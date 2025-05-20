import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from core.retrieval import retrieve_relevant_chunks, calculate_similarity, calculate_penalized_score


# --- CONFIG ---
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN") 

HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "food_places"

# --- INIT ---

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(COLLECTION_NAME)

# Hugging Face inference client
hf_client = InferenceClient(
    model=HF_MODEL,
    token=HUGGINGFACE_TOKEN
)

# --- RAG COMPONENTS ---

def retrieve_chunks(query, top_k=5, metadata_filter=None, top_j=2):
    # Use the same retrieval logic as before, but group and summarize chunks per article
    results = retrieve_relevant_chunks(query, collection, embedder, metadata_filter=metadata_filter, top_k=top_k, top_j=top_j)
    summarized_chunks = []
    for article in results:
        # Combine all top chunks for this article into a single summary
        combined_content = "\n---\n".join([chunk["doc"] for chunk in article["chunks"]])
        # Use the metadata from the first chunk as representative
        meta = article["chunks"][0]["meta"] if article["chunks"] else {}
        # Calculate similarity between query and combined content
        similarity = calculate_similarity(embedder, query, combined_content)
        penalized_score = calculate_penalized_score(embedder, combined_content, query)
        summarized_chunks.append({
            "content": combined_content,
            "metadata": meta,
            "similarity": similarity,
            "penalized_score": penalized_score
        })
    if not summarized_chunks:
        return [
            {"content": "Sorry, I couldn't find relevant food places.", "metadata": {}, "similarity": 0.0}
        ]
    return summarized_chunks

def generate_prompt_context(context_chunks):
    formatted_chunks = []
    for chunk in context_chunks:
        metadata = chunk.get('metadata', {})
        name = metadata.get('name', 'Unknown Name')
        location = metadata.get('location', 'Unknown Location')
        cuisine = metadata.get('cuisine_type', 'Unknown Cuisine')
        venue_type = metadata.get('venue_type', 'Unknown Venue Type')
        # Format the chunk with metadata, now with combined content
        formatted_chunk = f"Name: {name}\nLocation: {location}\nCuisine: {cuisine}\nVenue Type: {venue_type}\nContent: {chunk['content']}"
        formatted_chunks.append(formatted_chunk)
    return "\n---\n".join(formatted_chunks)

def generate_chat_response(query, context):
    system_prompt = (
    "You are a helpful assistant that recommends food places in Singapore based on the given context.\n\n"
    "When answering:\n"
    "- Be concise, friendly, and informative.\n"
    "- Use the context provided to extract real data. Do not make up information.\n"
    "- Ignore any information that is not relevant the food venue in the context. Do not include it in the response.\n"
    "- If available, include the following in the response for each place:\n"
    "  🍽️ Name: name of the place.\n"
    "  ⏰ Opening Hours: If known, show opening hours.\n"
    "  📍 Location: General area like Central, East, etc.\n"
    "  🍜 Cuisine / Tags: Mention notable types of food served.\n"
    "  🏷️ Venue Type: Mention if it's a cafe, restaurant, etc.\n"
    "  💬 Review Summary: Summarize public or user reviews into one or two sentence, highlighting for why that place was chosen based on the user's query.\n\n"
    "Format the output with these emoji headers for better readability."
)


    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}"},
        {"role": "user", "content": query}
    ]

    try:
        response = hf_client.chat_completion(
            messages=messages,
            temperature=0.5,
            max_tokens=500,
            top_p=0.7,
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"[ERROR] Hugging Face chat completion failed: {e}")
        return "Sorry, I couldn't generate a response."

# --- METADATA EXTRACTION ---
def extract_metadata_filter(query):
    # Pipline to extract metadata from the query
    metadata = {}
    # merge metadata with any extracted regions
    regions = extract_region_from_query(query)
    if regions != None:
        metadata.update(regions)

    # If metadata is empty, return None
    if not metadata:
        return None
    return metadata

def extract_region_from_query(query):
    # Simple heuristic to extract region from the query
    regions = ["Central", "East", "West", "North", "South"]
    matched_regions = []
    for region in regions:
        if region.lower() in query.lower():
            matched_regions.append(region)
    
    if len(matched_regions) == 1:
        return {'region': matched_regions[0]}
    elif len(matched_regions) > 1:
        return {'region': {'$in':matched_regions}}
    return None

# --- GUARDRAIL SYSTEM ---
def apply_guardrails(query, context_chunks):
    # List of guardrail rules, each returns (should_override, response)
    for rule in [guardrail_no_results]:
        should_override, response = rule(query, context_chunks)
        if should_override:
            return True, response
    return False, None

def guardrail_no_results(query, context_chunks):
    # If the only chunk is the default sorry message, override
    if (
        len(context_chunks) == 1 and
        context_chunks[0]["content"].startswith("Sorry, I couldn't find relevant food places")
    ):
        return True, context_chunks[0]["content"]
    return False, None

# --- MAIN CHAT FUNCTION ---
def answer_question(query):
    # Extract identifiable metadata from the query
    metadata_filter = extract_metadata_filter(query)
    chunks = retrieve_chunks(query, metadata_filter=metadata_filter)
    # Guardrail check
    should_override, guardrail_response = apply_guardrails(query, chunks)
    if should_override:
        return guardrail_response
    context = generate_prompt_context(chunks)
    return generate_chat_response(query, context)

# --- INTERACTIVE LOOP ---

if __name__ == "__main__":
    while True:
        query = input("\nAsk me about food in Singapore: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = answer_question(query)
        print("\n🤖", response)
