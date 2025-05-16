import os
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv


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

def retrieve_chunks(query, top_k=5, metadata_filter=None):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=metadata_filter)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    if not documents:
        return [
            {"content": "Sorry, I couldn't find relevant food places.", "metadata": {}}
        ]
    return [
        {"content": doc, "metadata": meta} for doc, meta in zip(documents, metadatas)
    ]

def generate_prompt_context(context_chunks):
    formatted_chunks = []
    for chunk in context_chunks:
        metadata = chunk.get('metadata', {})
        name = metadata.get('name', 'Unknown Name')
        location = metadata.get('location', 'Unknown Location')
        cuisine = metadata.get('cuisine_type', 'Unknown Cuisine')
        venue_type = metadata.get('venue_type', 'Unknown Venue Type')
        # Format the chunk with metadata
        formatted_chunk = f"Name: {name}\nLocation: {location}\nCuisine: {cuisine}\nVenue Type:{venue_type}\nContent: {chunk['content']}"
        formatted_chunks.append(formatted_chunk)
    return "\n---\n".join(formatted_chunks)

def generate_chat_response(query, context):
    system_prompt = (
    "You are a helpful assistant that recommends food places in Singapore based on the given context.\n\n"
    "When answering:\n"
    "- Be concise, friendly, and informative.\n"
    "- Use the context provided to extract real data. Do not make up information.\n"
    "- If available, include the following in the response for each place:\n"
    "  🍽️ \033[1mName\033[1m: name of the place.\n"
    "  ⏰ \033[1mOpening Hours\033[1m: If known, show opening hours.\n"
    "  📍 \033[1mLocation\033[1m: General area like Central, East, etc.\n"
    "  🍜 \033[1mCuisine / Tags\033[1m: Mention notable types of food served.\n"
    "  🏷️ \033[1mVenue Type\033[1m: Mention if it's a cafe, restaurant, etc.\n"
    "  💬 \033[1mReview Summary\033[1m: Summarize public or user reviews into one or two sentence, highlighting for why that place was chosen based on the user's query.\n\n"
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

# --- MAIN CHAT FUNCTION ---

def answer_question(query):
    # Extract identifiable metadata from the query
    metadata_filter = extract_metadata_filter(query)

    chunks = retrieve_chunks(query,metadata_filter=metadata_filter)
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
