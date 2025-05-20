import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import sys
import argparse
from dotenv import load_dotenv
import math
from collections import defaultdict
import numpy as np
from core.retrieval import retrieve_relevant_chunks

load_dotenv()

# Load unwanted keywords from environment variable (comma-separated)
UNWANTED_KEYWORDS = os.getenv("UNWANTED_KEYWORDS", "Read more at:").split(",")
UNWANTED_KEYWORDS = [kw.strip() for kw in UNWANTED_KEYWORDS if kw.strip()]
UNWANTED_PENALTY = float(os.getenv("UNWANTED_PENALTY", "0.2"))
UNWANTED_PENALTY_SCALE = float(os.getenv("UNWANTED_PENALTY_SCALE", "2.0"))
UNWANTED_PENALTY_MAXLEN = int(os.getenv("UNWANTED_PENALTY_MAXLEN", "500"))

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

# Load sentence-transformers model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Init Chroma DB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_collection(name="food_places")

def retrieve(query, top_k=5, top_j=2):
    results = retrieve_relevant_chunks(query, collection, model, top_k=top_k, top_j=top_j)
    print(f"\nTop {top_k} articles for query: '{query}' (showing up to {top_j} unique chunks per article)\n")
    for i, article in enumerate(results):
        article_name = article["article_name"]
        unique_chunks = article["chunks"]
        print(f"Article {i+1}: {article_name} , Unique chunks: {len(unique_chunks)}")
        for j, chunk in enumerate(unique_chunks):
            print(f"  Chunk {j+1} (similarity: {chunk['similarity']:.4f}, penalty: {chunk['penalty']:.4f}, penalized: {chunk['penalized_score']:.4f}):")
            print(f"    Location: {chunk['meta'].get('location', '')}")
            print(f"    Cuisine: {chunk['meta'].get('cuisine_type', '')}")
            print(f"    Region: {chunk['meta'].get('region', '')}")
            print(f"    Chunk: {chunk['meta'].get('chunk', '')}")
            print(f"    Text: {chunk['doc'][:300]}{'...' if len(chunk['doc']) > 300 else ''}\n")
    if not results:
        print("No results found.")

def browse():
    print("\nBrowsing all documents in the collection. Press Enter to see next, or 'q' to quit.\n")
    # ChromaDB does not support direct iteration, so we fetch all ids first
    all_ids = collection.get()["ids"]
    batch_size = 1
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i+batch_size]
        batch = collection.get(ids=batch_ids)
        for doc, meta in zip(batch["documents"], batch["metadatas"]):
            print(f"Name: {meta.get('name', '')}")
            print(f"Location: {meta.get('location', '')}")
            print(f"Cuisine: {meta.get('cuisine_type', '')}")
            print(f"Region: {meta.get('region', '')}")
            print(f"Chunk: {meta.get('chunk', '')}")
            print(f"Text: {doc[:500]}{'...' if len(doc) > 500 else ''}\n")
            user_input = input("Press Enter for next, 'q' to quit: ")
            if user_input.strip().lower() == 'q':
                return

def main():
    parser = argparse.ArgumentParser(description="Test retrieval or browse Chroma DB collection.")
    parser.add_argument('--query', type=str, help='User query for retrieval mode')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to show')
    parser.add_argument('--browse', action='store_true', help='Browse the collection interactively')
    args = parser.parse_args()

    if args.browse:
        browse()
    elif args.query:
        retrieve(args.query, args.top_k)
    else:
        print("Please provide either --query or --browse. Use -h for help.")

if __name__ == "__main__":
    main()
