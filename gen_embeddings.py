import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid
import json
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

CLEANED_DATA_PATH = os.getenv("CLEANED_DATA_PATH", "cleaned/cleaned_data.json")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")

with open(CLEANED_DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} processed articles from {CLEANED_DATA_PATH}")

persist_directory = CHROMA_DB_PATH

# Load sentence-transformers model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Init Chroma DB
client = chromadb.PersistentClient(path=persist_directory)

collection = client.get_collection(name="food_places")

if collection is None:
    print("Collection not found. Creating a new one.")
    collection = client.create_collection(name="food_places")

# Clear the collection if needed
collection.delete(where={"$exists": True})

# enumerate over dataset with tqdm
for i, entry in tqdm(enumerate(dataset), total=len(dataset)):
    chunks = entry.get("article_text", [''])

    for j, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(uuid.uuid4())],
            metadatas=[{
                "name": entry["name"],
                "location": entry.get("location", ""),
                "cuisine_type": entry.get("cuisine_type", ""),
                "region": ", ".join(entry.get("regions", [])),
                "source_index": i,
                "chunk": j
            }]
        )

print(f"📀 Saved {len(dataset)} article embeddings into local vector DB")
