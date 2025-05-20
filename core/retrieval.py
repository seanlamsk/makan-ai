import os
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Load unwanted keywords and penalty settings from environment
UNWANTED_KEYWORDS = os.getenv("UNWANTED_KEYWORDS", "Read more at:").split(",")
UNWANTED_KEYWORDS = [kw.strip() for kw in UNWANTED_KEYWORDS if kw.strip()]
UNWANTED_PENALTY = float(os.getenv("UNWANTED_PENALTY", "0.2"))
UNWANTED_PENALTY_SCALE = float(os.getenv("UNWANTED_PENALTY_SCALE", "2.0"))
UNWANTED_PENALTY_MAXLEN = int(os.getenv("UNWANTED_PENALTY_MAXLEN", "500"))

def calculate_similarity(model, chunk, query):
    """
    Calculate the cosine similarity between a chunk and a query.
    """
    chunk_embedding = model.encode(chunk).tolist()
    query_embedding = model.encode(query).tolist()
    sim = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
    return sim

def calculate_penalized_score(model, doc, query):
    sim = calculate_similarity(model, doc, query)
    # Initialize penalty
    penalty = 0.0
    for kw in UNWANTED_KEYWORDS:
        if kw and kw.lower() in doc.lower():
            # Length of the chunk
            chunk_len = len(doc)
            # Normalize the length to a maximum of UNWANTED_PENALTY_MAXLEN
            norm_len = min(chunk_len, UNWANTED_PENALTY_MAXLEN) / UNWANTED_PENALTY_MAXLEN
            # Exponential penalty based on the normalized length
            exp_penalty = UNWANTED_PENALTY * math.exp(UNWANTED_PENALTY_SCALE * (1 - norm_len))
            penalty += exp_penalty
    penalized_score = sim - penalty
    return penalized_score


def retrieve_relevant_chunks(query, collection, model, metadata_filter = None, top_k=5, top_j=2):
    """
    Retrieve top K articles and top J relevant chunks per article from a ChromaDB collection.
    Returns a list of dicts: [{article_name, chunks: [chunk_info, ...]}, ...]
    Each chunk_info contains: penalized_score, similarity, penalty, doc, meta
    """
    embedding = model.encode(query).tolist()
    n_results = max(100, top_k * top_j * 5)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
        where=metadata_filter
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    article_best_chunk = {}
    for doc, meta, dist in zip(docs, metas, dists):
        article_name = meta.get('name', '')
        if article_name not in article_best_chunk or dist < article_best_chunk[article_name][0]:
            article_best_chunk[article_name] = (dist, doc, meta)

    sorted_articles = sorted(article_best_chunk.items(), key=lambda x: x[1][0])
    top_articles = [name for name, _ in sorted_articles[:top_k]]

    results = []
    for article_name in top_articles:
        article_data = collection.get(where={"name": article_name})
        article_docs = article_data["documents"]
        article_metas = article_data["metadatas"]
        chunk_scores = []
        for doc, meta in zip(article_docs, article_metas):
            chunk_embedding = model.encode(doc).tolist()
            # sim = np.dot(embedding, chunk_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(chunk_embedding))
            sim = calculate_similarity(model, doc, query)
            penalized_score = calculate_penalized_score(model, doc, query)
            print(f"Doc: {doc}, Meta: {meta}, Similarity: {sim}, Penalized Score: {penalized_score}")
            chunk_scores.append({
                "penalized_score": penalized_score,
                "similarity": sim,
                "doc": doc,
                "meta": meta
            })
        # Sort and filter to unique chunk indices
        chunk_scores.sort(reverse=True, key=lambda x: x["penalized_score"])
        seen_chunks = set()
        unique_chunks = []
        for chunk in chunk_scores:
            chunk_idx = chunk["meta"].get('chunk', None)
            if chunk_idx not in seen_chunks:
                unique_chunks.append(chunk)
                seen_chunks.add(chunk_idx)
            if len(unique_chunks) >= top_j:
                break
        results.append({
            "article_name": article_name,
            "chunks": unique_chunks
        })
    return results
