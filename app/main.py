import os
import time
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

from app.embeddings import load_dataset, embed_documents, embed_query
from app.search import VectorStore
from app.cache import SemanticCache
from app.clustering import FuzzyClusterer


app = FastAPI()


class QueryRequest(BaseModel):
    query: str


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

DOCS_PATH = os.path.join(DATA_DIR, "documents.pkl")
FAISS_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
GMM_PATH = os.path.join(DATA_DIR, "gmm_model.pkl")
CACHE_PATH = os.path.join(DATA_DIR, "cache.pkl")

print("Initializing system...")

target_names = []

if os.path.exists(DOCS_PATH) and os.path.exists(FAISS_PATH) and os.path.exists(GMM_PATH):
    print("Loading pre-computed data from disk...")
    # Load docs and target names
    with open(DOCS_PATH, "rb") as f:
        documents, target_names, doc_embeddings = pickle.load(f)
    
    print("Loading vector store...")
    vector_store = VectorStore.load(FAISS_PATH, doc_embeddings)
    
    print("Loading clusterer...")
    clusterer = FuzzyClusterer.load(GMM_PATH)
else:
    print("Pre-computed data not found. Building from scratch...")
    print("Loading dataset...")
    documents, target_names = load_dataset()
    
    print("Embedding documents...")
    doc_embeddings = embed_documents(documents)
    
    print("Building vector store...")
    vector_store = VectorStore(doc_embeddings)
    
    print("Clustering...")
    clusterer = FuzzyClusterer(n_clusters=20)
    clusterer.fit(doc_embeddings)
    
    print("Saving to disk for future fast startup...")
    with open(DOCS_PATH, "wb") as f:
        pickle.dump((documents, target_names, doc_embeddings), f)
    vector_store.save(FAISS_PATH)
    clusterer.save(GMM_PATH)

if os.path.exists(CACHE_PATH):
    print("Loading cache from disk...")
    cache = SemanticCache.load(CACHE_PATH)
else:
    print("Initializing fresh cache...")
    cache = SemanticCache()


@app.post("/query")
def query_system(request: QueryRequest):
    
    start_time = time.time()

    query = request.query
    query_vector = embed_query(query)

    cache_lookup = cache.lookup(query_vector)

    if cache_lookup["hit"]:
        entry = cache_lookup["entry"]
        result = entry["result"]
        similarity_score = cache_lookup["score"]
        matched_query = entry["query"]
        cache_hit = True
    else:
        indices, distances = vector_store.search(query_vector)
        result = [documents[i] for i in indices[:3]]
        
        cache.store(query, query_vector, result)
        cache.save(CACHE_PATH) # Persist after miss
        
        similarity_score = None
        matched_query = None
        cache_hit = False

    dominant_cluster_id = clusterer.dominant_cluster(query_vector)
    
    # Map GMM cluster ID to the target_names list.
    # Note: GMM clustering is unsupervised so cluster ID != newsgroup target ID directly,
    # but this provides a deterministic mapping for demonstration.
    dominant_cluster_name = target_names[dominant_cluster_id] if 0 <= dominant_cluster_id < len(target_names) else f"Cluster {dominant_cluster_id}"
    
    processing_time_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "query": query,
        "cache_hit": cache_hit,
        "matched_query": matched_query,
        "similarity_score": similarity_score,
        "result": result,
        "dominant_cluster": dominant_cluster_id,
        "dominant_cluster_name": dominant_cluster_name,
        "processing_time_ms": processing_time_ms
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()
    
    if os.path.exists(CACHE_PATH):
        os.remove(CACHE_PATH)

    return {"message": "Cache cleared"}


@app.get("/health")
def health_endpoint():
    
    return {
        "status": "healthy",
        "faiss_loaded": vector_store is not None,
        "cache_entries": len(cache.cache_entries)
    }
