from fastapi import FastAPI
from pydantic import BaseModel

from app.embeddings import load_dataset, embed_documents, embed_query
from app.search import VectorStore
from app.cache import SemanticCache
from app.clustering import FuzzyClusterer


app = FastAPI()


class QueryRequest(BaseModel):
    query: str


print("Loading dataset...")

documents = load_dataset()

print("Embedding documents...")

doc_embeddings = embed_documents(documents)

print("Building vector store...")

vector_store = VectorStore(doc_embeddings)

print("Clustering...")

clusterer = FuzzyClusterer(n_clusters=20)
clusterer.fit(doc_embeddings)

cache = SemanticCache()


@app.post("/query")
def query_system(request: QueryRequest):

    query = request.query

    query_vector = embed_query(query)

    cache_lookup = cache.lookup(query_vector)

    if cache_lookup["hit"]:

        entry = cache_lookup["entry"]

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": entry["query"],
            "similarity_score": cache_lookup["score"],
            "result": entry["result"],
            "dominant_cluster": clusterer.dominant_cluster(query_vector)
        }

    # Cache miss → compute result
    indices, distances = vector_store.search(query_vector)

    result_docs = [documents[i] for i in indices[:3]]

    result = result_docs

    cache.store(query, query_vector, result)

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": result,
        "dominant_cluster": clusterer.dominant_cluster(query_vector)
    }


@app.get("/cache/stats")
def cache_stats():

    return cache.stats()


@app.delete("/cache")
def clear_cache():

    cache.clear()

    return {"message": "Cache cleared"}
