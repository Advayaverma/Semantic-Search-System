Semantic Search System
Fuzzy Clustering + Semantic Cache + FastAPI

A lightweight semantic search system built using document embeddings, fuzzy clustering, and a semantic cache. The system exposes a REST API using FastAPI and can be deployed using Docker.

The goal of this project is to demonstrate how semantic search pipelines can combine vector embeddings, probabilistic clustering, and caching to improve retrieval performance and reduce redundant computation.

## System Architecture

```
20 Newsgroups Dataset
        │
        ▼
Sentence Embeddings (MiniLM-L6-v2)
        │
        ▼
Vector Database (FAISS)
        │
        ▼
Fuzzy Clustering (Gaussian Mixture)
        │
        ▼
Semantic Cache
        │
        ▼
FastAPI API
```
Dataset

The project uses the 20 Newsgroups, which contains approximately 20,000 newsgroup posts across 20 different topics.

To improve embedding quality, preprocessing removes:

headers

footers

quoted replies

This reduces noise from email metadata and previous messages.

Embeddings

Documents are embedded using the all-MiniLM-L6-v2.

Reasons for choosing this model:

fast inference

strong semantic similarity performance

small model size

widely used for semantic search

Each document is converted into a 384-dimensional embedding vector.

Vector Database

Embeddings are stored using FAISS (Facebook AI Similarity Search).

Advantages:

fast nearest-neighbor search

optimized for high-dimensional vectors

scalable for large datasets

The vector store allows fast retrieval of semantically similar documents.

Fuzzy Clustering

The system uses a Gaussian Mixture Model to perform soft clustering.

Unlike hard clustering, each document receives probabilities across clusters.

Example:

Cluster 12 probability: 0.79
Cluster 0 probability: 0.13
Cluster 9 probability: 0.06

This captures the fact that documents can belong to multiple semantic topics.

Semantic Cache

Traditional caches only work when queries are identical.

This system implements a semantic cache that detects similar queries using embeddings.

Example:

Query 1: space shuttle launch
Query 2: NASA rocket launch

These queries have similar embeddings, so the system can reuse previous results.

Benefits:

avoids recomputation

improves response time

reduces embedding workload

API Endpoints

The system exposes an API using FastAPI.

Query Endpoint
POST /query

Example request:

{
  "query": "space shuttle launch"
}

Example response:

{
  "query": "...",
  "cache_hit": true,
  "matched_query": "...",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3
}
Cache Statistics
GET /cache/stats

Example response:

{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
Clear Cache
DELETE /cache

Resets the semantic cache and statistics.

Running the Project
Local Setup

Install dependencies:

pip install -r requirements.txt

Start the server:

uvicorn app.main:app --reload

Open:

http://localhost:8000/docs

This shows the interactive API documentation.

Docker Deployment

The project can also be run inside Docker.

Build the image:

docker build -t semantic-search .

Run the container:

docker run -p 8000:8000 semantic-search

Then open:

http://localhost:8000/docs
Project Structure
semantic-search-system
│
├── app
│   ├── __init__.py
│   ├── cache.py
│   ├── cluster_analysis.py
│   ├── clustering.py
│   ├── embeddings.py
│   ├── main.py
│   └── search.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
└── README.md
Design Decisions

Sentence Transformer Embeddings

Chosen for speed, accuracy, and lightweight deployment.

Gaussian Mixture Clustering

Allows probabilistic cluster membership instead of hard assignments.

Semantic Cache

Implemented from first principles without Redis or external caching systems.

FastAPI

Provides a fast, modern API layer with automatic documentation.

Future Improvements

Potential extensions include:

persistent embedding storage

cluster visualization

distributed caching

larger datasets

GPU acceleration

Conclusion

This project demonstrates how vector embeddings, fuzzy clustering, and semantic caching can be combined to build a scalable semantic search system.

The architecture highlights how semantic similarity can improve caching and reduce redundant computation in NLP systems.