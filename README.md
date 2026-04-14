# Semantic Search System  
### Fuzzy Clustering + Semantic Cache + FastAPI

![Python](https://img.shields.io/badge/python-3.10-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/docker-containerized-blue)

A lightweight **semantic search system** built using document embeddings, fuzzy clustering, and a semantic cache.  
The system exposes a REST API using **FastAPI** and can be deployed using **Docker**.

The goal of this project is to demonstrate how semantic search pipelines combine **vector embeddings, probabilistic clustering, and caching** to improve retrieval performance and reduce redundant computation.

---

# System Architecture

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

---

# Dataset

The project uses the **20 Newsgroups dataset**, which contains approximately **20,000 discussion posts across 20 different topics**.

To improve embedding quality, preprocessing removes:

- headers  
- footers  
- quoted replies  

This reduces noise from email metadata and previous messages.

---

# Embeddings

Documents are embedded using the **all-MiniLM-L6-v2 sentence transformer model**.

Reasons for choosing this model:

- fast inference
- strong semantic similarity performance
- small model size
- widely used for semantic search

Each document is converted into a **384-dimensional embedding vector**.

---

# Vector Database

Embeddings are stored using **FAISS (Facebook AI Similarity Search)**.

Advantages:

- fast nearest-neighbor search
- optimized for high-dimensional vectors
- scalable for large datasets

The vector store allows fast retrieval of semantically similar documents.

---

# Fuzzy Clustering

The system uses a **Gaussian Mixture Model (GMM)** to perform **soft clustering**.

Unlike hard clustering methods, each document receives probabilities across clusters.

Example:


Cluster 12 probability: 0.79
Cluster 0 probability: 0.13
Cluster 9 probability: 0.06


This captures the fact that documents may belong to **multiple semantic topics simultaneously**.

---

# Semantic Cache

Traditional caches only work when queries are identical.

This project implements a **semantic cache** that detects queries with similar meaning.

Example:


Query 1: space shuttle launch
Query 2: NASA rocket launch


Since these queries are semantically similar, the system can **reuse cached results**.

Benefits:

- reduces redundant computation
- improves response time
- increases system efficiency

---

# API Endpoints

The system exposes a REST API using **FastAPI**.

### Query Endpoint


POST /query


Example request:

```json
{
  "query": "space shuttle launch"
}
```

Example response:

```json
{
  "query": "...",
  "cache_hit": true,
  "matched_query": "...",
  "similarity_score": 0.91,
  "result": "...",
  "dominant_cluster": 3,
  "dominant_cluster_name": "sci.space",
  "processing_time_ms": 1.45
}
```
Cache Statistics
```
GET /cache/stats
```

Example response:

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.405
}
```

Clear Cache
```
DELETE /cache
```
Resets the semantic cache and statistics.

System Health
```
GET /health
```
Checks if the system is running and if the models are loaded.

Example response:
```json
{
  "status": "healthy",
  "faiss_loaded": true,
  "cache_entries": 17
}
```

# Running the Project

***Local Setup***

Install dependencies:
```bash
pip install -r requirements.txt
```

Start the API server:
```bash
uvicorn app.main:app --reload
```

Open the API documentation:
```
http://localhost:8000/docs
```

# Docker Deployment

Build the Docker image:

```bash
docker build -t semantic-search .
```

Run the container:
```bash
docker run -p 8000:8000 semantic-search
```

Then open:
```
http://localhost:8000/docs
```
# Project Structure
```
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
```
# Design Decisions
**Sentence Transformer Embeddings**

Chosen for fast inference and strong semantic representation.

**Gaussian Mixture Clustering**

Allows probabilistic cluster membership instead of rigid cluster assignments.

**Semantic Cache**

Implemented from first principles without external caching frameworks.

**FastAPI**

Provides a fast and lightweight API layer with automatic documentation.

# Future Improvements

Possible extensions include:

- interactive frontend web application

- cluster visualization

- distributed caching

- GPU acceleration

- scaling to larger datasets

# Conclusion

This project demonstrates how semantic embeddings, fuzzy clustering, and intelligent caching can be combined to build an efficient semantic search system.

The architecture highlights how semantic similarity can reduce redundant computation while maintaining high-quality retrieval.