import numpy as np


class SemanticCache:

    def __init__(self, similarity_threshold=0.85, max_size=1000):

        self.cache_entries = []
        self.cache_embeddings = None
        self.threshold = similarity_threshold
        self.max_size = max_size

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_vector):

        if self.cache_embeddings is None or len(self.cache_entries) == 0:
            self.miss_count += 1
            return {"hit": False}

        # Vectorized dot product for cosine similarity (embeddings are pre-normalized)
        scores = np.dot(self.cache_embeddings, query_vector)

        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        if best_score >= self.threshold:

            self.hit_count += 1

            return {
                "hit": True,
                "entry": self.cache_entries[best_idx],
                "score": float(best_score)
            }

        self.miss_count += 1

        return {"hit": False}

    def store(self, query, embedding, result):

        if len(self.cache_entries) >= self.max_size:
            # Simple FIFO cache eviction
            self.cache_entries.pop(0)
            self.cache_embeddings = self.cache_embeddings[1:]

        self.cache_entries.append({
            "query": query,
            "embedding": embedding,
            "result": result
        })

        if self.cache_embeddings is None:
            self.cache_embeddings = np.array([embedding])
        else:
            self.cache_embeddings = np.vstack([self.cache_embeddings, embedding])

    def stats(self):

        total = len(self.cache_entries)

        hit_rate = (
            self.hit_count / (self.hit_count + self.miss_count)
            if (self.hit_count + self.miss_count) > 0
            else 0
        )

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "max_size": self.max_size
        }

    def clear(self):

        self.cache_entries = []
        self.cache_embeddings = None
        self.hit_count = 0
        self.miss_count = 0

    def save(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                "entries": self.cache_entries,
                "embeddings": self.cache_embeddings,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count
            }, f)

    @classmethod
    def load(cls, filepath, similarity_threshold=0.85, max_size=1000):
        import pickle
        instance = cls(similarity_threshold, max_size)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            instance.cache_entries = data["entries"]
            instance.cache_embeddings = data["embeddings"]
            instance.hit_count = data["hit_count"]
            instance.miss_count = data["miss_count"]
        return instance

        