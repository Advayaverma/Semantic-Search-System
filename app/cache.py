import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticCache:

    def __init__(self, similarity_threshold=0.85):

        self.cache = []
        self.threshold = similarity_threshold

        self.hit_count = 0
        self.miss_count = 0

    def lookup(self, query_vector):

        best_score = 0
        best_entry = None

        for entry in self.cache:

            score = cosine_similarity(
                [query_vector],
                [entry["embedding"]]
            )[0][0]

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold:

            self.hit_count += 1

            return {
                "hit": True,
                "entry": best_entry,
                "score": float(best_score)
            }

        self.miss_count += 1

        return {"hit": False}

    def store(self, query, embedding, result):

        self.cache.append({
            "query": query,
            "embedding": embedding,
            "result": result
        })

    def stats(self):

        total = len(self.cache)

        hit_rate = (
            self.hit_count / (self.hit_count + self.miss_count)
            if (self.hit_count + self.miss_count) > 0
            else 0
        )

        return {
            "total_entries": total,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate
        }

    def clear(self):

        self.cache = []
        self.hit_count = 0
        self.miss_count = 0
        