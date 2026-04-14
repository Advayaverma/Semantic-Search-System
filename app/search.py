import faiss
import numpy as np

class VectorStore:

    def __init__(self, embeddings):

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.embeddings = embeddings

    def search(self, query_vector, k=5):

        query_vector = np.array([query_vector]).astype("float32")

        scores, indices = self.index.search(query_vector, k)

        return indices[0], scores[0]

    def save(self, filepath):
        faiss.write_index(self.index, filepath)

    @classmethod
    def load(cls, filepath, embeddings):
        instance = cls.__new__(cls)
        instance.index = faiss.read_index(filepath)
        instance.embeddings = embeddings
        return instance