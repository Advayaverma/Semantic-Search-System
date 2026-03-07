import faiss
import numpy as np

class VectorStore:

    def __init__(self, embeddings):

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.embeddings = embeddings

    def search(self, query_vector, k=5):

        query_vector = np.array([query_vector]).astype("float32")

        distances, indices = self.index.search(query_vector, k)

        return indices[0], distances[0]