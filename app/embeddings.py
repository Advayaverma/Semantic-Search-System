from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer
import numpy as np

# Load dataset
def load_dataset():

    dataset = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    documents = dataset.data

    # Remove very short documents (noise)
    documents = [doc for doc in documents if len(doc) > 50]

    return documents


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_documents(documents):

    embeddings = model.encode(documents, show_progress_bar=True)

    return np.array(embeddings)


def embed_query(query):

    embedding = model.encode([query])[0]

    return embedding