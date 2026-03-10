import numpy as np
from app.embeddings import load_dataset, embed_documents
from app.clustering import FuzzyClusterer

print("Loading dataset...")
docs = load_dataset()

print("Embedding...")
embeddings = embed_documents(docs)

print("Clustering...")
clusterer = FuzzyClusterer(20)
clusterer.fit(embeddings)

probs = clusterer.get_membership(embeddings)

for cluster_id in range(5):

    print("\nCLUSTER", cluster_id)

    # documents strongly belonging to this cluster
    scores = probs[:, cluster_id]

    top_indices = np.argsort(scores)[-5:]

    for i in top_indices:
        print("Score:", f"{scores[i]:.6f}")
        print(docs[i][:200])
        print()



print("\n========== FUZZY / BOUNDARY DOCUMENTS ==========\n")

for i in range(10):

    top_clusters = np.argsort(probs[i])[-3:][::-1]

    print("Document:", docs[i][:150])
    
    for c in top_clusters:
        print(f"Cluster {c} probability:", round(probs[i][c], 4))
    
    print()