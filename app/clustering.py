from sklearn.mixture import GaussianMixture
import numpy as np


class FuzzyClusterer:

    def __init__(self, n_clusters=20):

        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',
            random_state=42
        )

    def fit(self, embeddings):

        self.model.fit(embeddings)

    def get_membership(self, embeddings):

        probs = self.model.predict_proba(embeddings)

        return probs

    def dominant_cluster(self, vector):

        probs = self.model.predict_proba([vector])[0]

        return int(np.argmax(probs))
        