from sklearn.mixture import GaussianMixture
import numpy as np


class FuzzyClusterer:

    def __init__(self, n_clusters=20):

        self.model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="diag",
            reg_covar=1e-3,
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

    def save(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)

    @classmethod
    def load(cls, filepath):
        import pickle
        instance = cls.__new__(cls)
        with open(filepath, 'rb') as f:
            instance.model = pickle.load(f)
        return instance
        