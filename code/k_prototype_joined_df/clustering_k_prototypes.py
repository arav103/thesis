import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes


class KPrototypesClusterer:
    """Runs K-Prototypes clustering and cost analysis."""

    def __init__(self, data: pd.DataFrame, categorical_indices: list[int]):
        self.data = data
        self.categorical_indices = categorical_indices
        self.model = None
        self.clusters = None

    def evaluate_cost(self, max_clusters: int = 20) -> pd.DataFrame:
        cost = []
        for k in range(1, max_clusters + 1):
            kprototype = KPrototypes(n_jobs=-1, n_clusters=k, init="Huang")
            kprototype.fit_predict(self.data.values, categorical=self.categorical_indices)
            cost.append(kprototype.cost_)
            print(f"Cluster {k}: cost = {kprototype.cost_:.2f}")
        return pd.DataFrame({"Cluster": range(1, max_clusters + 1), "Cost": cost})

    def fit(self, n_clusters: int) -> np.ndarray:
        self.model = KPrototypes(n_clusters=n_clusters, init="Huang", verbose=2, random_state=3)
        self.clusters = self.model.fit_predict(self.data.values, categorical=self.categorical_indices)
        return self.clusters
