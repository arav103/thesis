import numpy as np
import pandas as pd
from kmodes.kmodes import KModes


class KModesClusterer:
    """Performs K-Modes clustering and cost evaluation."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.model = None
        self.clusters = None

    def evaluate_cost(self, max_clusters: int = 10) -> pd.DataFrame:
        cost = []
        for k in range(1, max_clusters + 1):
            kmodes = KModes(n_clusters=k, init='Cao', n_jobs=-1)
            kmodes.fit_predict(self.data)
            cost.append(kmodes.cost_)
            print(f"Cluster {k}: cost = {kmodes.cost_:.2f}")
        return pd.DataFrame({"Cluster": range(1, max_clusters + 1), "Cost": cost})

    def fit(self, n_clusters: int) -> np.ndarray:
        self.model = KModes(n_clusters=n_clusters, init='Cao', verbose=1)
        self.clusters = self.model.fit_predict(self.data)
        return self.clusters
