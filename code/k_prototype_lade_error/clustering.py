import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


class KPrototypesClusterer:
    """Performs clustering and evaluates metrics."""

    def __init__(self, data: pd.DataFrame, categorical_indices: list[int]):
        self.data = data
        self.categorical_indices = categorical_indices
        self.results = {}

    def evaluate_clusters(self, k_min: int = 2, k_max: int = 10) -> pd.DataFrame:
        cost, ch_scores, db_scores, sil_scores = [], [], [], []

        for k in range(k_min, k_max + 1):
            model = KPrototypes(n_jobs=-1, n_clusters=k, init='Huang', random_state=3)
            clusters = model.fit_predict(self.data, categorical=self.categorical_indices)
            cost.append(model.cost_)
            ch_scores.append(calinski_harabasz_score(self.data, clusters))
            db_scores.append(davies_bouldin_score(self.data, clusters))
            sil_scores.append(silhouette_score(self.data, clusters))
            print(f"Cluster {k}: CH={ch_scores[-1]:.2f}, DB={db_scores[-1]:.2f}, Silhouette={sil_scores[-1]:.2f}")

        self.results = pd.DataFrame({
            'Clusters': range(k_min, k_max + 1),
            'Cost': cost,
            'Calinski-Harabasz': ch_scores,
            'Davies-Bouldin': db_scores,
            'Silhouette': sil_scores
        })

        return self.results
