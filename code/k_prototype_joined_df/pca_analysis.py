import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class PCAAnalyzer:
    """Performs PCA and exports component loadings."""

    def __init__(self, data: pd.DataFrame, n_components: int = 4):
        self.data = data
        self.pca = PCA(n_components=n_components)

    def fit_transform(self) -> pd.DataFrame:
        reduced = self.pca.fit_transform(self.data)
        return pd.DataFrame(reduced, columns=[f"PC{i + 1}" for i in range(self.pca.n_components_)])

    def get_loadings(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.pca.components_,
            columns=self.data.columns,
            index=[f"PC{i + 1}" for i in range(self.pca.n_components_)]
        )

    def get_explained_variance(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_
