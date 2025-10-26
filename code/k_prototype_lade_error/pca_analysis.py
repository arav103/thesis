import pandas as pd
from sklearn.decomposition import PCA


class PCAAnalyzer:
    """Performs PCA and returns variance, loadings, and transformed features."""

    def __init__(self, data: pd.DataFrame, n_components: int = 4):
        self.data = data
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit_transform(self) -> pd.DataFrame:
        reduced_features = self.pca.fit_transform(self.data)
        explained = self.pca.explained_variance_ratio_
        print(f"Cumulative variance: {explained.cumsum()}")
        return pd.DataFrame(reduced_features, columns=[f"PC{i + 1}" for i in range(self.n_components)])

    def get_loadings(self) -> pd.DataFrame:
        loadings = self.pca.components_
        return pd.DataFrame(loadings, columns=self.data.columns, index=[f"PC{i + 1}" for i in range(self.n_components)])
