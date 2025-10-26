import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram


class Visualizer:
    """Handles plots for clustering metrics and PCA analysis."""

    @staticmethod
    def plot_cost(df: pd.DataFrame, save_path: str):
        plt.figure(figsize=(8, 5))
        sns.lineplot(x="Cluster", y="Cost", data=df, marker="o")
        plt.title("K-Prototypes Cost vs Cluster Count")
        plt.xlabel("Clusters (k)")
        plt.ylabel("Cost")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_dendrogram(centers, save_path: str):
        plt.figure(figsize=(10, 7))
        dendrogram(linkage(centers, method="ward"))
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Samples")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

    @staticmethod
    def plot_heatmap(df: pd.DataFrame, save_path: str):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, cmap="vlag")
        plt.title("Cluster Heatmap")
        plt.tight_layout()
        plt.savefig(save_path, dpi=600)
        plt.close()
