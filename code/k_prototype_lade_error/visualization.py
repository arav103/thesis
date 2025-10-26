import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Visualizer:
    """Generates and saves clustering and PCA plots."""

    @staticmethod
    def plot_metrics(df: pd.DataFrame, save_dir: str):
        metrics = ['Calinski-Harabasz', 'Davies-Bouldin', 'Silhouette', 'Cost']
        for metric in metrics:
            plt.figure()
            sns.lineplot(x='Clusters', y=metric, data=df, marker='o')
            plt.title(f"{metric} vs Number of Clusters")
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{metric.lower().replace(' ', '_')}.png", dpi=300)
            plt.close()
