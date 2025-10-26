from k_prototype_joined_df.clustering_kmodes import KModesClusterer
from k_prototype_joined_df.preprocessing import DataPreprocessor
from k_prototype_joined_df.pca_analysis import PCAAnalyzer
from k_prototype_joined_df.visualization import Visualizer


def main():
    drop_indices = [13, 16, 17, 18, 21, 22, 32, 33, 34, 35, 43, 44, 47, 48, 51, 53, 54, 63, 65, 66, 67, 69, 70, 73, 74,
                    75, 76, 77, 78, 81, 82, 86, 88, 89, 90, 91, 92]
    column_ranges = [slice(0, 7), slice(46, 52), [55]]

    # Preprocessing
    dp = DataPreprocessor("data/error_clustering.csv")
    df = dp.load_data(drop_indices)
    df = dp.encode_categorical(column_ranges)
    df = dp.scale_data()

    # Clustering
    clusterer = KModesClusterer(df)
    cost_df = clusterer.evaluate_cost(max_clusters=12)
    Visualizer.plot_cost(cost_df, "outputs/plots/cost_plot.png")

    clusters = clusterer.fit(n_clusters=7)

    # PCA
    pca = PCAAnalyzer(df)
    reduced = pca.fit_transform()
    loadings = pca.get_loadings()
    loadings.to_excel("outputs/pcaloadings.xlsx")

    # Hierarchical
    Visualizer.plot_dendrogram(clusterer.model.cluster_centroids_, "outputs/plots/dendrogram.png")


if __name__ == "__main__":
    main()
