from k_prototype_joined_df.clustering_k_prototypes import KPrototypesClusterer
from k_prototype_joined_df.pca_analysis import PCAAnalyzer
from k_prototype_joined_df.preprocessing import DataPreprocessor
from k_prototype_joined_df.visualization import Visualizer


def main():


    drop_indices = [13, 16, 17, 18, 21, 22, 32, 33, 34, 35, 43, 44, 47, 48, 51, 53, 54, 63, 65, 66, 73, 75, 76, 77, 78,
                    82, 86, 88, 89, 90, 91, 92]
    dp = DataPreprocessor("data/joinedfile.csv")
    df = dp.load_data(drop_indices)
    dp.convert_types([slice(0, 7), slice(46, 57)], [slice(7, 45), slice(58, 61)])
    dp.fill_missing([slice(46, 57)])
    df = dp.scale_columns(
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 58, 59, 60])


    categorical_indices = [0, 1, 2, 3, 4, 5, 6, 7, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 61]
    clusterer = KPrototypesClusterer(df, categorical_indices)
    cost_df = clusterer.evaluate_cost(20)
    Visualizer.plot_cost(cost_df, "outputs/plots/cost_plot.png")
    clusters = clusterer.fit(8)


    pca = PCAAnalyzer(df)
    reduced = pca.fit_transform()
    loadings = pca.get_loadings()
    loadings.to_excel("outputs/pcaloadings.xlsx")


    Visualizer.plot_dendrogram(clusterer.model.cluster_centroids_, "outputs/plots/dendrogram.png")
    Visualizer.plot_heatmap(df, "outputs/plots/heatmap.png")


if __name__ == "__main__":
    main()
