from k_prototype_lade_error.clustering import KPrototypesClusterer
from k_prototype_lade_error.data_processing import DataPreprocessor
from k_prototype_lade_error.pca_analysis import PCAAnalyzer
from k_prototype_lade_error.visualization import Visualizer


def main():
    categorical_indices = [0, 1, 2, 3, 4, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    numeric_indices = list(range(6, 37))

    dp = DataPreprocessor("data/my_porsche_joinedfile_dropped.csv")
    df = dp.load_data()
    df = dp.encode_and_scale(categorical_indices, numeric_indices)

    clusterer = KPrototypesClusterer(df, categorical_indices)
    results = clusterer.evaluate_clusters(2, 10)

    Visualizer.plot_metrics(results, "outputs/plots")

    pca = PCAAnalyzer(df, n_components=4)
    reduced = pca.fit_transform()
    loadings = pca.get_loadings()
    loadings.to_excel("outputs/pcaloadings.xlsx", index=True)


if __name__ == "__main__":
    main()
