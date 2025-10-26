from sklearn.model_selection import train_test_split
from supervised_rf.data_preprocessor import DataPreprocessor
from supervised_rf.feature_selector import FeatureSelector
from supervised_rf.rf_trainer import RandomForestTrainer
from supervised_rf.visualizer import plot_auc, plot_feature_importance, plot_dendrogram

def main():
    data_path = 'data/continued_charging_error.csv'
    target_col = 'Error 1'
    drop_cols = ['Error 2','Error 3','Error 4','Error 5','bms_istmodus_error']
    categorical_slices = [slice(0,5), slice(36,56)]

    pre = DataPreprocessor(categorical_slices)
    df = pre.load_data(data_path)
    df = pre.encode_categoricals(df)
    df = pre.scale_features(df)
    X, y = pre.finalize(df, drop_cols, target_col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    trainer = RandomForestTrainer()
    sweep_df = trainer.sweep_estimators(X_train, y_train, X_test, y_test, grid=[1,2,4,8,16,32,48,64,82,108])
    plot_auc(sweep_df, 'outputs/rf/plots/auc_sweep.png')

    clf = trainer.train_final(X_train, y_train, n_estimators=22, max_depth=35)
    metrics = trainer.evaluate(clf, X_test, y_test)
    print(metrics)

    imp = trainer.feature_importance(clf, X_test, y_test)
    plot_feature_importance(imp, 'outputs/rf/plots/importance_full.png', 'Feature Importance')

    selector = FeatureSelector(distance_threshold=1.1)
    mandatory = ['charge_serviceprovidername','charge_platform','charge_partnername']
    final_feats, linkage = selector.select_features(X, mandatory)
    plot_dendrogram(linkage, X.columns.tolist(), 'outputs/rf/plots/dendrogram.png')

if __name__ == "__main__":
    main()
