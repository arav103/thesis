import os

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_auc(df, path):
    ensure_dir(os.path.dirname(path))
    plt.figure()
    plt.plot(df['n_estimators'], df['train_auc'], label='Train AUC')
    plt.plot(df['n_estimators'], df['test_auc'], label='Test AUC')
    plt.legend(); plt.xlabel('n_estimators'); plt.ylabel('AUC')
    plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

def plot_feature_importance(imp, path, title):
    ensure_dir(os.path.dirname(path))
    imp.plot(kind='bar', figsize=(12,5))
    plt.title(title); plt.ylabel('Normalized Importance'); plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()

def plot_dendrogram(linkage, labels, path):
    ensure_dir(os.path.dirname(path))
    plt.figure(figsize=(12,8))
    dendrogram(linkage, labels=labels, leaf_rotation=90)
    plt.title('Feature Correlation Dendrogram')
    plt.tight_layout()
    plt.savefig(path, dpi=300); plt.close()
