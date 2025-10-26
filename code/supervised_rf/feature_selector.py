from collections import defaultdict

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


class FeatureSelector:
    def __init__(self, distance_threshold=1.1):
        self.distance_threshold = distance_threshold

    def select_features(self, X, mandatory_features=None, drop_if_exists=None):
        corr = spearmanr(X).correlation
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1)
        distance = 1 - np.abs(corr)
        linkage = hierarchy.ward(squareform(distance))

        cluster_ids = hierarchy.fcluster(linkage, self.distance_threshold, criterion='distance')
        cluster_map = defaultdict(list)
        for idx, cid in enumerate(cluster_ids):
            cluster_map[cid].append(idx)
        selected = [X.columns[v[0]] for v in cluster_map.values()]

        if drop_if_exists:
            selected = [f for f in selected if f not in drop_if_exists]
        if mandatory_features:
            for f in mandatory_features:
                if f not in selected:
                    selected.append(f)
        return selected, linkage
