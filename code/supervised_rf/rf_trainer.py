from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc, hamming_loss
from sklearn.model_selection import train_test_split


@dataclass
class RandomForestTrainer:
    """Trains/evaluates a RandomForest and computes permutation importances."""
    test_size: float = 0.3
    random_state: int = 42
    oob_score: bool = True

    def split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, ...]:
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)

    def sweep_estimators(
        self, X_train, y_train, X_test, y_test, grid: List[int]
    ) -> pd.DataFrame:
        rows = []
        for n in grid:
            clf = RandomForestClassifier(
                n_estimators=n, n_jobs=-1, random_state=20,
                oob_score=self.oob_score, bootstrap=True, warm_start=True
            )
            clf.fit(X_train, y_train)
            # Using predicted probabilities for ROC (better than labels)
            if hasattr(clf, "predict_proba"):
                train_score = self._roc_auc(y_train, clf.predict_proba(X_train)[:, 1])
                test_score = self._roc_auc(y_test, clf.predict_proba(X_test)[:, 1])
            else:
                train_score = self._roc_auc(y_train, clf.predict(X_train))
                test_score = self._roc_auc(y_test, clf.predict(X_test))
            rows.append({"n_estimators": n, "train_auc": train_score, "test_auc": test_score})
        return pd.DataFrame(rows)

    def fit_final(
        self, X_train, y_train, n_estimators: int = 200, max_depth: int | None = None
    ) -> RandomForestClassifier:
        clf = RandomForestClassifier(
            n_estimators=n_estimators, n_jobs=-1, random_state=15,
            oob_score=self.oob_score, bootstrap=True, max_depth=max_depth
        )
        clf.fit(X_train, y_train)
        return clf

    def metrics(self, clf, X_test, y_test) -> dict:
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        else:
            y_score = clf.predict(X_test)
        y_pred = (y_score >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        return {
            "auc": auc(fpr, tpr),
            "hamming_loss": hamming_loss(y_test, y_pred),
            "oob_score": getattr(clf, "oob_score_", None),
            "roc": (fpr, tpr),
        }

    def permutation_importance(self, clf, X_test, y_test, repeats: int = 50, seed: int = 11):
        result = permutation_importance(clf, X_test, y_test, n_repeats=repeats, random_state=seed, n_jobs=-1)
        importances = pd.Series(result.importances_mean, index=X_test.columns)
        norm = (importances - importances.min()) / (importances.max() - importances.min() + 1e-12)
        return norm.sort_values(ascending=False)

    @staticmethod
    def _roc_auc(y_true, y_score) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        return auc(fpr, tpr)
