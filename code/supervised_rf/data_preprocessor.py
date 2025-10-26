from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


@dataclass
class DataPreprocessor:
    """Prepares data for supervised learning (encoding, scaling, target split)."""
    categorical_slices: Sequence[slice]
    drop_columns: Iterable[str]
    target_col: str

    def load(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path, low_memory=False)
        return df

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        le = LabelEncoder()
        for s in self.categorical_slices:
            for col in df.iloc[:, s].columns:
                df[col] = le.fit_transform(df[col].astype(str))
        return df

    def scale_all(self, df: pd.DataFrame) -> pd.DataFrame:
        # After label-encoding, we can scale all numeric columns.
        scaler = MinMaxScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns].astype(float))
        return df

    def finalize(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        df = df.drop(columns=list(self.drop_columns), errors="ignore")
        df = df.fillna(-0.1)
        y = df[self.target_col].astype(int)
        X = df.drop(columns=[self.target_col])
        return X, y
