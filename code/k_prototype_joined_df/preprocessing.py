import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    """Handles data loading, cleaning, encoding, and scaling."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None

    def load_data(self, drop_indices: list[int]) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path)
        self.df.drop(columns=self.df.columns[drop_indices], inplace=True)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def convert_types(self, categorical_slices: list[slice], numeric_slices: list[slice]) -> None:
        for cat_slice in categorical_slices:
            self.df.iloc[:, cat_slice] = self.df.iloc[:, cat_slice].astype(str)
        for num_slice in numeric_slices:
            self.df.iloc[:, num_slice] = self.df.iloc[:, num_slice].astype(float)

    def fill_missing(self, categorical_slices: list[slice], fill_value: str = "missing value") -> None:
        for cat_slice in categorical_slices:
            self.df.iloc[:, cat_slice] = self.df.iloc[:, cat_slice].fillna(fill_value)

    def scale_columns(self, column_indices: list[int]) -> pd.DataFrame:
        scaler = MinMaxScaler()
        data = self.df.values
        data[:, column_indices] = scaler.fit_transform(data[:, column_indices])
        self.df = pd.DataFrame(data, columns=self.df.columns)
        return self.df
