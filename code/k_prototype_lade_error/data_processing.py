import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


class DataPreprocessor:
    """Handles loading, cleaning, encoding, and scaling of the dataset."""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None

    def load_data(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path, low_memory=False)
        self.df.drop_duplicates(inplace=True)
        return self.df

    def encode_and_scale(self, categorical_indices: list[int], numeric_indices: list[int]) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        encoder = LabelEncoder()
        scaler = MinMaxScaler()

        # Encode categorical columns
        for idx in categorical_indices:
            self.df.iloc[:, idx] = encoder.fit_transform(self.df.iloc[:, idx].astype(str))

        # Scale numeric columns
        self.df.iloc[:, numeric_indices] = scaler.fit_transform(self.df.iloc[:, numeric_indices].astype(float))

        return self.df
