# data/loaders/csv_loader.py

import pandas as pd
from .base_loader import BaseLoader

class CSVLoader(BaseLoader):
    """
    Loader de CSV genérico.
    """

    def __init__(self, path, target_column, feature_columns=None):
        self.path = path
        self.target_column = target_column
        self.feature_columns = feature_columns

    def load(self):
        df = pd.read_csv(self.path)

        if self.feature_columns is None:
            X = df.drop(columns=[self.target_column]).values
        else:
            X = df[self.feature_columns].values

        y = df[self.target_column].values

        metadata = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "feature_names": self.feature_columns or list(df.drop(columns=[self.target_column]).columns),
            "task": "classification"
        }

        return X, y, metadata

