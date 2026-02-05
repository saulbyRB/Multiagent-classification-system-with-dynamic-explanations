# data/loaders/sklearn_loader.py

from sklearn import datasets
from .base_loader import BaseLoader
import numpy as np

class SklearnLoader(BaseLoader):
    """
    Loader para datasets de sklearn.
    """

    def __init__(self, dataset_func):
        """
        dataset_func: función de sklearn que devuelve objeto con .data y .target
        """
        self.dataset_func = dataset_func

    def load(self):
        ds = self.dataset_func()
        X = ds.data
        y = ds.target
        metadata = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "feature_names": ds.feature_names if hasattr(ds, "feature_names") else [f"f{i}" for i in range(X.shape[1])],
            "task": "classification"
        }
        return X, y, metadata

