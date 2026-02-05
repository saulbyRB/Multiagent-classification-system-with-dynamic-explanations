# models/base_model.py

from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):

    def __init__(self, name: str):
        self.name = name
        self.is_trained = False
        self.state = {
            "trained": False,
            "n_fit": 0,
            "n_predict": 0,
            "last_prediction": None,
            "last_explanation": None
        }

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def predict_proba(self, X):
        raise NotImplementedError

    def get_confidence(self, X):
        try:
            proba = self.predict_proba(X)
        except NotImplementedError:
            return None
        if proba is None:
            return None
        return np.max(proba, axis=1)

    def capabilities(self) -> dict:
        return {
            "predict_proba": False,
            "confidence": True,
            "shap": False,
            "lime": False,
            "counterfactual": False
        }

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "trained": self.is_trained,
            "capabilities": self.capabilities()
        }

