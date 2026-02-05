# models/base_model.py

from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Interfaz base para modelos de Machine Learning.

    Esta clase define el contrato mínimo que deben cumplir
    todos los clasificadores utilizados por agentes del sistema.
    """

    def __init__(self, name: str):
        self.name = name
        self.is_trained = False

    # ==================== Entrenamiento ====================

    @abstractmethod
    def fit(self, X, y):
        """
        Entrena el modelo.
        """
        pass

    # ==================== Predicción ====================

    @abstractmethod
    def predict(self, X):
        """
        Devuelve la clase predicha.
        """
        pass

    def predict_proba(self, X):
        """
        Devuelve probabilidades de clase si están disponibles.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} no implementa predict_proba()"
        )

    # ==================== Confianza ====================

    def get_confidence(self, X):
        """
        Devuelve una medida escalar de confianza asociada a la predicción.
        Por defecto: máxima probabilidad.
        """
        if hasattr(self, "predict_proba"):
            proba = self.predict_proba(X)
            if proba is None:
                return None
            return np.max(proba, axis=1)
        return None

    # ==================== Metadata ====================

    def get_metadata(self) -> dict:
        """
        Devuelve información descriptiva del modelo.
        """
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "trained": self.is_trained
        }
