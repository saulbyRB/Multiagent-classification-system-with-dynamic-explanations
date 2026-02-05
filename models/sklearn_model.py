# models/sklearn_model.py

import numpy as np
from sklearn.base import BaseEstimator
from models.base_model import BaseModel


class SklearnModel(BaseModel):
    """
    Wrapper genérico para modelos de scikit-learn.

    Permite integrar clasificadores sklearn en el sistema
    multiagente mediante una interfaz común.
    """

    def __init__(self, name: str, model: BaseEstimator):
        super().__init__(name=name)

        if not isinstance(model, BaseEstimator):
            raise TypeError(
                "El modelo debe ser una instancia de sklearn.base.BaseEstimator"
            )

        self.model = model
        self.model_class = model.__class__.__name__

    # ==================== Entrenamiento ====================

    def fit(self, X, y):
        """
        Entrena el modelo sklearn.
        """
        self.model.fit(X, y)
        self.is_trained = True
        self.state["trained"] = True
        self.state["n_fit"] += 1

        return self

    # ==================== Predicción ====================

    def predict(self, X):
        """
        Devuelve la clase predicha.
        """
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Devuelve probabilidades de clase si el modelo las soporta.
        """
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado")

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)

        return None

    # ==================== Confianza ====================

    def get_confidence(self, X):
        """
        Devuelve la confianza asociada a la predicción.

        Estrategia:
        - Si hay predict_proba: max(prob)
        - Si hay decision_function: distancia normalizada
        - Si no: None
        """
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado")

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)
            return np.max(proba, axis=1)

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(X)
            scores = np.atleast_2d(scores)
            return np.max(np.abs(scores), axis=1)

        return None

    # ==================== Metadata ====================

    def get_metadata(self) -> dict:
        """
        Devuelve metadata enriquecida del modelo.
        """
        base_meta = super().get_metadata()

        base_meta.update({
            "backend": "scikit-learn",
            "model_class": self.model_class,
            "parameters": self.model.get_params()
        })

        return base_meta
    
    def capabilities(self):
        return {
            "predict_proba": hasattr(self.model, "predict_proba"),
            "confidence": True,
            "shap": True,
            "lime": True,
            "counterfactual": True
        }

