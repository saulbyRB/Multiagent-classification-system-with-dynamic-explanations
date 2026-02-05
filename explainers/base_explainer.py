# explainers/base_explainer.py

from abc import ABC, abstractmethod


class BaseExplainer(ABC):
    """
    Interfaz base para mecanismos de explicabilidad.

    Un explainer recibe un modelo entrenado y una instancia
    (o conjunto de instancias) y devuelve una explicación
    estructurada y serializable.
    """

    def __init__(self, name: str, scope: str = "local"):
        """
        Parameters
        ----------
        name : str
            Nombre del explainer (e.g. 'shap', 'lime')
        scope : str
            'local' o 'global'
        """
        self.name = name
        self.scope = scope

    # ==================== Explicación ====================

    @abstractmethod
    def explain(self, model, X, **kwargs) -> dict:
        """
        Genera una explicación para una o varias instancias.

        Parameters
        ----------
        model : BaseModel
            Modelo entrenado
        X : array-like
            Instancia(s) a explicar

        Returns
        -------
        dict
            Explicación estructurada
        """
        pass

    # ==================== Metadata ====================

    def get_metadata(self) -> dict:
        """
        Devuelve metadata del explainer.
        """
        return {
            "name": self.name,
            "scope": self.scope,
            "class": self.__class__.__name__
        }
    
    def _build_base_explanation(
        self,
        model,
        X,
        instance_id=None
    ) -> dict:
        x = X if instance_id is None else X[instance_id:instance_id + 1]

        prediction = model.predict(x)
        confidence = model.get_confidence(x)

        return {
            "explainer": self.name,
            "scope": self.scope,
            "instance_id": instance_id,
            "prediction": prediction.tolist() if prediction is not None else None,
            "confidence": confidence.tolist() if confidence is not None else None,
            "model": model.name
        }

