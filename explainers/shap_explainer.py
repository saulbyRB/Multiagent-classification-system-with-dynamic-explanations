# explainers/shap_explainer.py

import numpy as np
import shap
from explainers.base_explainer import BaseExplainer


class ShapExplainer(BaseExplainer):
    """
    Explainer SHAP para clasificadores.
    Genera explicaciones locales basadas en importancia de features.
    """

    def __init__(self, background_data, feature_names=None):
        super().__init__(name="shap", scope="local")
        self.background_data = background_data
        self.feature_names = feature_names

        self._explainer = None

    def _build_explainer(self, model):
        """
        Construye el explainer SHAP dinámicamente.
        """

        def model_callable(X):
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            return model.predict(X)

        self._explainer = shap.Explainer(
            model_callable,
            self.background_data
        )

    def explain(self, model, X, instance_id=0, **kwargs) -> dict:
        """
        Genera explicación SHAP para una instancia concreta.
        """
        if self._explainer is None:
            self._build_explainer(model)

        X = np.asarray(X)
        x_instance = X[instance_id:instance_id + 1]

        shap_values = self._explainer(x_instance)

        values = shap_values.values
        if values.ndim == 3:  # multiclase
            values = values[..., 1]

        base = self._build_base_explanation(
            model=model,
            X=X,
            instance_id=instance_id
        )

        base["details"] = {
            "type": "feature_importance",
            "feature_names": self.feature_names,
            "values": values.flatten().tolist()
        }

        return base
