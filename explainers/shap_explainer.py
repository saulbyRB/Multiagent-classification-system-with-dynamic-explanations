# explainers/shap_explainer.py

import numpy as np
import shap
from explainers.base_explainer import BaseExplainer


class ShapExplainer(BaseExplainer):
    """
    Explainer SHAP para clasificadores.
    Genera explicaciones locales basadas en importancia de features.
    """

    def __init__(self, feature_names=None):
        super().__init__(name="shap", scope="local")
        self.feature_names = feature_names
        self.background_data = None
        self._explainer = None
        self._current_model = None  # referencia al modelo actual

    def set_background(self, X):
        """
        Establece los datos de background para SHAP.
        """
        self.background_data = np.asarray(X)
        self._explainer = None
        self._current_model = None

    def invalidate(self):
        """
        Fuerza reconstrucción del explainer en la próxima llamada.
        Debe llamarse desde ClassifierAgent cada vez que el modelo se re-entrena.
        """
        self._explainer = None
        self._current_model = None

    def _build_explainer(self, model):
        """
        Construye el explainer SHAP vinculado al modelo actual.
        """
        if self.background_data is None:
            raise RuntimeError("No se ha establecido background_data para SHAP.")

        # Guardamos referencia al modelo para detectar si cambia
        self._current_model = model

        def model_callable(X):
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            return model.predict(X)

        self._explainer = shap.Explainer(model_callable, self.background_data)

    def explain(self, model, X, instance_id=0, **kwargs) -> dict:
        """
        Genera explicación SHAP para una instancia concreta.
        """
        # Reconstruir si no hay explainer o si el modelo cambió
        if self._explainer is None or self._current_model is not model:
            self._build_explainer(model)

        X = np.asarray(X)
        x_instance = X[instance_id:instance_id + 1]

        shap_values = self._explainer(x_instance)
        values = shap_values.values

        if values.ndim == 3:
            # Multiclase: tomamos los SHAP values de la clase predicha
            # en lugar de siempre clase 1 (que era incorrecto para Iris con 3 clases)
            pred_class = int(model.predict(x_instance)[0])
            values = values[..., pred_class]  # shape: (1, n_features)

        # Clip: similitud coseno necesita vectores finitos, SHAP puede dar NaN o muy pequeños
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

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