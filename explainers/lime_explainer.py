# explainers/lime_explainer.py

import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from explainers.base_explainer import BaseExplainer


class LimeExplainer(BaseExplainer):
    """
    Explainer LIME para clasificación tabular.
    """

    def __init__(self, feature_names=None, class_names=None, discretize_continuous=True):
        """
        feature_names : lista de nombres de features
        class_names : lista de nombres de clases
        discretize_continuous : bool, si discretizar variables continuas
        """
        super().__init__(name="lime", scope="local")
        self.feature_names = feature_names
        self.class_names = class_names
        self.discretize_continuous = discretize_continuous

        self.background_data = None
        self._explainer = None

    def set_background(self, X):
        """
        Establece los datos de entrenamiento / background para LIME.
        """
        self.background_data = np.asarray(X)
        self._explainer = None  # forzar reconstrucción

    def _build_explainer(self, model):
        """
        Construye el LimeTabularExplainer dinámicamente.
        """
        if self.background_data is None:
            raise RuntimeError("No se ha establecido background_data para LIME.")

        self._explainer = LimeTabularExplainer(
            training_data=self.background_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            discretize_continuous=self.discretize_continuous,
            mode="classification"
        )

    def explain(self, model, X, instance_id=0, num_features=None, **kwargs) -> dict:
        """
        Genera explicación LIME para una instancia concreta.
        """
        if self._explainer is None:
            self._build_explainer(model)

        X = np.asarray(X)
        x_instance = X[instance_id:instance_id + 1]

        lime_exp = self._explainer.explain_instance(
            x_instance.flatten(),
            model.predict_proba,
            num_features=num_features or X.shape[1]
        )

        # Convertir a lista de valores (importancia de features)
        values = np.zeros(X.shape[1])
        for idx, val in lime_exp.as_list():
            if isinstance(idx, int):
                values[idx] = val
            else:
                # si viene el nombre, buscar índice
                try:
                    values[self.feature_names.index(idx)] = val
                except Exception:
                    pass

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
