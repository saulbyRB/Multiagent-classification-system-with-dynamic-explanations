# explainers/lime_explainer.py

import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from explainers.base_explainer import BaseExplainer


class LimeExplainer(BaseExplainer):
    """
    Explainer LIME para clasificación tabular.
    """

    def __init__(
        self,
        training_data,
        feature_names,
        class_names,
        discretize_continuous=True
    ):
        super().__init__(name="lime", scope="local")

        self.feature_names = feature_names
        self.class_names = class_names

        self._explainer = LimeTabularExplainer(
            training_data=np.asarray(training_data),
            feature_names=feature_names,
            class_names=class_names,
            discretize_continuous=discretize_continuous,
            mode="classification"
        )

    def explain(self, model, X, instance_id=0, num_features=None, **kwargs) -> dict:
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
