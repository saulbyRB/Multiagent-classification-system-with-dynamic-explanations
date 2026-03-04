import re
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from explainers.base_explainer import BaseExplainer


class LimeExplainer(BaseExplainer):
    """
    Explainer LIME para clasificación tabular.

    Mejoras para redes neuronales (modelos con frontera de decisión no-lineal):

    1. kernel_width adaptativo: para redes neuronales se usa un kernel_width
       más estrecho (0.40 * sqrt(n_features)) en lugar del default de LIME
       (0.75 * sqrt(n_features)). Esto reduce el vecindario de linealización
       y evita que features de alta varianza (como proline) dominen por el
       tamaño del muestreo, no por su relevancia real para el modelo.

    2. num_samples aumentado: más muestras de perturbación compensan el
       kernel más estrecho y reducen la varianza de la estimación local.

    3. sample_around_instance=True: fuerza a LIME a muestrear perturbaciones
       centradas en la instancia concreta, no en la media del training set.
       Crítico para instancias en zonas no-lineales de la frontera.
    """

    _KERNEL_WIDTH_FACTOR = {
        "default": 0.75,
        "torch":   0.40,
    }
    _NUM_SAMPLES = {
        "default": 5000,
        "torch":   8000,
    }

    def __init__(self, feature_names=None, class_names=None,
                 discretize_continuous=False):
        super().__init__(name="lime", scope="local")
        self.feature_names         = feature_names
        self.class_names           = class_names
        self.discretize_continuous = discretize_continuous

        self.background_data = None
        self._explainer      = None
        self._model_type     = "default"
        self._n_features     = None

    def set_background(self, X):
        self.background_data = np.asarray(X)
        self._explainer      = None
        self._n_features     = self.background_data.shape[1]

    def _is_torch_model(self, model):
        return hasattr(model, "nn_model") and hasattr(model, "device")

    def _build_explainer(self, model):
        if self.background_data is None:
            raise RuntimeError("No se ha establecido background_data para LIME.")

        self._model_type = "torch" if self._is_torch_model(model) else "default"

        n_feat       = self.background_data.shape[1]
        kw_factor    = self._KERNEL_WIDTH_FACTOR[self._model_type]
        kernel_width = kw_factor * np.sqrt(n_feat)

        self._explainer = LimeTabularExplainer(
            training_data=self.background_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            discretize_continuous=self.discretize_continuous,
            mode="classification",
            kernel_width=kernel_width,
            sample_around_instance=True,
            random_state=42,
        )

    def _parse_feature_index(self, label):
        if isinstance(label, int):
            return label
        if self.feature_names is None:
            return None
        if label in self.feature_names:
            return self.feature_names.index(label)
        sorted_names = sorted(self.feature_names, key=len, reverse=True)
        for name in sorted_names:
            pattern = r'(?<![a-zA-Z_])' + re.escape(name) + r'(?![a-zA-Z_0-9])'
            if re.search(pattern, label):
                return self.feature_names.index(name)
        return None

    def explain(self, model, X, instance_id=0, num_features=None, **kwargs) -> dict:
        if self._explainer is None:
            self._build_explainer(model)

        X          = np.asarray(X)
        x_instance = X[instance_id:instance_id + 1]

        num_samples = self._NUM_SAMPLES[self._model_type]

        lime_exp = self._explainer.explain_instance(
            x_instance.flatten(),
            model.predict_proba,
            num_features=num_features or X.shape[1],
            num_samples=num_samples,
        )

        values = np.zeros(X.shape[1])
        for label, val in lime_exp.as_list():
            idx = self._parse_feature_index(label)
            if idx is not None and 0 <= idx < X.shape[1]:
                values[idx] = val

        base = self._build_base_explanation(
            model=model, X=X, instance_id=instance_id
        )
        base["details"] = {
            "type":          "feature_importance",
            "feature_names": self.feature_names,
            "values":        values.flatten().tolist()
        }
        return base