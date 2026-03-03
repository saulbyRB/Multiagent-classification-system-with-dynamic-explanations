import re
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from explainers.base_explainer import BaseExplainer


class LimeExplainer(BaseExplainer):
    """
    Explainer LIME para clasificación tabular.
    """

    def __init__(self, feature_names=None, class_names=None, discretize_continuous=True):
        super().__init__(name="lime", scope="local")
        self.feature_names = feature_names
        self.class_names = class_names
        self.discretize_continuous = discretize_continuous

        self.background_data = None
        self._explainer = None

    def set_background(self, X):
        self.background_data = np.asarray(X)
        self._explainer = None

    def _build_explainer(self, model):
        if self.background_data is None:
            raise RuntimeError("No se ha establecido background_data para LIME.")

        # En LimeExplainer._build_explainer()
        self._explainer = LimeTabularExplainer(
            training_data=self.background_data,
            feature_names=self.feature_names,
            class_names=self.class_names,
            discretize_continuous=False,   # ← evita el sesgo de binning
            mode="classification"
        )

    def _parse_feature_index(self, label):
        """
        Extrae el índice de feature de una etiqueta LIME.

        LIME puede devolver:
          - int directamente
          - nombre limpio: "proline"
          - condición discretizada: "proline > 755.00" / "755.00 < proline <= 900.00"

        Devuelve el índice en self.feature_names, o None si no se puede resolver.
        """
        if isinstance(label, int):
            return label

        if self.feature_names is None:
            return None

        # 1) Coincidencia exacta con el nombre de feature
        if label in self.feature_names:
            return self.feature_names.index(label)

        # 2) Buscar el nombre de feature dentro de la condición
        #    Ordenar por longitud descendente para evitar match parcial
        sorted_names = sorted(self.feature_names, key=len, reverse=True)
        for name in sorted_names:
            # Buscar como palabra completa (no subcadena de otro nombre)
            pattern = r'(?<![a-zA-Z_])' + re.escape(name) + r'(?![a-zA-Z_0-9])'
            if re.search(pattern, label):
                return self.feature_names.index(name)

        return None

    def explain(self, model, X, instance_id=0, num_features=None, **kwargs) -> dict:
        if self._explainer is None:
            self._build_explainer(model)

        X = np.asarray(X)
        x_instance = X[instance_id:instance_id + 1]

        lime_exp = self._explainer.explain_instance(
            x_instance.flatten(),
            model.predict_proba,
            num_features=num_features or X.shape[1]
        )

        values = np.zeros(X.shape[1])
        for label, val in lime_exp.as_list():
            idx = self._parse_feature_index(label)
            if idx is not None and 0 <= idx < X.shape[1]:
                values[idx] = val

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