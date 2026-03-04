import numpy as np
import shap
import torch
from explainers.base_explainer import BaseExplainer


class ShapExplainer(BaseExplainer):
    """
    Explainer SHAP para clasificadores.

    Para modelos PyTorch (TorchModel): usa GradientExplainer, que calcula
    gradientes reales de la red mediante backprop. Es más estable y fiel
    que PermutationExplainer/SamplingExplainer para redes neuronales.

    Para modelos sklearn u otros: usa shap.Explainer genérico (KernelExplainer
    o TreeExplainer según el modelo).
    """

    def __init__(self, feature_names=None):
        super().__init__(name="shap", scope="local")
        self.feature_names   = feature_names
        self.background_data = None
        self._explainer      = None
        self._current_model  = None
        self._is_torch       = False

    def set_background(self, X):
        self.background_data = np.asarray(X)
        self._explainer      = None
        self._current_model  = None

    def invalidate(self):
        self._explainer     = None
        self._current_model = None

    def _is_torch_model(self, model):
        """Detecta si el modelo es un TorchModel con nn_model accesible."""
        return hasattr(model, "nn_model") and hasattr(model, "device")

    def _build_explainer(self, model):
        if self.background_data is None:
            raise RuntimeError("No se ha establecido background_data para SHAP.")

        self._current_model = model
        self._is_torch = self._is_torch_model(model)

        if self._is_torch:
            # ── GradientExplainer: usa backprop real de la red ────────────
            # Ventajas frente a PermutationExplainer:
            # 1. Gradientes exactos en lugar de estimaciones por permutación
            # 2. Mucho más estable iteración a iteración
            # 3. Respeta la geometría interna de la red (activaciones, pesos)
            # Usamos un subconjunto del background para eficiencia (max 100)
            nn_model = model.nn_model
            device   = model.device
            nn_model.eval()

            bg = self.background_data
            if len(bg) > 100:
                idx = np.random.choice(len(bg), 100, replace=False)
                bg  = bg[idx]

            bg_tensor = torch.tensor(bg, dtype=torch.float32, device=device)
            self._explainer = shap.GradientExplainer(nn_model, bg_tensor)

        else:
            # ── Explainer genérico para sklearn u otros ───────────────────
            def model_callable(X):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)
                return model.predict(X)

            self._explainer = shap.Explainer(model_callable, self.background_data)

    def explain(self, model, X, **kwargs) -> dict:
        instance_id = kwargs.get("instance_id", 0)

        if self._explainer is None or self._current_model is not model:
            self._build_explainer(model)

        X          = np.asarray(X)
        x_instance = X[instance_id:instance_id + 1]

        if self._is_torch:
            # GradientExplainer devuelve lista de arrays (uno por clase)
            device    = model.device
            nn_model  = model.nn_model
            nn_model.eval()

            x_tensor  = torch.tensor(x_instance, dtype=torch.float32,
                                     device=device)
            shap_vals = self._explainer.shap_values(x_tensor)
            # shap_vals: lista de [1 x n_features] (una por clase)
            pred_class = int(model.predict(x_instance)[0])

            if isinstance(shap_vals, list):
                values = np.array(shap_vals[pred_class]).flatten()
            else:
                # array [1, n_features, n_classes]
                values = shap_vals[0, :, pred_class]

        else:
            shap_values = self._explainer(x_instance)
            values      = shap_values.values

            if values.ndim == 3:
                pred_class = int(model.predict(x_instance)[0])
                values = values[..., pred_class]

            values = values.flatten()

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        base = self._build_base_explanation(
            model=model, X=X, instance_id=instance_id
        )
        base["details"] = {
            "type":          "feature_importance",
            "feature_names": self.feature_names,
            "values":        values.flatten().tolist()
        }
        return base