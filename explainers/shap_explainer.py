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

    # ── Configuración de background local para sklearn ───────────────────────
    # Para modelos sklearn (GB, RF…), SHAP usa PermutationExplainer con el
    # background completo → importancias globales. Limitando el background
    # a los K vecinos más cercanos de la instancia, SHAP mide importancias
    # locales coherentes con lo que LIME también capta localmente.
    N_LOCAL_NEIGHBORS = 30   # vecinos para background local sklearn

    def _build_explainer(self, model):
        if self.background_data is None:
            raise RuntimeError("No se ha establecido background_data para SHAP.")

        self._current_model = model
        self._is_torch = self._is_torch_model(model)

        if self._is_torch:
            # ── GradientExplainer: usa backprop real de la red ────────────
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
            # ── Explainer local para sklearn ──────────────────────────────
            # background_data completo guardado para calcular vecinos en explain().
            # El explainer se construye con background global como fallback,
            # pero en explain() se reconstruye localmente si hay instancia.
            def model_callable(X):
                if hasattr(model, "predict_proba"):
                    return model.predict_proba(X)
                return model.predict(X)

            self._model_callable = model_callable
            # Explainer global como fallback (se usa si no hay instancia en explain)
            self._explainer = shap.Explainer(model_callable, self.background_data)

    def _build_local_explainer(self, model, instance):
        """
        Construye un shap.Explainer con background = K vecinos más cercanos
        de la instancia en el training set.

        Esto hace que SHAP mida importancias locales (qué features importan
        CERCA de esta instancia) en lugar de globales (qué features importan
        en todo el dataset). Resultado: mayor coherencia con LIME, que también
        opera localmente.
        """
        bg   = self.background_data
        inst = np.asarray(instance, dtype=float).reshape(1, -1)

        # Distancia euclidiana normalizada por std del background
        std  = np.std(bg, axis=0)
        std  = np.where(std > 1e-8, std, 1.0)
        norm = (bg - inst) / std
        dists = np.linalg.norm(norm, axis=1)

        k = min(self.N_LOCAL_NEIGHBORS, len(bg))
        neighbor_idxs = np.argsort(dists)[:k]
        local_bg      = bg[neighbor_idxs]

        def model_callable(X):
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X)
            return model.predict(X)

        return shap.Explainer(model_callable, local_bg)

    def explain(self, model, X, **kwargs) -> dict:
        instance_id = kwargs.get("instance_id", 0)

        X          = np.asarray(X)
        x_instance = X[instance_id:instance_id + 1]

        # Para modelos no-torch: reconstruir explainer local en cada llamada.
        # Es más costoso que reutilizar, pero garantiza que el background
        # refleja el vecindario real de la instancia actual, no el de la
        # instancia anterior. Para torch se reutiliza (GradientExplainer
        # no depende de la instancia para construirse).
        if self._current_model is not model:
            self._build_explainer(model)

        if not self._is_torch and self.background_data is not None:
            # Siempre local para sklearn
            active_explainer = self._build_local_explainer(model, x_instance)
        elif self._explainer is None:
            self._build_explainer(model)
            active_explainer = self._explainer
        else:
            active_explainer = self._explainer

        if self._is_torch:
            # GradientExplainer devuelve lista de arrays (uno por clase)
            device    = model.device
            nn_model  = model.nn_model
            nn_model.eval()

            x_tensor  = torch.tensor(x_instance, dtype=torch.float32,
                                     device=device)
            shap_vals = active_explainer.shap_values(x_tensor)
            # shap_vals: lista de [1 x n_features] (una por clase)
            pred_class = int(model.predict(x_instance)[0])

            if isinstance(shap_vals, list):
                values = np.array(shap_vals[pred_class]).flatten()
            else:
                # array [1, n_features, n_classes]
                values = shap_vals[0, :, pred_class]

        else:
            shap_values = active_explainer(x_instance)
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