import numpy as np
from explainers.base_explainer import BaseExplainer


class PermutationExplainer(BaseExplainer):
    """
    Explainer basado en Permutation Importance local.

    Para cada feature i, permuta sus valores en un subconjunto de puntos
    del vecindario de la instancia y mide el cambio en P(clase_predicha).
    El resultado es un vector de importancias con signo:

        valor[i] = mean_j [ P(clase | x_j) - P(clase | x_j con feature_i permutada) ]

    - valor[i] > 0: permutar la feature REDUCE P(clase) → la feature es
      positiva para la predicción actual.
    - valor[i] < 0: permutar la feature AUMENTA P(clase) → la feature
      está actuando en contra de la clase predicha (caso raro en instancias
      bien clasificadas, relevante en instancias de frontera).
    - valor[i] ≈ 0: la feature no afecta la predicción localmente.

    Diferencias clave respecto a SHAP y LIME:
    - No asume linealidad local (LIME sí lo hace).
    - No usa gradientes (SHAP GradientExplainer sí los usa).
    - Mide el efecto real de eliminar la información de una feature
      (mediante permutación) en el vecindario concreto de la instancia.
    - Es interpretable directamente como "cuánto cambia P si esta feature
      se vuelve irrelevante".

    Parámetros:
        feature_names:   nombres de las features (opcional, para logging).
        n_neighbors:     número de vecinos del background para el probe set.
                         Más vecinos → estimación más estable pero más lenta.
        n_permutations:  número de permutaciones por feature por vecino.
                         Más permutaciones → menor varianza pero más tiempo.
        random_state:    semilla base para reproducibilidad. Se combina con
                         el ID de la instancia para que instancias distintas
                         produzcan permutaciones distintas pero reproducibles.
    """

    def __init__(self, feature_names=None, n_neighbors=15,
                 n_permutations=5, random_state=42):
        super().__init__(name="permutation", scope="local")
        self.feature_names   = feature_names
        self.n_neighbors     = n_neighbors
        self.n_permutations  = n_permutations
        self.random_state    = random_state

        self.background_data = None
        self._feature_std    = None

    # ── Background ────────────────────────────────────────────────────────────

    def set_background(self, X):
        self.background_data = np.asarray(X, dtype=float)
        std = np.std(self.background_data, axis=0)
        self._feature_std = np.where(std > 1e-8, std, 1.0)

    def invalidate(self):
        # PermutationExplainer no guarda estado del modelo — siempre recomputa
        pass

    # ── Core ──────────────────────────────────────────────────────────────────

    def _get_probe_set(self, instance):
        """
        Selecciona N vecinos más cercanos del background para la instancia.

        Distancia euclidiana normalizada por std del background. Los vecinos
        más cercanos comparten el contexto local de la instancia — la frontera
        de decisión es la misma, así que las permutaciones miden el efecto
        real en esa zona.
        """
        bg   = self.background_data
        inst = np.asarray(instance, dtype=float).reshape(1, -1)

        norm_bg = (bg - inst) / (self._feature_std + 1e-8)
        dists   = np.linalg.norm(norm_bg, axis=1)

        k = min(self.n_neighbors, len(bg))
        neighbor_idxs = np.argsort(dists)[:k]

        # Incluir siempre la instancia concreta en el probe set
        return np.vstack([inst, bg[neighbor_idxs]])

    def _permute_feature(self, probe_set, feature_idx, rng):
        """
        Genera una versión del probe set con la feature i permutada.

        La permutación mezcla los valores de la feature entre los puntos
        del probe set — la feature pierde toda correlación con las demás
        pero la distribución marginal se preserva.
        """
        perturbed = probe_set.copy()
        # Permutación de los valores de la feature entre todos los puntos
        perm_vals = rng.permutation(probe_set[:, feature_idx])
        perturbed[:, feature_idx] = perm_vals
        return perturbed

    def explain(self, model, X, instance_id=0, **kwargs) -> dict:
        if self.background_data is None:
            raise RuntimeError(
                "No se ha establecido background_data para PermutationExplainer."
            )

        X          = np.asarray(X, dtype=float)
        x_instance = X[instance_id:instance_id + 1]
        n_features = X.shape[1]

        pred_class = int(model.predict(x_instance)[0])
        probe_set  = self._get_probe_set(x_instance)

        # Semilla reproducible pero específica a la instancia
        rng = np.random.RandomState(self.random_state + instance_id)

        # ── P(clase_predicha) original para cada punto del probe set ─────────
        try:
            p_original = model.predict_proba(probe_set)[:, pred_class]
        except Exception:
            p_original = (model.predict(probe_set) == pred_class).astype(float)

        # ── Importancia por permutación para cada feature ────────────────────
        importances = np.zeros(n_features)

        for fi in range(n_features):
            drops = []
            for _ in range(self.n_permutations):
                perturbed = self._permute_feature(probe_set, fi, rng)
                try:
                    p_perturbed = model.predict_proba(perturbed)[:, pred_class]
                except Exception:
                    p_perturbed = (
                        model.predict(perturbed) == pred_class
                    ).astype(float)

                # Caída media de P(clase) al permutar esta feature
                # Positivo → feature ayuda a la predicción
                # Negativo → feature perjudica (raro, informativo en frontera)
                drops.append(float(np.mean(p_original - p_perturbed)))

            importances[fi] = float(np.mean(drops))

        # Normalizar: escalar al rango [-1, 1] para coherencia con SHAP/LIME
        max_abs = np.abs(importances).max()
        if max_abs > 1e-8:
            importances = importances / max_abs

        # ── Construir resultado ───────────────────────────────────────────────
        base = self._build_base_explanation(
            model=model, X=X, instance_id=instance_id
        )
        base["details"] = {
            "type":          "feature_importance",
            "feature_names": self.feature_names,
            "values":        importances.tolist(),
        }
        return base