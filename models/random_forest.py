
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel
from models.mentor import MentorMixin


class RandomForest(SklearnModel, MentorMixin):

    # ── Configuración de shiperparámetros ──────────────────────────────────────
    # n_estimators_min: mínimo de árboles en fine-tune.
    # Con 5 árboles la votación es frágil (3 vs 2 decide la clase).
    # Con 20+ la mayoría es robusta ante réplicas ruidosas del mentor.
    N_ESTIMATORS_MIN  = 20
    # max_depth_max: techo para evitar sobreajuste acumulativo.
    MAX_DEPTH_MIN     = 3      # profundidad mínima operativa: depth=2 es insuficiente
    MAX_DEPTH_MAX     = 8
    # random_state_base: semilla base para reproducibilidad.
    # Se incrementa por iteración → determinista pero variable entre iters.
    RANDOM_STATE_BASE = 42

    def __init__(self, n_estimators=50, max_depth=None, **kwargs):
        # Si el caller pasa random_state explícitamente, respetarlo como base;
        # si no, usar RANDOM_STATE_BASE. En ambos casos eliminarlo de kwargs
        # para evitar colisión con el parámetro explícito de RandomForestClassifier.
        init_seed = kwargs.pop("random_state", self.RANDOM_STATE_BASE)
        super().__init__(RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=init_seed,
            **kwargs
        ))
        self.hyperparams    = {"n_estimators": n_estimators, "max_depth": max_depth}
        self.reward_history = []
        self._iter          = 0
        # Guardar la semilla inicial para la secuencia determinista
        self.RANDOM_STATE_BASE = init_seed

    def evaluate_performance(self, X, y):
        yp = self.model.predict(X)
        return 0.7 * accuracy_score(y, yp) + 0.3 * f1_score(y, yp, average="weighted")

    def adjust_from_feedback(self, signals, X_train, y_train, X_test, y_test):
        strategy      = signals.get("strategy", "adjust")
        trend         = signals.get("trend", 0.0)
        mentor_vector = signals.get("mentor_vector", None)
        instance      = signals.get("instance", None)
        target_pred   = signals.get("target_pred", None)

        reward = self.evaluate_performance(X_test, y_test)
        self.reward_history.append(reward)

        if strategy == "keep":
            self._iter += 1
            return

        # ── Factor de ajuste ──────────────────────────────────────────────────
        # Si delta < 0 (reward bajó): factor=1.0 → no crecer max_depth.
        # Evita el ciclo: baja reward → factor alto → más profundidad →
        # sobreajuste → baja reward → ...
        base = 1.05
        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            if delta < 0:
                factor = 1.0
            else:
                factor = float(np.clip(base * (1.0 + np.tanh(-delta * 5)), 1.01, 1.15))
        else:
            factor = base

        if trend < -0.05:
            factor = 1.0  # tendencia negativa sostenida: no expandir

        if strategy == "soft_adjust":
            factor = 1.0 + (factor - 1.0) * 0.4

        # ── max_depth con techo ───────────────────────────────────────────────
        if self.hyperparams["max_depth"] is not None:
            new_depth = int(self.hyperparams["max_depth"] * factor)
            self.hyperparams["max_depth"] = min(max(new_depth, self.MAX_DEPTH_MIN), self.MAX_DEPTH_MAX)

        # ── n_estimators mínimo seguro ────────────────────────────────────────
        n_est = max(self.hyperparams["n_estimators"], self.N_ESTIMATORS_MIN)

        # ── max_depth mínimo operativo (incluso cuando factor=1.0) ────────────
        # Garantiza que el modelo siempre tenga suficiente profundidad para
        # capturar interacciones de features en instancias frontera,
        # independientemente de si el factor de ajuste permitió crecer o no.
        if self.hyperparams["max_depth"] is not None:
            self.hyperparams["max_depth"] = max(
                self.hyperparams["max_depth"], self.MAX_DEPTH_MIN
            )

        # ── random_state determinista por iteración ───────────────────────────
        # seed = base + iter garantiza reproducibilidad sin varianza aleatoria.
        seed = self.RANDOM_STATE_BASE + self._iter

        self.model.set_params(
            n_estimators=n_est,
            max_depth=self.hyperparams["max_depth"],
            random_state=seed,
        )

        # ── Entrenamiento ─────────────────────────────────────────────────────
        if mentor_vector is not None and instance is not None and target_pred is not None:
            print(f"[RF] Fine-tune con mentor | target={target_pred}")
            X_f, y_f, w_f = self._build_mentor_dataset(
                X_train, y_train, instance, target_pred, mentor_vector,
                n_replicas=15,    # más réplicas → ancla más robusta
                base_weight=5.0,  # coherente con sklearn_mentor_mixin default
            )
            self.model.fit(X_f, y_f, sample_weight=w_f)
        else:
            self.model.fit(X_train, y_train)

        print(f"[RF] n_est={n_est} max_depth={self.hyperparams['max_depth']} "
              f"seed={seed} mentor={mentor_vector is not None}")

        self._iter += 1

