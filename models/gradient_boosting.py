# gradient_boosting.py
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel


class GradientBoostingModel(SklearnModel):

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, **kwargs):
        super().__init__(
            GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                **kwargs
            )
        )
        self.hyperparams = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
        }
        self.reward_history = []
        self._lr_direction = 1  # +1 bajando lr, -1 subiéndola

    def evaluate_performance(self, X, y):
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1  = f1_score(y, y_pred, average="weighted")
        return 0.5 * acc + 0.5 * f1

    def adjust_from_feedback(self, signals, X_train, y_train, X_test, y_test):
        reward   = self.evaluate_performance(X_test, y_test)
        trend    = signals.get("trend", 0.0)       # del aggregator
        strategy = signals.get("strategy", "adjust")

        self.reward_history.append(reward)

        if strategy == "keep":
            return

        # ── Magnitud del ajuste ──────────────────────────────────────────
        base_factor = 1.04
        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            adaptive_factor = base_factor * (1.0 + np.tanh(abs(delta) * 3))
            adaptive_factor = np.clip(adaptive_factor, 1.01, 1.2)
        else:
            adaptive_factor = base_factor

        if strategy == "soft_adjust":
            adaptive_factor = 1 + (adaptive_factor - 1) / 2

        # ── Dirección basada en tendencia del grupo ───────────────────────
        # Si trend < 0 (el agente está empeorando según el aggregator)
        # invertimos la dirección del lr: en lugar de seguir bajándolo,
        # lo subimos para explorar otro régimen
        if trend < -0.05:
            self._lr_direction *= -1
            print(f"[AdvancedGB] Tendencia negativa ({trend:+.3f}) → "
                  f"invirtiendo dirección lr")

        if self._lr_direction > 0:
            # Dirección normal: bajar lr
            new_lr = self.hyperparams["learning_rate"] / adaptive_factor
        else:
            # Dirección invertida: subir lr
            new_lr = self.hyperparams["learning_rate"] * adaptive_factor

        self.hyperparams["learning_rate"] = float(np.clip(new_lr, 0.01, 0.9))
        self.hyperparams["n_estimators"]  = min(
            int(self.hyperparams["n_estimators"] * adaptive_factor), 300
        )

        self.model.set_params(
            **self.hyperparams,
            random_state=np.random.randint(0, 1000)
        )
        self.model.fit(X_train, y_train)

        print(f"[AdvancedGB] Ajuste | "
              f"n_estimators={self.hyperparams['n_estimators']} | "
              f"lr={self.hyperparams['learning_rate']:.4f} | "
              f"dir={'↓' if self._lr_direction > 0 else '↑'}")