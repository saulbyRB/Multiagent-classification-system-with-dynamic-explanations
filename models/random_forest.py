# random_forest.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel


class RandomForest(SklearnModel):

    def __init__(self, n_estimators=50, max_depth=None, **kwargs):
        super().__init__(
            RandomForestClassifier(n_estimators=n_estimators,
                                   max_depth=max_depth, **kwargs)
        )
        self.hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth
        }
        self.reward_history = []

    def evaluate_performance(self, X, y):
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1  = f1_score(y, y_pred, average="weighted")
        return 0.7 * acc + 0.3 * f1

    def adjust_from_feedback(self, signals, X_train, y_train, X_test, y_test):
        reward   = self.evaluate_performance(X_test, y_test)
        trend    = signals.get("trend", 0.0)
        strategy = signals.get("strategy", "adjust")

        self.reward_history.append(reward)

        if strategy == "keep":
            print("[AdvancedRF] Estrategia: keep, no se ajusta")
            return

        base_factor = 1.05
        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            adaptive_factor = base_factor * (1.0 + np.tanh(-delta * 5))
            adaptive_factor = np.clip(adaptive_factor, 1.01, 1.2)
        else:
            adaptive_factor = base_factor

        # Si el aggregator indica que este agente está empeorando,
        # reducimos el factor (menos árboles, más conservador)
        # en lugar de seguir creciendo agresivamente
        if trend < -0.05:
            adaptive_factor = 1 + (adaptive_factor - 1) * 0.25
            print(f"[AdvancedRF] Tendencia negativa ({trend:+.3f}) → "
                  f"frenando crecimiento (factor={adaptive_factor:.3f})")

        factor = adaptive_factor if strategy == "adjust" else (1 + (adaptive_factor - 1) / 2)

        # Solo crecer si el beneficio marginal justifica el coste
        if self.hyperparams["n_estimators"] >= 150 and abs(delta) < 0.01:
            print("[AdvancedRF] Plateau detectado → manteniendo n_estimators")
            self.model.fit(X_train, y_train)
            return

        if self.hyperparams["max_depth"] is not None:
            self.hyperparams["max_depth"] = max(
                int(self.hyperparams["max_depth"] * factor), 2
            )

        self.model.set_params(
            n_estimators=self.hyperparams["n_estimators"],
            max_depth=self.hyperparams["max_depth"],
            random_state=np.random.randint(0, 1000)
        )
        self.model.fit(X_train, y_train)

        print(f"[AdvancedRF] Ajuste (factor={adaptive_factor:.3f}) | "
              f"n_estimators={self.hyperparams['n_estimators']} | "
              f"max_depth={self.hyperparams['max_depth']}")