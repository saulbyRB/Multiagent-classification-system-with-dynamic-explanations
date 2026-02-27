# advanced_rf_model.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel

class RandomForest(SklearnModel):
    """
    RandomForest con ajuste adaptativo basado en reward recibido.
    """

    def __init__(self, n_estimators=50, max_depth=None, **kwargs):
        super().__init__(RandomForestClassifier(n_estimators=n_estimators,
                                                max_depth=max_depth,
                                                **kwargs))
        self.hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth
        }
        self.reward_history = []

    def evaluate_performance(self, X, y):
        """
        Calcula reward compuesto para feedback
        """
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted")
        # reward ponderado: 70% acc, 30% f1
        reward = 0.7 * acc + 0.3 * f1
        return reward

    def adjust_from_feedback(self, signals, X_train, y_train, X_test, y_test):
        """
        Ajusta hiperparámetros usando reward adaptativo
        """
        # Calculamos reward actual
        reward = self.evaluate_performance(X_test, y_test)
        self.reward_history.append(reward)

        # Factor base
        base_factor = 1.05

        # Si tenemos al menos un histórico, calculamos delta
        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            # delta < 0 → empeora → aumento agresivo
            adaptive_factor = base_factor * (1.0 + np.tanh(-delta * 5))
            adaptive_factor = min(max(adaptive_factor, 1.01), 1.2)
        else:
            adaptive_factor = base_factor

        # Estrategia de ajuste según signals o decision
        strategy = signals.get("strategy", "adjust")  # keep | soft_adjust | adjust

        if strategy == "keep":
            print("[AdvancedRF] Estrategia: keep, no se ajusta")
            return

        # Ajuste hiperparámetros
        if strategy in ["adjust", "soft_adjust"]:
            factor = adaptive_factor if strategy == "adjust" else (1 + (adaptive_factor - 1)/2)

            # n_estimators
            self.hyperparams["n_estimators"] = min(
                int(self.hyperparams["n_estimators"] * factor), 200
            )

            # max_depth
            if self.hyperparams["max_depth"] is not None:
                new_depth = int(self.hyperparams["max_depth"] * factor)
                self.hyperparams["max_depth"] = max(new_depth, 2)

            # Reentrenamos modelo con nuevos hiperparámetros
            self.model.set_params(
                n_estimators=self.hyperparams["n_estimators"],
                max_depth=self.hyperparams["max_depth"],
                random_state=np.random.randint(0, 1000)
            )
            self.model.fit(X_train, y_train)

        print(f"[AdvancedRF] Ajuste realizado (factor={adaptive_factor:.3f}). "
              f"n_estimators={self.hyperparams['n_estimators']}, "
              f"max_depth={self.hyperparams['max_depth']}")
