# logistic_regression.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel


class Logisticregression(SklearnModel):

    def __init__(self, C=1.0, penalty="l2", max_iter=500, **kwargs):
        super().__init__(
            LogisticRegression(
                C=C, penalty=penalty,
                solver="liblinear", max_iter=max_iter, **kwargs
            )
        )
        self.hyperparams = {"C": C, "penalty": penalty}
        self.reward_history = []

    def evaluate_performance(self, X, y):
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1  = f1_score(y, y_pred, average="weighted")
        return 0.6 * acc + 0.4 * f1

    def adjust_from_feedback(self, signals, X_train, y_train, X_test, y_test):
        reward   = self.evaluate_performance(X_test, y_test)
        trend    = signals.get("trend", 0.0)
        strategy = signals.get("strategy", "adjust")

        self.reward_history.append(reward)

        if strategy == "keep":
            print("[LogReg] keep → sin cambios")
            return

        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            factor = 1.0 + np.clip(delta, -0.1, 0.1)
        else:
            factor = 1.05

        if strategy == "soft_adjust":
            factor = 1 + (factor - 1) / 2

        # Si empeora según el aggregator, invertir el ajuste de C:
        # en lugar de reducir regularización (C↑), aumentarla (C↓)
        if trend < -0.05:
            factor = 1 / factor
            print(f"[LogReg] Tendencia negativa ({trend:+.3f}) → "
                  f"aumentando regularización")

        self.hyperparams["C"] = float(np.clip(
            self.hyperparams["C"] * factor, 0.01, 10.0
        ))

        self.model.set_params(C=self.hyperparams["C"])
        self.model.fit(X_train, y_train)

        print(f"[LogReg] Ajuste: C={self.hyperparams['C']:.4f}")