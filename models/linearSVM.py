# linear_svm.py
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel


class LinearSVM(SklearnModel):

    def __init__(self, C=1.0, max_iter=3000, **kwargs):
        super().__init__(
            LinearSVC(C=C, max_iter=max_iter, **kwargs)
        )
        self.hyperparams = {"C": C}
        self.reward_history = []
        self._C_direction = 1  # +1 aumentando C, -1 reduciéndolo

    def evaluate_performance(self, X, y):
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1  = f1_score(y, y_pred, average="weighted")
        return 0.5 * acc + 0.5 * f1

    def adjust_from_feedback(self, signals, X_train, y_train, X_test, y_test):
        reward   = self.evaluate_performance(X_test, y_test)
        trend    = signals.get("trend", 0.0)
        strategy = signals.get("strategy", "adjust")

        self.reward_history.append(reward)

        if strategy == "keep":
            print("[SVM] keep → sin cambios")
            return

        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            factor = 1.0 + np.tanh(-delta * 3)
        else:
            factor = 1.1

        if strategy == "soft_adjust":
            factor = 1 + (factor - 1) / 2

        # Si empeora según el aggregator, invertir dirección de C
        if trend < -0.05:
            self._C_direction *= -1
            print(f"[SVM] Tendencia negativa ({trend:+.3f}) → "
                  f"invirtiendo dirección C")

        if self._C_direction > 0:
            new_C = self.hyperparams["C"] * factor   # más margen blando
        else:
            new_C = self.hyperparams["C"] / factor   # más margen duro

        self.hyperparams["C"] = float(np.clip(new_C, 0.01, 20.0))

        self.model.set_params(C=self.hyperparams["C"])
        self.model.fit(X_train, y_train)

        print(f"[SVM] Ajuste: C={self.hyperparams['C']:.4f} "
              f"dir={'↑' if self._C_direction > 0 else '↓'}")