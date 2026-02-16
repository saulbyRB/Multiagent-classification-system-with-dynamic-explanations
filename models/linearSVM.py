# advanced_svm_model.py
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel


class LinearSVM(SklearnModel):
    """
    Linear SVM con ajuste adaptativo del margen (C).
    Especialista en fronteras robustas.
    """

    def __init__(self, C=1.0, max_iter=3000, **kwargs):
        super().__init__(
            LinearSVC(C=C, max_iter=max_iter, **kwargs)
        )

        self.hyperparams = {"C": C}
        self.reward_history = []

    def evaluate_performance(self, X, y):
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1  = f1_score(y, y_pred, average="weighted")
        return 0.5 * acc + 0.5 * f1

    def adjust_from_feedback(self, signals, X_train, y_train, X_test, y_test):
        reward = self.evaluate_performance(X_test, y_test)
        self.reward_history.append(reward)

        strategy = signals.get("strategy", "adjust")
        if strategy == "keep":
            print("[SVM] keep → sin cambios")
            return

        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            # margen más grande si empeora
            factor = 1.0 + np.tanh(-delta * 3)
        else:
            factor = 1.1

        if strategy == "soft_adjust":
            factor = 1 + (factor - 1) / 2

        self.hyperparams["C"] = np.clip(
            self.hyperparams["C"] * factor, 0.01, 20.0
        )

        self.model.set_params(C=self.hyperparams["C"])
        self.model.fit(X_train, y_train)

        print(f"[SVM] Ajuste: C={self.hyperparams['C']:.4f}")
