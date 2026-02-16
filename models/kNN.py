# advanced_knn_model.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel


class KNN(SklearnModel):
    """
    k-NN adaptativo con ajuste del vecindario.
    Explorador local de patrones.
    """

    def __init__(self, n_neighbors=5, weights="distance", **kwargs):
        super().__init__(
            KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                **kwargs
            )
        )

        self.hyperparams = {
            "n_neighbors": n_neighbors,
            "weights": weights
        }
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
            print("[kNN] keep → sin cambios")
            return

        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            # empeora → más vecinos (más suavidad)
            factor = 1.0 + np.tanh(-delta * 4)
        else:
            factor = 1.2

        if strategy == "soft_adjust":
            factor = 1 + (factor - 1) / 2

        new_k = int(self.hyperparams["n_neighbors"] * factor)
        self.hyperparams["n_neighbors"] = int(np.clip(new_k, 3, 30))

        self.model.set_params(n_neighbors=self.hyperparams["n_neighbors"])
        self.model.fit(X_train, y_train)

        print(f"[kNN] Ajuste: k={self.hyperparams['n_neighbors']}")
