# knn.py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel


class KNN(SklearnModel):

    def __init__(self, n_neighbors=5, weights="distance", **kwargs):
        super().__init__(
            KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, **kwargs)
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
        reward   = self.evaluate_performance(X_test, y_test)
        trend    = signals.get("trend", 0.0)
        strategy = signals.get("strategy", "adjust")

        self.reward_history.append(reward)

        if strategy == "keep":
            print("[kNN] keep → sin cambios")
            return

        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            factor = 1.0 + np.tanh(-delta * 4)
        else:
            factor = 1.2

        if strategy == "soft_adjust":
            factor = 1 + (factor - 1) / 2

        # Si el aggregator dice que empeoramos, reducimos k en lugar
        # de aumentarlo — buscamos fronteras más locales
        if trend < -0.05:
            factor = 1 / factor  # invertir: menos vecinos
            print(f"[kNN] Tendencia negativa ({trend:+.3f}) → "
                  f"reduciendo k (más local)")

        new_k = int(self.hyperparams["n_neighbors"] * factor)
        self.hyperparams["n_neighbors"] = int(np.clip(new_k, 3, 30))

        self.model.set_params(n_neighbors=self.hyperparams["n_neighbors"])
        self.model.fit(X_train, y_train)

        print(f"[kNN] Ajuste: k={self.hyperparams['n_neighbors']}")