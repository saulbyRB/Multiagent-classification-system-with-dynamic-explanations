import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel


class GradientBoostingModel(SklearnModel):
    """
    Gradient Boosting con ajuste adaptativo
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        **kwargs
    ):
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

    def evaluate_performance(self, X, y):
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="weighted")
        return 0.5 * acc + 0.5 * f1

    def adjust_from_feedback(self, signals, X_train, y_train, X_test, y_test):
        reward = self.evaluate_performance(X_test, y_test)
        self.reward_history.append(reward)

        base_factor = 1.04

        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            adaptive_factor = base_factor * (1.0 + np.tanh(-delta * 3))
            adaptive_factor = np.clip(adaptive_factor, 1.01, 1.2)
        else:
            adaptive_factor = base_factor

        strategy = signals.get("strategy", "adjust")

        if strategy == "keep":
            return

        factor = adaptive_factor if strategy == "adjust" else (1 + (adaptive_factor - 1) / 2)

        self.hyperparams["n_estimators"] = min(
            int(self.hyperparams["n_estimators"] * factor), 300
        )

        self.hyperparams["learning_rate"] = max(
            self.hyperparams["learning_rate"] / factor, 0.01
        )

        self.model.set_params(**self.hyperparams,random_state=np.random.randint(0, 1000))
        self.model.fit(X_train, y_train)

        print(
            f"[AdvancedGB] Ajuste | "
            f"n_estimators={self.hyperparams['n_estimators']} | "
            f"lr={self.hyperparams['learning_rate']:.4f}"
        )
