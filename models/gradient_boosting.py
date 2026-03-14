
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel
from models.mentor import MentorMixin


class GradientBoostingModel(SklearnModel, MentorMixin):

    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, **kwargs):
        super().__init__(GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, **kwargs))
        self.hyperparams = {"n_estimators": n_estimators,
                            "learning_rate": learning_rate,
                            "max_depth": max_depth}
        self.reward_history = []

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
            return
        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            factor = float(np.clip(1.0 + np.tanh(abs(delta) * 3) * 0.15, 1.01, 1.2))
        else:
            factor = 1.05
        if trend < -0.05:
            factor = 1.0 + (factor - 1.0) * 0.3
        if strategy == "soft_adjust":
            factor = 1.0 + (factor - 1.0) * 0.5
        hp = self.hyperparams
        hp["n_estimators"]   = int(np.clip(hp["n_estimators"] * factor, 10, 300))
        hp["learning_rate"]  = float(np.clip(hp["learning_rate"] / factor, 0.01, 0.5))
        hp["max_depth"]      = int(np.clip(hp["max_depth"], 2, 8))
        self.model.set_params(**hp)
        if mentor_vector is not None and instance is not None and target_pred is not None:
            print(f"[GB] Fine-tune con mentor | target={target_pred}")
            X_f, y_f, w_f = self._build_mentor_dataset(X_train, y_train, instance, target_pred, mentor_vector)
            self.model.fit(X_f, y_f, sample_weight=w_f)
        else:
            self.model.fit(X_train, y_train)
        print(f"[GB] n_est={hp['n_estimators']} lr={hp['learning_rate']:.4f} mentor={mentor_vector is not None}")
