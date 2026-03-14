
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel
from models.mentor import MentorMixin


class KNNModel(SklearnModel, MentorMixin):

    def __init__(self, n_neighbors=5, **kwargs):
        super().__init__(KNeighborsClassifier(n_neighbors=n_neighbors, **kwargs))
        self.hyperparams = {"n_neighbors": n_neighbors}
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
            # kNN: reducir k si va mal (mas fino), aumentar si va bien (mas estable)
            direction = -1 if delta < 0 else 1
            step = max(1, int(self.hyperparams["n_neighbors"] * 0.1))
            if strategy == "soft_adjust":
                step = max(1, step // 2)
        else:
            direction, step = -1, 1
        if trend < -0.05:
            direction = -1  # reducir k para mas sensibilidad local
        self.hyperparams["n_neighbors"] = int(np.clip(
            self.hyperparams["n_neighbors"] + direction * step, 1, 30))
        self.model.set_params(n_neighbors=self.hyperparams["n_neighbors"])
        if mentor_vector is not None and instance is not None and target_pred is not None:
            print(f"[kNN] Fine-tune con mentor | target={target_pred}")
            X_f, y_f, w_f = self._build_mentor_dataset(
                X_train, y_train, instance, target_pred, mentor_vector,
                n_replicas=20, base_weight=8.0)  # mas replicas para kNN
            self.model.fit(X_f, y_f)  # kNN no soporta sample_weight en fit
        else:
            self.model.fit(X_train, y_train)
        print(f"[kNN] k={self.hyperparams['n_neighbors']} mentor={mentor_vector is not None}")


