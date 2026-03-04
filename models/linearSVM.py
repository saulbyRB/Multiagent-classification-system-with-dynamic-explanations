import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel
from models.mentor import MentorMixin


class LinearSVM(SklearnModel, MentorMixin):

    def __init__(self, C=1.0, max_iter=2000, **kwargs):
        base = LinearSVC(C=C, max_iter=max_iter, **kwargs)
        super().__init__(CalibratedClassifierCV(base))
        self.hyperparams = {"C": C, "max_iter": max_iter}
        self._base_params = kwargs
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
            c_factor = float(np.clip(1.0 + np.tanh(-delta * 5) * 0.3, 0.7, 1.5))
        else:
            c_factor = 1.1
        if trend < -0.05:
            c_factor = 1.0 + (c_factor - 1.0) * 0.3
        if strategy == "soft_adjust":
            c_factor = 1.0 + (c_factor - 1.0) * 0.5
        self.hyperparams["C"] = float(np.clip(self.hyperparams["C"] * c_factor, 0.001, 100.0))
        base = LinearSVC(C=self.hyperparams["C"],
                         max_iter=self.hyperparams["max_iter"],
                         **self._base_params)
        self.model = CalibratedClassifierCV(base)
        if mentor_vector is not None and instance is not None and target_pred is not None:
            print(f"[SVM] Fine-tune con mentor | target={target_pred}")
            X_f, y_f, w_f = self._build_mentor_dataset(X_train, y_train, instance, target_pred, mentor_vector)
            self.model.fit(X_f, y_f, sample_weight=w_f)
        else:
            self.model.fit(X_train, y_train)
        print(f"[SVM] C={self.hyperparams['C']:.4f} mentor={mentor_vector is not None}")