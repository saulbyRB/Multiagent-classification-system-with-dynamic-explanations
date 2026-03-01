# xgboost_model.py
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from models.sklearn_model import SklearnModel


class XGBoostModel(SklearnModel):

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 max_depth=6, subsample=0.8, colsample_bytree=0.8, **kwargs):
        super().__init__(
            XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                use_label_encoder=False,
                eval_metric="logloss",
                **kwargs
            )
        )
        self.hyperparams = {
            "n_estimators":   n_estimators,
            "learning_rate":  learning_rate,
            "max_depth":      max_depth,
            "subsample":      subsample,
            "colsample_bytree": colsample_bytree,
        }
        self.reward_history = []
        self._lr_direction = 1  # +1 bajando lr, -1 subiéndola

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
            print("[AdvancedXGB] keep → sin ajuste")
            return

        base_factor = 1.03
        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            adaptive_factor = base_factor * (1.0 + np.tanh(-delta * 4))
            adaptive_factor = np.clip(adaptive_factor, 1.01, 1.15)
        else:
            adaptive_factor = base_factor

        factor = adaptive_factor if strategy == "adjust" else (1 + (adaptive_factor - 1) / 2)

        # Si empeora según el aggregator, invertir dirección del lr
        if trend < -0.05:
            self._lr_direction *= -1
            print(f"[AdvancedXGB] Tendencia negativa ({trend:+.3f}) → "
                  f"invirtiendo dirección lr")

        self.hyperparams["n_estimators"] = min(
            int(self.hyperparams["n_estimators"] * factor), 400
        )
        self.hyperparams["max_depth"] = min(
            max(int(self.hyperparams["max_depth"] * factor), 3), 12
        )

        if self._lr_direction > 0:
            new_lr = self.hyperparams["learning_rate"] / factor
        else:
            new_lr = self.hyperparams["learning_rate"] * factor

        self.hyperparams["learning_rate"] = float(np.clip(new_lr, 0.01, 0.9))

        self.model.set_params(**self.hyperparams)
        self.model.fit(X_train, y_train)

        print(f"[AdvancedXGB] Ajuste (factor={adaptive_factor:.3f}) | "
              f"n_estimators={self.hyperparams['n_estimators']} | "
              f"lr={self.hyperparams['learning_rate']:.4f} | "
              f"depth={self.hyperparams['max_depth']} | "
              f"dir={'↓' if self._lr_direction > 0 else '↑'}")