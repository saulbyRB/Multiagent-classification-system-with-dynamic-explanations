# torch_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.base_model import BaseModel
import copy


class TorchModel(BaseModel):

    def __init__(self, nn_model: nn.Module, device="cpu",
                 lr=1e-3, min_lr=1e-5, max_lr=1e-2,
                 epochs=10, batch_size=32, criterion=None):
        super().__init__()
        self.nn_model  = nn_model.to(device)
        self.model     = self.nn_model   # alias para compatibilidad con explainers
        self.device    = device

        self.lr      = lr
        self.base_lr = lr
        self.min_lr  = min_lr
        self.max_lr  = max_lr

        self.epochs     = epochs
        self.batch_size = batch_size
        self.criterion  = criterion or nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.lr)

        self.is_trained = False
        self.classes_   = None

        self.reward_history   = []
        self.lr_history       = []
        self.loss_history     = []
        self.current_iteration = 0

    # ── Entrenamiento ──────────────────────────────────────────────────

    def fit(self, X, y):
        self._train_loop(X, y, epochs=self.epochs)
        self.is_trained = True
        self.classes_   = np.unique(y)
        return self

    def partial_fit(self, X, y, epochs=1):
        self._train_loop(X, y, epochs=epochs)
        self.is_trained = True
        return self

    def _train_loop(self, X, y, epochs):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long,    device=self.device)

        loader = DataLoader(TensorDataset(X_t, y_t),
                            batch_size=self.batch_size, shuffle=True)
        self.nn_model.train()

        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.nn_model(xb), yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            self.loss_history.append(epoch_loss / len(loader))

    # ── Adaptación MAS ─────────────────────────────────────────────────

    def adjust_from_feedback(self, signals, X_train, y_train):
        strategy     = signals.get("strategy", "soft_adjust")
        reward       = signals.get("reward", None)
        trend        = signals.get("trend", 0.0)       # del aggregator
        global_trend = signals.get("global_trend", 0.0)

        if reward is not None:
            self.reward_history.append(reward)

        # ── Ajuste de lr guiado por tendencia ───────────────────────────
        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]

            if trend < -0.05:
                # El aggregator dice que estamos empeorando:
                # invertir la dirección del lr
                factor = 1.0 - np.tanh(abs(delta) * 5) * 0.3
                print(f"[TorchModel] Tendencia negativa ({trend:+.3f}) → "
                      f"subiendo lr (factor={factor:.3f})")
            else:
                # Mejorando o estable: seguir bajando lr suavemente
                factor = 1.0 + np.tanh(-delta * 5) * 0.2

            self.lr = float(np.clip(self.lr * factor, self.min_lr, self.max_lr))
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr

        self.lr_history.append(self.lr)

        # ── Capas entrenables según estrategia ──────────────────────────
        if strategy == "keep":
            trainable_ratio = 0.0
        elif strategy == "soft_adjust":
            trainable_ratio = 0.3
        else:
            trainable_ratio = 1.0

        self._freeze_layers(trainable_ratio)

        # ── Épocas según estrategia ─────────────────────────────────────
        if strategy == "keep":
            epochs = 0
        elif strategy == "soft_adjust":
            epochs = 1
        else:
            epochs = min(3, self.epochs)

        if epochs > 0:
            self.partial_fit(X_train, y_train, epochs=epochs)

        self.current_iteration += 1

    # ── Inferencia ─────────────────────────────────────────────────────

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado")
        self.nn_model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return torch.argmax(self.nn_model(X_t), dim=1).cpu().numpy()

    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado")
        self.nn_model.eval()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return F.softmax(self.nn_model(X_t), dim=1).cpu().numpy()

    def get_confidence(self, X):
        return np.max(self.predict_proba(X), axis=1)

    # ── Utilidades ─────────────────────────────────────────────────────

    def _freeze_layers(self, trainable_ratio: float):
        params = list(self.nn_model.parameters())
        n = len(params)
        if n == 0:
            return
        freeze_until = n - int(np.ceil(n * trainable_ratio))
        for i, p in enumerate(params):
            p.requires_grad = i >= freeze_until

    def get_state(self):
        return {
            "model_state":     copy.deepcopy(self.nn_model.state_dict()),
            "optimizer_state": copy.deepcopy(self.optimizer.state_dict()),
            "lr":     self.lr,
            "epochs": self.epochs
        }

    def set_state(self, state):
        self.nn_model.load_state_dict(state["model_state"])
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.lr     = state["lr"]
        self.epochs = state["epochs"]

    def get_metadata(self):
        meta = super().get_metadata()
        meta.update({
            "backend":     "torch",
            "model_class": self.nn_model.__class__.__name__,
            "parameters":  sum(p.numel() for p in self.nn_model.parameters()),
            "current_lr":  self.lr,
            "iterations":  self.current_iteration
        })
        return meta

    def capabilities(self):
        return {
            "predict_proba": True,
            "confidence":    True,
            "shap":          True,
            "lime":          True,
            "counterfactual": True,
            "adaptive":      True
        }