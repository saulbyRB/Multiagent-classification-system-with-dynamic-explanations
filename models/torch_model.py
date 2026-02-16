import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.base_model import BaseModel


class TorchModel(BaseModel):
    """
    TorchModel adaptativo para entornos multi-agente (MAS-aware).
    """

    def __init__(
        self,
        nn_model: nn.Module,
        device="cpu",
        lr=1e-3,
        min_lr=1e-5,
        max_lr=1e-2,
        epochs=10,
        batch_size=32,
        criterion=None
    ):
        super().__init__()
        self.model = nn_model.to(device)
        self.device = device

        # Optimización
        self.lr = lr
        self.base_lr = lr
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = criterion or nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr
        )

        # Estado
        self.is_trained = False
        self.classes_ = None

        # MAS tracking
        self.reward_history = []
        self.lr_history = []
        self.loss_history = []
        self.current_iteration = 0

    # --------------------------------------------------
    # Entrenamiento base
    # --------------------------------------------------

    def fit(self, X, y):
        self._train_loop(X, y, epochs=self.epochs)
        self.is_trained = True
        self.classes_ = np.unique(y)
        return self

    def partial_fit(self, X, y, epochs=1):
        """
        Entrenamiento incremental (clave para MAS).
        """
        self._train_loop(X, y, epochs=epochs)
        self.is_trained = True
        return self

    def _train_loop(self, X, y, epochs):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()

        for _ in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            self.loss_history.append(epoch_loss / len(loader))

    # --------------------------------------------------
    # Adaptación MAS
    # --------------------------------------------------

    def adjust_from_feedback(self, signals, X_train, y_train):
        """
        Ajuste adaptativo controlado por el MAS.
        """
        strategy = signals.get("strategy", "soft_adjust")
        reward = signals.get("reward", None)

        strategy = signals.get("strategy", "soft_adjust")

        if strategy == "keep":
            trainable_ratio = 0.0
        elif strategy == "soft_adjust":
            trainable_ratio = 0.3
        else:  # adjust
            trainable_ratio = 1.0

        self._freeze_layers(trainable_ratio)


        if reward is not None:
            self.reward_history.append(reward)

        # --- Ajuste de learning rate ---
        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]

            # Control suave (evita oscilaciones)
            factor = 1.0 + np.tanh(-delta * 5) * 0.2
            self.lr = np.clip(
                self.lr * factor,
                self.min_lr,
                self.max_lr
            )

            for g in self.optimizer.param_groups:
                g["lr"] = self.lr

        self.lr_history.append(self.lr)

        # --- Decidir epochs ---
        if strategy == "keep":
            epochs = 0
        elif strategy == "soft_adjust":
            epochs = 1
        else:  # adjust
            epochs = min(3, self.epochs)

        
        # --- Entrenamiento incremental ---
        if epochs > 0:
            self.partial_fit(X_train, y_train, epochs=epochs)

        self.current_iteration += 1

    # --------------------------------------------------
    # Inferencia
    # --------------------------------------------------

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            return torch.argmax(logits, dim=1).cpu().numpy()

    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            return F.softmax(logits, dim=1).cpu().numpy()

    def get_confidence(self, X):
        probs = self.predict_proba(X)
        return np.max(probs, axis=1)

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    def get_metadata(self):
        meta = super().get_metadata()
        meta.update({
            "backend": "torch",
            "model_class": self.model.__class__.__name__,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "current_lr": self.lr,
            "iterations": self.current_iteration
        })
        return meta

    def capabilities(self):
        return {
            "predict_proba": True,
            "confidence": True,
            "shap": True,
            "lime": True,
            "counterfactual": True,
            "adaptive": True
        }

    def _freeze_layers(self, trainable_ratio: float):
        """
        Congela capas dejando entrenable solo un porcentaje final
        de los parámetros del modelo.
        """
        params = list(self.model.parameters())
        n_total = len(params)

        if n_total == 0:
            return

        n_trainable = int(np.ceil(n_total * trainable_ratio))
        freeze_until = n_total - n_trainable

        for i, p in enumerate(params):
            p.requires_grad = i >= freeze_until
