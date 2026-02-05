import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from models.base_model import BaseModel

class TorchModel(BaseModel):
    """
    Wrapper genérico para cualquier red neuronal definida por el usuario.
    """

    def __init__(self, name, nn_model: nn.Module, device="cpu", lr=1e-3, epochs=10, batch_size=32, criterion=None):
        super().__init__(name=name)
        self.model = nn_model.to(device)
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_trained = False
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.classes_ = None

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for xb, yb in loader:
                self.optimizer.zero_grad()
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()

        self.is_trained = True
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds

    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("El modelo no ha sido entrenado")
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def get_confidence(self, X):
        probs = self.predict_proba(X)
        return np.max(probs, axis=1) if probs is not None else None

    def get_metadata(self):
        meta = super().get_metadata()
        meta.update({
            "backend": "torch",
            "model_class": self.model.__class__.__name__,
            "parameters": sum(p.numel() for p in self.model.parameters())
        })
        return meta

    def capabilities(self):
        return {
            "predict_proba": True,
            "confidence": True,
            "shap": True,     # con wrapper adecuado
            "lime": True,
            "counterfactual": True
        }
