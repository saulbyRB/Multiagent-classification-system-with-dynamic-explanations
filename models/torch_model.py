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
        self.model     = self.nn_model
        self.device    = device

        self.lr      = lr
        self.base_lr = lr
        self.min_lr  = min_lr
        self.max_lr  = max_lr

        self.epochs     = epochs
        self.batch_size = batch_size
        self.criterion  = criterion or nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.nn_model.parameters(), lr=self.lr)

        self.is_trained        = False
        self.classes_          = None
        self.reward_history    = []
        self.lr_history        = []
        self.loss_history      = []
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

    def _train_loop(self, X, y, epochs, sample_weights=None):
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long,    device=self.device)

        if sample_weights is not None:
            w_t = torch.tensor(sample_weights, dtype=torch.float32,
                               device=self.device)
            dataset = TensorDataset(X_t, y_t, w_t)
        else:
            dataset = TensorDataset(X_t, y_t)

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.nn_model.train()

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                self.optimizer.zero_grad()
                if sample_weights is not None:
                    xb, yb, wb = batch
                    logits = self.nn_model(xb)
                    # Loss ponderada muestra a muestra
                    loss = (F.cross_entropy(logits, yb, reduction="none") * wb).mean()
                else:
                    xb, yb = batch
                    loss = self.criterion(self.nn_model(xb), yb)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            self.loss_history.append(epoch_loss / len(loader))

    # ── Adaptación MAS ─────────────────────────────────────────────────

    def adjust_from_feedback(self, signals, X_train, y_train):
        strategy      = signals.get("strategy", "soft_adjust")
        reward        = signals.get("reward", None)
        trend         = signals.get("trend", 0.0)
        mentor_vector = signals.get("mentor_vector", None)   # ← nuevo
        instance      = signals.get("instance", None)        # ← nuevo
        target_pred   = signals.get("target_pred", None)     # ← nuevo

        if reward is not None:
            self.reward_history.append(reward)

        # ── Ajuste de lr ────────────────────────────────────────────────
        if len(self.reward_history) >= 2:
            delta = self.reward_history[-1] - self.reward_history[-2]
            if trend < -0.05:
                factor = 1.0 - np.tanh(abs(delta) * 5) * 0.3
                print(f"[TorchModel] Tendencia negativa ({trend:+.3f}) → "
                      f"subiendo lr (factor={factor:.3f})")
            else:
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

        # ── Fine-tune guiado por mentor ─────────────────────────────────
        # Si hay mentor_vector, instance y target_pred disponibles,
        # construimos un mini-batch aumentado donde la instancia se replica
        # varias veces con la clase correcta, ponderando las features
        # importantes según el mentor.
        if (mentor_vector is not None
                and instance is not None
                and target_pred is not None
                and strategy != "keep"):

            print(f"[TorchModel] Fine-tune con guía de mentor | "
                  f"target={target_pred}")

            X_aug, y_aug, w_aug = self._build_mentor_batch(
                X_train, y_train, instance, target_pred, mentor_vector
            )

            # Descongelar todas las capas para el fine-tune dirigido
            self._freeze_layers(1.0)

            # Fine-tune con el batch aumentado (pocas épocas, lr reducido)
            lr_mentor = self.lr * 0.5
            for g in self.optimizer.param_groups:
                g["lr"] = lr_mentor

            self._train_loop(X_aug, y_aug, epochs=2, sample_weights=w_aug)

            # Restaurar lr original
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr

            # Re-aplicar freeze para el entrenamiento normal posterior
            self._freeze_layers(trainable_ratio)

        # ── Entrenamiento normal ────────────────────────────────────────
        if epochs > 0:
            self.partial_fit(X_train, y_train, epochs=epochs)

        self.current_iteration += 1

    def _build_mentor_batch(self, X_train, y_train, instance,
                             target_pred, mentor_vector, n_replicas=10):
        """
        Construye un batch aumentado para el fine-tune guiado:

        1. Toma el train set original.
        2. Replica la instancia N veces con el target correcto,
           con peso alto (proporcional a importancia de features del mentor).
        3. Los ejemplos del train set tienen peso 1.0.
        4. Las réplicas tienen peso = base_weight * feature_importance_factor.

        El factor de importancia refuerza la señal en las features
        que el mentor considera relevantes, añadiendo pequeñas perturbaciones
        gaussianas para que el modelo no sobreajuste a un único punto.
        """
        x_inst = np.asarray(instance, dtype=float).reshape(1, -1)
        n_feat = x_inst.shape[1]

        # Importancia de features según el mentor (normalizada a [0,1])
        mv = np.abs(mentor_vector[:n_feat]) if len(mentor_vector) >= n_feat \
             else np.abs(mentor_vector)
        mv_norm = mv / (mv.max() + 1e-8)

        # Generar réplicas con ruido gaussiano ponderado por importancia
        noise_scale = 0.05  # pequeño para no alejarse del punto original
        replicas_x = []
        replicas_w = []

        for _ in range(n_replicas):
            noise = np.random.randn(n_feat) * noise_scale * mv_norm
            x_rep = x_inst.flatten() + noise
            # Peso de la réplica: más alto para features importantes
            importance_factor = 1.0 + float(mv_norm.mean()) * 2.0
            replicas_x.append(x_rep)
            replicas_w.append(importance_factor)

        replicas_x = np.array(replicas_x, dtype=float)
        replicas_y = np.full(n_replicas, target_pred, dtype=int)
        replicas_w = np.array(replicas_w, dtype=float)

        # Combinar con train set
        X_aug = np.vstack([X_train, replicas_x])
        y_aug = np.concatenate([y_train, replicas_y])
        # Pesos: 1.0 para train original, importance_factor para réplicas
        w_aug = np.concatenate([
            np.ones(len(X_train), dtype=float),
            replicas_w
        ])

        return X_aug, y_aug, w_aug

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
            "predict_proba":  True,
            "confidence":     True,
            "shap":           True,
            "lime":           True,
            "counterfactual": True,
            "adaptive":       True,
            "mentor_guided":  True,
        }