# transformer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

#sklearn e imblearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import accuracy_score

from utils.evaluacion import compute_classification_metrics

# ===== Dataset y utilidades =====

class _HeartDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D, shape={X.shape}")
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.astype(int), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

# ===== Modelo Transformer tabular sencillo =====

class TransformerClassifier(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            num_classes: int,
            d_model: int = 64, 
            nhead: int = 4, 
            num_layers: int = 2, 
            dropout: float = 0.2
        ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True, 
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)     # (B, d)
        x = x.unsqueeze(1)        # (B, 1, d)
        x = self.encoder(x)       # (B, 1, d)
        x = x.squeeze(1)          # (B, d)
        return self.classifier(x)


@dataclass
class TrainConfig:
    epochs: int = 20           # como tu script monolítico
    lr: float = 1e-3
    batch_size: int = 32
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.2
    device: Optional[str] = None  # "cuda" | "cpu" | None => auto


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    ) -> float:
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total_loss += float(loss.detach().cpu().item())
    return total_loss

def fit_transformer(
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_val: np.ndarray, 
    y_val: np.ndarray,
    *, 
    cfg: TrainConfig = TrainConfig()
) -> Tuple[TransformerClassifier, Dict[str, float]]:
    if X_train.ndim == 1:
        X_train = X_train.reshape(1, -1)
    if X_val.ndim == 1:
        X_val = X_val.reshape(1, -1)

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError(
            f"Dimensión de características inconsistente entre train ({X_train.shape[1]}) "
            f"y val ({X_val.shape[1]})."
        )
    
    device_str  = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    n_features = X_train.shape[1]
    n_classes = int(len(np.unique(y_train)))
    if n_classes < 2:
        raise ValueError("Se requiere al menos 2 clases para entrenar el Transformer.")

    model = TransformerClassifier(
        input_dim=n_features,
        num_classes=n_classes,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    train_loader = DataLoader(
        _HeartDataset(X_train, y_train), 
        batch_size=cfg.batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        _HeartDataset(X_val, y_val), 
        batch_size=cfg.batch_size, 
        shuffle=False
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        train_loss = _train_one_epoch(model, train_loader, opt, loss_fn, device_str )

    # --- Recopilar predicciones y probabilidades para métricas completas ---
    model.eval()
    all_labels: list[int] = []
    all_preds: list[int] = []
    all_probs: Optional[list[float]] = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            prob_logits = torch.softmax(logits, dim=1)

            preds = torch.argmax(prob_logits, dim=1)

            all_labels.extend(yb.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

            # Si es binario, guardamos prob. de la clase 1
            if prob_logits.shape[1] == 2:
                probs = prob_logits[:, 1].cpu().numpy()
                all_probs.extend(probs.tolist())
            else:
                all_probs = None  # multi-clase: dejamos que sea None

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    if all_probs is not None:
        y_proba: Optional[np.ndarray] = np.array(all_probs)
    else:
        y_proba = None

    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    return model, metrics

class SklearnTransformerClassifier(BaseEstimator, ClassifierMixin):
    # Wrapper sklearn alrededor del Transformer en PyTorch.
    #
    # - Implementa fit/predict/predict_proba.
    # - Es compatible con Pipeline, GridSearchCV, etc.
    # - Internamente puede usar un pequeño conjunto de validación para
    #   monitorizar métricas, pero eso NO afecta la interfaz sklearn.

    def __init__(
        self,
        epochs: int = 20,
        lr: float = 1e-3,
        batch_size: int = 32,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: Optional[str] = None,
        val_size: float = 0.2,
        random_state: Optional[int] = None,
    ):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.val_size = val_size
        self.random_state = random_state

    def fit(self, X, y):
        # Entrena el modelo Transformer.
        #
        # X: array-like de shape (n_muestras, n_features)
        # y: array-like con etiquetas (se asume problema binario 0/1)
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D, shape={X.shape}")

        # Guardamos clases originales y las mapeamos a índices 0..K-1
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(
                f"Se requieren al menos 2 clases; se encontraron {n_classes}."
            )

        # División train/val interna si val_size > 0
        if self.val_size and self.val_size > 0.0:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_idx,
                test_size=self.val_size,
                stratify=y_idx,
                random_state=self.random_state,
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y_idx, y_idx

        # Configuración y entrenamiento usando la función de bajo nivel
        cfg = TrainConfig(
            epochs=self.epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            device=self.device,
        )

        model, val_metrics = fit_transformer(
            X_train, y_train, X_val, y_val, cfg=cfg
        )

        self.model_ = model
        self.validation_metrics_ = val_metrics
        self.n_features_in_ = X.shape[1]
        self.device_ = next(self.model_.parameters()).device
        self.is_fitted_ = True
        return self

    def _predict_logits(self, X) -> np.ndarray:
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError(f"X debe ser 2D, shape={X.shape}")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"n_features_in_ inconsistente: se esperaba {self.n_features_in_}, "
                f"pero se recibió {X.shape[1]}."
            )

        self.model_.eval()
        with torch.no_grad():
            xb = torch.tensor(X, dtype=torch.float32).to(self.device_)
            logits = self.model_(xb)
            return logits.cpu().numpy()

    def predict_proba(self, X) -> np.ndarray:
        """
        Devuelve las probabilidades por clase en el orden self.classes_.
        """
        logits = self._predict_logits(X)
        probs_t = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=1)
        probs = probs_t.numpy()
        return probs

    def predict(self, X) -> np.ndarray:
        """
        Predicción de etiquetas (en el espacio original de y).
        """
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]

    def score(self, X, y) -> float:
        """
        Accuracy, para ser coherente con la API de sklearn.
        """
        y = np.asarray(y)
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)