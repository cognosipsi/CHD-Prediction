# transformer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils.evaluacion import compute_classification_metrics
# ...existing code...

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
    def __init__(self, input_dim: int, num_classes: int,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
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


def _train_one_epoch(model, loader, opt, loss_fn, device) -> float:
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        total += float(loss.detach().cpu().item())
    return total

def fit_transformer(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *, cfg: TrainConfig = TrainConfig()
) -> Tuple[TransformerClassifier, Dict[str, float]]:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(
        input_dim=X_train.shape[1], num_classes=2,
        d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.num_layers, dropout=cfg.dropout
    ).to(device)

    train_loader = DataLoader(_HeartDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(_HeartDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(cfg.epochs):
        _ = _train_one_epoch(model, train_loader, opt, loss_fn, device)
        # si quieres, imprime: print(f"[Transformer] Epoch {epoch+1}/{cfg.epochs} - loss={loss:.4f}")

    # --- Recopilar predicciones y probabilidades para métricas completas ---
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for xb, yb in val_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().detach().numpy()
        preds = torch.argmax(logits, dim=1).cpu().detach().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.numpy().tolist())

    metrics = compute_classification_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    return model, metrics

# === NUEVO: wrapper esperado por el pipeline ===
def transformer_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> Dict[str, float]:
    """
    Envoltorio que entrena con fit_transformer y devuelve
    (accuracy, precision, recall, f1, auc) en ese orden, como espera el pipeline.
    """
    _, metrics = fit_transformer(
        X_train, y_train, X_test, y_test,
        cfg=TrainConfig(
            epochs=epochs, lr=lr, batch_size=batch_size,
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
        )
    )
    return metrics