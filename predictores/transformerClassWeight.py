# transformer.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ===== Dataset =====
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

# ===== Modelo =====
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
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 32
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.2
    device: Optional[str] = None  # "cuda" | "cpu" | None => auto
    seed: Optional[int] = 42

def _set_seed(seed: Optional[int]):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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

@torch.no_grad()
def _evaluate(model, loader, device) -> Dict[str, float]:
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.numpy().tolist())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}

def _class_weights(y: np.ndarray) -> np.ndarray:
    """
    Pesos inversos a la frecuencia de cada clase para CrossEntropyLoss.
    """
    y = np.asarray(y)
    n = y.size
    counts = np.bincount(y, minlength=2).astype(float)
    # Peso mayor para la clase minoritaria
    weights = n / (2.0 * np.clip(counts, 1.0, None))
    return weights

def fit_transformer(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    *, cfg: TrainConfig = TrainConfig()
) -> Tuple[TransformerClassifier, Dict[str, float]]:
    _set_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerClassifier(
        input_dim=X_train.shape[1], num_classes=2,
        d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.num_layers, dropout=cfg.dropout
    ).to(device)

    train_loader = DataLoader(_HeartDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(_HeartDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # === Pérdida con pesos por clase (reduce "todo 0" en datasets desbalanceados) ===
    w = torch.tensor(_class_weights(y_train), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    for _ in range(cfg.epochs):
        _ = _train_one_epoch(model, train_loader, opt, loss_fn, device)

    metrics = _evaluate(model, val_loader, device)
    return model, metrics

def transformer_evaluator(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    mask: np.ndarray,
    *,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dropout: float = 0.2,
    metric: str = "accuracy",
    maximize: bool = True,
) -> float:
    # Se dejó por compatibilidad (no usado por WOA en tu pipeline actual)
    m = np.asarray(mask).astype(int).ravel()
    if m.sum() == 0:
        m[np.random.randint(0, m.size)] = 1
    Xtr = X_train[:, m == 1]
    Xv  = X_val[:,   m == 1]
    _, metrics = fit_transformer(
        Xtr, y_train, Xv, y_val,
        cfg=TrainConfig(epochs=epochs, lr=lr, batch_size=batch_size,
                        d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
    )
    val = float(metrics.get(metric, metrics["accuracy"]))
    return val if maximize else (1.0 - val)

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
) -> Tuple[float, float, float, float, float]:
    """
    Devuelve (accuracy, precision, recall, f1, auc).
    """
    _, metrics = fit_transformer(
        X_train, y_train, X_test, y_test,
        cfg=TrainConfig(
            epochs=epochs, lr=lr, batch_size=batch_size,
            d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout
        )
    )
    return (
        float(metrics.get("accuracy", 0.0)),
        float(metrics.get("precision", 0.0)),
        float(metrics.get("recall", 0.0)),
        float(metrics.get("f1", 0.0)),
        float(metrics.get("auc", float("nan"))),
    )
