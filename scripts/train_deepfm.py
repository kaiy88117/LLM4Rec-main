from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import MissingArtifactError, get_artifact_paths, get_torch_device
from ranking_models import DeepFM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the DeepFM ranking model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--embedding-dim", type=int, default=16)
    return parser.parse_args()


def _safe_auc(y_true, y_pred) -> float | None:
    unique = set(float(item) for item in y_true)
    if len(unique) < 2:
        return None
    return roc_auc_score(y_true, y_pred)


def train_deepfm_model(
    samples_path: Path | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    embedding_dim: int = 16,
) -> dict[str, float | int | None | str]:
    paths = get_artifact_paths()
    samples_path = samples_path or paths.ctr_samples_csv

    if not samples_path.exists():
        raise MissingArtifactError(
            f"CTR samples not found at '{samples_path.relative_to(PROJECT_ROOT)}'. "
            "Run `python -m scripts.prepare_movielens` first."
        )

    train_df = pd.read_csv(samples_path)
    feature_cols = [col for col in train_df.columns if col.startswith("u_") or col.startswith("v_")]
    if not feature_cols:
        raise MissingArtifactError(
            "CTR samples exist but contain no feature columns. "
            "Rebuild artifacts with `python -m scripts.prepare_movielens`."
        )

    X = train_df[feature_cols].values
    y = train_df["label"].values
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(set(y)) > 1 else None,
    )

    device = get_torch_device()
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
    )

    model = DeepFM(X.shape[1], embedding_dim=embedding_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor).squeeze().detach().cpu().numpy()
        val_labels = y_val_tensor.squeeze().detach().cpu().numpy()

    auc = _safe_auc(val_labels, val_preds)
    metrics = {
        "rows": int(len(train_df)),
        "feature_dim": int(X.shape[1]),
        "epochs": int(epochs),
        "embedding_dim": int(embedding_dim),
        "loss": round(last_loss, 6),
        "auc": round(float(auc), 6) if auc is not None else None,
        "log_loss": round(float(log_loss(val_labels, val_preds, labels=[0, 1])), 6),
        "device": str(device),
    }

    paths.deepfm_model_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(X.shape[1]),
            "embedding_dim": int(embedding_dim),
        },
        paths.deepfm_model_pt,
    )
    paths.deepfm_model_pt.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


if __name__ == "__main__":
    args = parse_args()
    result = train_deepfm_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
    )
    print(json.dumps(result, indent=2))
