from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import MissingArtifactError, get_artifact_paths, get_torch_device, require_paths
from ranking_models import TwoTowerRecall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Two-Tower recall model.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    return parser.parse_args()


def _safe_auc(y_true, y_pred) -> float | None:
    unique = set(float(item) for item in y_true)
    if len(unique) < 2:
        return None
    return roc_auc_score(y_true, y_pred)


def train_two_tower_model(
    samples_path: Path | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
) -> dict[str, float | int | None | str]:
    paths = get_artifact_paths()
    samples_path = samples_path or paths.ctr_samples_csv

    if not samples_path.exists():
        raise MissingArtifactError(
            f"CTR samples not found at '{samples_path.relative_to(PROJECT_ROOT)}'. "
            "Run `python -m scripts.prepare_movielens` first."
        )

    train_df = pd.read_csv(samples_path)
    user_cols = [col for col in train_df.columns if col.startswith("u_")]
    item_cols = [col for col in train_df.columns if col.startswith("v_")]
    if not user_cols or not item_cols:
        raise MissingArtifactError(
            "CTR samples exist but contain no user/item feature columns. "
            "Rebuild artifacts with `python -m scripts.prepare_movielens`."
        )

    user_x = train_df[user_cols].values.astype(np.float32)
    item_x = train_df[item_cols].values.astype(np.float32)
    labels = train_df["label"].values.astype(np.float32)

    ux_train, ux_val, ix_train, ix_val, y_train, y_val = train_test_split(
        user_x,
        item_x,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels if len(set(labels.tolist())) > 1 else None,
    )

    device = get_torch_device()
    train_dataset = TensorDataset(
        torch.tensor(ux_train, dtype=torch.float32, device=device),
        torch.tensor(ix_train, dtype=torch.float32, device=device),
        torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    ux_val_tensor = torch.tensor(ux_val, dtype=torch.float32, device=device)
    ix_val_tensor = torch.tensor(ix_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    model = TwoTowerRecall(
        input_dim=len(user_cols),
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        for ux_batch, ix_batch, y_batch in train_loader:
            logits = model(ux_batch, ix_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        val_logits = model(ux_val_tensor, ix_val_tensor)
        val_preds = torch.sigmoid(val_logits).squeeze().detach().cpu().numpy()
        val_labels = y_val_tensor.squeeze().detach().cpu().numpy()

    auc = _safe_auc(val_labels, val_preds)
    metrics = {
        "rows": int(len(train_df)),
        "feature_dim": int(len(user_cols)),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "embedding_dim": int(embedding_dim),
        "loss": round(last_loss, 6),
        "auc": round(float(auc), 6) if auc is not None else None,
        "device": str(device),
    }

    paths.two_tower_model_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(len(user_cols)),
            "hidden_dim": int(hidden_dim),
            "embedding_dim": int(embedding_dim),
            "user_feature_cols": user_cols,
            "item_feature_cols": item_cols,
        },
        paths.two_tower_model_pt,
    )
    paths.two_tower_model_pt.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def export_two_tower_item_embeddings(
    catalog_path: Path | None = None,
    model_path: Path | None = None,
) -> dict[str, int | str]:
    paths = get_artifact_paths()
    catalog_path = catalog_path or paths.movie_catalog_csv
    model_path = model_path or paths.two_tower_model_pt
    require_paths(
        [catalog_path, model_path],
        "Run `python -m scripts.prepare_movielens` and `python -m scripts.train_two_tower` first.",
    )

    catalog = pd.read_csv(catalog_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    item_feature_cols = checkpoint["item_feature_cols"]

    genre_classes = [col.replace("v_", "") for col in item_feature_cols]
    item_vectors = []
    for genres in catalog["genres"].fillna("").tolist():
        genre_set = set([genre for genre in str(genres).split("|") if genre and genre != "(no genres listed)"])
        item_vectors.append([1.0 if genre in genre_set else 0.0 for genre in genre_classes])

    device = get_torch_device()
    model = TwoTowerRecall(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    item_tensor = torch.tensor(np.array(item_vectors, dtype=np.float32), dtype=torch.float32, device=device)
    with torch.no_grad():
        item_embeddings = model.encode_item(item_tensor)
        item_embeddings = torch.nn.functional.normalize(item_embeddings, dim=1)
        item_embeddings = item_embeddings.detach().cpu().numpy().astype(np.float32)

    np.save(paths.two_tower_movie_embeddings_npy, item_embeddings)
    paths.two_tower_movie_id_map_txt.write_text(
        "\n".join(str(int(movie_id)) for movie_id in catalog["movie_id"].astype(int).tolist()),
        encoding="utf-8",
    )
    return {
        "rows": int(len(catalog)),
        "embedding_dim": int(item_embeddings.shape[1]),
        "embeddings_path": str(paths.two_tower_movie_embeddings_npy),
    }


if __name__ == "__main__":
    args = parse_args()
    paths = get_artifact_paths()
    metrics = train_two_tower_model(
        samples_path=paths.ctr_samples_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
    )
    exported = export_two_tower_item_embeddings(
        catalog_path=paths.movie_catalog_csv,
        model_path=paths.two_tower_model_pt,
    )
    print(json.dumps({"train_metrics": metrics, "item_export": exported}, indent=2))
