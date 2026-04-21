from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import torch

from project_config import get_artifact_paths, get_torch_device, require_paths
from ranking_models import TwoTowerRecall


@lru_cache(maxsize=1)
def load_two_tower_assets() -> dict[str, Any]:
    paths = get_artifact_paths()
    require_paths(
        [
            paths.movie_catalog_csv,
            paths.two_tower_model_pt,
            paths.two_tower_movie_embeddings_npy,
            paths.two_tower_movie_id_map_txt,
        ],
        build_hint="Run `python -m scripts.prepare_movielens` and `python -m scripts.train_two_tower` first.",
    )

    catalog = pd.read_csv(paths.movie_catalog_csv)
    checkpoint = torch.load(paths.two_tower_model_pt, map_location="cpu")
    device = get_torch_device()

    model = TwoTowerRecall(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    item_embeddings = np.load(paths.two_tower_movie_embeddings_npy).astype(np.float32)
    movie_id_map = [
        int(line)
        for line in paths.two_tower_movie_id_map_txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    user_feature_cols = checkpoint["user_feature_cols"]

    return {
        "catalog": catalog,
        "model": model,
        "item_embeddings": item_embeddings,
        "movie_id_map": movie_id_map,
        "user_feature_cols": user_feature_cols,
        "device": device,
    }


def two_tower_recall_by_history(
    history_movie_ids: list[int],
    k: int = 50,
    exclude_movie_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    assets = load_two_tower_assets()
    catalog = assets["catalog"]
    model = assets["model"]
    item_embeddings = assets["item_embeddings"]
    movie_id_map = assets["movie_id_map"]
    user_feature_cols = assets["user_feature_cols"]
    device = assets["device"]

    history_df = catalog[catalog["movie_id"].isin(history_movie_ids)]
    genre_counts: dict[str, int] = {}
    for genres in history_df["genres"].fillna("").tolist():
        for genre in str(genres).split("|"):
            if not genre or genre == "(no genres listed)":
                continue
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    feature_values = []
    for col in user_feature_cols:
        genre = col.replace("u_", "")
        feature_values.append(1.0 if genre_counts.get(genre, 0) > 0 else 0.0)

    user_tensor = torch.tensor([feature_values], dtype=torch.float32, device=device)
    with torch.no_grad():
        user_embedding = model.encode_user(user_tensor)
        user_embedding = torch.nn.functional.normalize(user_embedding, dim=1)
        user_embedding = user_embedding.detach().cpu().numpy()[0]

    scores = item_embeddings @ user_embedding
    exclude = set(exclude_movie_ids or set())
    exclude.update(history_movie_ids)

    ranked_indices = np.argsort(scores)[::-1]
    results: list[tuple[int, float]] = []
    for idx in ranked_indices:
        movie_id = movie_id_map[idx]
        if movie_id in exclude:
            continue
        results.append((movie_id, float(scores[idx])))
        if len(results) >= k:
            break
    return results
