from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import find_movielens_dir, get_artifact_paths, get_torch_device, require_paths
from ranking_models import CTRMLP, DeepFM
from scripts.prepare_movielens import load_movies, load_ratings
from tools.recall_baselines import content_recall_by_history, itemcf_recall, load_catalog_assets
from tools.two_tower_recall import two_tower_recall_by_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation for recall and ranking baselines.")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--target-size", type=int, default=3)
    parser.add_argument("--min-history", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=500)
    parser.add_argument("--num-negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_eval_users(ratings, target_size: int, min_history: int, max_users: int | None):
    positive = ratings[ratings["rating"] >= 4].sort_values(["user_id", "timestamp"])
    users = []
    for user_id, group in positive.groupby("user_id"):
        movie_ids = group["movie_id"].astype(int).tolist()
        if len(movie_ids) < min_history + target_size:
            continue
        users.append(
            {
                "user_id": int(user_id),
                "history": movie_ids[:-target_size],
                "targets": movie_ids[-target_size:],
            }
        )
        if max_users and len(users) >= max_users:
            break
    return users


def recall_at_k(preds: list[int], targets: set[int]) -> float:
    return len(set(preds) & targets) / max(len(targets), 1)


def hitrate_at_k(preds: list[int], targets: set[int]) -> float:
    return 1.0 if set(preds) & targets else 0.0


def ndcg_at_k(preds: list[int], targets: set[int]) -> float:
    dcg = 0.0
    for rank, movie_id in enumerate(preds, start=1):
        if movie_id in targets:
            dcg += 1.0 / np.log2(rank + 1)

    ideal_hits = min(len(targets), len(preds))
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def summarize_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "Recall@K": round(float(np.mean([row["recall"] for row in rows])), 6),
        "HitRate@K": round(float(np.mean([row["hitrate"] for row in rows])), 6),
        "NDCG@K": round(float(np.mean([row["ndcg"] for row in rows])), 6),
    }


def evaluate_recall(users, k: int) -> dict[str, dict[str, float]]:
    model_rows = {
        "itemcf": [],
        "content_tfidf_faiss": [],
        "two_tower": [],
    }

    for user in users:
        targets = set(user["targets"])
        history = user["history"]

        itemcf_preds = [movie_id for movie_id, _ in itemcf_recall(history, k=k)]
        content_preds = [movie_id for movie_id, _ in content_recall_by_history(history, k=k)]
        two_tower_preds = [movie_id for movie_id, _ in two_tower_recall_by_history(history, k=k)]

        model_rows["itemcf"].append(
            {
                "recall": recall_at_k(itemcf_preds, targets),
                "hitrate": hitrate_at_k(itemcf_preds, targets),
                "ndcg": ndcg_at_k(itemcf_preds, targets),
            }
        )
        model_rows["content_tfidf_faiss"].append(
            {
                "recall": recall_at_k(content_preds, targets),
                "hitrate": hitrate_at_k(content_preds, targets),
                "ndcg": ndcg_at_k(content_preds, targets),
            }
        )
        model_rows["two_tower"].append(
            {
                "recall": recall_at_k(two_tower_preds, targets),
                "hitrate": hitrate_at_k(two_tower_preds, targets),
                "ndcg": ndcg_at_k(two_tower_preds, targets),
            }
        )

    return {name: summarize_metric_rows(rows) for name, rows in model_rows.items()}


def load_rank_models():
    paths = get_artifact_paths()
    require_paths([paths.ctr_model_pt], "Run `python -m scripts.prepare_movielens` first.")
    device = get_torch_device()

    mlp_ckpt = torch.load(paths.ctr_model_pt, map_location="cpu")
    mlp = CTRMLP(int(mlp_ckpt["input_dim"]))
    mlp.load_state_dict(mlp_ckpt["state_dict"])
    mlp.to(device).eval()

    models = {"mlp": mlp}
    if paths.deepfm_model_pt.exists():
        deepfm_ckpt = torch.load(paths.deepfm_model_pt, map_location="cpu")
        deepfm = DeepFM(
            int(deepfm_ckpt["input_dim"]),
            embedding_dim=int(deepfm_ckpt.get("embedding_dim", 16)),
        )
        deepfm.load_state_dict(deepfm_ckpt["state_dict"])
        deepfm.to(device).eval()
        models["deepfm"] = deepfm

    return models, device


def evaluate_ranking(users, ratings, k: int, num_negatives: int, seed: int) -> dict[str, dict[str, float | None]]:
    assets = load_catalog_assets()
    catalog = assets["catalog"]
    genre_classes = sorted({genre for genres in catalog["genres_list"] for genre in genres})
    movie_genre_map = {
        int(row.movie_id): row.genres_list
        for row in catalog[["movie_id", "genres_list"]].itertuples(index=False)
    }
    all_movie_ids = set(catalog["movie_id"].astype(int).tolist())
    rated_map = (
        ratings.groupby("user_id")["movie_id"]
        .apply(lambda s: set(int(item) for item in s.tolist()))
        .to_dict()
    )

    models, device = load_rank_models()
    rng = random.Random(seed)
    results: dict[str, list[dict[str, float]]] = {name: [] for name in models}

    for user in users:
        user_id = user["user_id"]
        history = user["history"]
        targets = user["targets"]
        target_set = set(targets)

        preferred_genres = []
        for movie_id in history:
            preferred_genres.extend(movie_genre_map.get(int(movie_id), []))
        user_vec = [1 if genre in set(preferred_genres) else 0 for genre in genre_classes]

        unrated = list(all_movie_ids - rated_map.get(user_id, set()))
        if len(unrated) < num_negatives:
            negatives = unrated
        else:
            negatives = rng.sample(unrated, num_negatives)
        candidate_ids = list(dict.fromkeys(targets + negatives))
        labels = np.array([1 if movie_id in target_set else 0 for movie_id in candidate_ids], dtype=np.int32)

        X = np.array(
            [
                user_vec
                + [1 if genre in set(movie_genre_map.get(int(movie_id), [])) else 0 for genre in genre_classes]
                for movie_id in candidate_ids
            ],
            dtype=np.float32,
        )
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)

        for name, model in models.items():
            with torch.no_grad():
                scores = model(X_tensor).squeeze().detach().cpu().numpy()
            if np.isscalar(scores):
                scores = np.array([float(scores)])

            ranked_ids = [candidate_ids[idx] for idx in np.argsort(scores)[::-1][:k]]
            auc = roc_auc_score(labels, scores) if len(set(labels.tolist())) > 1 else None
            results[name].append(
                {
                    "recall": recall_at_k(ranked_ids, target_set),
                    "hitrate": hitrate_at_k(ranked_ids, target_set),
                    "ndcg": ndcg_at_k(ranked_ids, target_set),
                    "auc": float(auc) if auc is not None else None,
                }
            )

    summary = {}
    for name, rows in results.items():
        auc_values = [row["auc"] for row in rows if row["auc"] is not None]
        summary[name] = {
            "Recall@K": round(float(np.mean([row["recall"] for row in rows])), 6),
            "HitRate@K": round(float(np.mean([row["hitrate"] for row in rows])), 6),
            "NDCG@K": round(float(np.mean([row["ndcg"] for row in rows])), 6),
            "AUC": round(float(np.mean(auc_values)), 6) if auc_values else None,
        }
    return summary


def main() -> None:
    args = parse_args()
    movielens_dir = find_movielens_dir()
    ratings_path = movielens_dir / "ratings.dat"
    movies_path = movielens_dir / "movies.dat"

    ratings = load_ratings(ratings_path)
    _ = load_movies(movies_path)
    users = build_eval_users(
        ratings=ratings,
        target_size=args.target_size,
        min_history=args.min_history,
        max_users=args.max_users,
    )

    output = {
        "eval_users": len(users),
        "k": args.k,
        "target_size": args.target_size,
        "recall": evaluate_recall(users, k=args.k),
        "ranking": evaluate_ranking(
            users=users,
            ratings=ratings,
            k=args.k,
            num_negatives=args.num_negatives,
            seed=args.seed,
        ),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
