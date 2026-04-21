from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
import torch

from project_config import get_artifact_paths, get_torch_device, require_paths
from ranking_models import CTRMLP
from tools.recall_baselines import content_recall_by_query, load_catalog_assets
from tools.rec_explainer import generate_explanation


GENRE_ALIASES = {
    "action": "Action",
    "\u52a8\u4f5c": "Action",
    "adventure": "Adventure",
    "\u5192\u9669": "Adventure",
    "animation": "Animation",
    "\u52a8\u753b": "Animation",
    "children": "Children's",
    "childrens": "Children's",
    "\u513f\u7ae5": "Children's",
    "comedy": "Comedy",
    "\u559c\u5267": "Comedy",
    "crime": "Crime",
    "\u72af\u7f6a": "Crime",
    "documentary": "Documentary",
    "\u7eaa\u5f55\u7247": "Documentary",
    "drama": "Drama",
    "\u5267\u60c5": "Drama",
    "fantasy": "Fantasy",
    "\u5947\u5e7b": "Fantasy",
    "film-noir": "Film-Noir",
    "\u9ed1\u8272\u7535\u5f71": "Film-Noir",
    "horror": "Horror",
    "\u6050\u6016": "Horror",
    "musical": "Musical",
    "\u97f3\u4e50": "Musical",
    "mystery": "Mystery",
    "\u60ac\u7591": "Mystery",
    "romance": "Romance",
    "\u7231\u60c5": "Romance",
    "sci-fi": "Sci-Fi",
    "scifi": "Sci-Fi",
    "\u79d1\u5e7b": "Sci-Fi",
    "thriller": "Thriller",
    "\u60ca\u609a": "Thriller",
    "war": "War",
    "\u6218\u4e89": "War",
    "western": "Western",
    "\u897f\u90e8": "Western",
}


def _split_genres(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [genre for genre in str(value).split("|") if genre and genre != "(no genres listed)"]


def _normalize_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+|[\u4e00-\u9fff]+", text.lower())


def _extract_preference_terms(user_input: str | list[str]) -> tuple[str, list[str]]:
    if isinstance(user_input, list):
        raw_query = " ".join(str(item) for item in user_input if str(item).strip())
    else:
        raw_query = str(user_input).strip()

    tokens = _normalize_tokens(raw_query)
    preferences: list[str] = []
    for token in tokens:
        mapped = GENRE_ALIASES.get(token)
        if mapped and mapped not in preferences:
            preferences.append(mapped)

    raw_query_lower = raw_query.lower()
    for alias, mapped in GENRE_ALIASES.items():
        if alias in raw_query_lower and mapped not in preferences:
            preferences.append(mapped)

    return raw_query, preferences


def _binarize_terms(term_list: list[str], classes: list[str]) -> list[int]:
    term_set = set(term_list)
    return [1 if item in term_set else 0 for item in classes]


def _load_runtime_assets() -> dict[str, Any]:
    paths = get_artifact_paths()
    require_paths(
        [
            paths.ctr_model_pt,
        ],
        build_hint="Run `python -m scripts.prepare_movielens` from the project root.",
    )

    recall_assets = load_catalog_assets()
    catalog = recall_assets["catalog"]
    stats = recall_assets["stats"]
    checkpoint = torch.load(paths.ctr_model_pt, map_location="cpu")
    genre_classes = sorted({genre for genres in catalog["genres_list"] for genre in genres})
    input_dim = int(checkpoint.get("input_dim", len(genre_classes) * 2))
    model = CTRMLP(input_dim)
    model.load_state_dict(checkpoint["state_dict"])
    device = get_torch_device()
    model = model.to(device)
    model.eval()

    return {
        "catalog": catalog,
        "stats": stats,
        "genre_classes": genre_classes,
        "model": model,
        "device": device,
    }


def _fallback_popular_movies(k: int) -> list[dict]:
    assets = _load_runtime_assets()
    catalog = assets["catalog"]
    stats = assets["stats"].sort_values(
        by=["popularity_score", "rating_count"],
        ascending=False,
    ).head(k)
    candidate_df = stats.merge(
        catalog[["movie_id", "title", "year", "genres", "genres_list"]],
        on=["movie_id", "title", "genres"],
        how="left",
    )
    return [
        {
            "movie_id": int(row.movie_id),
            "title": row.title,
            "year": int(row.year) if not pd.isna(row.year) else None,
            "score": round(float(row.popularity_score), 4),
            "genres": row.genres_list,
            "reason": "No clear preference was detected, so globally strong MovieLens results are returned first.",
        }
        for row in candidate_df.itertuples(index=False)
    ]


def recommend_videos(user_input: str | list[str], k: int = 5) -> list[dict]:
    raw_query, preferred_genres = _extract_preference_terms(user_input)
    if not raw_query:
        return _fallback_popular_movies(k)

    assets = _load_runtime_assets()
    catalog = assets["catalog"]
    genre_classes = assets["genre_classes"]
    model = assets["model"]
    device = assets["device"]
    stats = assets["stats"]

    recalls = content_recall_by_query(
        raw_query=raw_query,
        preferred_terms=preferred_genres,
        k=max(k * 10, 20),
        exclude_movie_ids=set(),
    )
    if not recalls:
        return _fallback_popular_movies(k)

    candidate_ids = [movie_id for movie_id, _ in recalls]
    candidate_df = catalog[catalog["movie_id"].isin(candidate_ids)].copy()
    if candidate_df.empty:
        return _fallback_popular_movies(k)

    recall_scores = {movie_id: score for movie_id, score in recalls}
    candidate_df["recall_score"] = candidate_df["movie_id"].map(recall_scores).fillna(0.0)
    candidate_df = candidate_df.merge(
        stats[["movie_id", "avg_rating", "rating_count", "popularity_score"]],
        on="movie_id",
        how="left",
    )
    candidate_df[["avg_rating", "rating_count", "popularity_score"]] = candidate_df[
        ["avg_rating", "rating_count", "popularity_score"]
    ].fillna(0.0)

    user_vec = _binarize_terms(preferred_genres, genre_classes)
    movie_vecs = [_binarize_terms(genres, genre_classes) for genres in candidate_df["genres_list"]]
    X_pred = np.array([user_vec + movie_vec for movie_vec in movie_vecs], dtype=np.float32)
    X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32, device=device)

    with torch.no_grad():
        rank_scores = model(X_pred_tensor).squeeze().detach().cpu().numpy()
    if np.isscalar(rank_scores):
        rank_scores = np.array([float(rank_scores)])

    candidate_df["rank_score"] = rank_scores
    popularity = candidate_df["popularity_score"].to_numpy(dtype=np.float32)
    popularity_max = float(popularity.max()) if len(popularity) else 0.0
    if popularity_max > 0:
        popularity = popularity / popularity_max
    candidate_df["final_score"] = (
        0.6 * candidate_df["rank_score"]
        + 0.25 * candidate_df["recall_score"]
        + 0.15 * popularity
    )
    candidate_df = candidate_df.sort_values(by="final_score", ascending=False).head(k)

    results = []
    for row in candidate_df.itertuples(index=False):
        explanation = generate_explanation(
            user_tags=preferred_genres,
            video_tags=row.genres_list,
            score=float(row.final_score),
            title=row.title,
        )
        results.append(
            {
                "movie_id": int(row.movie_id),
                "title": row.title,
                "year": int(row.year) if not pd.isna(row.year) else None,
                "score": round(float(row.final_score), 4),
                "genres": row.genres_list,
                "reason": explanation,
            }
        )
    return results


def format_recommendations(results: list[dict]) -> str:
    if not results:
        return "No suitable movies were found. Try a clearer genre or title keyword."

    lines = []
    for idx, item in enumerate(results, start=1):
        year_text = f" ({item['year']})" if item.get("year") else ""
        genres_text = " / ".join(item.get("genres", [])) or "Unknown"
        lines.append(
            f"{idx}. {item['title']}{year_text}\n"
            f"   Genres: {genres_text}\n"
            f"   Score: {item['score']:.4f}\n"
            f"   Reason: {item['reason']}"
        )
    return "\n".join(lines)
