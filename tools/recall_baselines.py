from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import faiss
import joblib
import numpy as np
import pandas as pd

from project_config import get_artifact_paths, require_paths


def _split_genres(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [genre for genre in str(value).split("|") if genre and genre != "(no genres listed)"]


@lru_cache(maxsize=1)
def load_catalog_assets() -> dict[str, Any]:
    paths = get_artifact_paths()
    require_paths(
        [
            paths.movie_catalog_csv,
            paths.movie_stats_csv,
            paths.recall_index_faiss,
            paths.recall_vectorizer_joblib,
            paths.movie_id_map_txt,
            paths.itemcf_neighbors_json,
        ],
        build_hint="Run `python -m scripts.prepare_movielens` from the project root.",
    )

    catalog = pd.read_csv(paths.movie_catalog_csv)
    catalog["genres_list"] = catalog["genres"].apply(_split_genres)
    stats = pd.read_csv(paths.movie_stats_csv)
    vectorizer = joblib.load(paths.recall_vectorizer_joblib)
    index = faiss.read_index(str(paths.recall_index_faiss))
    movie_id_map = [
        int(line)
        for line in paths.movie_id_map_txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    itemcf_neighbors = json.loads(paths.itemcf_neighbors_json.read_text(encoding="utf-8"))
    return {
        "catalog": catalog,
        "stats": stats,
        "vectorizer": vectorizer,
        "index": index,
        "movie_id_map": movie_id_map,
        "itemcf_neighbors": itemcf_neighbors,
    }


def build_profile_text_from_history(history_movie_ids: list[int]) -> str:
    assets = load_catalog_assets()
    catalog = assets["catalog"]
    history_df = catalog[catalog["movie_id"].isin(history_movie_ids)]
    if history_df.empty:
        return ""

    parts = []
    for row in history_df.itertuples(index=False):
        parts.append(f"{row.title_clean} {str(row.genres).replace('|', ' ')}")
    return " ".join(parts)


def content_recall_by_query(
    raw_query: str,
    preferred_terms: list[str] | None = None,
    k: int = 50,
    exclude_movie_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    assets = load_catalog_assets()
    vectorizer = assets["vectorizer"]
    index = assets["index"]
    movie_id_map = assets["movie_id_map"]

    query_text = " ".join((preferred_terms or []) + [raw_query]).strip()
    if not query_text:
        return []

    query_vec = vectorizer.transform([query_text]).astype(np.float32).toarray()
    if np.linalg.norm(query_vec) == 0:
        return []

    faiss.normalize_L2(query_vec)
    search_size = min(max(k * 3, k), len(movie_id_map))
    distances, indices = index.search(query_vec, search_size)

    exclude_movie_ids = exclude_movie_ids or set()
    results: list[tuple[int, float]] = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        movie_id = movie_id_map[idx]
        if movie_id in exclude_movie_ids:
            continue
        results.append((movie_id, float(score)))
        if len(results) >= k:
            break
    return results


def content_recall_by_history(
    history_movie_ids: list[int],
    k: int = 50,
    exclude_movie_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    profile_text = build_profile_text_from_history(history_movie_ids)
    return content_recall_by_query(
        raw_query=profile_text,
        preferred_terms=[],
        k=k,
        exclude_movie_ids=exclude_movie_ids or set(history_movie_ids),
    )


def itemcf_recall(
    history_movie_ids: list[int],
    k: int = 50,
    exclude_movie_ids: set[int] | None = None,
) -> list[tuple[int, float]]:
    assets = load_catalog_assets()
    neighbors = assets["itemcf_neighbors"]
    exclude = set(exclude_movie_ids or set())
    exclude.update(history_movie_ids)

    scores: dict[int, float] = {}
    for movie_id in history_movie_ids:
        for neighbor_id, score in neighbors.get(str(int(movie_id)), []):
            neighbor_id = int(neighbor_id)
            if neighbor_id in exclude:
                continue
            scores[neighbor_id] = scores.get(neighbor_id, 0.0) + float(score)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return ranked[:k]
