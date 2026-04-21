from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import get_artifact_paths


POSITIVE_THRESHOLD = 4
NEGATIVE_THRESHOLD = 2


def _genres_to_list(genres_value: str) -> list[str]:
    if not genres_value:
        return []
    return [
        genre
        for genre in str(genres_value).split("|")
        if genre and genre != "(no genres listed)"
    ]


def _binarize_terms(tag_list: list[str], classes: list[str]) -> list[int]:
    tag_set = set(tag_list)
    return [1 if item in tag_set else 0 for item in classes]


def build_ctr_samples(
    movies_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    output_path: Path | None = None,
    max_positive_per_user: int = 20,
    negatives_per_positive: int = 2,
    random_seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(random_seed)
    paths = get_artifact_paths()
    output_path = output_path or paths.ctr_samples_csv

    movies = movies_df.copy()
    movies["genres_list"] = movies["genres"].apply(_genres_to_list)
    genre_classes = sorted({genre for genres in movies["genres_list"] for genre in genres})
    movie_genre_map = {
        int(row.movie_id): row.genres_list
        for row in movies[["movie_id", "genres_list"]].itertuples(index=False)
    }
    all_movie_ids = list(movie_genre_map.keys())

    ratings = ratings_df.copy()
    ratings["movie_id"] = ratings["movie_id"].astype(int)

    samples: list[dict[str, int]] = []

    for user_id, user_ratings in ratings.groupby("user_id"):
        positives = user_ratings[user_ratings["rating"] >= POSITIVE_THRESHOLD]["movie_id"].tolist()
        negatives = user_ratings[user_ratings["rating"] <= NEGATIVE_THRESHOLD]["movie_id"].tolist()

        if not positives:
            continue

        if max_positive_per_user > 0 and len(positives) > max_positive_per_user:
            positives = rng.sample(positives, max_positive_per_user)

        preferred_genres: list[str] = []
        for movie_id in user_ratings[user_ratings["rating"] >= POSITIVE_THRESHOLD]["movie_id"].tolist():
            preferred_genres.extend(movie_genre_map.get(int(movie_id), []))

        if not preferred_genres:
            continue

        positive_count = len(positives)
        target_negative_count = max(positive_count * negatives_per_positive, positive_count)

        explicit_negatives = [movie_id for movie_id in negatives if movie_id in movie_genre_map]
        sampled_negatives = explicit_negatives[:target_negative_count]

        if len(sampled_negatives) < target_negative_count:
            rated_movie_ids = set(user_ratings["movie_id"].tolist())
            unrated_candidates = [
                movie_id for movie_id in all_movie_ids if movie_id not in rated_movie_ids
            ]
            if unrated_candidates:
                sample_size = min(
                    target_negative_count - len(sampled_negatives),
                    len(unrated_candidates),
                )
                sampled_negatives.extend(rng.sample(unrated_candidates, sample_size))

        user_vec = _binarize_terms(preferred_genres, genre_classes)

        for movie_id in positives:
            movie_vec = _binarize_terms(movie_genre_map.get(int(movie_id), []), genre_classes)
            samples.append(
                {
                    "user_id": int(user_id),
                    "movie_id": int(movie_id),
                    **{f"u_{genre}": value for genre, value in zip(genre_classes, user_vec)},
                    **{f"v_{genre}": value for genre, value in zip(genre_classes, movie_vec)},
                    "label": 1,
                }
            )

        for movie_id in sampled_negatives:
            movie_vec = _binarize_terms(movie_genre_map.get(int(movie_id), []), genre_classes)
            samples.append(
                {
                    "user_id": int(user_id),
                    "movie_id": int(movie_id),
                    **{f"u_{genre}": value for genre, value in zip(genre_classes, user_vec)},
                    **{f"v_{genre}": value for genre, value in zip(genre_classes, movie_vec)},
                    "label": 0,
                }
            )

    train_df = pd.DataFrame(samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_path, index=False)
    return train_df


def load_movie_catalog(catalog_path: Path | None = None) -> pd.DataFrame:
    paths = get_artifact_paths()
    catalog_path = catalog_path or paths.movie_catalog_csv
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Movie catalog not found at '{catalog_path}'. Run `python -m scripts.prepare_movielens` first."
        )
    return pd.read_csv(catalog_path)


def load_ratings(ratings_path: Path) -> pd.DataFrame:
    return pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )


if __name__ == "__main__":
    from project_config import find_movielens_dir

    artifact_paths = get_artifact_paths()
    movielens_dir = find_movielens_dir()
    movies = load_movie_catalog()
    ratings = load_ratings(movielens_dir / "ratings.dat")
    df = build_ctr_samples(movies, ratings, output_path=artifact_paths.ctr_samples_csv)
    print(json.dumps({"rows": len(df), "output": str(artifact_paths.ctr_samples_csv)}, indent=2))
