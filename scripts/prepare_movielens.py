from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_config import find_movielens_dir, get_artifact_paths
from scripts.build_ctr_samples import build_ctr_samples
from scripts.train_ctr import train_ctr_model
from scripts.train_two_tower import train_two_tower_model, export_two_tower_item_embeddings


YEAR_PATTERN = re.compile(r"\((\d{4})\)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local MovieLens 1M artifacts for LLM4Rec.")
    parser.add_argument("--encoder", choices=["tfidf"], default="tfidf")
    parser.add_argument("--max-positive-per-user", type=int, default=20)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


def load_movies(movies_path: Path) -> pd.DataFrame:
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    movies["movie_id"] = movies["movie_id"].astype(int)
    movies["year"] = movies["title"].apply(extract_year)
    movies["title_clean"] = movies["title"].apply(remove_year_suffix)
    movies["genres"] = movies["genres"].fillna("")
    movies["full_text"] = movies.apply(
        lambda row: f"{row['title_clean']} Genres: {row['genres'].replace('|', ' ')}",
        axis=1,
    )
    return movies


def load_ratings(ratings_path: Path) -> pd.DataFrame:
    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    return ratings


def load_users(users_path: Path) -> pd.DataFrame:
    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        encoding="latin-1",
    )
    users["user_id"] = users["user_id"].astype(int)
    return users


def extract_year(title: str) -> int | None:
    match = YEAR_PATTERN.search(str(title))
    return int(match.group(1)) if match else None


def remove_year_suffix(title: str) -> str:
    return YEAR_PATTERN.sub("", str(title)).strip()


def build_movie_catalog(movies: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    catalog = movies[["movie_id", "title", "title_clean", "year", "genres", "full_text"]].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(output_path, index=False)
    return catalog


def build_movie_stats(ratings: pd.DataFrame, movies: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    stats = (
        ratings.groupby("movie_id")
        .agg(avg_rating=("rating", "mean"), rating_count=("rating", "count"))
        .reset_index()
    )
    stats["popularity_score"] = stats["avg_rating"] * np.log1p(stats["rating_count"])
    stats = stats.merge(
        movies[["movie_id", "title", "genres"]],
        on="movie_id",
        how="left",
    )
    stats.to_csv(output_path, index=False)
    return stats


def build_user_behavior(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    users: pd.DataFrame,
    output_path: Path,
) -> dict[str, dict[str, object]]:
    movie_genres = {
        int(row.movie_id): [
            genre
            for genre in str(row.genres).split("|")
            if genre and genre != "(no genres listed)"
        ]
        for row in movies[["movie_id", "genres"]].itertuples(index=False)
    }
    user_meta = users.set_index("user_id").to_dict("index")
    behavior: dict[str, dict[str, object]] = {}

    for user_id, user_ratings in ratings.groupby("user_id"):
        positive = user_ratings[user_ratings["rating"] >= 4].sort_values(
            "timestamp",
            ascending=False,
        )
        watched_movies = positive["movie_id"].astype(int).head(20).tolist()
        preferred_genres: list[str] = []
        for movie_id in positive["movie_id"].astype(int).tolist():
            preferred_genres.extend(movie_genres.get(movie_id, []))

        genre_counts = (
            pd.Series(preferred_genres).value_counts().head(5).index.tolist()
            if preferred_genres
            else []
        )
        profile = {
            "watched_movies": watched_movies,
            "preferred_genres": genre_counts,
        }
        if user_id in user_meta:
            profile["profile"] = {
                "gender": user_meta[user_id]["gender"],
                "age": int(user_meta[user_id]["age"]),
                "occupation": int(user_meta[user_id]["occupation"]),
            }
        behavior[f"user_{int(user_id)}"] = profile

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(behavior, indent=2), encoding="utf-8")
    return behavior


def build_itemcf_neighbors(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    output_path: Path,
    positive_threshold: int = 4,
    topn: int = 100,
) -> dict[str, list[list[float | int]]]:
    positive = ratings[ratings["rating"] >= positive_threshold][["user_id", "movie_id"]].drop_duplicates()

    user_ids = sorted(positive["user_id"].unique().tolist())
    movie_ids = sorted(movies["movie_id"].astype(int).unique().tolist())
    user_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    movie_index = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

    row_idx = positive["movie_id"].map(movie_index).to_numpy()
    col_idx = positive["user_id"].map(user_index).to_numpy()
    data = np.ones(len(positive), dtype=np.float32)

    item_user_matrix = sparse.csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(movie_ids), len(user_ids)),
        dtype=np.float32,
    )

    sim_matrix = cosine_similarity(item_user_matrix, dense_output=True)
    np.fill_diagonal(sim_matrix, 0.0)

    neighbors: dict[str, list[list[float | int]]] = {}
    for movie_id in movie_ids:
        idx = movie_index[movie_id]
        scores = sim_matrix[idx]
        top_indices = np.argsort(scores)[::-1][:topn]
        movie_neighbors = []
        for neighbor_idx in top_indices:
            score = float(scores[neighbor_idx])
            if score <= 0:
                continue
            movie_neighbors.append([int(movie_ids[neighbor_idx]), round(score, 6)])
        neighbors[str(int(movie_id))] = movie_neighbors

    output_path.write_text(json.dumps(neighbors), encoding="utf-8")
    return neighbors


def build_recall_artifacts(
    catalog: pd.DataFrame,
    output_embeddings: Path,
    output_index: Path,
    output_vectorizer: Path,
    output_config: Path,
    output_id_map: Path,
) -> dict[str, str | int]:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    embeddings = vectorizer.fit_transform(catalog["full_text"]).astype(np.float32).toarray()
    faiss.normalize_L2(embeddings)

    np.save(output_embeddings, embeddings)
    joblib.dump(vectorizer, output_vectorizer)
    output_id_map.write_text(
        "\n".join(str(int(movie_id)) for movie_id in catalog["movie_id"].tolist()),
        encoding="utf-8",
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(output_index))

    config = {
        "encoder_type": "tfidf",
        "dimension": int(embeddings.shape[1]),
        "index_type": "IndexFlatIP",
    }
    output_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main() -> None:
    args = parse_args()
    paths = get_artifact_paths()
    movielens_dir = find_movielens_dir()

    movies = load_movies(movielens_dir / "movies.dat")
    ratings = load_ratings(movielens_dir / "ratings.dat")
    users = load_users(movielens_dir / "users.dat")

    catalog = build_movie_catalog(movies, paths.movie_catalog_csv)
    build_movie_stats(ratings, movies, paths.movie_stats_csv)
    build_user_behavior(ratings, movies, users, paths.user_behavior_json)
    build_itemcf_neighbors(ratings, movies, paths.itemcf_neighbors_json)
    build_recall_artifacts(
        catalog=catalog,
        output_embeddings=paths.recall_embeddings_npy,
        output_index=paths.recall_index_faiss,
        output_vectorizer=paths.recall_vectorizer_joblib,
        output_config=paths.recall_config_json,
        output_id_map=paths.movie_id_map_txt,
    )
    build_ctr_samples(
        movies_df=catalog,
        ratings_df=ratings,
        output_path=paths.ctr_samples_csv,
        max_positive_per_user=args.max_positive_per_user,
        negatives_per_positive=args.negatives_per_positive,
    )
    metrics = train_ctr_model(
        samples_path=paths.ctr_samples_csv,
        epochs=args.epochs,
    )
    two_tower_metrics = train_two_tower_model(
        samples_path=paths.ctr_samples_csv,
        epochs=args.epochs,
    )
    two_tower_export = export_two_tower_item_embeddings()

    summary = {
        "movielens_dir": str(movielens_dir.relative_to(PROJECT_ROOT)),
        "catalog_rows": int(len(catalog)),
        "ratings_rows": int(len(ratings)),
        "artifacts_dir": str(paths.movie_catalog_csv.parent.relative_to(PROJECT_ROOT)),
        "model_path": str(paths.ctr_model_pt.relative_to(PROJECT_ROOT)),
        "train_metrics": metrics,
        "two_tower_metrics": two_tower_metrics,
        "two_tower_export": two_tower_export,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
