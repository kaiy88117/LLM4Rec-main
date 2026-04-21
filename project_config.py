from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_ML_DIR = DATA_DIR / "ml-1m"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"


class LLM4RecError(RuntimeError):
    """Base error for repository-specific runtime failures."""


class MissingRawDataError(LLM4RecError):
    """Raised when local MovieLens raw files cannot be found."""


class MissingArtifactError(LLM4RecError):
    """Raised when generated recommendation artifacts are missing."""


@dataclass(frozen=True)
class ArtifactPaths:
    movie_catalog_csv: Path
    movie_stats_csv: Path
    user_behavior_json: Path
    itemcf_neighbors_json: Path
    recall_index_faiss: Path
    recall_embeddings_npy: Path
    recall_vectorizer_joblib: Path
    recall_config_json: Path
    movie_id_map_txt: Path
    two_tower_model_pt: Path
    two_tower_movie_embeddings_npy: Path
    two_tower_movie_id_map_txt: Path
    ctr_samples_csv: Path
    ctr_model_pt: Path
    deepfm_model_pt: Path


def ensure_runtime_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def get_artifact_paths() -> ArtifactPaths:
    ensure_runtime_dirs()
    return ArtifactPaths(
        movie_catalog_csv=PROCESSED_DIR / "movies_catalog.csv",
        movie_stats_csv=PROCESSED_DIR / "movie_stats.csv",
        user_behavior_json=PROCESSED_DIR / "user_behavior.json",
        itemcf_neighbors_json=PROCESSED_DIR / "itemcf_neighbors.json",
        recall_index_faiss=PROCESSED_DIR / "movie_embeddings.faiss",
        recall_embeddings_npy=PROCESSED_DIR / "movie_embeddings.npy",
        recall_vectorizer_joblib=PROCESSED_DIR / "recall_vectorizer.joblib",
        recall_config_json=PROCESSED_DIR / "recall_config.json",
        movie_id_map_txt=PROCESSED_DIR / "movie_id_map.txt",
        two_tower_model_pt=MODELS_DIR / "two_tower_model.pt",
        two_tower_movie_embeddings_npy=PROCESSED_DIR / "two_tower_movie_embeddings.npy",
        two_tower_movie_id_map_txt=PROCESSED_DIR / "two_tower_movie_id_map.txt",
        ctr_samples_csv=PROCESSED_DIR / "train_ctr_samples.csv",
        ctr_model_pt=MODELS_DIR / "ctr_mlp_model.pt",
        deepfm_model_pt=MODELS_DIR / "deepfm_model.pt",
    )


def get_torch_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_movielens_dir() -> Path:
    candidates = [RAW_ML_DIR, RAW_ML_DIR / "ml-1m"]
    required = ("movies.dat", "ratings.dat", "users.dat")

    for candidate in candidates:
        if all((candidate / name).exists() for name in required):
            return candidate

    hint = (
        "Expected local MovieLens 1M files under "
        f"'{RAW_ML_DIR.relative_to(PROJECT_ROOT)}' or "
        f"'{(RAW_ML_DIR / 'ml-1m').relative_to(PROJECT_ROOT)}'."
    )
    raise MissingRawDataError(
        "MovieLens 1M raw data is missing. "
        "Place 'movies.dat', 'ratings.dat', and 'users.dat' in "
        f"'{RAW_ML_DIR.relative_to(PROJECT_ROOT)}'. {hint}"
    )


def require_paths(paths: list[Path], build_hint: str) -> None:
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return

    missing_text = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
    raise MissingArtifactError(
        f"Required files are missing: {missing_text}. {build_hint}"
    )
