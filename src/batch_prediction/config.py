"""Paths, limits and environment-backed settings."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def env_path(name: str, default: Path) -> Path:
    """Return an absolute path from an environment variable or a project default."""
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


USERS_PATH = env_path("BATCH_PREDICTION_USERS_PATH", DATA_DIR / "users.csv")
VIDEOS_PATH = env_path("BATCH_PREDICTION_VIDEOS_PATH", DATA_DIR / "videos.csv")
MODEL_PATH = env_path("BATCH_PREDICTION_MODEL_PATH", MODELS_DIR / "best_model.joblib")
METADATA_PATH = env_path(
    "BATCH_PREDICTION_METADATA_PATH", MODELS_DIR / "model_metadata.json"
)

# Specification section 3.1/4.2. One number, enforced in three places: the
# Streamlit page truncates, the request model rejects, and the route re-checks.
MAX_BATCH_SIZE = 100

# Confidence bands. These describe how far a prediction sits from a coin flip,
# not how likely it is to be correct.
CONFIDENCE_HIGH = 0.80
CONFIDENCE_MEDIUM = 0.60

API_BASE_URL = os.getenv("BATCH_PREDICTION_API_URL", "http://127.0.0.1:8000")
