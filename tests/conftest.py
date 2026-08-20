"""Shared fixtures.

The tests run against the real shipped model and the real lookup tables. That
is deliberate: the batch route's contract is about how it behaves with genuine
identifiers and genuine failures, and a stubbed store would not exercise the
identifier resolution that produces most per-row errors.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from batch_prediction.api import create_app
from batch_prediction.config import DATA_DIR, METADATA_PATH, MODEL_PATH
from batch_prediction.features import FeatureStore

VALID_USER = "user_000001"
VALID_VIDEO = "video_0000001"      # duration 55 seconds
SHORT_VIDEO = "video_0000003"      # duration 20 seconds


@pytest.fixture(scope="session")
def store() -> FeatureStore:
    return FeatureStore.from_csv(str(DATA_DIR / "users.csv"), str(DATA_DIR / "videos.csv"))


@pytest.fixture(scope="session")
def client() -> TestClient:
    app = create_app(
        model_path=MODEL_PATH,
        metadata_path=METADATA_PATH,
        users_path=DATA_DIR / "users.csv",
        videos_path=DATA_DIR / "videos.csv",
    )
    return TestClient(app)


@pytest.fixture
def small_store() -> FeatureStore:
    """A two-video store, including a zero-duration row the real data lacks."""
    users = pd.DataFrame({"user_id": ["user_000001", "user_000002"]})
    videos = pd.DataFrame({
        "video_id": ["video_0000001", "video_zero"],
        "duration_seconds": [50, 0],
    })
    return FeatureStore(users=users, videos=videos)


def request_payload(user_id=VALID_USER, video_id=VALID_VIDEO, watch_time=30.0, hour=None):
    body = {"user_id": user_id, "video_id": video_id, "watch_time": watch_time}
    if hour is not None:
        body["hour_of_day"] = hour
    return body
