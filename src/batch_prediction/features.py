"""Feature construction for batch scoring.

The served model takes two features, ``watch_time_seconds`` and
``watch_ratio``. A request carries neither directly: it carries ``watch_time``
and a ``video_id``, and the ratio needs the video's duration. That lookup, plus
identifier resolution, is what this module exists for.

One distinction is easy to miss: the **API contract** and the **feature set**
are different things. ``hour_of_day`` is accepted, validated and echoed back
because the specification defines it as an input. It is not a model feature.
Accepting a field and training on it are separate decisions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

# The model's inputs, in the order the estimator was fitted on.
FEATURE_COLUMNS = ["watch_time_seconds", "watch_ratio"]

MAX_WATCH_TIME_SECONDS = 3600.0
USER_ID_PATTERN = re.compile(r"user_[A-Za-z0-9_-]+")
VIDEO_ID_PATTERN = re.compile(r"video_[A-Za-z0-9_-]+")


def validate_ids(user_id: str, video_id: str) -> None:
    """Validate identifier shape without assuming a fixed digit count."""
    if not USER_ID_PATTERN.fullmatch(str(user_id)):
        raise ValueError(
            "user_id must start with 'user_' and contain only letters, digits, "
            "underscores or hyphens"
        )
    if not VIDEO_ID_PATTERN.fullmatch(str(video_id)):
        raise ValueError(
            "video_id must start with 'video_' and contain only letters, digits, "
            "underscores or hyphens"
        )


def _required_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(sorted(missing))}")


@dataclass
class FeatureStore:
    """Identifier resolution and online feature construction.

    The store exists for two reasons even though the model uses only two
    features: an unknown identifier must produce an error rather than a
    prediction, and ``watch_ratio`` needs the video's duration, which a request
    does not carry.
    """

    users: pd.DataFrame
    videos: pd.DataFrame

    def __post_init__(self) -> None:
        self.user_ids = set(self.users["user_id"].astype(str))
        self.video_ids = set(self.videos["video_id"].astype(str))
        self.durations = dict(
            zip(
                self.videos["video_id"].astype(str),
                pd.to_numeric(self.videos["duration_seconds"], errors="coerce").fillna(0.0),
                strict=True,
            )
        )

    @classmethod
    def from_csv(cls, users_path: str, videos_path: str) -> "FeatureStore":
        users = pd.read_csv(users_path, usecols=["user_id"])
        videos = pd.read_csv(videos_path, usecols=["video_id", "duration_seconds"])
        _required_columns(users, {"user_id"}, "users.csv")
        _required_columns(videos, {"video_id", "duration_seconds"}, "videos.csv")
        return cls(users=users, videos=videos)

    def build_row(self, user_id: str, video_id: str, watch_time: float) -> dict[str, float]:
        """Build the feature values for one request.

        Raises ``ValueError`` for a malformed identifier or an out-of-range
        watch time, and ``KeyError`` for an identifier that is well formed but
        absent. Batch scoring maps both onto a per-row ``error`` string rather
        than failing the whole request.
        """
        validate_ids(user_id, video_id)
        user_id, video_id = str(user_id), str(video_id)
        if user_id not in self.user_ids:
            raise KeyError(f"unknown user_id: {user_id}")
        if video_id not in self.video_ids:
            raise KeyError(f"unknown video_id: {video_id}")
        try:
            seconds = float(watch_time)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"watch_time is not a number: {watch_time!r}") from exc
        if seconds != seconds:  # NaN
            raise ValueError("watch_time is not a number: nan")
        if not 0 <= seconds <= MAX_WATCH_TIME_SECONDS:
            raise ValueError(f"watch_time must be between 0 and {MAX_WATCH_TIME_SECONDS:.0f}")
        duration = float(self.durations.get(video_id, 0.0))
        ratio = min(max(seconds / duration, 0.0), 1.0) if duration else 0.0
        return {"watch_time_seconds": seconds, "watch_ratio": ratio}

    def build_one(self, user_id: str, video_id: str, watch_time: float) -> pd.DataFrame:
        """Single-row frame, for the one-off ``/predict`` route."""
        return pd.DataFrame([self.build_row(user_id, video_id, watch_time)],
                            columns=FEATURE_COLUMNS)

    def build_many(
        self, requests: list[tuple[str, str, float]]
    ) -> tuple[pd.DataFrame, list[int], dict[int, str]]:
        """Build features for a batch, isolating per-row failures.

        Returns the frame of rows that could be built, the original index of
        each of those rows, and a mapping from original index to error message
        for the rows that could not. Specification section 4.4: one malformed
        row must not cost the caller the other ninety-nine.
        """
        rows: list[dict[str, float]] = []
        kept: list[int] = []
        errors: dict[int, str] = {}
        for position, (user_id, video_id, watch_time) in enumerate(requests):
            try:
                rows.append(self.build_row(user_id, video_id, watch_time))
                kept.append(position)
            except KeyError as exc:
                errors[position] = str(exc).strip("'")
            except ValueError as exc:
                errors[position] = str(exc)
        frame = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        return frame, kept, errors
