"""Turning an uploaded table into a request body.

These rules come from section 3.1 of docs/SPEC.md. They live here rather than
in the Streamlit page so they can be tested without starting a browser, and so
a caller that is not the page — a script, a notebook — gets the same
conversions.
"""

from __future__ import annotations

import pandas as pd

from .config import MAX_BATCH_SIZE

REQUIRED_COLUMNS = ["user_id", "video_id", "watch_time"]
OPTIONAL_COLUMNS = ["hour_of_day"]


def missing_columns(frame: pd.DataFrame) -> list[str]:
    """Required columns absent from an uploaded frame, in specification order."""
    return [column for column in REQUIRED_COLUMNS if column not in frame.columns]


def normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the type conversions from specification 3.1.4.

    Identifiers become strings, ``watch_time`` becomes a float and
    ``hour_of_day`` becomes a nullable integer. Values that cannot be converted
    become null rather than raising: a batch reports bad rows, it does not
    refuse the whole file because of one of them.
    """
    frame = frame.copy()
    frame["user_id"] = frame["user_id"].astype(str).str.strip()
    frame["video_id"] = frame["video_id"].astype(str).str.strip()
    frame["watch_time"] = pd.to_numeric(frame["watch_time"], errors="coerce").astype(float)
    if "hour_of_day" in frame.columns:
        frame["hour_of_day"] = pd.to_numeric(frame["hour_of_day"], errors="coerce").astype("Int64")
    return frame


def prepare(frame: pd.DataFrame, limit: int = MAX_BATCH_SIZE) -> pd.DataFrame:
    """Keep the columns the API knows about, truncate, and convert types.

    Truncation is specification 3.1.3: the service caps a batch at 100, so the
    page sends 100 rather than sending 500 and getting a 400 back.
    """
    keep = [*REQUIRED_COLUMNS, *[c for c in OPTIONAL_COLUMNS if c in frame.columns]]
    return normalise(frame[keep].head(limit))


def to_payload(frame: pd.DataFrame) -> dict[str, list[dict[str, object]]]:
    """Build the ``/predict/batch`` body from a prepared frame.

    ``hour_of_day`` is omitted per row when it is absent or unreadable, because
    the field is optional and sending null would fail schema validation for the
    whole batch.
    """
    requests: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        watch_time = row["watch_time"]
        item: dict[str, object] = {
            "user_id": str(row["user_id"]),
            "video_id": str(row["video_id"]),
            "watch_time": 0.0 if pd.isna(watch_time) else float(watch_time),
        }
        hour = row.get("hour_of_day")
        if hour is not None and not pd.isna(hour):
            item["hour_of_day"] = int(hour)
        requests.append(item)
    return {"requests": requests}
