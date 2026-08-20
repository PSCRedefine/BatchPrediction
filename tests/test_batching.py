"""Upload handling rules (specification 3.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from batch_prediction.batching import (
    REQUIRED_COLUMNS,
    missing_columns,
    normalise,
    prepare,
    to_payload,
)
from batch_prediction.config import MAX_BATCH_SIZE


def frame(rows: int = 3, **overrides) -> pd.DataFrame:
    data = {
        "user_id": [f"user_00000{i}" for i in range(rows)],
        "video_id": [f"video_000000{i}" for i in range(rows)],
        "watch_time": [10.0] * rows,
    }
    data.update(overrides)
    return pd.DataFrame(data)


class TestMissingColumns:
    def test_complete_frame_reports_nothing(self):
        assert missing_columns(frame()) == []

    def test_reports_each_absent_column(self):
        assert missing_columns(pd.DataFrame({"user_id": ["user_1"]})) == ["video_id", "watch_time"]

    def test_optional_column_is_never_required(self):
        assert missing_columns(frame()) == []
        assert "hour_of_day" not in REQUIRED_COLUMNS

    def test_extra_columns_are_not_a_problem(self):
        assert missing_columns(frame().assign(note=["a", "b", "c"])) == []


class TestNormalise:
    def test_identifiers_become_strings(self):
        out = normalise(pd.DataFrame({"user_id": [1], "video_id": [2], "watch_time": ["5"]}))
        assert out["user_id"].iloc[0] == "1"
        assert isinstance(out["video_id"].iloc[0], str)

    def test_identifiers_are_stripped(self):
        out = normalise(frame(1, user_id=["  user_000001  "]))
        assert out["user_id"].iloc[0] == "user_000001"

    def test_watch_time_becomes_float(self):
        out = normalise(frame(1, watch_time=["42"]))
        assert out["watch_time"].dtype == float
        assert out["watch_time"].iloc[0] == 42.0

    def test_unreadable_watch_time_becomes_nan_not_an_exception(self):
        out = normalise(frame(1, watch_time=["abc"]))
        assert np.isnan(out["watch_time"].iloc[0])

    def test_hour_of_day_becomes_nullable_integer(self):
        out = normalise(frame(2, hour_of_day=["14", None]))
        assert out["hour_of_day"].iloc[0] == 14
        assert pd.isna(out["hour_of_day"].iloc[1])

    def test_absent_hour_of_day_is_left_absent(self):
        assert "hour_of_day" not in normalise(frame())

    def test_does_not_mutate_the_caller_frame(self):
        original = frame(1, watch_time=["7"])
        normalise(original)
        assert original["watch_time"].iloc[0] == "7"


class TestPrepare:
    def test_truncates_to_the_batch_limit(self):
        assert len(prepare(frame(MAX_BATCH_SIZE + 50))) == MAX_BATCH_SIZE

    def test_keeps_a_short_frame_whole(self):
        assert len(prepare(frame(7))) == 7

    def test_drops_columns_the_api_does_not_know(self):
        out = prepare(frame(2).assign(note=["a", "b"], watch_ratio=[0.1, 0.2]))
        assert list(out.columns) == REQUIRED_COLUMNS

    def test_carries_the_optional_column_through(self):
        out = prepare(frame(2, hour_of_day=[9, 10]))
        assert list(out.columns) == [*REQUIRED_COLUMNS, "hour_of_day"]


class TestToPayload:
    def test_produces_one_request_per_row(self):
        payload = to_payload(prepare(frame(4)))
        assert len(payload["requests"]) == 4

    def test_omits_hour_of_day_when_absent(self):
        item = to_payload(prepare(frame(1)))["requests"][0]
        assert "hour_of_day" not in item

    def test_omits_hour_of_day_when_unreadable(self):
        item = to_payload(prepare(frame(1, hour_of_day=["nope"])))["requests"][0]
        assert "hour_of_day" not in item

    def test_includes_hour_of_day_as_an_integer(self):
        item = to_payload(prepare(frame(1, hour_of_day=["14"])))["requests"][0]
        assert item["hour_of_day"] == 14
        assert isinstance(item["hour_of_day"], int)

    def test_unreadable_watch_time_is_sent_as_zero(self):
        item = to_payload(prepare(frame(1, watch_time=["abc"])))["requests"][0]
        assert item["watch_time"] == 0.0

    def test_values_are_json_native_types(self):
        item = to_payload(prepare(frame(1, hour_of_day=[3])))["requests"][0]
        assert isinstance(item["user_id"], str)
        assert isinstance(item["watch_time"], float)
        assert type(item["hour_of_day"]) is int

    @pytest.mark.parametrize("rows", [1, 25, MAX_BATCH_SIZE])
    def test_payload_never_exceeds_the_limit(self, rows):
        assert len(to_payload(prepare(frame(rows)))["requests"]) <= MAX_BATCH_SIZE
