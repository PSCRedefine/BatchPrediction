"""Feature construction and per-row fault isolation (specification 3.1, 4.4)."""

from __future__ import annotations

import numpy as np
import pytest

from batch_prediction.features import (
    FEATURE_COLUMNS,
    MAX_WATCH_TIME_SECONDS,
    validate_ids,
)

from .conftest import SHORT_VIDEO, VALID_USER, VALID_VIDEO


class TestValidateIds:
    def test_accepts_well_formed_identifiers(self):
        validate_ids("user_000001", "video_0000001")

    def test_accepts_identifiers_of_any_length(self):
        validate_ids("user_a", "video_b-1_2")

    @pytest.mark.parametrize("user_id", ["000001", "usr_000001", "", "user 1"])
    def test_rejects_malformed_user_id(self, user_id):
        with pytest.raises(ValueError, match="user_id"):
            validate_ids(user_id, "video_0000001")

    @pytest.mark.parametrize("video_id", ["0000001", "vid_1", "", "video 1"])
    def test_rejects_malformed_video_id(self, video_id):
        with pytest.raises(ValueError, match="video_id"):
            validate_ids("user_000001", video_id)


class TestBuildRow:
    def test_returns_the_two_model_features(self, store):
        row = store.build_row(VALID_USER, VALID_VIDEO, 30.0)
        assert set(row) == set(FEATURE_COLUMNS)

    def test_watch_ratio_is_watch_time_over_duration(self, store):
        row = store.build_row(VALID_USER, SHORT_VIDEO, 10.0)   # 20 second video
        assert row["watch_ratio"] == pytest.approx(0.5)

    def test_watch_ratio_is_clipped_at_one(self, store):
        row = store.build_row(VALID_USER, SHORT_VIDEO, 600.0)
        assert row["watch_ratio"] == 1.0

    def test_zero_duration_video_yields_ratio_zero(self, small_store):
        row = small_store.build_row("user_000001", "video_zero", 12.0)
        assert row["watch_ratio"] == 0.0
        assert row["watch_time_seconds"] == 12.0

    def test_unknown_user_raises_key_error(self, store):
        with pytest.raises(KeyError, match="unknown user_id"):
            store.build_row("user_999999999", VALID_VIDEO, 10.0)

    def test_unknown_video_raises_key_error(self, store):
        with pytest.raises(KeyError, match="unknown video_id"):
            store.build_row(VALID_USER, "video_999999999", 10.0)

    def test_malformed_identifier_raises_value_error(self, store):
        with pytest.raises(ValueError):
            store.build_row("nope", VALID_VIDEO, 10.0)

    def test_negative_watch_time_raises_value_error(self, store):
        with pytest.raises(ValueError, match="watch_time must be between"):
            store.build_row(VALID_USER, VALID_VIDEO, -1.0)

    def test_absurd_watch_time_raises_value_error(self, store):
        with pytest.raises(ValueError, match="watch_time must be between"):
            store.build_row(VALID_USER, VALID_VIDEO, MAX_WATCH_TIME_SECONDS + 1)

    def test_non_numeric_watch_time_raises_value_error(self, store):
        with pytest.raises(ValueError, match="not a number"):
            store.build_row(VALID_USER, VALID_VIDEO, "abc")

    def test_nan_watch_time_raises_value_error(self, store):
        with pytest.raises(ValueError, match="not a number"):
            store.build_row(VALID_USER, VALID_VIDEO, np.nan)


class TestBuildMany:
    def test_all_valid_rows_are_kept_in_order(self, store):
        frame, kept, errors = store.build_many([
            (VALID_USER, VALID_VIDEO, 10.0),
            (VALID_USER, SHORT_VIDEO, 20.0),
        ])
        assert kept == [0, 1]
        assert errors == {}
        assert len(frame) == 2
        assert list(frame.columns) == FEATURE_COLUMNS

    def test_one_bad_row_does_not_lose_the_others(self, store):
        frame, kept, errors = store.build_many([
            (VALID_USER, VALID_VIDEO, 10.0),
            ("user_999999999", VALID_VIDEO, 10.0),
            (VALID_USER, SHORT_VIDEO, 5.0),
        ])
        assert kept == [0, 2]
        assert set(errors) == {1}
        assert "unknown user_id" in errors[1]
        assert len(frame) == 2

    def test_every_row_can_fail(self, store):
        frame, kept, errors = store.build_many([
            ("bad", VALID_VIDEO, 1.0),
            (VALID_USER, "bad", 1.0),
        ])
        assert kept == []
        assert len(frame) == 0
        assert set(errors) == {0, 1}

    def test_empty_batch_returns_empty_frame(self, store):
        frame, kept, errors = store.build_many([])
        assert kept == [] and errors == {} and len(frame) == 0
