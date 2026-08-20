"""API contract tests (specification sections 3.2, 3.3 and 4)."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from batch_prediction.api import confidence_label, create_app, predict_probabilities
from batch_prediction.config import MAX_BATCH_SIZE

from .conftest import SHORT_VIDEO, VALID_USER, VALID_VIDEO, request_payload


class TestHealth:
    def test_reports_ok_when_loaded(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["store_loaded"] is True
        assert body["max_batch_size"] == MAX_BATCH_SIZE

    def test_reports_uptime_and_model_name(self, client):
        body = client.get("/health").json()
        assert body["uptime_seconds"] >= 0.0
        assert body["model_name"] != "unknown"

    def test_stays_serviceable_when_the_model_is_missing(self, tmp_path):
        app = create_app(model_path=tmp_path / "absent.joblib")
        with TestClient(app) as degraded:
            body = degraded.get("/health").json()
            assert body["status"] == "degraded"
            assert body["model_loaded"] is False
            assert degraded.post("/predict/batch",
                                 json={"requests": [request_payload()]}).status_code == 503


class TestModelInfo:
    def test_exposes_features_threshold_and_limit(self, client):
        body = client.get("/model/info").json()
        assert body["features"] == ["watch_time_seconds", "watch_ratio"]
        assert 0.0 < body["threshold"] < 1.0
        assert body["max_batch_size"] == MAX_BATCH_SIZE


class TestSinglePredict:
    def test_scores_a_valid_request(self, client):
        body = client.post("/predict", json=request_payload()).json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["confidence"] in {"low", "medium", "high"}

    def test_unknown_identifier_is_404(self, client):
        response = client.post("/predict", json=request_payload(user_id="user_999999999"))
        assert response.status_code == 404

    def test_defaults_hour_of_day_when_absent(self, client):
        body = client.post("/predict", json=request_payload()).json()
        assert 0 <= body["hour_of_day"] <= 23


class TestBatchHappyPath:
    def test_returns_one_result_per_request_in_order(self, client):
        payload = {"requests": [
            request_payload(watch_time=5.0),
            request_payload(video_id=SHORT_VIDEO, watch_time=15.0),
            request_payload(watch_time=55.0),
        ]}
        body = client.post("/predict/batch", json=payload).json()
        assert body["batch_size"] == 3
        assert body["successful"] == 3
        assert body["failed"] == 0
        assert [item["index"] for item in body["results"]] == [0, 1, 2]

    def test_every_probability_is_a_valid_probability(self, client):
        payload = {"requests": [request_payload(watch_time=t) for t in (0.0, 10.0, 60.0)]}
        body = client.post("/predict/batch", json=payload).json()
        for item in body["results"]:
            assert 0.0 <= item["probability"] <= 1.0
            assert item["confidence"] in {"low", "medium", "high"}
            assert isinstance(item["predicted_engaged"], bool)

    def test_reports_batch_size_and_response_time(self, client):
        body = client.post("/predict/batch", json={"requests": [request_payload()]}).json()
        assert body["batch_size"] == 1
        assert body["response_time_ms"] >= 0.0
        assert body["model_name"] and body["model_version"]

    def test_echoes_optional_hour_of_day(self, client):
        payload = {"requests": [request_payload(hour=14), request_payload()]}
        body = client.post("/predict/batch", json=payload).json()
        assert body["results"][0]["hour_of_day"] == 14
        assert body["results"][1]["hour_of_day"] is None

    def test_accepts_a_full_size_batch(self, client):
        payload = {"requests": [request_payload()] * MAX_BATCH_SIZE}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        assert response.json()["successful"] == MAX_BATCH_SIZE

    def test_longer_watch_time_scores_no_lower(self, client):
        payload = {"requests": [request_payload(watch_time=1.0), request_payload(watch_time=55.0)]}
        results = client.post("/predict/batch", json=payload).json()["results"]
        assert results[1]["probability"] >= results[0]["probability"]


class TestBatchFaultTolerance:
    """Specification 3.3 and 4.4: one bad row must not fail the batch."""

    def test_unknown_user_fails_only_its_own_row(self, client):
        payload = {"requests": [
            request_payload(),
            request_payload(user_id="user_999999999"),
            request_payload(video_id=SHORT_VIDEO),
        ]}
        body = client.post("/predict/batch", json=payload).json()
        assert body["batch_size"] == 3
        assert body["successful"] == 2
        assert body["failed"] == 1
        assert "unknown user_id" in body["results"][1]["error"]
        assert body["results"][1]["probability"] is None
        assert body["results"][0]["probability"] is not None
        assert body["results"][2]["probability"] is not None

    def test_unknown_video_fails_only_its_own_row(self, client):
        payload = {"requests": [request_payload(video_id="video_999999999"), request_payload()]}
        body = client.post("/predict/batch", json=payload).json()
        assert body["successful"] == 1
        assert "unknown video_id" in body["results"][0]["error"]

    def test_failed_rows_are_excluded_from_successful(self, client):
        payload = {"requests": [request_payload(user_id="user_999999999")] * 3}
        body = client.post("/predict/batch", json=payload).json()
        assert body["successful"] == 0
        assert body["failed"] == 3
        assert all(item["error"] for item in body["results"])

    def test_failed_row_still_echoes_its_input(self, client):
        payload = {"requests": [request_payload(user_id="user_999999999", watch_time=42.0)]}
        item = client.post("/predict/batch", json=payload).json()["results"][0]
        assert item["user_id"] == "user_999999999"
        assert item["watch_time"] == 42.0
        assert item["index"] == 0


class TestBatchLimits:
    def test_over_the_limit_is_400(self, client):
        payload = {"requests": [request_payload()] * (MAX_BATCH_SIZE + 1)}
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 400
        assert str(MAX_BATCH_SIZE) in response.json()["detail"]

    def test_empty_batch_is_400(self, client):
        response = client.post("/predict/batch", json={"requests": []})
        assert response.status_code == 400

    def test_malformed_row_is_422_not_400(self, client):
        response = client.post("/predict/batch",
                               json={"requests": [{"user_id": VALID_USER}]})
        assert response.status_code == 422

    def test_out_of_range_watch_time_is_rejected_by_the_schema(self, client):
        response = client.post("/predict/batch",
                               json={"requests": [request_payload(watch_time=-5.0)]})
        assert response.status_code == 422

    def test_missing_requests_key_is_422(self, client):
        assert client.post("/predict/batch", json={}).status_code == 422


class TestProbabilityHelpers:
    def test_regressor_output_is_clipped(self):
        class Regressor:
            def predict(self, frame):
                return np.array([-0.4, 0.5, 1.9])

        assert list(predict_probabilities(Regressor(), None)) == [0.0, 0.5, 1.0]

    def test_classifier_uses_the_positive_column(self):
        class Classifier:
            def predict_proba(self, frame):
                return np.array([[0.7, 0.3], [0.1, 0.9]])

        assert list(predict_probabilities(Classifier(), None)) == pytest.approx([0.3, 0.9])

    @pytest.mark.parametrize("probability,expected", [
        (0.5, "low"), (0.55, "low"), (0.85, "medium"), (0.05, "high"), (0.99, "high"),
    ])
    def test_confidence_measures_distance_from_a_coin_flip(self, probability, expected):
        assert confidence_label(probability) == expected
