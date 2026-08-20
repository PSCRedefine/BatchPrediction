"""Streamlit console for batch engagement prediction.

Implements section 2 of docs/SPEC.md. Two ways in — a CSV upload for bulk work
and a manual builder for a handful of rows — converging on one results view.

The page holds no model. Every prediction goes through ``call_api``, so what
you see here is exactly what an API client would get.
"""

from __future__ import annotations

import os
import time
from io import StringIO
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from batch_prediction.batching import missing_columns, normalise, prepare, to_payload
from batch_prediction.config import MAX_BATCH_SIZE

API_BASE_URL = os.getenv("BATCH_PREDICTION_API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Cognitive Shorts", page_icon="📱", layout="wide")


def call_api(path: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    """Call the prediction service, turning transport failures into data.

    Returns a dict that either holds the parsed response or an ``error`` key.
    Raising here would leave the page half-rendered, so every failure comes
    back as a value the caller can display.
    """
    url = f"{API_BASE_URL}/{path.lstrip('/')}"
    try:
        response = (
            requests.post(url, json=payload, timeout=timeout)
            if payload is not None
            else requests.get(url, timeout=timeout)
        )
    except requests.exceptions.ConnectionError:
        return {"error": "API server is offline"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except requests.exceptions.RequestException as exc:
        return {"error": str(exc)}

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return {"error": detail if isinstance(detail, str) else str(detail)}
    try:
        return response.json()
    except ValueError:
        return {"error": "the server returned a response that is not JSON"}


def run_batch(frame: pd.DataFrame) -> None:
    """Submit a batch and store the outcome for the results section."""
    started = time.perf_counter()
    response = call_api("predict/batch", to_payload(frame))
    elapsed = (time.perf_counter() - started) * 1000
    if "error" in response:
        st.session_state.batch_error = response["error"]
        st.session_state.batch_response = None
        return
    st.session_state.batch_error = None
    st.session_state.batch_response = response
    st.session_state.batch_round_trip_ms = elapsed


def render_results() -> None:
    """Metrics, table, download and distribution — specification 2.4."""
    response = st.session_state.get("batch_response")
    if st.session_state.get("batch_error"):
        st.error(f"❌ Batch prediction failed: {st.session_state.batch_error}")
        return
    if not response:
        return

    st.success("✅ Batch prediction complete!")
    results = pd.DataFrame(response["results"])
    scored = results["probability"].dropna() if "probability" in results else pd.Series(dtype=float)

    total = int(response.get("batch_size", len(results)))
    successful = int(response.get("successful", len(scored)))
    average = float(scored.mean()) if not scored.empty else 0.0
    elapsed = float(response.get("response_time_ms", st.session_state.get("batch_round_trip_ms", 0)))

    one, two, three, four = st.columns(4)
    one.metric("Total Requests", total)
    two.metric("Successful", successful)
    three.metric("Avg Probability", f"{average:.3f}")
    four.metric("Response Time", f"{elapsed:.0f} ms")

    if successful < total:
        st.warning(f"{total - successful} of {total} rows could not be scored. "
                   "The reason for each is in the `error` column below.")

    st.subheader("Results")
    st.dataframe(results, use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Download Results CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"batch_results_{int(time.time())}.csv",
        mime="text/csv",
    )

    if not scored.empty:
        st.subheader("Prediction Probability Distribution")
        figure = px.histogram(
            scored.to_frame("probability"), x="probability", nbins=20,
            labels={"probability": "Predicted probability", "count": "Count"},
        )
        figure.update_layout(
            xaxis_title="Predicted probability", yaxis_title="Count",
            bargap=0.05, showlegend=False, height=360,
        )
        threshold = response.get("threshold")
        if threshold is not None:
            figure.add_vline(x=float(threshold), line_dash="dash",
                             annotation_text=f"threshold {float(threshold):.3f}")
        st.plotly_chart(figure, use_container_width=True)
    else:
        st.info("No row was scored successfully, so there is no distribution to plot.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.session_state.setdefault("manual_batch", [])
st.session_state.setdefault("batch_response", None)
st.session_state.setdefault("batch_error", None)

health = call_api("health")

with st.sidebar:
    st.header("Cognitive Shorts")
    page = st.selectbox("Select Page", ["Batch Prediction", "Model Info"])
    st.divider()
    if "error" in health:
        st.error(f"API: {health['error']}")
    else:
        st.success(f"API: {health.get('status', 'unknown')}")
        st.caption(f"Max batch size: {health.get('max_batch_size', MAX_BATCH_SIZE)}")

if page == "Model Info":
    st.title("ℹ️ Model Info")
    info = call_api("model/info")
    if "error" in info:
        st.error(f"❌ Could not load model info: {info['error']}")
    else:
        left, right = st.columns(2)
        left.metric("Model", info.get("model_name", "unknown"))
        right.metric("Decision threshold", f"{float(info.get('threshold', 0.5)):.3f}")
        st.write("**Features**:", ", ".join(info.get("features", [])))
        st.json(info.get("metadata", {}), expanded=False)
    st.stop()

st.title("📊 Batch Prediction")
st.caption("Score up to 100 user-video interactions in one call.")

# --- CSV upload mode -------------------------------------------------------
st.subheader("Upload a CSV")
uploaded = st.file_uploader(
    "Upload CSV file with user interactions",
    type=["csv"],
    help="CSV should have columns: user_id, video_id, watch_time",
)

csv_frame: pd.DataFrame | None = None
if uploaded is not None:
    try:
        csv_frame = pd.read_csv(StringIO(uploaded.getvalue().decode("utf-8")))
    except Exception as exc:  # a malformed upload is the user's problem to see
        st.error(f"❌ Could not read the CSV: {exc}")
        csv_frame = None

    if csv_frame is not None:
        missing = missing_columns(csv_frame)
        if missing:
            st.error(f"Missing required columns: {missing}")
            csv_frame = None
        else:
            st.success(f"✅ Loaded {len(csv_frame)} rows")
            if len(csv_frame) > MAX_BATCH_SIZE:
                st.warning(
                    f"File has {len(csv_frame)} rows. "
                    f"Only first {MAX_BATCH_SIZE} will be processed."
                )
            st.write("**Data Preview**")
            st.dataframe(csv_frame.head(5), use_container_width=True, hide_index=True)

            prepared = prepare(csv_frame)
            unreadable = int(prepared["watch_time"].isna().sum())
            if unreadable:
                st.warning(f"{unreadable} row(s) have a watch_time that is not a number "
                           "and are sent as 0.")

            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner(f"Scoring {len(prepared)} rows..."):
                    run_batch(prepared)

st.divider()

# --- Manual batch input mode ----------------------------------------------
st.subheader("Or build a batch by hand")
st.info("💡 Upload a CSV file above for bulk processing, or add individual requests below")

with st.form("add_request", clear_on_submit=True):
    one, two, three = st.columns([2, 2, 1])
    manual_user = one.text_input("User ID", placeholder="user_000001")
    manual_video = two.text_input("Video ID", placeholder="video_0000001")
    manual_watch = three.number_input("Watch Time", min_value=0.0, max_value=3600.0,
                                      value=30.0, step=1.0)
    if st.form_submit_button("➕ Add Request"):
        if not manual_user.strip() or not manual_video.strip():
            st.error("User ID and Video ID are both required.")
        elif len(st.session_state.manual_batch) >= MAX_BATCH_SIZE:
            st.error(f"The batch already holds {MAX_BATCH_SIZE} requests.")
        else:
            st.session_state.manual_batch.append({
                "user_id": manual_user.strip(),
                "video_id": manual_video.strip(),
                "watch_time": float(manual_watch),
            })

batch = st.session_state.manual_batch
if batch:
    st.write(f"**Current Batch ({len(batch)} requests)**")
    st.dataframe(pd.DataFrame(batch), use_container_width=True, hide_index=True)
    left, right = st.columns([1, 1])
    if left.button("🗑️ Clear All"):
        st.session_state.manual_batch = []
        st.rerun()
    if right.button("🚀 Process Batch", type="primary"):
        with st.spinner(f"Scoring {len(batch)} rows..."):
            run_batch(normalise(pd.DataFrame(batch)))

st.divider()
render_results()
