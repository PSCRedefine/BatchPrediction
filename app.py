"""Streamlit console for batch engagement prediction.

Implements section 2 of docs/SPEC.md. Two ways in — a CSV upload for bulk work
and a manual builder for a handful of rows — converging on one results view.

The page holds no model and no data. Every prediction goes through ``call_api``,
so what you see here is exactly what an API client would get.
"""

from __future__ import annotations

import os
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from batch_prediction.batching import missing_columns, normalise, prepare, to_payload
from batch_prediction.config import MAX_BATCH_SIZE

API_BASE_URL = os.getenv("BATCH_PREDICTION_API_URL", "http://127.0.0.1:8000")
SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "sample_batch_requests.csv"

st.set_page_config(page_title="Cognitive Shorts", page_icon="📱", layout="wide")


def call_api(path: str, payload: dict[str, Any] | None = None, timeout: int = 30) -> dict[str, Any]:
    """Call the prediction service, turning transport failures into data.

    Returns a dict that either holds the parsed response or an ``error`` key.
    Raising here would leave the page half-rendered, so every failure comes back
    as a value the caller can display.
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


def results_table(results: pd.DataFrame) -> None:
    """Render the results with the probability drawn as a bar.

    A column of six-decimal floats is hard to scan. The bar makes the spread
    visible at a glance, which matters here because the model's outputs sit in
    a narrow band and repeat across rows.
    """
    columns = {
        "index": st.column_config.NumberColumn("#", width="small"),
        "user_id": st.column_config.TextColumn("User"),
        "video_id": st.column_config.TextColumn("Video"),
        "watch_time": st.column_config.NumberColumn("Watch (s)", format="%.1f"),
        "hour_of_day": st.column_config.NumberColumn("Hour", format="%d"),
        "probability": st.column_config.ProgressColumn(
            "Probability", min_value=0.0, max_value=1.0, format="%.4f"
        ),
        "confidence": st.column_config.TextColumn("Confidence"),
        "predicted_engaged": st.column_config.CheckboxColumn("Engaged?"),
        "error": st.column_config.TextColumn("Error", width="medium"),
    }
    for column in ("error", "confidence"):
        if column in results.columns:
            results = results.assign(**{column: results[column].fillna("")})
    st.dataframe(
        results,
        width='stretch',
        hide_index=True,
        column_config={k: v for k, v in columns.items() if k in results.columns},
    )


def render_results() -> None:
    """Metrics, table, download and distribution — specification 2.4."""
    if st.session_state.get("batch_error"):
        st.error(f"❌ Batch prediction failed: {st.session_state.batch_error}")
        return
    response = st.session_state.get("batch_response")
    if not response:
        return

    st.header("Results & Analytics")
    st.success("✅ Batch prediction complete!")

    results = pd.DataFrame(response["results"])
    scored = results["probability"].dropna() if "probability" in results else pd.Series(dtype=float)

    total = int(response.get("batch_size", len(results)))
    successful = int(response.get("successful", len(scored)))
    failed = total - successful
    average = float(scored.mean()) if not scored.empty else 0.0
    elapsed = float(response.get("response_time_ms",
                                 st.session_state.get("batch_round_trip_ms", 0.0)))
    threshold = float(response.get("threshold", 0.5))

    one, two, three, four = st.columns(4)
    one.metric("Total Requests", total)
    two.metric("Successful", successful, delta=f"-{failed} failed" if failed else None,
               delta_color="inverse" if failed else "normal")
    three.metric("Avg Probability", f"{average:.3f}")
    four.metric("Response Time", f"{elapsed:.0f} ms",
                help="Server-side processing time reported by the API")

    if failed:
        st.warning(
            f"{failed} of {total} rows could not be scored. Each one carries the reason in "
            "its `error` column — the rest of the batch was scored normally."
        )

    flagged = int((scored >= threshold).sum())
    five, six, seven = st.columns(3)
    five.metric("Flagged as engaged", flagged,
                help=f"Rows at or above the decision threshold of {threshold:.3f}")
    six.metric("Flag rate", f"{(flagged / successful * 100 if successful else 0):.1f}%")
    seven.metric("Throughput", f"{(total / (elapsed / 1000) if elapsed else 0):,.0f} rows/s")

    tabs = st.tabs([f"All ({total})", f"Scored ({successful})", f"Failed ({failed})"])
    with tabs[0]:
        results_table(results)
    with tabs[1]:
        results_table(results[results["probability"].notna()]
                      .drop(columns=["error"], errors="ignore"))
    with tabs[2]:
        if failed:
            results_table(results[results["probability"].isna()][
                ["index", "user_id", "video_id", "watch_time", "error"]])
        else:
            st.success("Every row was scored.")

    st.download_button(
        "📥 Download Results CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name=f"batch_predictions_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        type="primary",
    )

    if scored.empty:
        st.info("No row was scored successfully, so there is no distribution to plot.")
        return

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Prediction Probability Distribution")
        figure = px.histogram(scored.to_frame("probability"), x="probability", nbins=20)
        figure.update_traces(marker_line_width=1, marker_line_color="rgba(0,0,0,0.35)")
        figure.add_vline(x=threshold, line_dash="dash", line_color="#ff4b4b",
                         annotation_text=f"threshold {threshold:.3f}",
                         annotation_position="top right")
        figure.update_layout(xaxis_title="Predicted probability", yaxis_title="Count",
                             bargap=0.05, showlegend=False, height=340,
                             margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(figure, width='stretch')
    with right:
        st.subheader("Highest Scoring Rows")
        top = (results[results["probability"].notna()]
               .nlargest(5, "probability")[["user_id", "video_id", "watch_time", "probability"]])
        st.dataframe(
            top, width='stretch', hide_index=True,
            column_config={
                "user_id": st.column_config.TextColumn("User"),
                "video_id": st.column_config.TextColumn("Video"),
                "watch_time": st.column_config.NumberColumn("Watch (s)", format="%.1f"),
                "probability": st.column_config.ProgressColumn(
                    "Probability", min_value=0.0, max_value=1.0, format="%.4f"),
            },
        )
        counts = (results["confidence"].dropna().value_counts()
                  .reindex(["low", "medium", "high"]).fillna(0).astype(int))
        st.caption(
            f"Confidence: {counts['high']} high · {counts['medium']} medium · {counts['low']} low. "
            "Confidence is distance from a coin flip, not a claim about correctness."
        )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.session_state.setdefault("manual_batch", [])
st.session_state.setdefault("batch_response", None)
st.session_state.setdefault("batch_error", None)
st.session_state.setdefault("sample_frame", None)

# A shareable demo link: /?demo=1 loads the bundled sample and scores it, so the
# results view can be reached without a file picker. Used for the README
# screenshots, which is why they are reproducible rather than hand-staged.
if st.query_params.get("demo") == "1" and st.session_state.batch_response is None:
    if SAMPLE_PATH.exists():
        st.session_state.sample_frame = pd.read_csv(SAMPLE_PATH)

health = call_api("health")
offline = "error" in health

with st.sidebar:
    st.header("⚙️ Controls")
    if offline:
        st.error(f"❌ API Status: {health['error']}")
    else:
        st.success(f"✅ API Status: {health.get('status', 'unknown').title()}")
        st.metric("Uptime", f"{health.get('uptime_seconds', 0):.0f}s")
    st.divider()
    page = st.selectbox("Select Page", ["Batch Prediction", "Model Info"])
    st.divider()
    if not offline:
        st.caption("**Model**")
        st.caption(f"`{health.get('model_name', 'unknown')}`")
        st.caption(f"Version {health.get('version', '?')} · max batch {health.get('max_batch_size', MAX_BATCH_SIZE)}")

if page == "Model Info":
    st.title("ℹ️ Model Info")
    info = call_api("model/info")
    if "error" in info:
        st.error(f"❌ Could not load model info: {info['error']}")
    else:
        one, two, three = st.columns(3)
        one.metric("Model", info.get("model_name", "unknown"))
        two.metric("Decision threshold", f"{float(info.get('threshold', 0.5)):.3f}")
        three.metric("Max batch size", info.get("max_batch_size", MAX_BATCH_SIZE))
        st.write("**Features**: " + ", ".join(f"`{f}`" for f in info.get("features", [])))
        st.caption("Why only two features, and why this model — see docs/MODEL_SELECTION.md.")
        st.json(info.get("metadata", {}), expanded=False)
    st.stop()

st.title("📱 Cognitive Shorts Recommendation System")
st.caption("Production-ready ML system for predicting user engagement")
st.header("📊 Batch Prediction")

if offline:
    st.error(f"❌ {health['error']} — start it with `uvicorn batch_prediction.api:app`")

# --- CSV upload mode -------------------------------------------------------
uploaded = st.file_uploader(
    "Upload CSV file with user interactions",
    type=["csv"],
    help="CSV should have columns: user_id, video_id, watch_time",
)

if SAMPLE_PATH.exists() and uploaded is None:
    if st.button("Use the bundled sample (20 rows, 2 of them invalid)"):
        st.session_state.sample_frame = pd.read_csv(SAMPLE_PATH)
if uploaded is not None:
    st.session_state.sample_frame = None

csv_frame: pd.DataFrame | None = st.session_state.sample_frame
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
    else:
        st.success(f"✅ Loaded {len(csv_frame)} rows")
        if len(csv_frame) > MAX_BATCH_SIZE:
            st.warning(f"File has {len(csv_frame)} rows. "
                       f"Only first {MAX_BATCH_SIZE} will be processed.")
        st.subheader("Data Preview")
        st.dataframe(csv_frame.head(5), width='stretch')

        prepared = prepare(csv_frame)
        unreadable = int(prepared["watch_time"].isna().sum())
        if unreadable:
            st.warning(f"{unreadable} row(s) have a watch_time that is not a number "
                       "and are sent as 0.")
        autorun = (st.query_params.get("demo") == "1"
                   and st.session_state.batch_response is None and not offline)
        if st.button("🚀 Run Batch Prediction", type="primary", disabled=offline) or autorun:
            with st.spinner(f"Scoring {len(prepared)} rows..."):
                run_batch(prepared)

st.divider()

# --- Manual batch input mode ----------------------------------------------
st.header("Manual Batch Input")
st.info("💡 Upload a CSV file above for bulk processing, or add individual requests below")

with st.form("add_request", clear_on_submit=True):
    one, two, three = st.columns([2, 2, 1])
    manual_user = one.text_input("User ID", placeholder="user_000001")
    manual_video = two.text_input("Video ID", placeholder="video_0000001")
    manual_watch = three.number_input("Watch Time", min_value=0.0, max_value=3600.0,
                                      value=45.0, step=1.0)
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
    st.subheader(f"Current Batch ({len(batch)} requests)")
    st.dataframe(pd.DataFrame(batch), width='stretch', hide_index=True)
    left, right = st.columns([1, 1])
    if left.button("🗑️ Clear All"):
        st.session_state.manual_batch = []
        st.rerun()
    if right.button("🚀 Process Batch", type="primary", disabled=offline):
        with st.spinner(f"Scoring {len(batch)} rows..."):
            run_batch(normalise(pd.DataFrame(batch)))

st.divider()
render_results()
