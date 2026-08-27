# Batch Prediction

## 1. Overview
### 1.1 Page Purpose
Batch Prediction is a core functional module in the front-end UI (based on Streamlit) of the Cognitive Shorts system. This page allows the user to submit multiple "user-video-watch behaviour" records in one go, call the back-end ML model interface, and quickly obtain the predicted interaction probabilities for this batch of data

## 2. Page Requirements (UI)
The page is divided into two main functional areas: **CSV File Upload Mode** and **Manual Batch Input Mode**

### 2.1 Page Title and Navigation
- **Title**: `📊 Batch Prediction`
- **Entry point**: select "Batch Prediction" in the left-hand Sidebar navigation bar

### 2.2 CSV File Upload Mode
Suitable for prediction scenarios with a relatively large volume of data (up to 100 rows)

**UI elements:**
1. **File upload component**:
   - Prompt text: "Upload CSV file with user interactions"
   - Help text: "CSV should have columns: user_id, video_id, watch_time"
   - Format restriction: only `.csv` is supported
2. **Status message**: after a successful load, display "✅ Loaded {N} rows"
3. **Data Preview**: display a table of the first 5 rows of the parsed CSV
4. **Warning**: if the uploaded data exceeds 100 rows, display a yellow warning bar: "File has {N} rows. Only first 100 will be processed."
5. **Action button**: primary-colour button "🚀 Run Batch Prediction"

### 2.3 Manual Batch Input Mode
Suitable for small, ad hoc data prediction scenarios

**UI elements:**
1. **Information message**: blue information bar "💡 Upload a CSV file above for bulk processing, or add individual requests below"
2. **Form input area (Add Request)**:
   - `User ID` input field
   - `Video ID` input field
   - `Watch Time` numeric input field
   - "➕ Add Request" submit button
3. **Current batch list area (Current Batch)**:
   - Title: "Current Batch ({N} requests)"
   - Data table: display the details of all requests that have been added.
   - "🗑️ Clear All" button: clear the current list.
   - "🚀 Process Batch" primary-colour button: submit the current list for prediction

### 2.4 Prediction Results Display Area (Results & Analytics)
Displayed after the prediction is triggered and a response is received from the back end.

**UI elements:**
1. **Success message**: "✅ Batch prediction/complete!"
2. **Key metrics panel (Metrics)**:
   - `Total Requests`: total number of requests
   - `Successful`: number of successful predictions
   - `Avg Probability`: average predicted probability (to 3 decimal places)
   - `Response Time`: total response time (milliseconds)
3. **Results table (Results Dataframe)**: display the complete data including prediction results or error messages
4. **Export button**: "📥 Download Results CSV", allowing the user to download the complete prediction results
5. **Visualisation chart (Visualization)**:
   - Chart type: bar chart / histogram (Histogram)
   - Title: "Prediction Probability Distribution"
   - X axis: predicted probability distribution
   - Y axis: frequency (Count)

---

## 3. Functional Requirements

### 3.1 Input and Data Validation Rules
**CSV mode:**
1. **Required column validation**: the uploaded CSV must contain the three columns `user_id`, `video_id` and `watch_time`. If any one of them is missing, the process must be aborted and a red error bar displayed: "Missing required columns: [...]".
2. **Optional columns**: an optional `hour_of_day` column is supported.
3. **Quantity limit**: to protect back-end resources, a single batch prediction is forcibly truncated to the first 100 rows (`df.head(100)`).
4. **Data type conversion**:
   - `user_id`, `video_id` converted to string (`str`)
   - `watch_time` converted to float (`float`)
   - `hour_of_day` (if present) converted to integer (`int`)

**Manual mode:**
1. **Empty value validation**: when Add is clicked, `user_id` and `video_id` must not be empty
2. **State management**: use the Streamlit `session_state` to persist the list of requests added by the current user

### 3.2 Interface Communication Specification
The front end communicates with the back end uniformly through the wrapped `call_api("predict/batch", payload)` function

**Request:**
- **Method**: POST
- **Endpoint**: `/predict/batch`
- **Payload format**:
  ```json
  {
    "requests": [
      {
        "user_id": "user_123",
        "video_id": "video_456",
        "watch_time": 45.0,
        "hour_of_day": 14  // optional
      }
    ]
  }
  ```

**Response Handling:**
- **Normal response (HTTP 200)**: parse the returned JSON and extract the `results` array, `batch_size` and `response_time_ms` to render the page.
- **Business exception (HTTP 4xx/5xx)**: catch the exception information and display a red error bar: "❌ Batch prediction failed: {error}".
- **Network exception**: handle connection refused or timeout, and display "API server is offline" or "Request timeout".

### 3.3 Result Data Processing
1. **Successfully predicted data**: the `results` returned by the back end contain `probability` (probability value) and `confidence` (confidence level); the front end must merge these into the original request data for display.
2. **Failed prediction data** (partial failure): the back end uses a fault-tolerant mechanism. If a single record is invalid (for example, malformed), the whole batch is not interrupted and that result returns an `error` field. The front end must display that `error` information faithfully in the table, and must exclude it when calculating the "Successful" metric.

---

## 4. Back-end Support Requirements (API Requirements)
1. **Interface definition**: provide a `/predict/batch` route that accepts a `BatchPredictionRequest` object containing a list of `PredictionRequest`.
2. **Concurrency limit**: limit the `requests` array to `max_items=100` at the model level; a request exceeding the limit must return 400 Bad Request directly
4. **Fault tolerance**: when iterating over the request list, each **individual record** must be wrapped in `try-except`. A feature engineering or prediction failure on a single record should not cause the whole batch to crash; a failed entry should return a dictionary giving the reason for the error
5. **Model invocation**:
   - Supports probability prediction (`predict_proba`) and regression prediction (compatible with LightGBM)
   - Probability values are forcibly clipped (`np.clip`) to the range `[0.0, 1.0]`.
6. **Response assembly**: return a unified response structure containing all processing results, the total number of requests and the back-end processing time

