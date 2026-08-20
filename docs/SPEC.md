# Batch Prediction

## 1. 概述
### 1.1 页面定位
Batch Prediction（批量预测）是 Cognitive Shorts 系统前端 UI（基于 Streamlit）中的一个核心功能模块。该页面允许用户一次性提交多条“用户-视频-观看行为”数据，调用后端 ML 模型接口，快速获取这批数据的预测互动概率

## 2. 页面需求 (UI)
页面整体划分为两大功能区：**CSV 文件上传模式** 和 **手动批量录入模式**

### 2.1 页面标题与导航
- **标题**：`📊 Batch Prediction`
- **入口**：左侧 Sidebar 导航栏选择 "Batch Prediction"

### 2.2 CSV 文件上传模式 (File Upload Mode)
适用于数据量较大（100条以内）的预测场景

**UI 元素：**
1. **文件上传组件**：
   - 提示文案："Upload CSV file with user interactions"
   - 帮助说明："CSV should have columns: user_id, video_id, watch_time"
   - 限制格式：仅支持 `.csv`
2. **状态提示**：成功加载后显示 "✅ Loaded {N} rows"
3. **数据预览区 (Data Preview)**：展示解析后的 CSV 前 5 行数据表格
4. **警告提示 (Warning)**：如果上传数据超过 100 行，显示黄条警告："File has {N} rows. Only first 100 will be processed."
5. **执行按钮**：主色调按钮 "🚀 Run Batch Prediction"

### 2.3 手动批量录入模式 (Manual Batch Input Mode)
适用于少量、临时性的数据预测场景

**UI 元素：**
1. **说明信息**：蓝条提示 "💡 Upload a CSV file above for bulk processing, or add individual requests below"
2. **表单录入区 (Add Request)**：
   - `User ID` 输入框
   - `Video ID` 输入框
   - `Watch Time` 数字输入框
   - "➕ Add Request" 提交按钮
3. **当前批次列表区 (Current Batch)**：
   - 标题："Current Batch ({N} requests)"
   - 数据表格：展示已添加的所有请求明细。
   - "🗑️ Clear All" 按钮：清空当前列表。
   - "🚀 Process Batch" 主色调按钮：提交当前列表进行预测

### 2.4 预测结果展示区 (Results & Analytics)
点击预测并收到后端响应后展示。

**UI 元素：**
1. **成功提示**："✅ Batch prediction/complete!"
2. **核心指标看板 (Metrics)**：
   - `Total Requests`：总请求数
   - `Successful`：成功预测数
   - `Avg Probability`：平均预测概率（保留3位小数）
   - `Response Time`：总响应时间（毫秒）
3. **结果表格 (Results Dataframe)**：展示包含预测结果或错误信息的完整数据
4. **导出按钮**："📥 Download Results CSV"，允许用户下载完整的预测结果
5. **可视化图表 (Visualization)**：
   - 图表类型：柱状图/直方图 (Histogram)
   - 标题："Prediction Probability Distribution"
   - X轴：预测概率分布
   - Y轴：频次 (Count)

---

## 3. 功能需求 (Functional Requirements)

### 3.1 输入与数据校验规则
**CSV 模式：**
1. **必填列校验**：上传的 CSV 必须包含 `user_id`, `video_id`, `watch_time` 三列。缺少任何一列则中止流程并显示红条错误 "Missing required columns: [...]"。
2. **选填列**：支持可选的 `hour_of_day` 列。
3. **数量限制**：为保护后端资源，单次批量预测强制截断前 100 条数据（`df.head(100)`）。
4. **数据类型转换**：
   - `user_id`, `video_id` 转换为字符串 (`str`)
   - `watch_time` 转换为浮点数 (`float`)
   - `hour_of_day` (如存在) 转换为整数 (`int`)

**手动模式：**
1. **空值校验**：点击添加时，`user_id` 和 `video_id` 不能为空
2. **状态管理**：使用 Streamlit `session_state` 持久化保存当前用户添加的请求列表

### 3.2 接口通信规范
前端统一通过封装的 `call_api("predict/batch", payload)` 函数与后端通信

**请求 (Request):**
- **Method**: POST
- **Endpoint**: `/predict/batch`
- **Payload 格式**:
  ```json
  {
    "requests": [
      {
        "user_id": "user_123",
        "video_id": "video_456",
        "watch_time": 45.0,
        "hour_of_day": 14  // 可选
      }
    ]
  }
  ```

**响应处理 (Response Handling):**
- **正常响应 (HTTP 200)**：解析返回的 JSON，提取 `results` 数组、`batch_size` 和 `response_time_ms` 进行页面渲染。
- **业务异常 (HTTP 4xx/5xx)**：捕获异常信息，展示红条错误 "❌ Batch prediction failed: {error}"。
- **网络异常**：处理连接拒绝或超时，展示 "API server is offline" 或 "Request timeout"。

### 3.3 结果数据处理
1. **成功预测的数据**：后端返回的 `results` 中包含 `probability` (概率值) 和 `confidence` (置信度)，前端需将其合并到原始请求数据中展示。
2. **失败预测的数据**（部分失败）：后端采用容错机制，如果某条数据异常（如格式不符），不中断整个批次，该条结果会返回 `error` 字段。前端需在表格中如实展示该 `error` 信息，且在统计 "Successful" 指标时将其排除。

---

## 4. 后端支持需求 (API Requirements)
1. **接口定义**：提供 `/predict/batch` 路由，接收包含 `PredictionRequest` 列表的 `BatchPredictionRequest` 对象。
2. **并发限制**：在模型层面上限制 `requests` 数组 `max_items=100`，超限直接返回 400 Bad Request
4. **容错机制**：遍历请求列表时，需对**单条数据**进行 `try-except` 包裹。单条数据特征工程或预测失败不应导致整个批次崩溃，失败的条目应返回错误原因字典
5. **模型调用**：
   - 支持概率预测 (`predict_proba`) 和回归预测 (兼容 LightGBM)
   - 概率值强制裁剪 (`np.clip`) 到 `[0.0, 1.0]` 范围内。
6. **响应组装**：返回包含所有处理结果、总请求量和后端处理耗时的统一响应结构

