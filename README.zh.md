# 前端说明（Wellbeing App）

> NTU / NUS 学生 wellbeing 小应用。学生每天记几个数字——睡眠、喝水、运动——
> 应用把它们和课表结合，给出 wellbeing 建议。
>
> 仓库基于 [agent-service-toolkit](https://github.com/JoshuaC215/agent-service-toolkit)：
> LangGraph agents 跑在 FastAPI 服务后面，前端是 Streamlit。
>
> 英文版 toolkit 说明见 [`README.md`](README.md)。本文件只讲**前端部分**。

---

## 分工

**前端（本文件负责的范围）**——下面列的这些文件，全是确定性逻辑：课表、日历查询、
数据录入、图表。唯一的模型调用是读课表截图的一次 vision call，以及建议卡片对
agent 服务的调用。

| 文件 | 作用 |
|---|---|
| `src/app.py` | 入口。`streamlit run src/app.py`，`st.navigation` 下三个页面 |
| `src/academic_calendar.py` | NTU / NUS 的 AY2026-27 学期日期硬编码。`academic_context(date, school)` 返回教学周和阶段；`parse_weeks("Wk9,12")` 处理单课的周次限制 |
| `src/timetable_input.py` | 课表录入。Gemini 读截图成课程块；可视化网格用于核对。存 `data/timetable.csv` |
| `src/daily_log.py` | 每日录入。上课时长来自课表，所以只问睡眠、喝水、运动。喝水是水瓶 SVG（改成了 CSS），250 ml 一下。存 `data/daily_log.csv`。顶部是建议卡片区 |
| `src/health_dashboard.py` | 指标卡、10pm–10pm 的一天节奏条、趋势折线图。读 `data/daily_log.csv`，缺失时用 placeholder（只补 sleep/water/movement，class 数据只从真实课表来） |
| `src/suggestions.py` | **前端 ↔ agent 服务的接口层 + 契约**（详见下方） |

---

## 怎么跑

用仓库自带的虚拟环境：

```bash
cd Hackathon-jet2holiday
.venv\Scripts\streamlit.exe run src\app.py
```

默认开在 `http://localhost:8501`。想手动开浏览器就加 `--server.headless true`。

> 同时只跑一个 streamlit。改了代码没生效，多半是有旧进程还占着端口，或者你看的
> 不是最新那个窗口。

---

## 三个页面

| 页面 | 文件 | 内容 |
|---|---|---|
| **Today** | `daily_log.py` | 顶部：定点建议卡片。然后：当天课表（含课程时间轴）、昨晚睡眠、喝水、运动（时间段选择） |
| **Timetable** | `timetable_input.py` | 一次性录入。截图导入 → 可编辑网格 → Save。网格和周统计只显示**本教学周**在上的课（标题 "This week" / "Week N total"），其余的仍在 "Edit rows" 里 |
| **Trends** | `health_dashboard.py` | Today 快照指标卡 → "Your two weeks"（节奏条）→ "Your two weeks trend"（折线图） |

---

## 数据流

```
timetable_input.py  ->  data/timetable.csv   ->  daily_log.py / health_dashboard.py
daily_log.py        ->  data/daily_log.csv   ->  health_dashboard.py / 后端 agents
daily_log.py        ->  data/suggestions.json（定点建议缓存）
```

这三个文件是前端和后端之间的契约。每个模块能单独跑，后端也能直接读同样的文件。

**`data/` 目录的分工**（队友 `data-preparation` PR 已合入）：

| 路径 | 谁的 | 进版本库？ |
|---|---|---|
| `data/real/`、`data/simulated/`、`data/docs/` | 队友整理的公开数据 + 模拟测试数据 + 数据字典 | 是 |
| `data/timetable.csv`、`data/daily_log.csv`、`data/suggestions.json` | 前端运行时写的，每人各自的 | 否（`.gitignore` 里按文件名单独忽略） |

### `data/daily_log.csv` 列

```
date, sleep_hours, sleep_start, exercise_minutes, exercise_blocks,
water_ml, class_hours, class_start, class_end
```

- `exercise_blocks`：`HH:MM-HH:MM;HH:MM-HH:MM`（分号分隔的运动时间段）
- `exercise_minutes`：各段时长之和（重叠只算一次）

### `data/timetable.csv` 列

```
day, start, end, name, weeks, school
```

- `weeks`：`""`（每周）/ `"9,12"` / `"2-13"`

---

## 建议卡片 & 后端接口契约


```python
SUGGESTIONS_KEY = "suggestions"          # ChatMessage.custom_data 上的 key
CONTEXT_KEY     = "daily_context"        # agent_config 上的 key
SCHEMA_VERSION  = 1
DEFAULT_AGENT   = "langgraph-supervisor-agent"
TONES           = ("positive", "nudge", "watch")
```

### 请求：前端 → agent

`agent_config` 传结构化数字，`message` 传一段英文 brief（含"不诊断/不给数值目标"约束）：

```python
agent_config = {"daily_context": {
    "school": "NTU", "date": "2026-09-04", "teaching_week": 4, "phase": "teaching",
    "sleep_hours": 6.2, "sleep_start": "01:10",
    "class_hours": 4.0, "class_start": "09:30", "class_end": "16:20",
    "exercise_minutes": 0, "exercise_blocks": [["18:00", "18:30"]],
    "water_ml": 500,
}}
```

### 响应：agent → 前端

放到最终消息（或专门的 `type="custom"` 消息）的 `custom_data` 上：

```json
{"suggestions": {
  "schema_version": 1,
  "generated_at": "2026-09-04T14:12:00Z",
  "summary": "一句话总览（可选）",
  "support": {"label": "NTU UCS", "url": "https://..."},
  "cards": [{
    "headline": "标题",
    "reasoning": "为什么",
    "action": "今天具体做一件什么事",
    "metrics": ["sleep_hours", "class_start"],
    "tone": "watch",
    "source_agent": "sleep-coach",
    "id": "sleep-debt"
  }]
}}
```

`support` 只在检测到 distress 时给，前端会置顶显示求助链接（对应"应用不自己处理
危机，而是引导到 NTU UCS / NUS UHC"的要求）。

### 子 agent 进度

`stream_suggestions()` 会把任何形似 `TaskData` 的 `type="custom"` 消息解析成
`ProgressEvent`，前端用 `st.status` 实时显示（让多 agent 架构在 demo 里看得见）。

### 服务地址

`suggestions.service_url()`：优先 `AGENT_URL` 环境变量，否则
`http://{HOST=0.0.0.0}:{PORT=8080}`。

---

## 定点建议机制（check-in）

每天 5 个固定时段：`08:00 / 12:00 / 16:00 / 20:00 / 22:00`
（`daily_log.CHECK_IN_SLOTS`）。

- **当天、到点的时段还没生成** → 自动生成一次（进度条流式），结果存 session +
  `data/suggestions.json`
- 已生成的时段 → 顶部时段选择器，可回看，每个有自己的 Refresh
- 8am 前 → "First check-in at 8am"
- 其他日期 → 列出 5 个时段 + Generate 按钮
- check-in 区块显示在页面顶部，但代码里**最后才渲染**（`st.container()` 占位），
  所以 20–40s 的生成不卡页面其他部分

**限制**：前端只能在"有人打开 app 且已过某时段"时补上该时段。**app 关着时不会
自己推**——真正的无人值守定时推送需要一个调度器（后端 cron，或 toolkit 自带定时），
那是后端的活。

---

## 本地开发：预览建议卡片

后端还没起时，点生成会走"连不上"分支（不编假数据）。想预览卡片效果，用一个假后端：

```bash
# 终端 1：假后端（一次性脚本，不在仓库里，自己写一个返回上面契约形状的 /stream 即可）
python stub_agent.py     # 监听 127.0.0.1:8899

# 终端 2：streamlit 指向它
$env:AGENT_URL="http://127.0.0.1:8899"
.venv\Scripts\streamlit.exe run src\app.py
```

`src/suggestions.py` 直接跑（`python src/suggestions.py`）会打印 context、prompt，
并用 `on_error_sample=True` 展示离线兜底卡片。

---


## 健康内容红线

不诊断，不给食物或运动的数值目标。Dashboard 上的范围标"usual range"，不是目标。
用户表达痛苦时，界面应引导到真实支持渠道（NTU UCS、NUS UHC），而不是让模型处理。

---

## 本轮改动清单

- **修复 Trends 页崩溃**（`load_data` / `_placeholder_data` 半截重命名的 NameError）
- **Today**：`From your timetable` 下加课程时间轴（圆点 + 线段 + 课程编号）
- **Trends**：
  - 没有 daily_log 记录的天，class 数据直接从 `timetable.csv` 取（不再编）
  - 节奏条按每节课分段画（原来是一整条）
  - 加绿色 movement 条 + 图例
  - 节奏条锚点从 6pm 改 10pm
  - 页面重排：Today 置顶 → "Your two weeks" 分区 → "Your two weeks trend" 折线图
- **daily_log**：
  - Movement 输入改成时间段双滑块 + 多段 + 重叠合并，新增 `exercise_blocks` 列
  - 水瓶可视化从 SVG 改 CSS（`st.html` 会删 SVG）
  - 日期选择器上限设为今天
- **timetable_input**：
  - 删掉编造的 seed 课表，改成重新载入已保存的
  - 网格和周统计只算本教学周（"This week" / "Week N total"），其余在 Edit rows 可见
- **新增 `src/suggestions.py`**：前端 ↔ agent 服务接口层 + 契约
- **Today**：定点建议卡片（5 个固定时段 + session/磁盘缓存 + 子 agent 进度条）

---

## 待办

- Today 顶部 suggestion cards 的 UI 打磨（错误态、加载态的视觉）
- 后端按契约实现 `custom_data.suggestions` 后，去掉/替换 stub
- 无人值守的定点推送（需要调度器，后端）
- 节奏条跨 22:00 边界的睡眠段目前会被裁掉（边缘情况，暂不处理）
