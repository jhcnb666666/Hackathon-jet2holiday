# Student Wellness Pipeline

这是一个面向学生的日程和生活习惯建议系统。

系统在指定时间读取学生当天的日程、睡眠、运动、饮食和工作数据，选择对应的小模型得到分数，再交给 AWS Bedrock LLM 生成一条简单、具体的英文建议。

## Student-state tutoring experiment

This fork includes a research MVP for testing whether a trained student-state model improves an
LLM tutor's intervention choices. It independently prepares the public health survey extracts,
trains logistic, MLP and LSTM knowledge-tracing models, exposes a compact learned state to a
two-stage tutoring agent, and evaluates four information conditions without leaking the oracle
label into the prompt.

Start with a fast end-to-end model check:

```sh
python scripts/prepare_health_data.py
python scripts/train_student_model.py --quick
```

After configuring an LLM provider in `.env`, run the intervention ablation:

```sh
python scripts/evaluate_tutor.py --scenarios 48
```

The complete design, commands, outputs and interpretation limits are documented in
[`docs/STUDENT_STATE_EXPERIMENT.md`](docs/STUDENT_STATE_EXPERIMENT.md).

It includes a [LangGraph](https://langchain-ai.github.io/langgraph/) agent, a [FastAPI](https://fastapi.tiangolo.com/) service to serve it, a client to interact with the service, and a [Streamlit](https://streamlit.io/) app that uses the client to provide a chat interface. Data structures and settings are built with [Pydantic](https://github.com/pydantic/pydantic).

## 当前状态

本地 demo 已经可以在不调用 AWS 的情况下跑通。当前已经完成日程和生活数据结构、Physical Activity 小模型、SleepModel 适配器、CSV 读取器、指定时间触发器、总 pipeline、AWS Bedrock 建议适配器和本地 demo。

尚未完成 Eating 小模型、Workload 小模型、正式 HTTP API、后台定时服务，以及 AWS Bedrock 的真实调用验证。

## 关键文件

- `src/agents/student_wellness.py`：总 pipeline。
- `src/agents/physical_activity_model.py`：Physical Activity 小模型。
- `src/agents/sleep_model.py`：Sleep 模型适配器。
- `models/sleep_model.json`：Sleep 模型产物。
- `src/agents/aws_advice_model.py`：AWS Bedrock 英文建议适配器。
- `src/data/wellness_loader.py`：CSV 数据读取器。
- `src/schema/student_wellness.py`：统一数据结构。
- `src/agents/scheduler.py`：指定时间触发器。
- `scripts/run_wellness_demo.py`：本地端到端 demo。
- `PIPELINE.md`：pipeline 详细说明。

## 数据文件

真实数据位于 `data/real/`，包括 sleep、physical activity、eating habits、work 和 demographics。当前只接入睡眠和运动数据。GPA 以及 GPA 质量标记都会被忽略。

## 本地运行

要求 Python 3.12 以上，并安装 `uv`。先运行 `uv sync`，再设置 `PYTHONPATH=src`，最后运行 `scripts/run_wellness_demo.py`。本地 demo 不访问 AWS，不消耗 API 额度。

## AWS Bedrock

真实建议生成使用 AWS Bedrock。配置 `USE_AWS_BEDROCK=true`、`AWS_REGION` 和 `AWS_BEDROCK_MODEL_ID`，并通过系统环境变量提供临时 AWS 凭证。不要把真实凭证写入代码、README、`.env.example` 或 Git。

## 设计原则

每个小模型只负责一个生活维度，统一输出 `MetricScore`。大模型负责综合日程和分数，不替代小模型评分。建议输出统一使用英文。系统提供生活方式建议，不提供医疗诊断。
