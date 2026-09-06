# 当前 Pipeline

## 一句话说明

系统在指定时间读取学生当天日程和生活数据，选择需要的 specialist models，得到各项分数，最后由 AWS Bedrock LLM 生成一条英文建议。

## 数据流

```text
StudentSchedule + WellnessSignals
                |
                v
         DailyTrigger
                |
                v
          ModelSelector
                |
                v
  Specialist Models (one or more)
                |
                v
          MetricScore list
                |
                v
       AWS Bedrock AdviceModel
                |
                v
          WellnessReport
```

## 每个组件负责什么

### 1. `StudentSchedule`

描述当天安排，例如课程、学习、吃饭、运动和睡眠时间。

### 2. `WellnessSignals`

描述学生的生活数据。目前预留以下维度：

- sleep
- physical activity
- eating
- workload
- demographics as context only

GPA 不在这个 schema 中，也不会进入任何模型。

### 3. `ModelSelector`

根据日程和触发时间返回模型名称，例如：

```python
["sleep", "physical_activity"]
```

它只负责选择，不负责评分。

### 4. `SpecialistModel`

每个小模型都实现：

```python
async def score(schedule, signals) -> MetricScore:
    ...
```

返回统一格式：指标名称、0 到 100 分、等级和证据。

### 5. `AdviceModel`

接收日程和所有小模型的分数，生成一条英文建议。它是唯一应该调用 AWS Bedrock 的综合建议组件。

## 当前代码状态

- `StudentWellnessPipeline`：总编排器，已完成。
- `DailyTrigger`：指定时间触发，已与 signals 接口对齐。
- `PhysicalActivityModel`：本地 baseline，可直接作为 specialist 使用，不调用 API。
- `SleepModel`：仓库有训练产物和测试，但当前本地缺少 handoff 文档所引用的运行时代码文件，需要补齐后再注册。
- Eating、Workload：只有接口位置，尚未实现具体模型。
- AWS AdviceModel：接口已定义，具体 Bedrock 实现需要在应用入口注入。

## 组装方式

```python
pipeline = StudentWellnessPipeline(
    models={
        "physical_activity": physical_activity_model,
        "sleep": sleep_model,
    },
    selector=selector,
    advice=aws_advice_model,
)

report = await pipeline.run(schedule, signals, triggered_at=time(20, 0))
```

## 设计原则

- specialist model 只分析自己的维度。
- specialist model 不调用其他 specialist model。
- 大模型不直接替代 specialist scoring。
- 所有模型使用统一的 `MetricScore`。
- 不调用外部模型的测试不会消耗 AWS API 额度。
- 健康建议不是医疗诊断。
