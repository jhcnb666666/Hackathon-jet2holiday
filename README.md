# 学生健康日程建议系统

这是一个给学生生成日常健康建议的项目。

系统会在指定时间读取学生当天的日程，以及睡眠、运动、饮食、工作等生活数据，然后完成三件事：

1. 判断需要调用哪些生活维度的小模型。
2. 让小模型分别给对应维度打分。
3. 把日程和这些分数交给大语言模型，生成一条简单、具体的英文建议。

## 当前进度

目前已经搭好通用框架，并实现了 Physical Activity 小模型。

Physical Activity 模型暂时是本地 Python 代码，不会调用 LLM，也不会消耗 API 额度。它只看：

- 每周运动了几天
- 每周力量训练几天
- 运动分钟数

它会返回 0 到 100 的分数、英文等级和英文证据。例如：

```json
{
  "metric": "physical_activity",
  "score": 62.5,
  "level": "attention",
  "evidence": [
    "2 physically active days per week",
    "1 strength-training day per week"
  ]
}
```

## GPA 不参与计算

数据文件里虽然有 GPA，但这个项目不会使用 GPA 做输入、标签或评分依据。

当前 Physical Activity 数据来自：

```text
physical_activity_cleaned.csv
```

其中 GPA 相关列会被忽略。GPA 不能代表当天的健康状态，也不是我们想生成健康建议的依据。

## 项目结构

```text
src/
├── agents/
│   ├── physical_activity_model.py  # Physical Activity 小模型
│   ├── student_wellness.py         # 总流程：选择、打分、生成建议
│   └── scheduler.py                 # 指定时间触发
├── schema/
│   └── student_wellness.py          # 日程、生活数据、分数和建议的数据格式
└── core/
    ├── llm.py                       # LLM，包括 AWS Bedrock 配置
    └── settings.py                  # 项目配置
```

## 总体流程

```text
学生日程 + 生活数据
        ↓
选择需要调用的小模型
        ↓
小模型分别输出分数
        ↓
LLM 综合日程和分数
        ↓
生成一条英文建议
```

现在框架只要求每个小模型实现同一个接口：

```python
async def score(schedule, signals) -> MetricScore:
    ...
```

以后可以按同样方式加入：

- Sleep Model
- Eating Model
- Workload Model
- 其他生活习惯模型

## Physical Activity 评分方式

当前是一个简单的基线版本：

- 运动天数：60%
- 力量训练天数：20%
- 运动分钟数：20%

分数越高，代表当前运动情况越好：

- `good`：分数大于等于 80
- `attention`：分数大于等于 50 且小于 80
- `poor`：分数小于 50

这不是医疗诊断，只是为了给后面的建议生成提供一个统一的参考指标。

## AWS 配置

真实 LLM 调用使用 AWS Bedrock。凭证只通过环境变量读取，不要写进代码或提交到 Git：

```env
USE_AWS_BEDROCK=true
AWS_REGION=ap-southeast-1
AWS_BEDROCK_MODEL_ID=global.anthropic.claude-haiku-4-5-20251001-v1:0
```

临时凭证示例：

```bat
set AWS_ACCESS_KEY_ID=your_access_key
set AWS_SECRET_ACCESS_KEY=your_secret_key
set AWS_SESSION_TOKEN=your_session_token
```

不要把真实凭证发到聊天、写进 `.env.example` 或提交到仓库。

## 安装依赖

项目使用 Python 3.12 或更高版本，以及 `uv` 管理依赖：

```bash
uv sync
```

## 重要说明

- 当前 Physical Activity 模型不会调用 API。
- 当前还没有实现 Sleep、Eating 和 Workload 的具体小模型。
- 总流程已经预留了这些模型的接口。
- 只有最终建议生成器需要调用 AWS LLM 时，才会消耗 API 额度。
- 建议输出和 Physical Activity 模型输出统一使用英文。
- 系统只提供生活方式建议，不提供医疗诊断。
