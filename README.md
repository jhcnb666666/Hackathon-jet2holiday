# Student Wellness Pipeline

This project provides daily lifestyle recommendations for students.

At a scheduled time, the system reads the student's daily schedule and lifestyle data such as sleep, physical activity, eating habits, and work. It selects the relevant specialist models, collects their scores, and sends the schedule and scores to an AWS Bedrock LLM to generate one simple, practical recommendation in English.

GPA is not used. It is not included in the lifestyle data, does not affect specialist model scores, and is not sent to the recommendation model.

## Current status

The local demo can run end to end without calling AWS. The current implementation includes the schedule and lifestyle data schemas, a Physical Activity model, a SleepModel adapter, a CSV loader, a scheduled trigger, the main pipeline, an AWS Bedrock recommendation adapter, and a local demo.

The following parts are not implemented yet: the Eating model, the Workload model, a production HTTP API, a background scheduling service, and real AWS Bedrock call verification.

## Key files

- `src/agents/student_wellness.py`: the main pipeline.
- `src/agents/physical_activity_model.py`: the Physical Activity specialist model.
- `src/agents/sleep_model.py`: the Sleep model adapter.
- `models/sleep_model.json`: the Sleep model artifact.
- `src/agents/aws_advice_model.py`: the AWS Bedrock recommendation adapter.
- `src/data/wellness_loader.py`: the CSV data loader.
- `src/schema/student_wellness.py`: shared data schemas.
- `src/agents/scheduler.py`: the scheduled trigger.
- `scripts/run_wellness_demo.py`: the local end-to-end demo.
- `PIPELINE.md`: detailed pipeline documentation.

## Data files

The real datasets are located in `data/real/` and include sleep, physical activity, eating habits, work, and demographics data. Only sleep and physical activity data are currently connected to the pipeline. GPA and the GPA quality flag are ignored.

## Running locally

Python 3.12 or later and `uv` are required. Run `uv sync`, set `PYTHONPATH=src`, and then run `scripts/run_wellness_demo.py`. The local demo does not access AWS or consume API quota.

## AWS Bedrock

Real recommendations use AWS Bedrock. Set `USE_AWS_BEDROCK=true`, `AWS_REGION`, and `AWS_BEDROCK_MODEL_ID`, then provide temporary AWS credentials through system environment variables. Do not put real credentials in the code, README, `.env.example`, or Git.

You also need to make sure that the AWS account has permission to call the selected Bedrock model.

## Design principles

Each specialist model is responsible for one lifestyle dimension and returns the shared `MetricScore` format. The LLM combines the schedule and specialist scores; it does not replace specialist scoring. Recommendations are always generated in English. The system provides lifestyle suggestions and does not provide medical diagnoses.
