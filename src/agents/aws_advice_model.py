"""AWS Bedrock final advice adapter. Not used by local tests."""
from schema.student_wellness import MetricScore, Recommendation, StudentSchedule


class AWSBedrockAdviceModel:
    def __init__(self, model):
        self.model = model

    async def generate(self, schedule: StudentSchedule, scores: list[MetricScore]) -> Recommendation:
        structured = self.model.with_structured_output(Recommendation)
        return await structured.ainvoke([
            ("system", "Generate one concise, practical recommendation in English. Do not diagnose or mention GPA."),
            ("human", str({"schedule": schedule.model_dump(), "scores": [s.model_dump() for s in scores]})),
        ])
