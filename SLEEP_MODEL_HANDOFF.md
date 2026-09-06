# Sleep model handoff

## Scope

This component summarizes sleep survey cases for the wellness LLM pipeline. It does not
predict GPA, diagnose a sleep disorder, or call an external LLM/API.

## Inputs used for training

- `sleep_duration`
- `bedtime`
- `wake_time`

The loader uses an explicit feature allowlist. `record_id`, `current_gpa`,
`gpa_quality_flag`, and any future non-sleep columns are ignored.

## Model

The survey has no human-labelled sleep-quality target. The training script therefore uses
deterministic categorical k-modes rather than manufacturing a supervised target. The saved
JSON artifact contains four learned pattern prototypes and their sample shares.

Runtime scoring is deliberately separate from pattern learning:

- The learned pattern supplies a compact survey summary for the LLM.
- The score represents adult sleep-duration adequacy using the 7+ hour reference.
- Bedtime and wake time are reported as evidence but are not treated as a diagnosis.

## Train

```bash
python scripts/train_sleep_model.py \
  --input data/sleep_cleaned.csv \
  --output models/sleep_model.json
```

The checked-in artifact was trained from all 614 supplied cases.

## Integrate

```python
from agents.physical_activity_model import PhysicalActivityModel
from agents.sleep_model import SleepModel

models = {
    "sleep": SleepModel(),
    "physical_activity": PhysicalActivityModel(),
}
```

`SleepModel` implements the existing specialist contract:

```python
async def score(schedule, signals) -> MetricScore:
    ...
```

Example output:

```json
{
  "metric": "sleep",
  "score": 65.0,
  "level": "attention",
  "evidence": [
    "Sleep duration: 6.0 hours (is below the adult 7+ hour reference)",
    "Sleep onset: 01:00",
    "Wake time: 07:00",
    "Nearest survey pattern: 6 hours, Between 12am and 2am, Between 8am and 10am (37.1% of training cases)"
  ]
}
```

## Files

- `src/sleep_clustering.py`: feature loading, k-modes training, and survey binning
- `src/agents/sleep_model.py`: pipeline adapter and duration score
- `scripts/train_sleep_model.py`: command-line training entry point
- `models/sleep_model.json`: trained portable model artifact
- `tests/agents/test_sleep_model.py`: training and runtime tests
- `data/sleep_cleaned.csv`: supplied anonymized survey cases

## References

- CDC: https://www.cdc.gov/sleep/about/index.html
- AASM/SRS: https://aasm.org/resources/pdf/pressroom/adult-sleep-duration-consensus.pdf

