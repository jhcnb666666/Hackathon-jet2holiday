import pandas as pd

from student_model.data import build_sequence_samples, split_student_ids
from student_model.synthetic import SyntheticConfig, generate_interactions


def test_synthetic_interactions_are_reproducible_and_ordered():
    config = SyntheticConfig(students=8, interactions_per_student=10, seed=7)
    first = generate_interactions(config)
    second = generate_interactions(config)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 80
    assert first["is_synthetic"].all()
    assert first["correct"].isin([0, 1]).all()
    assert (
        first.groupby("student_id")["interaction_index"]
        .apply(lambda values: values.tolist() == list(range(10)))
        .all()
    )


def test_student_split_is_disjoint_and_samples_have_next_targets():
    frame = generate_interactions(SyntheticConfig(students=20, interactions_per_student=10))
    split = split_student_ids(frame)
    assert set(split["train"]).isdisjoint(split["validation"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["validation"]).isdisjoint(split["test"])
    samples = build_sequence_samples(frame, split["train"], min_history=4)
    assert samples
    assert min(len(sample.concept_ids) for sample in samples) >= 4
    assert all(sample.next_correct in {0.0, 1.0} for sample in samples)
