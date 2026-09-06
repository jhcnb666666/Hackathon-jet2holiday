from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class StudentLSTM(nn.Module):
    """Multi-task sequence model for next-response and current-concept state prediction."""

    def __init__(
        self,
        num_concepts: int,
        num_questions: int,
        numeric_feature_count: int = 5,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        layers: int = 1,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.config = {
            "num_concepts": num_concepts,
            "num_questions": num_questions,
            "numeric_feature_count": numeric_feature_count,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "layers": layers,
            "dropout": dropout,
        }
        self.concept_embedding = nn.Embedding(num_concepts, embedding_dim)
        self.question_embedding = nn.Embedding(num_questions, embedding_dim)
        self.numeric_projection = nn.Sequential(
            nn.Linear(numeric_feature_count, embedding_dim),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim * 3,
            hidden_size=hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.shared = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.next_correct_head = nn.Linear(hidden_dim, 1)
        self.mastery_head = nn.Linear(hidden_dim, 1)
        self.struggling_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        concept_ids: torch.Tensor,
        question_ids: torch.Tensor,
        numeric_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedded = torch.cat(
            [
                self.concept_embedding(concept_ids),
                self.question_embedding(question_ids),
                self.numeric_projection(numeric_features),
            ],
            dim=-1,
        )
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        representation = self.shared(hidden[-1])
        return {
            "next_correct_logit": self.next_correct_head(representation).squeeze(-1),
            "mastery": torch.sigmoid(self.mastery_head(representation).squeeze(-1)),
            "struggling_logit": self.struggling_head(representation).squeeze(-1),
        }


class FixedWidthStudentModel(nn.Module):
    """Logistic or MLP baseline over aggregate history features."""

    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = ()) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden_dims:
            layers.extend([nn.Linear(previous, width), nn.ReLU(), nn.Dropout(0.15)])
            previous = width
        self.encoder = nn.Sequential(*layers) if layers else nn.Identity()
        self.next_correct_head = nn.Linear(previous, 1)
        self.mastery_head = nn.Linear(previous, 1)
        self.struggling_head = nn.Linear(previous, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        representation = self.encoder(features)
        return {
            "next_correct_logit": self.next_correct_head(representation).squeeze(-1),
            "mastery": torch.sigmoid(self.mastery_head(representation).squeeze(-1)),
            "struggling_logit": self.struggling_head(representation).squeeze(-1),
        }


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    next_correct: torch.Tensor,
    mastery: torch.Tensor,
    struggling: torch.Tensor,
) -> torch.Tensor:
    correctness_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["next_correct_logit"], next_correct
    )
    mastery_loss = nn.functional.mse_loss(outputs["mastery"], mastery)
    struggling_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["struggling_logit"], struggling
    )
    return correctness_loss + 0.4 * mastery_loss + 0.5 * struggling_loss
