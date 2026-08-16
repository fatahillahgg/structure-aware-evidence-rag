from dataclasses import dataclass
from typing import Literal

from context_evaluator import ContextEvaluation


RetrievalAction = Literal[
    "ANSWER",
    "REWRITE",
    "EXPAND",
    "DECOMPOSE",
    "RETRY_DIRECT",
    "ABSTAIN",
]


@dataclass(frozen=True)
class ControllerDecision:
    action: RetrievalAction
    reason: str


def choose_action(
    evaluation: ContextEvaluation,
    current_mode: str,
    attempt: int,
    max_attempts: int = 2,
) -> ControllerDecision:
    """Choose the next retrieval action from an evidence assessment."""
    if evaluation.status == "sufficient" and evaluation.confidence >= 0.65:
        return ControllerDecision("ANSWER", "evidence is sufficient and confident")

    if (
        evaluation.status == "partial"
        and evaluation.confidence >= 0.75
        and evaluation.supported_aspects
    ):
        return ControllerDecision(
            "ANSWER",
            "evidence supports a grounded partial answer; disclose missing aspects",
        )

    if attempt >= max_attempts:
        return ControllerDecision("ABSTAIN", "corrective retrieval budget exhausted")

    if evaluation.status == "conflicting":
        if current_mode != "direct":
            return ControllerDecision("RETRY_DIRECT", "conflicting evidence needs an independent direct search")
        return ControllerDecision("EXPAND", "conflicting evidence needs broader query coverage")

    if evaluation.status == "insufficient":
        if current_mode == "direct":
            return ControllerDecision("REWRITE", "direct evidence is insufficient")
        if current_mode != "decompose" and len(evaluation.missing_aspects) > 1:
            return ControllerDecision("DECOMPOSE", "multiple missing aspects need separate searches")
        return ControllerDecision("EXPAND", "evidence is insufficient and needs broader queries")

    if evaluation.status == "partial":
        if len(evaluation.missing_aspects) > 1 and current_mode != "decompose":
            return ControllerDecision("DECOMPOSE", "multiple aspects are still missing")
        return ControllerDecision("EXPAND", "some answer aspects are still missing")

    return ControllerDecision("ABSTAIN", "unknown context status")
