import unittest

from context_evaluator import ContextEvaluation, SupportedAspect
from retrieval_controller import choose_action


def evaluation(status: str, confidence: float = 0.8, missing: list[str] | None = None):
    return ContextEvaluation(status, [], missing or [], [], [], confidence)


class RetrievalControllerTests(unittest.TestCase):
    def test_sufficient_evidence_answers(self) -> None:
        decision = choose_action(evaluation("sufficient"), "direct", attempt=0)
        self.assertEqual(decision.action, "ANSWER")

    def test_insufficient_direct_evidence_rewrites(self) -> None:
        decision = choose_action(evaluation("insufficient"), "direct", attempt=0)
        self.assertEqual(decision.action, "REWRITE")

    def test_multiple_missing_aspects_decompose(self) -> None:
        decision = choose_action(
            evaluation("partial", missing=["accuracy", "optimizer"]),
            "rewrite",
            attempt=0,
        )
        self.assertEqual(decision.action, "DECOMPOSE")

    def test_confident_partial_evidence_answers_with_disclosure(self) -> None:
        context = ContextEvaluation(
            "partial",
            [SupportedAspect("VGG16 accuracy", ["chunk-1"])],
            ["CNN accuracy"],
            [],
            ["chunk-1"],
            0.82,
        )

        decision = choose_action(context, "direct", attempt=0)

        self.assertEqual(decision.action, "ANSWER")

    def test_conflicting_rewritten_evidence_retries_direct(self) -> None:
        decision = choose_action(evaluation("conflicting"), "rewrite", attempt=0)
        self.assertEqual(decision.action, "RETRY_DIRECT")

    def test_budget_exhaustion_abstains(self) -> None:
        decision = choose_action(evaluation("partial", missing=["accuracy"]), "expand", attempt=2)
        self.assertEqual(decision.action, "ABSTAIN")


if __name__ == "__main__":
    unittest.main()
