import unittest
from unittest.mock import Mock

from langchain_core.documents import Document

from context_evaluator import evaluate_context


class ContextEvaluatorTests(unittest.TestCase):
    def test_evaluator_parses_and_filters_evidence_ids(self) -> None:
        llm = Mock()
        llm.invoke.return_value.content = (
            '{"status":"partial",'
            '"supported_aspects":[{"aspect":"VGG16 test accuracy",'
            '"evidence_ids":["chunk-12","unknown"]}],'
            '"missing_aspects":["CNN test accuracy"],'
            '"conflicting_aspects":[],"evidence_ids":["chunk-12","unknown"],'
            '"confidence":0.82}'
        )
        results = [
            (
                Document(
                    page_content="VGG16 achieved 98.18%.",
                    metadata={"chunk_id": "chunk-12", "section_path": ["Results"]},
                ),
                8.2,
            )
        ]

        evaluation = evaluate_context("Compare VGG16 and CNN accuracy.", results, llm)

        self.assertEqual(evaluation.status, "partial")
        self.assertEqual(evaluation.evidence_ids, ["chunk-12"])
        self.assertEqual(evaluation.supported_aspects[0].aspect, "VGG16 test accuracy")
        self.assertEqual(evaluation.confidence, 0.82)

    def test_empty_context_is_insufficient_without_llm_call(self) -> None:
        llm = Mock()

        evaluation = evaluate_context("What is the accuracy?", [], llm)

        self.assertEqual(evaluation.status, "insufficient")
        self.assertEqual(evaluation.evidence_ids, [])
        llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
