import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from query_analyzer import analyze_query
from rag import answer_question, rewrite_query


class QueryAnalyzerTests(unittest.TestCase):
    def test_standalone_question_uses_direct_retrieval(self) -> None:
        self.assertEqual(analyze_query("What accuracy did VGG16 achieve?"), "direct")

    def test_context_dependent_question_uses_rewriting(self) -> None:
        self.assertEqual(analyze_query("How did it perform?"), "rewrite")

    def test_multi_part_question_uses_decomposition(self) -> None:
        self.assertEqual(
            analyze_query("Compare VGG16 and MobileNetV2, and explain the limitation."),
            "decompose",
        )

    def test_follow_up_question_about_other_models_uses_rewriting(self) -> None:
        self.assertEqual(analyze_query("What about the other models?"), "rewrite")

    def test_query_analyzer_rejects_empty_questions(self) -> None:
        with self.assertRaises(ValueError):
            analyze_query("  ")


class QueryRewriteTests(unittest.TestCase):
    def test_rewrite_query_returns_model_query(self) -> None:
        llm = Mock()
        llm.invoke.return_value.content = "VGG16 classification accuracy on lung CT images"

        with patch("rag.create_llm", return_value=llm):
            rewritten = rewrite_query("How did that model do?")

        self.assertEqual(rewritten, "VGG16 classification accuracy on lung CT images")

    def test_rewrite_query_rejects_empty_questions(self) -> None:
        with self.assertRaises(ValueError):
            rewrite_query("  ")

    def test_answer_retrieves_using_rewritten_query(self) -> None:
        llm = Mock()
        llm.invoke.side_effect = [
            Mock(content="VGG16 classification accuracy"),
            Mock(content="VGG16 achieved the highest accuracy."),
        ]
        retrieved = [(Document(page_content="VGG16 achieved 98.18%."), 0.9)]

        with patch("rag.create_llm", return_value=llm):
            with patch("rag.retrieve", return_value=retrieved) as retrieve_mock:
                with patch("rag.evaluate_context"):
                    answer = answer_question("How did it perform?", k=1)

        retrieve_mock.assert_called_once_with(
            "VGG16 classification accuracy", k=1, index_path=unittest.mock.ANY
        )
        self.assertIn("VGG16 achieved the highest accuracy.", answer)
        self.assertIn("[Source chunk 1]", answer)

    def test_answer_retrieves_directly_for_standalone_question(self) -> None:
        llm = Mock()
        llm.invoke.return_value.content = "VGG16 achieved the highest accuracy."
        retrieved = [(Document(page_content="VGG16 achieved 98.18%."), 0.9)]

        with patch("rag.create_llm", return_value=llm):
            with patch("rag.retrieve", return_value=retrieved) as retrieve_mock:
                with patch("rag.evaluate_context"):
                    answer_question("What accuracy did VGG16 achieve?", k=1)

        retrieve_mock.assert_called_once_with(
            "What accuracy did VGG16 achieve?", k=1, index_path=unittest.mock.ANY
        )
        self.assertEqual(llm.invoke.call_count, 1)

    def test_direct_retrieval_falls_back_to_rewrite_for_low_score(self) -> None:
        llm = Mock()
        llm.invoke.side_effect = [
            Mock(content="VGG16 classification accuracy"),
            Mock(content="VGG16 achieved the highest accuracy."),
        ]
        retrieved = [(Document(page_content="weak match"), -0.5)]
        fallback = [(Document(page_content="VGG16 achieved 98.18%."), 0.8)]

        with patch("rag.create_llm", return_value=llm):
            with patch("rag.retrieve", side_effect=[retrieved, fallback]) as retrieve_mock:
                with patch("rag.evaluate_context"):
                    answer_question("What accuracy did VGG16 achieve?", k=1)

        self.assertEqual(retrieve_mock.call_count, 2)
        self.assertEqual(llm.invoke.call_count, 2)


if __name__ == "__main__":
    unittest.main()
