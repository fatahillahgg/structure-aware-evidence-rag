import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from rag import answer_question, rewrite_query


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
                answer = answer_question("How did it perform?", k=1)

        retrieve_mock.assert_called_once_with(
            "VGG16 classification accuracy", k=1, index_path=unittest.mock.ANY
        )
        self.assertEqual(answer, "VGG16 achieved the highest accuracy.")


if __name__ == "__main__":
    unittest.main()
