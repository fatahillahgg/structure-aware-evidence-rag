import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from evaluate import keyword_recall, token_f1
from retrieval import hybrid_retrieve, reciprocal_rank_fusion


class RetrievalTests(unittest.TestCase):
    def test_rrf_prefers_documents_present_in_both_rankings(self) -> None:
        shared = Document(page_content="shared")
        dense_only = Document(page_content="dense only")
        bm25_only = Document(page_content="bm25 only")

        results = reciprocal_rank_fusion(
            [
                [(shared, 0.1), (dense_only, 0.2)],
                [(shared, 2.0), (bm25_only, 1.0)],
            ]
        )

        self.assertEqual(results[0][0].page_content, "shared")
        self.assertGreater(results[0][1], results[1][1])

    def test_hybrid_retrieval_fuses_dense_and_bm25_results(self) -> None:
        dense = [(Document(page_content="dense"), 0.1)]
        sparse = [(Document(page_content="sparse"), 2.0)]

        with patch("retrieval.dense_retrieve", return_value=dense) as dense_mock:
            with patch("retrieval.bm25_retrieve", return_value=sparse) as sparse_mock:
                results = hybrid_retrieve("test query", k=1)

        dense_mock.assert_called_once()
        sparse_mock.assert_called_once()
        self.assertEqual(len(results), 1)

    def test_hybrid_retrieval_rejects_empty_queries(self) -> None:
        with self.assertRaises(ValueError):
            hybrid_retrieve("  ")


class EvaluationMetricTests(unittest.TestCase):
    def test_keyword_recall(self) -> None:
        self.assertEqual(keyword_recall("VGG16 achieved 98.18%", ["VGG16", "98.18%"]), 1.0)
        self.assertEqual(keyword_recall("VGG16", ["VGG16", "ImageNet"]), 0.5)

    def test_token_f1_is_perfect_for_same_text(self) -> None:
        self.assertEqual(token_f1("VGG16 achieved 98.18%", "VGG16 achieved 98.18%"), 1.0)


if __name__ == "__main__":
    unittest.main()
