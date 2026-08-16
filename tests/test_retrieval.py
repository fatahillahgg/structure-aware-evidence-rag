import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from chunker import chunk_documents
from evaluate import keyword_recall, token_f1
from retrieval import (
    deduplicate_evidence,
    hybrid_retrieve,
    normalize_evidence_text,
    reciprocal_rank_fusion,
    rerank,
)


class RetrievalTests(unittest.TestCase):
    def test_structure_aware_chunks_preserve_section_metadata(self) -> None:
        documents = [
            Document(
                page_content="# Paper\n\n## Results\n\nVGG16 achieved 98.18%.",
                metadata={"title": "Paper", "year": 2024},
            )
        ]

        chunks = chunk_documents(documents, chunk_size=80, chunk_overlap=10)

        results_chunk = next(chunk for chunk in chunks if chunk.metadata["section"] == "Results")
        self.assertEqual(results_chunk.metadata["section_path"], ["Paper", "Results"])
        self.assertEqual(results_chunk.metadata["year"], 2024)
        self.assertTrue(results_chunk.metadata["chunk_id"].startswith("document-0-chunk-"))

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
                with patch("retrieval.rerank", side_effect=lambda query, items, k: items[:k]) as rerank_mock:
                    results = hybrid_retrieve("test query", k=1)

        dense_mock.assert_called_once()
        sparse_mock.assert_called_once()
        rerank_mock.assert_called_once()
        self.assertEqual(len(results), 1)

    def test_rerank_orders_candidates_by_cross_encoder_score(self) -> None:
        first = Document(page_content="first")
        second = Document(page_content="second")

        with patch("retrieval._create_reranker") as model_factory:
            model_factory.return_value.predict.return_value = [0.1, 0.9]
            results = rerank("query", [(first, 1.0), (second, 0.5)], k=2)

        self.assertEqual([document.page_content for document, _ in results], ["second", "first"])
        self.assertEqual(results[0][1], 0.9)

    def test_evidence_normalization_collapses_whitespace(self) -> None:
        self.assertEqual(
            normalize_evidence_text("VGG16\n achieved   98.18%."),
            "vgg16 achieved 98.18%.",
        )

    def test_deduplicate_evidence_removes_near_duplicates_after_reranking(self) -> None:
        duplicate = Document(page_content="VGG16 achieved 98.18% test accuracy.")
        near_duplicate = Document(page_content="VGG16 achieved 98.18% test accuracy. ")
        distinct = Document(page_content="MobileNetV2 achieved 93.64% test accuracy.")

        results = deduplicate_evidence(
            [(duplicate, 0.9), (near_duplicate, 0.8), (distinct, 0.7)],
            k=2,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].page_content, duplicate.page_content)
        self.assertEqual(results[1][0].page_content, distinct.page_content)

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
