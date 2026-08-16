import re
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from chunker import chunk_documents
from loader import load_paper
from observability import trace_event, trace_span
from vector_store import DEFAULT_INDEX_PATH, load_vector_store


load_dotenv()
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[._-][a-z0-9]+)*%?", text.lower())


@lru_cache(maxsize=1)
def _bm25_corpus() -> tuple[list[Document], BM25Okapi]:
    documents = chunk_documents(load_paper())
    index = BM25Okapi([_tokenize(document.page_content) for document in documents])
    return documents, index


def dense_retrieve(
    query: str,
    k: int = 4,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> list[tuple[Document, float]]:
    """Retrieve the most relevant paper chunks and their FAISS scores."""
    return load_vector_store(index_path).similarity_search_with_score(query, k=k)


def bm25_retrieve(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """Retrieve chunks using sparse Okapi BM25 term matching."""
    documents, index = _bm25_corpus()
    scores = index.get_scores(_tokenize(query))
    ranked_indices = sorted(range(len(documents)), key=lambda item: scores[item], reverse=True)
    return [(documents[index], float(scores[index])) for index in ranked_indices[:k]]


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[Document, float]]],
    k: int = 60,
) -> list[tuple[Document, float]]:
    """Merge ranked results using Reciprocal Rank Fusion (RRF)."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranked_list in ranked_lists:
        for rank, (document, _) in enumerate(ranked_list, start=1):
            key = document.page_content
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            documents[key] = document

    ranked_documents = sorted(scores, key=scores.get, reverse=True)
    return [(documents[key], scores[key]) for key in ranked_documents]


@lru_cache(maxsize=1)
def _create_reranker() -> CrossEncoder:
    """Load the local cross-encoder once per process."""
    return CrossEncoder(RERANKER_MODEL)


def rerank(
    query: str,
    candidates: list[tuple[Document, float]],
    k: int,
) -> list[tuple[Document, float]]:
    """Rerank RRF candidates using query-document relevance scores."""
    if not candidates:
        return []

    pairs = [(query, document.page_content) for document, _ in candidates]
    relevance_scores = _create_reranker().predict(pairs)
    reranked = [
        (document, float(score))
        for (document, _), score in zip(candidates, relevance_scores, strict=True)
    ]
    return sorted(reranked, key=lambda item: item[1], reverse=True)[:k]


def normalize_evidence_text(text: str) -> str:
    """Normalize evidence text for duplicate comparison without changing its source."""
    return re.sub(r"\s+", " ", text).strip().lower()


def deduplicate_evidence(
    candidates: list[tuple[Document, float]],
    k: int,
    similarity_threshold: float = 0.92,
) -> list[tuple[Document, float]]:
    """Remove exact and near-duplicate evidence while preserving rank order."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be greater than 0 and at most 1")

    kept: list[tuple[Document, float]] = []
    normalized_kept: list[str] = []
    duplicate_count = 0
    for document, score in candidates:
        normalized = normalize_evidence_text(document.page_content)
        if not normalized:
            duplicate_count += 1
            continue
        is_duplicate = any(
            normalized == previous
            or SequenceMatcher(None, normalized, previous).ratio() >= similarity_threshold
            for previous in normalized_kept
        )
        if is_duplicate:
            duplicate_count += 1
            continue
        kept.append((document, score))
        normalized_kept.append(normalized)
        if len(kept) >= k:
            break

    trace_event(
        "evidence_deduplicated",
        input_count=len(candidates),
        output_count=len(kept),
        duplicate_count=duplicate_count,
        similarity_threshold=similarity_threshold,
    )
    return kept


def hybrid_retrieve(
    query: str,
    k: int = 4,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    candidate_k: int | None = None,
    rrf_k: int = 60,
) -> list[tuple[Document, float]]:
    """Fuse dense and sparse rankings, then rerank the RRF candidates."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if k < 1:
        raise ValueError("k must be at least 1")
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1")

    candidates = max(k, candidate_k or max(10, k * 3))
    with trace_span("dense_retrieval", k=candidates):
        dense_results = dense_retrieve(query, k=candidates, index_path=index_path)
    with trace_span("bm25_retrieval", k=candidates):
        sparse_results = bm25_retrieve(query, k=candidates)
    fused = reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k)
    with trace_span("cross_encoder_rerank", candidate_count=len(fused), k=len(fused)):
        reranked = rerank(query, fused, k=len(fused))
    results = deduplicate_evidence(reranked, k=k)
    trace_event(
        "retrieval_finished",
        dense_count=len(dense_results),
        sparse_count=len(sparse_results),
        fused_count=len(fused),
        result_count=len(results),
        top_score=results[0][1] if results else None,
    )
    return results


def retrieve(
    query: str,
    k: int = 4,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> list[tuple[Document, float]]:
    """Retrieve with dense+sparse hybrid search and RRF ranking."""
    return hybrid_retrieve(query, k=k, index_path=index_path)


def build_context(
    query: str,
    k: int = 4,
    index_path: str | Path = DEFAULT_INDEX_PATH,
) -> str:
    """Return retrieved chunks as a single context string for an LLM prompt."""
    results = hybrid_retrieve(query, k=k, index_path=index_path)
    return "\n\n---\n\n".join(document.page_content for document, _ in results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve chunks with dense+sparse RRF search")
    parser.add_argument("query", help="Question or search query")
    parser.add_argument("-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()

    for number, (document, score) in enumerate(
        hybrid_retrieve(args.query, k=args.k, index_path=args.index_path),
        start=1,
    ):
        section = " > ".join(document.metadata.get("section_path", []))
        chunk_id = document.metadata.get("chunk_id", f"chunk-{number}")
        print(f"[{number}] score={score:.4f} chunk_id={chunk_id} section={section}")
        print(document.page_content)
        print()
