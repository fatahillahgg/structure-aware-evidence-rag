import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from chunker import chunk_documents
from loader import load_paper
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
    fused = reciprocal_rank_fusion(
        [
            dense_retrieve(query, k=candidates, index_path=index_path),
            bm25_retrieve(query, k=candidates),
        ],
        k=rrf_k,
    )
    return rerank(query, fused, k=k)


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
        print(f"[{number}] score={score:.4f}")
        print(document.page_content)
        print()
