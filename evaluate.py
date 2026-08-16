import argparse
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from embedder import create_embedder
from query_analyzer import analyze_query
from rag import _merge_results, answer_question, decompose_query, rewrite_query
from retrieval import DEFAULT_INDEX_PATH, retrieve


DEFAULT_EVAL_PATH = Path(__file__).parent / "data" / "evaluation" / "dataset.json"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "evaluation" / "latest"


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", text.lower()).strip()


def keyword_recall(text: str, keywords: list[str]) -> float:
    """Return the fraction of expected keyword phrases found in text."""
    if not keywords:
        return 1.0
    normalised_text = _normalise(text)
    matched = sum(_normalise(keyword) in normalised_text for keyword in keywords)
    return matched / len(keywords)


def token_f1(prediction: str, reference: str) -> float:
    """Calculate a simple unigram F1 against the expected answer."""
    predicted = _normalise(prediction).split()
    expected = _normalise(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)

    predicted_counts = defaultdict(int)
    expected_counts = defaultdict(int)
    for token in predicted:
        predicted_counts[token] += 1
    for token in expected:
        expected_counts[token] += 1

    overlap = sum(min(count, expected_counts[token]) for token, count in predicted_counts.items())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


@lru_cache(maxsize=1)
def _evaluation_embedder():
    return create_embedder()


def semantic_similarity(prediction: str, reference: str) -> float:
    """Calculate cosine similarity using the local normalized embedder."""
    if not prediction.strip() or not reference.strip():
        return 0.0
    prediction_vector = _evaluation_embedder().embed_query(prediction)
    reference_vector = _evaluation_embedder().embed_query(reference)
    return max(
        0.0,
        sum(left * right for left, right in zip(prediction_vector, reference_vector, strict=True)),
    )


def has_source_citation(answer: str) -> bool:
    return bool(re.search(r"\[Source chunk \d+\]", answer))


def is_abstention(answer: str) -> bool:
    return answer.startswith("I do not have enough reliable evidence")


def evaluate_item(
    item: dict[str, Any],
    k: int,
    generate: bool,
    query_mode: str,
) -> dict[str, Any]:
    history = item.get("history")
    actual_query_mode = (
        analyze_query(item["question"], history=history)
        if query_mode == "analyzer"
        else query_mode
    )
    if actual_query_mode == "rewrite":
        retrieval_query = rewrite_query(item["question"], history=history)
        results = retrieve(retrieval_query, k=k)
    elif actual_query_mode == "decompose":
        subqueries = decompose_query(item["question"], history=history)
        retrieval_query = " | ".join(subqueries)
        results = _merge_results(
            item["question"],
            [retrieve(subquery, k=k) for subquery in subqueries],
            k=k,
        )
    else:
        retrieval_query = item["question"]
        results = retrieve(retrieval_query, k=k)
    context = "\n".join(document.page_content for document, _ in results)
    if generate and actual_query_mode == "decompose":
        answer = answer_question(item["question"], k=k, history=history)
    elif generate:
        answer = answer_question(item["question"], k=k, retrieval_query=retrieval_query)
    else:
        answer = ""
    keywords = item.get("expected_keywords", [])

    return {
        "id": item["id"],
        "category": item.get("category", "uncategorized"),
        "difficulty": item.get("difficulty", "unknown"),
        "question": item["question"],
        "query_mode": actual_query_mode,
        "expected_query_mode": item.get("query_type"),
        "routing_correct": (
            actual_query_mode == item["query_type"]
            if query_mode == "analyzer" and item.get("query_type")
            else None
        ),
        "retrieval_query": retrieval_query,
        "answer": answer,
        "retrieval_keyword_recall": keyword_recall(context, keywords),
        "answer_keyword_recall": keyword_recall(answer, keywords) if generate else None,
        "answer_token_f1": token_f1(answer, item["expected_answer"]) if generate else None,
        "answer_semantic_similarity": (
            semantic_similarity(answer, item["expected_answer"]) if generate else None
        ),
        "answer_has_citation": has_source_citation(answer) if generate else None,
        "answer_abstained": is_abstention(answer) if generate else None,
    }


def _average(rows: list[dict[str, Any]], key: str) -> float:
    values = [row[key] for row in rows if row[key] is not None]
    return sum(values) / len(values) if values else 0.0


def make_chart(rows: list[dict[str, Any]], output_path: Path, generate: bool) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    categories = list(grouped)
    metrics = [("Retrieval keyword recall", "retrieval_keyword_recall")]
    if generate:
        metrics.extend(
            [
                ("Answer keyword recall", "answer_keyword_recall"),
                ("Answer token F1", "answer_token_f1"),
                ("Answer semantic similarity", "answer_semantic_similarity"),
            ]
        )

    width = 0.8 / len(metrics)
    positions = list(range(len(categories)))
    fig, axis = plt.subplots(figsize=(11, 6))
    for offset, (label, key) in enumerate(metrics):
        values = [_average(grouped[category], key) for category in categories]
        axis.bar(
            [position + (offset - (len(metrics) - 1) / 2) * width for position in positions],
            values,
            width=width,
            label=label,
        )

    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Hybrid RAG Evaluation: Dense + BM25 + RRF")
    axis.set_xticks(positions, [category.replace("_", " ").title() for category in categories])
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and chart naive RAG performance")
    parser.add_argument("--eval-path", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument(
        "--query-mode",
        choices=("analyzer", "direct", "rewrite", "decompose"),
        default="analyzer",
        help="Use the analyzer, original question, or an LLM-rewritten query for retrieval",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Evaluate retrieval without answer-generation calls",
    )
    args = parser.parse_args()

    items = json.loads(args.eval_path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("Evaluation file must contain a non-empty JSON list")

    rows = [
        evaluate_item(
            item,
            k=args.k,
            generate=not args.retrieval_only,
            query_mode=args.query_mode,
        )
        for item in items
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.json"
    chart_path = args.output_dir / f"{args.query_mode}_rag_performance.png"
    results_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    make_chart(rows, chart_path, generate=not args.retrieval_only)

    print(f"Evaluated {len(rows)} questions")
    print(f"Retrieval keyword recall: {_average(rows, 'retrieval_keyword_recall'):.1%}")
    routing_values = [row["routing_correct"] for row in rows if row["routing_correct"] is not None]
    if routing_values:
        print(f"Query routing accuracy: {sum(routing_values) / len(routing_values):.1%}")
    if not args.retrieval_only:
        print(f"Answer keyword recall: {_average(rows, 'answer_keyword_recall'):.1%}")
        print(f"Answer token F1: {_average(rows, 'answer_token_f1'):.1%}")
        print(
            "Answer semantic similarity: "
            f"{_average(rows, 'answer_semantic_similarity'):.1%}"
        )
        citation_values = [row["answer_has_citation"] for row in rows if row["answer_has_citation"] is not None]
        print(f"Answers with citations: {sum(citation_values) / len(citation_values):.1%}")
        abstention_values = [row["answer_abstained"] for row in rows if row["answer_abstained"] is not None]
        print(f"Answer rate: {1 - (sum(abstention_values) / len(abstention_values)):.1%}")
        print(f"Abstention rate: {sum(abstention_values) / len(abstention_values):.1%}")
    print(f"Chart: {chart_path}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
