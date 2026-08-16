import argparse
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from context_evaluator import ContextEvaluation, evaluate_context
from observability import trace_event, trace_request, trace_span
from query_analyzer import analyze_query
from retrieval import DEFAULT_INDEX_PATH, deduplicate_evidence, rerank, retrieve
from retrieval_controller import ControllerDecision, choose_action


SYSTEM_PROMPT = """You answer questions about the supplied research paper.
Use only the context provided below. If the answer is not in the context, say
that you do not have enough information in the paper. Do not invent facts.
Cite supporting context chunks using [Source chunk N]. If the context assessment
is partial, answer the supported part and briefly disclose what is missing.
Keep the answer concise and mention relevant numbers when available."""

QUERY_REWRITE_PROMPT = """You rewrite user questions for retrieval from one research paper.
Convert the question into one concise, standalone search query.
Resolve conversational references when possible, preserve model names, metrics,
numbers, and technical terms, and do not answer the question.
Return only the rewritten query, with no explanation."""

QUERY_DECOMPOSE_PROMPT = """You decompose a multi-part question about one research paper.
Return 2 or 3 concise standalone retrieval questions, one per line.
Preserve model names, metrics, numbers, and technical terms. Do not answer them.
Return only the questions, with no numbering or explanation."""

QUERY_EXPAND_PROMPT = """You create alternative search queries for one research paper question.
Return 2 or 3 concise queries, one per line. Use different terminology while
preserving model names, metrics, numbers, and technical terms. Do not answer.
Return only the queries, with no numbering or explanation."""

MAX_CORRECTIVE_ATTEMPTS = 2


@lru_cache(maxsize=1)
def create_llm() -> ChatOpenAI:
    """Create a Gemini chat model through OpenRouter."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise RuntimeError("Set OPENROUTER_API_KEY in .env before running the RAG")

    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        temperature=0,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_SITE_NAME", "rag-paper"),
        },
    )


def _history_text(history: Sequence[object] | None) -> str:
    if not history:
        return ""

    messages: list[str] = []
    for item in history[-6:]:
        if isinstance(item, dict):
            role = str(item.get("role", "user"))
            content = item.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            messages.append(f"{role}: {content}")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            messages.extend((f"user: {item[0]}", f"assistant: {item[1]}"))

    return "\n".join(messages)


def rewrite_query(question: str, history: Sequence[object] | None = None) -> str:
    """Rewrite a user question into a standalone retrieval query."""
    if not question.strip():
        raise ValueError("question must not be empty")

    history_context = _history_text(history)
    human_message = (
        f"Conversation history:\n{history_context}\n\nCurrent question: {question}"
        if history_context
        else question
    )
    with trace_span("query_rewrite", has_history=bool(history)):
        response = create_llm().invoke(
            [
                ("system", QUERY_REWRITE_PROMPT),
                ("human", human_message),
            ]
        )
    rewritten = str(response.content).strip()
    trace_event("query_rewritten", query=rewritten)
    return rewritten or question.strip()


def decompose_query(question: str, history: Sequence[object] | None = None) -> list[str]:
    """Split a multi-part question into standalone retrieval queries."""
    history_context = _history_text(history)
    human_message = (
        f"Conversation history:\n{history_context}\n\nCurrent question: {question}"
        if history_context
        else question
    )
    with trace_span("query_decompose", has_history=bool(history)):
        response = create_llm().invoke(
            [("system", QUERY_DECOMPOSE_PROMPT), ("human", human_message)]
        )
    queries = [line.strip(" -\t") for line in str(response.content).splitlines()]
    queries = [query for query in queries if query]
    selected = queries[:3] or [question.strip()]
    trace_event("query_decomposed", subquery_count=len(selected), subqueries=selected)
    return selected


def expand_query(question: str, history: Sequence[object] | None = None) -> list[str]:
    """Generate alternative retrieval queries for corrective retrieval."""
    history_context = _history_text(history)
    human_message = (
        f"Conversation history:\n{history_context}\n\nCurrent question: {question}"
        if history_context
        else question
    )
    with trace_span("query_expand", has_history=bool(history)):
        response = create_llm().invoke(
            [("system", QUERY_EXPAND_PROMPT), ("human", human_message)]
        )
    queries = [line.strip(" -\t") for line in str(response.content).splitlines()]
    queries = [query for query in queries if query]
    selected = list(dict.fromkeys(queries[:3] or [question.strip()]))
    trace_event("query_expanded", query_count=len(selected), queries=selected)
    return selected


def _merge_results(
    question: str,
    query_results: list[list[tuple[object, float]]],
    k: int,
) -> list[tuple[object, float]]:
    candidates: dict[str, tuple[object, float]] = {}
    for results in query_results:
        for document, score in results:
            key = document.page_content
            if key not in candidates or score > candidates[key][1]:
                candidates[key] = (document, score)
    reranked = rerank(question, list(candidates.values()), k=len(candidates))
    merged = deduplicate_evidence(reranked, k=k)
    trace_event(
        "evidence_merged",
        input_result_count=sum(len(results) for results in query_results),
        unique_candidate_count=len(candidates),
        output_result_count=len(merged),
    )
    return merged


def _ensure_citation(answer: str, source_count: int) -> str:
    if re.search(r"\[Source chunk \d+\]", answer):
        return answer
    sources = ", ".join(f"[Source chunk {number}]" for number in range(1, source_count + 1))
    return f"{answer}\n\nSources: {sources}" if sources else answer


def _format_context(results: list[tuple[object, float]]) -> str:
    formatted: list[str] = []
    for number, (document, _) in enumerate(results, start=1):
        section_path = " > ".join(document.metadata.get("section_path", []))
        chunk_id = document.metadata.get("chunk_id", f"chunk-{number}")
        source_label = f"[Source chunk {number}; {chunk_id}; section: {section_path}]"
        formatted.append(f"{source_label}\n{document.page_content}")
    return "\n\n---\n\n".join(formatted)


def _retrieve_for_mode(
    question: str,
    mode: str,
    k: int,
    index_path: str | Path,
    history: Sequence[object] | None,
) -> list[tuple[object, float]]:
    if mode == "direct":
        return retrieve(question.strip(), k=k, index_path=index_path)
    if mode == "rewrite":
        return retrieve(rewrite_query(question, history=history), k=k, index_path=index_path)
    if mode == "decompose":
        queries = decompose_query(question, history=history)
    elif mode == "expand":
        queries = expand_query(question, history=history)
    else:
        raise ValueError(f"unknown retrieval mode: {mode}")

    return _merge_results(
        question,
        [retrieve(query, k=k, index_path=index_path) for query in queries],
        k=k,
    )


def _abstention_message(evaluation: ContextEvaluation) -> str:
    missing = ", ".join(evaluation.missing_aspects) or "the requested details"
    return f"I do not have enough reliable evidence in the paper to answer this fully. Missing: {missing}."


def _answer_question(
    question: str,
    k: int = 4,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    retrieval_query: str | None = None,
    history: Sequence[object] | None = None,
) -> str:
    """Retrieve paper context and generate a grounded answer."""
    current_mode = analyze_query(question, history=history) if retrieval_query is None else "direct"
    trace_event("retrieval_route_selected", mode=current_mode)
    results = (
        retrieve(retrieval_query, k=k, index_path=index_path)
        if retrieval_query is not None
        else _retrieve_for_mode(question, current_mode, k, index_path, history)
    )

    context_evaluation: ContextEvaluation
    decision: ControllerDecision
    for attempt in range(MAX_CORRECTIVE_ATTEMPTS + 1):
        context_evaluation = evaluate_context(question, results, llm=create_llm())
        decision = choose_action(context_evaluation, current_mode, attempt)
        trace_event(
            "retrieval_controller_decision",
            action=decision.action,
            reason=decision.reason,
            attempt=attempt,
            mode=current_mode,
            status=context_evaluation.status,
            confidence=context_evaluation.confidence,
        )
        if decision.action == "ANSWER":
            break
        if decision.action == "ABSTAIN":
            trace_event("answer_abstained", reason=decision.reason)
            return _abstention_message(context_evaluation)
        current_mode = "direct" if decision.action == "RETRY_DIRECT" else decision.action.lower()
        results = _retrieve_for_mode(question, current_mode, k, index_path, history)

    context = _format_context(results)

    trace_event(
        "context_selected",
        chunk_count=len(results),
        top_score=results[0][1] if results else None,
    )
    with trace_span("answer_generation", chunk_count=len(results)):
        response = create_llm().invoke(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    "Context sufficiency assessment:\n"
                    f"{context_evaluation.to_dict()}\n\n"
                    f"Context from the paper:\n\n{context}\n\nQuestion: {question}",
                ),
            ]
        )
    return _ensure_citation(str(response.content), len(results))


def answer_question(
    question: str,
    k: int = 4,
    index_path: str | Path = DEFAULT_INDEX_PATH,
    retrieval_query: str | None = None,
    history: Sequence[object] | None = None,
) -> str:
    """Trace and answer one grounded question."""
    with trace_request(question):
        return _answer_question(
            question,
            k=k,
            index_path=index_path,
            retrieval_query=retrieval_query,
            history=history,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ask questions using naive RAG")
    parser.add_argument("question", help="Question about the research paper")
    parser.add_argument("-k", type=int, default=4, help="Number of chunks to retrieve")
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX_PATH)
    args = parser.parse_args()

    print(answer_question(args.question, k=args.k, index_path=args.index_path))
