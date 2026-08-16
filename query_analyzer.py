import re
from typing import Literal, Sequence

from observability import trace_event

QueryMode = Literal["direct", "rewrite", "decompose"]

_CONTEXT_DEPENDENT_PATTERN = re.compile(
    r"\b(?:it|they|them|their|these|those|he|she|his|her)\b"
    r"|\b(?:above|previous|earlier)\b"
    r"|\b(?:what about|other models|another model)\b",
    re.IGNORECASE,
)
_MULTI_PART_PATTERN = re.compile(
    r"\b(?:compare|contrast|difference|differences|rank|respectively)\b"
    r"|\band\s+(?:explain|describe|why|how|what)\b"
    r"|[,;]\s*(?:and|also|plus)\b",
    re.IGNORECASE,
)


def analyze_query(question: str, history: Sequence[object] | None = None) -> QueryMode:
    """Choose direct retrieval, rewriting, or decomposition."""
    if not question.strip():
        raise ValueError("question must not be empty")

    if _MULTI_PART_PATTERN.search(question):
        trace_event("query_analyzed", mode="decompose", question=question)
        return "decompose"

    if history and _CONTEXT_DEPENDENT_PATTERN.search(question):
        trace_event("query_analyzed", mode="rewrite", has_history=True, question=question)
        return "rewrite"

    if _CONTEXT_DEPENDENT_PATTERN.search(question):
        trace_event("query_analyzed", mode="rewrite", has_history=False, question=question)
        return "rewrite"
    trace_event("query_analyzed", mode="direct", has_history=bool(history), question=question)
    return "direct"
