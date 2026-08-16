import json
import re
from dataclasses import dataclass
from typing import Literal, Sequence

from langchain_core.documents import Document

from observability import trace_event, trace_span


ContextStatus = Literal["sufficient", "partial", "insufficient", "conflicting"]


@dataclass(frozen=True)
class SupportedAspect:
    aspect: str
    evidence_ids: list[str]


@dataclass(frozen=True)
class ContextEvaluation:
    status: ContextStatus
    supported_aspects: list[SupportedAspect]
    missing_aspects: list[str]
    conflicting_aspects: list[str]
    evidence_ids: list[str]
    confidence: float

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "supported_aspects": [
                {"aspect": item.aspect, "evidence_ids": item.evidence_ids}
                for item in self.supported_aspects
            ],
            "missing_aspects": self.missing_aspects,
            "conflicting_aspects": self.conflicting_aspects,
            "evidence_ids": self.evidence_ids,
            "confidence": self.confidence,
        }


EVALUATOR_SYSTEM_PROMPT = """You evaluate whether retrieved evidence is sufficient to answer a question.
Use only the supplied evidence. Do not answer the question.
Use evidence IDs exactly as provided. Identify supported aspects, missing aspects,
and contradictions between evidence chunks.

Return only valid JSON matching this schema:
{
  "status": "sufficient|partial|insufficient|conflicting",
  "supported_aspects": [{"aspect": "...", "evidence_ids": ["..."]}],
  "missing_aspects": ["..."],
  "conflicting_aspects": ["..."],
  "evidence_ids": ["..."],
  "confidence": 0.0
}

Choose sufficient when all answer-relevant aspects are supported, partial when
some but not all are supported, insufficient when the evidence cannot answer the
question, and conflicting when evidence contains materially incompatible claims."""


def _evidence_id(document: Document, number: int) -> str:
    return str(document.metadata.get("chunk_id", f"chunk-{number}"))


def _format_evidence(results: Sequence[tuple[Document, float]]) -> tuple[str, set[str]]:
    blocks: list[str] = []
    evidence_ids: set[str] = set()
    for number, (document, score) in enumerate(results, start=1):
        evidence_id = _evidence_id(document, number)
        evidence_ids.add(evidence_id)
        section = " > ".join(document.metadata.get("section_path", []))
        blocks.append(
            f"Evidence ID: {evidence_id}\n"
            f"Section: {section}\n"
            f"Reranker score: {score:.4f}\n"
            f"Text: {document.page_content}"
        )
    return "\n\n---\n\n".join(blocks), evidence_ids


def _extract_json(content: str) -> dict:
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("context evaluator response must be a JSON object")
    return parsed


def _parse_evaluation(payload: dict, allowed_ids: set[str]) -> ContextEvaluation:
    statuses = {"sufficient", "partial", "insufficient", "conflicting"}
    status = payload.get("status")
    if status not in statuses:
        raise ValueError(f"invalid context status: {status!r}")

    def strings(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    supported: list[SupportedAspect] = []
    raw_supported = payload.get("supported_aspects", [])
    if isinstance(raw_supported, list):
        for item in raw_supported:
            if not isinstance(item, dict) or not str(item.get("aspect", "")).strip():
                continue
            ids = [item_id for item_id in strings(item.get("evidence_ids")) if item_id in allowed_ids]
            supported.append(SupportedAspect(str(item["aspect"]).strip(), ids))

    evidence_ids = [item_id for item_id in strings(payload.get("evidence_ids")) if item_id in allowed_ids]
    for item in supported:
        evidence_ids.extend(item.evidence_ids)
    evidence_ids = list(dict.fromkeys(evidence_ids))
    confidence = float(payload.get("confidence", 0.0))
    confidence = min(1.0, max(0.0, confidence))
    return ContextEvaluation(
        status=status,
        supported_aspects=supported,
        missing_aspects=strings(payload.get("missing_aspects")),
        conflicting_aspects=strings(payload.get("conflicting_aspects")),
        evidence_ids=evidence_ids,
        confidence=confidence,
    )


def evaluate_context(
    question: str,
    results: Sequence[tuple[Document, float]],
    llm,
) -> ContextEvaluation:
    """Evaluate evidence sufficiency without generating an answer."""
    evidence, allowed_ids = _format_evidence(results)
    if not results:
        evaluation = ContextEvaluation("insufficient", [], [question], [], [], 1.0)
        trace_event("context_evaluated", **evaluation.to_dict())
        return evaluation

    prompt = f"Question:\n{question}\n\nRetrieved evidence:\n{evidence}"
    try:
        with trace_span("context_sufficiency_evaluation", evidence_count=len(results)):
            response = llm.invoke(
                [("system", EVALUATOR_SYSTEM_PROMPT), ("human", prompt)]
            )
        payload = _extract_json(str(response.content))
        evaluation = _parse_evaluation(payload, allowed_ids)
    except Exception as error:
        trace_event(
            "context_evaluation_failed",
            error_type=type(error).__name__,
            error=str(error),
        )
        evaluation = ContextEvaluation(
            "insufficient",
            [],
            ["Context evaluator failed to produce a valid assessment"],
            [],
            [],
            0.0,
        )
    trace_event("context_evaluated", **evaluation.to_dict())
    return evaluation
