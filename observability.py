import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator


_TRACE_ID: ContextVar[str | None] = ContextVar("rag_trace_id", default=None)
_WRITE_LOCK = threading.Lock()


def current_trace_id() -> str | None:
    return _TRACE_ID.get()


def trace_event(event: str, **fields: object) -> None:
    """Write one structured trace event when RAG_TRACE_PATH is configured."""
    trace_path = os.getenv("RAG_TRACE_PATH")
    if not trace_path:
        return

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": event,
        "trace_id": current_trace_id(),
        **fields,
    }
    destination = Path(trace_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _WRITE_LOCK:
        with destination.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str) + "\n")


@contextmanager
def trace_request(question: str) -> Iterator[str]:
    trace_id = uuid.uuid4().hex
    token = _TRACE_ID.set(trace_id)
    started = time.perf_counter()
    trace_event("request_started", question=question)
    try:
        yield trace_id
    except Exception as error:
        trace_event(
            "request_failed",
            error_type=type(error).__name__,
            error=str(error),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        raise
    else:
        trace_event(
            "request_finished",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    finally:
        _TRACE_ID.reset(token)


@contextmanager
def trace_span(name: str, **fields: object) -> Iterator[None]:
    started = time.perf_counter()
    trace_event(f"{name}_started", **fields)
    try:
        yield
    except Exception as error:
        trace_event(
            f"{name}_failed",
            error_type=type(error).__name__,
            error=str(error),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        raise
    else:
        trace_event(
            f"{name}_finished",
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
