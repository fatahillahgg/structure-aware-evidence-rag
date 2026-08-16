import json
import os
import tempfile
import unittest
from pathlib import Path

from observability import trace_event, trace_request


class ObservabilityTests(unittest.TestCase):
    def test_trace_request_writes_structured_lifecycle_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "traces.jsonl"
            previous_path = os.environ.get("RAG_TRACE_PATH")
            os.environ["RAG_TRACE_PATH"] = str(trace_path)
            try:
                with trace_request("What accuracy did VGG16 achieve?"):
                    trace_event("test_event", value=1)
            finally:
                if previous_path is None:
                    os.environ.pop("RAG_TRACE_PATH", None)
                else:
                    os.environ["RAG_TRACE_PATH"] = previous_path

            records = [json.loads(line) for line in trace_path.read_text().splitlines()]

        self.assertEqual(
            [record["event"] for record in records],
            ["request_started", "test_event", "request_finished"],
        )
        self.assertTrue(records[0]["trace_id"])
        self.assertEqual(records[0]["trace_id"], records[-1]["trace_id"])


if __name__ == "__main__":
    unittest.main()
