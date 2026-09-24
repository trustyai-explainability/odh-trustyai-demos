"""Contract tests for the container-backed TrustyAI client."""

from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO_ROOT))

from common.trustyai_client import TrustyAIClient, compact_request_inputs


class TrustyAIClientTests(unittest.TestCase):
    """Exercise the real HTTP client against a local HTTP server."""

    def test_client_calls_info_upload_and_mmd_without_source_imports(self) -> None:
        """The notebook-facing client uses the service HTTP contract."""
        received: list[tuple[str, dict[str, object] | None]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/info":
                    self.send_error(404)
                    return
                self._send_json(200, {"status": "ready"})

            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                document = json.loads(self.rfile.read(length).decode("utf-8"))
                received.append((self.path, document))
                self._send_json(200, {"status": "success"})

            def _send_json(self, status: int, document: dict[str, object]) -> None:
                encoded = json.dumps(document).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = TrustyAIClient(f"http://127.0.0.1:{server.server_port}")
            self.assertEqual(client.info(), {"status": "ready"})
            self.assertEqual(client.upload({"data_tag": "IMAGE_BASELINE"}), {"status": "success"})
            self.assertEqual(client.compute_mmd({"modelId": "image-mmd-model"}), {"status": "success"})
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        self.assertEqual(
            [path for path, _ in received],
            ["/data/upload", "/metrics/drift/mmd"],
        )

    def test_compact_request_inputs_preserves_shape_and_does_not_mutate_source(self) -> None:
        """Large image transport data becomes small without changing the source document."""
        document = {
            "request": {
                "inputs": [
                    {"shape": [2], "datatype": "BYTES", "data": ["large-a", "large-b"]}
                ]
            },
            "response": {"outputs": [{"data": [0.1, 0.2]}]},
        }

        compacted = compact_request_inputs(document)

        self.assertEqual(compacted["request"]["inputs"][0]["data"], ["", ""])
        self.assertEqual(document["request"]["inputs"][0]["data"], ["large-a", "large-b"])


if __name__ == "__main__":
    unittest.main()
