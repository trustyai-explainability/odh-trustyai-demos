"""Minimal stateless KServe V2 predictor for the image drift demo."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import re
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

import numpy as np
from PIL import Image

DEMO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DEMO_ROOT not in sys.path:
    sys.path.insert(0, DEMO_ROOT)

from common.image_features import BATCH_SIZE, EMBEDDING_DIM, embed_image_batch

INFER_PATH = re.compile(r"^/v2/models/([^/]+)/infer$")
MAX_REQUEST_BYTES = 32 * 1024 * 1024
REQUEST_SLOTS = threading.BoundedSemaphore(2)


class RequestValidationError(ValueError):
    """Raised when a request is not a supported KServe V2 image request."""


def _positive_shape(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RequestValidationError("input shape must contain one positive integer")
    return value


def _decode_input(payload: dict[str, object]) -> list[bytes]:
    inputs = payload.get("inputs")
    if not isinstance(inputs, list) or len(inputs) != 1:
        raise RequestValidationError("payload must contain exactly one input")

    image_input = inputs[0]
    if not isinstance(image_input, dict):
        raise RequestValidationError("image input must be an object")
    if image_input.get("name") != "image":
        raise RequestValidationError("input name must be 'image'")
    if image_input.get("datatype") != "BYTES":
        raise RequestValidationError("image input datatype must be BYTES")

    shape = image_input.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        raise RequestValidationError("image input shape must be rank 1: [N]")
    expected_count = _positive_shape(shape[0])
    if expected_count > BATCH_SIZE:
        raise RequestValidationError(f"image input batch cannot exceed {BATCH_SIZE}")

    encoded_values = image_input.get("data")
    if not isinstance(encoded_values, list) or len(encoded_values) != expected_count:
        raise RequestValidationError(
            f"image input data must contain {expected_count} base64 values"
        )

    decoded_values: list[bytes] = []
    for index, encoded in enumerate(encoded_values):
        if not isinstance(encoded, str):
            raise RequestValidationError(f"image data at index {index} must be a string")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise RequestValidationError(
                f"image data at index {index} is not valid base64"
            ) from exc
        if not decoded:
            raise RequestValidationError(f"image data at index {index} is empty")
        decoded_values.append(decoded)
    return decoded_values


def predict_request(payload: dict[str, object], *, model_name: str) -> dict[str, object]:
    """Validate a KServe V2 request and return one FP32 embedding output."""
    if not isinstance(payload, dict):
        raise RequestValidationError("request body must be a JSON object")
    if not isinstance(model_name, str) or not model_name:
        raise RequestValidationError("model name must be non-empty")

    image_bytes = _decode_input(payload)
    embedding = embed_image_batch(image_bytes)
    request_id = payload.get("id", "response")
    if not isinstance(request_id, str):
        raise RequestValidationError("request id must be a string when provided")

    return {
        "model_name": model_name,
        "id": request_id,
        "outputs": [
            {
                "name": "embedding",
                "shape": [len(image_bytes), EMBEDDING_DIM],
                "datatype": "FP32",
                "data": embedding.reshape(-1).astype(float).tolist(),
            }
        ],
    }


class PredictorServer(ThreadingHTTPServer):
    """HTTP server carrying the configured model name."""

    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], model_name: str) -> None:
        super().__init__(address, PredictorHandler)
        self.model_name = model_name


class PredictorHandler(BaseHTTPRequestHandler):
    """Serve KServe V2 health and inference endpoints."""

    server: PredictorServer

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(30)

    def _send_json(self, status: HTTPStatus, body: dict[str, object]) -> None:
        encoded = json.dumps(body, allow_nan=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/v2/health/live":
            self._send_json(HTTPStatus.OK, {"status": "live"})
        elif path == "/v2/health/ready":
            self._send_json(HTTPStatus.OK, {"status": "ready"})
        else:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        match = INFER_PATH.fullmatch(path)
        if match is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "endpoint not found"})
            return

        requested_model = unquote(match.group(1))
        if requested_model != self.server.model_name:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "model not found"})
            return

        if not REQUEST_SLOTS.acquire(blocking=False):
            self._send_json(HTTPStatus.TOO_MANY_REQUESTS, {"error": "predictor is busy"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise RequestValidationError("request body is empty")
            if content_length > MAX_REQUEST_BYTES:
                raise RequestValidationError(
                    f"request body exceeds {MAX_REQUEST_BYTES} bytes"
                )
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
            response = predict_request(payload, model_name=self.server.model_name)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc}"})
        except RequestValidationError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except (OSError, ValueError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        else:
            self._send_json(HTTPStatus.OK, response)
        finally:
            REQUEST_SLOTS.release()


def _self_test() -> None:
    """Check the local V2 contract using one synthetic in-memory PNG."""
    image = Image.new("RGB", (32, 24), (32, 128, 224))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    response = predict_request(
        {
            "id": "self-test",
            "inputs": [
                {
                    "name": "image",
                    "shape": [1],
                    "datatype": "BYTES",
                    "data": [encoded],
                }
            ],
        },
        model_name="image-mmd-model",
    )
    output = response["outputs"][0]
    if output["shape"] != [1, EMBEDDING_DIM] or output["datatype"] != "FP32":
        raise RuntimeError("predictor self-test returned an invalid output schema")
    if not np.isfinite(np.asarray(output["data"], dtype=np.float32)).all():
        raise RuntimeError("predictor self-test returned non-finite values")
    print("predictor self-test passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default=os.getenv("HOST", "0.0.0.0")  # nosec B104
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME", "image-mmd-model"),
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test()
        return 0

    server = PredictorServer((args.host, args.port), args.model_name)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
