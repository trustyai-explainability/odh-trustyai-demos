"""Small HTTP client for local and remote TrustyAI demo workflows."""

from __future__ import annotations

import json
import ssl
from copy import deepcopy
from dataclasses import dataclass, field
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener


class TrustyAIClientError(RuntimeError):
    """Raised when the TrustyAI HTTP contract cannot be completed."""


def compact_request_inputs(document: dict[str, object]) -> dict[str, object]:
    """Replace transport-only request data with small placeholders.

    The TrustyAI PVC serializer stores request and response rows together. Raw
    base64 image inputs exceed its default row-size limit, while MMD only uses
    the embedding output columns. Preserve the request schema and row count but
    remove the large transport payload before a local data upload.
    """
    compacted = deepcopy(document)
    request = compacted.get("request", compacted)
    if not isinstance(request, dict):
        return compacted
    inputs = request.get("inputs")
    if not isinstance(inputs, list):
        return compacted
    for item in inputs:
        if isinstance(item, dict) and isinstance(item.get("data"), list):
            item["data"] = [""] * len(item["data"])
    return compacted


class _NoRedirectHandler(HTTPRedirectHandler):
    """Prevent redirects from carrying bearer tokens to another host."""

    def redirect_request(self, *args: object, **kwargs: object) -> None:
        raise TrustyAIClientError("TrustyAI redirects are disabled")


@dataclass(frozen=True)
class TrustyAIClient:
    """Call the TrustyAI API without importing the service implementation."""

    base_url: str
    token: str | None = None
    allowed_hosts: frozenset[str] = field(default_factory=frozenset)
    ca_bundle: str | None = None
    insecure: bool = False
    timeout: float = 60.0

    def __post_init__(self) -> None:
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("TrustyAI URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "TrustyAI URL must not contain credentials, query, or fragment data"
            )

        hostname = parsed.hostname.lower()
        if self.allowed_hosts and hostname not in self.allowed_hosts:
            raise ValueError(f"TrustyAI host {hostname!r} is not in TRUSTYAI_ALLOWED_HOSTS")
        if not self.allowed_hosts and hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Set TRUSTYAI_ALLOWED_HOSTS before using a non-loopback TrustyAI URL")
        if self.ca_bundle and parsed.scheme != "https":
            raise ValueError("TRUSTYAI_CA_BUNDLE requires an HTTPS TrustyAI URL")
        if self.insecure and hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Insecure TrustyAI mode is limited to loopback URLs")

    def info(self) -> dict[str, object]:
        """Return the service model/observation information."""
        return self._request("/info")

    def upload(self, document: dict[str, object]) -> dict[str, object]:
        """Upload one request/response document to TrustyAI."""
        return self._request("/data/upload", document)

    def mmd_definition(self) -> dict[str, object]:
        """Return the MMD endpoint definition."""
        return self._request("/metrics/drift/mmd/definition")

    def compute_mmd(self, document: dict[str, object]) -> dict[str, object]:
        """Compute MMD for the most recently uploaded observation batch."""
        return self._request("/metrics/drift/mmd", document)

    def _request(
        self,
        path: str,
        document: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if not path.startswith("/"):
            raise ValueError("TrustyAI API paths must start with '/'")

        body = None
        headers = {"Accept": "application/json"}
        method = "GET" if document is None else "POST"
        if document is not None:
            body = json.dumps(document, allow_nan=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        request = Request(
            self.base_url.rstrip("/") + path,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self._opener().open(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TrustyAIClientError(
                f"TrustyAI {method} {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (OSError, URLError, TimeoutError) as exc:
            raise TrustyAIClientError(
                f"Unable to reach TrustyAI at {self.base_url}: {exc}"
            ) from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TrustyAIClientError(
                f"TrustyAI {method} {path} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise TrustyAIClientError(
                f"TrustyAI {method} {path} returned a non-object JSON value"
            )
        return payload

    def _opener(self):
        handlers: list[object] = [_NoRedirectHandler()]
        parsed = urlparse(self.base_url)
        if parsed.scheme == "https":
            if self.insecure:
                context = ssl._create_unverified_context()
            else:
                context = ssl.create_default_context(cafile=self.ca_bundle)
            handlers.append(HTTPSHandler(context=context))
        return build_opener(*handlers)
