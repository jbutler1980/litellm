"""BIP embedding nonce forwarding for the Infinity (and, unchanged, Voyage) embedding adapters.

The proxy-side callback (``bip_nonce_forwarding``) authorizes exactly one provider header,
``X-BIP-Embedding-Nonce``, and marks the request with the private ``_proxy_forward_headers``
flag. ``litellm.embedding`` must forward that single header to Infinity, and nothing else the
caller supplied, and must not change what Voyage receives.
"""

import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Final

import pytest

import litellm

NONCE: Final = "A" * 43
OTHER_NONCE: Final = "B" * 43
NONCE_HEADER: Final = "x-bip-embedding-nonce"


class RecordingEmbeddingServer:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []  # mutable-ok: the handler thread appends what the backend received
        recorder: Final = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length: Final = int(self.headers.get("Content-Length", "0"))
                body: Final = json.loads(self.rfile.read(length) or b"{}")
                recorder.requests.append(
                    {
                        "path": self.path,
                        "headers": {name.lower(): value for name, value in self.headers.items()},
                        "body": body,
                    }
                )
                payload: Final = json.dumps(
                    {
                        "object": "list",
                        "model": "recording-model",
                        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: object) -> None:
                return

        self._server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread: Final = Thread(target=self._server.serve_forever, daemon=True)

    @property
    def api_base(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    def __enter__(self) -> "RecordingEmbeddingServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)
        self._server.server_close()


@pytest.fixture
def backend() -> Iterator[RecordingEmbeddingServer]:
    with RecordingEmbeddingServer() as server:
        yield server


async def test_infinity_forwards_only_the_proxy_authorized_nonce_header(backend: RecordingEmbeddingServer) -> None:
    response = await litellm.aembedding(
        model="infinity/recording-model",
        input=["structural fixture"],
        api_base=backend.api_base,
        api_key="backend-only-key",
        headers={
            "X-BIP-Embedding-Nonce": NONCE,
            "X-Caller-Arbitrary": "caller-only",
            "Cookie": "caller-only",
            "Authorization": "Bearer caller-only",
        },
        _proxy_forward_headers=True,
    )

    assert response.model == "recording-model"
    assert len(backend.requests) == 1, backend.requests
    seen = backend.requests[0]["headers"]
    assert seen[NONCE_HEADER] == NONCE, seen
    assert seen["authorization"] == "Bearer backend-only-key", seen
    assert "cookie" not in seen, seen
    assert "x-caller-arbitrary" not in seen, seen
    assert "_proxy_forward_headers" not in backend.requests[0]["body"], backend.requests[0]["body"]


async def test_infinity_ignores_a_caller_supplied_nonce_without_the_proxy_marker(
    backend: RecordingEmbeddingServer,
) -> None:
    await litellm.aembedding(
        model="infinity/recording-model",
        input=["structural fixture"],
        api_base=backend.api_base,
        api_key="backend-only-key",
        extra_headers={"X-BIP-Embedding-Nonce": OTHER_NONCE},
    )
    await litellm.aembedding(
        model="infinity/recording-model",
        input=["structural fixture"],
        api_base=backend.api_base,
        api_key="backend-only-key",
        headers={"X-BIP-Embedding-Nonce": OTHER_NONCE},
        _proxy_forward_headers="yes",
    )

    assert len(backend.requests) == 2, backend.requests
    assert all(NONCE_HEADER not in request["headers"] for request in backend.requests), backend.requests


async def test_voyage_never_receives_the_nonce_even_with_the_proxy_marker(backend: RecordingEmbeddingServer) -> None:
    await litellm.aembedding(
        model="voyage/recording-model",
        input=["structural fixture"],
        api_base=backend.api_base,
        api_key="backend-only-key",
        headers={"X-BIP-Embedding-Nonce": NONCE, "X-Caller-Arbitrary": "caller-only"},
        _proxy_forward_headers=True,
    )

    assert len(backend.requests) == 1, backend.requests
    seen = backend.requests[0]["headers"]
    assert NONCE_HEADER not in seen, seen
    assert "x-caller-arbitrary" not in seen, seen
