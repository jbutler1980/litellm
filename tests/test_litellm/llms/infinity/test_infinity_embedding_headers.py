import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

import litellm


@pytest.mark.asyncio
async def test_infinity_forwards_only_proxy_authorized_headers():
    observed_headers: list[dict[str, str]] = []

    class RecordingHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            observed_headers.append(
                {name.lower(): value for name, value in self.headers.items()}
            )
            body = json.dumps(
                {
                    "object": "list",
                    "model": "recording-model",
                    "data": [
                        {
                            "object": "embedding",
                            "index": 0,
                            "embedding": [0.1, 0.2],
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api_base = f"http://127.0.0.1:{server.server_port}"
    nonce = "A" * 43

    try:
        response = await litellm.aembedding(
            model="infinity/recording-model",
            input=["structural fixture"],
            api_base=api_base,
            api_key="backend-only-key",
            headers={
                "X-BIP-Embedding-Nonce": nonce,
                "X-Caller-Arbitrary": "caller-only",
                "Cookie": "caller-only",
                "Authorization": "Bearer caller-only",
            },
            _proxy_forward_headers=True,
        )
        ordinary_response = await litellm.aembedding(
            model="infinity/recording-model",
            input=["structural fixture"],
            api_base=api_base,
            api_key="backend-only-key",
            extra_headers={"X-BIP-Embedding-Nonce": "B" * 43},
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    assert response.model == "recording-model"
    assert ordinary_response.model == "recording-model"
    assert len(observed_headers) == 2
    assert observed_headers[0]["x-bip-embedding-nonce"] == nonce
    assert observed_headers[0]["authorization"] == "Bearer backend-only-key"
    assert "cookie" not in observed_headers[0]
    assert "x-caller-arbitrary" not in observed_headers[0]
    assert "x-bip-embedding-nonce" not in observed_headers[1]
