"""A BIP-attested embedding request is sent to the backend exactly once.

``AsyncHTTPHandler.post`` re-sends a POST on a new connection when the first attempt dies with
``httpx.RemoteProtocolError`` or ``httpx.ConnectError``. After the backend has read the request
that is a replay of work the backend may already have done, and BIP accounts for each attempt at
its own boundary. A request carrying a well-formed ``X-BIP-Embedding-Nonce`` must therefore
surface the failure instead; every other request keeps the ordinary resend.
"""

import asyncio
import contextlib
import socket
import struct
from collections.abc import AsyncIterator
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

NONCE: Final = "A" * 43
NONCE_HEADER: Final = "X-BIP-Embedding-Nonce"


class DisconnectAfterReadServer:
    """Reads one complete HTTP request, records it, then drops the connection without a response."""

    def __init__(self, *, reset: bool) -> None:
        self.reset: Final = reset
        self.requests: list[dict[str, str]] = []  # mutable-ok: the handler appends each request the backend fully read
        self._server: asyncio.AbstractServer | None = None

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        headers = {
            name.strip().lower(): value.strip()
            for name, _, value in (line.partition(":") for line in head.decode("latin-1").split("\r\n")[1:])
        }
        await reader.readexactly(int(headers.get("content-length", "0")))
        self.requests.append(headers)
        if self.reset:
            raw: Final = writer.get_extra_info("socket")
            raw.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        writer.close()

    async def start(self) -> str:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        return f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}/embeddings"

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()


@pytest.fixture(params=["aiohttp", "httpx"])
def transport(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", request.param == "httpx")
    return request.param


disconnects = pytest.mark.parametrize("disconnect", ["clean_close", "reset"])


@contextlib.asynccontextmanager
async def _backend(disconnect: str) -> AsyncIterator[tuple[DisconnectAfterReadServer, str]]:
    server = DisconnectAfterReadServer(reset=disconnect == "reset")
    url = await server.start()
    try:
        yield server, url
    finally:
        await server.stop()


async def _post(url: str, headers: dict[str, str]) -> None:
    handler = AsyncHTTPHandler(timeout=httpx.Timeout(5.0))
    try:
        with pytest.raises(httpx.TransportError):
            await handler.post(url, json={"input": ["x"]}, headers=headers)
    finally:
        await handler.close()


@disconnects
async def test_attested_post_is_sent_once_when_the_backend_drops_the_connection_after_reading_it(
    transport: str, disconnect: str
) -> None:
    async with _backend(disconnect) as (server, url):
        await _post(url, {NONCE_HEADER: NONCE, "Content-Type": "application/json"})

    assert len(server.requests) == 1, (
        f"transport={transport} disconnect={disconnect}: backend read {len(server.requests)} requests"
    )
    assert server.requests[0][NONCE_HEADER.lower()] == NONCE


@disconnects
@pytest.mark.parametrize("nonce", ["A" * 42, "A" * 44, "A" * 42 + "!", ""])
async def test_a_malformed_nonce_does_not_disable_the_ordinary_resend(
    nonce: str, transport: str, disconnect: str
) -> None:
    async with _backend(disconnect) as (ordinary_server, ordinary_url):
        await _post(ordinary_url, {"Content-Type": "application/json"})
    async with _backend(disconnect) as (server, url):
        await _post(url, {NONCE_HEADER: nonce, "Content-Type": "application/json"})

    assert len(server.requests) == len(ordinary_server.requests), (
        f"transport={transport} disconnect={disconnect}: a malformed nonce changed the resend behaviour "
        f"({len(server.requests)} vs {len(ordinary_server.requests)} requests)"
    )


@disconnects
async def test_the_ordinary_resend_is_kept_for_requests_without_a_nonce(transport: str, disconnect: str) -> None:
    """The transport only raises a resendable error (RemoteProtocolError/ConnectError) in two of the four cases."""
    resendable = (transport, disconnect) in {("httpx", "clean_close"), ("aiohttp", "reset")}

    async with _backend(disconnect) as (server, url):
        await _post(url, {"Content-Type": "application/json"})

    assert len(server.requests) == (2 if resendable else 1), (
        f"transport={transport} disconnect={disconnect}: backend read {len(server.requests)} requests"
    )


@disconnects
async def test_an_attested_embedding_call_through_the_infinity_adapter_is_sent_once(
    transport: str, disconnect: str
) -> None:
    async with _backend(disconnect) as (server, url):
        with pytest.raises(Exception):
            await litellm.aembedding(
                model="infinity/recording-model",
                input=["structural fixture"],
                api_base=url.removesuffix("/embeddings"),
                api_key="backend-only-key",
                headers={NONCE_HEADER: NONCE},
                _proxy_forward_headers=True,
                max_retries=0,
                num_retries=0,
            )

    assert len(server.requests) == 1, (
        f"transport={transport} disconnect={disconnect}: backend read {len(server.requests)} requests"
    )
    assert server.requests[0][NONCE_HEADER.lower()] == NONCE
