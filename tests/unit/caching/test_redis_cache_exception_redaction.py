"""Redis failures must not copy the failed command, and with it the cached value, into operator logs.

redis-py re-wraps every pipelined failure as ``Command # N (<the whole command, key and value
included>) of pipeline caused error: <reason>``. A 92 KB line of that shape was seen in production
from ``POST /v1/embeddings``. The value argument of these log calls is already bounded; the
EXCEPTION text must be bounded too: class name, at most 120 bytes of head, at most 60 bytes of
tail, and the number of bytes dropped.
"""

import contextlib
import logging
from typing import Final

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.caching.redis_cache import RedisCache

VALUE_MARKER: Final = "TENANT-EMBEDDING-VALUE"
COMMAND_BYTES: Final = 90_000
MAX_LOG_LINE_BYTES: Final = 400


def _pipeline_failure_text(reason: str = "READONLY You can't write against a read only replica.") -> str:
    """The exact shape redis-py's pipeline error annotation produces."""
    half = "0.123456789, " * (COMMAND_BYTES // 26)
    command = f"SET litellm:cache-key [{half}{VALUE_MARKER}, {half}] EX 3600"
    assert len(command) >= COMMAND_BYTES
    return f"Command # 1 ({command}) of pipeline caused error: {reason}"


def _pipeline_failure() -> Exception:
    from redis.exceptions import ReadOnlyError

    return ReadOnlyError(_pipeline_failure_text())


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> RedisCache:
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    with patch("asyncio.get_running_loop", side_effect=RuntimeError("No running event loop")):
        return RedisCache()


def _failing_async_client() -> MagicMock:
    client = MagicMock()
    pipe = MagicMock()
    pipe.execute = AsyncMock(side_effect=_pipeline_failure())
    client.pipeline.return_value.__aenter__ = AsyncMock(return_value=pipe)
    client.pipeline.return_value.__aexit__ = AsyncMock(return_value=False)
    client.ping = AsyncMock(side_effect=_pipeline_failure())
    client.get = AsyncMock(side_effect=_pipeline_failure())
    client.mget = AsyncMock(side_effect=_pipeline_failure())
    return client


def _assert_bounded(records: list[logging.LogRecord], printed: str = "") -> None:
    assert records or printed, "the failure must be reported somewhere, or this test proves nothing"
    for record in records:
        rendered = record.getMessage() + (record.exc_text or "")
        assert len(rendered.encode()) <= MAX_LOG_LINE_BYTES, (record.levelname, len(rendered.encode()), rendered[:200])
        assert VALUE_MARKER not in rendered, rendered[:200]
        assert rendered.count("0.123456789") <= 10, "more of the value than the 120-byte head allows"
    for line in printed.splitlines():
        assert len(line.encode()) <= MAX_LOG_LINE_BYTES, (len(line.encode()), line[:200])
        assert VALUE_MARKER not in line, line[:200]
        assert line.count("0.123456789") <= 10, "more of the value than the 120-byte head allows"


@pytest.mark.asyncio
async def test_pipeline_write_failure_logs_a_bounded_exception_without_the_command(
    cache: RedisCache, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        patch.object(cache, "init_async_client", return_value=_failing_async_client()),
        caplog.at_level(logging.DEBUG, logger="LiteLLM"),
    ):
        await cache.async_set_cache_pipeline([("litellm:cache-key", [0.1] * 4)], ttl=60)

    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(error_records) == 1, [r.getMessage()[:120] for r in caplog.records]
    message = error_records[0].getMessage()
    assert "ReadOnlyError" in message, message
    assert "READONLY You can't write against a read only replica." in message, message
    assert "[truncated" in message and "bytes]" in message, message
    _assert_bounded(caplog.records)


@pytest.mark.asyncio
async def test_pipeline_with_ttls_write_failure_logs_a_bounded_exception(
    cache: RedisCache, caplog: pytest.LogCaptureFixture
) -> None:
    with (
        patch.object(cache, "init_async_client", return_value=_failing_async_client()),
        caplog.at_level(logging.DEBUG, logger="LiteLLM"),
    ):
        await cache.async_set_cache_pipeline_with_ttls([("litellm:cache-key", [0.1] * 4, 60.0)])

    assert [r for r in caplog.records if r.levelno >= logging.ERROR], "the failure must still be logged"
    _assert_bounded(caplog.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["ping", "async_batch_get_cache", "async_get_cache"])
async def test_async_failures_log_a_bounded_exception(
    method: str, cache: RedisCache, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import litellm._logging as logging_module

    monkeypatch.setattr(logging_module, "set_verbose", True)
    arguments = {
        "ping": (),
        "async_batch_get_cache": (["litellm:cache-key"],),
        "async_get_cache": ("litellm:cache-key",),
    }
    with (
        patch.object(cache, "init_async_client", return_value=_failing_async_client()),
        patch.object(cache, "_async_commands", return_value=_failing_async_client()),
        caplog.at_level(logging.DEBUG, logger="LiteLLM"),
        contextlib.suppress(Exception),
    ):
        await getattr(cache, method)(*arguments[method])

    _assert_bounded(caplog.records, capsys.readouterr().out)


@pytest.mark.parametrize("method", ["sync_ping", "set_cache", "get_cache", "batch_get_cache"])
def test_sync_failures_log_a_bounded_exception(
    method: str, cache: RedisCache, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    import litellm._logging as logging_module

    monkeypatch.setattr(logging_module, "set_verbose", True)
    failure = _pipeline_failure()
    client = MagicMock()
    for name in ("ping", "set", "get", "mget"):
        getattr(client, name).side_effect = failure
    cache.redis_client = client
    arguments = {
        "sync_ping": (),
        "set_cache": ("litellm:cache-key", [0.1] * 4),
        "get_cache": ("litellm:cache-key",),
        "batch_get_cache": (["litellm:cache-key"],),
    }
    with caplog.at_level(logging.DEBUG, logger="LiteLLM"), contextlib.suppress(Exception):
        getattr(cache, method)(*arguments[method])

    _assert_bounded(caplog.records, capsys.readouterr().out)


def test_the_health_ping_setup_logs_a_bounded_exception(caplog: pytest.LogCaptureFixture) -> None:
    client = MagicMock()
    client.ping.side_effect = _pipeline_failure()
    with (
        patch("litellm._redis.get_redis_client", return_value=client),
        patch("asyncio.get_running_loop", side_effect=RuntimeError("no running event loop")),
        caplog.at_level(logging.DEBUG, logger="LiteLLM"),
        contextlib.suppress(Exception),
    ):
        RedisCache(host="127.0.0.1", port=6379)

    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "the sync health ping failure must still be logged"
    for record in errors:
        extra_error = str(getattr(record, "error", ""))
        assert len(extra_error.encode()) <= MAX_LOG_LINE_BYTES, len(extra_error.encode())
    _assert_bounded(caplog.records)


def test_short_exception_text_is_kept_whole_with_its_class_name() -> None:
    from litellm.caching.redis_cache import _bounded_exception_text

    assert _bounded_exception_text(OSError("redis unavailable")) == "OSError: redis unavailable"


def test_head_and_tail_are_cut_on_bytes_and_account_for_the_dropped_bytes() -> None:
    from litellm.caching.redis_cache import _bounded_exception_text

    text = "H" * 120 + "m" * 1000 + "T" * 60
    rendered = _bounded_exception_text(ValueError(text))

    assert rendered == f"ValueError: {'H' * 120} …[truncated 1000 bytes]… {'T' * 60}", rendered
    astral = _bounded_exception_text(ValueError("\U0001f600" * 200))
    assert len(astral.encode()) < 400, len(astral.encode())


def test_an_exception_that_cannot_be_rendered_does_not_raise() -> None:
    from litellm.caching.redis_cache import _bounded_exception_text

    class Hostile(Exception):
        def __str__(self) -> str:
            raise KeyboardInterrupt

    assert _bounded_exception_text(Hostile()) == "UnrenderableException: <exception could not be rendered>"
