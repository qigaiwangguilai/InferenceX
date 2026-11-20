# SPDX-License-Identifier: Apache-2.0

"""HTTP transport error handling for streaming benchmark requests."""

import asyncio
import math
import os
from functools import partial

import aiohttp
from aiohttp.client_proto import ResponseHandler


def request_timeout() -> aiohttp.ClientTimeout:
    """Read finite request deadlines without shortening long-context runs by default."""
    def seconds(name: str, default: str | None = None) -> float | None:
        """Validate an optional timeout supplied through the environment."""
        raw = os.environ.get(name, default)
        if raw is None:
            return None
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a finite positive number of seconds"
            ) from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number of seconds")
        return value

    return aiohttp.ClientTimeout(
        total=seconds("BENCH_CLIENT_TOTAL_TIMEOUT_S", str(6 * 60 * 60)),
        sock_read=seconds("BENCH_CLIENT_SOCK_READ_TIMEOUT_S"),
    )


class _BenchmarkResponseHandler(ResponseHandler):
    """Wake body readers when aiohttp reports a protocol error or loses transport."""

    def set_exception(self, exc: BaseException, *args, **kwargs) -> None:
        """Propagate protocol failures to the separately awaited response body."""
        super().set_exception(exc, *args, **kwargs)
        payload = self._payload
        if payload is not None and not payload.is_eof() and payload.exception() is None:
            payload.set_exception(
                aiohttp.ClientPayloadError(f"Response transport failed: {exc}"), exc
            )

    def connection_lost(self, exc: BaseException | None) -> None:
        """Fail an unfinished body even if the parser's EOF cleanup misses it."""
        payload = self._payload
        try:
            super().connection_lost(exc)
        finally:
            if payload is not None and not payload.is_eof():
                error = payload.exception()
                if error is None:
                    error = aiohttp.ClientPayloadError(
                        "Response connection closed before the body completed"
                    )
                # feed_data() may have woken readline before the parser failed.
                # readline can consume that partial line and install a NEW
                # waiter without checking the exception again. Reapply even
                # an existing error so this late waiter is also woken.
                payload.set_exception(error)


class BenchmarkTCPConnector(aiohttp.TCPConnector):
    """Use a local protocol workaround without globally patching aiohttp."""

    def __init__(self) -> None:
        """Preserve connector defaults and replace only the protocol factory."""
        super().__init__()
        # aiohttp 3.13.5 can leave StreamReader pending after a parser failure.
        # This private integration is isolated here and covered by raw TCP tests.
        self._factory = partial(_BenchmarkResponseHandler, loop=asyncio.get_running_loop())
