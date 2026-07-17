"""Configurable exponential-backoff retry support.

Two entry points are provided:

* :func:`call_with_retry` — imperative helper: ``call_with_retry(fn, a, b)``.
* :func:`retry` — decorator form for wrapping a function definition.

Both share the same backoff calculation and both raise
:class:`~src.common.exceptions.RetryExhaustedError` (chaining the last
underlying exception) once ``max_attempts`` is exhausted.
"""

from __future__ import annotations

import functools
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, TypeVar

from src.common.config import RetrySettings
from src.common.exceptions import PipelineError, RetryExhaustedError
from src.common.logging_utils import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# By default only exceptions explicitly marked retryable=True are retried.
DEFAULT_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (PipelineError,)


@dataclass(frozen=True)
class RetryPolicy:
    """Backoff policy. Mirrors :class:`src.common.config.RetrySettings`."""

    max_attempts: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    multiplier: float = 2.0
    jitter: bool = True

    @classmethod
    def from_settings(cls, settings: RetrySettings) -> "RetryPolicy":
        return cls(
            max_attempts=settings.max_attempts,
            base_delay_seconds=settings.base_delay_seconds,
            max_delay_seconds=settings.max_delay_seconds,
            multiplier=settings.multiplier,
            jitter=settings.jitter,
        )

    def delay_for_attempt(self, attempt: int) -> float:
        """Delay (seconds) to wait before the given retry attempt (1-indexed)."""

        raw = self.base_delay_seconds * (self.multiplier ** (attempt - 1))
        capped = min(self.max_delay_seconds, raw)
        if self.jitter:
            return random.uniform(0, capped)
        return capped


def _is_retryable(exc: BaseException, retryable_exceptions: Iterable[type[BaseException]]) -> bool:
    if isinstance(exc, PipelineError):
        return exc.retryable
    return isinstance(exc, tuple(retryable_exceptions))


def call_with_retry(
    fn: Callable[..., T],
    *args: Any,
    policy: RetryPolicy | None = None,
    retryable_exceptions: Iterable[type[BaseException]] = DEFAULT_RETRYABLE_EXCEPTIONS,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    operation_name: str | None = None,
    **kwargs: Any,
) -> T:
    """Invoke ``fn(*args, **kwargs)``, retrying on retryable failures.

    Args:
        fn: Callable to invoke.
        policy: Backoff configuration. Defaults to ``RetryPolicy()``.
        retryable_exceptions: Non-``PipelineError`` exception types that
            should also be retried. ``PipelineError`` subclasses use their
            own ``retryable`` class attribute instead.
        on_retry: Optional callback ``(attempt, exception, delay)`` invoked
            before each sleep, useful for metrics hooks.
        operation_name: Label used in log messages / the final error.

    Raises:
        RetryExhaustedError: if all attempts fail. The final underlying
            exception is chained via ``__cause__``.
    """

    policy = policy or RetryPolicy()
    name = operation_name or getattr(fn, "__name__", "operation")

    last_exc: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, filtered below
            last_exc = exc
            if not _is_retryable(exc, retryable_exceptions):
                raise
            if attempt >= policy.max_attempts:
                break
            delay = policy.delay_for_attempt(attempt)
            logger.warning(
                "Retrying %s after failure (attempt %d/%d): %s. Sleeping %.2fs",
                name,
                attempt,
                policy.max_attempts,
                exc,
                delay,
            )
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            time.sleep(delay)

    raise RetryExhaustedError(
        f"{name} failed after {policy.max_attempts} attempt(s)",
        cause=last_exc,
    ) from last_exc


def retry(
    policy: RetryPolicy | None = None,
    retryable_exceptions: Iterable[type[BaseException]] = DEFAULT_RETRYABLE_EXCEPTIONS,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator form of :func:`call_with_retry`."""

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return call_with_retry(
                fn,
                *args,
                policy=policy,
                retryable_exceptions=retryable_exceptions,
                on_retry=on_retry,
                operation_name=fn.__name__,
                **kwargs,
            )

        return wrapper

    return decorator