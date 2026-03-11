"""Error classification for managed agent execution."""

from __future__ import annotations


class AgentTickError(Exception):
    """Base class for agent tick errors."""

    retryable: bool = False
    needs_human: bool = False


class RetryableError(AgentTickError):
    """Transient error that should be retried with backoff."""

    retryable = True


class FatalError(AgentTickError):
    """Permanent error that requires user intervention."""

    retryable = False


class EscalateError(AgentTickError):
    """Agent is uncertain and needs human input."""

    retryable = False
    needs_human = True


_RETRYABLE_PATTERNS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "temporary",
    "unavailable",
    "503",
    "429",
    "502",
)

_FATAL_PATTERNS = (
    "permission",
    "access denied",
    "unauthorized",
    "forbidden",
    "invalid api key",
    "invalid_api_key",
    "not found",
    "401",
    "403",
)


def classify_error(exc: Exception) -> AgentTickError:
    """Classify an arbitrary exception into a RetryableError or FatalError."""
    if isinstance(exc, AgentTickError):
        return exc

    msg = str(exc).lower()

    # Check fatal patterns first (more specific)
    if isinstance(exc, PermissionError):
        return FatalError(str(exc))
    for pattern in _FATAL_PATTERNS:
        if pattern in msg:
            return FatalError(str(exc))

    # Check retryable patterns
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return RetryableError(str(exc))
    for pattern in _RETRYABLE_PATTERNS:
        if pattern in msg:
            return RetryableError(str(exc))

    # Default: assume retryable (better to retry than to give up)
    return RetryableError(str(exc))


def retry_delay(attempt: int) -> int:
    """Exponential backoff delay in seconds: min(10 * 2^attempt, 300)."""
    return min(10 * (2**attempt), 300)
