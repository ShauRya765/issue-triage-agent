"""Cost and latency telemetry for LLM calls and graph nodes.

Two things are recorded, and they are deliberately kept apart:

- `LLMCall` -- one model request: tokens in, tokens out, wall time, the model
  that actually served it, and which graph node made it.
- `NodeTiming` -- one node execution's wall time, whether or not it called a
  model. Timing every node is what lets the breakdown be checked rather than
  assumed: measured over real issues, `extract` is ~93% of a run (median 5.3s of
  5.7s) and the `fetch_context` GitHub lookup only ~0.5s, so the model call is
  the latency story. That was worth confirming rather than guessing.

Both are plain dicts so they serialise into the Postgres checkpoint with no
custom encoder, and both are *records*, not decisions -- nothing here feeds
app.policy.

Collection is context-scoped rather than global: `collector()` binds a fresh
list for the duration of a block, and `record_call` appends to whichever
collector is currently bound. The binding is a ContextVar, and a new thread
starts with an empty context, so a collector must be opened in the same thread
that makes the calls it is meant to capture. That is why the eval opens one
inside each worker (eval/replay.py) rather than once around the pool.
"""

import contextvars
import os
from contextlib import contextmanager
from typing import TypedDict

# Anthropic list price for claude-sonnet-5, USD per million tokens, as of
# 2026-06. Overridable from the environment because the rate is a deployment
# fact, not a property of this code: a different model, a negotiated rate or a
# partner platform (Bedrock/Vertex bill separately) all change it, and none of
# those should require a code edit to price correctly.
_DEFAULT_INPUT_USD_PER_MTOK = 2.00
_DEFAULT_OUTPUT_USD_PER_MTOK = 10.00

# Cached input is billed at neither the input rate nor for free. Anthropic
# charges a premium to *write* an entry (1.25x base input on the default
# 5-minute TTL) and a large discount to *read* one (0.1x base input). Defaults
# below are those multipliers applied to the Sonnet 5 input rate; both are
# overridable for the same reason the base rates are -- a different model or a
# partner platform changes them, and a wrong multiplier would silently
# misreport what caching saved, which is the one number the feature exists to
# produce.
_DEFAULT_CACHE_WRITE_USD_PER_MTOK = 2.50
_DEFAULT_CACHE_READ_USD_PER_MTOK = 0.20


class LLMCall(TypedDict):
    node: str
    model: str
    # Uncached input only. Tokens served from or written to the prompt cache are
    # counted in the two fields below and never here, so the three are disjoint
    # and sum to what the request actually read.
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    wall_ms: float


class NodeTiming(TypedDict):
    node: str
    wall_ms: float


_calls: contextvars.ContextVar[list[LLMCall] | None] = contextvars.ContextVar(
    "usage_calls", default=None
)
_node: contextvars.ContextVar[str] = contextvars.ContextVar("usage_node", default="")


def _read_rate(name: str, default: float) -> float:
    """One env-configured USD/Mtok rate, read at call time not import time."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        # A malformed rate must not silently price everything at zero and
        # make the run look free.
        raise ValueError(f"{name} must be a number, got {raw!r}") from None


def rates() -> tuple[float, float]:
    """(input, output) USD per million tokens, read at call time not import time."""
    return (
        _read_rate("COST_PER_MTOK_INPUT", _DEFAULT_INPUT_USD_PER_MTOK),
        _read_rate("COST_PER_MTOK_OUTPUT", _DEFAULT_OUTPUT_USD_PER_MTOK),
    )


def cache_rates() -> tuple[float, float]:
    """(write, read) USD per million tokens for prompt-cached input."""
    return (
        _read_rate("COST_PER_MTOK_CACHE_WRITE", _DEFAULT_CACHE_WRITE_USD_PER_MTOK),
        _read_rate("COST_PER_MTOK_CACHE_READ", _DEFAULT_CACHE_READ_USD_PER_MTOK),
    )


def cost_usd(
    input_tokens: int,
    output_tokens: int,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> float:
    """Price one call. Cache tokens default to zero so old callers are unchanged.

    The three input figures are disjoint: passing the same token in two of them
    would double-bill it. extract_usage is what guarantees that split.
    """
    input_rate, output_rate = rates()
    write_rate, read_rate = cache_rates()
    return (
        input_tokens * input_rate
        + output_tokens * output_rate
        + cache_write_tokens * write_rate
        + cache_read_tokens * read_rate
    ) / 1_000_000


@contextmanager
def collector():
    """Bind a fresh list of LLMCalls for this block, and yield it."""
    calls: list[LLMCall] = []
    token = _calls.set(calls)
    try:
        yield calls
    finally:
        _calls.reset(token)


@contextmanager
def node(name: str):
    """Tag calls made inside this block as belonging to a named graph node."""
    token = _node.set(name)
    try:
        yield
    finally:
        _node.reset(token)


def record_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    wall_ms: float,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
) -> None:
    """Append one call to the active collector, or drop it if none is bound.

    Dropping is deliberate: app.extract is called directly by tests and by
    scripts that have no interest in telemetry, and instrumentation must never
    be the reason a triage run fails.

    The cache figures default to zero so a caller that knows nothing about
    caching records a well-formed row rather than a row missing two keys.
    """
    calls = _calls.get()
    if calls is None:
        return
    calls.append(
        {
            "node": _node.get() or "unknown",
            "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cache_write_tokens": int(cache_write_tokens),
            "cache_read_tokens": int(cache_read_tokens),
            "wall_ms": round(wall_ms, 1),
        }
    )


class Usage(TypedDict):
    """One call's billable quantities, already split by how each is priced."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int


def extract_usage(response) -> Usage:
    """Pull the billable token counts off a langchain_anthropic reply.

    `usage_metadata` is langchain's normalised shape and is the primary source.
    The model name comes from `response_metadata`, which reports what actually
    served the request rather than what was asked for -- an alias resolves to a
    concrete model, and the cost belongs to the latter. Missing fields degrade
    to zero/"unknown" rather than raising: a run must not fail because
    accounting is unavailable.

    The subtraction matters. langchain's `usage_metadata["input_tokens"]` is
    *inclusive* of cache reads and writes (it adds them back so total_tokens is
    meaningful), while Anthropic's own `input_tokens` excludes them. Pricing
    langchain's figure at the input rate would bill every cached token twice --
    once at 1.0x here and again at 0.1x or 1.25x as a cache token -- and the
    saving this instrumentation exists to measure would vanish into a rate it
    never paid. What is returned here is uncached input only.
    """
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    raw = metadata.get("usage", {}) or {}
    if not usage:
        # Older/other shapes put the raw Anthropic block here instead. That one
        # already excludes cache tokens, so no subtraction is wanted.
        usage = raw

    details = usage.get("input_token_details", {}) or {}
    cache_write = int(
        details.get("cache_creation", raw.get("cache_creation_input_tokens", 0)) or 0
    )
    cache_read = int(
        details.get("cache_read", raw.get("cache_read_input_tokens", 0)) or 0
    )
    reported_input = int(usage.get("input_tokens", 0) or 0)
    # Only langchain's normalised shape double-counts; guard against going
    # negative if a future version stops doing so.
    uncached_input = (
        max(0, reported_input - cache_write - cache_read) if details else reported_input
    )

    return Usage(
        model=str(metadata.get("model") or "unknown"),
        input_tokens=uncached_input,
        output_tokens=int(usage.get("output_tokens", 0) or 0),
        cache_write_tokens=cache_write,
        cache_read_tokens=cache_read,
    )


def summarise(calls: list[LLMCall], timings: list[NodeTiming]) -> dict:
    """Aggregate one run: tokens, cost, and per-node latency.

    Per-node latency sums by name rather than assuming one execution each: a
    resumed run executes `execute` again after the human approves, and on a
    re-run of the same thread every node runs a second time. Summing keeps the
    total honest about what the thread actually spent.
    """
    input_tokens = sum(c["input_tokens"] for c in calls)
    output_tokens = sum(c["output_tokens"] for c in calls)
    cache_write_tokens = sum(c.get("cache_write_tokens", 0) for c in calls)
    cache_read_tokens = sum(c.get("cache_read_tokens", 0) for c in calls)

    per_node: dict[str, dict] = {}
    for timing in timings:
        entry = per_node.setdefault(
            timing["node"], {"runs": 0, "wall_ms": 0.0, "llm_calls": 0}
        )
        entry["runs"] += 1
        entry["wall_ms"] = round(entry["wall_ms"] + timing["wall_ms"], 1)
    for call in calls:
        entry = per_node.setdefault(
            call["node"], {"runs": 0, "wall_ms": 0.0, "llm_calls": 0}
        )
        entry["llm_calls"] += 1

    input_rate, output_rate = rates()
    write_rate, read_rate = cache_rates()
    return {
        "llm_calls": len(calls),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cache_read_tokens": cache_read_tokens,
        "total_tokens": (
            input_tokens + output_tokens + cache_write_tokens + cache_read_tokens
        ),
        "cost_usd": round(
            cost_usd(
                input_tokens, output_tokens, cache_write_tokens, cache_read_tokens
            ),
            6,
        ),
        "rates_usd_per_mtok": {
            "input": input_rate,
            "output": output_rate,
            "cache_write": write_rate,
            "cache_read": read_rate,
        },
        "total_wall_ms": round(sum(t["wall_ms"] for t in timings), 1),
        "per_node": per_node,
    }
