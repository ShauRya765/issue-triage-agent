"""The only place an LLM is called.

extract_facts asks the model to report what the issue text says, never what
should be done about it. The prompt is deliberately narrow: it asks for
booleans plus a short verbatim quote as evidence for each, so a reviewer can
check the model's claim against the source text without re-reading the whole
issue. If the model returns anything that doesn't parse as the expected JSON
shape, this returns {} rather than raising -- a failed extraction should
degrade to "we know nothing," which app.policy already treats conservatively
(kind defaults elsewhere, not here), not crash the graph.
"""

import json
import logging
import os
import random
import re
import time

from langchain_anthropic import ChatAnthropic

from app import usage
from app.policy import COMPONENTS

logger = logging.getLogger(__name__)

# Retried status codes: 429 (rate limit), 529 (Anthropic "overloaded"), and the
# transient 5xx family. Running the eval with --concurrency makes 429 the
# expected case rather than an edge case, and without a retry it would land as
# a silent {} -- scored as a miss, which would understate accuracy for a reason
# that has nothing to do with the prompt.
_RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_MAX_ATTEMPTS = 5
_MAX_BACKOFF_SECONDS = 30.0


def _status_code(exc: Exception) -> int | None:
    """Dig a status code out of an SDK exception, whatever shape it arrives in."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _is_retryable(exc: Exception) -> bool:
    """True for rate limits, overload and transient network faults.

    Checked by duck-typing rather than by importing anthropic's exception
    classes: anthropic is a transitive dependency of langchain-anthropic and is
    not pinned here, so its exception hierarchy is not ours to rely on.
    """
    if (status := _status_code(exc)) is not None:
        return status in _RETRY_STATUS
    name = type(exc).__name__
    return name.endswith(("ConnectionError", "TimeoutError", "APITimeoutError"))


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with full jitter, so parallel workers desynchronise."""
    ceiling = min(_MAX_BACKOFF_SECONDS, 2.0**attempt)
    return random.uniform(0, ceiling)


_SYSTEM_PROMPT_TEMPLATE = """You extract structured facts from a GitHub issue. You do not \
decide priority, you do not decide labels, and you do not decide whether the \
issue has "enough" information -- you only report what the text says.

Respond with JSON only, no prose, matching exactly this shape:

{
  "has_version": boolean,
  "has_reproduction": boolean,
  "has_logs": boolean,
  "has_expected_behaviour": boolean,
  "claimed_area": string or null,
  "area_confidence": "high" | "medium" | "low",
  "kind": "bug" | "feature" | "question" | "docs",
  "evidence": {
    "<fact_name>": "<short verbatim quote from the issue supporting that fact>"
  }
}

Rules:
- has_version: true only if a specific Next.js version is stated (e.g. "14.2.3").
- has_reproduction: true only if there's a repo link, sandbox link, or concrete
  minimal steps to reproduce -- not just "it happens sometimes".
- has_logs: true only if actual error output, stack trace, or log lines are
  included verbatim.
- has_expected_behaviour: true only if the reporter states what they expected
  to happen, distinct from what happened.
- claimed_area: which product area the text points at. You MUST copy one value
  exactly from the allowed list below, or return null. Do not invent an area
  name, do not reword one, and do not return a plausible-sounding area that is
  not on the list -- if nothing on the list fits, null is the correct answer.
  This is a guess for a human to check, not a decision.
- area_confidence: how confident you are in claimed_area, based on how
  explicitly the text points at that area.
- kind: the single best category for what this issue is asking for.
- Only include an evidence entry for a fact you're asserting true, or for
  claimed_area/kind. Quotes must be copied verbatim from the issue, not
  paraphrased.

Allowed values for claimed_area (copy one exactly, or use null):
{components}
"""


def _system_prompt() -> str:
    """Inject the repo's real label vocabulary into the prompt.

    The vocabulary is code-owned -- it comes from policy.COMPONENTS, derived by
    scripts/check_labels.py from the repo itself. Without it the model reliably
    invented plausible-but-nonexistent areas ("next-devtools", "Use Cache",
    "Telemetry") with high confidence, and policy.component then dropped every
    one to needs-triage. That cost ~half the eval's accuracy.

    This does not move the decision into the model: it still only reports which
    of a known set the text points at, and policy.component still decides
    whether that claim is usable.
    """
    listing = "\n".join(f"- {name}" for name in sorted(COMPONENTS))
    # Plain replace, not .format(): the prompt embeds a literal JSON schema, so
    # str.format would try to read those braces as replacement fields.
    return _SYSTEM_PROMPT_TEMPLATE.replace("{components}", listing)


def _build_user_prompt(title: str, body: str) -> str:
    return f"Issue title: {title}\n\nIssue body:\n{body}"


def _content_to_text(content) -> str:
    """Flatten a model response into plain text.

    langchain_anthropic returns a plain string for simple replies, but newer
    models can return a *list of content blocks* -- [{"type": "text", "text":
    ...}, ...]. Passing that list straight to re.search raises TypeError, which
    the caller swallows into {}, so an otherwise perfect extraction silently
    became "no information". Normalise both shapes here.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content or "")


def _parse_response(content) -> dict:
    text = _content_to_text(content)
    # Models sometimes wrap JSON in a code fence despite instructions; strip it.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    required = {
        "has_version",
        "has_reproduction",
        "has_logs",
        "has_expected_behaviour",
        "claimed_area",
        "area_confidence",
        "kind",
    }
    if not required.issubset(data.keys()):
        return {}

    data.setdefault("evidence", {})
    return data


def extract_facts(title: str, body: str) -> dict:
    """Call the model once and return parsed facts, or {} on any failure.

    Note there is no temperature setting: newer Claude models reject the
    parameter outright (400, "`temperature` is deprecated for this model"),
    and passing it silently turned every extraction into {} -- which looks
    exactly like "the issue contained no information" rather than like a
    misconfiguration.
    """
    model_name = os.environ.get("MODEL", "claude-sonnet-5")
    messages = [
        ("system", _system_prompt()),
        ("human", _build_user_prompt(title, body)),
    ]

    for attempt in range(_MAX_ATTEMPTS):
        try:
            model = ChatAnthropic(model=model_name, max_tokens=1024)
            started = time.perf_counter()
            response = model.invoke(messages)
            wall_ms = (time.perf_counter() - started) * 1000
            # Record before parsing: a reply that fails to parse still cost
            # tokens and time, and hiding it would understate the real spend.
            served_model, input_tokens, output_tokens = usage.extract_usage(response)
            usage.record_call(served_model, input_tokens, output_tokens, wall_ms)
            parsed = _parse_response(response.content)
            if not parsed:
                logger.warning(
                    "extract_facts: model replied but the response did not parse"
                )
            return parsed
        except Exception as exc:
            # Rate limits and overload are worth waiting out; a bad key or a
            # rejected parameter will fail identically every time, so retrying
            # it just multiplies the delay before the real error is logged.
            if _is_retryable(exc) and attempt < _MAX_ATTEMPTS - 1:
                delay = _backoff_seconds(attempt)
                logger.warning(
                    "extract_facts: retryable %s (%s), sleeping %.1fs (attempt %d/%d)",
                    type(exc).__name__,
                    _status_code(exc),
                    delay,
                    attempt + 1,
                    _MAX_ATTEMPTS,
                )
                time.sleep(delay)
                continue
            # Still fail closed -- policy treats {} conservatively and the graph
            # keeps running. But log it: a bad model name, an expired key or a
            # rejected parameter is indistinguishable from a contentless issue
            # once this returns {}, and that silence hid a 400 for a whole run.
            logger.warning("extract_facts failed (%s): %s", type(exc).__name__, exc)
            return {}

    return {}
