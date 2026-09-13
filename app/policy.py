"""Every decision the triage agent makes, in one place.

This module makes no LLM calls and imports nothing from app.graph or
app.extract. It only ever sees the structured Facts a model has already
extracted (app.state.Facts) plus the issue's real GitHub labels -- never raw
issue text beyond what the caller passes through in facts["body"]. That is a
deliberate boundary: the model extracts, this module decides. If you find
yourself wanting an LLM call in here, the decision belongs in extract.py's
facts schema instead, as a new fact, not a new import here.
"""

import os
import re
from typing import Literal

from app.state import Action, Facts, ReporterContext

# Real component labels used on vercel/next.js, derived by scripts/check_labels.py
# (see README). Anything not in this set is not a component vercel/next.js
# maintainers actually apply.
_DEFAULT_COMPONENTS = (
    "Turbopack,Runtime,Performance,Dynamic Routes,Output,Linking and Navigating,"
    "Middleware,Internationalization (i18n),TypeScript,Cache Components,Webpack,"
    "Error Overlay,Headers,Parallel & Intercepting Routes,Metadata,Redirects,"
    "Module Resolution,Pages Router,Not Found,Form (next/form),Cookies,"
    "create-next-app,Script (next/script),SWC,React,Route Handlers"
)

COMPONENTS: frozenset[str] = frozenset(
    name.strip()
    for name in os.environ.get("COMPONENTS", _DEFAULT_COMPONENTS).split(",")
    if name.strip()
)

LABEL_PREFIX = os.environ.get("LABEL_PREFIX", "")

# P0 signals that do not depend on the model's opinion: an existing security
# label, or the issue body matching a known data-loss / build-breaking
# pattern. Deliberately narrow and literal -- false negatives here just mean
# a P0 gets scored P1, which a human reviewer will catch; false positives
# would erode trust in the label.
_SECURITY_LABEL_RE = re.compile(r"security", re.IGNORECASE)

_DATA_LOSS_RE = re.compile(
    r"\b(data\s*loss|lost\s+(all\s+)?(my\s+)?data|wiped\s+(my\s+)?(data|database|files)|"
    r"deleted\s+(all\s+)?(my\s+)?(data|files)|corrupt(ed|ing)?\s+(data|database))\b",
    re.IGNORECASE,
)

_BUILD_BREAKING_RE = re.compile(
    r"\b(build\s+(is\s+)?broken|breaks?\s+the\s+build|build\s+(is\s+)?fail(s|ing)?|"
    r"cannot\s+build|can'?t\s+build|fails?\s+to\s+build)\b",
    re.IGNORECASE,
)


def _has_security_signal(labels: list[str]) -> bool:
    return any(_SECURITY_LABEL_RE.search(label) for label in labels)


def _has_pattern_signal(body: str) -> bool:
    return bool(_DATA_LOSS_RE.search(body) or _BUILD_BREAKING_RE.search(body))


# How many prior issues on this repo make someone an established reporter.
# Set by judgment, not measurement: one prior issue is noise, and the point is
# to catch people who file regularly and whose reports have proven actionable.
ESTABLISHED_REPORTER_ISSUES = 3

_LADDER = ["P3", "P2", "P1", "P0"]


def _escalate(priority_value: str) -> str:
    """Move one step up the ladder. Never reaches P0.

    P0 is reserved for the hard signals in priority() -- a security label or a
    data-loss/build-breaking pattern. Who filed an issue is never on its own
    grounds to call something a P0, or the label stops meaning "drop
    everything" and starts meaning "someone important is watching".
    """
    index = _LADDER.index(priority_value)
    return _LADDER[min(index + 1, _LADDER.index("P1"))]


def _context_warrants_escalation(context: ReporterContext | None) -> bool:
    """Whether reporter context justifies one step up. Absent/unknown never does."""
    if context is None or context.get("unknown"):
        return False

    # A maintainer filing an issue has already done the triage a stranger's
    # report still needs: they know the codebase and wouldn't file noise.
    if context.get("tier") == "internal":
        return True

    # A contributor who files regularly has a track record on this repo.
    return (
        context.get("tier") == "contributor"
        and context.get("prior_issues", 0) >= ESTABLISHED_REPORTER_ISSUES
    )


def priority(
    facts: Facts,
    labels: list[str],
    context: ReporterContext | None = None,
) -> Literal["P0", "P1", "P2", "P3"]:
    """Decide priority by rule. P0 requires a matched signal, never the model's say-so.

    context is optional so this stays callable (and testable) with nothing but
    the issue itself. When present it can raise the result by at most one step,
    and never to P0 -- see _escalate.
    """
    body = facts.get("body", "") or ""
    if _has_security_signal(labels) or _has_pattern_signal(body):
        return "P0"

    kind = facts.get("kind")
    if kind == "bug":
        if (
            facts.get("has_reproduction")
            and facts.get("has_logs")
            and facts.get("has_version")
        ):
            base = "P1"
        elif facts.get("has_reproduction"):
            base = "P2"
        else:
            base = "P3"
    else:
        base = "P3"

    if _context_warrants_escalation(context):
        return _escalate(base)
    return base


def component(facts: Facts) -> str:
    """Return the claimed area only if it's a known label and confidently claimed."""
    claimed = facts.get("claimed_area")
    if claimed in COMPONENTS and facts.get("area_confidence") == "high":
        return f"{LABEL_PREFIX}{claimed}" if LABEL_PREFIX else claimed
    return "needs-triage"


def _missing_info_comment(facts: Facts) -> str:
    missing = []
    if not facts.get("has_reproduction"):
        missing.append("a minimal reproduction (repo link or sandbox)")
    if not facts.get("has_version"):
        missing.append("the Next.js version you're using")
    if not facts.get("has_logs"):
        missing.append("relevant error output or logs")
    if not facts.get("has_expected_behaviour"):
        missing.append("what you expected to happen")

    body = (
        "Thanks for the report! To help us triage this, could you add "
        + ", ".join(missing)
        + "?"
    )
    return body


def actions(facts: Facts, priority_value: str, component_value: str) -> list[Action]:
    """Build the proposed action list from already-decided priority/component.

    A bug without a reproduction gets a priority label and a request for the
    missing info, but no component label: routing an unreproducible bug to a
    team is worse than leaving it in needs-triage, since it wastes that
    team's time chasing a report that can't yet be confirmed.
    """
    result: list[Action] = [Action(type="add_label", label=priority_value, body=None)]

    is_unreproducible_bug = facts.get("kind") == "bug" and not facts.get(
        "has_reproduction"
    )

    if is_unreproducible_bug:
        result.append(
            Action(type="comment", label=None, body=_missing_info_comment(facts))
        )
        return result

    result.append(Action(type="add_label", label=component_value, body=None))

    missing_any = not (
        facts.get("has_reproduction")
        and facts.get("has_version")
        and facts.get("has_logs")
        and facts.get("has_expected_behaviour")
    )
    if missing_any:
        result.append(
            Action(type="comment", label=None, body=_missing_info_comment(facts))
        )

    return result


# --- autonomy ---------------------------------------------------------
#
# The gate that decides whether a run may execute without a human. Like every
# other decision in this module it is a pure function over already-extracted
# facts, so "when is the agent allowed to act alone?" is a question you can
# read, test offline, and change deliberately -- not an emergent property of a
# prompt. It is written to fail towards the human: every branch that isn't
# positively known to be safe returns a reason, and an unrecognised state
# falls through to requiring review.


def review_reason(
    facts: Facts,
    priority_value: str,
    component_value: str,
    proposed: list[Action],
    context: ReporterContext | None = None,
) -> str | None:
    """Why this run needs a human, or None if it may execute autonomously.

    Returning the *reason* rather than a bare bool is deliberate: it gets
    recorded on the graph state, so a run that paused can be explained after
    the fact without re-deriving the decision.
    """
    if priority_value == "P0":
        # A P0 is by definition the case where being wrong is most expensive.
        return "P0 issues always get a human"

    if any(action["type"] == "comment" for action in proposed):
        # A label is cheap to remove and nobody is notified. A comment emails
        # every subscriber and can't be unsent -- a wrong one is a public
        # mistake on someone else's issue.
        return "posting a comment to a reporter needs a human"

    if component_value == "needs-triage":
        # We couldn't route it, which is exactly when a human adds value.
        return "component could not be determined"

    if facts.get("area_confidence") != "high":
        return "low model confidence in the claimed area"

    if not facts.get("kind"):
        # extract_facts returns {} on any failure; the graph fills defaults.
        # An empty kind means we're acting on an extraction that didn't happen.
        return "fact extraction failed"

    if context is not None and context.get("unknown"):
        # We asked for reporter context and couldn't get it. Acting anyway
        # means acting on a rule whose inputs we know are incomplete.
        return "reporter context unavailable"

    return None


def requires_review(
    facts: Facts,
    priority_value: str,
    component_value: str,
    proposed: list[Action],
    context: ReporterContext | None = None,
) -> bool:
    """True when a human must approve before anything executes."""
    return (
        review_reason(facts, priority_value, component_value, proposed, context)
        is not None
    )


# --- validation -------------------------------------------------------


class InvalidAction(ValueError):
    """Raised when a human-edited action list isn't safe to execute."""


_VALID_PRIORITIES = frozenset({"P0", "P1", "P2", "P3"})
MAX_COMMENT_CHARS = 4000


def validate_actions(proposed: list) -> list[Action]:
    """Check a human-supplied action list before it reaches the GitHub client.

    The approve endpoint lets a reviewer replace the proposed actions wholesale,
    which is the point -- but it means arbitrary JSON reaches execute_node. Without
    this, a typo'd key surfaces as a KeyError *partway through* execution, after
    earlier actions in the list have already hit the API and can't be taken back.
    Validating the whole list up front makes execution all-or-nothing.

    Labels are checked against the vocabulary the repo actually uses, so an
    approved-but-misspelled label can't silently create a brand new label on
    someone else's repo.
    """
    if not isinstance(proposed, list):
        raise InvalidAction("actions must be a list")

    allowed_labels = COMPONENTS | _VALID_PRIORITIES | {"needs-triage"}
    validated: list[Action] = []

    for index, action in enumerate(proposed):
        where = f"actions[{index}]"

        if not isinstance(action, dict):
            raise InvalidAction(f"{where} must be an object")

        action_type = action.get("type")
        if action_type not in ("add_label", "comment"):
            raise InvalidAction(
                f"{where}.type must be 'add_label' or 'comment', got {action_type!r}"
            )

        if action_type == "add_label":
            label = action.get("label")
            if not isinstance(label, str) or not label.strip():
                raise InvalidAction(f"{where}.label must be a non-empty string")
            bare = label[len(LABEL_PREFIX) :] if LABEL_PREFIX else label
            if bare not in allowed_labels:
                raise InvalidAction(
                    f"{where}.label {label!r} is not a known label for this repo"
                )
            validated.append(Action(type="add_label", label=label, body=None))
        else:
            body = action.get("body")
            if not isinstance(body, str) or not body.strip():
                raise InvalidAction(f"{where}.body must be a non-empty string")
            if len(body) > MAX_COMMENT_CHARS:
                raise InvalidAction(
                    f"{where}.body exceeds {MAX_COMMENT_CHARS} characters"
                )
            validated.append(Action(type="comment", label=None, body=body))

    return validated
