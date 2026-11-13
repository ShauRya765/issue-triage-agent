"""Shared state shape threaded through the LangGraph graph.

This is the only place the graph's data shape is defined. Every node reads
and writes a subset of these keys; nothing here is decided by an LLM -- it is
just the record of what has happened to one issue as it moves through the
graph.
"""

import operator
from typing import Annotated, Literal, TypedDict

from app.usage import LLMCall, NodeTiming


# Named so the functions that produce these values can be annotated with the
# same type the TypedDicts expect, instead of widening to str at the boundary
# and losing the check exactly where a typo would do damage.
Priority = Literal["P0", "P1", "P2", "P3"]
Tier = Literal["internal", "contributor", "external"]
Intent = Literal["bug_report", "feature_request", "question", "duplicate", "other"]
GateOutcome = Literal["proceed", "clarify", "handoff"]


class Issue(TypedDict):
    repo: str
    number: int
    title: str
    body: str
    labels: list[str]
    url: str
    # Who filed it. Needed to look up reporter context; GitHub can report a
    # deleted account as null, so this may be "".
    reporter: str
    # GitHub's own relationship field: OWNER / MEMBER / COLLABORATOR /
    # CONTRIBUTOR / FIRST_TIME_CONTRIBUTOR / NONE. Comes free on the issue
    # payload -- no extra request.
    author_association: str


class Evidence(TypedDict, total=False):
    has_version: str
    has_reproduction: str
    has_logs: str
    has_expected_behaviour: str
    claimed_area: str
    kind: str


class Facts(TypedDict):
    """What the extract node produces. Model-authored, never a decision.

    Every field here except `body` is the model's report of what the issue text
    says. `body` is declared here deliberately rather
    than smuggled in as an undeclared key: app.policy needs the raw text to
    match its data-loss and build-breaking patterns, and those patterns must run
    against what the reporter actually wrote, not against a model's paraphrase
    of it. Keeping it on the same dict is what lets policy stay a pure function
    of its arguments, with no second fetch and no graph import.
    """

    has_version: bool
    has_reproduction: bool
    has_logs: bool
    has_expected_behaviour: bool
    claimed_area: str | None
    area_confidence: Literal["high", "medium", "low"]
    kind: Literal["bug", "feature", "question", "docs"]
    evidence: Evidence
    # Passed through untouched from the issue -- NOT model output. The only
    # field here that no model authored.
    body: str


class ReporterContext(TypedDict):
    """What we know about the person who filed the issue, fetched not inferred.

    This is the "customer context" step: it is looked up by app.context from
    the GitHub API, never produced by the model. app.policy reads it to decide
    urgency, the same way a human triager would check whether a reporter is a
    maintainer or a first-time filer before deciding how fast to move.

    On a commercial support desk the equivalent fields would come from a CRM
    (plan tier, contract value, open ticket count). The shape is the same: a
    fetched record about the requester, kept separate from the request text.
    """

    login: str
    # Normalised from author_association into the three bands policy cares
    # about: "internal", "contributor", "external".
    tier: Tier
    # Total issues this person has previously filed on this repo.
    prior_issues: int
    # True when the lookup failed (rate limit, network, deleted account).
    # policy must treat this as "no signal", never as "external".
    unknown: bool


class Action(TypedDict):
    type: Literal["add_label", "comment"]
    label: str | None
    body: str | None


class IntentClassification(TypedDict):
    """The model's intent call. Model-authored; the gate decides what it means."""

    intent: Intent
    confidence: float
    rationale: str


class SlotState(TypedDict):
    # Verbatim quote from the issue or a reply, or None if not supplied.
    value: str | None
    # Computed by app.intents.mark_slots, not reported by the model.
    filled: bool


class Turn(TypedDict):
    """One message in the clarification conversation, in posting order."""

    role: Literal["agent", "reporter"]
    body: str
    # GitHub comment id for reporter replies; None for agent comments (which
    # may never have been posted, under DRY_RUN).
    comment_id: int | None
    at: str


class GateDecision(TypedDict):
    """One pass through the intent gate, recorded so the console can show why."""

    outcome: GateOutcome
    reason: str
    # How many clarification comments had been sent when this decision was made.
    attempts: int


class Correction(TypedDict):
    """A reviewer's change to model output, kept as before/after for eval."""

    field: Literal["intent", "slot"]
    # Slot name for field="slot"; None for intent.
    name: str | None
    before: str | None
    after: str | None


class TriageState(TypedDict, total=False):
    issue: Issue
    facts: Facts
    context: ReporterContext
    priority: str
    component: str
    proposed_actions: list[Action]
    approved_actions: list[Action]
    executed_keys: list[str]
    execution_log: list[str]
    # Set by the routing step: whether this run stopped for a human, and why.
    # Recorded on the state so an autonomous run is auditable after the fact.
    review_required: bool
    review_reason: str
    # Intent and slot extraction (app.intents), and the clarification loop.
    intent: IntentClassification
    slots: dict[str, SlotState]
    missing_slots: list[str]
    clarification_attempts: int
    conversation: list[Turn]
    # Appended to on every gate pass, so the full history of why the run went
    # where it went survives across clarification turns.
    gate_decisions: Annotated[list[GateDecision], operator.add]
    # Set when the intent gate hands off: what is still missing, for the human.
    handoff_summary: str
    corrections: list[Correction]
    # Telemetry. Both lists use operator.add so each node appends its own record
    # instead of overwriting the previous node's -- without a reducer, the last
    # node to return would be the only one represented.
    llm_calls: Annotated[list[LLMCall], operator.add]
    node_timings: Annotated[list[NodeTiming], operator.add]
    # Aggregate of the two lists above, recomputed by every node so a run that
    # is parked at the human_review interrupt still reports what it has spent so
    # far. See app.usage.summarise for the shape.
    usage: dict


def facts_from_raw(raw: dict, body: str) -> Facts:
    """Fill a raw extractor dict out into the full Facts shape.

    Lives here, next to the TypedDict it satisfies, because two callers build
    Facts from the same raw extractor output -- the extract node on a live run,
    and eval.gate when it re-scores a stored results file. If they filled the
    defaults differently the offline measurement would be of a gate that never
    ran, so there is one function and both call it.

    `body` is a parameter rather than a key read out of `raw` on purpose: it is
    the only field on Facts no model authored, and app.policy matches its
    data-loss and build-breaking patterns against it. Taking it from the issue
    keeps model output out of that path by construction.
    """
    return Facts(
        has_version=bool(raw.get("has_version", False)),
        has_reproduction=bool(raw.get("has_reproduction", False)),
        has_logs=bool(raw.get("has_logs", False)),
        has_expected_behaviour=bool(raw.get("has_expected_behaviour", False)),
        claimed_area=raw.get("claimed_area"),
        area_confidence=raw.get("area_confidence", "low"),
        kind=raw.get("kind", "question"),
        evidence=raw.get("evidence", {}),
        body=body,
    )
