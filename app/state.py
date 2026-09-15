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
    # Telemetry. Both lists use operator.add so each node appends its own record
    # instead of overwriting the previous node's -- without a reducer, the last
    # node to return would be the only one represented.
    llm_calls: Annotated[list[LLMCall], operator.add]
    node_timings: Annotated[list[NodeTiming], operator.add]
    # Aggregate of the two lists above, recomputed by every node so a run that
    # is parked at the human_review interrupt still reports what it has spent so
    # far. See app.usage.summarise for the shape.
    usage: dict
