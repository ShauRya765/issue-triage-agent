"""The LangGraph wiring: extract -> decide -> propose -> human_review -> execute.

Every decision lives in app.policy; this module only calls it. The only job
here is sequencing, the durable pause, and turning approved actions into
GitHub API calls with idempotency keys so a resumed/replayed run can't
double-label or double-comment.
"""

import functools
import hashlib
import os
import time

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from psycopg import Connection
from psycopg.rows import dict_row

from app import context as context_module
from app import github, policy, usage
from app.extract import extract_facts
from app.state import Action, Facts, TriageState
from app.usage import NodeTiming


def extract_node(state: TriageState) -> dict:
    issue = state["issue"]
    raw = extract_facts(issue["title"], issue["body"])

    facts: Facts = {
        "has_version": bool(raw.get("has_version", False)),
        "has_reproduction": bool(raw.get("has_reproduction", False)),
        "has_logs": bool(raw.get("has_logs", False)),
        "has_expected_behaviour": bool(raw.get("has_expected_behaviour", False)),
        "claimed_area": raw.get("claimed_area"),
        "area_confidence": raw.get("area_confidence", "low"),
        "kind": raw.get("kind", "question"),
        "evidence": raw.get("evidence", {}),
        # Passed through untouched -- deliberately read from the issue, not from
        # `raw`, so no model output can reach policy's pattern matching. Built
        # here rather than assigned afterwards so the literal satisfies Facts.
        "body": issue["body"],
    }
    return {"facts": facts}


def context_node(state: TriageState) -> dict:
    """Look up who filed this. A fetch, not a judgment -- see app.context."""
    return {"context": context_module.fetch_reporter_context(state["issue"])}


def decide_node(state: TriageState) -> dict:
    facts = state["facts"]
    labels = state["issue"]["labels"]
    reporter_context = state.get("context")
    priority_value = policy.priority(facts, labels, reporter_context)
    component_value = policy.component(facts)
    return {"priority": priority_value, "component": component_value}


def propose_node(state: TriageState) -> dict:
    proposed = policy.actions(state["facts"], state["priority"], state["component"])

    # Decide here, store on the state, and let route_after_propose just read it.
    # Conditional edges run on every resume, so computing the decision inside
    # the router would let it flip mid-run if anything it reads changed.
    reason = policy.review_reason(
        state["facts"],
        state["priority"],
        state["component"],
        proposed,
        state.get("context"),
    )
    return {
        "proposed_actions": proposed,
        "review_required": reason is not None,
        "review_reason": reason or "",
    }


def route_after_propose(state: TriageState) -> str:
    """Send the run to a human, or straight to execution.

    AUTONOMOUS is off by default, so the shipped behaviour is the safe one:
    every run stops for review. Turning it on delegates only the cases
    policy.review_reason positively cleared.
    """
    if not _autonomous_enabled():
        return "human_review"
    # Default True: a state missing the flag has not been cleared for autonomy.
    return "human_review" if state.get("review_required", True) else "auto_approve"


def auto_approve_node(state: TriageState) -> dict:
    """Approve the proposal unchanged, on the agent's own authority."""
    return {"approved_actions": state["proposed_actions"]}


def human_review_node(state: TriageState) -> dict:
    """Pause here. Resuming with Command(resume=actions) supplies the approved list."""
    approved = interrupt(
        {
            "issue": state["issue"],
            "facts": state["facts"],
            "context": state.get("context"),
            "priority": state["priority"],
            "component": state["component"],
            "proposed_actions": state["proposed_actions"],
            # Why this stopped here rather than running autonomously. Empty
            # when AUTONOMOUS is off, since then everything stops regardless.
            "review_reason": state.get("review_reason", ""),
        }
    )
    return {"approved_actions": approved}


def _idempotency_key(repo: str, number: int, action: Action) -> str:
    if action["type"] == "add_label":
        payload = f"add_label:{action['label']}"
    else:
        payload = (
            f"comment:{hashlib.sha256((action['body'] or '').encode()).hexdigest()}"
        )
    return f"{repo}#{number}:{payload}"


def execute_node(state: TriageState) -> dict:
    issue = state["issue"]
    repo, number = issue["repo"], issue["number"]
    executed = list(state.get("executed_keys", []))
    log = list(state.get("execution_log", []))

    for action in state.get("approved_actions", []):
        key = _idempotency_key(repo, number, action)
        if key in executed:
            continue

        if action["type"] == "add_label":
            result = github.add_label(repo, number, action["label"])
        else:
            result = github.comment(repo, number, action["body"])

        executed.append(key)
        log.append(result)

    return {"executed_keys": executed, "execution_log": log}


def _instrumented(name: str, fn):
    """Wrap a node so it records its own wall time and any LLM calls it made.

    The aggregate is recomputed here, on every node, rather than once at the end
    of the graph. A run that stops at the human_review interrupt never reaches a
    final node, and GET /runs/{number} still has to be able to report what that
    run cost -- so the freshest aggregate has to be on the state at every pause
    point, not just at END.

    The collector is opened inside the wrapper so it lives in whichever thread
    LangGraph chose to run the node on; see app.usage for why that matters.

    One gap, deliberately left: human_review_node raises to pause, so the
    execution that hits the interrupt records nothing. It is timed on the
    resuming execution instead, and that is the honest number anyway -- the
    interval a human spends deciding is not the agent's latency.
    """

    @functools.wraps(fn)
    def wrapper(state: TriageState) -> dict:
        with usage.collector() as calls, usage.node(name):
            started = time.perf_counter()
            update = fn(state)
            wall_ms = (time.perf_counter() - started) * 1000

        timing: NodeTiming = {"node": name, "wall_ms": round(wall_ms, 1)}
        merged = dict(update or {})
        merged["llm_calls"] = calls
        merged["node_timings"] = [timing]
        # The reducers have not run yet, so combine by hand to summarise over
        # this node's contribution plus everything already on the state.
        merged["usage"] = usage.summarise(
            list(state.get("llm_calls", [])) + calls,
            list(state.get("node_timings", [])) + [timing],
        )
        return merged

    return wrapper


def _autonomous_enabled() -> bool:
    """Read at call time, not import time, so tests and the API can toggle it."""
    return os.environ.get("AUTONOMOUS", "false").lower() == "true"


def build_graph(checkpointer: PostgresSaver):
    graph = StateGraph(TriageState)

    # Every node goes through _instrumented, including the ones that make no
    # model call -- otherwise the breakdown could only ever confirm that the LLM
    # call is slow. Measured on real issues: extract ~5.3s, fetch_context ~0.5s,
    # decide and propose effectively free.
    def add(name: str, fn) -> None:
        graph.add_node(name, _instrumented(name, fn))

    add("extract", extract_node)
    # Node name differs from the "context" state key on purpose: LangGraph
    # rejects a node that shadows a key in the state schema.
    add("fetch_context", context_node)
    add("decide", decide_node)
    add("propose", propose_node)
    add("human_review", human_review_node)
    add("auto_approve", auto_approve_node)
    add("execute", execute_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "fetch_context")
    graph.add_edge("fetch_context", "decide")
    graph.add_edge("decide", "propose")

    # The only branch in the graph: pause for a human, or self-approve.
    # Both paths converge on the same execute node, so an autonomous run and
    # an approved run execute through identical code -- including the
    # idempotency check.
    graph.add_conditional_edges(
        "propose",
        route_after_propose,
        {"human_review": "human_review", "auto_approve": "auto_approve"},
    )
    graph.add_edge("human_review", "execute")
    graph.add_edge("auto_approve", "execute")
    graph.add_edge("execute", END)

    return graph.compile(checkpointer=checkpointer)


def thread_config(repo: str, number: int) -> dict:
    return {"configurable": {"thread_id": f"{repo}#{number}"}}


_conn = None
_compiled_graph = None


def _connect(database_url: str) -> Connection:
    """Open the checkpoint connection with settings PostgresSaver requires.

    autocommit=True and row_factory=dict_row are what PostgresSaver expects.
    prepare_threshold=None disables psycopg's prepared statements: Supabase's
    transaction pooler (port 6543) is pgbouncer in transaction mode, which
    rejects them ("prepared statement ... already exists"). Disabling costs a
    little per-query planning and makes the checkpointer work on either the
    pooler or a direct/session connection.
    """
    return Connection.connect(
        database_url,
        autocommit=True,
        prepare_threshold=None,
        row_factory=dict_row,
    )


def get_graph():
    """Lazily open the Postgres checkpoint connection and compile the graph once."""
    global _conn, _compiled_graph
    if _compiled_graph is None:
        database_url = os.environ["DATABASE_URL"]
        _conn = _connect(database_url)
        checkpointer = PostgresSaver(_conn)
        checkpointer.setup()
        _compiled_graph = build_graph(checkpointer)
    return _compiled_graph
