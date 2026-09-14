"""Graph-level tests: routing, the durable pause, and idempotency.

These use MemorySaver rather than Postgres. That tests the graph's control
flow -- which branch runs, what pauses, what executes -- but deliberately not
durability across a process restart, which only a real database can show.
See README for how to confirm that part against Supabase.

extract_facts and the reporter lookup are stubbed, so nothing here needs an
API key or the network.
"""

from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app import usage
from app.graph import build_graph, thread_config
from app.main import _initial_state

ISSUE = {
    "repo": "owner/repo",
    "number": 7,
    "title": "Turbopack crash",
    "body": "it crashes",
    "labels": [],
    "url": "https://example.test/7",
    "reporter": "octocat",
    "author_association": "NONE",
}

CLEAN_FACTS = {
    "has_version": True,
    "has_reproduction": True,
    "has_logs": True,
    "has_expected_behaviour": True,
    "claimed_area": "Turbopack",
    "area_confidence": "high",
    "kind": "bug",
    "evidence": {},
}

KNOWN_CONTEXT = {
    "login": "octocat",
    "tier": "external",
    "prior_issues": 0,
    "unknown": False,
}


@pytest.fixture
def calls():
    return []


@pytest.fixture
def graph_env(monkeypatch, calls):
    """Build a graph with the model, the context lookup and GitHub all stubbed."""
    monkeypatch.setenv("SHADOW_MODE", "true")

    def record_label(repo, number, label):
        calls.append(("add_label", label))
        return f"labeled {label}"

    def record_comment(repo, number, body):
        calls.append(("comment", body))
        return "commented"

    with (
        patch("app.graph.extract_facts", return_value=dict(CLEAN_FACTS)),
        patch("app.context.fetch_reporter_context", return_value=dict(KNOWN_CONTEXT)),
        patch("app.github.add_label", record_label),
        patch("app.github.comment", record_comment),
    ):
        yield build_graph(MemorySaver())


def is_paused(graph, config) -> bool:
    return any(task.interrupts for task in graph.get_state(config).tasks)


def start(graph, config, issue=None):
    return graph.invoke(_initial_state(graph, config, issue or ISSUE), config)


# --- routing ----------------------------------------------------------


def test_pauses_for_human_by_default(graph_env, monkeypatch, calls):
    """AUTONOMOUS unset: every run stops, even a clean one."""
    monkeypatch.delenv("AUTONOMOUS", raising=False)
    config = thread_config("owner/repo", 7)
    start(graph_env, config)

    assert is_paused(graph_env, config)
    assert calls == [], "nothing should execute before approval"


def test_autonomous_run_executes_without_a_human(graph_env, monkeypatch, calls):
    monkeypatch.setenv("AUTONOMOUS", "true")
    config = thread_config("owner/repo", 7)
    start(graph_env, config)

    assert not is_paused(graph_env, config)
    assert calls == [("add_label", "P1"), ("add_label", "Turbopack")]


def test_autonomous_still_pauses_on_p0(graph_env, monkeypatch, calls):
    monkeypatch.setenv("AUTONOMOUS", "true")
    config = thread_config("owner/repo", 8)
    issue = {**ISSUE, "number": 8, "body": "this breaks the build"}
    start(graph_env, config, issue)

    state = graph_env.get_state(config)
    assert is_paused(graph_env, config)
    assert state.values["priority"] == "P0"
    assert state.values["review_reason"] == "P0 issues always get a human"
    assert calls == []


# --- the human path ---------------------------------------------------


def test_approval_executes_the_proposed_actions(graph_env, monkeypatch, calls):
    monkeypatch.delenv("AUTONOMOUS", raising=False)
    config = thread_config("owner/repo", 7)
    start(graph_env, config)

    proposed = graph_env.get_state(config).values["proposed_actions"]
    graph_env.invoke(Command(resume=proposed), config)

    assert calls == [("add_label", "P1"), ("add_label", "Turbopack")]
    assert not is_paused(graph_env, config)


def test_human_can_edit_the_action_list(graph_env, monkeypatch, calls):
    """The reviewer's edited list is what executes, not the proposal."""
    monkeypatch.delenv("AUTONOMOUS", raising=False)
    config = thread_config("owner/repo", 7)
    start(graph_env, config)

    edited = [{"type": "add_label", "label": "P3", "body": None}]
    graph_env.invoke(Command(resume=edited), config)

    assert calls == [("add_label", "P3")]


def test_interrupt_payload_carries_context_and_reason(graph_env, monkeypatch):
    monkeypatch.setenv("AUTONOMOUS", "true")
    config = thread_config("owner/repo", 9)
    issue = {**ISSUE, "number": 9, "body": "this breaks the build"}
    start(graph_env, config, issue)

    payload = graph_env.get_state(config).tasks[0].interrupts[0].value
    assert payload["context"] == KNOWN_CONTEXT
    assert payload["review_reason"]
    assert payload["priority"] == "P0"


# --- idempotency ------------------------------------------------------


def test_resuming_a_finished_thread_executes_nothing_more(
    graph_env, monkeypatch, calls
):
    monkeypatch.delenv("AUTONOMOUS", raising=False)
    config = thread_config("owner/repo", 7)
    start(graph_env, config)

    proposed = graph_env.get_state(config).values["proposed_actions"]
    graph_env.invoke(Command(resume=proposed), config)
    graph_env.invoke(Command(resume=proposed), config)

    assert calls == [("add_label", "P1"), ("add_label", "Turbopack")]


def test_rerunning_the_same_issue_does_not_re_execute(graph_env, monkeypatch, calls):
    """Regression: starting a second run used to reset the idempotency ledger.

    The thread id is permanent per issue, so seeding executed_keys with [] on
    every start wiped the record of what had already been done and let a
    re-run comment on the same issue twice.
    """
    monkeypatch.delenv("AUTONOMOUS", raising=False)
    config = thread_config("owner/repo", 7)

    for _ in range(2):
        start(graph_env, config)
        proposed = graph_env.get_state(config).values["proposed_actions"]
        graph_env.invoke(Command(resume=proposed), config)

    assert calls == [("add_label", "P1"), ("add_label", "Turbopack")]


# --- cost and latency telemetry ---------------------------------------


@pytest.fixture
def metered_graph(monkeypatch):
    """Like graph_env, but the stubbed extractor reports token usage."""
    monkeypatch.setenv("SHADOW_MODE", "true")
    monkeypatch.setenv("COST_PER_MTOK_INPUT", "2")
    monkeypatch.setenv("COST_PER_MTOK_OUTPUT", "10")

    def metered_extract(title, body):
        usage.record_call("claude-sonnet-5", 1500, 120, 900.0)
        return dict(CLEAN_FACTS)

    with (
        patch("app.graph.extract_facts", metered_extract),
        patch("app.context.fetch_reporter_context", return_value=dict(KNOWN_CONTEXT)),
        patch("app.github.add_label", lambda *a: "ok"),
        patch("app.github.comment", lambda *a: "ok"),
    ):
        yield build_graph(MemorySaver())


def test_every_node_is_timed_not_just_the_llm_one(metered_graph, monkeypatch):
    """A node that makes no model call still has to be timed and reported."""
    monkeypatch.delenv("AUTONOMOUS", raising=False)
    config = thread_config("owner/repo", 20)
    start(metered_graph, config)

    per_node = metered_graph.get_state(config).values["usage"]["per_node"]
    assert {"extract", "fetch_context", "decide", "propose"} <= set(per_node)
    assert per_node["fetch_context"]["llm_calls"] == 0
    assert per_node["extract"]["llm_calls"] == 1


def test_usage_is_available_while_parked_at_the_interrupt(metered_graph, monkeypatch):
    """A run awaiting approval still has to be able to report what it spent."""
    monkeypatch.delenv("AUTONOMOUS", raising=False)
    config = thread_config("owner/repo", 21)
    start(metered_graph, config)

    assert is_paused(metered_graph, config)
    usage_summary = metered_graph.get_state(config).values["usage"]
    assert usage_summary["llm_calls"] == 1
    assert usage_summary["total_tokens"] == 1620
    assert usage_summary["cost_usd"] == pytest.approx(1500 * 2e-6 + 120 * 1e-5)


def test_telemetry_accumulates_across_the_pause_without_recounting(
    metered_graph, monkeypatch
):
    """Resuming must not re-run extract, nor reset the ledger it wrote."""
    monkeypatch.delenv("AUTONOMOUS", raising=False)
    config = thread_config("owner/repo", 22)
    start(metered_graph, config)

    proposed = metered_graph.get_state(config).values["proposed_actions"]
    metered_graph.invoke(Command(resume=proposed), config)

    values = metered_graph.get_state(config).values
    assert values["usage"]["llm_calls"] == 1, "extract ran once, not twice"
    assert len(values["llm_calls"]) == 1
    assert "execute" in values["usage"]["per_node"]


def test_a_second_run_of_the_same_issue_adds_to_the_bill(metered_graph, monkeypatch):
    """Re-running really does call the model again; the cost must say so."""
    monkeypatch.delenv("AUTONOMOUS", raising=False)
    config = thread_config("owner/repo", 23)

    for _ in range(2):
        start(metered_graph, config)
        proposed = metered_graph.get_state(config).values["proposed_actions"]
        metered_graph.invoke(Command(resume=proposed), config)

    assert metered_graph.get_state(config).values["usage"]["llm_calls"] == 2
