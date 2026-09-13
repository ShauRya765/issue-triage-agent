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

    with patch("app.graph.extract_facts", return_value=dict(CLEAN_FACTS)), patch(
        "app.context.fetch_reporter_context", return_value=dict(KNOWN_CONTEXT)
    ), patch("app.github.add_label", record_label), patch(
        "app.github.comment", record_comment
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
