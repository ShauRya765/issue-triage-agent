"""Unit tests for app.usage -- the cost and latency accounting.

Cost is a number someone will quote in a meeting, so the arithmetic and the
rate configuration are worth pinning down. All offline: no model is called, and
usage is read off hand-built response stand-ins shaped like the real ones.
"""

import pytest

from app import usage


class FakeResponse:
    """Stands in for a langchain_anthropic reply, which is all extract_usage reads."""

    def __init__(self, usage_metadata=None, response_metadata=None):
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata


# --- rates ------------------------------------------------------------


def test_default_rates_are_sonnet_5_list_price(monkeypatch):
    monkeypatch.delenv("COST_PER_MTOK_INPUT", raising=False)
    monkeypatch.delenv("COST_PER_MTOK_OUTPUT", raising=False)
    assert usage.rates() == (2.00, 10.00)


def test_rates_are_overridable_from_the_environment(monkeypatch):
    """A negotiated rate or a different model must not need a code edit."""
    monkeypatch.setenv("COST_PER_MTOK_INPUT", "5")
    monkeypatch.setenv("COST_PER_MTOK_OUTPUT", "25")
    assert usage.rates() == (5.0, 25.0)


def test_blank_rate_falls_back_to_the_default(monkeypatch):
    """An empty env var is an unset one, not a request to price at zero."""
    monkeypatch.setenv("COST_PER_MTOK_INPUT", "   ")
    assert usage.rates()[0] == 2.00


def test_malformed_rate_raises_rather_than_pricing_at_zero(monkeypatch):
    """Silently pricing at $0 would make a run look free. Fail loudly instead."""
    monkeypatch.setenv("COST_PER_MTOK_INPUT", "two dollars")
    with pytest.raises(ValueError, match="COST_PER_MTOK_INPUT"):
        usage.rates()


def test_rates_are_read_at_call_time(monkeypatch):
    monkeypatch.setenv("COST_PER_MTOK_INPUT", "1")
    first = usage.rates()[0]
    monkeypatch.setenv("COST_PER_MTOK_INPUT", "3")
    assert first == 1.0 and usage.rates()[0] == 3.0


# --- cost -------------------------------------------------------------


def test_cost_is_priced_per_million_tokens(monkeypatch):
    monkeypatch.setenv("COST_PER_MTOK_INPUT", "2")
    monkeypatch.setenv("COST_PER_MTOK_OUTPUT", "10")
    # 1M input at $2 plus 1M output at $10.
    assert usage.cost_usd(1_000_000, 1_000_000) == pytest.approx(12.0)
    # The realistic shape: ~2.3k in, ~400 out.
    assert usage.cost_usd(2300, 400) == pytest.approx(2300 * 2e-6 + 400 * 1e-5)


def test_cost_of_nothing_is_zero():
    assert usage.cost_usd(0, 0) == 0.0


# --- collection -------------------------------------------------------


def test_collector_captures_calls_made_inside_it():
    with usage.collector() as calls:
        with usage.node("extract"):
            usage.record_call("claude-sonnet-5", 100, 20, 123.45)

    assert calls == [
        {
            "node": "extract",
            "model": "claude-sonnet-5",
            "input_tokens": 100,
            "output_tokens": 20,
            "wall_ms": 123.5,
        }
    ]


def test_recording_without_a_collector_is_a_no_op():
    """Instrumentation must never be the reason a triage run fails."""
    usage.record_call("claude-sonnet-5", 100, 20, 1.0)  # no collector bound


def test_calls_outside_a_node_are_tagged_unknown():
    with usage.collector() as calls:
        usage.record_call("claude-sonnet-5", 1, 1, 1.0)
    assert calls[0]["node"] == "unknown"


def test_collectors_do_not_leak_into_each_other():
    with usage.collector() as outer:
        with usage.node("extract"):
            usage.record_call("m", 1, 1, 1.0)
        with usage.collector() as inner:
            with usage.node("other"):
                usage.record_call("m", 2, 2, 2.0)

    assert len(outer) == 1 and len(inner) == 1
    assert outer[0]["input_tokens"] == 1
    assert inner[0]["input_tokens"] == 2


def test_node_tag_is_restored_after_the_block():
    with usage.collector() as calls:
        with usage.node("outer"):
            with usage.node("inner"):
                usage.record_call("m", 1, 1, 1.0)
            usage.record_call("m", 1, 1, 1.0)

    assert [c["node"] for c in calls] == ["inner", "outer"]


# --- reading usage off a response -------------------------------------


def test_usage_is_read_from_usage_metadata():
    response = FakeResponse(
        usage_metadata={"input_tokens": 1500, "output_tokens": 120},
        response_metadata={"model": "claude-sonnet-5"},
    )
    assert usage.extract_usage(response) == ("claude-sonnet-5", 1500, 120)


def test_model_comes_from_what_actually_served_the_request():
    """response_metadata reports the resolved model; cost belongs to that one."""
    response = FakeResponse(
        usage_metadata={"input_tokens": 1, "output_tokens": 1},
        response_metadata={"model": "claude-sonnet-5"},
    )
    assert usage.extract_usage(response)[0] == "claude-sonnet-5"


def test_usage_falls_back_to_the_raw_anthropic_block():
    response = FakeResponse(
        usage_metadata=None,
        response_metadata={
            "model": "m",
            "usage": {"input_tokens": 7, "output_tokens": 3},
        },
    )
    assert usage.extract_usage(response) == ("m", 7, 3)


def test_missing_usage_degrades_to_zero_rather_than_raising():
    """Accounting being unavailable must not fail the run."""
    assert usage.extract_usage(FakeResponse()) == ("unknown", 0, 0)


def test_null_token_counts_are_treated_as_zero():
    response = FakeResponse(
        usage_metadata={"input_tokens": None, "output_tokens": None},
        response_metadata={},
    )
    assert usage.extract_usage(response) == ("unknown", 0, 0)


# --- aggregation ------------------------------------------------------


def test_summarise_totals_tokens_and_cost(monkeypatch):
    monkeypatch.setenv("COST_PER_MTOK_INPUT", "2")
    monkeypatch.setenv("COST_PER_MTOK_OUTPUT", "10")
    calls = [
        {
            "node": "extract",
            "model": "m",
            "input_tokens": 1000,
            "output_tokens": 100,
            "wall_ms": 900.0,
        },
        {
            "node": "extract",
            "model": "m",
            "input_tokens": 500,
            "output_tokens": 50,
            "wall_ms": 400.0,
        },
    ]
    timings = [{"node": "extract", "wall_ms": 1300.0}]
    summary = usage.summarise(calls, timings)

    assert summary["llm_calls"] == 2
    assert summary["input_tokens"] == 1500
    assert summary["output_tokens"] == 150
    assert summary["total_tokens"] == 1650
    assert summary["cost_usd"] == pytest.approx(1500 * 2e-6 + 150 * 1e-5)


def test_per_node_latency_covers_nodes_that_made_no_llm_call():
    """Nodes that call no model still have to appear, or the breakdown is partial."""
    timings = [
        {"node": "extract", "wall_ms": 900.0},
        {"node": "fetch_context", "wall_ms": 1500.0},
    ]
    summary = usage.summarise([], timings)

    assert summary["per_node"]["fetch_context"]["wall_ms"] == 1500.0
    assert summary["per_node"]["fetch_context"]["llm_calls"] == 0


def test_repeated_node_executions_are_summed_not_overwritten():
    """execute runs again after a human approves; the total must reflect both."""
    timings = [
        {"node": "execute", "wall_ms": 100.0},
        {"node": "execute", "wall_ms": 250.0},
    ]
    summary = usage.summarise([], timings)

    assert summary["per_node"]["execute"]["runs"] == 2
    assert summary["per_node"]["execute"]["wall_ms"] == 350.0
    assert summary["total_wall_ms"] == 350.0


def test_summary_records_the_rates_it_used(monkeypatch):
    """So a stored figure can be re-checked against the price it assumed."""
    monkeypatch.setenv("COST_PER_MTOK_INPUT", "7")
    monkeypatch.setenv("COST_PER_MTOK_OUTPUT", "9")
    assert usage.summarise([], [])["rates_usd_per_mtok"] == {
        "input": 7.0,
        "output": 9.0,
    }


def test_summarise_of_an_empty_run_is_all_zero():
    summary = usage.summarise([], [])

    assert summary["llm_calls"] == 0
    assert summary["total_tokens"] == 0
    assert summary["cost_usd"] == 0.0
    assert summary["per_node"] == {}
