"""Unit tests for eval.replay's sample selection and scoring arithmetic.

The headline accuracy number is only defensible if the things around it are
right: which issues get excluded, and whether precision/recall mean what the
README says they mean. All pure functions over hand-built rows, so this runs
offline like the rest of the suite -- no issues are fetched and no model is
called.
"""

import pytest

from eval import replay


def make_issue(number: int, labels: list[str]) -> dict:
    return {
        "repo": "owner/repo",
        "number": number,
        "title": "t",
        "body": "b",
        "labels": labels,
        "url": f"https://example.test/{number}",
        "reporter": "octocat",
        "author_association": "NONE",
    }


def make_row(actual: str, predicted: str, extraction_failed: bool = False) -> dict:
    return {
        "number": 1,
        "url": "u",
        "actual": actual,
        "predicted": predicted,
        "correct": predicted == actual,
        "extraction_failed": extraction_failed,
    }


# --- the bot-closed exclusion -----------------------------------------


def test_bot_closed_when_every_label_is_automation():
    assert replay.is_bot_closed(["invalid link"])
    assert replay.is_bot_closed(["locked"])
    assert replay.is_bot_closed(["locked", "invalid link"])


def test_unlabelled_issue_is_not_bot_closed():
    """No labels means untriaged, which is a different exclusion."""
    assert not replay.is_bot_closed([])


def test_automation_label_alongside_a_real_one_is_not_bot_closed():
    """A locked issue a maintainer also routed still has judgment to score."""
    assert not replay.is_bot_closed(["locked", "Turbopack"])


def test_check_labels_script_shares_this_exclusion():
    """The vocabulary check must not drift from the eval's definition."""
    from scripts import check_labels

    assert check_labels.is_bot_closed is replay.is_bot_closed


# --- sample selection -------------------------------------------------


def test_selection_keeps_only_single_component_non_bot_issues():
    issues = [
        make_issue(1, ["invalid link"]),  # bot-closed
        make_issue(2, []),  # untriaged
        make_issue(3, ["Turbopack"]),  # scorable
        make_issue(4, ["Turbopack", "Runtime"]),  # ambiguous
        make_issue(5, ["locked", "Runtime"]),  # scorable
    ]
    selected, funnel = replay.select_scorable(issues, limit=100)

    assert [issue["number"] for issue, _ in selected] == [3, 5]
    assert [actual for _, actual in selected] == ["Turbopack", "Runtime"]
    assert funnel["bot_closed"] == 1
    assert funnel["no_single_component"] == 2
    assert funnel["eligible"] == 2


def test_funnel_counts_all_eligible_even_past_the_limit():
    """The limit caps what gets scored, not what the funnel reports as available.

    Otherwise a --n smaller than the sample would make the eligible count look
    like the qualifying rate, which is the figure used to size the next fetch.
    """
    issues = [make_issue(n, ["Turbopack"]) for n in range(10)]
    selected, funnel = replay.select_scorable(issues, limit=3)

    assert len(selected) == 3
    assert funnel["eligible"] == 10
    assert funnel["scored"] == 3


# --- outcome rates ----------------------------------------------------


def test_outcome_buckets_partition_the_sample():
    rows = [
        make_row("Turbopack", "Turbopack"),
        make_row("Turbopack", "needs-triage"),
        make_row("Runtime", "Turbopack"),
        make_row("Runtime", "Runtime"),
    ]
    summary = replay.summarise(rows, "owner/repo", "vtest", {}, requested_n=4)

    assert summary["accuracy"] == 0.5
    assert summary["abstention_rate"] == 0.25
    assert summary["mislabel_rate"] == 0.25
    assert (
        summary["accuracy"] + summary["abstention_rate"] + summary["mislabel_rate"]
        == 1.0
    )


def test_abstention_is_not_counted_as_a_mislabel():
    """The distinction is the whole point: one can never execute autonomously."""
    rows = [make_row("Turbopack", "needs-triage") for _ in range(4)]
    summary = replay.summarise(rows, "owner/repo", "vtest", {}, requested_n=4)

    assert summary["abstention_rate"] == 1.0
    assert summary["mislabel_rate"] == 0.0


def test_extraction_failures_are_surfaced_not_dropped():
    rows = [
        make_row("Turbopack", "needs-triage", extraction_failed=True),
        make_row("Runtime", "Runtime"),
    ]
    summary = replay.summarise(rows, "owner/repo", "vtest", {}, requested_n=2)

    assert summary["extraction_failures"] == 1
    assert summary["scored_n"] == 2


# --- per-component precision and recall -------------------------------


def test_precision_and_recall_per_component():
    rows = [
        make_row("Turbopack", "Turbopack"),  # Turbopack tp
        make_row("Turbopack", "needs-triage"),  # Turbopack fn
        make_row("Runtime", "Turbopack"),  # Turbopack fp, Runtime fn
        make_row("Runtime", "Runtime"),  # Runtime tp
    ]
    metrics = {m["component"]: m for m in replay.per_component_metrics(rows)}

    assert metrics["Turbopack"] == {
        "component": "Turbopack",
        "support": 2,
        "predicted_n": 2,
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
    }
    # Runtime was never wrongly applied, so abstaining cost recall, not precision.
    assert metrics["Runtime"]["precision"] == 1.0
    assert metrics["Runtime"]["recall"] == 0.5


def test_abstention_never_costs_precision():
    """Declining to route is invisible to precision by construction."""
    rows = [
        make_row("Turbopack", "Turbopack"),
        make_row("Turbopack", "needs-triage"),
        make_row("Turbopack", "needs-triage"),
    ]
    metrics = {m["component"]: m for m in replay.per_component_metrics(rows)}

    assert metrics["Turbopack"]["precision"] == 1.0
    assert metrics["Turbopack"]["recall"] == pytest.approx(1 / 3)


def test_needs_triage_is_not_reported_as_a_component():
    rows = [make_row("Turbopack", "needs-triage")]
    assert [m["component"] for m in replay.per_component_metrics(rows)] == ["Turbopack"]


def test_precision_is_none_for_a_component_never_predicted():
    """None, not 0.0 -- a label that was never applied has no precision to report."""
    rows = [make_row("Runtime", "needs-triage")]
    metrics = {m["component"]: m for m in replay.per_component_metrics(rows)}

    assert metrics["Runtime"]["precision"] is None
    assert metrics["Runtime"]["recall"] == 0.0


# --- LABEL_PREFIX -----------------------------------------------------


def test_prefix_is_stripped_before_comparison(monkeypatch):
    """With LABEL_PREFIX set, predictions must still compare to the repo's names."""
    monkeypatch.setattr(replay.policy, "LABEL_PREFIX", "area: ")
    assert replay._strip_prefix("area: Turbopack") == "Turbopack"


def test_prefix_stripping_is_a_noop_by_default():
    assert replay._strip_prefix("Turbopack") == "Turbopack"
    assert replay._strip_prefix("needs-triage") == "needs-triage"


# --- confidence interval ----------------------------------------------


def test_interval_narrows_as_the_sample_grows():
    """The whole point of scaling the eval up: same rate, tighter bound."""
    small_lo, small_hi = replay.wilson_interval(10, 15)
    large_lo, large_hi = replay.wilson_interval(100, 150)

    assert (small_hi - small_lo) > (large_hi - large_lo)
    assert small_lo < 0.667 < small_hi
    assert large_lo < 0.667 < large_hi


def test_interval_stays_within_bounds_at_the_extremes():
    """Wilson, not the normal approximation, so 100% doesn't run past 1.0."""
    for correct, total in ((0, 10), (10, 10), (1, 1)):
        lo, hi = replay.wilson_interval(correct, total)
        assert 0.0 <= lo <= hi <= 1.0


def test_interval_of_an_empty_sample_is_degenerate():
    assert replay.wilson_interval(0, 0) == (0.0, 0.0)


# --- naive baselines --------------------------------------------------


def test_majority_class_baseline_is_the_dominant_share():
    rows = [make_row("Turbopack", "x") for _ in range(6)] + [
        make_row("Runtime", "x") for _ in range(4)
    ]
    base = replay.baselines(rows)

    assert base["majority_class"] == "Turbopack"
    assert base["majority_support"] == 6
    assert base["majority_accuracy"] == 0.6


def test_random_baseline_is_the_sum_of_squared_frequencies():
    """Guessing L with probability p(L) scores sum(p^2), not 1/len(labels)."""
    rows = [make_row("Turbopack", "x") for _ in range(6)] + [
        make_row("Runtime", "x") for _ in range(4)
    ]
    base = replay.baselines(rows)

    assert base["random_weighted_accuracy"] == pytest.approx(0.6**2 + 0.4**2)


def test_random_baseline_of_a_single_class_is_certainty():
    rows = [make_row("Turbopack", "x") for _ in range(5)]
    base = replay.baselines(rows)

    assert base["majority_accuracy"] == 1.0
    assert base["random_weighted_accuracy"] == 1.0


def test_random_baseline_beats_uniform_on_a_skewed_distribution():
    """The weighted baseline is the harder, fairer bar -- that's why it's used."""
    rows = [make_row("Turbopack", "x") for _ in range(90)] + [
        make_row(f"Other{i}", "x") for i in range(10)
    ]
    base = replay.baselines(rows)
    uniform = 1 / base["distinct_components"]

    assert base["random_weighted_accuracy"] > uniform


def test_majority_tie_resolves_deterministically():
    """Equal supports must not depend on dict insertion order."""
    forward = replay.baselines([make_row("Alpha", "x"), make_row("Beta", "x")])
    backward = replay.baselines([make_row("Beta", "x"), make_row("Alpha", "x")])

    assert forward["majority_class"] == backward["majority_class"] == "Beta"


def test_baselines_of_an_empty_sample_are_zero():
    base = replay.baselines([])

    assert base["n"] == 0
    assert base["majority_class"] is None
    assert base["random_weighted_accuracy"] == 0.0


def test_baselines_ride_along_in_the_summary():
    rows = [make_row("Turbopack", "Turbopack"), make_row("Runtime", "needs-triage")]
    summary = replay.summarise(rows, "owner/repo", "vtest", {}, requested_n=2)

    assert summary["baselines"]["majority_accuracy"] == 0.5


def test_an_agent_that_only_learned_the_majority_class_shows_no_lift():
    """The number that would expose a classifier doing nothing useful.

    A model that always answers Turbopack scores exactly the majority baseline,
    so the ratio is 1.0 -- which is what the report is built to make obvious
    rather than let a respectable-looking 37% stand unqualified.
    """
    rows = [make_row("Turbopack", "Turbopack") for _ in range(6)] + [
        make_row("Runtime", "Turbopack") for _ in range(4)
    ]
    summary = replay.summarise(rows, "owner/repo", "vtest", {}, requested_n=10)

    assert summary["accuracy"] == summary["baselines"]["majority_accuracy"] == 0.6


# --- cost and latency -------------------------------------------------


def make_timed_row(
    wall_ms: float, input_tokens: int = 100, output_tokens: int = 10
) -> dict:
    row = make_row("Turbopack", "Turbopack")
    row.update(
        {
            "wall_ms": wall_ms,
            "llm_calls": 1,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": replay.usage.cost_usd(input_tokens, output_tokens),
        }
    )
    return row


def test_percentile_uses_nearest_rank_so_p95_is_an_observed_value():
    values = [float(v) for v in range(1, 101)]  # 1..100
    assert replay._percentile(values, 0.95) == 95.0
    assert replay._percentile(values, 0.50) == 50.0
    assert replay._percentile(values, 1.0) == 100.0


def test_percentile_of_one_value_is_that_value():
    assert replay._percentile([7.0], 0.95) == 7.0


def test_percentile_of_nothing_is_zero():
    assert replay._percentile([], 0.95) == 0.0


def test_latency_reports_median_and_p95_not_just_a_mean():
    """The distribution has a retry tail; a mean would hide it."""
    rows = [make_timed_row(w) for w in (100, 200, 300, 400, 9000)]
    latency = replay.cost_and_latency(rows)["latency_ms"]

    assert latency["median"] == 300.0
    assert latency["p95"] == 9000.0
    assert latency["max"] == 9000.0
    assert latency["min"] == 100.0


def test_cost_totals_and_divides_by_issue(monkeypatch):
    monkeypatch.setenv("COST_PER_MTOK_INPUT", "2")
    monkeypatch.setenv("COST_PER_MTOK_OUTPUT", "10")
    rows = [make_timed_row(100, input_tokens=1000, output_tokens=100) for _ in range(4)]
    cost = replay.cost_and_latency(rows)["cost"]

    assert cost["input_tokens"] == 4000
    assert cost["output_tokens"] == 400
    assert cost["per_issue_usd"] == pytest.approx(1000 * 2e-6 + 100 * 1e-5)
    assert cost["total_usd"] == pytest.approx(4 * cost["per_issue_usd"], rel=1e-4)


def test_latency_states_what_it_measures():
    """Eval latency is extraction only -- it must not read as agent latency."""
    latency = replay.cost_and_latency([make_timed_row(100)])["latency_ms"]
    assert "extraction only" in latency["measures"]


def test_cost_and_latency_survive_rows_without_telemetry():
    """Old result files have no wall_ms; re-summarising one must not crash."""
    result = replay.cost_and_latency([make_row("Turbopack", "Turbopack")])

    assert result["latency_ms"]["median"] == 0.0
    assert result["cost"]["total_usd"] == 0.0


def test_summary_carries_cost_and_latency():
    rows = [make_timed_row(100), make_timed_row(300)]
    summary = replay.summarise(rows, "owner/repo", "vtest", {}, requested_n=2)

    assert summary["latency_ms"]["median"] == 200.0
    assert summary["cost"]["total_usd"] > 0
