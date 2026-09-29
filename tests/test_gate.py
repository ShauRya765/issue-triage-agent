"""Unit tests for the offline autonomy-gate replay.

The gate measurement is only worth reading if it replays the *shipped* gate.
These tests pin the two ways it could quietly stop doing that: the refactor
that exposed every tripped condition must not change which reason the graph
acts on, and the offline scorer must build Facts the same way the extract node
does. Everything here is hand-built rows and stub issues, so it runs offline
like the rest of the suite -- no model, no GitHub, no results file on disk.
"""

import pytest

from app import policy
from app.state import facts_from_raw
from eval import gate


def make_facts(**overrides) -> dict:
    raw = {
        "has_version": True,
        "has_reproduction": True,
        "has_logs": True,
        "has_expected_behaviour": True,
        "claimed_area": "Turbopack",
        "area_confidence": "high",
        "kind": "bug",
    }
    raw.update(overrides)
    return raw


def make_issue(number: int = 1, body: str = "it is broken", labels=None) -> dict:
    return {
        "repo": "owner/repo",
        "number": number,
        "title": "t",
        "body": body,
        "labels": ["Turbopack"] if labels is None else labels,
        "url": f"https://example.test/{number}",
        "reporter": "octocat",
        "author_association": "NONE",
    }


def make_row(number: int = 1, actual: str = "Turbopack", **facts_overrides) -> dict:
    facts = make_facts(**facts_overrides)
    return {
        "number": number,
        "url": f"https://example.test/{number}",
        "actual": actual,
        "predicted": "Turbopack",
        "correct": actual == "Turbopack",
        "extraction_failed": False,
        "facts": facts,
        "reporter_context": {
            "login": "octocat",
            "tier": "external",
            "prior_issues": 0,
            "unknown": False,
        },
    }


# --- the refactor that made the measurement possible ------------------


def test_review_reason_is_the_first_of_review_reasons():
    """The graph acts on one reason; the breakdown reads all of them."""
    facts = facts_from_raw(make_facts(area_confidence="low", kind=None), "body")
    proposed = [{"type": "comment", "label": None, "body": "more info?"}]

    reasons = policy.review_reasons(facts, "P0", "needs-triage", proposed, None)
    reason = policy.review_reason(facts, "P0", "needs-triage", proposed, None)

    assert len(reasons) == 5
    assert reason == reasons[0] == policy.P0_REASON


def test_a_clean_run_trips_nothing_and_needs_no_human():
    facts = facts_from_raw(make_facts(), "a plain bug report")
    proposed = [{"type": "add_label", "label": "P1", "body": None}]
    context = {
        "login": "octocat",
        "tier": "external",
        "prior_issues": 0,
        "unknown": False,
    }

    assert policy.review_reasons(facts, "P1", "Turbopack", proposed, context) == []
    assert policy.review_reason(facts, "P1", "Turbopack", proposed, context) is None


def test_every_reason_string_is_declared_in_gate_conditions():
    """Otherwise a new condition would vanish from the breakdown."""
    facts = facts_from_raw(make_facts(area_confidence="low", kind=None), "data loss")
    proposed = [{"type": "comment", "label": None, "body": "?"}]
    context = {"login": "x", "tier": "external", "prior_issues": 0, "unknown": True}

    reasons = policy.review_reasons(facts, "P0", "needs-triage", proposed, context)

    assert set(reasons) == set(policy.GATE_CONDITIONS)
    assert reasons == list(policy.GATE_CONDITIONS), "order must match the checks"


def test_reasons_do_not_short_circuit():
    """Each condition is reported on its own, not only when it happens to be first."""
    facts = facts_from_raw(make_facts(area_confidence="medium"), "body")
    proposed = [{"type": "add_label", "label": "P1", "body": None}]

    reasons = policy.review_reasons(facts, "P1", "needs-triage", proposed, None)

    assert policy.NO_COMPONENT_REASON in reasons
    assert policy.LOW_CONFIDENCE_REASON in reasons


def test_extract_node_and_gate_build_facts_identically():
    """Both callers go through state.facts_from_raw, so they cannot drift."""
    from app import graph

    issue = make_issue(body="the body")
    raw = make_facts()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(graph, "extract_facts", lambda title, body: dict(raw))
        live = graph.extract_node({"issue": issue})["facts"]

    offline = gate.score_row(make_row(), issue)
    assert live == facts_from_raw(raw, issue["body"])
    assert offline["component"] == policy.component(live)


# --- scoring one row --------------------------------------------------


def test_a_confident_complete_bug_clears_the_gate():
    scored = gate.score_row(make_row(), make_issue())

    assert scored["cleared"] is True
    assert scored["reasons"] == []
    assert scored["outcome"] == "correct"


def test_a_data_loss_body_is_stopped_as_p0():
    """The body comes from the cache, not the row -- P0 patterns need real text."""
    issue = make_issue(body="the migration wiped my database")

    scored = gate.score_row(make_row(), issue)

    assert scored["priority"] == "P0"
    assert scored["reasons"] == [policy.P0_REASON]
    assert scored["cleared"] is False


def test_a_missing_repro_is_stopped_by_the_comment_it_would_post():
    issue = make_issue()
    row = make_row(has_reproduction=False)

    scored = gate.score_row(row, issue)

    assert policy.COMMENT_REASON in scored["reasons"]
    assert scored["cleared"] is False


def test_unknown_reporter_context_stops_the_run():
    row = make_row()
    row["reporter_context"]["unknown"] = True

    scored = gate.score_row(row, make_issue())

    assert scored["reasons"] == [policy.NO_CONTEXT_REASON]
    assert scored["context_known"] is False


def test_a_row_without_recorded_context_is_marked_unmeasured():
    """Absent context must read as 'not looked up', never as a clean lookup."""
    row = make_row()
    del row["reporter_context"]

    scored = gate.score_row(row, make_issue())

    assert scored["context_recorded"] is False
    assert policy.NO_CONTEXT_REASON not in scored["reasons"]


def test_prediction_drift_against_the_recorded_run_is_visible():
    row = make_row()
    row["predicted"] = "Runtime"

    summary = gate.summarise_gate([gate.score_row(row, make_issue())])

    assert summary["prediction_drift_n"] == 1


# --- the four headline numbers ----------------------------------------


def _scored(cleared: bool, outcome: str, reasons: list[str]) -> dict:
    return {
        "number": 1,
        "url": "u",
        "actual": "Turbopack",
        "predicted": "Turbopack",
        "recorded_predicted": "Turbopack",
        "correct": outcome == "correct",
        "outcome": outcome,
        "priority": "P2",
        "component": "Turbopack",
        "cleared": cleared,
        "reasons": reasons,
        "context_known": True,
        "context_recorded": True,
    }


def test_headline_numbers():
    scored = [
        _scored(True, "correct", []),
        _scored(True, "correct", []),
        _scored(True, "mislabelled", []),
        _scored(False, "mislabelled", [policy.LOW_CONFIDENCE_REASON]),
        _scored(
            False,
            "abstained",
            [policy.NO_COMPONENT_REASON, policy.LOW_CONFIDENCE_REASON],
        ),
        _scored(False, "abstained", [policy.NO_COMPONENT_REASON]),
    ]

    summary = gate.summarise_gate(scored)

    assert summary["cleared_n"] == 3
    assert summary["clearance_rate"] == pytest.approx(0.5)
    assert summary["mislabels_n"] == 2
    assert summary["mislabels_cleared_n"] == 1
    assert summary["abstentions_n"] == 2
    assert summary["abstentions_stopped_twice_n"] == 1
    # One wrong label in three unsupervised runs.
    assert summary["cleared_mislabel_rate"] == pytest.approx(1 / 3)
    assert summary["cleared_accuracy"] == pytest.approx(2 / 3)


def test_sole_reason_separates_load_bearing_conditions_from_decoration():
    scored = [
        # Low confidence rides along with a condition that would have stopped
        # this run anyway -- it changed no outcome here.
        _scored(
            False,
            "abstained",
            [policy.NO_COMPONENT_REASON, policy.LOW_CONFIDENCE_REASON],
        ),
        _scored(False, "mislabelled", [policy.NO_COMPONENT_REASON]),
    ]

    by_condition = {
        row["condition"]: row for row in gate.summarise_gate(scored)["by_condition"]
    }

    assert by_condition[policy.LOW_CONFIDENCE_REASON]["tripped"] == 1
    assert by_condition[policy.LOW_CONFIDENCE_REASON]["sole_reason"] == 0
    assert by_condition[policy.NO_COMPONENT_REASON]["sole_reason"] == 1
    assert by_condition[policy.NO_COMPONENT_REASON]["mislabels_caught_solely"] == 1


def test_every_condition_appears_in_the_breakdown_even_at_zero():
    """A condition that never fires has to be visible as never firing."""
    summary = gate.summarise_gate([_scored(True, "correct", [])])

    assert [row["condition"] for row in summary["by_condition"]] == list(
        policy.GATE_CONDITIONS
    )
    assert all(row["tripped"] == 0 for row in summary["by_condition"])


# --- refusing to score what it cannot score ---------------------------


def test_results_without_facts_are_rejected_by_name():
    old_shape = {"version": "v4", "rows": [{"number": 1, "predicted": "Turbopack"}]}

    with pytest.raises(gate.UnscorableResults, match="v4"):
        gate._require_gate_fields(old_shape)


def test_results_with_facts_are_accepted():
    gate._require_gate_fields({"version": "v6", "rows": [make_row()]})
