"""Unit tests for app.policy.

No API key, no network: these exercise pure functions over hand-built facts
dicts. That's the point -- if these pass without ANTHROPIC_API_KEY set, the
decisions genuinely don't depend on the model.
"""

import pytest

from app import policy


def make_facts(**overrides) -> dict:
    facts = {
        "body": "",
        "has_version": True,
        "has_reproduction": True,
        "has_logs": True,
        "has_expected_behaviour": True,
        "claimed_area": None,
        "area_confidence": "low",
        "kind": "bug",
        "evidence": {},
    }
    facts.update(overrides)
    return facts


# --- priority ---------------------------------------------------------


def test_priority_p0_on_security_label():
    facts = make_facts(kind="bug")
    assert policy.priority(facts, ["security", "bug"]) == "P0"


def test_priority_p0_on_data_loss_pattern():
    facts = make_facts(body="Upgrading wiped my database, all rows gone.")
    assert policy.priority(facts, []) == "P0"


def test_priority_p0_on_build_breaking_pattern():
    facts = make_facts(body="This change breaks the build on every CI run.")
    assert policy.priority(facts, []) == "P0"


def test_priority_p0_takes_precedence_over_missing_info():
    # Even a bug with no repro/logs is still P0 if it's a security report.
    facts = make_facts(has_reproduction=False, has_logs=False, has_version=False)
    assert policy.priority(facts, ["type: security"]) == "P0"


def test_priority_p1_fully_reproducible_bug():
    facts = make_facts(has_reproduction=True, has_logs=True, has_version=True)
    assert policy.priority(facts, []) == "P1"


def test_priority_p2_reproducible_but_missing_logs_or_version():
    facts = make_facts(has_reproduction=True, has_logs=False, has_version=False)
    assert policy.priority(facts, []) == "P2"


def test_priority_p3_bug_without_reproduction():
    facts = make_facts(has_reproduction=False)
    assert policy.priority(facts, []) == "P3"


def test_priority_p3_for_non_bug_kinds():
    for kind in ("feature", "question", "docs"):
        facts = make_facts(
            kind=kind, has_reproduction=True, has_logs=True, has_version=True
        )
        assert policy.priority(facts, []) == "P3"


def test_priority_ignores_model_opinion_fields():
    # Facts carrying an out-of-band "priority" or "severity" key (as if a
    # model tried to sneak an opinion in) must have zero effect.
    facts = make_facts(has_reproduction=True, has_logs=True, has_version=True)
    facts["priority"] = "P0"
    facts["severity"] = "critical"
    assert policy.priority(facts, []) == "P1"


# --- component ----------------------------------------------------------


def test_component_returns_claimed_area_when_known_and_high_confidence():
    facts = make_facts(claimed_area="Turbopack", area_confidence="high")
    assert policy.component(facts) == "Turbopack"


def test_component_needs_triage_when_confidence_not_high():
    facts = make_facts(claimed_area="Turbopack", area_confidence="medium")
    assert policy.component(facts) == "needs-triage"


def test_component_needs_triage_when_area_unknown():
    facts = make_facts(claimed_area="Some Made Up Area", area_confidence="high")
    assert policy.component(facts) == "needs-triage"


def test_component_needs_triage_when_area_is_none():
    facts = make_facts(claimed_area=None, area_confidence="high")
    assert policy.component(facts) == "needs-triage"


# --- actions --------------------------------------------------------------


def test_actions_unreproducible_bug_requests_info_and_skips_component_label():
    facts = make_facts(kind="bug", has_reproduction=False)
    result = policy.actions(facts, "P3", "Turbopack")

    label_actions = [a for a in result if a["type"] == "add_label"]
    comment_actions = [a for a in result if a["type"] == "comment"]

    assert len(label_actions) == 1
    assert label_actions[0]["label"] == "P3"
    assert len(comment_actions) == 1
    assert "reproduction" in comment_actions[0]["body"]


def test_actions_reproducible_bug_gets_component_label():
    facts = make_facts(
        kind="bug",
        has_reproduction=True,
        has_logs=True,
        has_version=True,
        has_expected_behaviour=True,
    )
    result = policy.actions(facts, "P1", "Turbopack")

    label_actions = [a for a in result if a["type"] == "add_label"]
    labels = {a["label"] for a in label_actions}

    assert labels == {"P1", "Turbopack"}
    assert not any(a["type"] == "comment" for a in result)


def test_actions_requests_info_when_some_fields_missing_even_with_repro():
    facts = make_facts(
        kind="bug",
        has_reproduction=True,
        has_logs=False,
        has_version=True,
        has_expected_behaviour=True,
    )
    result = policy.actions(facts, "P2", "Turbopack")

    assert any(a["type"] == "comment" for a in result)
    assert any(a["type"] == "add_label" and a["label"] == "Turbopack" for a in result)


def test_actions_never_applies_component_label_to_unreproducible_bug_even_with_high_confidence_area():
    facts = make_facts(
        kind="bug",
        has_reproduction=False,
        claimed_area="Turbopack",
        area_confidence="high",
    )
    component_value = policy.component(facts)
    result = policy.actions(facts, "P3", component_value)

    assert not any(
        a["type"] == "add_label" and a["label"] == "Turbopack" for a in result
    )


# --- reporter context / escalation ------------------------------------
#
# Context can raise priority one step and never to P0. These run offline like
# everything else here: the context dict is hand-built, exactly as app.context
# would have returned it, so the escalation rules are testable without GitHub.


def make_context(**overrides) -> dict:
    context = {
        "login": "octocat",
        "tier": "external",
        "prior_issues": 0,
        "unknown": False,
    }
    context.update(overrides)
    return context


def test_priority_unchanged_without_context():
    facts = make_facts(has_reproduction=False)
    assert policy.priority(facts, []) == "P3"


def test_external_reporter_gets_no_escalation():
    facts = make_facts(has_reproduction=False)
    context = make_context(tier="external")
    assert policy.priority(facts, [], context) == "P3"


def test_internal_reporter_escalates_one_step():
    facts = make_facts(has_reproduction=False)
    context = make_context(tier="internal")
    assert policy.priority(facts, [], context) == "P2"


def test_established_contributor_escalates():
    facts = make_facts(has_reproduction=False)
    context = make_context(tier="contributor", prior_issues=5)
    assert policy.priority(facts, [], context) == "P2"


def test_new_contributor_does_not_escalate():
    facts = make_facts(has_reproduction=False)
    context = make_context(tier="contributor", prior_issues=1)
    assert policy.priority(facts, [], context) == "P3"


def test_unknown_context_never_escalates():
    """A failed lookup must not act like a signal in either direction."""
    facts = make_facts(has_reproduction=False)
    context = make_context(tier="internal", unknown=True)
    assert policy.priority(facts, [], context) == "P3"


def test_context_never_escalates_to_p0():
    """P0 stays reserved for hard signals, whoever filed the issue."""
    facts = make_facts()  # full info -> base P1
    context = make_context(tier="internal")
    assert policy.priority(facts, [], context) == "P1"


def test_context_does_not_downgrade_a_real_p0():
    facts = make_facts(body="this wiped my database")
    context = make_context(tier="external")
    assert policy.priority(facts, [], context) == "P0"


# --- autonomy gate ----------------------------------------------------


def _clean_autonomous_case():
    """Facts/priority/component/actions that policy should clear to run alone."""
    facts = make_facts(claimed_area="Turbopack", area_confidence="high")
    proposed = [{"type": "add_label", "label": "P1", "body": None}]
    return facts, "P1", "Turbopack", proposed


def test_clean_label_only_run_needs_no_human():
    facts, priority_value, component_value, proposed = _clean_autonomous_case()
    assert (
        policy.review_reason(
            facts, priority_value, component_value, proposed, make_context()
        )
        is None
    )
    assert not policy.requires_review(
        facts, priority_value, component_value, proposed, make_context()
    )


def test_p0_always_needs_a_human():
    facts, _, component_value, proposed = _clean_autonomous_case()
    assert policy.requires_review(facts, "P0", component_value, proposed)


def test_comment_action_always_needs_a_human():
    facts, priority_value, component_value, _ = _clean_autonomous_case()
    proposed = [{"type": "comment", "label": None, "body": "please add a repro"}]
    assert policy.requires_review(facts, priority_value, component_value, proposed)


def test_needs_triage_component_needs_a_human():
    facts, priority_value, _, proposed = _clean_autonomous_case()
    assert policy.requires_review(facts, priority_value, "needs-triage", proposed)


def test_low_confidence_needs_a_human():
    facts, priority_value, component_value, proposed = _clean_autonomous_case()
    facts["area_confidence"] = "medium"
    assert policy.requires_review(facts, priority_value, component_value, proposed)


def test_unknown_reporter_context_needs_a_human():
    facts, priority_value, component_value, proposed = _clean_autonomous_case()
    context = make_context(unknown=True)
    assert policy.requires_review(
        facts, priority_value, component_value, proposed, context
    )


def test_failed_extraction_needs_a_human():
    facts, priority_value, component_value, proposed = _clean_autonomous_case()
    facts["kind"] = None
    assert policy.requires_review(facts, priority_value, component_value, proposed)


# --- validation of human-edited actions -------------------------------


def test_validate_accepts_proposed_shape():
    facts = make_facts(claimed_area="Turbopack", area_confidence="high")
    proposed = policy.actions(facts, "P1", "Turbopack")
    assert policy.validate_actions(proposed) == proposed


def test_validate_rejects_non_list():
    with pytest.raises(policy.InvalidAction):
        policy.validate_actions({"type": "add_label", "label": "P1"})


def test_validate_rejects_unknown_type():
    with pytest.raises(policy.InvalidAction):
        policy.validate_actions([{"type": "close_issue", "label": None, "body": None}])


def test_validate_rejects_unknown_label():
    """An approved typo must not create a new label on someone else's repo."""
    with pytest.raises(policy.InvalidAction):
        policy.validate_actions(
            [{"type": "add_label", "label": "Turbopakc", "body": None}]
        )


def test_validate_rejects_empty_comment():
    with pytest.raises(policy.InvalidAction):
        policy.validate_actions([{"type": "comment", "label": None, "body": "   "}])


def test_validate_rejects_oversized_comment():
    body = "x" * (policy.MAX_COMMENT_CHARS + 1)
    with pytest.raises(policy.InvalidAction):
        policy.validate_actions([{"type": "comment", "label": None, "body": body}])


def test_validate_rejects_label_action_missing_label():
    with pytest.raises(policy.InvalidAction):
        policy.validate_actions([{"type": "add_label", "label": None, "body": None}])
