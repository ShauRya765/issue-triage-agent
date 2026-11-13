"""Slot config and the code-side filled/missing decision. No model involved."""

import pytest
from pydantic import ValidationError

from app import intents
from app.intents import IntentResult, SlotExtraction


def test_config_covers_every_intent():
    config = intents.load_config()
    assert set(config["intents"]) == set(intents.INTENTS)


def test_required_slots_match_the_spec():
    assert intents.required_slots("bug_report") == [
        "version",
        "steps_to_reproduce",
        "expected_behavior",
        "actual_behavior",
    ]
    assert intents.required_slots("feature_request") == ["use_case"]
    assert intents.required_slots("question") == []
    assert intents.required_slots("duplicate") == []
    assert intents.required_slots("other") == []


def test_every_required_slot_has_a_question():
    config = intents.load_config()
    for spec in config["intents"].values():
        for slot in spec["required"]:
            assert config["slots"][slot]["question"].strip()


def test_config_rejects_unknown_required_slot(tmp_path):
    bad = tmp_path / "slots.yaml"
    bad.write_text(
        "intents:\n"
        + "".join(
            f"  {name}:\n    intro: hi\n    required: [{'nope' if name == 'bug_report' else ''}]\n"
            for name in intents.INTENTS
        )
        + "slots: {}\n"
    )
    with pytest.raises(intents.InvalidSlotConfig, match="undefined slot 'nope'"):
        intents.load_config.__wrapped__(bad)


def test_config_rejects_missing_intent(tmp_path):
    bad = tmp_path / "slots.yaml"
    bad.write_text("intents:\n  bug_report:\n    required: []\nslots: {}\n")
    with pytest.raises(intents.InvalidSlotConfig, match="must match"):
        intents.load_config.__wrapped__(bad)


def test_confidence_must_be_a_probability():
    with pytest.raises(ValidationError):
        IntentResult(intent="bug_report", confidence=1.5, rationale="x")
    with pytest.raises(ValidationError):
        IntentResult(intent="bug_report", confidence=-0.1, rationale="x")


def test_unknown_intent_is_rejected():
    with pytest.raises(ValidationError):
        IntentResult(intent="rant", confidence=0.9, rationale="x")


def test_mark_slots_fills_only_quoted_values():
    slots = intents.mark_slots(
        [
            SlotExtraction(name="version", value="15.1.3"),
            SlotExtraction(name="expected_behavior", value="   "),
            SlotExtraction(name="actual_behavior", value=None),
        ]
    )
    assert slots["version"] == {"value": "15.1.3", "filled": True}
    assert slots["expected_behavior"]["filled"] is False
    assert slots["actual_behavior"]["filled"] is False


def test_mark_slots_drops_invented_slot_names():
    slots = intents.mark_slots([{"name": "favourite_colour", "value": "blue"}])
    assert "favourite_colour" not in slots


def test_mark_slots_keeps_earlier_fills_across_turns():
    first = intents.mark_slots([{"name": "version", "value": "15.1.3"}])
    second = intents.mark_slots(
        [{"name": "version", "value": None}, {"name": "use_case", "value": "CI"}],
        previous=first,
    )
    assert second["version"] == {"value": "15.1.3", "filled": True}
    assert second["use_case"]["filled"] is True


def test_missing_slots_is_per_intent_and_ordered():
    slots = intents.mark_slots([{"name": "expected_behavior", "value": "a 200"}])
    assert intents.missing_slots("bug_report", slots) == [
        "version",
        "steps_to_reproduce",
        "actual_behavior",
    ]
    assert intents.missing_slots("question", slots) == []
