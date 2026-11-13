"""Intent schema and slot configuration.

Two halves, kept apart on purpose:

- IntentResult is what the model returns: an intent, a confidence, a one-line
  rationale, and a verbatim quote (or null) for every slot it was asked about.
  It is a report, never a decision.
- mark_slots and required_slots are plain code over config/slots.yaml. They
  decide which slots an intent needs and which of those are filled. The model
  is never asked "is this issue complete?" -- it is asked what the text says,
  and completeness is computed here.

The config is loaded once and validated against the Intent literal at import,
so a typo in slots.yaml fails at startup rather than routing every issue of
some intent through an empty slot list.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import get_args

import yaml
from pydantic import BaseModel, Field

from app.state import Intent, SlotState

CONFIG_PATH = Path(
    os.environ.get("SLOTS_CONFIG", Path(__file__).parent.parent / "config" / "slots.yaml")
)

INTENTS: tuple[str, ...] = get_args(Intent)


class InvalidSlotConfig(ValueError):
    """config/slots.yaml does not describe the intents the code knows about."""


class SlotExtraction(BaseModel):
    name: str = Field(description="Slot name, copied exactly from the list given.")
    value: str | None = Field(
        description=(
            "A short verbatim quote from the issue or replies that supplies this "
            "slot, or null if the text does not supply it."
        )
    )


class IntentResult(BaseModel):
    """The model's structured report for one issue (title, body and replies)."""

    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="One line explaining the intent chosen.")
    slots: list[SlotExtraction] = Field(default_factory=list)


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> dict:
    """Read and validate slots.yaml. Cached: the file is static per process."""
    raw = yaml.safe_load(Path(path).read_text())
    intents = raw.get("intents") or {}
    slots = raw.get("slots") or {}

    if set(intents) != set(INTENTS):
        raise InvalidSlotConfig(
            f"slots.yaml intents {sorted(intents)} must match {sorted(INTENTS)}"
        )
    for intent, spec in intents.items():
        for slot in spec.get("required") or []:
            if slot not in slots:
                raise InvalidSlotConfig(f"{intent} requires undefined slot {slot!r}")
        if spec.get("required") and not (spec.get("intro") or "").strip():
            raise InvalidSlotConfig(f"{intent} has required slots but no intro text")
    for name, spec in slots.items():
        if not (spec.get("question") or "").strip():
            raise InvalidSlotConfig(f"slot {name!r} has no question text")

    return raw


def required_slots(intent: str) -> list[str]:
    """The slots an issue of this intent needs, in config order."""
    spec = load_config()["intents"].get(intent) or {}
    return list(spec.get("required") or [])


def all_slot_names() -> list[str]:
    """Every slot defined, across intents.

    The model is asked about all of them regardless of the intent it picks, so a
    reviewer who corrects the intent already has the new intent's slots
    extracted -- no second model call.
    """
    return list(load_config()["slots"])


def mark_slots(
    extracted: list[SlotExtraction] | list[dict],
    previous: dict[str, SlotState] | None = None,
) -> dict[str, SlotState]:
    """Merge the model's quotes into slot state. Decides filled vs missing.

    A slot is filled when the model gave a non-blank quote for it. Once filled
    it stays filled: a later reply that does not repeat the version does not
    un-say it. Names the config does not define are dropped rather than
    trusted, so the model cannot invent a slot.
    """
    known = set(all_slot_names())
    merged: dict[str, SlotState] = {
        name: SlotState(value=None, filled=False) for name in known
    }
    for name, slot in (previous or {}).items():
        if name in known:
            merged[name] = SlotState(value=slot.get("value"), filled=bool(slot.get("filled")))

    for item in extracted:
        data = item.model_dump() if isinstance(item, SlotExtraction) else dict(item)
        name, value = data.get("name"), data.get("value")
        if name not in known or merged[name]["filled"]:
            continue
        if isinstance(value, str) and value.strip():
            merged[name] = SlotState(value=value.strip(), filled=True)

    return merged


def missing_slots(intent: str, slots: dict[str, SlotState]) -> list[str]:
    """Required slots for this intent that are not yet filled, in config order."""
    return [
        name for name in required_slots(intent) if not slots.get(name, {}).get("filled")
    ]
