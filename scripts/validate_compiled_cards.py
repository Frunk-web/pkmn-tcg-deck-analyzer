from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("data/compiled_cards/auto/compiled_cards_all.json")


ALLOWED_CARD_STATUSES = {"complete", "partial", "needs_human_review"}
REQUIRED_CARD_KEYS = {
    "schema_version",
    "effect_group_id",
    "representative_card_id",
    "same_effect_card_ids",
    "identity",
    "printed",
    "gameplay",
    "sources",
    "compiled_effects",
    "parser",
}
REQUIRED_EFFECT_KEYS = {
    "effect_id",
    "source",
    "kind",
    "timing",
    "playability",
    "costs",
    "choices",
    "steps",
    "duration",
    "usage_limit",
    "parser",
}


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_card(card: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    missing = REQUIRED_CARD_KEYS - set(card.keys())
    if missing:
        errors.append(f"missing card keys: {sorted(missing)}")

    status = card.get("parser", {}).get("status")
    if status not in ALLOWED_CARD_STATUSES:
        errors.append(f"invalid parser.status: {status!r}")

    if not card.get("effect_group_id"):
        errors.append("missing effect_group_id")

    if not card.get("representative_card_id"):
        errors.append("missing representative_card_id")

    same_effect_ids = card.get("same_effect_card_ids")
    if not isinstance(same_effect_ids, list) or not same_effect_ids:
        errors.append("same_effect_card_ids must be a non-empty list")

    identity = card.get("identity", {})
    if not identity.get("name"):
        errors.append("identity.name is missing")
    if not identity.get("supertype"):
        errors.append("identity.supertype is missing")

    effects = card.get("compiled_effects")
    if not isinstance(effects, list):
        errors.append("compiled_effects must be a list")
        return errors

    seen_effect_ids: set[str] = set()

    for idx, effect in enumerate(effects):
        if not isinstance(effect, dict):
            errors.append(f"compiled_effects[{idx}] is not an object")
            continue

        missing_effect = REQUIRED_EFFECT_KEYS - set(effect.keys())
        if missing_effect:
            errors.append(f"effect {idx} missing keys: {sorted(missing_effect)}")

        effect_id = effect.get("effect_id")
        if not effect_id:
            errors.append(f"effect {idx} missing effect_id")
        elif effect_id in seen_effect_ids:
            errors.append(f"duplicate effect_id: {effect_id}")
        else:
            seen_effect_ids.add(effect_id)

        steps = effect.get("steps")
        if not isinstance(steps, list):
            errors.append(f"effect {effect_id} steps must be a list")
        else:
            for sidx, step in enumerate(steps):
                if not isinstance(step, dict):
                    errors.append(f"effect {effect_id} step {sidx} is not an object")
                    continue
                if not step.get("op"):
                    errors.append(f"effect {effect_id} step {sidx} missing op")

        estatus = effect.get("parser", {}).get("status")
        if estatus not in ALLOWED_CARD_STATUSES:
            errors.append(f"effect {effect_id} invalid parser.status: {estatus!r}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate compiled Pokémon card JSON.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--max-errors", type=int, default=100)
    args = parser.parse_args()

    payload = load_payload(args.input)
    cards = payload.get("compiled_cards", [])

    if not isinstance(cards, list):
        raise SystemExit("compiled_cards must be a list")

    status_counts = Counter()
    supertype_counts = Counter()
    total_effects = 0
    error_rows: list[dict[str, Any]] = []

    for card in cards:
        status_counts[card.get("parser", {}).get("status")] += 1
        supertype_counts[card.get("identity", {}).get("supertype")] += 1
        total_effects += len(card.get("compiled_effects") or [])

        errors = validate_card(card)
        if errors:
            error_rows.append({
                "effect_group_id": card.get("effect_group_id"),
                "representative_card_id": card.get("representative_card_id"),
                "name": card.get("identity", {}).get("name"),
                "errors": errors,
            })

    summary = {
        "input": str(args.input),
        "cards": len(cards),
        "effects": total_effects,
        "status_counts": dict(status_counts),
        "supertype_counts": dict(supertype_counts),
        "validation_error_cards": len(error_rows),
        "sample_errors": error_rows[: args.max_errors],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if error_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
