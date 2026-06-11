from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/all_cards.csv")
DEFAULT_OUTPUT_DIR = Path("data/card_batches")


IMPORTANT_COLUMNS = [
    "card_id",
    "name",
    "supertype",
    "subtypes",
    "types",
    "hp",
    "evolves_from",
    "rules",
    "abilities_text",
    "attacks_text",
    "combined_text",
    "set_id",
    "set_name",
    "set_series",
    "set_release_date",
    "number",
    "rarity",
    "regulation_mark",
    "legal_standard",
    "legal_expanded",
    "legal_unlimited",
    "raw_rules_json",
    "raw_abilities_json",
    "raw_attacks_json",
    "raw_card_json",
]


def clean_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if isinstance(value, str):
        value = value.strip()
        return value if value else None

    return value


def maybe_json_loads(value: Any) -> Any:
    value = clean_value(value)

    if value is None:
        return None

    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def split_pipe_list(value: Any) -> list[str]:
    value = clean_value(value)

    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    return [part.strip() for part in str(value).split("|") if part.strip()]


def card_priority(row: pd.Series) -> tuple[int, int, int]:
    """
    Lower tuple sorts earlier.

    We prioritize cards that matter most for turn simulation:
    1. Trainer cards, because they are mostly pure effects.
    2. Energy cards, because they affect attachments/costs.
    3. Pokémon with Abilities.
    4. Pokémon with attacks.
    5. Everything else.
    """

    supertype = str(row.get("supertype") or "").lower()
    subtypes = str(row.get("subtypes") or "").lower()
    abilities = str(row.get("abilities_text") or "").strip()
    attacks = str(row.get("attacks_text") or "").strip()

    if supertype == "trainer":
        if "supporter" in subtypes:
            return (0, 0, 0)
        if "item" in subtypes:
            return (0, 1, 0)
        if "stadium" in subtypes:
            return (0, 2, 0)
        if "tool" in subtypes:
            return (0, 3, 0)
        return (0, 9, 0)

    if supertype == "energy":
        return (1, 0, 0)

    if abilities:
        return (2, 0, 0)

    if attacks:
        return (3, 0, 0)

    return (9, 0, 0)


def row_to_review_card(row: pd.Series) -> dict[str, Any]:
    raw_card = maybe_json_loads(row.get("raw_card_json"))

    card = {
        "card_id": clean_value(row.get("card_id")),
        "name": clean_value(row.get("name")),
        "supertype": clean_value(row.get("supertype")),
        "subtypes": split_pipe_list(row.get("subtypes")),
        "types": split_pipe_list(row.get("types")),
        "hp": clean_value(row.get("hp")),
        "evolves_from": clean_value(row.get("evolves_from")),
        "rules": split_pipe_list(row.get("rules")),
        "abilities_text": clean_value(row.get("abilities_text")),
        "attacks_text": clean_value(row.get("attacks_text")),
        "combined_text": clean_value(row.get("combined_text")),
        "set": {
            "id": clean_value(row.get("set_id")),
            "name": clean_value(row.get("set_name")),
            "series": clean_value(row.get("set_series")),
            "release_date": clean_value(row.get("set_release_date")),
        },
        "number": clean_value(row.get("number")),
        "rarity": clean_value(row.get("rarity")),
        "regulation_mark": clean_value(row.get("regulation_mark")),
        "legalities": {
            "standard": clean_value(row.get("legal_standard")),
            "expanded": clean_value(row.get("legal_expanded")),
            "unlimited": clean_value(row.get("legal_unlimited")),
        },
        "raw": {
            "rules": maybe_json_loads(row.get("raw_rules_json")),
            "abilities": maybe_json_loads(row.get("raw_abilities_json")),
            "attacks": maybe_json_loads(row.get("raw_attacks_json")),
            "card": raw_card,
        },
        "compilation_request": {
            "target_schema": "schemas/pokemon_card_schema_v1.json",
            "effect_ops_reference": "docs/pokemon_effect_ops_v1.md",
            "task": "Normalize this card and compile rules, abilities, attacks, trainer text, and energy text into simulator-ready compiled_effects.",
            "allowed_statuses": ["complete", "partial", "needs_human_review"],
        },
    }

    return card


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create small JSON batches for card compilation review."
    )

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-cards", type=int, default=None)

    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="Only include Standard-legal cards.",
    )

    parser.add_argument(
        "--only-with-text",
        action="store_true",
        help="Only include cards with rules, abilities, or attacks text.",
    )

    parser.add_argument(
        "--ids",
        type=Path,
        default=None,
        help="Optional text file with one card_id per line. Use this to batch a decklist first.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)

    missing = [col for col in IMPORTANT_COLUMNS if col not in df.columns]
    if missing:
        print("Warning: missing expected columns:")
        for col in missing:
            print(f"  - {col}")

    if args.standard_only:
        df = df[df["legal_standard"].str.lower() == "legal"].copy()

    if args.only_with_text:
        has_text = (
            df["rules"].fillna("").str.strip().ne("")
            | df["abilities_text"].fillna("").str.strip().ne("")
            | df["attacks_text"].fillna("").str.strip().ne("")
            | df["combined_text"].fillna("").str.strip().ne("")
        )
        df = df[has_text].copy()

    if args.ids:
        wanted_ids = {
            line.strip()
            for line in args.ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        df = df[df["card_id"].isin(wanted_ids)].copy()

    df["priority"] = df.apply(card_priority, axis=1)

    sort_columns = ["priority", "set_release_date", "name", "card_id"]
    existing_sort_columns = [col for col in sort_columns if col in df.columns]

    df = df.sort_values(
        existing_sort_columns,
        ascending=[True, False, True, True][: len(existing_sort_columns)],
    ).copy()

    if args.max_cards is not None:
        df = df.head(args.max_cards).copy()

    cards = [row_to_review_card(row) for _, row in df.iterrows()]

    total_batches = math.ceil(len(cards) / args.batch_size)

    for batch_index in range(total_batches):
        start = batch_index * args.batch_size
        end = start + args.batch_size
        batch_cards = cards[start:end]

        batch_number = batch_index + 1
        output_path = args.output_dir / f"card_batch_{batch_number:04d}.json"

        payload = {
            "batch_id": f"card_batch_{batch_number:04d}",
            "schema_version": "pokemon-card-compilation-batch/v1",
            "instructions_for_chatgpt": [
                "For each card, create a normalized simulator card definition.",
                "Preserve raw text.",
                "Fill identity, printed, gameplay, sources, compiled_effects, and parser fields.",
                "Use parser.status='partial' when the text is not safely compilable.",
                "Do not invent effects not supported by card text.",
                "Return a single JSON object with compiled_cards as a list.",
            ],
            "count": len(batch_cards),
            "cards": batch_cards,
        }

        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Wrote {output_path} with {len(batch_cards)} cards.")

    print()
    print(f"Total cards selected: {len(cards)}")
    print(f"Total batches written: {total_batches}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()