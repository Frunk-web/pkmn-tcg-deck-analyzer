from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/all_cards.csv")
DEFAULT_OUTPUT_DIR = Path("data/card_batches_unique")
DEFAULT_GROUPS_JSON = Path("data/unique_effect_groups.json")
DEFAULT_GROUPS_CSV = Path("data/unique_effect_groups.csv")


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
    "image_small",
    "image_large",
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


def normalize_text(value: Any) -> str:
    value = clean_value(value)
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


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


def canonicalize(value: Any) -> Any:
    """
    Create stable, comparable JSON-like values for hashing.
    This keeps list order where order matters, but normalizes whitespace in strings.
    """

    if value is None:
        return None

    if isinstance(value, str):
        return normalize_text(value)

    if isinstance(value, list):
        return [canonicalize(item) for item in value]

    if isinstance(value, dict):
        return {str(k): canonicalize(value[k]) for k in sorted(value.keys())}

    return value


def stable_hash(payload: Any, length: int = 16) -> str:
    serialized = json.dumps(canonicalize(payload), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


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


def safe_raw_card(row: pd.Series) -> dict[str, Any]:
    raw_card = maybe_json_loads(row.get("raw_card_json"))
    return raw_card if isinstance(raw_card, dict) else {}


def extract_attack_signature(row: pd.Series, raw_card: dict[str, Any]) -> Any:
    raw_attacks = maybe_json_loads(row.get("raw_attacks_json"))

    if isinstance(raw_attacks, list):
        return [
            {
                "name": attack.get("name"),
                "cost": attack.get("cost"),
                "convertedEnergyCost": attack.get("convertedEnergyCost"),
                "damage": attack.get("damage"),
                "text": attack.get("text"),
            }
            for attack in raw_attacks
            if isinstance(attack, dict)
        ]

    raw_card_attacks = raw_card.get("attacks")
    if isinstance(raw_card_attacks, list):
        return [
            {
                "name": attack.get("name"),
                "cost": attack.get("cost"),
                "convertedEnergyCost": attack.get("convertedEnergyCost"),
                "damage": attack.get("damage"),
                "text": attack.get("text"),
            }
            for attack in raw_card_attacks
            if isinstance(attack, dict)
        ]

    return normalize_text(row.get("attacks_text"))


def extract_ability_signature(row: pd.Series, raw_card: dict[str, Any]) -> Any:
    raw_abilities = maybe_json_loads(row.get("raw_abilities_json"))

    if isinstance(raw_abilities, list):
        return [
            {
                "name": ability.get("name"),
                "type": ability.get("type"),
                "text": ability.get("text"),
            }
            for ability in raw_abilities
            if isinstance(ability, dict)
        ]

    raw_card_abilities = raw_card.get("abilities")
    if isinstance(raw_card_abilities, list):
        return [
            {
                "name": ability.get("name"),
                "type": ability.get("type"),
                "text": ability.get("text"),
            }
            for ability in raw_card_abilities
            if isinstance(ability, dict)
        ]

    return normalize_text(row.get("abilities_text"))


def extract_rules_signature(row: pd.Series, raw_card: dict[str, Any]) -> Any:
    raw_rules = maybe_json_loads(row.get("raw_rules_json"))

    if isinstance(raw_rules, list):
        return raw_rules

    raw_card_rules = raw_card.get("rules")
    if isinstance(raw_card_rules, list):
        return raw_card_rules

    return split_pipe_list(row.get("rules"))


def simulation_signature(row: pd.Series) -> dict[str, Any]:
    """
    Safe dedupe key.

    This intentionally includes identity and static gameplay fields, not just text.
    That way alternate arts / rarities / reprints group together, but different cards
    with coincidentally similar text do not get merged by accident.
    """

    raw_card = safe_raw_card(row)

    return {
        "name": normalize_text(row.get("name")),
        "supertype": normalize_text(row.get("supertype")),
        "subtypes": split_pipe_list(row.get("subtypes")),
        "types": split_pipe_list(row.get("types")),
        "hp": normalize_text(row.get("hp")),
        "evolvesFrom": normalize_text(row.get("evolves_from")),
        "rules": extract_rules_signature(row, raw_card),
        "abilities": extract_ability_signature(row, raw_card),
        "attacks": extract_attack_signature(row, raw_card),
        "weaknesses": raw_card.get("weaknesses") or [],
        "resistances": raw_card.get("resistances") or [],
        "retreatCost": raw_card.get("retreatCost") or [],
        "convertedRetreatCost": raw_card.get("convertedRetreatCost"),
        "ancientTrait": raw_card.get("ancientTrait"),
    }


def row_to_printing_summary(row: pd.Series) -> dict[str, Any]:
    return {
        "card_id": clean_value(row.get("card_id")),
        "name": clean_value(row.get("name")),
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
        "images": {
            "small": clean_value(row.get("image_small")),
            "large": clean_value(row.get("image_large")),
        },
    }


def row_to_review_card(row: pd.Series, group_rows: pd.DataFrame, effect_group_id: str) -> dict[str, Any]:
    raw_card = maybe_json_loads(row.get("raw_card_json"))
    printings = [row_to_printing_summary(group_row) for _, group_row in group_rows.iterrows()]

    card = {
        "effect_group_id": effect_group_id,
        "representative_card_id": clean_value(row.get("card_id")),
        "same_effect_card_ids": [p["card_id"] for p in printings],
        "same_effect_printing_count": len(printings),
        "same_effect_printings": printings,
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
        "images": {
            "small": clean_value(row.get("image_small")),
            "large": clean_value(row.get("image_large")),
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
            "task": (
                "Normalize this representative card and compile rules, abilities, attacks, "
                "trainer text, and energy text into simulator-ready compiled_effects. "
                "The compiled behavior applies to every card_id in same_effect_card_ids. "
                "Preserve same_effect_printings so duplicate arts/reprints can map back to the same compiled behavior."
            ),
            "allowed_statuses": ["complete", "partial", "needs_human_review"],
        },
    }

    return card


def apply_filters(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
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

    if args.max_cards is not None:
        df = df.head(args.max_cards).copy()

    return df


def sort_for_review(df: pd.DataFrame) -> pd.DataFrame:
    priorities = df.apply(card_priority, axis=1)
    df = df.copy()
    df["priority_major"] = [p[0] for p in priorities]
    df["priority_minor"] = [p[1] for p in priorities]
    df["priority_patch"] = [p[2] for p in priorities]

    sort_columns = [
        "priority_major",
        "priority_minor",
        "priority_patch",
        "set_release_date",
        "name",
        "card_id",
    ]
    existing_sort_columns = [col for col in sort_columns if col in df.columns]
    ascending = [True, True, True, False, True, True][: len(existing_sort_columns)]

    return df.sort_values(existing_sort_columns, ascending=ascending).copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deduplicated JSON batches for Pokémon TCG card compilation review."
    )

    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--groups-json", type=Path, default=DEFAULT_GROUPS_JSON)
    parser.add_argument("--groups-csv", type=Path, default=DEFAULT_GROUPS_CSV)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Optional limit before deduplication. Useful only for quick testing.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=None,
        help="Optional limit after deduplication. Useful for making a test batch.",
    )

    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="Only include Standard-legal cards. Do not use this if you want all cards.",
    )

    parser.add_argument(
        "--only-with-text",
        action="store_true",
        help="Only include cards with rules, abilities, attacks, or combined text.",
    )

    parser.add_argument(
        "--ids",
        type=Path,
        default=None,
        help="Optional text file with one card_id per line. Use this to batch a decklist first.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.groups_json.parent.mkdir(parents=True, exist_ok=True)
    args.groups_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)

    missing = [col for col in IMPORTANT_COLUMNS if col not in df.columns]
    if missing:
        print("Warning: missing expected columns:")
        for col in missing:
            print(f"  - {col}")

    original_count = len(df)
    df = apply_filters(df, args)
    filtered_count = len(df)

    df = sort_for_review(df)

    df["simulation_signature_json"] = df.apply(
        lambda row: json.dumps(canonicalize(simulation_signature(row)), ensure_ascii=False, sort_keys=True),
        axis=1,
    )
    df["effect_group_id"] = df["simulation_signature_json"].apply(
        lambda value: "eg_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    )

    # Build group metadata in sorted review order.
    group_cards: list[dict[str, Any]] = []
    group_summary_rows: list[dict[str, Any]] = []

    for effect_group_id, group_rows in df.groupby("effect_group_id", sort=False):
        group_rows = sort_for_review(group_rows.drop(columns=[c for c in ["priority_major", "priority_minor", "priority_patch"] if c in group_rows.columns]))
        representative = group_rows.iloc[0]
        representative_card_id = clean_value(representative.get("card_id"))
        same_effect_card_ids = [clean_value(x) for x in group_rows["card_id"].tolist()]

        group_cards.append(row_to_review_card(representative, group_rows, effect_group_id))

        group_summary_rows.append(
            {
                "effect_group_id": effect_group_id,
                "representative_card_id": representative_card_id,
                "name": clean_value(representative.get("name")),
                "supertype": clean_value(representative.get("supertype")),
                "subtypes": clean_value(representative.get("subtypes")),
                "types": clean_value(representative.get("types")),
                "same_effect_printing_count": len(same_effect_card_ids),
                "same_effect_card_ids": "|".join(str(x) for x in same_effect_card_ids if x),
                "rules": clean_value(representative.get("rules")),
                "abilities_text": clean_value(representative.get("abilities_text")),
                "attacks_text": clean_value(representative.get("attacks_text")),
            }
        )

    total_effect_groups_before_limit = len(group_cards)

    if args.max_groups is not None:
        group_cards = group_cards[: args.max_groups]
        group_summary_rows = group_summary_rows[: args.max_groups]

    output_groups = len(group_cards)
    total_batches = math.ceil(output_groups / args.batch_size) if output_groups else 0

    groups_payload = {
        "schema_version": "pokemon-card-effect-groups/v1",
        "source_file": str(args.input),
        "original_card_rows": original_count,
        "filtered_card_rows": filtered_count,
        "unique_effect_groups_total_before_limit": total_effect_groups_before_limit,
        "unique_effect_groups_written": output_groups,
        "dedupe_policy": {
            "mode": "safe",
            "description": (
                "Groups alternate arts, rarities, and reprints only when name, identity, "
                "static gameplay fields, rules, abilities, attacks, weakness, resistance, "
                "and retreat signature match."
            ),
        },
        "groups": [
            {
                "effect_group_id": row["effect_group_id"],
                "representative_card_id": row["representative_card_id"],
                "name": row["name"],
                "supertype": row["supertype"],
                "same_effect_printing_count": row["same_effect_printing_count"],
                "same_effect_card_ids": row["same_effect_card_ids"].split("|") if row["same_effect_card_ids"] else [],
            }
            for row in group_summary_rows
        ],
    }

    args.groups_json.write_text(
        json.dumps(groups_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(group_summary_rows).to_csv(args.groups_csv, index=False, encoding="utf-8-sig")

    for batch_index in range(total_batches):
        start = batch_index * args.batch_size
        end = start + args.batch_size
        batch_cards = group_cards[start:end]

        batch_number = batch_index + 1
        output_path = args.output_dir / f"unique_card_batch_{batch_number:04d}.json"

        payload = {
            "batch_id": f"unique_card_batch_{batch_number:04d}",
            "schema_version": "pokemon-card-compilation-batch/v1",
            "deduped": True,
            "instructions_for_chatgpt": [
                "Each item is one representative card for an effect group.",
                "Compile each representative card once.",
                "The compiled behavior applies to every card_id in same_effect_card_ids.",
                "Preserve same_effect_printings so alternate arts/reprints can map to the compiled behavior.",
                "For each representative card, create a normalized simulator card definition.",
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

        print(f"Wrote {output_path} with {len(batch_cards)} unique effect groups.")

    duplicate_rows_grouped_away_total = filtered_count - total_effect_groups_before_limit

    print()
    print(f"Original card rows: {original_count}")
    print(f"Filtered card rows: {filtered_count}")
    print(f"Unique effect groups before limit: {total_effect_groups_before_limit}")
    print(f"Unique effect groups written: {output_groups}")
    print(f"Duplicate rows grouped away: {duplicate_rows_grouped_away_total}")
    print(f"Total batches written: {total_batches}")
    print(f"Output directory: {args.output_dir}")
    print(f"Group map JSON: {args.groups_json}")
    print(f"Group map CSV: {args.groups_csv}")


if __name__ == "__main__":
    main()
