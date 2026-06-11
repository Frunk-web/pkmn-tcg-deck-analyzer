from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/all_cards.csv")
DEFAULT_OUTPUT_DIR = Path("data/reports")


REQUIRED_COLUMNS = [
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
        return None


def split_pipe_list(value: Any) -> list[str]:
    value = clean_value(value)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [part.strip() for part in str(value).split("|") if part.strip()]


def normalize_space(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_effect_lines(row: pd.Series) -> list[dict[str, str]]:
    """
    Returns text chunks that are useful for corpus analysis.

    For Trainer/Energy rules we use the raw rules list.
    For Pokémon, we analyze ability text and attack text separately when raw JSON is available.
    """
    chunks: list[dict[str, str]] = []

    rules = maybe_json_loads(row.get("raw_rules_json"))
    if isinstance(rules, list):
        for rule in rules:
            if clean_value(rule):
                chunks.append({"section": "rules", "text": normalize_space(str(rule))})
    else:
        rules_text = clean_value(row.get("rules"))
        if rules_text:
            for part in str(rules_text).split("|"):
                part = normalize_space(part)
                if part:
                    chunks.append({"section": "rules", "text": part})

    abilities = maybe_json_loads(row.get("raw_abilities_json"))
    if isinstance(abilities, list):
        for ability in abilities:
            name = ability.get("name") if isinstance(ability, dict) else None
            text = ability.get("text") if isinstance(ability, dict) else None
            if clean_value(text):
                chunks.append({
                    "section": "ability",
                    "text": normalize_space(str(text)),
                    "name": clean_value(name) or "",
                })

    attacks = maybe_json_loads(row.get("raw_attacks_json"))
    if isinstance(attacks, list):
        for attack in attacks:
            if not isinstance(attack, dict):
                continue
            name = clean_value(attack.get("name")) or ""
            damage = clean_value(attack.get("damage")) or ""
            text = clean_value(attack.get("text")) or ""
            if damage:
                chunks.append({
                    "section": "attack_damage",
                    "text": normalize_space(f"{name}: damage {damage}"),
                    "name": name,
                })
            if text:
                chunks.append({
                    "section": "attack_text",
                    "text": normalize_space(str(text)),
                    "name": name,
                })

    # Fallback if raw data was missing.
    if not chunks:
        combined = clean_value(row.get("combined_text"))
        if combined:
            chunks.append({"section": "combined", "text": normalize_space(str(combined))})

    return chunks


def classify_text(text: str) -> list[str]:
    t = text.lower()
    tags: list[str] = []

    patterns = [
        ("draw", r"\bdraw\b"),
        ("search_deck", r"\bsearch your deck\b|\bsearch .* deck\b"),
        ("shuffle_deck", r"\bshuffle\b"),
        ("reveal", r"\breveal\b"),
        ("discard", r"\bdiscard\b"),
        ("heal", r"\bheal\b|recovers? from"),
        ("switch", r"\bswitch\b|active spot"),
        ("attach_energy", r"\battach\b.*\benergy\b"),
        ("move_energy", r"\bmove\b.*\benergy\b"),
        ("discard_energy", r"\bdiscard\b.*\benergy\b"),
        ("special_condition", r"poisoned|burned|asleep|confused|paralyzed|special condition"),
        ("damage", r"\bdamage\b|damage counters?"),
        ("coin_flip", r"\bflip a coin\b|coin"),
        ("choose_mode", r"\bchoose 1\b|\bchoose one\b"),
        ("look_top_deck", r"\blook at the top\b"),
        ("bench", r"\bbench\b|benched"),
        ("prize", r"\bprize\b"),
        ("hand_shuffle", r"shuffle.*hand|hand.*shuffle"),
        ("from_discard", r"discard pile"),
        ("global_supporter_rule", r"you may play only 1 supporter card"),
        ("rule_box", r"rule box"),
        ("pokemon_ex_v", r"pokémon ex|pokemon ex|pokémon v|pokemon v"),
    ]

    for tag, pattern in patterns:
        if re.search(pattern, t):
            tags.append(tag)

    if not tags:
        tags.append("unclassified")

    return tags


def normalize_template(text: str) -> str:
    """
    Creates a rough text template for frequency analysis.
    It intentionally keeps words but replaces numbers and energy/color names.
    """
    text = normalize_space(text)
    text = re.sub(r"\b\d+\b", "{N}", text)
    text = re.sub(r"\b(Water|Fire|Grass|Lightning|Psychic|Fighting|Darkness|Metal|Dragon|Colorless|Fairy)\b", "{TYPE}", text)
    text = re.sub(r"\b(Basic|Stage 1|Stage 2)\b", "{STAGE}", text)
    text = re.sub(r"\bPokémon ex\b|\bPokemon ex\b", "Pokémon ex", text, flags=re.IGNORECASE)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Pokémon TCG card text corpus.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--only-with-text", action="store_true")
    parser.add_argument("--standard-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print("Warning: missing expected columns:")
        for col in missing:
            print(f"  - {col}")

    original_rows = len(df)

    if args.standard_only and "legal_standard" in df.columns:
        df = df[df["legal_standard"].str.lower() == "legal"].copy()

    if args.only_with_text:
        has_text = (
            df.get("rules", "").fillna("").str.strip().ne("")
            | df.get("abilities_text", "").fillna("").str.strip().ne("")
            | df.get("attacks_text", "").fillna("").str.strip().ne("")
            | df.get("combined_text", "").fillna("").str.strip().ne("")
        )
        df = df[has_text].copy()

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()

    supertype_counts = Counter(df["supertype"].fillna("").replace("", "Unknown"))
    subtype_counts: Counter[str] = Counter()
    tag_counts: Counter[str] = Counter()
    template_counts: Counter[str] = Counter()
    line_records: list[dict[str, Any]] = []
    card_text_records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        for subtype in split_pipe_list(row.get("subtypes")):
            subtype_counts[subtype] += 1

        chunks = split_effect_lines(row)
        card_tags: set[str] = set()

        for chunk in chunks:
            text = normalize_space(chunk["text"])
            if not text:
                continue

            tags = classify_text(text)
            template = normalize_template(text)

            for tag in tags:
                tag_counts[tag] += 1
                card_tags.add(tag)

            template_counts[template] += 1

            line_records.append({
                "card_id": row.get("card_id"),
                "name": row.get("name"),
                "supertype": row.get("supertype"),
                "subtypes": row.get("subtypes"),
                "section": chunk.get("section"),
                "chunk_name": chunk.get("name", ""),
                "tags": "|".join(tags),
                "template": template,
                "text": text,
            })

        card_text_records.append({
            "card_id": row.get("card_id"),
            "name": row.get("name"),
            "supertype": row.get("supertype"),
            "subtypes": row.get("subtypes"),
            "tag_count": len(card_tags),
            "tags": "|".join(sorted(card_tags)),
            "combined_text": row.get("combined_text"),
        })

    def write_counter_csv(path: Path, counter: Counter[str], key_name: str) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[key_name, "count"])
            writer.writeheader()
            for key, count in counter.most_common():
                writer.writerow({key_name: key, "count": count})

    write_counter_csv(args.output_dir / "supertype_counts.csv", supertype_counts, "supertype")
    write_counter_csv(args.output_dir / "subtype_counts.csv", subtype_counts, "subtype")
    write_counter_csv(args.output_dir / "effect_tag_counts.csv", tag_counts, "tag")
    write_counter_csv(args.output_dir / "effect_template_frequency.csv", template_counts, "template")

    pd.DataFrame(line_records).to_csv(args.output_dir / "effect_text_lines.csv", index=False)
    pd.DataFrame(card_text_records).to_csv(args.output_dir / "card_text_tags.csv", index=False)

    summary = {
        "source_file": str(args.input),
        "original_rows": original_rows,
        "analyzed_rows": len(df),
        "effect_text_chunks": len(line_records),
        "unique_templates": len(template_counts),
        "supertype_counts": dict(supertype_counts.most_common()),
        "top_effect_tags": dict(tag_counts.most_common(30)),
        "top_templates": [
            {"template": template, "count": count}
            for template, count in template_counts.most_common(50)
        ],
        "outputs": {
            "supertype_counts": str(args.output_dir / "supertype_counts.csv"),
            "subtype_counts": str(args.output_dir / "subtype_counts.csv"),
            "effect_tag_counts": str(args.output_dir / "effect_tag_counts.csv"),
            "effect_template_frequency": str(args.output_dir / "effect_template_frequency.csv"),
            "effect_text_lines": str(args.output_dir / "effect_text_lines.csv"),
            "card_text_tags": str(args.output_dir / "card_text_tags.csv"),
        },
    }

    (args.output_dir / "effect_corpus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
