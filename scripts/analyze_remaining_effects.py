from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_COVERAGE = Path("data/reports/compiler_coverage.json")
DEFAULT_REVIEW_QUEUE = Path("data/reports/review_queue.csv")
DEFAULT_OUT_DIR = Path("data/reports/long_tail")


def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


FAMILIES: list[tuple[str, list[re.Pattern[str]]]] = [
    ("attack_lock_or_attack_tax", [
        re.compile(r"can't (?:use )?attacks?", re.I),
        re.compile(r"can't attack", re.I),
        re.compile(r"tries to (?:use an )?attack", re.I),
        re.compile(r"attack(?:s)? cost .* more", re.I),
        re.compile(r"attack doesn't happen", re.I),
        re.compile(r"opponent flips a coin.*(?:attack|turn ends)", re.I),
    ]),
    ("damage_prevention_or_reduction", [
        re.compile(r"prevent all (?:damage|effects)", re.I),
        re.compile(r"prevent .* damage", re.I),
        re.compile(r"reduced by \d+", re.I),
        re.compile(r"takes? \d+ less damage", re.I),
        re.compile(r"not knocked out", re.I),
        re.compile(r"remaining HP becomes", re.I),
        re.compile(r"has no Weakness", re.I),
    ]),
    ("damage_scaling_or_conditional_damage", [
        re.compile(r"does \d+ (?:more )?damage for each", re.I),
        re.compile(r"does \d+ damage times", re.I),
        re.compile(r"plus \d+ more damage for each", re.I),
        re.compile(r"already has any damage counters.*does \d+ more damage", re.I),
        re.compile(r"if .* this attack does \d+ more damage", re.I),
        re.compile(r"same number of cards in your hand", re.I),
        re.compile(r"more cards in your hand", re.I),
        re.compile(r"for each .* attached", re.I),
        re.compile(r"for each .* in (?:your|opponent's) discard", re.I),
        re.compile(r"for each .* in play", re.I),
        re.compile(r"for each .* Retreat Cost", re.I),
    ]),
    ("spread_or_bench_damage", [
        re.compile(r"Benched Pokémon", re.I),
        re.compile(r"each of your opponent's Pokémon", re.I),
        re.compile(r"each of your own Benched", re.I),
        re.compile(r"Choose \d+ of your opponent's Pokémon", re.I),
    ]),
    ("damage_counters", [
        re.compile(r"put \d+ damage counters?", re.I),
        re.compile(r"move all damage counters", re.I),
        re.compile(r"until its remaining HP is", re.I),
        re.compile(r"remove \d+ damage counters?", re.I),
        re.compile(r"damage counters .* any way you like", re.I),
    ]),
    ("special_conditions", [
        re.compile(r"is now (?:Asleep|Confused|Paralyzed|Poisoned|Burned)", re.I),
        re.compile(r"Special Conditions?", re.I),
        re.compile(r"choose (?:1|a) Special Condition", re.I),
    ]),
    ("energy_attachment_or_acceleration", [
        re.compile(r"attach .* Energy .* from (?:your )?(?:discard pile|deck|hand)", re.I),
        re.compile(r"search your deck for .* Energy .* attach", re.I),
        re.compile(r"provides .* Energy", re.I),
    ]),
    ("energy_discard_move_or_bounce", [
        re.compile(r"discard .* Energy", re.I),
        re.compile(r"move .* Energy", re.I),
        re.compile(r"put .* Energy .* into (?:their|your) hand", re.I),
        re.compile(r"attached to .* Energy", re.I),
    ]),
    ("search_deck", [
        re.compile(r"search your deck", re.I),
        re.compile(r"look at the top \d+ cards", re.I),
        re.compile(r"look at the bottom \d+ cards", re.I),
    ]),
    ("draw_shuffle_hand", [
        re.compile(r"draw \d+ cards?", re.I),
        re.compile(r"draw a card", re.I),
        re.compile(r"shuffle your hand into your deck", re.I),
        re.compile(r"draw cards until", re.I),
    ]),
    ("hand_disruption", [
        re.compile(r"opponent's hand", re.I),
        re.compile(r"opponent reveals", re.I),
        re.compile(r"without looking", re.I),
        re.compile(r"random card from your opponent", re.I),
        re.compile(r"opponent discards? \d+ cards? from their hand", re.I),
        re.compile(r"look at your opponent's hand", re.I),
    ]),
    ("switch_or_gust", [
        re.compile(r"switch .* Active Pokémon", re.I),
        re.compile(r"switch in .* Benched Pokémon", re.I),
        re.compile(r"opponent switches", re.I),
        re.compile(r"new Active Pokémon", re.I),
    ]),
    ("evolution_devolution_or_levelup", [
        re.compile(r"evolves? from", re.I),
        re.compile(r"evolve", re.I),
        re.compile(r"devolve", re.I),
        re.compile(r"LV\.X", re.I),
        re.compile(r"previous level", re.I),
    ]),
    ("return_to_hand_or_deck", [
        re.compile(r"return this Pokémon", re.I),
        re.compile(r"put this Pokémon and all attached cards into your hand", re.I),
        re.compile(r"shuffle this Pokémon and all cards attached", re.I),
        re.compile(r"put a card from your discard pile on top of your deck", re.I),
    ]),
    ("discard_recovery", [
        re.compile(r"from your discard pile into your hand", re.I),
        re.compile(r"put .* from your discard pile", re.I),
        re.compile(r"search your discard pile", re.I),
    ]),
    ("trainer_tool_stadium_rules", [
        re.compile(r"Stadium", re.I),
        re.compile(r"Pokémon Tool", re.I),
        re.compile(r"Trainer card", re.I),
        re.compile(r"Retreat Cost .* attached", re.I),
        re.compile(r"last card in your hand", re.I),
        re.compile(r"only if .* Knocked Out", re.I),
    ]),
    ("copy_or_use_other_attack", [
        re.compile(r"use .* as this attack", re.I),
        re.compile(r"can use the attack", re.I),
        re.compile(r"copy", re.I),
    ]),
    ("gx_vstar_rule_or_special_once", [
        re.compile(r"GX attack", re.I),
        re.compile(r"VSTAR Power", re.I),
        re.compile(r"can't use more than 1", re.I),
    ]),
]

CORE_FAMILIES = {
    "attack_lock_or_attack_tax",
    "damage_prevention_or_reduction",
    "damage_scaling_or_conditional_damage",
    "spread_or_bench_damage",
    "damage_counters",
    "special_conditions",
    "energy_attachment_or_acceleration",
    "energy_discard_move_or_bounce",
    "search_deck",
    "draw_shuffle_hand",
    "hand_disruption",
    "switch_or_gust",
    "evolution_devolution_or_levelup",
}


def classify_text(text: str) -> tuple[str, list[str]]:
    text = normalize_space(text)
    matches: list[str] = []
    for family, patterns in FAMILIES:
        if any(p.search(text) for p in patterns):
            matches.append(family)
    if not matches:
        return "other_long_tail", []
    return matches[0], matches


def priority_for(count: int, family: str, all_families: list[str]) -> str:
    if count >= 10:
        return "high"
    if count >= 4 and (family in CORE_FAMILIES or any(f in CORE_FAMILIES for f in all_families)):
        return "medium"
    if family == "other_long_tail":
        return "low"
    return "low_medium"


def read_coverage(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_unparsed_blob(blob: str) -> list[str]:
    blob = normalize_space(blob)
    if not blob:
        return []
    parts = [normalize_space(x) for x in blob.split(" || ")]
    return [x for x in parts if x]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify remaining unparsed Pokémon TCG effects into mechanic families.")
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    coverage = read_coverage(args.coverage)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    classified_rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    family_unique_texts: Counter[str] = Counter()
    family_samples: dict[str, str] = {}

    top_unparsed = coverage.get("top_unparsed_text", []) or []
    for item in top_unparsed[: args.top_n]:
        text = normalize_space(item.get("text", ""))
        count = int(item.get("count", 0) or 0)
        primary, all_families = classify_text(text)
        priority = priority_for(count, primary, all_families)
        classified_rows.append({
            "count": count,
            "priority": priority,
            "primary_family": primary,
            "all_families": "|".join(all_families),
            "text": text,
        })
        family_counts[primary] += count
        family_unique_texts[primary] += 1
        family_samples.setdefault(primary, text)

    write_csv(
        out_dir / "remaining_unparsed_text_classified.csv",
        classified_rows,
        ["count", "priority", "primary_family", "all_families", "text"],
    )

    family_rows = []
    for family, total_count in family_counts.most_common():
        family_rows.append({
            "primary_family": family,
            "weighted_occurrences_in_top_unparsed": total_count,
            "unique_texts_in_top_unparsed": family_unique_texts[family],
            "sample_text": family_samples.get(family, ""),
        })

    write_csv(
        out_dir / "remaining_effect_family_summary.csv",
        family_rows,
        ["primary_family", "weighted_occurrences_in_top_unparsed", "unique_texts_in_top_unparsed", "sample_text"],
    )

    review_rows: list[dict[str, Any]] = []
    review_family_counts: Counter[str] = Counter()
    if args.review_queue.exists():
        with args.review_queue.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts = split_unparsed_blob(row.get("unparsed_text", ""))
                families = []
                for text in texts:
                    primary, all_families = classify_text(text)
                    families.append(primary)
                    review_family_counts[primary] += 1
                primary_family = families[0] if families else "none"
                review_rows.append({
                    "effect_group_id": row.get("effect_group_id", ""),
                    "representative_card_id": row.get("representative_card_id", ""),
                    "name": row.get("name", ""),
                    "supertype": row.get("supertype", ""),
                    "subtypes": row.get("subtypes", ""),
                    "same_effect_printing_count": row.get("same_effect_printing_count", ""),
                    "unparsed_text_count": row.get("unparsed_text_count", ""),
                    "primary_family": primary_family,
                    "all_families": "|".join(sorted(set(families))),
                    "unparsed_text": row.get("unparsed_text", ""),
                })

        write_csv(
            out_dir / "review_queue_classified.csv",
            review_rows,
            [
                "effect_group_id", "representative_card_id", "name", "supertype", "subtypes",
                "same_effect_printing_count", "unparsed_text_count", "primary_family", "all_families", "unparsed_text",
            ],
        )

    summary = {
        "source_coverage": str(args.coverage),
        "source_review_queue": str(args.review_queue) if args.review_queue.exists() else None,
        "coverage": coverage.get("coverage"),
        "status_counts": coverage.get("status_counts"),
        "top_unparsed_items_classified": len(classified_rows),
        "family_summary": family_rows,
        "review_queue_family_counts": review_family_counts.most_common(),
        "outputs": {
            "remaining_unparsed_text_classified": str(out_dir / "remaining_unparsed_text_classified.csv"),
            "remaining_effect_family_summary": str(out_dir / "remaining_effect_family_summary.csv"),
            "review_queue_classified": str(out_dir / "review_queue_classified.csv") if review_rows else None,
        },
    }

    with (out_dir / "long_tail_review_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "status_counts": summary["status_counts"],
        "coverage": summary["coverage"],
        "top_family_summary": family_rows[:15],
        "outputs": summary["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
