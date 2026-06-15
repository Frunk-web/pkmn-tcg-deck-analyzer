from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_COVERAGE = Path("data/reports/compiler_coverage.json")
DEFAULT_REVIEW_QUEUE = Path("data/reports/review_queue.csv")
DEFAULT_OUT_DIR = Path("data/reports/long_tail")

TURN1_RELEVANCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


FAMILIES: list[tuple[str, str, str, list[re.Pattern[str]]]] = [
    # ------------------------------------------------------------------
    # Specific high-frequency / high-clarity battle families first.
    # Ordering matters: the first matching family becomes primary.
    # These split older "other_long_tail" / overly broad buckets into
    # compiler-actionable families.
    # ------------------------------------------------------------------
    (
        "damage_retaliation_or_reflect",
        "none",
        "Retaliation / reflect damage formulas are common battle-text patterns and should compile into conditional damage modifiers.",
        [
            re.compile(r"was damaged by an attack during your opponent's last turn.*does that much more damage", re.I),
            re.compile(r"this attack does that much more damage", re.I),
            re.compile(r"does damage equal to the damage done to", re.I),
            re.compile(r"does the same amount of damage", re.I),
            re.compile(r"if .* was damaged .* last turn", re.I),
        ],
    ),
    (
        "attack_cost_modification",
        "none",
        "Attack-cost modifiers should compile as conditional cost modifiers, separate from Energy movement/discard effects.",
        [
            re.compile(r"ignore all Energy in this Pokémon's attack costs", re.I),
            re.compile(r"ignore .* Energy .* attack costs", re.I),
            re.compile(r"attack(?:s)? cost .* less", re.I),
            re.compile(r"attack(?:s)? cost .* more", re.I),
            re.compile(r"(?:no|without) Energy .* attack", re.I),
            re.compile(r"you still need the necessary Energy", re.I),
        ],
    ),
    (
        "coin_damage_plus_special_condition",
        "none",
        "Coin-flip attacks that both scale damage and apply a Special Condition are a reusable battle pattern.",
        [
            re.compile(r"flip \d+ coins?.*damage times the number of heads.*(?:Paralyzed|Poisoned|Asleep|Confused|Burned|Special Condition)", re.I),
            re.compile(r"flip \d+ coins?.*number of heads.*(?:Defending|Active) Pokémon is now", re.I),
            re.compile(r"if .* coins? .* heads.*(?:Paralyzed|Poisoned|Asleep|Confused|Burned)", re.I),
        ],
    ),
    (
        "gust_before_or_during_damage",
        "none",
        "Gust/switch-before-damage effects combine switching with attack resolution and should be separated from spread damage.",
        [
            re.compile(r"before doing damage.*switch .* Benched Pokémon", re.I),
            re.compile(r"before doing damage.*choose .* Benched Pokémon.*switch", re.I),
            re.compile(r"switch .* opponent's Benched Pokémon .* Active", re.I),
            re.compile(r"switch .* Defending Pokémon", re.I),
        ],
    ),
    (
        "damage_scaling_by_revealed_hand_or_trainers",
        "none",
        "Damage based on revealed hand contents is a conditional damage formula, not a Trainer rule.",
        [
            re.compile(r"opponent reveals their hand.*damage .* each Trainer", re.I),
            re.compile(r"look at your opponent's hand.*damage .* each", re.I),
            re.compile(r"for each Trainer card .* opponent", re.I),
            re.compile(r"for each .* card .* opponent's hand", re.I),
        ],
    ),
    (
        "energy_provides_or_type_modifier",
        "high",
        "Energy-providing/modifying effects are common and should compile as attachment-provided energy or modifier rules.",
        [
            re.compile(r"provides? (?:\[?[A-Za-z]+\]?|Colorless|all types of) Energy", re.I),
            re.compile(r"provides .* Energy", re.I),
            re.compile(r"counts as .* Energy", re.I),
            re.compile(r"as long as this card is attached.*provides", re.I),
        ],
    ),
    (
        "self_recovery_or_recycle_from_discard",
        "medium",
        "Self-recovery/recycle effects from discard should compile into discard-to-deck/hand movement effects.",
        [
            re.compile(r"if this Pokémon is in your discard pile.*(?:bottom|top) of your deck", re.I),
            re.compile(r"put this Pokémon on the bottom of your deck", re.I),
            re.compile(r"put this card from your discard pile", re.I),
            re.compile(r"shuffle .* from your discard pile into your deck", re.I),
        ],
    ),
    (
        "between_turns_damage_counter_healing",
        "none",
        "Between-turn damage-counter removal is a battle/status pattern separate from damage-counter placement.",
        [
            re.compile(r"at any time between turns.*remove \d+ damage counters?", re.I),
            re.compile(r"between turns.*remove .* damage counters?", re.I),
        ],
    ),
    (
        "healing_or_damage_counter_removal",
        "none",
        "Healing and damage-counter removal should compile into heal_damage / remove_damage_counters operations.",
        [
            re.compile(r"heal \d+ damage from", re.I),
            re.compile(r"remove \d+ damage counters? from", re.I),
            re.compile(r"remove .* damage counters?", re.I),
            re.compile(r"heal .* damage", re.I),
        ],
    ),
    (
        "retreat_cost_modification",
        "low",
        "Retreat-cost changes should compile as conditional retreat-cost modifiers, separate from attack costs.",
        [
            re.compile(r"has no Retreat Cost", re.I),
            re.compile(r"Retreat Cost .* is 0", re.I),
            re.compile(r"Retreat Cost .* less", re.I),
            re.compile(r"Retreat Cost .* more", re.I),
            re.compile(r"can't retreat", re.I),
            re.compile(r"cannot retreat", re.I),
        ],
    ),
    (
        "self_shuffle_or_return_to_deck",
        "medium",
        "Self shuffle / return-to-deck effects should compile into zone movement of this Pokémon and attached cards.",
        [
            re.compile(r"shuffle this Pokémon and all attached cards into your deck", re.I),
            re.compile(r"shuffle this Pokémon .* into your deck", re.I),
            re.compile(r"return this Pokémon and all cards attached to it to your hand", re.I),
            re.compile(r"put this Pokémon and all attached cards .* into your deck", re.I),
        ],
    ),
    (
        "status_condition_immunity",
        "none",
        "Special-condition immunity/prevention is a battle/status rule family.",
        [
            re.compile(r"can't be (?:Paralyzed|Poisoned|Asleep|Confused|Burned|affected by a Special Condition)", re.I),
            re.compile(r"cannot be (?:Paralyzed|Poisoned|Asleep|Confused|Burned)", re.I),
            re.compile(r"prevent .* Special Condition", re.I),
        ],
    ),
    (
        "opponent_draw_or_may_draw",
        "low",
        "Opponent draw effects should compile as opponent draw decisions but usually matter less for solo consistency.",
        [
            re.compile(r"your opponent draws? (?:a|\d+) cards?", re.I),
            re.compile(r"opponent may draw", re.I),
        ],
    ),
    (
        "future_damage_bonus_or_mark",
        "none",
        "Future-turn damage bonuses/marks are delayed battle modifiers.",
        [
            re.compile(r"until the end of your next turn.*attack does \d+ more damage", re.I),
            re.compile(r"during your next turn.*attack does \d+ more damage", re.I),
            re.compile(r"if an attack damages the Defending Pokémon.*does \d+ more damage", re.I),
        ],
    ),
    (
        "direct_damage_ignore_weakness_resistance",
        "none",
        "Direct damage that ignores Weakness/Resistance should compile as targeted damage with weakness/resistance flags.",
        [
            re.compile(r"does \d+ damage to 1 of your opponent's Pokémon.*isn't affected by Weakness or Resistance", re.I),
            re.compile(r"damage isn't affected by Weakness or Resistance", re.I),
            re.compile(r"don't apply Weakness and Resistance", re.I),
        ],
    ),
    (
        "attachment_duration_or_delayed_discard",
        "medium",
        "Delayed discard of an attached card is a Tool/attachment duration rule.",
        [
            re.compile(r"if this card is attached .* discard it at the end of", re.I),
            re.compile(r"discard it at the end of your opponent's turn", re.I),
            re.compile(r"discard this card at the end of", re.I),
        ],
    ),
    (
        "conditional_damage_vs_pokemon_ex_or_damaged",
        "none",
        "Conditional damage based on defending Pokémon subtype or existing damage counters should compile as damage modifiers.",
        [
            re.compile(r"if the Defending Pokémon is Pokémon-ex.*does .* more damage", re.I),
            re.compile(r"if the Defending Pokémon .* has any damage counters", re.I),
            re.compile(r"already has any damage counters.*more damage", re.I),
        ],
    ),
    (
        "bounce_defending_pokemon_to_hand",
        "none",
        "Bounce effects return the defending/active Pokémon and attachments to hand.",
        [
            re.compile(r"opponent returns the Defending Pokémon and all cards attached to it to .* hand", re.I),
            re.compile(r"return the Defending Pokémon and all cards attached", re.I),
            re.compile(r"put the Defending Pokémon and all attached cards into .* hand", re.I),
        ],
    ),
    (
        "copy_supporter_or_trainer_effect",
        "medium",
        "Copying Supporter/Trainer effects should compile as a copy-effect / deferred execution pattern.",
        [
            re.compile(r"search your opponent's discard pile for a Supporter card and use the effect", re.I),
            re.compile(r"use the effect of that card as the effect of this attack", re.I),
            re.compile(r"copy .* Supporter", re.I),
        ],
    ),
    (
        "mill_or_deck_discard",
        "low",
        "Deck discard/mill effects should compile as opponent/self deck-to-discard movement.",
        [
            re.compile(r"discard the top card from your opponent's deck", re.I),
            re.compile(r"discard the top \d+ cards? of .* deck", re.I),
            re.compile(r"put the top card of .* deck .* discard", re.I),
        ],
    ),
    # ------------------------------------------------------------------
    # General setup/card-access families.
    # ------------------------------------------------------------------
    (
        "search_deck",
        "high",
        "Search/look-at-deck effects can directly change Turn-1 access lines.",
        [
            re.compile(r"search your deck", re.I),
            re.compile(r"look at the top \d+ cards", re.I),
            re.compile(r"look at the bottom \d+ cards", re.I),
            re.compile(r"reveal .* put .* into your hand", re.I),
        ],
    ),
    (
        "draw_shuffle_hand",
        "high",
        "Draw/refresh effects directly affect opening-hand and Turn-1 consistency.",
        [
            re.compile(r"draw \d+ cards?", re.I),
            re.compile(r"draw a card", re.I),
            re.compile(r"draw cards until", re.I),
            re.compile(r"shuffle your hand into your deck", re.I),
        ],
    ),
    (
        "energy_attachment_or_acceleration",
        "high",
        "Energy attach/acceleration matters for setup goals and can be a cost/enabler.",
        [
            re.compile(r"attach .* Energy .* from (?:your )?(?:discard pile|deck|hand)", re.I),
            re.compile(r"search your deck for .* Energy .* attach", re.I),
        ],
    ),
    (
        "energy_discard_move_or_bounce",
        "high",
        "Energy discard/move/bounce is often a cost or setup transformation.",
        [
            re.compile(r"discard .* Energy", re.I),
            re.compile(r"move .* Energy", re.I),
            re.compile(r"put .* Energy .* into (?:their|your) hand", re.I),
            re.compile(r"attached to .* Energy", re.I),
            re.compile(r"Basic [A-Za-z]+ Energy", re.I),
        ],
    ),
    (
        "evolution_devolution_or_levelup",
        "high",
        "Evolution effects matter for setup-package goals like Basic + evolution + Wally.",
        [
            re.compile(r"evolves? from", re.I),
            re.compile(r"evolve", re.I),
            re.compile(r"devolve", re.I),
            re.compile(r"LV\.X", re.I),
            re.compile(r"previous level", re.I),
        ],
    ),
    (
        "discard_recovery",
        "medium",
        "Discard recovery can matter after costs, but is usually less central than draw/search.",
        [
            re.compile(r"from your discard pile into your hand", re.I),
            re.compile(r"put .* from your discard pile", re.I),
            re.compile(r"search your discard pile", re.I),
        ],
    ),
    (
        "trainer_tool_stadium_rules",
        "medium",
        "Trainer/Stadium/Tool restrictions can affect whether a line is legal.",
        [
            re.compile(r"Stadium", re.I),
            re.compile(r"Pokémon Tool", re.I),
            re.compile(r"Trainer card", re.I),
            re.compile(r"Retreat Cost .* attached", re.I),
            re.compile(r"last card in your hand", re.I),
            re.compile(r"only if .* Knocked Out", re.I),
        ],
    ),
    (
        "switch_or_gust",
        "medium",
        "Switching can matter for active/bench setup goals, but not basic starting-hand statements.",
        [
            re.compile(r"switch .* Active Pokémon", re.I),
            re.compile(r"switch in .* Benched Pokémon", re.I),
            re.compile(r"opponent switches", re.I),
            re.compile(r"new Active Pokémon", re.I),
        ],
    ),
    (
        "hand_disruption",
        "low",
        "Opponent-hand effects are usually not relevant to solo Turn-1 consistency diagnostics.",
        [
            re.compile(r"opponent's hand", re.I),
            re.compile(r"opponent reveals", re.I),
            re.compile(r"without looking", re.I),
            re.compile(r"random card from your opponent", re.I),
            re.compile(r"opponent discards? \d+ cards? from their hand", re.I),
            re.compile(r"look at your opponent's hand", re.I),
        ],
    ),
    (
        "gx_vstar_rule_or_special_once",
        "medium",
        "Once-per-game/once-per-turn rules can matter if they gate draw/search/setup effects.",
        [
            re.compile(r"GX attack", re.I),
            re.compile(r"VSTAR Power", re.I),
            re.compile(r"can't use more than 1", re.I),
            re.compile(r"once during your turn", re.I),
        ],
    ),
    (
        "return_to_hand_or_deck",
        "low",
        "Bounce effects are usually full-game or attack-line effects, rarely core starting setup.",
        [
            re.compile(r"return this Pokémon", re.I),
            re.compile(r"put this Pokémon and all attached cards into your hand", re.I),
            re.compile(r"shuffle this Pokémon and all cards attached", re.I),
            re.compile(r"put a card from your discard pile on top of your deck", re.I),
        ],
    ),
    (
        "attack_lock_or_attack_tax",
        "none",
        "Attack locks/taxes are battle simulation concerns, not starting-hand consistency.",
        [
            re.compile(r"can't (?:use )?attacks?", re.I),
            re.compile(r"can't attack", re.I),
            re.compile(r"tries to (?:use an )?attack", re.I),
            re.compile(r"attack doesn't happen", re.I),
            re.compile(r"opponent flips a coin.*(?:attack|turn ends)", re.I),
        ],
    ),
    (
        "damage_prevention_or_reduction",
        "none",
        "Damage prevention is important for full battle simulation, not Turn-1 access.",
        [
            re.compile(r"prevent all (?:damage|effects)", re.I),
            re.compile(r"prevent .* damage", re.I),
            re.compile(r"reduced by \d+", re.I),
            re.compile(r"takes? \d+ less damage", re.I),
            re.compile(r"not knocked out", re.I),
            re.compile(r"remaining HP becomes", re.I),
            re.compile(r"has no Weakness", re.I),
        ],
    ),
    (
        "damage_scaling_or_conditional_damage",
        "none",
        "Damage formulas are battle simulation concerns, not starting-hand consistency.",
        [
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
        ],
    ),
    (
        "spread_or_bench_damage",
        "none",
        "Bench/spread damage is battle simulation, not card-access consistency.",
        [
            re.compile(r"does .* damage to .* Benched Pokémon", re.I),
            re.compile(r"damage to each of your opponent's Pokémon", re.I),
            re.compile(r"each of your opponent's Pokémon", re.I),
            re.compile(r"each of your own Benched", re.I),
            re.compile(r"Choose \d+ of your opponent's Pokémon", re.I),
        ],
    ),
    (
        "damage_counters",
        "none",
        "Damage-counter placement is battle simulation, not starting-hand consistency.",
        [
            re.compile(r"put \d+ damage counters?", re.I),
            re.compile(r"move all damage counters", re.I),
            re.compile(r"until its remaining HP is", re.I),
            re.compile(r"remove \d+ damage counters?", re.I),
            re.compile(r"damage counters .* any way you like", re.I),
        ],
    ),
    (
        "special_conditions",
        "none",
        "Special Conditions are battle simulation concerns for this milestone.",
        [
            re.compile(r"is now (?:Asleep|Confused|Paralyzed|Poisoned|Burned)", re.I),
            re.compile(r"Special Conditions?", re.I),
            re.compile(r"choose (?:1|a) Special Condition", re.I),
        ],
    ),
    (
        "copy_or_use_other_attack",
        "none",
        "Copying attacks belongs to full battle simulation.",
        [
            re.compile(r"use .* as this attack", re.I),
            re.compile(r"can use the attack", re.I),
            re.compile(r"copy", re.I),
        ],
    ),
]

# Families that should generally be prioritized for Turn-1 consistency work.
CORE_FAMILIES = {family for family, relevance, _, _ in FAMILIES if relevance in {"high", "medium"}}


def classify_text(text: str) -> tuple[str, list[str], str, str]:
    text = normalize_space(text)
    matches: list[str] = []
    relevance_labels: list[str] = []
    notes: list[str] = []
    for family, turn1_relevance, turn1_note, patterns in FAMILIES:
        if any(p.search(text) for p in patterns):
            matches.append(family)
            relevance_labels.append(turn1_relevance)
            notes.append(turn1_note)
    if not matches:
        return "other_long_tail", [], "unknown", "Needs review; Turn-1 impact is not known yet."
    primary = matches[0]
    best_relevance = max(relevance_labels, key=lambda x: TURN1_RELEVANCE_RANK.get(x, 0))
    return primary, matches, best_relevance, notes[0]


def priority_for(count: int, family: str, all_families: list[str], turn1_relevance: str) -> str:
    """Global compiler-priority label for a single unparsed text row.

    This is intentionally based on frequency, not Turn-1 relevance. Turn-1
    relevance is still reported as a separate column, but compiler roadmap work
    should primarily follow repeated effect families across the whole corpus.
    """
    if count >= 25:
        return "high"
    if count >= 10:
        return "medium"
    if count >= 4:
        return "low"
    return "watch"


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
    family_turn1_counts: Counter[str] = Counter()
    family_samples: dict[str, str] = {}
    family_notes: dict[str, str] = {}

    top_unparsed = coverage.get("top_unparsed_text", []) or []
    for item in top_unparsed[: args.top_n]:
        text = normalize_space(item.get("text", ""))
        count = int(item.get("count", 0) or 0)
        primary, all_families, turn1_relevance, turn1_note = classify_text(text)
        priority = priority_for(count, primary, all_families, turn1_relevance)
        classified_rows.append({
            "count": count,
            "priority": priority,
            "turn1_relevance": turn1_relevance,
            "primary_family": primary,
            "all_families": "|".join(all_families),
            "turn1_note": turn1_note,
            "text": text,
        })
        family_counts[primary] += count
        family_unique_texts[primary] += 1
        family_turn1_counts[turn1_relevance] += count
        family_samples.setdefault(primary, text)
        family_notes.setdefault(primary, turn1_note)

    write_csv(
        out_dir / "remaining_unparsed_text_classified.csv",
        classified_rows,
        ["count", "priority", "turn1_relevance", "primary_family", "all_families", "turn1_note", "text"],
    )

    family_rows = []
    for family, total_count in family_counts.most_common():
        # Choose highest relevance observed in this family from classified rows.
        observed = [r["turn1_relevance"] for r in classified_rows if r["primary_family"] == family]
        best_relevance = max(observed, key=lambda x: TURN1_RELEVANCE_RANK.get(x, 0)) if observed else "unknown"
        family_rows.append({
            "primary_family": family,
            "turn1_relevance": best_relevance,
            "weighted_occurrences_in_top_unparsed": total_count,
            "unique_texts_in_top_unparsed": family_unique_texts[family],
            "sample_text": family_samples.get(family, ""),
            "turn1_note": family_notes.get(family, ""),
        })

    write_csv(
        out_dir / "remaining_effect_family_summary.csv",
        family_rows,
        ["primary_family", "turn1_relevance", "weighted_occurrences_in_top_unparsed", "unique_texts_in_top_unparsed", "sample_text", "turn1_note"],
    )

    review_rows: list[dict[str, Any]] = []
    review_family_counts: Counter[str] = Counter()
    review_turn1_counts: Counter[str] = Counter()

    if args.review_queue.exists():
        with args.review_queue.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts = split_unparsed_blob(row.get("unparsed_text", ""))
                families = []
                relevances = []
                for text in texts:
                    primary, all_families, turn1_relevance, _ = classify_text(text)
                    families.append(primary)
                    relevances.append(turn1_relevance)
                    review_family_counts[primary] += 1
                    review_turn1_counts[turn1_relevance] += 1
                primary_family = families[0] if families else "none"
                max_relevance = max(relevances, key=lambda x: TURN1_RELEVANCE_RANK.get(x, 0)) if relevances else "none"
                review_rows.append({
                    "effect_group_id": row.get("effect_group_id", ""),
                    "representative_card_id": row.get("representative_card_id", ""),
                    "name": row.get("name", ""),
                    "supertype": row.get("supertype", ""),
                    "subtypes": row.get("subtypes", ""),
                    "same_effect_printing_count": row.get("same_effect_printing_count", ""),
                    "unparsed_text_count": row.get("unparsed_text_count", ""),
                    "primary_family": primary_family,
                    "turn1_relevance": max_relevance,
                    "all_families": "|".join(sorted(set(families))),
                    "unparsed_text": row.get("unparsed_text", ""),
                })

        write_csv(
            out_dir / "review_queue_classified.csv",
            review_rows,
            [
                "effect_group_id",
                "representative_card_id",
                "name",
                "supertype",
                "subtypes",
                "same_effect_printing_count",
                "unparsed_text_count",
                "primary_family",
                "turn1_relevance",
                "all_families",
                "unparsed_text",
            ],
        )

    summary = {
        "source_coverage": str(args.coverage),
        "source_review_queue": str(args.review_queue) if args.review_queue.exists() else None,
        "coverage": coverage.get("coverage"),
        "status_counts": coverage.get("status_counts"),
        "top_unparsed_items_classified": len(classified_rows),
        "turn1_relevance_weighted_counts": dict(family_turn1_counts),
        "family_summary": family_rows,
        "review_queue_family_counts": review_family_counts.most_common(),
        "review_queue_turn1_relevance_counts": review_turn1_counts.most_common(),
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
        "turn1_relevance_weighted_counts": summary["turn1_relevance_weighted_counts"],
        "top_family_summary": family_rows[:15],
        "outputs": summary["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
