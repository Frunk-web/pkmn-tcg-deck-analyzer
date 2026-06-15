#!/usr/bin/env python3
"""
Analyze compiled Pokemon TCG card JSON for simulator-readiness.

This script is intentionally about *readiness*, not full game accuracy. It keeps
older global compiler/simulator coverage outputs and adds Turn-1 card-access
coverage views:

- parser status coverage
- compiled effect kinds
- primitive operation usage
- complete-card catalog for early simulator testing
- partial-card review samples
- Turn-1 relevant op coverage
- Turn-1 relevant partial cards
- coverage by era/year/series/regulation mark/supertype
- optional deck-specific readiness report

Run locally from the project root:

python scripts/analyze_simulator_readiness.py \
  --compiled data/compiled_cards/auto/compiled_cards_all.json \
  --out-dir data/reports/simulator_readiness

Optional deck-specific report:

python scripts/analyze_simulator_readiness.py \
  --compiled data/compiled_cards/auto/compiled_cards_all.json \
  --decklist data/decks/mega_lucario_turn1_deck_card_ids.txt \
  --out-dir data/reports/simulator_readiness
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

CORE_SIMULATOR_OPS = [
    "reference_global_rule",
    "draw_cards",
    "search_deck",
    "reveal_cards",
    "shuffle_deck",
    "move_card",
    "move_cards",
    "move_zone_to_zone",
    "put_card_into_hand",
    "put_card_on_bench",
    "attach_card",
    "attach_energy",
    "discard_card",
    "discard_cards",
    "discard_energy",
    "move_energy",
    "deal_attack_damage",
    "modify_attack_damage",
    "place_damage_counters",
    "heal_damage",
    "apply_special_condition",
    "remove_special_condition",
    "switch_active",
    "coin_flip",
    "flip_coins",
    "branch_on_result",
    "choose_target",
    "choose_amount",
    "choose_cards",
    "register_continuous_modifier",
    "register_trigger",
    "register_replacement_effect",
]

# Ops that can change whether a player can access cards / setup pieces by the
# end of the first turn. Damage-only operations are intentionally excluded.
TURN1_CARD_ACCESS_OPS = {
    "reference_global_rule",
    "register_usage_limit",
    "play_condition",
    "draw_cards",
    "draw_until_hand_size",
    "draw_until_hand_size_matches",
    "search_deck",
    "reveal_cards",
    "look_at_cards",
    "look_at_top_cards",
    "look_at_bottom_cards",
    "reorder_cards",
    "shuffle_deck",
    "choose_cards",
    "choose_card",
    "choose_target",
    "choose_amount",
    "move_card",
    "move_cards",
    "move_zone_to_zone",
    "put_card_into_hand",
    "put_card_on_bench",
    "play_card_to_bench",
    "attach_card",
    "attach_energy",
    "move_energy",
    "discard_card",
    "discard_cards",
    "discard_energy",
    "return_to_hand",
    "return_to_deck",
    "switch_active",
    "evolve_pokemon",
    "devolve_pokemon",
    "coin_flip",
    "flip_coins",
    "branch_on_result",
    "set_variable",
    "count_cards",
}

TURN1_RELEVANT_EFFECT_KINDS = {
    "ability",
    "ability_activated",
    "ability_triggered",
    "trainer_item",
    "trainer_supporter",
    "trainer_stadium",
    "trainer_tool",
    "energy",
    "special_energy",
    "rule",
}

TURN1_IRRELEVANT_OR_LOW_OPS = {
    "deal_attack_damage",
    "modify_attack_damage",
    "place_damage_counters",
    "heal_damage",
    "apply_special_condition",
    "remove_special_condition",
    "prevent_damage",
    "reduce_damage",
}

TURN1_UNPARSED_PATTERNS: list[tuple[str, str, list[re.Pattern[str]]]] = [
    (
        "search_or_tutor",
        "high",
        [
            re.compile(r"search your deck", re.I),
            re.compile(r"look at the top \d+ cards", re.I),
            re.compile(r"look at the bottom \d+ cards", re.I),
            re.compile(r"put .* into your hand", re.I),
        ],
    ),
    (
        "draw_or_hand_refresh",
        "high",
        [
            re.compile(r"draw \d+ cards?", re.I),
            re.compile(r"draw a card", re.I),
            re.compile(r"draw cards until", re.I),
            re.compile(r"shuffle your hand into your deck", re.I),
        ],
    ),
    (
        "energy_setup_or_cost",
        "high",
        [
            re.compile(r"attach .* Energy", re.I),
            re.compile(r"discard .* Energy", re.I),
            re.compile(r"move .* Energy", re.I),
            re.compile(r"Basic [A-Za-z]+ Energy", re.I),
        ],
    ),
    (
        "bench_or_play_pokemon",
        "high",
        [
            re.compile(r"put .* (?:on|onto) your Bench", re.I),
            re.compile(r"put .* Basic .* onto your Bench", re.I),
            re.compile(r"have .* in play", re.I),
        ],
    ),
    (
        "evolution_or_rare_candy_like",
        "high",
        [
            re.compile(r"evolve", re.I),
            re.compile(r"evolves? from", re.I),
            re.compile(r"devolve", re.I),
        ],
    ),
    (
        "switch_or_retreat",
        "medium",
        [
            re.compile(r"switch .* Active", re.I),
            re.compile(r"Retreat Cost", re.I),
            re.compile(r"new Active", re.I),
        ],
    ),
    (
        "usage_limit_or_turn_condition",
        "medium",
        [
            re.compile(r"once during your turn", re.I),
            re.compile(r"can't use more than 1", re.I),
            re.compile(r"during your first turn", re.I),
            re.compile(r"if you have .* in play", re.I),
        ],
    ),
    (
        "prize_or_setup_exception",
        "medium",
        [
            re.compile(r"Prize cards?", re.I),
            re.compile(r"as your Active Pokémon", re.I),
            re.compile(r"when you play this Pokémon", re.I),
        ],
    ),
    (
        "damage_or_battle_only",
        "low",
        [
            re.compile(r"damage", re.I),
            re.compile(r"damage counters?", re.I),
            re.compile(r"Special Condition", re.I),
            re.compile(r"Weakness|Resistance", re.I),
            re.compile(r"Knocked Out", re.I),
        ],
    ),
]

ERA_BUCKETS = [
    (1998, 2003, "1998-2003 early/classic"),
    (2004, 2010, "2004-2010 ex/dppt/hgss"),
    (2011, 2016, "2011-2016 bw/xy"),
    (2017, 2022, "2017-2022 sm/swsh"),
    (2023, 2100, "2023+ sv/current"),
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def iter_nested_steps(obj: Any) -> Iterable[Dict[str, Any]]:
    """Yield every dict that looks like a compiled step from nested effect objects."""
    if isinstance(obj, dict):
        if "op" in obj and isinstance(obj.get("op"), str):
            yield obj
        for v in obj.values():
            yield from iter_nested_steps(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from iter_nested_steps(item)


def flatten_texts(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: List[str] = []
        for x in value:
            out.extend(flatten_texts(x))
        return out
    if isinstance(value, dict):
        out: List[str] = []
        for x in value.values():
            out.extend(flatten_texts(x))
        return out
    return [str(value)]


def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def card_status(card: Dict[str, Any]) -> str:
    return (card.get("parser") or {}).get("status") or "unknown"


def card_supertype(card: Dict[str, Any]) -> str:
    return ((card.get("identity") or {}).get("supertype") or "unknown")


def card_name(card: Dict[str, Any]) -> str:
    return ((card.get("identity") or {}).get("name") or card.get("representative_card_id") or "unknown")


def card_subtypes(card: Dict[str, Any]) -> str:
    return "|".join((card.get("identity") or {}).get("subtypes") or [])


def card_types(card: Dict[str, Any]) -> str:
    return "|".join((card.get("identity") or {}).get("types") or [])


def effect_kinds(card: Dict[str, Any]) -> set[str]:
    return {e.get("kind") or "unknown" for e in (card.get("compiled_effects") or [])}


def card_ops(card: Dict[str, Any]) -> set[str]:
    ops: set[str] = set()
    for effect in card.get("compiled_effects") or []:
        for step in iter_nested_steps(effect.get("steps", [])):
            ops.add(step.get("op") or "unknown")
    return ops


def card_unparsed_texts(card: Dict[str, Any]) -> list[str]:
    return [normalize_space(x) for x in flatten_texts((card.get("parser") or {}).get("unparsed_text") or []) if normalize_space(x)]


def get_nested(obj: Any, *paths: Sequence[str]) -> Any:
    for path in paths:
        cur = obj
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok and cur not in (None, "", []):
            return cur
    return None


def recursive_find_key(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and v not in (None, "", []):
                return v
        for v in obj.values():
            found = recursive_find_key(v, keys)
            if found not in (None, "", []):
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = recursive_find_key(v, keys)
            if found not in (None, "", []):
                return found
    return None


def card_release_year(card: Dict[str, Any]) -> Optional[int]:
    raw = get_nested(
        card,
        ("printed", "release_date"),
        ("printed", "set_release_date"),
        ("printed", "set", "release_date"),
        ("sources", "raw_api_card", "set", "releaseDate"),
        ("sources", "raw_api_card", "set", "release_date"),
    )
    if raw is None:
        raw = recursive_find_key(card.get("same_effect_printings") or [], {"releaseDate", "release_date", "set_release_date"})
    if raw is None:
        raw = recursive_find_key(card, {"releaseDate", "release_date", "set_release_date"})
    if raw is None:
        return None
    match = re.search(r"(19\d{2}|20\d{2})", str(raw))
    return int(match.group(1)) if match else None


def era_for_year(year: Optional[int]) -> str:
    if year is None:
        return "unknown"
    for start, end, label in ERA_BUCKETS:
        if start <= year <= end:
            return label
    return "unknown"


def card_series(card: Dict[str, Any]) -> str:
    raw = get_nested(
        card,
        ("printed", "series"),
        ("printed", "set_series"),
        ("printed", "set", "series"),
        ("sources", "raw_api_card", "set", "series"),
    )
    if raw is None:
        raw = recursive_find_key(card.get("same_effect_printings") or [], {"series", "set_series"})
    return str(raw or "unknown")


def card_set_name(card: Dict[str, Any]) -> str:
    raw = get_nested(
        card,
        ("printed", "set_name"),
        ("printed", "set", "name"),
        ("sources", "raw_api_card", "set", "name"),
    )
    if raw is None:
        raw = recursive_find_key(card.get("same_effect_printings") or [], {"set_name"})
    return str(raw or "unknown")


def card_regulation_mark(card: Dict[str, Any]) -> str:
    raw = get_nested(
        card,
        ("gameplay", "regulation_mark"),
        ("printed", "regulation_mark"),
        ("sources", "raw_api_card", "regulationMark"),
        ("sources", "raw_api_card", "regulation_mark"),
    )
    if raw is None:
        raw = recursive_find_key(card.get("same_effect_printings") or [], {"regulationMark", "regulation_mark"})
    return str(raw or "unknown")


def classify_unparsed_turn1(text: str) -> tuple[str, str]:
    text = normalize_space(text)
    for family, relevance, patterns in TURN1_UNPARSED_PATTERNS:
        if any(p.search(text) for p in patterns):
            return family, relevance
    return "other_long_tail", "unknown"


def max_turn1_relevance(labels: Iterable[str]) -> str:
    rank = {"high": 4, "medium": 3, "low": 2, "unknown": 1, "none": 0}
    best = "none"
    for label in labels:
        if rank.get(label, 0) > rank.get(best, 0):
            best = label
    return best


def card_turn1_profile(card: Dict[str, Any]) -> Dict[str, Any]:
    ops = card_ops(card)
    kinds = effect_kinds(card)
    unparsed_texts = card_unparsed_texts(card)
    unparsed_classifications = [classify_unparsed_turn1(t) for t in unparsed_texts]
    unparsed_families = sorted({family for family, _ in unparsed_classifications})
    unparsed_relevance = max_turn1_relevance(relevance for _, relevance in unparsed_classifications)

    has_turn1_op = bool(ops & TURN1_CARD_ACCESS_OPS)
    has_low_or_battle_only_ops = bool(ops & TURN1_IRRELEVANT_OR_LOW_OPS)
    has_turn1_kind = bool(kinds & TURN1_RELEVANT_EFFECT_KINDS)

    if card_status(card) == "complete" and has_turn1_op:
        turn1_status = "ready"
    elif card_status(card) == "complete" and has_turn1_kind:
        turn1_status = "probably_ready_or_irrelevant"
    elif card_status(card) == "partial" and unparsed_relevance in {"high", "medium"}:
        turn1_status = "blocked_by_turn1_relevant_unparsed_text"
    elif card_status(card) == "partial" and has_turn1_op:
        turn1_status = "partially_ready_review_needed"
    elif card_status(card) == "partial" and unparsed_relevance in {"low", "unknown"}:
        turn1_status = "partial_but_likely_low_turn1_impact"
    elif has_low_or_battle_only_ops:
        turn1_status = "battle_effects_only_or_low_turn1_impact"
    else:
        turn1_status = "unknown_or_irrelevant"

    return {
        "ops": ops,
        "kinds": kinds,
        "unparsed_families": unparsed_families,
        "unparsed_relevance": unparsed_relevance,
        "turn1_status": turn1_status,
        "has_turn1_op": has_turn1_op,
    }


def summarize_groups(cards: list[Dict[str, Any]], group_fn) -> list[dict[str, Any]]:
    buckets: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for card in cards:
        buckets[str(group_fn(card))].append(card)

    rows = []
    for group, group_cards in sorted(buckets.items(), key=lambda kv: kv[0]):
        total = len(group_cards)
        parser_counts = Counter(card_status(c) for c in group_cards)
        turn1_counts = Counter(card_turn1_profile(c)["turn1_status"] for c in group_cards)
        high_blocked = turn1_counts.get("blocked_by_turn1_relevant_unparsed_text", 0)
        rows.append({
            "group": group,
            "cards": total,
            "complete": parser_counts.get("complete", 0),
            "partial": parser_counts.get("partial", 0),
            "complete_rate": round(parser_counts.get("complete", 0) / total, 4) if total else 0.0,
            "turn1_ready": turn1_counts.get("ready", 0),
            "turn1_ready_or_probably_ready": turn1_counts.get("ready", 0) + turn1_counts.get("probably_ready_or_irrelevant", 0),
            "turn1_blocked_high_or_medium": high_blocked,
            "turn1_blocked_rate": round(high_blocked / total, 4) if total else 0.0,
            "turn1_status_counts": json.dumps(dict(turn1_counts), ensure_ascii=False, sort_keys=True),
        })
    return rows


def read_decklist_ids(path: Optional[Path]) -> list[str]:
    if path is None or not path.exists():
        return []
    ids: list[str] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                ids.extend([parts[1]] * int(parts[0]))
            elif len(parts) >= 1:
                ids.append(parts[0])
    return ids


def build_card_id_index(cards: list[Dict[str, Any]]) -> dict[str, Dict[str, Any]]:
    index: dict[str, Dict[str, Any]] = {}
    for card in cards:
        rep = card.get("representative_card_id")
        if rep:
            index[str(rep)] = card
        for cid in card.get("same_effect_card_ids") or []:
            index[str(cid)] = card
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", default="data/compiled_cards/auto/compiled_cards_all.json", help="Compiled JSON produced by compile_cards_auto.py")
    parser.add_argument("--out-dir", default="data/reports/simulator_readiness", help="Output directory")
    parser.add_argument("--complete-only-export", default="complete_cards_for_sim_seed.json", help="Filename for complete cards export")
    parser.add_argument("--decklist", default=None, help="Optional decklist of card IDs for a deck-specific readiness report")
    args = parser.parse_args()

    compiled_path = Path(args.compiled)
    out_dir = Path(args.out_dir)
    decklist_path = Path(args.decklist) if args.decklist else None

    root = read_json(compiled_path)
    cards = root.get("compiled_cards", []) if isinstance(root, dict) else root
    if not isinstance(cards, list):
        raise SystemExit("Expected compiled JSON to contain a compiled_cards list or be a list.")

    status_counts = Counter(card_status(c) for c in cards)
    supertype_counts = Counter(card_supertype(c) for c in cards)
    status_by_supertype: Dict[Tuple[str, str], int] = Counter((card_supertype(c), card_status(c)) for c in cards)

    effect_kind_counts = Counter()
    op_counts = Counter()
    op_by_kind = Counter()
    steps_total = 0
    effects_total = 0
    cards_with_no_effects = []
    cards_with_unparsed = []
    complete_cards = []

    for card in cards:
        effects = card.get("compiled_effects") or []
        effects_total += len(effects)
        if not effects:
            cards_with_no_effects.append(card)
        for effect in effects:
            kind = effect.get("kind") or "unknown"
            effect_kind_counts[kind] += 1
            for step in iter_nested_steps(effect.get("steps", [])):
                op = step.get("op") or "unknown"
                op_counts[op] += 1
                op_by_kind[(kind, op)] += 1
                steps_total += 1
        unparsed = (card.get("parser") or {}).get("unparsed_text") or []
        if unparsed:
            cards_with_unparsed.append(card)
        if card_status(card) == "complete":
            complete_cards.append(card)

    total_cards = len(cards)
    complete_count = status_counts.get("complete", 0)
    partial_count = status_counts.get("partial", 0)
    complete_rate = round(complete_count / total_cards, 4) if total_cards else 0.0
    partial_rate = round(partial_count / total_cards, 4) if total_cards else 0.0

    core_op_rows = []
    for op in CORE_SIMULATOR_OPS:
        count = op_counts.get(op, 0)
        core_op_rows.append({
            "op": op,
            "compiled_occurrences": count,
            "present_in_compiled_cards": count > 0,
            "turn1_card_access_relevant": "yes" if op in TURN1_CARD_ACCESS_OPS else "no",
            "suggested_runtime_priority": "high" if count >= 50 else ("medium" if count >= 10 else "low"),
        })

    top_op_rows = [
        {
            "op": op,
            "count": count,
            "is_core_runtime_op": op in CORE_SIMULATOR_OPS,
            "turn1_card_access_relevant": "yes" if op in TURN1_CARD_ACCESS_OPS else "no",
        }
        for op, count in op_counts.most_common()
    ]
    top_kind_rows = [{"effect_kind": kind, "count": count} for kind, count in effect_kind_counts.most_common()]
    status_rows = [{"supertype": st, "status": status, "count": count} for (st, status), count in sorted(status_by_supertype.items())]

    complete_catalog_rows = []
    for card in complete_cards:
        kinds = sorted(effect_kinds(card))
        ops = sorted(card_ops(card))
        complete_catalog_rows.append({
            "effect_group_id": card.get("effect_group_id", ""),
            "representative_card_id": card.get("representative_card_id", ""),
            "name": card_name(card),
            "supertype": card_supertype(card),
            "subtypes": card_subtypes(card),
            "same_effect_printing_count": card.get("same_effect_printing_count", ""),
            "effect_count": len(card.get("compiled_effects") or []),
            "effect_kinds": "|".join(kinds),
            "ops": "|".join(ops),
        })

    partial_sample_rows = []
    for card in cards_with_unparsed[:500]:
        unparsed = card_unparsed_texts(card)
        profile = card_turn1_profile(card)
        partial_sample_rows.append({
            "effect_group_id": card.get("effect_group_id", ""),
            "representative_card_id": card.get("representative_card_id", ""),
            "name": card_name(card),
            "supertype": card_supertype(card),
            "subtypes": card_subtypes(card),
            "unparsed_text_count": len(unparsed),
            "turn1_status": profile["turn1_status"],
            "turn1_unparsed_relevance": profile["unparsed_relevance"],
            "turn1_unparsed_families": "|".join(profile["unparsed_families"]),
            "unparsed_text": " || ".join(unparsed)[:4000],
        })

    turn1_card_rows = []
    for card in cards:
        profile = card_turn1_profile(card)
        year = card_release_year(card)
        turn1_card_rows.append({
            "effect_group_id": card.get("effect_group_id", ""),
            "representative_card_id": card.get("representative_card_id", ""),
            "name": card_name(card),
            "supertype": card_supertype(card),
            "subtypes": card_subtypes(card),
            "types": card_types(card),
            "parser_status": card_status(card),
            "turn1_status": profile["turn1_status"],
            "turn1_unparsed_relevance": profile["unparsed_relevance"],
            "turn1_unparsed_families": "|".join(profile["unparsed_families"]),
            "has_turn1_compiled_op": "yes" if profile["has_turn1_op"] else "no",
            "ops": "|".join(sorted(profile["ops"])),
            "effect_kinds": "|".join(sorted(profile["kinds"])),
            "release_year": year or "",
            "era": era_for_year(year),
            "series": card_series(card),
            "set_name": card_set_name(card),
            "regulation_mark": card_regulation_mark(card),
            "unparsed_text": " || ".join(card_unparsed_texts(card))[:4000],
        })

    turn1_relevant_partial_rows = [
        r for r in turn1_card_rows
        if r["parser_status"] == "partial" and r["turn1_unparsed_relevance"] in {"high", "medium"}
    ]

    turn1_status_counts = Counter(r["turn1_status"] for r in turn1_card_rows)
    turn1_blocked_count = turn1_status_counts.get("blocked_by_turn1_relevant_unparsed_text", 0)
    turn1_ready_count = turn1_status_counts.get("ready", 0)

    by_era_rows = summarize_groups(cards, lambda c: era_for_year(card_release_year(c)))
    by_year_rows = summarize_groups(cards, lambda c: card_release_year(c) or "unknown")
    by_series_rows = summarize_groups(cards, card_series)
    by_regulation_rows = summarize_groups(cards, card_regulation_mark)
    by_supertype_rows = summarize_groups(cards, card_supertype)

    deck_rows: list[dict[str, Any]] = []
    deck_ids = read_decklist_ids(decklist_path)
    if deck_ids:
        index = build_card_id_index(cards)
        deck_id_counts = Counter(deck_ids)
        seen_groups: set[str] = set()
        for cid, qty in sorted(deck_id_counts.items()):
            card = index.get(cid)
            if not card:
                deck_rows.append({"card_id": cid, "count": qty, "found_in_compiled": "no"})
                continue
            key = str(card.get("effect_group_id") or card.get("representative_card_id") or cid)
            if key in seen_groups:
                continue
            seen_groups.add(key)
            profile = card_turn1_profile(card)
            year = card_release_year(card)
            deck_rows.append({
                "card_id": cid,
                "count": qty,
                "found_in_compiled": "yes",
                "representative_card_id": card.get("representative_card_id", ""),
                "name": card_name(card),
                "supertype": card_supertype(card),
                "subtypes": card_subtypes(card),
                "parser_status": card_status(card),
                "turn1_status": profile["turn1_status"],
                "turn1_unparsed_relevance": profile["unparsed_relevance"],
                "turn1_unparsed_families": "|".join(profile["unparsed_families"]),
                "ops": "|".join(sorted(profile["ops"])),
                "release_year": year or "",
                "era": era_for_year(year),
                "series": card_series(card),
                "regulation_mark": card_regulation_mark(card),
                "unparsed_text": " || ".join(card_unparsed_texts(card))[:4000],
            })

    summary = {
        "source_compiled": str(compiled_path),
        "source_decklist": str(decklist_path) if decklist_path else None,
        "schema_version": root.get("schema_version") if isinstance(root, dict) else None,
        "compiler_version": root.get("compiler_version") if isinstance(root, dict) else None,
        "cards_total": total_cards,
        "effects_total": effects_total,
        "compiled_steps_total": steps_total,
        "unique_ops": len(op_counts),
        "status_counts": dict(status_counts),
        "supertype_counts": dict(supertype_counts),
        "coverage": {
            "complete_rate": complete_rate,
            "partial_rate": partial_rate,
        },
        "turn1_card_access_readiness": {
            "status_counts": dict(turn1_status_counts),
            "ready_cards": turn1_ready_count,
            "blocked_by_turn1_relevant_unparsed_text": turn1_blocked_count,
            "blocked_by_turn1_relevant_unparsed_text_rate": round(turn1_blocked_count / total_cards, 4) if total_cards else 0.0,
            "partial_cards_with_high_or_medium_turn1_relevance": len(turn1_relevant_partial_rows),
            "note": "Turn-1 readiness focuses on draw/search/setup access effects, not full battle simulation.",
        },
        "complete_cards_available_for_sim_seed": complete_count,
        "cards_with_no_compiled_effects": len(cards_with_no_effects),
        "cards_with_unparsed_text": len(cards_with_unparsed),
        "top_ops": top_op_rows[:30],
        "top_effect_kinds": top_kind_rows[:30],
        "recommended_next_step": (
            "Use turn1_relevant_partial_cards.csv and the high-frequency rows in compiled_op_counts.csv "
            "to prioritize compiler/runtime work for Turn-1 consistency. Keep full battle effects as a separate roadmap."
        ),
        "outputs": {
            "summary": str(out_dir / "simulator_readiness_summary.json"),
            "turn1_readiness_summary": str(out_dir / "turn1_readiness_summary.json"),
            "core_runtime_op_priority": str(out_dir / "core_runtime_op_priority.csv"),
            "compiled_op_counts": str(out_dir / "compiled_op_counts.csv"),
            "effect_kind_counts": str(out_dir / "effect_kind_counts.csv"),
            "status_by_supertype": str(out_dir / "status_by_supertype.csv"),
            "complete_cards_catalog": str(out_dir / "complete_cards_catalog.csv"),
            "partial_cards_sample": str(out_dir / "partial_cards_sample.csv"),
            "turn1_card_access_cards": str(out_dir / "turn1_card_access_cards.csv"),
            "turn1_relevant_partial_cards": str(out_dir / "turn1_relevant_partial_cards.csv"),
            "turn1_coverage_by_era": str(out_dir / "turn1_coverage_by_era.csv"),
            "turn1_coverage_by_year": str(out_dir / "turn1_coverage_by_year.csv"),
            "turn1_coverage_by_series": str(out_dir / "turn1_coverage_by_series.csv"),
            "turn1_coverage_by_regulation_mark": str(out_dir / "turn1_coverage_by_regulation_mark.csv"),
            "turn1_coverage_by_supertype": str(out_dir / "turn1_coverage_by_supertype.csv"),
            "turn1_deck_readiness": str(out_dir / "turn1_deck_readiness.csv") if deck_rows else None,
            "complete_cards_for_sim_seed": str(out_dir / args.complete_only_export),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "simulator_readiness_summary.json", summary)
    write_json(out_dir / "turn1_readiness_summary.json", {
        "source_compiled": summary["source_compiled"],
        "source_decklist": summary["source_decklist"],
        "cards_total": total_cards,
        "compiler_version": summary["compiler_version"],
        "coverage": summary["coverage"],
        "turn1_card_access_readiness": summary["turn1_card_access_readiness"],
        "top_turn1_relevant_ops": [r for r in top_op_rows if r["turn1_card_access_relevant"] == "yes"][:30],
        "outputs": summary["outputs"],
    })

    write_csv(out_dir / "core_runtime_op_priority.csv", core_op_rows, ["op", "compiled_occurrences", "present_in_compiled_cards", "turn1_card_access_relevant", "suggested_runtime_priority"])
    write_csv(out_dir / "compiled_op_counts.csv", top_op_rows, ["op", "count", "is_core_runtime_op", "turn1_card_access_relevant"])
    write_csv(out_dir / "effect_kind_counts.csv", top_kind_rows, ["effect_kind", "count"])
    write_csv(out_dir / "status_by_supertype.csv", status_rows, ["supertype", "status", "count"])
    write_csv(out_dir / "complete_cards_catalog.csv", complete_catalog_rows, ["effect_group_id", "representative_card_id", "name", "supertype", "subtypes", "same_effect_printing_count", "effect_count", "effect_kinds", "ops"])
    write_csv(out_dir / "partial_cards_sample.csv", partial_sample_rows, ["effect_group_id", "representative_card_id", "name", "supertype", "subtypes", "unparsed_text_count", "turn1_status", "turn1_unparsed_relevance", "turn1_unparsed_families", "unparsed_text"])

    turn1_card_fields = [
        "effect_group_id", "representative_card_id", "name", "supertype", "subtypes", "types",
        "parser_status", "turn1_status", "turn1_unparsed_relevance", "turn1_unparsed_families",
        "has_turn1_compiled_op", "ops", "effect_kinds", "release_year", "era", "series",
        "set_name", "regulation_mark", "unparsed_text",
    ]
    write_csv(out_dir / "turn1_card_access_cards.csv", turn1_card_rows, turn1_card_fields)
    write_csv(out_dir / "turn1_relevant_partial_cards.csv", turn1_relevant_partial_rows, turn1_card_fields)

    group_fields = ["group", "cards", "complete", "partial", "complete_rate", "turn1_ready", "turn1_ready_or_probably_ready", "turn1_blocked_high_or_medium", "turn1_blocked_rate", "turn1_status_counts"]
    write_csv(out_dir / "turn1_coverage_by_era.csv", by_era_rows, group_fields)
    write_csv(out_dir / "turn1_coverage_by_year.csv", by_year_rows, group_fields)
    write_csv(out_dir / "turn1_coverage_by_series.csv", by_series_rows, group_fields)
    write_csv(out_dir / "turn1_coverage_by_regulation_mark.csv", by_regulation_rows, group_fields)
    write_csv(out_dir / "turn1_coverage_by_supertype.csv", by_supertype_rows, group_fields)

    if deck_rows:
        write_csv(out_dir / "turn1_deck_readiness.csv", deck_rows, [
            "card_id", "count", "found_in_compiled", "representative_card_id", "name", "supertype", "subtypes",
            "parser_status", "turn1_status", "turn1_unparsed_relevance", "turn1_unparsed_families",
            "ops", "release_year", "era", "series", "regulation_mark", "unparsed_text",
        ])

    seed_root = dict(root) if isinstance(root, dict) else {"schema_version": None, "compiled_cards": cards}
    seed_root["compiled_cards"] = complete_cards
    seed_root["simulator_seed_note"] = "Complete parser-status cards only. Use this as the initial simulator card pool."
    write_json(out_dir / args.complete_only_export, seed_root)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
