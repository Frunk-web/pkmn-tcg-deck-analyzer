#!/usr/bin/env python3
"""
Analyze compiled Pokemon TCG card JSON for simulator-readiness.

Reads the output from scripts/compile_cards_auto.py and summarizes:
- parser status coverage
- compiled effect kinds
- primitive operation usage
- complete-card catalog for early simulator testing
- partial-card review samples

Designed to run locally from the project root:

python scripts/analyze_simulator_readiness.py \
  --compiled data/compiled_cards/auto/compiled_cards_all.json \
  --out-dir data/reports/simulator_readiness
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


CORE_SIMULATOR_OPS = [
    "reference_global_rule",
    "draw_cards",
    "search_deck",
    "reveal_cards",
    "shuffle_deck",
    "move_card",
    "put_card_into_hand",
    "put_card_on_bench",
    "attach_card",
    "attach_energy",
    "discard_card",
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
    "register_continuous_modifier",
    "register_trigger",
    "register_replacement_effect",
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


def card_status(card: Dict[str, Any]) -> str:
    return (card.get("parser") or {}).get("status") or "unknown"


def card_supertype(card: Dict[str, Any]) -> str:
    return ((card.get("identity") or {}).get("supertype") or "unknown")


def card_name(card: Dict[str, Any]) -> str:
    return ((card.get("identity") or {}).get("name") or card.get("representative_card_id") or "unknown")


def card_subtypes(card: Dict[str, Any]) -> str:
    return "|".join((card.get("identity") or {}).get("subtypes") or [])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiled", default="data/compiled_cards/auto/compiled_cards_all.json", help="Compiled JSON produced by compile_cards_auto.py")
    parser.add_argument("--out-dir", default="data/reports/simulator_readiness", help="Output directory")
    parser.add_argument("--complete-only-export", default="complete_cards_for_sim_seed.json", help="Filename for complete cards export")
    args = parser.parse_args()

    compiled_path = Path(args.compiled)
    out_dir = Path(args.out_dir)
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
        core_op_rows.append({
            "op": op,
            "compiled_occurrences": op_counts.get(op, 0),
            "present_in_compiled_cards": op_counts.get(op, 0) > 0,
            "suggested_runtime_priority": "high" if op_counts.get(op, 0) >= 50 else ("medium" if op_counts.get(op, 0) >= 10 else "low"),
        })

    top_op_rows = [
        {"op": op, "count": count, "is_core_runtime_op": op in CORE_SIMULATOR_OPS}
        for op, count in op_counts.most_common()
    ]

    top_kind_rows = [
        {"effect_kind": kind, "count": count}
        for kind, count in effect_kind_counts.most_common()
    ]

    status_rows = [
        {
            "supertype": st,
            "status": status,
            "count": count,
        }
        for (st, status), count in sorted(status_by_supertype.items())
    ]

    complete_catalog_rows = []
    for card in complete_cards:
        kinds = sorted({e.get("kind") or "unknown" for e in (card.get("compiled_effects") or [])})
        ops = sorted({s.get("op") or "unknown" for e in (card.get("compiled_effects") or []) for s in iter_nested_steps(e.get("steps", []))})
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
        unparsed = (card.get("parser") or {}).get("unparsed_text") or []
        partial_sample_rows.append({
            "effect_group_id": card.get("effect_group_id", ""),
            "representative_card_id": card.get("representative_card_id", ""),
            "name": card_name(card),
            "supertype": card_supertype(card),
            "subtypes": card_subtypes(card),
            "unparsed_text_count": len(unparsed),
            "unparsed_text": " || ".join(flatten_texts(unparsed))[:4000],
        })

    summary = {
        "source_compiled": str(compiled_path),
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
        "complete_cards_available_for_sim_seed": complete_count,
        "cards_with_no_compiled_effects": len(cards_with_no_effects),
        "cards_with_unparsed_text": len(cards_with_unparsed),
        "top_ops": top_op_rows[:30],
        "top_effect_kinds": top_kind_rows[:30],
        "recommended_next_step": (
            "Start a minimal simulator using complete_cards_for_sim_seed.json and implement runtime handlers "
            "for the high-frequency core ops in core_runtime_op_priority.csv. Keep the partial cards as review/backlog."
        ),
        "outputs": {
            "summary": str(out_dir / "simulator_readiness_summary.json"),
            "core_runtime_op_priority": str(out_dir / "core_runtime_op_priority.csv"),
            "compiled_op_counts": str(out_dir / "compiled_op_counts.csv"),
            "effect_kind_counts": str(out_dir / "effect_kind_counts.csv"),
            "status_by_supertype": str(out_dir / "status_by_supertype.csv"),
            "complete_cards_catalog": str(out_dir / "complete_cards_catalog.csv"),
            "partial_cards_sample": str(out_dir / "partial_cards_sample.csv"),
            "complete_cards_for_sim_seed": str(out_dir / args.complete_only_export),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "simulator_readiness_summary.json", summary)
    write_csv(out_dir / "core_runtime_op_priority.csv", core_op_rows, ["op", "compiled_occurrences", "present_in_compiled_cards", "suggested_runtime_priority"])
    write_csv(out_dir / "compiled_op_counts.csv", top_op_rows, ["op", "count", "is_core_runtime_op"])
    write_csv(out_dir / "effect_kind_counts.csv", top_kind_rows, ["effect_kind", "count"])
    write_csv(out_dir / "status_by_supertype.csv", status_rows, ["supertype", "status", "count"])
    write_csv(out_dir / "complete_cards_catalog.csv", complete_catalog_rows, ["effect_group_id", "representative_card_id", "name", "supertype", "subtypes", "same_effect_printing_count", "effect_count", "effect_kinds", "ops"])
    write_csv(out_dir / "partial_cards_sample.csv", partial_sample_rows, ["effect_group_id", "representative_card_id", "name", "supertype", "subtypes", "unparsed_text_count", "unparsed_text"])

    # Keep the root metadata but only complete cards for the simulator seed export.
    seed_root = dict(root) if isinstance(root, dict) else {"schema_version": None, "compiled_cards": cards}
    seed_root["compiled_cards"] = complete_cards
    seed_root["simulator_seed_note"] = "Complete parser-status cards only. Use this as the initial simulator card pool."
    write_json(out_dir / args.complete_only_export, seed_root)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
