from __future__ import annotations

"""
Build a deck-scoped compiled-card subset for faster Turn 1 simulations.

This reads the full compiled_cards_all.json once, resolves a decklist against it,
and writes a much smaller compiled JSON containing only compiled groups needed
for that deck. Use the subset as --compiled for turn1 scripts to avoid repeatedly
loading all ~15k compiled groups.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_FINDER_PATH = os.path.join(ROOT, "scripts", "run_turn1_target_finder.py")


def load_target_finder_module():
    if not os.path.exists(TARGET_FINDER_PATH):
        raise RuntimeError(f"Missing dependency: {TARGET_FINDER_PATH}")
    spec = importlib.util.spec_from_file_location("turn1_target_finder_module", TARGET_FINDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TARGET_FINDER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_payload(path: Path) -> Dict[str, Any]:
    data = json.load(open(path, encoding="utf-8"))
    if isinstance(data, list):
        return {"schema_version": "pokemon-compiled-card-subset/v1", "compiled_cards": data}
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unsupported compiled JSON shape: {type(data).__name__}")


def group_ids(card: Dict[str, Any]) -> set[str]:
    ids = set()
    for key in ("representative_card_id", "card_id", "id"):
        value = card.get(key)
        if value:
            ids.add(str(value))
    for value in card.get("same_effect_card_ids", []) or []:
        if value:
            ids.add(str(value))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deck-scoped compiled-card subset for faster turn-1 simulation.")
    parser.add_argument("--compiled", default="data/compiled_cards/auto/compiled_cards_all.json")
    parser.add_argument("--decklist", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--include-name-matches", action="store_true", help="Also include compiled groups with the same canonical names as deck cards.")
    args = parser.parse_args()

    tf = load_target_finder_module()
    compiled_path = Path(args.compiled)
    deck_path = Path(args.decklist)
    payload = load_payload(compiled_path)
    compiled_cards = payload.get("compiled_cards") or payload.get("cards") or []

    raw_decklist = tf.parse_decklist(str(deck_path))
    resolved_deck, unresolved = tf.resolve_decklist(raw_decklist, compiled_cards)

    wanted_ids = {tf.card_id(c) for c in resolved_deck if tf.card_id(c)}
    wanted_names = {tf.norm(tf.card_name(c)) for c in resolved_deck if tf.card_name(c)}

    subset: List[Dict[str, Any]] = []
    for card in compiled_cards:
        ids = group_ids(card)
        include = bool(ids & wanted_ids)
        if not include and args.include_name_matches:
            name = ""
            identity = card.get("identity") or {}
            if isinstance(identity, dict):
                name = str(identity.get("name") or identity.get("canonical_name") or "")
            include = bool(name and tf.norm(name) in wanted_names)
        if include:
            subset.append(card)

    if args.out:
        out_path = Path(args.out)
    else:
        safe_name = deck_path.stem.replace("_card_ids", "")
        out_path = Path("data/compiled_cards/deck_subsets") / f"{safe_name}_compiled_subset.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "schema_version": "pokemon-compiled-card-subset/v1",
        "source_compiled": str(compiled_path),
        "source_decklist": str(deck_path),
        "compiled_cards": subset,
        "subset_metadata": {
            "deck_entries": len(raw_decklist),
            "resolved_deck_size": len(resolved_deck),
            "unresolved": unresolved,
            "unique_requested_ids": sorted(wanted_ids),
            "compiled_groups_in_source": len(compiled_cards),
            "compiled_groups_in_subset": len(subset),
        },
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "passed": not unresolved,
        "out": str(out_path),
        "resolved_deck_size": len(resolved_deck),
        "unresolved": unresolved,
        "compiled_groups_in_source": len(compiled_cards),
        "compiled_groups_in_subset": len(subset),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
