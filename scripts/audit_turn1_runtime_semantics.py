from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def get_cards(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        for key in ("compiled_cards", "cards", "data"):
            if isinstance(raw.get(key), list):
                return raw[key]
    if isinstance(raw, list):
        return raw
    return []


def card_name(card: Dict[str, Any]) -> str:
    return str(card.get("name") or card.get("card_name") or "")


def effect_name(effect: Dict[str, Any]) -> str:
    src = effect.get("source") if isinstance(effect.get("source"), dict) else {}
    return str(src.get("name") or "")


def effect_text(effect: Dict[str, Any]) -> str:
    src = effect.get("source") if isinstance(effect.get("source"), dict) else {}
    return str(src.get("text") or "")


def ops(effect: Dict[str, Any]) -> str:
    return ",".join(str(s.get("op") or "") for s in (effect.get("steps") or []) if isinstance(s, dict))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled", required=True)
    ap.add_argument("--examples", type=int, default=40)
    args = ap.parse_args()

    raw = json.loads(Path(args.compiled).read_text(encoding="utf-8"))
    cards = get_cards(raw)
    patch_counts = Counter()
    support_counts = Counter()
    by_effect_type = Counter()
    examples = []

    for card in cards:
        for effect in card.get("compiled_effects", []) or []:
            if not isinstance(effect, dict):
                continue
            rt = effect.get("turn1_runtime")
            if not isinstance(rt, dict):
                continue
            for p in rt.get("patches") or []:
                patch_counts[str(p)] += 1
            support_counts[str(rt.get("runtime_support") or "unknown")] += 1
            by_effect_type[str(rt.get("effect_type") or effect.get("kind") or "unknown")] += 1
            if len(examples) < args.examples:
                examples.append({
                    "card": card_name(card),
                    "effect": effect_name(effect),
                    "kind": effect.get("kind"),
                    "effect_type": rt.get("effect_type"),
                    "runtime_support": rt.get("runtime_support"),
                    "patches": rt.get("patches"),
                    "playability": rt.get("playability"),
                    "usage_limit": rt.get("usage_limit"),
                    "ops": ops(effect),
                    "text": effect_text(effect)[:220],
                })

    print(json.dumps({
        "compiled": args.compiled,
        "cards": len(cards),
        "effects_with_turn1_runtime": sum(patch_counts.values() > 0 for _ in [0]) if False else sum(1 for card in cards for e in card.get("compiled_effects", []) or [] if isinstance(e, dict) and isinstance(e.get("turn1_runtime"), dict)),
        "patch_counts": dict(patch_counts),
        "runtime_support_counts": dict(support_counts),
        "effect_type_counts": dict(by_effect_type),
        "examples": examples,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
