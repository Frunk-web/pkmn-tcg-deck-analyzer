from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tcgsim import RuntimeEngine, build_two_player_seed_state, filter_complete_cards, load_compiled_cards


def iter_effects(cards):
    for card in cards:
        for effect in card.get("compiled_effects", []) or []:
            yield card, effect


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled", default="data/reports/simulator_readiness/complete_cards_for_sim_seed.json")
    ap.add_argument("--max-effects", type=int, default=25)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--out", default="data/reports/simulator_readiness/sim_smoke_test_log.json")
    args = ap.parse_args()

    cards = load_compiled_cards(args.compiled)
    complete = filter_complete_cards(cards) or cards
    state = build_two_player_seed_state(complete, deck_size=20, seed=7)
    engine = RuntimeEngine(strict=args.strict)

    executed = 0
    op_counter = Counter()
    for card, effect in iter_effects(complete):
        if executed >= args.max_effects:
            break
        source = state.players[state.turn_player].active
        for step in effect.get("steps", []) or []:
            if step.get("op"):
                op_counter[step["op"]] += 1
        engine.execute_effect(state, effect, source_instance_id=source)
        executed += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result = {
        "compiled_source": args.compiled,
        "complete_cards_loaded": len(complete),
        "effects_executed": executed,
        "ops_seen_in_executed_effects": dict(op_counter.most_common()),
        "unsupported_ops_logged": [e for e in state.log if e.get("event") == "unsupported_op"][:50],
        "last_25_log_events": state.log[-25:],
        "state_summary": {
            "p1_hand": len(state.players["p1"].hand),
            "p1_deck": len(state.players["p1"].deck),
            "p1_active": state.players["p1"].active,
            "p2_active": state.players["p2"].active,
            "p2_active_damage_counters": state.cards[state.players["p2"].active].damage_counters if state.players["p2"].active else None,
            "p2_active_special_conditions": state.cards[state.players["p2"].active].special_conditions if state.players["p2"].active else [],
        },
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
