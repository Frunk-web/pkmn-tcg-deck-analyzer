from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tcgsim import RuntimeEngine, filter_complete_cards, load_compiled_cards
from tcgsim.loader import make_instance
from tcgsim.state import GameState, PlayerState, CardInstance


def identity(card: Dict[str, Any]) -> Dict[str, Any]:
    return card.get("identity", {}) or {}


def is_basic_pokemon(card: Dict[str, Any]) -> bool:
    ident = identity(card)
    return ident.get("supertype") == "Pokémon" and "Basic" in (ident.get("subtypes") or [])


def is_pokemon(card: Dict[str, Any]) -> bool:
    return identity(card).get("supertype") == "Pokémon"


def is_energy(card: Dict[str, Any]) -> bool:
    return identity(card).get("supertype") == "Energy"


def attack_damage_base(effect: Dict[str, Any]) -> int:
    for step in effect.get("steps", []) or []:
        if step.get("op") == "deal_attack_damage":
            amount = step.get("amount") or step.get("damage") or {}
            if isinstance(amount, dict):
                if isinstance(amount.get("base"), int):
                    return amount["base"]
                printed = str(amount.get("printed", ""))
                digits = "".join(ch for ch in printed if ch.isdigit())
                if digits:
                    return int(digits)
            if isinstance(amount, int):
                return amount
    return 0


def find_basic_attacker(cards: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    fallback: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
    for card in cards:
        if not is_basic_pokemon(card):
            continue
        for effect in card.get("compiled_effects", []) or []:
            if effect.get("kind") == "attack" and any(s.get("op") == "deal_attack_damage" for s in effect.get("steps", []) or []):
                dmg = attack_damage_base(effect)
                if dmg > 0:
                    return card, effect
                fallback = fallback or (card, effect)
    if fallback:
        return fallback
    for card in cards:
        if not is_pokemon(card):
            continue
        for effect in card.get("compiled_effects", []) or []:
            if effect.get("kind") == "attack":
                return card, effect
    raise ValueError("No Pokémon attack effect found in compiled seed")


def find_basic_pokemon(cards: List[Dict[str, Any]], exclude_id: Optional[str] = None) -> Dict[str, Any]:
    for card in cards:
        cid = card.get("representative_card_id") or card.get("card_id")
        if cid != exclude_id and is_basic_pokemon(card):
            return card
    for card in cards:
        cid = card.get("representative_card_id") or card.get("card_id")
        if cid != exclude_id and is_pokemon(card):
            return card
    raise ValueError("No Pokémon found in compiled seed")


def find_energy_or_synthetic(cards: List[Dict[str, Any]]) -> Dict[str, Any]:
    for card in cards:
        if is_energy(card):
            return card
    return {
        "schema_version": "synthetic/test-only",
        "representative_card_id": "synthetic-basic-energy",
        "card_id": "synthetic-basic-energy",
        "identity": {
            "name": "Synthetic Basic Energy",
            "canonical_name": "Synthetic Basic Energy",
            "supertype": "Energy",
            "subtypes": ["Basic"],
            "types": ["Colorless"],
            "tags": ["test_only"],
        },
        "compiled_effects": [],
        "parser": {"status": "complete"},
    }


def card_label(card: Dict[str, Any]) -> Dict[str, Any]:
    ident = identity(card)
    return {
        "card_id": card.get("representative_card_id") or card.get("card_id"),
        "name": ident.get("name"),
        "supertype": ident.get("supertype"),
        "subtypes": ident.get("subtypes", []),
    }


def add_instance(state: GameState, card_def: Dict[str, Any], iid: str, owner: str, zone: str) -> str:
    inst = make_instance(card_def, iid, owner, zone)
    state.cards[iid] = inst
    player = state.players[owner]
    if zone == "active":
        player.active = iid
    elif zone == "bench":
        player.bench.append(iid)
    elif zone == "hand":
        player.hand.append(iid)
    elif zone == "deck":
        player.deck.append(iid)
    elif zone == "prizes":
        player.prizes.append(iid)
    elif zone == "discard":
        player.discard.append(iid)
    else:
        inst.zone = zone
    return iid


def build_legalish_flow_state(cards: List[Dict[str, Any]], seed: int = 7) -> Tuple[GameState, Dict[str, Any]]:
    attacker, attack = find_basic_attacker(cards)
    attacker_id = attacker.get("representative_card_id") or attacker.get("card_id")
    defender = find_basic_pokemon(cards, exclude_id=attacker_id)
    bench = find_basic_pokemon(cards, exclude_id=defender.get("representative_card_id") or defender.get("card_id"))
    energy = find_energy_or_synthetic(cards)

    state = GameState(players={"p1": PlayerState("p1"), "p2": PlayerState("p2")}, cards={}, rng_seed=seed)

    # Active and bench Pokémon.
    add_instance(state, attacker, "p1-active-001", "p1", "active")
    add_instance(state, bench, "p1-bench-001", "p1", "bench")
    add_instance(state, defender, "p2-active-001", "p2", "active")
    add_instance(state, bench, "p2-bench-001", "p2", "bench")

    # Hands include Energy for attach tests.
    add_instance(state, energy, "p1-hand-energy-001", "p1", "hand")
    add_instance(state, energy, "p1-hand-energy-002", "p1", "hand")

    filler_defs = [c for c in cards if c is not attacker] or cards
    filler = cycle(filler_defs)
    for player in ("p1", "p2"):
        # Fill hand to 7.
        while len(state.players[player].hand) < 7:
            n = len(state.players[player].hand) + 1
            add_instance(state, next(filler), f"{player}-hand-{n:03d}", player, "hand")
        # Six prizes.
        for i in range(6):
            add_instance(state, next(filler), f"{player}-prize-{i+1:03d}", player, "prizes")
        # A legal-ish deck remainder.
        for i in range(46):
            add_instance(state, next(filler), f"{player}-deck-{i+1:03d}", player, "deck")

    selected = {
        "attacker": card_label(attacker),
        "defender": card_label(defender),
        "bench": card_label(bench),
        "energy": card_label(energy),
        "attack_effect_id": attack.get("effect_id"),
        "attack_damage_base": attack_damage_base(attack),
        "attack_text": attack.get("source", {}).get("text"),
    }
    return state, {"attack": attack, "selected": selected}


def passfail(name: str, passed: bool, **details: Any) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), **details}


def draw_for_turn(state: GameState, player: str) -> Optional[str]:
    if not state.players[player].deck:
        state.log_event("cannot_draw_for_turn_deck_empty", player=player)
        return None
    iid = state.players[player].deck.pop(0)
    state.players[player].hand.append(iid)
    state.cards[iid].zone = "hand"
    state.log_event("draw_for_turn", player=player, drawn=iid)
    return iid


def attach_energy_for_turn(state: GameState, player: str, target: str) -> Tuple[bool, Optional[str]]:
    flag = f"energy_attached_turn_{state.turn_number}_{player}"
    if state.players[player].flags.get(flag):
        state.log_event("attach_energy_denied_already_attached_this_turn", player=player, target=target)
        return False, None
    energy_id = None
    for iid in list(state.players[player].hand):
        if is_energy(state.cards[iid].definition):
            energy_id = iid
            break
    if energy_id is None:
        state.log_event("attach_energy_denied_no_energy_in_hand", player=player, target=target)
        return False, None
    state.players[player].hand.remove(energy_id)
    state.cards[target].attached_cards.append(energy_id)
    state.cards[energy_id].zone = "attached"
    state.cards[energy_id].controller = player
    state.players[player].flags[flag] = True
    state.log_event("attach_energy_for_turn", player=player, target=target, energy=energy_id)
    return True, energy_id


def end_turn(state: GameState) -> None:
    old = state.turn_player
    new = state.opponent_of(old)
    state.players[old].flags = {k: v for k, v in state.players[old].flags.items() if not k.startswith("energy_attached_turn_")}
    state.turn_player = new
    state.turn_number += 1
    state.phase = "draw_step"
    state.log_event("end_turn", old_turn_player=old, new_turn_player=new, turn_number=state.turn_number)


def summarize_state(state: GameState) -> Dict[str, Any]:
    p1 = state.players["p1"]
    p2 = state.players["p2"]
    return {
        "turn_player": state.turn_player,
        "turn_number": state.turn_number,
        "p1": {
            "deck": len(p1.deck),
            "hand": len(p1.hand),
            "prizes": len(p1.prizes),
            "active": p1.active,
            "bench": len(p1.bench),
            "active_attached": list(state.cards[p1.active].attached_cards) if p1.active else [],
        },
        "p2": {
            "deck": len(p2.deck),
            "hand": len(p2.hand),
            "prizes": len(p2.prizes),
            "active": p2.active,
            "bench": len(p2.bench),
            "active_damage_counters": state.cards[p2.active].damage_counters if p2.active else None,
            "active_zone": state.cards[p2.active].zone if p2.active else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled", default="data/reports/simulator_readiness/complete_cards_for_sim_seed.json")
    ap.add_argument("--out", default="data/reports/simulator_readiness/game_flow_test_report.json")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cards = filter_complete_cards(load_compiled_cards(args.compiled)) or load_compiled_cards(args.compiled)
    state, selected_ctx = build_legalish_flow_state(cards, seed=args.seed)
    engine = RuntimeEngine(strict=True)
    tests: List[Dict[str, Any]] = []

    # 1. Opening state invariants.
    tests.append(passfail(
        "setup_opening_state",
        state.players["p1"].active is not None
        and state.players["p2"].active is not None
        and len(state.players["p1"].hand) == 7
        and len(state.players["p1"].prizes) == 6
        and len(state.players["p1"].bench) >= 1,
        summary=summarize_state(state),
    ))

    # 2. Draw for turn.
    before_deck = len(state.players["p1"].deck)
    before_hand = len(state.players["p1"].hand)
    drawn = draw_for_turn(state, "p1")
    tests.append(passfail(
        "draw_for_turn_moves_top_deck_card_to_hand",
        drawn is not None and len(state.players["p1"].deck) == before_deck - 1 and len(state.players["p1"].hand) == before_hand + 1 and state.cards[drawn].zone == "hand",
        drawn=drawn,
        before={"deck": before_deck, "hand": before_hand},
        after={"deck": len(state.players["p1"].deck), "hand": len(state.players["p1"].hand)},
    ))

    # 3. Attach one Energy, then verify second attachment is blocked in same turn.
    active = state.players["p1"].active
    attached_before = len(state.cards[active].attached_cards)
    ok1, energy1 = attach_energy_for_turn(state, "p1", active)
    attached_mid = len(state.cards[active].attached_cards)
    ok2, energy2 = attach_energy_for_turn(state, "p1", active)
    attached_after = len(state.cards[active].attached_cards)
    tests.append(passfail(
        "attach_energy_once_per_turn",
        ok1 and energy1 is not None and attached_mid == attached_before + 1 and not ok2 and energy2 is None and attached_after == attached_mid,
        first_attach=energy1,
        second_attach_allowed=ok2,
        attached_cards=list(state.cards[active].attached_cards),
    ))

    # 4. Execute a real compiled attack effect and require damage to change.
    defender = state.players["p2"].active
    damage_before = state.cards[defender].damage_counters
    engine.execute_effect(state, selected_ctx["attack"], source_instance_id=active)
    damage_after = state.cards[defender].damage_counters
    tests.append(passfail(
        "compiled_attack_places_damage_on_opponent_active",
        damage_after > damage_before,
        attack_effect_id=selected_ctx["attack"].get("effect_id"),
        damage_before=damage_before,
        damage_after=damage_after,
        expected_base_damage=selected_ctx["selected"].get("attack_damage_base"),
    ))

    # 5. Switch active with bench.
    old_active = state.players["p1"].active
    old_bench = list(state.players["p1"].bench)
    context = {"self": "p1", "opponent": "p2", "source_instance_id": old_active, "choices": {}, "effect": {}}
    engine.execute_step(state, {"op": "switch_active", "player": "self"}, context)
    tests.append(passfail(
        "switch_active_promotes_bench_pokemon",
        state.players["p1"].active != old_active and old_active in state.players["p1"].bench,
        old_active=old_active,
        old_bench=old_bench,
        new_active=state.players["p1"].active,
        new_bench=list(state.players["p1"].bench),
    ))

    # 6. End turn resets once-per-turn attachment flag and passes turn.
    end_turn(state)
    tests.append(passfail(
        "end_turn_passes_turn_and_resets_energy_attach_flag",
        state.turn_player == "p2" and state.turn_number == 2 and not any(k.startswith("energy_attached_turn_") for k in state.players["p1"].flags),
        state={"turn_player": state.turn_player, "turn_number": state.turn_number, "p1_flags": state.players["p1"].flags},
    ))

    # 7. Knock Out cleanup can move a Pokémon to discard.
    ko_target = state.players["p2"].active
    engine.execute_step(state, {"op": "knock_out_pokemon", "target": "opponent.active"}, {"self": "p1", "opponent": "p2", "source_instance_id": state.players["p1"].active, "choices": {}, "effect": {}})
    tests.append(passfail(
        "knock_out_moves_target_to_discard_or_clears_active",
        state.cards[ko_target].zone == "discard" and ko_target in state.players["p2"].discard,
        knocked_out=ko_target,
        p2_active=state.players["p2"].active,
        p2_discard=list(state.players["p2"].discard),
        target_zone=state.cards[ko_target].zone,
    ))

    unsupported = [e for e in state.log if e.get("event") == "unsupported_op"]
    tests.append(passfail("no_unsupported_ops_during_game_flow_tests", len(unsupported) == 0, unsupported=unsupported[:20]))

    result = {
        "compiled_source": args.compiled,
        "cards_loaded": len(cards),
        "selected_cards": selected_ctx["selected"],
        "passed": all(t["passed"] for t in tests),
        "tests": tests,
        "state_summary": summarize_state(state),
        "last_50_log_events": state.log[-50:],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
