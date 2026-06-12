from __future__ import annotations

"""
Fast batch Turn 1 target-finding report for every unique card in a deck.

Version: v0.5

This script reuses the same simulated legal opening hands, prizes, and draw-for-turn
for every target card, then runs the turn-1 action policy from
scripts/run_turn1_target_finder.py for each target.

Why this exists
---------------
Running run_turn1_target_finder.py once per card would reshuffle 5,000 hands for
every card and make card-to-card comparisons noisy. This script instead:
  1. resolves the deck once,
  2. generates the same N legal starts once,
  3. evaluates every unique card against those exact same starts,
  4. combines exact probability.py baselines with simulated action increments.

Output
------
JSON: data/reports/simulator_readiness/turn1_all_cards.json
CSV:  data/reports/simulator_readiness/turn1_all_cards_summary.csv
Lines CSV: data/reports/simulator_readiness/turn1_all_cards_lines.csv

Requirement
-----------
This expects scripts/run_turn1_target_finder.py v0.8+ to be present, because it
reuses the same deck parser, exact probability integration, and turn-1 policy.

Version v0.5 delegates ability scoring/execution to run_turn1_target_finder.py v0.13+ so the all-card batch report also uses search-for-ability-requirement chains, such as Ultra Ball -> Solrock -> Lunar Cycle.
"""

import argparse
import copy
import csv
import importlib.util
import json
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TARGET_FINDER_PATH = os.path.join(ROOT, "scripts", "run_turn1_target_finder.py")


def load_target_finder_module():
    if not os.path.exists(TARGET_FINDER_PATH):
        raise RuntimeError(f"Missing dependency: {TARGET_FINDER_PATH}. Install run_turn1_target_finder.py first.")
    spec = importlib.util.spec_from_file_location("turn1_target_finder_module", TARGET_FINDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TARGET_FINDER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def pct(x: float) -> float:
    return round(100.0 * x, 4)


def ci95(successes: int, n: int) -> Dict[str, float]:
    if n <= 0:
        return {"low": 0.0, "high": 0.0}
    import math
    p = successes / n
    se = math.sqrt(max(p * (1.0 - p), 0.0) / n)
    return {"low": pct(max(0.0, p - 1.96 * se)), "high": pct(min(1.0, p + 1.96 * se))}


@dataclass
class BaseTrial:
    opening: List[Dict[str, Any]]
    prizes: List[Dict[str, Any]]
    library: List[Dict[str, Any]]
    mulligans: int


@dataclass
class TargetGoingAgg:
    trials: int = 0
    successes: int = 0
    opening: int = 0
    draw_for_turn: int = 0
    actions: int = 0
    all_target_copies_prized: int = 0
    mulligans: Counter = field(default_factory=Counter)
    stages: Counter = field(default_factory=Counter)
    lines: Counter = field(default_factory=Counter)

    def add(self, result: Dict[str, Any]) -> None:
        self.trials += 1
        found = bool(result.get("found"))
        stage = str(result.get("found_stage") or "not_found")
        self.stages[stage] += 1
        self.mulligans[int(result.get("mulligans") or 0)] += 1
        if result.get("all_target_copies_prized"):
            self.all_target_copies_prized += 1
        if found:
            self.successes += 1
            if stage == "opening_hand":
                self.opening += 1
            elif stage == "draw_for_turn":
                self.draw_for_turn += 1
            else:
                self.actions += 1
                line = str(result.get("line") or "none")
                if line != "none":
                    self.lines[line] += 1

    def summary(self) -> Dict[str, Any]:
        n = self.trials
        return {
            "trials": n,
            "successes": self.successes,
            "probability": round(self.successes / n, 8) if n else 0.0,
            "percent": pct(self.successes / n) if n else 0.0,
            "ci95_percent": ci95(self.successes, n),
            "found_in_opening_hand": {"successes": self.opening, "percent": pct(self.opening / n) if n else 0.0},
            "found_on_draw_for_turn": {"successes": self.draw_for_turn, "percent": pct(self.draw_for_turn / n) if n else 0.0},
            "found_after_actions": {"successes": self.actions, "percent": pct(self.actions / n) if n else 0.0},
            "all_target_copies_prized": {"trials": self.all_target_copies_prized, "percent": pct(self.all_target_copies_prized / n) if n else 0.0},
            "mulligan_distribution": [
                {"mulligans": k, "trials": v, "percent": pct(v / n) if n else 0.0}
                for k, v in sorted(self.mulligans.items())
            ],
            "found_stage_distribution": [
                {"stage": k, "trials": v, "percent": pct(v / n) if n else 0.0}
                for k, v in self.stages.most_common()
            ],
            "top_success_lines": [
                {"line": k, "count": v, "percent_of_trials": pct(v / n) if n else 0.0}
                for k, v in self.lines.most_common(20)
            ],
        }


def generate_base_trials(tf, deck: List[Dict[str, Any]], trials: int, seed: int, hand_size: int, prize_count: int, use_mulligans: bool) -> List[BaseTrial]:
    rng = random.Random(seed)
    bases: List[BaseTrial] = []
    for _ in range(trials):
        mulligans = 0
        while True:
            shuffled = list(deck)
            rng.shuffle(shuffled)
            opening = shuffled[:hand_size]
            rest = shuffled[hand_size:]
            if not use_mulligans or any(tf.is_basic_pokemon(c) for c in opening):
                break
            mulligans += 1
            if mulligans > 100:
                raise RuntimeError("Exceeded 100 mulligans in one trial; check Basic Pokémon count")
        prizes = rest[:prize_count]
        library = rest[prize_count:]
        bases.append(BaseTrial(opening=list(opening), prizes=list(prizes), library=list(library), mulligans=mulligans))
    return bases



# -----------------------------
# Generic ability layer for all-card Turn 1 evaluation
# -----------------------------


def ability_effects(tf, card: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return compiled effects that look like Pokémon Abilities.

    The compiler uses several effect_kind strings across eras, so this checks
    both effect_kind/kind and the effect_id. This intentionally excludes attacks.
    """
    out: List[Dict[str, Any]] = []
    if tf.card_supertype(card) != "Pokémon":
        return out
    for eff in tf.iter_effects(card):
        kind = str(eff.get("effect_kind") or eff.get("kind") or "").lower()
        eid = str(eff.get("effect_id") or eff.get("id") or "").lower()
        if "ability" in kind or "::ability" in eid:
            if not tf.effect_is_trivial_rule(eff):
                out.append(eff)
    return out


def ability_name_from_effect(effect: Dict[str, Any]) -> str:
    for key in ("name", "ability_name", "label", "title"):
        if effect.get(key):
            return str(effect.get(key))
    eid = str(effect.get("effect_id") or effect.get("id") or "")
    if "::" in eid:
        parts = [p for p in eid.split("::") if p]
        if parts:
            return parts[-1].replace("_", " ").title()
    text = str(effect.get("text") or effect.get("source_text") or "")
    if text:
        return text[:40]
    return "Ability"


def card_has_usable_ability(tf, card: Dict[str, Any]) -> bool:
    if tf.is_mega_kangaskhan_ex(card) or tf.is_meowth_ex(card):
        return True
    if is_teal_mask_ogerpon_ex(tf, card):
        return True
    for eff in ability_effects(tf, card):
        for step in tf.flatten_steps(eff):
            if step.get("op") in {
                "draw_cards", "draw_cards_per_coin_heads", "draw_until_hand_size",
                "search_deck", "look_at_top_cards", "look_at_cards", "reorder_cards",
                "choose_cards", "put_card_into_hand", "move_card", "move_cards",
            }:
                return True
    return False


def is_teal_mask_ogerpon_ex(tf, card: Dict[str, Any]) -> bool:
    return tf.norm(tf.card_name(card)) == tf.norm("Teal Mask Ogerpon ex")


def is_basic_grass_energy(tf, card: Dict[str, Any]) -> bool:
    if tf.card_supertype(card) != "Energy":
        return False
    name_n = tf.norm(tf.card_name(card))
    types = {tf.norm(x) for x in tf.card_types(card)}
    return "grass" in name_n or "grass" in types


def bench_capacity(st: Any) -> int:
    # We are not modeling Area Zero's expanded Bench yet. Keep the normal cap.
    return 5


def can_bench_basic_for_ability(tf, st: Any, card: Dict[str, Any], target_norm: str) -> bool:
    if not tf.is_basic_pokemon(card):
        return False
    if len(st.bench) >= bench_capacity(st):
        return False
    if card not in st.hand:
        return False
    # If this card is the target, the trial would already have succeeded from hand.
    # Do not bench/discard the target as a helper.
    if tf.target_matches(card, target_norm):
        return False
    return card_has_usable_ability(tf, card)


def ability_draw_power(tf, effect: Dict[str, Any]) -> int:
    total = 0
    for step in tf.flatten_steps(effect):
        op = step.get("op")
        if op in {"draw_cards", "draw_cards_per_coin_heads"}:
            total += tf.draw_amount_from_step(step, coin_heads=1)
        elif op == "draw_until_hand_size":
            total += tf.amount_value(step.get("target_hand_size"), default=0)
    return total


def ability_directly_searches_target(tf, effect: Dict[str, Any], target_norm: str, deck: Sequence[Dict[str, Any]]) -> bool:
    target_cards = [c for c in deck if tf.target_matches(c, target_norm)]
    if not target_cards:
        return False
    for step in tf.flatten_steps(effect):
        if step.get("op") == "search_deck":
            filt = tf.extract_filter(step)
            if any(tf.filter_allows_card(filt, tc) for tc in target_cards):
                return True
    return False


def score_generic_ability(tf, st: Any, source: Dict[str, Any], effect: Dict[str, Any], target_norm: str) -> float:
    """Heuristic score for a compiled Pokémon Ability.

    This is target-finding only. It prefers abilities that directly search the
    target, then draw effects, then look/reorder effects that currently see the
    target near the top of deck.
    """
    if ability_directly_searches_target(tf, effect, target_norm, st.deck):
        return 8800.0
    target_remaining = sum(1 for c in st.deck if tf.target_matches(c, target_norm))
    if target_remaining <= 0:
        return -1.0
    deck_size = max(1, len(st.deck))
    draw_power = ability_draw_power(tf, effect)
    draw_score = 1000.0 * (1.0 - tf.hypergeom_zero(deck_size, target_remaining, min(draw_power, deck_size))) if draw_power else 0.0
    look_score = 0.0
    for step in tf.flatten_steps(effect):
        if step.get("op") in {"look_at_top_cards", "look_at_cards"}:
            n = tf.amount_value(step.get("amount") or step.get("count") or step.get("number"), default=1)
            if any(tf.target_matches(c, target_norm) for c in st.deck[:max(0, n)]):
                look_score = max(look_score, 100.0)
    return max(draw_score, look_score)


def get_abilities_used(st: Any) -> set:
    used = getattr(st, "abilities_used", None)
    if used is None:
        used = set()
        setattr(st, "abilities_used", used)
    return used


def in_play_pokemon(st: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if st.active is not None:
        out.append(st.active)
    out.extend(list(st.bench))
    return out


def can_use_teal_dance(tf, st: Any) -> bool:
    used = int(st.counts.get("teal_dance_used", 0) or 0)
    available_sources = sum(1 for c in in_play_pokemon(st) if is_teal_mask_ogerpon_ex(tf, c))
    if used >= available_sources:
        return False
    return any(is_basic_grass_energy(tf, c) for c in st.hand)


def teal_dance_score(tf, st: Any, target_norm: str) -> float:
    if not can_use_teal_dance(tf, st):
        return -1.0
    target_remaining = sum(1 for c in st.deck if tf.target_matches(c, target_norm))
    if target_remaining <= 0:
        return -1.0
    return 1000.0 * (1.0 - tf.hypergeom_zero(max(1, len(st.deck)), target_remaining, 1))


def use_teal_dance(tf, st: Any, target_norm: str, stage: str) -> None:
    if not can_use_teal_dance(tf, st):
        return
    energy = None
    for c in list(st.hand):
        if is_basic_grass_energy(tf, c):
            energy = c
            break
    if energy is None:
        return
    st.hand.remove(energy)
    # We do not need a full attachment model for target-finding; the important
    # state change is that Grass leaves hand and Teal Dance draws 1.
    st.counts["teal_dance_used"] = int(st.counts.get("teal_dance_used", 0) or 0) + 1
    st.actions_used += 1
    st.line.append("Teal Dance")
    st.log.append({"event": "use_ability", "ability": "Teal Dance", "stage": stage, "source": "Teal Mask Ogerpon ex", "attached": tf.card_name(energy)})
    tf.draw_cards(st, 1, stage)
    if st.has_target_in_hand(target_norm):
        st.found = True
        st.found_stage = stage


def generic_ability_candidates(tf, st: Any, target_norm: str) -> List[Tuple[float, Dict[str, Any], Dict[str, Any]]]:
    used = get_abilities_used(st)
    out: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for source in in_play_pokemon(st):
        for idx, eff in enumerate(ability_effects(tf, source)):
            key = (id(source), str(eff.get("effect_id") or eff.get("id") or idx))
            if key in used:
                continue
            # Run Errand and Teal Dance have explicit models; skip any duplicate generic form.
            name_n = tf.norm(ability_name_from_effect(eff) + " " + str(eff.get("text") or ""))
            if "run errand" in name_n or "teal dance" in name_n or "last ditch" in name_n:
                continue
            score = score_generic_ability(tf, st, source, eff, target_norm)
            if score > 0:
                out.append((score, source, eff))
    return out


def use_generic_ability(tf, st: Any, source: Dict[str, Any], effect: Dict[str, Any], rng: random.Random, target_norm: str, going: str, enable_chain_search: bool) -> None:
    used = get_abilities_used(st)
    key = (id(source), str(effect.get("effect_id") or effect.get("id") or ability_name_from_effect(effect)))
    if key in used:
        return
    used.add(key)
    st.actions_used += 1
    ability_name = ability_name_from_effect(effect)
    st.line.append(ability_name)
    stage = f"after_use_{ability_name}"
    st.log.append({"event": "use_ability", "ability": ability_name, "stage": stage, "source": tf.card_name(source)})
    tf.execute_steps(st, tf.iter_steps(effect), rng, target_norm, going, stage, enable_chain_search)


def bench_basic_for_ability(tf, st: Any, card: Dict[str, Any], rng: random.Random, target_norm: str, going: str, enable_chain_search: bool) -> None:
    if not can_bench_basic_for_ability(tf, st, card, target_norm):
        return
    # Preserve special Meowth handling from the target-finder policy.
    if tf.is_meowth_ex(card):
        tf.play_card(st, card, rng, target_norm, going, enable_chain_search)
        return
    st.hand.remove(card)
    st.bench.append(card)
    st.actions_used += 1
    st.line.append(tf.card_name(card))
    stage = f"after_bench_{tf.card_name(card)}"
    st.log.append({"event": "play_basic_to_bench", "card": tf.card_name(card), "stage": stage})
    if is_teal_mask_ogerpon_ex(tf, card) and can_use_teal_dance(tf, st):
        use_teal_dance(tf, st, target_norm, f"{stage}_then_Teal_Dance")
        return
    # If it has a useful compiled ability, use the highest scoring one immediately.
    cands = generic_ability_candidates(tf, st, target_norm)
    cands = [x for x in cands if x[1] is card]
    if cands:
        cands.sort(key=lambda x: x[0], reverse=True)
        _, source, eff = cands[0]
        use_generic_ability(tf, st, source, eff, rng, target_norm, going, enable_chain_search)


def bench_basic_ability_score(tf, st: Any, card: Dict[str, Any], target_norm: str, going: str) -> float:
    if not can_bench_basic_for_ability(tf, st, card, target_norm):
        return -1.0
    if tf.is_meowth_ex(card):
        # Delegate to target-finder scoring for Meowth because it depends on Supporter availability.
        return tf.score_playable_card(card, st, target_norm, going, True)
    if is_teal_mask_ogerpon_ex(tf, card):
        # Score based on Teal Dance draw 1 if Grass is in hand.
        # Temporarily estimate as if benched.
        if any(is_basic_grass_energy(tf, c) for c in st.hand if c is not card):
            target_remaining = sum(1 for c in st.deck if tf.target_matches(c, target_norm))
            if target_remaining > 0:
                return 1000.0 * (1.0 - tf.hypergeom_zero(max(1, len(st.deck)), target_remaining, 1))
        return -1.0
    best = -1.0
    # Estimate compiled ability value after benching.
    for eff in ability_effects(tf, card):
        best = max(best, score_generic_ability(tf, st, card, eff, target_norm))
    return best


def simulate_target_from_base(
    tf,
    base: BaseTrial,
    target_norm: str,
    going: str,
    draw_for_turn: bool,
    max_actions: int,
    enable_chain_search: bool,
    rng: random.Random,
) -> Dict[str, Any]:
    opening = list(base.opening)
    prizes = list(base.prizes)
    library = list(base.library)

    target_copies_total = sum(1 for c in (opening + prizes + library) if tf.target_matches(c, target_norm))
    target_copies_prized = sum(1 for c in prizes if tf.target_matches(c, target_norm))
    target_copies_in_opening = sum(1 for c in opening if tf.target_matches(c, target_norm))

    active = tf.choose_optimal_active(opening, target_norm)
    hand_after_setup = list(opening)
    if active is not None:
        try:
            hand_after_setup.remove(active)
        except ValueError:
            pass

    st = tf.SimState(deck=library, hand=hand_after_setup, prizes=prizes, active=active)
    setattr(st, "abilities_used", set())

    if target_copies_in_opening > 0:
        st.found = True
        st.found_stage = "opening_hand"
    elif draw_for_turn:
        tf.draw_cards(st, 1, "draw_for_turn")
        if st.has_target_in_hand(target_norm):
            st.found = True
            st.found_stage = "draw_for_turn"

    while not st.found and st.actions_used < max_actions:
        scored: List[Tuple[float, Any]] = []

        # Normal Trainer / modeled-from-hand actions from target finder.
        playable = [c for c in list(st.hand) if tf.card_can_be_played_from_hand(c, going, st.supporter_used)]
        for c in playable:
            score = tf.score_playable_card(c, st, target_norm, going, enable_chain_search)
            if score > 0:
                scored.append((score, c))

        # Basic Pokémon can be benched if their abilities help find the target.
        for c in list(st.hand):
            score = tf.bench_basic_ability_score(st, c, target_norm, going)
            if score > 0:
                scored.append((score, {"_virtual_action": "BenchAbility", "card": c}))

        # Explicit active/bench abilities.
        ability_score = tf.run_errand_score(st, target_norm)
        if ability_score > 0:
            scored.append((ability_score, {"_virtual_action": "Run Errand"}))

        td_score = tf.teal_dance_score(st, target_norm)
        if td_score > 0:
            scored.append((td_score, {"_virtual_action": "Teal Dance"}))

        for score, source, eff in tf.generic_ability_candidates(st, target_norm):
            scored.append((score, {"_virtual_action": "GenericAbility", "source": source, "effect": eff}))

        if hasattr(tf, "ability_requirement_search_candidates"):
            for score, action in tf.ability_requirement_search_candidates(st, target_norm, going, enable_chain_search):
                scored.append((score, action))

        if not scored:
            break
        scored.sort(key=lambda x: (x[0], str(x[1])), reverse=True)
        _, chosen = scored[0]

        if isinstance(chosen, dict) and chosen.get("_virtual_action") == "Run Errand":
            tf.use_run_errand(st, target_norm, "after_use_Run_Errand")
        elif isinstance(chosen, dict) and chosen.get("_virtual_action") == "Teal Dance":
            tf.use_teal_dance(st, target_norm, "after_use_Teal_Dance")
        elif isinstance(chosen, dict) and chosen.get("_virtual_action") == "BenchAbility":
            tf.bench_basic_for_ability(st, chosen["card"], rng, target_norm, going, enable_chain_search)
        elif isinstance(chosen, dict) and chosen.get("_virtual_action") == "GenericAbility":
            tf.use_generic_ability(st, chosen["source"], chosen["effect"], rng, target_norm, going, enable_chain_search)
        elif isinstance(chosen, dict) and chosen.get("_virtual_action") == "AbilityRequirementSearch":
            tf.use_ability_requirement_search_chain(st, chosen, rng, target_norm, going, enable_chain_search)
        else:
            tf.play_card(st, chosen, rng, target_norm, going, enable_chain_search)

    return {
        "found": st.found,
        "found_stage": st.found_stage or "not_found",
        "line": " -> ".join(st.line) if st.line else "none",
        "mulligans": base.mulligans,
        "target_copies_total": target_copies_total,
        "target_copies_prized": target_copies_prized,
        "target_copies_in_opening": target_copies_in_opening,
        "all_target_copies_prized": target_copies_total > 0 and target_copies_prized == target_copies_total,
        "actions_used": st.actions_used,
    }

def unique_targets_from_deck(tf, deck: List[Dict[str, Any]], include_energy: bool, include_trainers: bool, include_pokemon: bool) -> List[Dict[str, Any]]:
    counts = Counter(tf.card_id(c) for c in deck)
    first_by_id: Dict[str, Dict[str, Any]] = {}
    for c in deck:
        first_by_id.setdefault(tf.card_id(c), c)

    targets = []
    for cid, c in first_by_id.items():
        supertype = tf.card_supertype(c)
        if supertype == "Energy" and not include_energy:
            continue
        if supertype == "Trainer" and not include_trainers:
            continue
        if supertype == "Pokémon" and not include_pokemon:
            continue
        targets.append({
            "target_card_id": cid,
            "target_name": tf.card_name(c),
            "target_norm": tf.norm(cid),
            "copies": counts[cid],
            "supertype": supertype,
            "subtypes": tf.card_subtypes(c),
        })
    targets.sort(key=lambda r: (r["supertype"], r["target_name"], r["target_card_id"]))
    return targets


def exact_plus_sim(tf, deck: List[Dict[str, Any]], target_norm: str, summary: Dict[str, Any], hand_size: int, prizes: int, draw_for_turn: bool) -> Dict[str, Any]:
    exact = tf.build_exact_probability_baselines(
        deck=deck,
        target_norm=target_norm,
        hand_size=hand_size,
        prize_count=prizes,
        draw_for_turn=draw_for_turn,
        max_mulligans=6,
    )
    if not exact.get("available"):
        return {"available": False, "exact": exact}

    p_seen_by_draw = float(exact.get("natural_draw_by_turn_1_has_target_percent", 0.0)) / 100.0
    n = int(summary.get("trials") or 0)
    opening = int(summary.get("found_in_opening_hand", {}).get("successes") or 0)
    draw = int(summary.get("found_on_draw_for_turn", {}).get("successes") or 0)
    actions = int(summary.get("found_after_actions", {}).get("successes") or 0)
    not_seen_after_draw = max(0, n - opening - draw)
    cond_action = (actions / not_seen_after_draw) if not_seen_after_draw else 0.0
    action_increment = (1.0 - p_seen_by_draw) * cond_action

    weighted_lines = []
    for row in summary.get("top_success_lines", []):
        line_count = int(row.get("count") or 0)
        cond_line = (line_count / not_seen_after_draw) if not_seen_after_draw else 0.0
        weighted_lines.append({
            "line": row.get("line"),
            "count": line_count,
            "percent_of_trials_raw_sim": row.get("percent_of_trials"),
            "conditional_on_not_seen_after_draw_percent": pct(cond_line),
            "exact_weighted_percent_of_trials": pct((1.0 - p_seen_by_draw) * cond_line),
        })

    return {
        "available": True,
        "exact_opening_hand_percent": exact.get("opening_hand_has_target_percent"),
        "exact_seen_by_draw_for_turn_percent": exact.get("natural_draw_by_turn_1_has_target_percent"),
        "exact_draw_for_turn_increment_percent": exact.get("draw_for_turn_increment_percent"),
        "simulated_action_success_given_not_seen_after_draw_percent": pct(cond_action),
        "exact_weighted_action_increment_percent": pct(action_increment),
        "final_exact_plus_sim_percent": pct(p_seen_by_draw + action_increment),
        "exact_prize_at_least_one_percent": exact.get("at_least_one_target_prized_after_legal_hand_percent"),
        "exact_prize_all_copies_percent": exact.get("all_target_copies_prized_after_legal_hand_percent"),
        "line_contributions": weighted_lines,
    }


def write_csvs(summary_path: str, lines_path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_fields = [
        "target_name", "target_card_id", "copies", "supertype", "subtypes", "going", "trials",
        "raw_sim_percent", "raw_ci95_low", "raw_ci95_high",
        "exact_opening_hand_percent", "exact_seen_by_draw_for_turn_percent",
        "simulated_action_success_given_not_seen_after_draw_percent", "exact_weighted_action_increment_percent",
        "final_exact_plus_sim_percent", "found_after_actions_raw_percent",
        "exact_prize_at_least_one_percent", "exact_prize_all_copies_percent",
        "top_line_1", "top_line_1_exact_weighted_percent", "top_line_2", "top_line_2_exact_weighted_percent",
    ]
    line_fields = [
        "target_name", "target_card_id", "copies", "supertype", "going", "line", "count",
        "percent_of_trials_raw_sim", "conditional_on_not_seen_after_draw_percent", "exact_weighted_percent_of_trials",
    ]

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for r in rows:
            for s in r["scenarios"]:
                ep = s.get("exact_plus_simulation", {})
                lines = ep.get("line_contributions", []) if ep.get("available") else []
                sm = s["summary"]
                w.writerow({
                    "target_name": r["target_name"],
                    "target_card_id": r["target_card_id"],
                    "copies": r["copies"],
                    "supertype": r["supertype"],
                    "subtypes": ";".join(r.get("subtypes") or []),
                    "going": s["going"],
                    "trials": sm["trials"],
                    "raw_sim_percent": sm["percent"],
                    "raw_ci95_low": sm["ci95_percent"]["low"],
                    "raw_ci95_high": sm["ci95_percent"]["high"],
                    "exact_opening_hand_percent": ep.get("exact_opening_hand_percent"),
                    "exact_seen_by_draw_for_turn_percent": ep.get("exact_seen_by_draw_for_turn_percent"),
                    "simulated_action_success_given_not_seen_after_draw_percent": ep.get("simulated_action_success_given_not_seen_after_draw_percent"),
                    "exact_weighted_action_increment_percent": ep.get("exact_weighted_action_increment_percent"),
                    "final_exact_plus_sim_percent": ep.get("final_exact_plus_sim_percent"),
                    "found_after_actions_raw_percent": sm.get("found_after_actions", {}).get("percent"),
                    "exact_prize_at_least_one_percent": ep.get("exact_prize_at_least_one_percent"),
                    "exact_prize_all_copies_percent": ep.get("exact_prize_all_copies_percent"),
                    "top_line_1": lines[0]["line"] if len(lines) > 0 else "",
                    "top_line_1_exact_weighted_percent": lines[0]["exact_weighted_percent_of_trials"] if len(lines) > 0 else "",
                    "top_line_2": lines[1]["line"] if len(lines) > 1 else "",
                    "top_line_2_exact_weighted_percent": lines[1]["exact_weighted_percent_of_trials"] if len(lines) > 1 else "",
                })

    with open(lines_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=line_fields)
        w.writeheader()
        for r in rows:
            for s in r["scenarios"]:
                ep = s.get("exact_plus_simulation", {})
                for line in ep.get("line_contributions", []) if ep.get("available") else []:
                    w.writerow({
                        "target_name": r["target_name"],
                        "target_card_id": r["target_card_id"],
                        "copies": r["copies"],
                        "supertype": r["supertype"],
                        "going": s["going"],
                        "line": line.get("line"),
                        "count": line.get("count"),
                        "percent_of_trials_raw_sim": line.get("percent_of_trials_raw_sim"),
                        "conditional_on_not_seen_after_draw_percent": line.get("conditional_on_not_seen_after_draw_percent"),
                        "exact_weighted_percent_of_trials": line.get("exact_weighted_percent_of_trials"),
                    })


def main() -> None:
    ap = argparse.ArgumentParser(description="Fast Turn 1 target-finding report for every unique card in a deck using the same simulated starts.")
    ap.add_argument("--compiled", default="data/compiled_cards/auto/compiled_cards_all.json")
    ap.add_argument("--decklist", required=True)
    ap.add_argument("--trials", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--going", choices=["first", "second", "both"], default="both")
    ap.add_argument("--hand-size", type=int, default=7)
    ap.add_argument("--prizes", type=int, default=6)
    ap.add_argument("--max-actions", type=int, default=20)
    ap.add_argument("--no-mulligans", action="store_true")
    ap.add_argument("--no-draw-for-turn", action="store_true")
    ap.add_argument("--complete-only", action="store_true")
    ap.add_argument("--chain-search", action="store_true")
    ap.add_argument("--include-energy", action="store_true", default=True)
    ap.add_argument("--exclude-energy", action="store_true")
    ap.add_argument("--include-trainers", action="store_true", default=True)
    ap.add_argument("--exclude-trainers", action="store_true")
    ap.add_argument("--include-pokemon", action="store_true", default=True)
    ap.add_argument("--exclude-pokemon", action="store_true")
    ap.add_argument("--limit-targets", type=int, default=0, help="Debug option: only evaluate the first N unique targets.")
    ap.add_argument("--out", default="data/reports/simulator_readiness/turn1_all_cards.json")
    ap.add_argument("--csv-out", default="data/reports/simulator_readiness/turn1_all_cards_summary.csv")
    ap.add_argument("--lines-csv-out", default="data/reports/simulator_readiness/turn1_all_cards_lines.csv")
    args = ap.parse_args()

    tf = load_target_finder_module()
    if tf.load_compiled_cards is None:
        raise RuntimeError("Could not import tcgsim through run_turn1_target_finder.py")

    cards = tf.load_compiled_cards(args.compiled)
    if args.complete_only:
        cards = tf.filter_complete_cards(cards)

    raw_decklist = tf.parse_decklist(args.decklist)
    deck, unresolved = tf.resolve_decklist(raw_decklist, cards)
    if unresolved:
        result = {
            "passed": False,
            "error": "Some decklist entries could not be resolved against the compiled card file.",
            "unresolved": unresolved,
            "resolved_deck_size": len(deck),
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        raise SystemExit(2)

    include_energy = args.include_energy and not args.exclude_energy
    include_trainers = args.include_trainers and not args.exclude_trainers
    include_pokemon = args.include_pokemon and not args.exclude_pokemon
    targets = unique_targets_from_deck(tf, deck, include_energy=include_energy, include_trainers=include_trainers, include_pokemon=include_pokemon)
    if args.limit_targets and args.limit_targets > 0:
        targets = targets[: args.limit_targets]

    goings = ["first", "second"] if args.going == "both" else [args.going]

    base_trials = generate_base_trials(
        tf=tf,
        deck=deck,
        trials=args.trials,
        seed=args.seed,
        hand_size=args.hand_size,
        prize_count=args.prizes,
        use_mulligans=not args.no_mulligans,
    )

    rows: List[Dict[str, Any]] = []
    for idx, target in enumerate(targets, start=1):
        target_norm = target["target_norm"]
        target_row = dict(target)
        target_row["scenarios"] = []
        for going in goings:
            rng = random.Random(args.seed + idx * 1_000_003 + (0 if going == "first" else 500_000_000))
            agg = TargetGoingAgg()
            for base in base_trials:
                result = simulate_target_from_base(
                    tf=tf,
                    base=base,
                    target_norm=target_norm,
                    going=going,
                    draw_for_turn=not args.no_draw_for_turn,
                    max_actions=args.max_actions,
                    enable_chain_search=args.chain_search,
                    rng=rng,
                )
                agg.add(result)
            summary = agg.summary()
            scenario = {
                "going": going,
                "summary": summary,
                "exact_plus_simulation": exact_plus_sim(
                    tf=tf,
                    deck=deck,
                    target_norm=target_norm,
                    summary=summary,
                    hand_size=args.hand_size,
                    prizes=args.prizes,
                    draw_for_turn=not args.no_draw_for_turn,
                ),
            }
            target_row["scenarios"].append(scenario)
        rows.append(target_row)

    result = {
        "passed": True,
        "compiled_source": args.compiled,
        "decklist_source": args.decklist,
        "trials": args.trials,
        "seed": args.seed,
        "deck_size": len(deck),
        "unique_targets_evaluated": len(rows),
        "same_starting_hands_used_for_all_targets": True,
        "same_prizes_and_draw_for_turn_used_for_all_targets": True,
        "assumptions": {
            "action_policy_source": "scripts/run_turn1_target_finder.py",
            "exact_probability_source": "src/probability.py through run_turn1_target_finder.py",
            "note": "Exact opening/draw baselines are analytic. Action increments are simulated conditional on not already seeing the target after draw-for-turn.",
            "ability_layer": "Enabled. Basic Pokémon with useful abilities can be benched, in-play Pokémon abilities can be activated, and compiled ability draw/search/look/reorder steps are evaluated. Explicit models include Teal Dance, Run Errand, and Last-Ditch Catch.",
        },
        "deck_summary": {
            "basic_pokemon": sum(1 for c in deck if tf.is_basic_pokemon(c)),
            "pokemon": sum(1 for c in deck if tf.card_supertype(c) == "Pokémon"),
            "trainer": sum(1 for c in deck if tf.card_supertype(c) == "Trainer"),
            "energy": sum(1 for c in deck if tf.card_supertype(c) == "Energy"),
            "top_cards": [{"name": k, "count": v} for k, v in Counter(tf.card_name(c) for c in deck).most_common(30)],
        },
        "targets": rows,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_csvs(args.csv_out, args.lines_csv_out, rows)

    # Compact console summary: top targets by exact+sim percent for going first/second.
    print(json.dumps({
        "passed": True,
        "trials": args.trials,
        "deck_size": len(deck),
        "unique_targets_evaluated": len(rows),
        "outputs": {
            "json": args.out,
            "summary_csv": args.csv_out,
            "lines_csv": args.lines_csv_out,
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
