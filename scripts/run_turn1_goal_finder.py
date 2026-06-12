from __future__ import annotations

"""
Turn 1 goal finder for Pokémon TCG deck consistency diagnostics.

Version: v0.2

This builds on scripts/run_turn1_target_finder.py. Instead of asking only
"can I find target card X?", it estimates whether a multi-card goal is met by
end of turn 1:

  - X and Y
  - X and Y and Z
  - at least one of A/B/C
  - package goals such as Riolu + Mega Lucario ex + Wally's Compassion

The policy is intentionally conservative and transparent. It reuses the
single-target solver's card/effect execution layer, but scores actions against
all currently missing goal pieces.

Version v0.2 adds exact natural baseline integration from src/probability.py.
The exact layer computes legal-opening-hand and draw-for-turn goal probabilities;
the simulator estimates only the action increment after the goal was not already
satisfied naturally.
"""

import argparse
import csv
import json
import copy
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import run_turn1_target_finder as tf
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Could not import scripts/run_turn1_target_finder.py. "
        "Copy this file into your repo's scripts/ directory beside run_turn1_target_finder.py."
    ) from exc

try:
    from src import probability as exact_probability
except Exception:  # pragma: no cover
    exact_probability = None


@dataclass
class GoalOption:
    raw: str
    norm: str


@dataclass
class GoalRequirement:
    label: str
    options: List[GoalOption]
    zone: str = "accessed"
    min_count: int = 1


@dataclass
class GoalTracker:
    accessed: List[Dict[str, Any]] = field(default_factory=list)
    accessed_names: List[str] = field(default_factory=list)
    seen_instance_ids: set = field(default_factory=set)

    def mark(self, cards: Iterable[Dict[str, Any]]) -> None:
        for c in cards:
            if not isinstance(c, dict):
                continue
            inst = c.get("_instance_id") or id(c)
            if inst in self.seen_instance_ids:
                continue
            self.seen_instance_ids.add(inst)
            self.accessed.append(c)
            self.accessed_names.append(tf.card_name(c))


def pct(x: float) -> float:
    return tf.pct(x)


def ci95(successes: int, n: int) -> Dict[str, float]:
    return tf.ci95(successes, n)


def default_report_file(filename: str) -> str:
    return os.path.join("data", "reports", "simulator_readiness", filename)


def parse_goal_string(goal: str, default_zone: str = "accessed") -> List[GoalRequirement]:
    """Parse a compact goal string.

    Commas separate AND requirements. A pipe creates an OR group inside a
    requirement.

    Examples:
      me1-76,me1-77,me1-132
      me1-76|me1-72,me1-77

    Zone override syntax is optional:
      me1-76@in_play,me1-77@hand,me1-132@hand
    """
    reqs: List[GoalRequirement] = []
    for idx, part in enumerate([p.strip() for p in str(goal or "").split(",") if p.strip()], start=1):
        zone = default_zone
        body = part
        if "@" in part:
            body, zone_part = part.rsplit("@", 1)
            body = body.strip()
            zone = zone_part.strip() or default_zone
        options = [GoalOption(raw=o.strip(), norm=tf.norm(o.strip())) for o in body.split("|") if o.strip()]
        if not options:
            continue
        label = " OR ".join(o.raw for o in options)
        reqs.append(GoalRequirement(label=label, options=options, zone=zone, min_count=1))
    return reqs


def parse_goal_file(path: str, default_zone: str = "accessed") -> Tuple[str, str, List[GoalRequirement]]:
    data = json.load(open(path, encoding="utf-8"))
    name = str(data.get("name") or os.path.splitext(os.path.basename(path))[0])
    mode = str(data.get("mode") or data.get("goal_mode") or "all").lower()
    reqs: List[GoalRequirement] = []
    for i, row in enumerate(data.get("requirements") or data.get("goals") or [], start=1):
        if isinstance(row, str):
            reqs.extend(parse_goal_string(row, default_zone=default_zone))
            continue
        if not isinstance(row, dict):
            continue
        zone = str(row.get("zone") or default_zone)
        min_count = int(row.get("min_count") or 1)
        raw_options = row.get("any_of") or row.get("options") or row.get("card") or row.get("id") or row.get("name")
        if isinstance(raw_options, str):
            options_raw = [raw_options]
        else:
            options_raw = list(raw_options or [])
        options = [GoalOption(raw=str(o), norm=tf.norm(o)) for o in options_raw if str(o).strip()]
        if not options:
            continue
        label = str(row.get("label") or " OR ".join(o.raw for o in options) or f"requirement_{i}")
        reqs.append(GoalRequirement(label=label, options=options, zone=zone, min_count=min_count))
    return name, mode, reqs



def instantiate_deck(deck: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deep-copy resolved card definitions into physical card instances.

    The resolver may return the same dict object for multiple copies of a card.
    For multi-card goals and min_count support, this script needs physical
    instances to be distinguishable.
    """
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(deck):
        cc = copy.deepcopy(c)
        cc["_instance_id"] = f"deckcopy_{i:03d}_{tf.card_id(cc)}"
        out.append(cc)
    return out

def card_matches_option(card: Dict[str, Any], option: GoalOption) -> bool:
    return tf.target_matches(card, option.norm)


def zone_cards(st: tf.SimState, tracker: GoalTracker, zone: str) -> List[Dict[str, Any]]:
    z = zone.lower().replace("-", "_")
    cards: List[Dict[str, Any]] = []
    if z in {"accessed", "seen", "obtained"}:
        cards.extend(tracker.accessed)
    elif z in {"hand"}:
        cards.extend(st.hand)
    elif z in {"in_play", "play", "board"}:
        if st.active is not None:
            cards.append(st.active)
        cards.extend(st.bench)
    elif z in {"hand_or_in_play", "hand_or_play", "available"}:
        cards.extend(st.hand)
        if st.active is not None:
            cards.append(st.active)
        cards.extend(st.bench)
    elif z == "discard":
        cards.extend(st.discard)
    elif z == "prizes":
        cards.extend(st.prizes)
    else:
        # Unknown zone defaults to accessed so old compact goals still work.
        cards.extend(tracker.accessed)
    return cards


def requirement_satisfied(req: GoalRequirement, st: tf.SimState, tracker: GoalTracker) -> bool:
    pool = zone_cards(st, tracker, req.zone)
    count = 0
    for c in pool:
        if any(card_matches_option(c, opt) for opt in req.options):
            count += 1
            if count >= req.min_count:
                return True
    return False


def goal_satisfied(reqs: Sequence[GoalRequirement], mode: str, st: tf.SimState, tracker: GoalTracker) -> bool:
    if not reqs:
        return False
    satisfied = [requirement_satisfied(req, st, tracker) for req in reqs]
    return any(satisfied) if mode == "any" else all(satisfied)


def missing_requirements(reqs: Sequence[GoalRequirement], mode: str, st: tf.SimState, tracker: GoalTracker) -> List[GoalRequirement]:
    if mode == "any":
        return [] if goal_satisfied(reqs, mode, st, tracker) else list(reqs)
    return [req for req in reqs if not requirement_satisfied(req, st, tracker)]


def all_goal_norms(reqs: Sequence[GoalRequirement]) -> List[str]:
    out: List[str] = []
    for r in reqs:
        for o in r.options:
            if o.norm and o.norm not in out:
                out.append(o.norm)
    return out

def goal_groups_for_exact_probability(
    deck: Sequence[Dict[str, Any]],
    reqs: Sequence[GoalRequirement],
    mode: str,
) -> List[Dict[str, Any]]:
    """Convert parsed goal requirements into src.probability goal groups.

    Each requirement is a group. Options inside a requirement are OR options.
    For top-level goal_mode="any", all requirement options are merged into a
    single OR group because probability.py's goal model is AND over groups.
    """

    def option_row(opt: GoalOption) -> Dict[str, Any]:
        matching = [c for c in deck if tf.target_matches(c, opt.norm)]
        return {
            "card_id": opt.raw,
            "count": len(matching),
            "is_basic": any(tf.is_basic_pokemon(c) for c in matching),
        }

    if mode == "any":
        merged_options: List[Dict[str, Any]] = []
        labels: List[str] = []
        for req in reqs:
            labels.append(req.label)
            for opt in req.options:
                row = option_row(opt)
                if row["count"] > 0:
                    merged_options.append(row)
        return [{"name": " OR ".join(labels), "min_count": 1, "options": merged_options}]

    groups: List[Dict[str, Any]] = []
    for req in reqs:
        options = []
        for opt in req.options:
            row = option_row(opt)
            if row["count"] > 0:
                options.append(row)
        groups.append({"name": req.label, "min_count": req.min_count, "options": options})
    return groups


def compute_exact_goal_baselines(
    deck: Sequence[Dict[str, Any]],
    reqs: Sequence[GoalRequirement],
    mode: str,
    hand_size: int,
    prize_count: int,
) -> Dict[str, Any]:
    """Use src.probability.py for exact natural goal baselines when available."""
    if exact_probability is None:
        return {"available": False, "reason": "Could not import src.probability."}

    deck_size = len(deck)
    basic_count = sum(1 for c in deck if tf.is_basic_pokemon(c))
    groups = goal_groups_for_exact_probability(deck, reqs, mode)

    try:
        natural = exact_probability.goal_natural_probabilities_given_legal_opening(
            deck_size=deck_size,
            basic_count=basic_count,
            goal_groups=groups,
            hand_size=hand_size,
        )
        prizes = exact_probability.goal_group_all_prized_probabilities_after_legal_hand(
            deck_size=deck_size,
            basic_count=basic_count,
            goal_groups=groups,
            prize_count=prize_count,
            hand_size=hand_size,
        )
        no_basic = exact_probability.p_no_basic_opening_7(deck_size, basic_count)
        return {
            "available": True,
            "source": "src/probability.py",
            "conditioning": "Exact natural probabilities are conditioned on keeping a legal opening hand with at least one Basic Pokémon.",
            "deck_size": deck_size,
            "basic_pokemon": basic_count,
            "goal_groups": groups,
            "no_basic_opening_hand_percent": pct(no_basic),
            "legal_opening_hand_percent": pct(1.0 - no_basic),
            **natural,
            "prize_lockout_by_group": prizes,
        }
    except Exception as exc:
        return {"available": False, "reason": f"Exact goal baseline failed: {exc}", "goal_groups": groups}


def add_exact_plus_simulation(scenario: Dict[str, Any], exact_baselines: Dict[str, Any]) -> None:
    """Attach exact+simulation hybrid estimates to one scenario.

    Simulation still determines the conditional action success rate. Exact math
    provides P(goal by natural draw). Final estimate is:

        P(natural) + P(not natural) * P(actions succeed | not natural)
    """
    sm = scenario.get("summary", {})
    trials = int(sm.get("trials") or 0)
    if not exact_baselines.get("available") or trials <= 0:
        scenario["exact_plus_simulation"] = {"available": False, "reason": exact_baselines.get("reason")}
        return

    by_stage = {row.get("stage"): int(row.get("count") or 0) for row in sm.get("success_by_stage", [])}
    natural_sim = by_stage.get("opening_hand", 0) + by_stage.get("draw_for_turn", 0)
    action_successes = by_stage.get("after_actions", 0)
    post_natural_trials = max(0, trials - natural_sim)
    conditional_action = action_successes / post_natural_trials if post_natural_trials else 0.0

    exact_open = float(exact_baselines.get("opening_hand_goal_probability") or 0.0)
    exact_draw = float(exact_baselines.get("after_turn_draw_goal_probability") or 0.0)
    final = exact_draw + (1.0 - exact_draw) * conditional_action
    action_increment = final - exact_draw

    line_rows = []
    for row in sm.get("top_success_lines", []) or []:
        count = int(row.get("count") or 0)
        conditional_line = count / post_natural_trials if post_natural_trials else 0.0
        line_rows.append({
            "line": row.get("line"),
            "count": count,
            "raw_percent_of_trials": row.get("percent"),
            "conditional_on_not_natural_percent": pct(conditional_line),
            "exact_weighted_percent_of_trials": pct((1.0 - exact_draw) * conditional_line),
        })

    scenario["exact_plus_simulation"] = {
        "available": True,
        "method": "Exact legal-opening/draw baseline from src/probability.py plus simulated conditional action success.",
        "exact_opening_hand_goal_percent": pct(exact_open),
        "exact_draw_for_turn_increment_percent": pct(exact_draw - exact_open),
        "exact_seen_by_draw_for_turn_percent": pct(exact_draw),
        "simulated_action_success_given_not_natural_percent": pct(conditional_action),
        "exact_weighted_action_increment_percent": pct(action_increment),
        "final_exact_plus_sim_percent": pct(final),
        "post_natural_trials": post_natural_trials,
        "line_contributions": line_rows,
    }


def choose_primary_target_norm(req: GoalRequirement, st: tf.SimState) -> Optional[str]:
    """Pick the best option norm for a requirement.

    Prefer an option that has copies left in deck, otherwise any option. This
    lets OR groups work naturally with searches.
    """
    for opt in req.options:
        if any(tf.target_matches(c, opt.norm) for c in st.deck):
            return opt.norm
    return req.options[0].norm if req.options else None


def snapshot_accessed(tracker: GoalTracker, st: tf.SimState) -> None:
    """Record cards that have been accessible/seen this turn.

    This intentionally includes hand, active, bench, and discard. If a card was
    in hand and later used/discarded, it was still accessed. It does not include
    deck or prizes.
    """
    cards = list(st.hand) + list(st.discard) + list(st.bench)
    if st.active is not None:
        cards.append(st.active)
    tracker.mark(cards)


def protected_discard_fodder_cards(st: tf.SimState, n: int, target_norm: str, stage: str) -> List[str]:
    """Goal-aware replacement for target_finder.discard_fodder_cards.

    For final-zone goals, avoid discarding any goal piece, not only the current
    single-target piece. For accessed goals this is less important, but still
    avoids throwing away unaccessed package pieces when possible.
    """
    protected_norms = list(getattr(st, "protected_goal_norms", []) or [])
    discarded: List[str] = []
    for _ in range(max(0, n)):
        candidates = [c for c in st.hand if not any(tf.target_matches(c, p) for p in protected_norms)]
        if not candidates:
            break

        def discard_priority(c: Dict[str, Any]) -> Tuple[int, int, int, str]:
            return (
                1 if tf.card_has_specific_play_effect(c) or tf.card_has_play_effect(c) else 0,
                1 if any(tf.card_directly_searches_target(c, p, st.deck) for p in protected_norms) else 0,
                tf.card_draw_power(c),
                tf.card_name(c),
            )

        candidates.sort(key=discard_priority)
        chosen = candidates[0]
        st.hand.remove(chosen)
        st.discard.append(chosen)
        discarded.append(tf.card_name(chosen))
    if discarded:
        st.log.append({"event": "discard_fodder_for_cost", "stage": stage, "discarded": discarded})
    return discarded


def protected_has_enough_discard_fodder(hand: Sequence[Dict[str, Any]], card: Dict[str, Any], target_norm: str) -> bool:
    # This function cannot see state, so it only protects the current target.
    # The stateful discard function above does the full goal-piece protection.
    cost = tf.card_known_discard_cost(card)
    if cost <= 0:
        return True
    others = [c for c in hand if c is not card and not tf.target_matches(c, target_norm)]
    return len(others) >= cost


def install_goal_safety_patches() -> None:
    # Monkey patch only inside this process. This lets target_finder's Ultra Ball
    # / discard-cost code avoid tossing goal pieces when we reuse it here.
    tf.discard_fodder_cards = protected_discard_fodder_cards
    tf.has_enough_discard_fodder = protected_has_enough_discard_fodder


def action_label(action: Any) -> str:
    if isinstance(action, dict) and action.get("_virtual_action"):
        if action.get("card"):
            return tf.card_name(action["card"])
        if action.get("source"):
            return tf.card_name(action["source"])
        if action.get("search_card"):
            return tf.card_name(action["search_card"])
        return str(action.get("_virtual_action"))
    if isinstance(action, dict):
        return tf.card_name(action)
    return str(action)


def score_actions_for_goal(
    st: tf.SimState,
    reqs: Sequence[GoalRequirement],
    mode: str,
    going: str,
    enable_chain_search: bool,
) -> List[Tuple[float, Any, str]]:
    missing = missing_requirements(reqs, mode, st, GoalTracker())
    # The caller passes a real tracker when checking satisfaction; for scoring,
    # missing reqs are recomputed by caller in a wrapper below. This function is
    # kept unused if called directly.
    return []


def score_candidate_for_missing_targets(
    st: tf.SimState,
    missing: Sequence[GoalRequirement],
    going: str,
    enable_chain_search: bool,
) -> List[Tuple[float, Any, str]]:
    scored: List[Tuple[float, Any, str]] = []
    target_norms = [n for req in missing for n in [choose_primary_target_norm(req, st)] if n]
    if not target_norms:
        return scored

    # Trainer / modeled-from-hand actions.
    playable = [c for c in list(st.hand) if tf.card_can_be_played_from_hand(c, going, st.supporter_used)]
    for c in playable:
        best_score = -1.0
        best_target = target_norms[0]
        for tn in target_norms:
            s = tf.score_playable_card(c, st, tn, going, enable_chain_search)
            if s > best_score:
                best_score = s
                best_target = tn
        if best_score > 0:
            scored.append((best_score, c, best_target))

    # Basic Pokémon from hand can be benched if their ability helps any missing piece.
    for c in list(st.hand):
        best_score = -1.0
        best_target = target_norms[0]
        for tn in target_norms:
            s = tf.bench_basic_ability_score(st, c, tn, going)
            if s > best_score:
                best_score = s
                best_target = tn
        if best_score > 0:
            scored.append((best_score, {"_virtual_action": "BenchAbility", "card": c}, best_target))

    # Explicit active/bench abilities and generic abilities.
    for tn in target_norms:
        s = tf.run_errand_score(st, tn)
        if s > 0:
            scored.append((s, {"_virtual_action": "Run Errand"}, tn))
        s = tf.teal_dance_score(st, tn)
        if s > 0:
            scored.append((s, {"_virtual_action": "Teal Dance"}, tn))
        for score, source, eff in tf.generic_ability_candidates(st, tn):
            if score > 0:
                scored.append((score, {"_virtual_action": "GenericAbility", "source": source, "effect": eff}, tn))
        for score, action in tf.ability_requirement_search_candidates(st, tn, going, enable_chain_search):
            if score > 0:
                scored.append((score, action, tn))

    # If a missing requirement card is already in hand/in play under a stricter
    # final-zone requirement, no action is needed. The main loop checks that.
    return scored


def execute_action(st: tf.SimState, action: Any, target_norm: str, rng: random.Random, going: str, enable_chain_search: bool) -> None:
    if isinstance(action, dict) and action.get("_virtual_action") == "Run Errand":
        tf.use_run_errand(st, target_norm, "after_use_Run_Errand")
    elif isinstance(action, dict) and action.get("_virtual_action") == "Teal Dance":
        tf.use_teal_dance(st, target_norm, "after_use_Teal_Dance")
    elif isinstance(action, dict) and action.get("_virtual_action") == "BenchAbility":
        tf.bench_basic_for_ability(st, action["card"], rng, target_norm, going, enable_chain_search)
    elif isinstance(action, dict) and action.get("_virtual_action") == "GenericAbility":
        tf.use_generic_ability(st, action["source"], action["effect"], rng, target_norm, going, enable_chain_search)
    elif isinstance(action, dict) and action.get("_virtual_action") == "AbilityRequirementSearch":
        tf.use_ability_requirement_search_chain(st, action, rng, target_norm, going, enable_chain_search)
    else:
        tf.play_card(st, action, rng, target_norm, going, enable_chain_search)


def trial_goal_prize_status(deck: List[Dict[str, Any]], prizes: Sequence[Dict[str, Any]], reqs: Sequence[GoalRequirement]) -> Dict[str, Any]:
    rows = []
    for req in reqs:
        total = 0
        prized = 0
        for c in deck:
            if any(card_matches_option(c, opt) for opt in req.options):
                total += 1
        for c in prizes:
            if any(card_matches_option(c, opt) for opt in req.options):
                prized += 1
        rows.append({"requirement": req.label, "total_copies": total, "prized_copies": prized, "all_prized": total > 0 and prized == total})
    return {"requirements": rows, "any_all_prized": any(r["all_prized"] for r in rows)}


def simulate_one_goal_trial(
    deck: List[Dict[str, Any]],
    rng: random.Random,
    reqs: Sequence[GoalRequirement],
    mode: str,
    going: str,
    hand_size: int,
    prize_count: int,
    use_mulligans: bool,
    draw_for_turn: bool,
    max_actions: int,
    enable_chain_search: bool,
    goal_zone: str,
) -> Dict[str, Any]:
    goal_norms = all_goal_norms(reqs)
    active_target_for_setup = goal_norms[0] if goal_norms else ""

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
    active = tf.choose_optimal_active(opening, active_target_for_setup)
    hand_after_setup = list(opening)
    if active is not None and active in hand_after_setup:
        hand_after_setup.remove(active)

    st = tf.SimState(deck=library, hand=hand_after_setup, prizes=list(prizes), active=active)
    st.protected_goal_norms = goal_norms
    tracker = GoalTracker()
    snapshot_accessed(tracker, st)
    if active is not None:
        st.log.append({"event": "choose_active", "active": tf.card_name(active)})

    natural_success_stage: Optional[str] = None
    if goal_satisfied(reqs, mode, st, tracker):
        natural_success_stage = "opening_hand"
    elif draw_for_turn:
        before_draw_n = len(st.hand)
        tf.draw_cards(st, 1, "draw_for_turn")
        snapshot_accessed(tracker, st)
        if goal_satisfied(reqs, mode, st, tracker):
            natural_success_stage = "draw_for_turn"

    success_stage = natural_success_stage

    while success_stage is None and st.actions_used < max_actions:
        missing = missing_requirements(reqs, mode, st, tracker)
        if not missing:
            success_stage = "after_actions"
            break
        scored = score_candidate_for_missing_targets(st, missing, going, enable_chain_search)
        if not scored:
            break

        scored.sort(key=lambda x: (x[0], action_label(x[1])), reverse=True)
        _, action, tn = scored[0]
        execute_action(st, action, tn, rng, going, enable_chain_search)
        snapshot_accessed(tracker, st)

        # Reset single-target found flag so the reused target finder keeps going
        # for the remaining package pieces.
        st.found = False
        st.found_stage = None

        if goal_satisfied(reqs, mode, st, tracker):
            success_stage = "after_actions"
            break

    final_missing = missing_requirements(reqs, mode, st, tracker)
    prize_status = trial_goal_prize_status(deck, prizes, reqs)
    line = " -> ".join(st.line) if st.line else "none"
    return {
        "success": success_stage is not None,
        "success_stage": success_stage or "not_met",
        "line": line,
        "mulligans": mulligans,
        "actions_used": st.actions_used,
        "active": tf.card_name(active) if active else None,
        "final_hand_size": len(st.hand),
        "final_deck_size": len(st.deck),
        "missing_requirements": [r.label for r in final_missing],
        "missing_count": len(final_missing),
        "goal_pieces_prized": prize_status,
        "accessed_goal_piece_names": sorted(set(
            tf.card_name(c)
            for c in tracker.accessed
            if any(any(card_matches_option(c, opt) for opt in req.options) for req in reqs)
        )),
        "log": st.log,
    }


def summarize_goal_trials(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results)
    successes = sum(1 for r in results if r["success"])
    by_stage = Counter(r["success_stage"] for r in results)
    lines = Counter(r["line"] for r in results if r["success"] and r["line"] != "none")
    missing = Counter()
    for r in results:
        if not r["success"]:
            if r.get("missing_requirements"):
                for m in r["missing_requirements"]:
                    missing[m] += 1
            else:
                missing["unknown"] += 1
    any_all_prized = sum(1 for r in results if r.get("goal_pieces_prized", {}).get("any_all_prized"))
    mulligans = Counter(r["mulligans"] for r in results)
    return {
        "trials": n,
        "successes": successes,
        "probability": round(successes / n, 6) if n else 0.0,
        "percent": pct(successes / n) if n else 0.0,
        "ci95_percent": ci95(successes, n),
        "success_by_stage": [{"stage": k, "count": v, "percent": pct(v / n)} for k, v in by_stage.most_common()],
        "natural_opening_or_draw_percent": pct((by_stage.get("opening_hand", 0) + by_stage.get("draw_for_turn", 0)) / n) if n else 0.0,
        "action_success_increment_percent": pct(by_stage.get("after_actions", 0) / n) if n else 0.0,
        "top_success_lines": [{"line": k, "count": v, "percent": pct(v / n)} for k, v in lines.most_common(25)],
        "top_missing_requirements_on_failure": [{"requirement": k, "count": v, "percent_of_trials": pct(v / n), "percent_of_failures": pct(v / max(1, n - successes))} for k, v in missing.most_common(25)],
        "any_required_piece_all_prized": {"trials": any_all_prized, "percent": pct(any_all_prized / n) if n else 0.0},
        "mulligans": [{"mulligans": k, "count": v, "percent": pct(v / n)} for k, v in sorted(mulligans.items())],
    }


def run_goal_scenario(args: argparse.Namespace, deck: List[Dict[str, Any]], reqs: Sequence[GoalRequirement], mode: str, going: str) -> Dict[str, Any]:
    rng = random.Random(args.seed + (0 if going == "first" else 1000003))
    results = [
        simulate_one_goal_trial(
            deck=deck,
            rng=rng,
            reqs=reqs,
            mode=mode,
            going=going,
            hand_size=args.hand_size,
            prize_count=args.prizes,
            use_mulligans=not args.no_mulligans,
            draw_for_turn=not args.no_draw_for_turn,
            max_actions=args.max_actions,
            enable_chain_search=args.chain_search,
            goal_zone=args.goal_zone,
        )
        for _ in range(args.trials)
    ]
    examples = [r for r in results if r["success"] and r["line"] != "none"][: args.example_lines]
    failures = [r for r in results if not r["success"]][: args.example_lines]
    return {
        "going": going,
        "summary": summarize_goal_trials(results),
        "example_successes": examples,
        "example_failures": failures,
    }


def write_json(path: str, result: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def write_summary_csv(path: str, goal_name: str, scenarios: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    rows = []
    for sc in scenarios:
        sm = sc.get("summary", {})
        rows.append({
            "goal_name": goal_name,
            "going": sc.get("going"),
            "trials": sm.get("trials"),
            "successes": sm.get("successes"),
            "raw_sim_percent": sm.get("percent"),
            "ci95_low": (sm.get("ci95_percent") or {}).get("low"),
            "ci95_high": (sm.get("ci95_percent") or {}).get("high"),
            "sim_natural_opening_or_draw_percent": sm.get("natural_opening_or_draw_percent"),
            "sim_action_success_increment_percent": sm.get("action_success_increment_percent"),
            "exact_seen_by_draw_for_turn_percent": ((sc.get("exact_plus_simulation") or {}).get("exact_seen_by_draw_for_turn_percent")),
            "sim_action_success_given_not_natural_percent": ((sc.get("exact_plus_simulation") or {}).get("simulated_action_success_given_not_natural_percent")),
            "exact_weighted_action_increment_percent": ((sc.get("exact_plus_simulation") or {}).get("exact_weighted_action_increment_percent")),
            "final_exact_plus_sim_percent": ((sc.get("exact_plus_simulation") or {}).get("final_exact_plus_sim_percent")),
            "top_line_1": ((sm.get("top_success_lines") or [{}])[0]).get("line"),
            "top_line_1_percent": ((sm.get("top_success_lines") or [{}])[0]).get("percent"),
            "top_missing_1": ((sm.get("top_missing_requirements_on_failure") or [{}])[0]).get("requirement"),
            "top_missing_1_percent_of_failures": ((sm.get("top_missing_requirements_on_failure") or [{}])[0]).get("percent_of_failures"),
            "any_required_piece_all_prized_percent": (sm.get("any_required_piece_all_prized") or {}).get("percent"),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def write_lines_csv(path: str, goal_name: str, scenarios: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = ["goal_name", "going", "line", "count", "raw_percent", "conditional_on_not_natural_percent", "exact_weighted_percent_of_trials"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sc in scenarios:
            exact_lines = {r.get("line"): r for r in ((sc.get("exact_plus_simulation") or {}).get("line_contributions") or [])}
            for row in (sc.get("summary", {}).get("top_success_lines") or []):
                ex = exact_lines.get(row.get("line"), {})
                writer.writerow({
                    "goal_name": goal_name,
                    "going": sc.get("going"),
                    "line": row.get("line"),
                    "count": row.get("count"),
                    "raw_percent": row.get("percent"),
                    "conditional_on_not_natural_percent": ex.get("conditional_on_not_natural_percent"),
                    "exact_weighted_percent_of_trials": ex.get("exact_weighted_percent_of_trials"),
                })


def print_compact(result: Dict[str, Any]) -> None:
    compact = {
        "passed": result.get("passed"),
        "goal_name": result.get("goal_name"),
        "goal_mode": result.get("goal_mode"),
        "goal_zone": result.get("goal_zone"),
        "trials": result.get("trials"),
        "deck_size": result.get("deck_summary", {}).get("deck_size"),
        "outputs": result.get("outputs"),
        "scenarios": [
            {
                "going": s.get("going"),
                "raw_sim_percent": s.get("summary", {}).get("percent"),
                "final_exact_plus_sim_percent": (s.get("exact_plus_simulation") or {}).get("final_exact_plus_sim_percent"),
                "exact_seen_by_draw_for_turn_percent": (s.get("exact_plus_simulation") or {}).get("exact_seen_by_draw_for_turn_percent"),
                "ci95_percent": s.get("summary", {}).get("ci95_percent"),
                "top_missing": (s.get("summary", {}).get("top_missing_requirements_on_failure") or [])[:3],
            }
            for s in result.get("scenarios", [])
        ],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Turn 1 multi-card goal finder: estimate P(X and Y and Z by end of Turn 1).")
    ap.add_argument("--compiled", default="data/compiled_cards/auto/compiled_cards_all.json")
    ap.add_argument("--decklist", required=True)
    ap.add_argument("--goal", help="Comma-separated goal pieces. Use | for OR inside a requirement, and @zone for a zone override.")
    ap.add_argument("--goal-file", help="JSON goal file with name/mode/requirements.")
    ap.add_argument("--goal-name", default=None)
    ap.add_argument("--goal-mode", choices=["all", "any"], default="all", help="all = every requirement; any = at least one requirement.")
    ap.add_argument("--goal-zone", default="accessed", choices=["accessed", "hand", "in_play", "hand_or_in_play"], help="Default zone for compact --goal pieces.")
    ap.add_argument("--going", choices=["first", "second", "both"], default="both")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--hand-size", type=int, default=7)
    ap.add_argument("--prizes", type=int, default=6)
    ap.add_argument("--max-actions", type=int, default=30)
    ap.add_argument("--no-mulligans", action="store_true")
    ap.add_argument("--no-draw-for-turn", action="store_true")
    ap.add_argument("--complete-only", action="store_true")
    ap.add_argument("--chain-search", action="store_true")
    ap.add_argument("--example-lines", type=int, default=5)
    ap.add_argument("--out", default=default_report_file("turn1_goal_finder.json"))
    ap.add_argument("--csv-out", default=default_report_file("turn1_goal_finder_summary.csv"))
    ap.add_argument("--lines-csv", default=default_report_file("turn1_goal_finder_lines.csv"))
    args = ap.parse_args()

    install_goal_safety_patches()

    if tf.load_compiled_cards is None:
        raise RuntimeError("Could not import tcgsim.load_compiled_cards from src/tcgsim.")

    if args.goal_file:
        goal_name, file_mode, reqs = parse_goal_file(args.goal_file, default_zone=args.goal_zone)
        if args.goal_mode == "all" and file_mode in {"all", "any"}:
            # Preserve file mode unless user explicitly provided --goal-mode any.
            mode = file_mode
        else:
            mode = args.goal_mode
    else:
        if not args.goal:
            raise SystemExit("Provide either --goal or --goal-file.")
        reqs = parse_goal_string(args.goal, default_zone=args.goal_zone)
        goal_name = args.goal_name or args.goal
        mode = args.goal_mode

    if args.goal_name:
        goal_name = args.goal_name
    if not reqs:
        raise SystemExit("No valid goal requirements were parsed.")

    cards = tf.load_compiled_cards(args.compiled)
    if args.complete_only:
        cards = tf.filter_complete_cards(cards)

    raw_decklist = tf.parse_decklist(args.decklist)
    deck, unresolved = tf.resolve_decklist(raw_decklist, cards)
    deck = instantiate_deck(deck)

    result: Dict[str, Any] = {
        "passed": False,
        "compiled_source": args.compiled,
        "decklist_source": args.decklist,
        "goal_name": goal_name,
        "goal_mode": mode,
        "goal_zone": args.goal_zone,
        "requirements": [
            {"label": r.label, "zone": r.zone, "min_count": r.min_count, "options": [o.raw for o in r.options]}
            for r in reqs
        ],
        "trials": args.trials,
        "seed": args.seed,
        "assumptions": {
            "goal_zone_accessed": "A card counts if it was in hand/in play/discard at any snapshot this turn; deck/prize cards do not count.",
            "hand_or_in_play": "Use --goal-zone hand_or_in_play or per-piece @hand/@in_play if you need final board/hand state rather than access.",
            "going_first_cannot_play_supporter_on_turn_1": True,
            "chain_search_enabled": bool(args.chain_search),
            "policy": "Greedy best-first action choice, scoring each legal action against currently missing goal pieces.",
        },
        "decklist_entries": [{"count": c, "name": n} for c, n in raw_decklist],
        "unresolved": unresolved,
        "outputs": {"json": args.out, "summary_csv": args.csv_out, "lines_csv": args.lines_csv},
    }

    if unresolved:
        result["error"] = "Some decklist entries could not be resolved."
        write_json(args.out, result)
        print_compact(result)
        sys.exit(2)

    result["deck_summary"] = {
        "deck_size": len(deck),
        "basic_pokemon": sum(1 for c in deck if tf.is_basic_pokemon(c)),
        "pokemon": sum(1 for c in deck if tf.card_supertype(c) == "Pokémon"),
        "trainer": sum(1 for c in deck if tf.is_trainer(c)),
        "energy": sum(1 for c in deck if tf.is_energy(c)),
        "top_cards": [{"name": k, "count": v} for k, v in Counter(tf.card_name(c) for c in deck).most_common(30)],
    }

    exact_baselines = compute_exact_goal_baselines(
        deck=deck,
        reqs=reqs,
        mode=mode,
        hand_size=args.hand_size,
        prize_count=args.prizes,
    )
    result["exact_goal_baselines"] = exact_baselines

    goings = ["first", "second"] if args.going == "both" else [args.going]
    scenarios = [run_goal_scenario(args, deck, reqs, mode, g) for g in goings]
    for sc in scenarios:
        add_exact_plus_simulation(sc, exact_baselines)
    result["scenarios"] = scenarios
    result["passed"] = True

    write_json(args.out, result)
    write_summary_csv(args.csv_out, goal_name, scenarios)
    write_lines_csv(args.lines_csv, goal_name, scenarios)
    print_compact(result)


if __name__ == "__main__":
    main()
