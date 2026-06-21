# TURN1_DEWRAP_DIRECT_GOAL_FILTER
# TURN1_V62_PRECISE_SEARCH_FILTER_COMPAT
# TURN1_V57_ACTION_BUDGET_SINGLE_COUNT
# TURN1_DIRECT_ACTION_BUDGET_V56
# TURN1_DIRECT_CARD_MATCH_CACHE_V54
# TURN1_DIRECT_CAPACITY_AND_SCORE_CACHE_V53
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
import re
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
    data = json.load(open(path, encoding="utf-8-sig"))
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


# ---------------------------------------------------------------------
# TURN1_DIRECT_CARD_MATCH_CACHE_V54
# ---------------------------------------------------------------------
# Direct hot-path fix, not a wrapper.
#
# The profiler showed millions of card_matches_option calls from
# requirement_satisfied / goal_satisfied. The original matching logic is kept;
# this patch injects a cache into the original function body and routes every
# original return through _turn1_card_goal_match_store(...).

_TURN1_CARD_MATCH_CACHE = {}


def _turn1_card_key_for_match(card):
    if isinstance(card, dict):
        ident = card.get("identity") or {}
        key = (
            card.get("card_id")
            or card.get("representative_card_id")
            or card.get("id")
            or ident.get("card_id")
            or ident.get("id")
            or ident.get("canonical_id")
        )
        if key:
            return ("id", str(key))
        name = card.get("name") or card.get("card_name") or ident.get("name") or ident.get("canonical_name")
        set_code = card.get("set_code") or ident.get("set_code") or card.get("set") or ident.get("set")
        number = card.get("number") or card.get("collector_number") or ident.get("number") or ident.get("collector_number")
        if name:
            return ("name", str(name), str(set_code or ""), str(number or ""))
    return ("obj", id(card))


def _turn1_goal_option_key_for_match(option):
    if option is None or isinstance(option, (str, int, float, bool)):
        return option
    if hasattr(option, "__dict__"):
        try:
            return (type(option).__name__, tuple(sorted((str(k), _turn1_goal_option_key_for_match(v)) for k, v in vars(option).items())))
        except Exception:
            return (type(option).__name__, id(option))
    if isinstance(option, dict):
        keep = {}
        for k, v in option.items():
            if str(k) in {"label", "name", "target", "target_norm", "options", "aliases", "card", "card_name", "set_code", "number", "collector_number"}:
                keep[str(k)] = _turn1_goal_option_key_for_match(v)
        if keep:
            return ("dict", tuple(sorted(keep.items())))
        return ("dict_id", id(option))
    if isinstance(option, (list, tuple)):
        return (type(option).__name__, tuple(_turn1_goal_option_key_for_match(x) for x in option))
    if isinstance(option, set):
        try:
            return ("set", tuple(sorted(_turn1_goal_option_key_for_match(x) for x in option)))
        except Exception:
            return ("set_id", id(option))
    return (type(option).__name__, repr(option))


def _turn1_card_goal_match_key(card, option):
    return (_turn1_card_key_for_match(card), _turn1_goal_option_key_for_match(option))


def _turn1_card_goal_match_store(key, value):
    result = bool(value)
    if len(_TURN1_CARD_MATCH_CACHE) < 500000:
        _TURN1_CARD_MATCH_CACHE[key] = result
    return result

def card_matches_option(card: Dict[str, Any], option: GoalOption) -> bool:
    _turn1_card_match_cache_key = _turn1_card_goal_match_key(card, option)
    _turn1_card_match_cached = _TURN1_CARD_MATCH_CACHE.get(_turn1_card_match_cache_key)
    if _turn1_card_match_cached is not None:
        return _turn1_card_goal_match_store(_turn1_card_match_cache_key, _turn1_card_match_cached)
    return _turn1_card_goal_match_store(_turn1_card_match_cache_key, tf.target_matches(card, option.norm))

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
    # TURN1_HARDEN_MIN_COUNT_REQUIREMENT_V19
    # Count distinct physical card instances, not merely "does this card exist".
    pool = zone_cards(st, tracker, req.zone)
    needed = max(1, int(getattr(req, "min_count", 1) or 1))

    count = 0
    seen_instances = set()

    for c in pool:
        if not isinstance(c, dict):
            continue

        if not any(card_matches_option(c, opt) for opt in req.options):
            continue

        inst = c.get("_instance_id") or id(c)

        if inst in seen_instances:
            continue

        seen_instances.add(inst)
        count += 1

        if count >= needed:
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

    context_rows = []
    for row in sm.get("top_success_line_contexts", []) or []:
        count = int(row.get("count") or 0)
        conditional_line = count / post_natural_trials if post_natural_trials else 0.0
        context_rows.append({
            "starting_hand_draw": row.get("starting_hand_draw"),
            "played": row.get("played"),
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
        "line_context_contributions": context_rows,
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
    # TURN1_ACTION_BUDGET_GUARD
    if _turn1_action_budget_exhausted(st):
        return []
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
    # TURN1_V57_BASE_EXECUTE_BUDGET_GUARD
    if not _turn1_action_budget_allows_next_action(st):
        return None
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




# ---------------------------------------------------------------------
# TURN1_GOAL_COMPATIBLE_ACCESS_FILTERS
# ---------------------------------------------------------------------
# Broad Turn-1 access guard:
# - A search/draw/search-like action can only help if its search filter is
#   compatible with the currently missing goal cards.
# - Active-only abilities may only be used if the source Pokémon is actually
#   Active.
#
# This fixes classes of errors such as:
# - Chien-Pao ex / Shivery Chill being used for a Pokémon goal even though it
#   searches Basic Water Energy.
# - Tatsugiri / Attract Customers being used for a Pokémon goal even though it
#   searches Supporters.
# - Active-only abilities being used from the Bench after Nest Ball.

_ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_BEFORE_GOAL_ACCESS_FILTER = score_candidate_for_missing_targets


def turn1_goal_access_filter_flatten_text(obj: Any, depth: int = 0) -> str:
    if obj is None or depth > 4:
        return ""

    if isinstance(obj, str):
        return obj

    if isinstance(obj, (int, float, bool)):
        return str(obj)

    if isinstance(obj, (list, tuple, set)):
        return " ".join(turn1_goal_access_filter_flatten_text(x, depth + 1) for x in obj)

    if isinstance(obj, dict):
        pieces: List[str] = []
        preferred = [
            "name",
            "label",
            "text",
            "effect",
            "effect_text",
            "ability_text",
            "abilities_text",
            "attacks_text",
            "combined_text",
            "rules",
            "source_text",
            "raw_text",
            "compiled_effects",
            "effects",
            "steps",
            "search_filter",
            "filters",
            "target",
            "zone",
            "from_zone",
            "to_zone",
            "destination",
            "owner",
            "player",
        ]

        for k in preferred:
            if k in obj:
                pieces.append(turn1_goal_access_filter_flatten_text(obj.get(k), depth + 1))

        # Include nested identity/gameplay/source too; these often contain the
        # original card text in compiled-card records.
        for k in ["identity", "gameplay", "source", "ability", "attack"]:
            if k in obj:
                pieces.append(turn1_goal_access_filter_flatten_text(obj.get(k), depth + 1))

        return " ".join(p for p in pieces if p)

    return str(obj)


def turn1_goal_access_filter_norm_text(s: str) -> str:
    s = str(s or "").lower()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = s.replace("{w}", " water ").replace("[w]", " water ")
    s = s.replace("{f}", " fighting ").replace("[f]", " fighting ")
    s = s.replace("{g}", " grass ").replace("[g]", " grass ")
    s = s.replace("{l}", " lightning ").replace("[l]", " lightning ")
    s = s.replace("{r}", " fire ").replace("[r]", " fire ")
    s = s.replace("{p}", " psychic ").replace("[p]", " psychic ")
    s = s.replace("{d}", " darkness ").replace("[d]", " darkness ")
    s = s.replace("{m}", " metal ").replace("[m]", " metal ")
    s = s.replace("{c}", " colorless ").replace("[c]", " colorless ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def turn1_goal_access_filter_action_text(action: Any) -> str:
    pieces: List[str] = []

    try:
        pieces.append(action_label(action))
    except Exception:
        pass

    if isinstance(action, dict):
        for k in ["source", "card", "search_card", "effect", "ability", "compiled_effect", "step"]:
            if k in action:
                pieces.append(turn1_goal_access_filter_flatten_text(action.get(k)))
    else:
        pieces.append(turn1_goal_access_filter_flatten_text(action))

    return turn1_goal_access_filter_norm_text(" ".join(p for p in pieces if p))


def turn1_goal_access_filter_card_name(card: Dict[str, Any]) -> str:
    try:
        return tf.card_name(card)
    except Exception:
        if isinstance(card, dict):
            ident = card.get("identity") if isinstance(card.get("identity"), dict) else {}
            return str(card.get("name") or ident.get("name") or "")
        return ""


def turn1_goal_access_filter_card_supertype(card: Dict[str, Any]) -> str:
    if not isinstance(card, dict):
        return ""
    ident = card.get("identity") if isinstance(card.get("identity"), dict) else {}
    return str(card.get("supertype") or ident.get("supertype") or "").lower()


def turn1_goal_access_filter_card_subtypes(card: Dict[str, Any]) -> List[str]:
    if not isinstance(card, dict):
        return []

    vals: List[Any] = []
    ident = card.get("identity") if isinstance(card.get("identity"), dict) else {}

    for source in [card, ident]:
        for k in ["subtypes", "subtype"]:
            v = source.get(k)
            if isinstance(v, list):
                vals.extend(v)
            elif v:
                vals.append(v)

    out: List[str] = []
    for v in vals:
        for part in str(v).replace("[", " ").replace("]", " ").replace("'", " ").replace('"', " ").split(","):
            p = part.strip().lower()
            if p:
                out.append(p)
    return out


def turn1_goal_access_filter_card_types(card: Dict[str, Any]) -> List[str]:
    if not isinstance(card, dict):
        return []

    vals: List[Any] = []
    ident = card.get("identity") if isinstance(card.get("identity"), dict) else {}

    for source in [card, ident]:
        v = source.get("types")
        if isinstance(v, list):
            vals.extend(v)
        elif v:
            vals.append(v)

    name = turn1_goal_access_filter_card_name(card).lower()
    text = turn1_goal_access_filter_norm_text(turn1_goal_access_filter_flatten_text(card))

    out = [str(v).strip().lower() for v in vals if str(v).strip()]
    for t in ["water", "fire", "grass", "lightning", "psychic", "fighting", "darkness", "metal", "dragon", "colorless"]:
        if t in name or f" {t} " in f" {text} ":
            if t not in out:
                out.append(t)
    return out


def turn1_goal_access_filter_is_basic_pokemon(card: Dict[str, Any]) -> bool:
    try:
        return bool(tf.is_basic_pokemon(card))
    except Exception:
        return turn1_goal_access_filter_card_supertype(card) == "pokémon" and any("basic" in s for s in turn1_goal_access_filter_card_subtypes(card))


def turn1_goal_access_filter_same_card_instance(a: Any, b: Any) -> bool:
    if not isinstance(a, dict) or not isinstance(b, dict):
        return False

    aid = a.get("_instance_id") or a.get("card_id") or a.get("id")
    bid = b.get("_instance_id") or b.get("card_id") or b.get("id")

    if aid and bid and aid == bid:
        return True

    return turn1_goal_access_filter_card_name(a) and turn1_goal_access_filter_card_name(a) == turn1_goal_access_filter_card_name(b)


def turn1_goal_access_filter_source_card_from_action(action: Any) -> Optional[Dict[str, Any]]:
    if isinstance(action, dict):
        for k in ["source", "card", "search_card"]:
            v = action.get(k)
            if isinstance(v, dict):
                return v
    if isinstance(action, dict):
        return action
    return None


def turn1_goal_access_filter_requires_active_spot(action_text: str) -> bool:
    t = turn1_goal_access_filter_norm_text(action_text)
    phrases = [
        "if this pokemon is in the active spot",
        "if this pokémon is in the active spot",
        "if this pokemon is your active pokemon",
        "if this pokémon is your active pokémon",
        "while this pokemon is in the active spot",
        "while this pokémon is in the active spot",
        "if this pokemon is active",
        "if this pokémon is active",
    ]
    return any(p in t for p in phrases)


def turn1_goal_access_filter_active_requirement_ok(action: Any, st: tf.SimState) -> bool:
    text = turn1_goal_access_filter_action_text(action)
    if not turn1_goal_access_filter_requires_active_spot(text):
        return True

    source = turn1_goal_access_filter_source_card_from_action(action)
    if source is None:
        return False

    return st.active is not None and turn1_goal_access_filter_same_card_instance(source, st.active)


def turn1_goal_access_filter_has_self_access_context(t: str) -> bool:
    t = turn1_goal_access_filter_norm_text(t)

    # Strong self-access phrases.
    if any(p in t for p in [
        "search your deck",
        "from your deck",
        "look at the top",
        "put it into your hand",
        "put them into your hand",
        "put it onto your bench",
        "put them onto your bench",
        "draw a card",
        "draw cards",
    ]):
        return True

    return False


def turn1_goal_access_filters_from_text(text: str) -> set:
    """
    Extract strong target-class restrictions from an effect/card text.

    The returned filters mean: this action is only allowed to count as Turn-1
    access if at least one missing goal card belongs to that same class.
    Unknown/unrestricted draw effects return an empty set and are allowed.
    """
    t = turn1_goal_access_filter_norm_text(text)

    if not turn1_goal_access_filter_has_self_access_context(t):
        return set()

    filters = set()

    # Trainer class searches.
    if "supporter" in t:
        filters.add("supporter")
    if "item card" in t or "item cards" in t:
        filters.add("item")
    if "stadium" in t:
        filters.add("stadium")
    if "pokemon tool" in t or "pokémon tool" in t or " tool card" in t:
        filters.add("tool")

    # Pokémon class searches.
    if "basic pokemon" in t or "basic pokémon" in t:
        filters.add("basic_pokemon")
    elif "pokemon" in t or "pokémon" in t:
        filters.add("pokemon")

    # Type-restricted Pokémon searches, e.g. Irida.
    for typ in ["water", "fire", "grass", "lightning", "psychic", "fighting", "darkness", "metal", "dragon", "colorless"]:
        if f"{typ} pokemon" in t or f"{typ} pokémon" in t:
            filters.add(f"{typ}_pokemon")

    # Energy class searches. Check specific first.
    for typ in ["water", "fire", "grass", "lightning", "psychic", "fighting", "darkness", "metal", "colorless"]:
        if f"basic {typ} energy" in t:
            filters.add(f"basic_{typ}_energy")
        elif f"{typ} energy" in t:
            filters.add(f"{typ}_energy")

    if "basic energy" in t:
        filters.add("basic_energy")
    elif "energy" in t:
        filters.add("energy")

    return filters


def turn1_goal_access_filter_goal_cards_for_req(st: tf.SimState, req: GoalRequirement) -> List[Dict[str, Any]]:
    pools: List[Dict[str, Any]] = []

    for attr in ["hand", "deck", "discard", "bench", "prizes"]:
        try:
            vals = getattr(st, attr, []) or []
            pools.extend([c for c in vals if isinstance(c, dict)])
        except Exception:
            pass

    try:
        if st.active is not None:
            pools.append(st.active)
    except Exception:
        pass

    out: List[Dict[str, Any]] = []
    seen = set()

    for c in pools:
        if not any(card_matches_option(c, opt) for opt in req.options):
            continue

        key = c.get("_instance_id") or c.get("card_id") or c.get("id") or id(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)

    return out


def turn1_goal_access_filter_matches_card(filter_name: str, card: Dict[str, Any]) -> bool:
    sup = turn1_goal_access_filter_card_supertype(card)
    subs = turn1_goal_access_filter_card_subtypes(card)
    types = turn1_goal_access_filter_card_types(card)
    name = turn1_goal_access_filter_card_name(card).lower()
    text = turn1_goal_access_filter_norm_text(turn1_goal_access_filter_flatten_text(card))

    if filter_name == "pokemon":
        return sup in {"pokémon", "pokemon"}
    if filter_name == "basic_pokemon":
        return sup in {"pokémon", "pokemon"} and turn1_goal_access_filter_is_basic_pokemon(card)
    if filter_name.endswith("_pokemon"):
        typ = filter_name.removesuffix("_pokemon")
        return sup in {"pokémon", "pokemon"} and typ in types

    if filter_name == "energy":
        return sup == "energy" or " energy" in name
    if filter_name == "basic_energy":
        return (sup == "energy" or " energy" in name) and ("basic" in name or any("basic" in s for s in subs) or "basic energy" in text)
    if filter_name.endswith("_energy"):
        base = filter_name.removesuffix("_energy")
        if base.startswith("basic_"):
            typ = base.removeprefix("basic_")
            return turn1_goal_access_filter_matches_card("basic_energy", card) and (typ in types or typ in name or f" {typ} " in f" {text} ")
        typ = base
        return (sup == "energy" or " energy" in name) and (typ in types or typ in name or f" {typ} " in f" {text} ")

    if filter_name == "supporter":
        return sup == "trainer" and (any("supporter" in s for s in subs) or "supporter" in name or "supporter" in text)
    if filter_name == "item":
        return sup == "trainer" and (any("item" == s or "item" in s for s in subs) or "item" in text)
    if filter_name == "stadium":
        return sup == "trainer" and (any("stadium" in s for s in subs) or "stadium" in text)
    if filter_name == "tool":
        return sup == "trainer" and (any("tool" in s for s in subs) or "tool" in text)

    return False


def turn1_goal_access_filters_compatible_with_missing(st: tf.SimState, filters: set, missing: Sequence[GoalRequirement]) -> bool:
    if not filters:
        return True

    for req in missing:
        goal_cards = turn1_goal_access_filter_goal_cards_for_req(st, req)

        # If we cannot identify the goal card type, do not block. This keeps the
        # guard conservative and avoids false negatives on unresolved cards.
        if not goal_cards:
            return True

        for c in goal_cards:
            if any(turn1_goal_access_filter_matches_card(f, c) for f in filters):
                return True

    return False


def turn1_action_allowed_for_missing_goal_access(action: Any, st: tf.SimState, missing: Sequence[GoalRequirement], going: str) -> bool:
    text = turn1_goal_access_filter_action_text(action)

    if not turn1_goal_access_filter_active_requirement_ok(action, st):
        try:
            st.log.append({
                "event": "blocked_illegal_active_only_ability",
                "action": action_label(action),
                "reason": "Ability text requires the source Pokémon to be in the Active Spot.",
            })
        except Exception:
            pass
        return False

    filters = turn1_goal_access_filters_from_text(text)

    if filters and not turn1_goal_access_filters_compatible_with_missing(st, filters, missing):
        try:
            st.log.append({
                "event": "blocked_incompatible_access_filter",
                "action": action_label(action),
                "filters": sorted(filters),
                "missing_requirements": [getattr(r, "label", str(r)) for r in missing],
                "reason": "Action searches a card class that cannot satisfy the current missing goal cards.",
            })
        except Exception:
            pass
        return False

    return True


def score_candidate_for_missing_targets(
    st: tf.SimState,
    missing: Sequence[GoalRequirement],
    going: str,
    enable_chain_search: bool,
) -> List[Tuple[float, Any, str]]:
    # TURN1_ACTION_BUDGET_GUARD
    if _turn1_action_budget_exhausted(st):
        return []
    scored = _ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_BEFORE_GOAL_ACCESS_FILTER(st, missing, going, enable_chain_search)

    if not scored:
        return scored

    filtered: List[Tuple[float, Any, str]] = []

    for score, action, target_norm in scored:
        if turn1_action_allowed_for_missing_goal_access(action, st, missing, going):
            filtered.append((score, action, target_norm))

    return filtered


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




# ---------------------------------------------------------------------
# TURN1_ACTUAL_START_CONTEXT_V13
# ---------------------------------------------------------------------

def turn1_unique_names_in_order(cards: Sequence[Dict[str, Any]]) -> List[str]:
    """
    Backward-compatible name, but now preserves multiplicity and order.

    Example:
      [Fighting Gong, Ultra Ball, Fighting Gong]
    becomes:
      ["Fighting Gong", "Ultra Ball", "Fighting Gong"]

    We intentionally do NOT dedupe anymore because the Turn 1 UI now needs
    to distinguish one copy from two copies in the actual simulated start.
    """
    out: List[str] = []

    for c in cards:
        if not isinstance(c, dict):
            continue

        name = tf.card_name(c)
        if not name:
            continue

        out.append(name)

    return out


def turn1_line_action_names(line: str) -> List[str]:
    raw = str(line or "").strip()

    if not raw or raw == "none":
        return []

    return [part.strip() for part in raw.split("->") if part.strip()]




# ---------------------------------------------------------------------
# TURN1_FORMAT_CARD_LIST_WITH_COUNTS_V21
# ---------------------------------------------------------------------

def turn1_format_card_list_with_counts(names: Sequence[str]) -> str:
    """
    Preserve first-seen order but compress duplicates.

    Example:
      ["Lunatone", "Ultra Ball", "Fighting Gong", "Fighting Gong"]
    -> "Lunatone, Ultra Ball, 2x Fighting Gong"
    """
    ordered = []
    counts = {}

    for name in names:
        name = str(name or "").strip()
        if not name:
            continue

        if name not in counts:
            ordered.append(name)
            counts[name] = 0

        counts[name] += 1

    parts = []
    for name in ordered:
        cnt = counts[name]
        if cnt <= 1:
            parts.append(name)
        else:
            parts.append(f"{cnt}x {name}")

    return ", ".join(parts) if parts else "—"


def turn1_relevant_start_draw_cards(
    natural_cards: Sequence[Dict[str, Any]],
    reqs: Sequence[GoalRequirement],
    line: str,
) -> List[str]:
    """
    Return the actual relevant cards that were naturally available before actions.

    This includes:
    - goal cards that were in the opening hand / active / natural draw
    - action cards from the line that were already naturally available

    IMPORTANT:
    We preserve multiplicity. If the natural start had two Fighting Gong,
    both copies are kept so the UI can show '2x Fighting Gong'.
    """
    natural_names = turn1_unique_names_in_order(natural_cards)
    natural_name_norms = [(name, tf.norm(name)) for name in natural_names]

    out: List[str] = []

    # 1. Actual goal pieces naturally available.
    for c in natural_cards:
        if not isinstance(c, dict):
            continue

        if any(
            any(card_matches_option(c, opt) for opt in req.options)
            for req in reqs
        ):
            name = tf.card_name(c)
            if name:
                out.append(name)

    # 2. Actual played cards that were naturally available.
    # Keep multiplicity up to what appears in the natural start.
    action_names = turn1_line_action_names(line)
    action_norms = [tf.norm(name) for name in action_names]

    used_indices = set()

    for action_norm in action_norms:
        if not action_norm:
            continue

        for idx, (natural_name, natural_norm) in enumerate(natural_name_norms):
            if idx in used_indices:
                continue

            if natural_norm == action_norm or natural_norm in action_norm or action_norm in natural_norm:
                # Only add if this exact natural instance wasn't already added via goal-piece logic.
                # We allow duplicates by instance position.
                out.append(natural_name)
                used_indices.add(idx)
                break

    return out


def turn1_start_draw_label(
    natural_cards: Sequence[Dict[str, Any]],
    reqs: Sequence[GoalRequirement],
    line: str,
) -> str:
    cards = turn1_relevant_start_draw_cards(natural_cards, reqs, line)
    return turn1_format_card_list_with_counts(cards)


def turn1_played_label(line: str) -> str:
    raw = str(line or "").strip()
    return raw if raw and raw != "none" else "—"




# ---------------------------------------------------------------------
# TURN1_EXCLUDE_PLAYED_PATHS_V15
# ---------------------------------------------------------------------

def turn1_parse_exclude_played(raw: str) -> List[str]:
    """
    Parse a comma/newline/pipe separated list of card names.

    Example:
      "Boss's Orders, Judge, Lillie's Determination"
    """
    if not raw:
        return []

    text = str(raw).replace("\n", ",").replace("|", ",")
    out: List[str] = []
    seen = set()

    for part in text.split(","):
        name = part.strip()

        if not name:
            continue

        # UI labels may look like: Boss's Orders [MEG 114]
        m = re.match(r"^(.*?)\s*\[[A-Za-z0-9]+\s+[A-Za-z0-9]+\]\s*$", name)
        if m:
            name = m.group(1).strip()

        normed = tf.norm(name)

        if normed and normed not in seen:
            seen.add(normed)
            out.append(name)

    return out


def turn1_action_names_from_line(line: str) -> List[str]:
    raw = str(line or "").strip()

    if not raw or raw == "none":
        return []

    return [p.strip() for p in raw.split("->") if p.strip()]


def turn1_excluded_played_matches(line: str, excluded_names: Sequence[str]) -> List[str]:
    """
    Return excluded card names that appear in a played action line.

    This checks played actions only, not the starting hand / natural draw.
    """
    actions = turn1_action_names_from_line(line)

    if not actions or not excluded_names:
        return []

    action_norms = [(a, tf.norm(a)) for a in actions]
    matches: List[str] = []
    seen = set()

    for excluded in excluded_names:
        ex_norm = tf.norm(excluded)

        if not ex_norm:
            continue

        for action_name, action_norm in action_norms:
            if not action_norm:
                continue

            if ex_norm == action_norm or ex_norm in action_norm or action_norm in ex_norm:
                if excluded not in seen:
                    seen.add(excluded)
                    matches.append(excluded)

    return matches


def turn1_apply_excluded_played_paths(results: List[Dict[str, Any]], raw_exclude_played: str) -> Dict[str, Any]:
    """
    Convert successful trials into failures when their played line used an excluded card.

    Natural successes remain valid because they did not play anything.
    """
    excluded_names = turn1_parse_exclude_played(raw_exclude_played)

    summary = {
        "enabled": bool(excluded_names),
        "excluded_card_names": excluded_names,
        "invalidated_successes": 0,
        "invalidated_by_card": {},
    }

    if not excluded_names:
        return summary

    invalidated_by_card = Counter()

    for r in results:
        if not r.get("success"):
            continue

        line = r.get("line") or "none"

        # Natural opening/draw successes do not play a line.
        if line == "none":
            continue

        matches = turn1_excluded_played_matches(line, excluded_names)

        if not matches:
            continue

        r["success"] = False
        r["success_stage"] = "excluded_path"
        r["excluded_by_played_cards"] = matches
        r["missing_requirements"] = [
            "Excluded path used: " + ", ".join(matches)
        ]

        summary["invalidated_successes"] += 1

        for m in matches:
            invalidated_by_card[m] += 1

    summary["invalidated_by_card"] = dict(invalidated_by_card)
    return summary




# ---------------------------------------------------------------------
# TURN1_OPPONENT_ONLY_TEXT_GUARD
# ---------------------------------------------------------------------
# General guard: opponent-only hand/deck disruption is not a consistency/search card.
# Examples this should block generically:
#   - your opponent shuffles their hand ... they draw cards
#   - look at your opponent's hand
#   - your opponent switches / discards / shuffles cards
# It should still allow real self-access effects:
#   - search your deck
#   - draw cards for yourself
#   - each player / both players draw effects


def turn1_opponent_only_filter_card_name(card: Any) -> str:
    try:
        return tf.card_name(card)
    except Exception:
        pass

    if isinstance(card, dict):
        ident = card.get("identity") or {}
        return str(card.get("name") or ident.get("name") or card.get("card_name") or "").strip()

    return str(getattr(card, "name", "") or "").strip()


def turn1_opponent_only_filter_flatten_text(obj: Any, limit: int = 30000) -> str:
    """Collect text from nested card dictionaries/lists without relying on one schema."""
    out: List[str] = []
    seen = set()

    def rec(x: Any) -> None:
        if sum(len(s) for s in out) > limit:
            return

        oid = id(x)
        if oid in seen:
            return

        if isinstance(x, (dict, list, tuple, set)):
            seen.add(oid)

        if isinstance(x, str):
            if x.strip():
                out.append(x)
            return

        if isinstance(x, dict):
            # Prefer effect/text/source fields, but fall back to all strings.
            preferred_keys = [
                "name", "card_name", "combined_text", "rules", "rules_text",
                "abilities_text", "attacks_text", "text", "effect_text", "raw_text",
                "source_text", "description", "effect", "effects", "compiled_effects",
                "steps", "op", "target", "owner", "zone", "source", "destination",
            ]
            for k in preferred_keys:
                if k in x:
                    rec(x.get(k))
            for k, v in x.items():
                if k not in preferred_keys:
                    rec(v)
            return

        if isinstance(x, (list, tuple, set)):
            for item in x:
                rec(item)
            return

    rec(obj)
    return " \n ".join(out)


def turn1_opponent_only_filter_norm_text(s: str) -> str:
    s = str(s or "").lower()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def turn1_text_has_self_access(t: str) -> bool:
    """True when the card text clearly gives us access/cards, not only the opponent."""
    if not t:
        return False

    # Symmetric draw effects can be real self-access. They are not opponent-only.
    if "each player" in t or "both players" in t:
        return True

    self_access_patterns = [
        r"search your deck",
        r"search deck",  # compiled text sometimes drops pronouns; allowed unless opponent-only text dominates
        r"from your deck",
        r"your deck for",
        r"put .* into your hand",
        r"put .* onto your bench",
        r"put .* on your bench",
        r"draw \d+ cards?",
        r"draw a card",
        r"draw cards until you",
        r"you draw",
        r"look at the top .* of your deck",
        r"reveal .* from your deck",
    ]

    return any(re.search(p, t) for p in self_access_patterns)


def turn1_text_has_opponent_only_disruption(t: str) -> bool:
    if not t:
        return False

    opponent_patterns = [
        r"your opponent[^.]*shuffl",
        r"your opponent[^.]*draw",
        r"your opponent[^.]*discard",
        r"your opponent[^.]*put",
        r"opponent's hand",
        r"opponent's deck",
        r"your opponent's hand",
        r"your opponent's deck",
        r"their hand",
        r"their deck",
        r"they draw",
        r"they shuffle",
        r"they put",
        r"they discard",
    ]

    return any(re.search(p, t) for p in opponent_patterns)


def turn1_compiled_step_is_opponent_only(card: Any) -> bool:
    """
    Schema-agnostic compiled-step guard.
    Blocks effects where draw/search/move-like operations target opponent/their zones.
    """
    if not isinstance(card, dict):
        return False

    compiled = card.get("compiled_effects") or card.get("effects") or []
    if not isinstance(compiled, list):
        return False

    saw_opponent_access = False
    saw_self_access = False

    def rec_step(x: Any) -> None:
        nonlocal saw_opponent_access, saw_self_access

        if isinstance(x, dict):
            blob = turn1_opponent_only_filter_norm_text(" ".join(str(v) for v in x.values() if not isinstance(v, (dict, list, tuple))))
            op = turn1_opponent_only_filter_norm_text(x.get("op") or x.get("operation") or x.get("type") or "")

            access_op = any(key in op for key in ["search", "draw", "move", "shuffle", "reveal", "look"])
            access_blob = any(key in blob for key in ["search", "draw", "hand", "deck", "shuffle"])

            if access_op or access_blob:
                if any(key in blob for key in ["opponent", "their hand", "their deck"]):
                    saw_opponent_access = True
                if any(key in blob for key in ["self", "player", "your deck", "your hand", "own", "my"]):
                    saw_self_access = True

            for v in x.values():
                if isinstance(v, (dict, list, tuple)):
                    rec_step(v)

        elif isinstance(x, (list, tuple)):
            for item in x:
                rec_step(item)

    rec_step(compiled)
    return saw_opponent_access and not saw_self_access


def turn1_card_is_opponent_only_access(card: Any) -> bool:
    """
    Generic block decision.

    Important: this is not a name blacklist. It uses card text/effect ownership.
    """
    blob = turn1_opponent_only_filter_norm_text(turn1_opponent_only_filter_flatten_text(card))

    # If compiled steps are explicitly opponent-only, block.
    if turn1_compiled_step_is_opponent_only(card):
        return True

    # If text has opponent-only disruption and no clear self-access, block.
    if turn1_text_has_opponent_only_disruption(blob):
        if not turn1_text_has_self_access(blob):
            return True

        # Strong opponent-only sentence forms. Ignore generic condition text like
        # "You may use this card only if..." and inspect the actual effect.
        opponent_effect_sentences = [
            sent.strip()
            for sent in re.split(r"[.;]", blob)
            if "opponent" in sent or "their hand" in sent or "their deck" in sent or sent.startswith("they ")
        ]
        if opponent_effect_sentences and not any(
            ("each player" in sent or "both players" in sent or "you draw" in sent or "search your deck" in sent or "from your deck" in sent)
            for sent in opponent_effect_sentences
        ):
            return True

    return False


# Guard fallback single-target scorer too. The multi-goal planner still delegates
# to target-finder scoring in some cases, so blocking only the executor is not enough.
try:
    _TURN1_ORIG_SCORE_PLAYABLE_CARD_BEFORE_OPPONENT_ONLY_GUARD = tf.score_playable_card

    def _turn1_score_playable_card_opponent_only_guard(card: Any, *args: Any, **kwargs: Any):
        if turn1_card_is_opponent_only_access(card):
            return 0
        return _TURN1_ORIG_SCORE_PLAYABLE_CARD_BEFORE_OPPONENT_ONLY_GUARD(card, *args, **kwargs)

    tf.score_playable_card = _turn1_score_playable_card_opponent_only_guard
except Exception:
    pass


def turn1_opponent_only_filter_action_names(line: str) -> List[str]:
    raw = str(line or "").strip()
    if not raw or raw == "none":
        return []
    return [p.strip() for p in raw.split("->") if p.strip()]


def turn1_build_opponent_only_blocked_name_map(deck: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    blocked: Dict[str, str] = {}

    for card in deck or []:
        if not isinstance(card, dict):
            continue

        if not turn1_card_is_opponent_only_access(card):
            continue

        name = turn1_opponent_only_filter_card_name(card)
        norm = tf.norm(name)
        if norm:
            blocked[norm] = name

    return blocked


def turn1_auto_block_opponent_only_played_cards(line: str, deck: Sequence[Dict[str, Any]]) -> List[str]:
    blocked_map = turn1_build_opponent_only_blocked_name_map(deck)
    if not blocked_map:
        return []

    out: List[str] = []
    seen = set()

    for action in turn1_opponent_only_filter_action_names(line):
        an = tf.norm(action)
        if not an:
            continue

        for blocked_norm, display in blocked_map.items():
            if an == blocked_norm or an in blocked_norm or blocked_norm in an:
                if display not in seen:
                    seen.add(display)
                    out.append(display)

    return out


def turn1_apply_opponent_only_filter(results: List[Dict[str, Any]], deck: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "enabled": True,
        "invalidated_successes": 0,
        "blocked_cards": {},
    }
    blocked_counts = Counter()

    for r in results:
        if not r.get("success"):
            continue

        line = r.get("line") or "none"
        if line == "none":
            continue

        blocked = turn1_auto_block_opponent_only_played_cards(line, deck)
        if not blocked:
            continue

        r["success"] = False
        r["success_stage"] = "blocked_opponent_only_access_action"
        r["blocked_opponent_only_access_cards"] = blocked
        r["missing_requirements"] = ["Blocked opponent-only access/disruption action: " + ", ".join(blocked)]
        summary["invalidated_successes"] += 1

        for b in blocked:
            blocked_counts[b] += 1

    summary["blocked_cards"] = dict(blocked_counts)
    return summary




# ---------------------------------------------------------------------
# TURN1_PRERETURN_INCOMPATIBLE_EFFECT_GUARD_V32
# ---------------------------------------------------------------------
# Hard guard for ability/effect labels that are not card names.
#
# Previous score-time filters can miss labels introduced by nested chain-search
# calls inside run_turn1_target_finder.py. This guard runs inside each trial
# immediately before the trial result is returned. If a successful line contains
# an effect label that only accesses an incompatible resource class, the trial is
# converted to a failure before any summaries / CSV rows are built.
#
# Example blocked for a Pokemon goal:
#   Shivery Chill -> searches Basic Water Energy only
#
# Example allowed:
#   Concealed Cards -> generic self draw
#   Buddy-Buddy Poffin -> Basic Pokemon access


def turn1_effect_label_actions_from_line(line):
    raw = str(line or "").strip()
    if not raw or raw == "none":
        return []
    return [p.strip() for p in raw.split("->") if p.strip()]


def turn1_effect_label_incompatibility_reason(action_label, deck, reqs):
    ok, reason = _turn1_effect_goal_compat_label_compatible_with_goal(action_label, deck, reqs)
    if ok:
        return None

    # If the effect-goal compatibility layer says this was not an effect label, do not block it here.
    if reason == "not_effect_label":
        return None

    return reason


# TURN1_EFFECT_GOAL_COMPAT_PATCHED_EFFECT_LABEL_COMPAT
def turn1_incompatible_effect_labels_in_line(line, deck, reqs):
    blocked = []
    for action in turn1_effect_label_actions_from_line(line):
        reason = turn1_effect_label_incompatibility_reason(action, deck, reqs)
        if reason:
            blocked.append({"action": action, "reason": reason})
    return blocked



# ---------------------------------------------------------------------
# TURN1_ACTIVE_COMPILED_SEARCH_RUNTIME_V41
# ---------------------------------------------------------------------
# Root fix for active-only compiled search abilities.
#
# The target-finder/goal-finder stack can score hand-played search cards, but
# active-only abilities like Chien-Pao ex's Shivery Chill need two extra pieces:
#   1. setup must choose the source Pokemon as Active when its compiled ability
#      can satisfy the current goal;
#   2. the goal loop must execute the Active Pokemon's compiled search effect
#      using the effect's actual filter, not as a generic/untyped action.
#
# This is generic. It does not special-case Chien-Pao by name. It looks for
# compiled/source ability text with an Active Spot condition and search_deck
# steps whose filters allow currently-missing goal cards.
#
# It does NOT assume free retreating. If the source Pokemon is not Active, this
# runtime does not use an Active-only ability.


def _turn1_v41_norm(value):
    import re as _re
    import unicodedata as _unicodedata

    s = str(value or "")
    s = _unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not _unicodedata.combining(ch))
    s = s.lower().replace("’", "'").replace("`", "'")
    s = _re.sub(r"[^a-z0-9{}]+", " ", s)
    return _re.sub(r"\s+", " ", s).strip()


def _turn1_v41_flatten_strings(obj, max_items=5000):
    out = []
    seen = set()

    def rec(x):
        if len(out) >= max_items:
            return

        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(x, str):
            if x.strip():
                out.append(x)
            return

        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k.strip():
                    out.append(k)
                rec(v)
            return

        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)

    rec(obj)
    return " ".join(out)


def _turn1_v41_card_name(card):
    try:
        return tf.card_name(card)
    except Exception:
        pass

    if isinstance(card, dict):
        ident = card.get("identity") or {}
        return (
            card.get("name")
            or card.get("card_name")
            or ident.get("name")
            or ident.get("canonical_name")
            or ""
        )

    return ""


def _turn1_v41_ability_name(effect):
    try:
        return tf.ability_name_from_effect(effect)
    except Exception:
        pass

    if isinstance(effect, dict):
        source = effect.get("source") or {}
        return (
            effect.get("name")
            or effect.get("ability_name")
            or source.get("name")
            or source.get("ability_name")
            or "Ability"
        )

    return "Ability"


def _turn1_v41_effect_blob(effect):
    parts = []

    for fn in [
        getattr(tf, "ability_text_blob", None),
        getattr(tf, "effect_text_blob", None),
    ]:
        if callable(fn):
            try:
                val = fn(effect)
                if val:
                    parts.append(str(val))
            except Exception:
                pass

    parts.append(_turn1_v41_flatten_strings(effect))
    return " ".join(parts)


def _turn1_v41_filter_blob(filt):
    try:
        return tf.filter_text_blob(filt)
    except Exception:
        return _turn1_v41_flatten_strings(filt)


def _turn1_v41_iter_steps(effect):
    try:
        return list(tf.iter_steps(effect))
    except Exception:
        if isinstance(effect, dict):
            steps = effect.get("steps") or effect.get("compiled_steps") or []
            return list(steps) if isinstance(steps, list) else []
        return []


def _turn1_v41_extract_filter(step):
    try:
        return tf.extract_filter(step)
    except Exception:
        if isinstance(step, dict):
            return step.get("filter") or step.get("filters") or step.get("target_filter") or {}
        return {}


def _turn1_v41_search_amount(step):
    try:
        return int(tf.search_amount(step))
    except Exception:
        pass

    if isinstance(step, dict):
        for key in ["amount", "count", "max", "quantity"]:
            val = step.get(key)
            if isinstance(val, int):
                return val
            if isinstance(val, dict):
                for subkey in ["value", "base"]:
                    if isinstance(val.get(subkey), int):
                        return int(val[subkey])
            if isinstance(val, str):
                m = re.search(r"\d+", val)
                if m:
                    return int(m.group(0))
    return 1


def _turn1_v41_effect_requires_active(effect):
    blob = _turn1_v41_norm(_turn1_v41_effect_blob(effect))
    return (
        "active spot" in blob
        or "active pokemon" in blob
        or "active pokémon" in blob
        or "this pokemon is active" in blob
        or "this pokémon is active" in blob
    )


def _turn1_v41_effect_has_search_deck(effect):
    return any(isinstance(step, dict) and step.get("op") == "search_deck" for step in _turn1_v41_iter_steps(effect))


def _turn1_v41_filter_allows_card(filt, card):
    try:
        return bool(tf.filter_allows_card(filt, card))
    except Exception:
        return False


def _turn1_v41_card_matches_req(card, req):
    try:
        return any(card_matches_option(card, opt) for opt in req.options)
    except Exception:
        return False


def _turn1_v41_req_current_count(req, st, tracker):
    try:
        pool = zone_cards(st, tracker, req.zone)
        return sum(1 for c in pool if _turn1_v41_card_matches_req(c, req))
    except Exception:
        return 0


def _turn1_v41_req_needed_count(req):
    try:
        return max(1, int(getattr(req, "min_count", 1) or 1))
    except Exception:
        return 1


def _turn1_v41_missing_reqs_from_state(reqs, mode, st, tracker):
    missing = []
    for req in reqs:
        if _turn1_v41_req_current_count(req, st, tracker) < _turn1_v41_req_needed_count(req):
            missing.append(req)
    if mode == "any" and len(missing) < len(reqs):
        return []
    return missing


def _turn1_v41_effect_can_search_target_norm(effect, target_norm):
    if not target_norm or not _turn1_v41_effect_has_search_deck(effect):
        return False

    # First use the compiled filter, if present.
    for step in _turn1_v41_iter_steps(effect):
        if not isinstance(step, dict) or step.get("op") != "search_deck":
            continue
        filt = _turn1_v41_extract_filter(step)
        blob = _turn1_v41_norm(_turn1_v41_filter_blob(filt) + " " + _turn1_v41_effect_blob(effect))
        target = _turn1_v41_norm(target_norm)

        if target and target in blob:
            return True

        # Class-level compatibility for common goal names.
        if "energy" in target:
            if "energy" not in blob:
                return False
            if "water" in target and "water" not in blob and "{w}" not in blob:
                return False
            if "basic" in target and "basic" not in blob:
                return False
            return True

        if "pokemon" in target or "pokemon" in blob or "pokémon" in blob:
            if "energy" in blob and "pokemon" not in blob and "pokémon" not in blob:
                return False
            return "pokemon" in blob or "pokémon" in blob or "bench" in blob

    return False


_ORIG_TURN1_V41_CHOOSE_OPTIMAL_ACTIVE = tf.choose_optimal_active


def _turn1_v41_choose_optimal_active(opening, target_norm):
    """Choose a legal opener Active. Prefer a Basic whose Active-only compiled
    search ability can satisfy the current target class.
    """
    try:
        basics = [c for c in opening if tf.is_basic_pokemon(c)]
    except Exception:
        basics = []

    for card in basics:
        try:
            effects = list(tf.ability_effects(card))
        except Exception:
            effects = []

        for eff in effects:
            if not _turn1_v41_effect_requires_active(eff):
                continue
            if _turn1_v41_effect_can_search_target_norm(eff, target_norm):
                return card

    return _ORIG_TURN1_V41_CHOOSE_OPTIMAL_ACTIVE(opening, target_norm)


tf.choose_optimal_active = _turn1_v41_choose_optimal_active


def _turn1_v41_select_missing_goal_cards_from_deck(st, reqs, mode, tracker, filt, amount):
    selected = []
    seen_ids = set()

    missing = _turn1_v41_missing_reqs_from_state(reqs, mode, st, tracker)
    if not missing:
        return selected

    for req in missing:
        current = _turn1_v41_req_current_count(req, st, tracker)
        needed = _turn1_v41_req_needed_count(req)
        deficit = max(0, needed - current)

        while deficit > 0 and len(selected) < amount:
            chosen_idx = None
            for i, c in enumerate(st.deck):
                cid = id(c)
                if cid in seen_ids:
                    continue
                if not _turn1_v41_card_matches_req(c, req):
                    continue
                if not _turn1_v41_filter_allows_card(filt, c):
                    continue
                chosen_idx = i
                break

            if chosen_idx is None:
                break

            card = st.deck.pop(chosen_idx)
            selected.append(card)
            seen_ids.add(id(card))
            deficit -= 1

    return selected


def turn1_v41_execute_active_compiled_search_if_useful(st, reqs, mode, tracker, rng, going):
    source = getattr(st, "active", None)
    if source is None:
        return False

    try:
        effects = list(tf.ability_effects(source))
    except Exception:
        effects = []

    if not effects:
        return False

    for eff in effects:
        if not _turn1_v41_effect_requires_active(eff):
            continue
        if not _turn1_v41_effect_has_search_deck(eff):
            continue

        try:
            key = tf.ability_usage_key(source, eff)
            if key in st.abilities_used:
                continue
        except Exception:
            key = f"v41:{_turn1_v41_card_name(source)}:{_turn1_v41_ability_name(eff)}"
            if key in getattr(st, "abilities_used", set()):
                continue

        # Active-only requirement is already satisfied by source == st.active.
        total_selected = []

        for step in _turn1_v41_iter_steps(eff):
            if not isinstance(step, dict) or step.get("op") != "search_deck":
                continue

            filt = _turn1_v41_extract_filter(step)
            amount = max(1, _turn1_v41_search_amount(step))
            selected = _turn1_v41_select_missing_goal_cards_from_deck(
                st,
                reqs,
                mode,
                tracker,
                filt,
                amount,
            )
            total_selected.extend(selected)

        if not total_selected:
            continue

        ability_name = _turn1_v41_ability_name(eff)
        st.hand.extend(total_selected)
        try:
            st.abilities_used.add(key)
        except Exception:
            pass
        st.actions_used += 1
        st.line.append(ability_name)
        try:
            rng.shuffle(st.deck)
        except Exception:
            pass
        try:
            st.log.append({
                "event": "active_compiled_search_selected_v41",
                "source": _turn1_v41_card_name(source),
                "ability": ability_name,
                "selected": [_turn1_v41_card_name(c) for c in total_selected],
                "reason": "Active-only compiled search ability legally matched missing goal requirements.",
            })
        except Exception:
            pass
        return True

    return False


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
    st._turn1_goal_reqs = list(reqs)
    st._turn1_goal_mode = mode
    tracker = GoalTracker()
    st._turn1_goal_tracker = tracker
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

    # Actual cards naturally available before any action line starts.
    # This includes the chosen Active Pokémon, cards remaining in hand after setup,
    # and the natural draw if one happened.
    natural_start_draw_cards = list(st.hand)
    if st.active is not None:
        natural_start_draw_cards.append(st.active)

    # TURN1_ENSURE_NATURAL_START_DRAW_CARDS_V14
    # Actual cards naturally available before any action line starts.
    # This includes the chosen Active Pokémon, cards remaining in hand after setup,
    # and the natural draw if one happened.
    natural_start_draw_cards = list(st.hand)
    if st.active is not None:
        natural_start_draw_cards.append(st.active)

    success_stage = natural_success_stage

    while success_stage is None and st.actions_used < max_actions:
        missing = missing_requirements(reqs, mode, st, tracker)
        if not missing:
            success_stage = "after_actions"
            break
        # TURN1_APPLY_ACTIVE_COMPILED_SEARCH_RUNTIME_V41
        # Before falling back to hand-played search cards, try a legal Active-only
        # compiled search ability against the whole missing goal package.
        if turn1_v41_execute_active_compiled_search_if_useful(st, reqs, mode, tracker, rng, going):
            snapshot_accessed(tracker, st)
            if goal_satisfied(reqs, mode, st, tracker):
                success_stage = "after_actions"
                break
            try:
                st.found = False
                st.found_stage = None
            except Exception:
                pass
            continue

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
    # TURN1_APPLY_PRERETURN_INCOMPATIBLE_EFFECT_GUARD_V32
    turn1_blocked_effect_labels = []
    if success_stage is not None and line != "none":
        turn1_blocked_effect_labels = turn1_incompatible_effect_labels_in_line(line, deck, reqs)
        if turn1_blocked_effect_labels:
            success_stage = None

    return {
        "success": success_stage is not None,
        "success_stage": success_stage or "not_met",
        "line": line,
        "starting_hand_draw": locals().get("starting_hand_draw", turn1_start_draw_label(locals().get("natural_start_draw_cards", []), reqs, locals().get("line", ""))),
        "played": locals().get("played", turn1_played_label(locals().get("line", ""))),
        "mulligans": mulligans,
        "actions_used": st.actions_used,
        "active": tf.card_name(active) if active else None,
        "final_hand_size": len(st.hand),
        "final_deck_size": len(st.deck),
        "missing_requirements": (["Blocked incompatible effect label: " + ", ".join(b.get("action", "?") for b in turn1_blocked_effect_labels)] if locals().get("turn1_blocked_effect_labels") else [r.label for r in final_missing]),
        "missing_count": (1 if locals().get("turn1_blocked_effect_labels") else len(final_missing)),
        "goal_pieces_prized": prize_status,
        "accessed_goal_piece_names": sorted(set(
            tf.card_name(c)
            for c in tracker.accessed
            if any(any(card_matches_option(c, opt) for opt in req.options) for req in reqs)
        )),
        "log": st.log,
    }


def turn1_direct_goal_filter_is_blocked_missing_requirement(value: Any) -> bool:
    # Return True for diagnostic block labels that should not be counted as unmet goals.
    s = str(value or "").lower()
    markers = [
        "blocked incompatible search target",
        "blocked opponent-dependent access",
        "blocked repeat active compiled",
        "blocked repeated active compiled",
        "blocked_opponent_dependent",
        "blocked_search_target_type",
        "blocked_incompatible_search_filter",
    ]
    return any(m in s for m in markers)


def summarize_goal_trials(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    # TURN1_DEWRAP_DIRECT_GOAL_FILTER
    n = len(results)
    successes = sum(1 for r in results if r["success"])
    by_stage = Counter(r["success_stage"] for r in results)
    lines = Counter(r["line"] for r in results if r["success"] and r["line"] != "none")
    line_contexts = Counter(
        (
            r.get("starting_hand_draw") or "—",
            r.get("played") or r.get("line") or "—",
            r.get("line") or "none",
        )
        for r in results
        if r["success"] and r["line"] != "none"
    )
    missing = Counter()
    for r in results:
        if not r["success"]:
            if r.get("missing_requirements"):
                for m in r["missing_requirements"]:
                    if turn1_direct_goal_filter_is_blocked_missing_requirement(m):
                        continue
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
        "top_success_line_contexts": [
            {
                "starting_hand_draw": k[0],
                "played": k[1],
                "line": k[2],
                "count": v,
                "percent": pct(v / n),
            }
            for k, v in line_contexts.most_common(25)
        ],
        "top_missing_requirements_on_failure": [
            {
                "requirement": k,
                "count": v,
                "percent_of_trials": pct(v / n),
                "percent_of_failures": pct(v / max(1, n - successes)),
            }
            for k, v in missing.most_common(25)
        ],
        "any_required_piece_all_prized": {"trials": any_all_prized, "percent": pct(any_all_prized / n) if n else 0.0},
        "mulligans": [{"mulligans": k, "count": v, "percent": pct(v / n)} for k, v in sorted(mulligans.items())],
    }





# ---------------------------------------------------------------------
# TURN1_EFFECT_NAME_COMPATIBILITY_GUARD_V29
# ---------------------------------------------------------------------
# Previous guards checked played card names. However the line can also contain
# ability/effect names, for example:
#   Ultra Ball -> Shivery Chill
#   Tatsugiri -> Attract Customers
#
# Those are not card names, so card-name filters miss them.
#
# This guard finds ability/effect labels in the played line, looks up the owning
# deck card text, classifies what that effect can access, and blocks the line if
# the effect's access class cannot satisfy the current goal.
#
# Examples:
#   Shivery Chill      -> Basic Water Energy only -> invalid for Pokémon goals
#   Attract Customers  -> Supporter only          -> invalid for Pokémon goals
#   Buddy-Buddy Poffin -> Basic Pokémon           -> valid for Pokémon goals


def turn1_effect_name_compat_norm_text(s):
    import re as _re
    import unicodedata as _unicodedata

    s = str(s or "")
    s = _unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not _unicodedata.combining(ch))
    s = s.lower()
    s = s.replace("’", "'").replace("`", "'")
    s = _re.sub(r"\s+", " ", s)
    return s.strip()


def turn1_effect_name_compat_flatten_strings(obj, max_items=5000):
    out = []
    seen = set()

    def rec(x):
        if len(out) >= max_items:
            return

        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(x, str):
            if x.strip():
                out.append(x)
            return

        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k.strip():
                    out.append(k)
                rec(v)
            return

        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)
            return

    rec(obj)
    return " ".join(out)


def turn1_effect_name_compat_card_name(card):
    try:
        return tf.card_name(card)
    except Exception:
        if isinstance(card, dict):
            ident = card.get("identity") or {}
            return (
                card.get("name")
                or card.get("card_name")
                or ident.get("name")
                or ident.get("canonical_name")
                or ""
            )
        return ""


def turn1_effect_name_compat_card_supertype(card):
    if not isinstance(card, dict):
        return ""

    ident = card.get("identity") or {}
    return str(
        card.get("supertype")
        or ident.get("supertype")
        or card.get("card_type")
        or ""
    )


def turn1_effect_name_compat_card_subtype_blob(card):
    if not isinstance(card, dict):
        return ""

    ident = card.get("identity") or {}
    parts = []

    for source in [card, ident]:
        for key in ["subtype", "subtypes", "trainerType", "types"]:
            val = source.get(key)
            if isinstance(val, list):
                parts.extend(str(x) for x in val)
            elif val:
                parts.append(str(val))

    return " ".join(parts)


def turn1_effect_name_compat_classes_for_goal_card(card):
    blob = turn1_effect_name_compat_norm_text(
        " ".join(
            [
                turn1_effect_name_compat_card_name(card),
                turn1_effect_name_compat_card_supertype(card),
                turn1_effect_name_compat_card_subtype_blob(card),
                turn1_effect_name_compat_flatten_strings(card),
            ]
        )
    )

    classes = set()

    if "pokemon" in blob or "pokémon" in blob:
        classes.add("pokemon")
        if "basic" in blob:
            classes.add("basic_pokemon")

    if "energy" in blob:
        classes.add("energy")
        if "basic" in blob:
            classes.add("basic_energy")
        if "water" in blob:
            classes.add("water_energy")
            if "basic" in blob:
                classes.add("basic_water_energy")
        if "fighting" in blob:
            classes.add("fighting_energy")

    if "trainer" in blob:
        classes.add("trainer")
    if "supporter" in blob:
        classes.add("supporter")
        classes.add("trainer")
    if "item" in blob:
        classes.add("item")
        classes.add("trainer")
    if "stadium" in blob:
        classes.add("stadium")
        classes.add("trainer")

    return classes


def turn1_effect_name_compat_goal_access_classes(reqs, deck):
    classes = set()

    for req in reqs:
        for card in deck:
            try:
                if any(card_matches_option(card, opt) for opt in req.options):
                    classes.update(turn1_effect_name_compat_classes_for_goal_card(card))
            except Exception:
                continue

    return classes


def turn1_effect_name_compat_access_target_classes_from_text(text):
    """
    Classify what an effect can access.

    This intentionally looks at target phrases, not the source card type.
    So Chien-Pao's text mentioning "this Pokémon is in the Active Spot" does
    not make Shivery Chill a Pokémon search.
    """
    import re as _re

    t = turn1_effect_name_compat_norm_text(text)
    classes = set()

    # Supporter-only access, e.g. Tatsugiri's Attract Customers.
    if "supporter" in t:
        classes.add("supporter")
        classes.add("trainer")

    # Energy-only access, e.g. Shivery Chill.
    if "basic water energy" in t:
        classes.update(["energy", "basic_energy", "water_energy", "basic_water_energy"])
    elif "basic energy" in t:
        classes.update(["energy", "basic_energy"])
    elif _re.search(r"(search|reveal|choose|put).{0,160}energy", t):
        classes.add("energy")

    # Pokémon access. Keep this tied to search/reveal/put phrases so that
    # "if this Pokémon is in the Active Spot" does not count.
    if _re.search(r"(search|reveal|choose|put|bench).{0,180}(basic pokemon|basic pokémon|basic pokémon)", t):
        classes.update(["pokemon", "basic_pokemon"])
    elif _re.search(r"(search|reveal|choose|put|bench).{0,180}(pokemon|pokémon|pokémon)", t):
        classes.add("pokemon")
    elif "put onto your bench" in t or "put it onto your bench" in t:
        classes.add("pokemon")

    # Trainer/item/stadium access.
    if _re.search(r"(search|reveal|choose|put).{0,160}item", t):
        classes.update(["trainer", "item"])
    if _re.search(r"(search|reveal|choose|put).{0,160}stadium", t):
        classes.update(["trainer", "stadium"])

    # Generic draw is not restricted by class; leave empty so it is not blocked
    # by this compatibility guard.
    return classes


def turn1_effect_name_compat_deck_card_name_norms(deck):
    out = set()
    for card in deck:
        name = turn1_effect_name_compat_card_name(card)
        if name:
            out.add(turn1_effect_name_compat_norm_text(name))
    return out


def turn1_effect_name_compat_effect_texts_for_action_label(action_label, deck):
    """
    Return possible owning card/effect text blobs for an action label.

    This is meant for labels like:
      Shivery Chill
      Attract Customers

    It skips normal card names. Card-name actions are handled by the normal
    card filters and executor.
    """
    action_norm = turn1_effect_name_compat_norm_text(action_label)

    if not action_norm:
        return []

    card_name_norms = turn1_effect_name_compat_deck_card_name_norms(deck)

    # If the action is literally a card name, do not treat it as an ability label.
    if action_norm in card_name_norms:
        return []

    matches = []

    for card in deck:
        blob = turn1_effect_name_compat_flatten_strings(card)
        blob_norm = turn1_effect_name_compat_norm_text(blob)

        if action_norm and action_norm in blob_norm:
            matches.append(blob)

    return matches


def turn1_effect_name_compat_action_labels(line):
    raw = str(line or "").strip()
    if not raw or raw == "none":
        return []

    return [p.strip() for p in raw.split("->") if p.strip()]


def turn1_effect_name_compat_incompatible_actions(line, deck, reqs):
    goal_classes = turn1_effect_name_compat_goal_access_classes(reqs, deck)

    if not goal_classes:
        return []

    blocked = []

    for action in turn1_effect_name_compat_action_labels(line):
        effect_texts = turn1_effect_name_compat_effect_texts_for_action_label(action, deck)

        if not effect_texts:
            continue

        combined_classes = set()
        for txt in effect_texts:
            combined_classes.update(turn1_effect_name_compat_access_target_classes_from_text(txt))

        # If we could not classify target classes, do not block.
        if not combined_classes:
            continue

        # Broad compatibility.
        # If an effect can only access Energy, it cannot satisfy a Pokémon goal.
        # If an effect can only access Supporters, it cannot satisfy a Pokémon goal.
        compatible = bool(combined_classes.intersection(goal_classes))

        # Basic Pokémon can satisfy Pokémon goals.
        if "basic_pokemon" in combined_classes and "pokemon" in goal_classes:
            compatible = True

        # Basic Energy / typed Energy can satisfy Energy goals.
        if (
            {"basic_energy", "basic_water_energy", "water_energy", "fighting_energy"}.intersection(combined_classes)
            and "energy" in goal_classes
        ):
            compatible = True

        if not compatible:
            blocked.append(
                {
                    "action": action,
                    "access_classes": sorted(combined_classes),
                    "goal_classes": sorted(goal_classes),
                }
            )

    return blocked


def turn1_apply_effect_name_compatibility_filter(results, deck, reqs):
    summary = {
        "enabled": True,
        "invalidated_successes": 0,
        "invalidated_by_action": {},
    }

    by_action = {}

    for r in results:
        if not r.get("success"):
            continue

        line = r.get("line") or "none"
        blocked = turn1_effect_name_compat_incompatible_actions(line, deck, reqs)

        if not blocked:
            continue

        actions = [b["action"] for b in blocked]

        r["success"] = False
        r["success_stage"] = "incompatible_effect_name_access"
        r["blocked_incompatible_effect_actions_v29"] = blocked
        r["missing_requirements"] = [
            "Incompatible effect access: " + ", ".join(actions)
        ]

        summary["invalidated_successes"] += 1

        for action in actions:
            by_action[action] = by_action.get(action, 0) + 1

    summary["invalidated_by_action"] = by_action
    return summary




# ---------------------------------------------------------------------
# TURN1_PRE_SUMMARY_EFFECT_LABEL_GUARD_V30
# ---------------------------------------------------------------------
# Conservative pre-summary guard for ability/effect labels.
#
# A line may contain card names:
#   Ultra Ball
#   Nest Ball
#
# or ability/effect names:
#   Shivery Chill
#   Attract Customers
#   Concealed Cards
#
# Card-name actions go through normal card logic.
# Effect-name actions must prove that their effect can help the current goal.
# This blocks cases like:
#   Shivery Chill in a Pokémon goal
# because Shivery Chill searches Basic Water Energy, not Pokémon.

def turn1_pre_summary_effect_label_is_compatible(action_label, deck, reqs):
    ok, reason = _turn1_effect_goal_compat_label_compatible_with_goal(action_label, deck, reqs)
    return ok, reason


# TURN1_EFFECT_GOAL_COMPAT_PATCHED_PRE_SUMMARY_EFFECT_LABEL_COMPAT
def turn1_pre_summary_effect_label_actions_from_line(line):
    raw = str(line or "").strip()

    if not raw or raw == "none":
        return []

    return [p.strip() for p in raw.split("->") if p.strip()]


def turn1_apply_pre_summary_effect_label_compatibility_guard(results, deck, reqs):
    """
    Mutate results in-place before summarize_goal_trials runs.
    """
    invalidated = 0

    for r in results:
        if not r.get("success"):
            continue

        line = r.get("line") or "none"

        if line == "none":
            continue

        blocked = []

        for action in turn1_pre_summary_effect_label_actions_from_line(line):
            ok, reason = turn1_pre_summary_effect_label_is_compatible(action, deck, reqs)

            if not ok:
                blocked.append({"action": action, "reason": reason})

        if not blocked:
            continue

        actions = [b["action"] for b in blocked]

        r["success"] = False
        r["success_stage"] = "blocked_incompatible_effect_label_v30"
        r["blocked_incompatible_effect_labels_v30"] = blocked
        r["missing_requirements"] = [
            "Blocked incompatible effect label: " + ", ".join(actions)
        ]

        invalidated += 1

    return {
        "enabled": True,
        "invalidated_successes": invalidated,
    }




# ---------------------------------------------------------------------
# TURN1_SCORE_TIME_ACCESS_COMPAT_FILTER
# ---------------------------------------------------------------------
# The previous Shivery Chill / Attract Customers guards were mostly post-hoc:
# they tried to invalidate successful rows after the single-target engine had
# already chosen a bad ability. This guard fixes the root at candidate scoring
# time by removing incompatible ability/effect actions before the greedy planner
# can choose them.
#
# Example blocked for a Pokémon goal:
#   Shivery Chill      -> searches Basic Water Energy
#   Attract Customers  -> searches a Supporter
#
# Example allowed:
#   Concealed Cards    -> generic self draw
#   Buddy-Buddy Poffin -> searches Basic Pokémon

_ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_BEFORE_SCORE_ACCESS_COMPAT = score_candidate_for_missing_targets


def turn1_score_access_compat_norm(s):
    import re as _re
    import unicodedata as _unicodedata

    s = str(s or "")
    s = _unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not _unicodedata.combining(ch))
    s = s.lower().replace("’", "'").replace("`", "'")
    s = _re.sub(r"\s+", " ", s)
    return s.strip()


def turn1_score_access_compat_flatten_strings(obj, max_items=5000):
    out = []
    seen = set()

    def rec(x):
        if len(out) >= max_items:
            return

        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(x, str):
            if x.strip():
                out.append(x)
            return

        if isinstance(x, dict):
            for k, v in x.items():
                # Include string keys because effect schemas sometimes encode
                # useful terms like ability names / operation names as keys.
                if isinstance(k, str) and k.strip():
                    out.append(k)
                rec(v)
            return

        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)
            return

    rec(obj)
    return " ".join(out)


def turn1_score_access_compat_card_identity_blob(card):
    if not isinstance(card, dict):
        return ""

    ident = card.get("identity") or {}
    parts = []

    for source in (card, ident):
        for key in ["name", "canonical_name", "supertype", "subtype", "subtypes", "types", "rules"]:
            val = source.get(key)
            if isinstance(val, list):
                parts.extend(str(x) for x in val)
            elif val:
                parts.append(str(val))

    try:
        name = tf.card_name(card)
        if name:
            parts.append(name)
    except Exception:
        pass

    return " ".join(parts)


def turn1_score_access_compat_classes_for_goal_card(card):
    blob = turn1_score_access_compat_norm(turn1_score_access_compat_card_identity_blob(card))
    classes = set()

    if "pokemon" in blob or "pokémon" in blob:
        classes.add("pokemon")
        if "basic" in blob:
            classes.add("basic_pokemon")

    if "energy" in blob:
        classes.add("energy")
        if "basic" in blob:
            classes.add("basic_energy")
        if "water" in blob:
            classes.update(["water_energy", "basic_water_energy"])
        if "fighting" in blob:
            classes.add("fighting_energy")
        if "grass" in blob:
            classes.add("grass_energy")
        if "lightning" in blob or "electric" in blob:
            classes.add("lightning_energy")
        if "fire" in blob:
            classes.add("fire_energy")
        if "psychic" in blob:
            classes.add("psychic_energy")
        if "darkness" in blob or "dark" in blob:
            classes.add("darkness_energy")
        if "metal" in blob:
            classes.add("metal_energy")

    if "trainer" in blob:
        classes.add("trainer")
    if "supporter" in blob:
        classes.update(["trainer", "supporter"])
    if "item" in blob:
        classes.update(["trainer", "item"])
    if "stadium" in blob:
        classes.update(["trainer", "stadium"])
    if "tool" in blob:
        classes.update(["trainer", "tool"])

    return classes


def turn1_score_access_compat_all_known_cards_from_state(st):
    cards = []
    for attr in ["deck", "hand", "discard", "prizes", "bench"]:
        try:
            val = getattr(st, attr, []) or []
            cards.extend([c for c in val if isinstance(c, dict)])
        except Exception:
            pass
    try:
        if isinstance(st.active, dict):
            cards.append(st.active)
    except Exception:
        pass
    return cards


def turn1_score_access_compat_goal_classes(missing, st):
    classes = set()
    cards = turn1_score_access_compat_all_known_cards_from_state(st)

    for req in missing or []:
        for c in cards:
            try:
                if any(card_matches_option(c, opt) for opt in req.options):
                    classes.update(turn1_score_access_compat_classes_for_goal_card(c))
            except Exception:
                continue

    return classes


def turn1_score_access_compat_effect_text_for_action(action):
    """Return the tightest effect/action text available for a candidate action."""
    if isinstance(action, dict) and action.get("_virtual_action"):
        chunks = []

        # Prefer the effect object first. This avoids classifying Chien-Pao as
        # a Pokémon search just because the source card's identity says Pokémon.
        for key in ["effect", "compiled_effect", "ability", "attack", "step"]:
            if key in action:
                chunks.append(turn1_score_access_compat_flatten_strings(action.get(key)))

        # Some actions store the useful text inside these fields.
        for key in ["source_effect", "search_effect", "requirement", "raw_effect"]:
            if key in action:
                chunks.append(turn1_score_access_compat_flatten_strings(action.get(key)))

        text = " ".join(x for x in chunks if x.strip())

        if text.strip():
            return text

        # Fallback to source/card only when no effect text exists.
        for key in ["source", "card", "search_card"]:
            if key in action:
                return turn1_score_access_compat_flatten_strings(action.get(key))

        return turn1_score_access_compat_flatten_strings(action)

    if isinstance(action, dict):
        return turn1_score_access_compat_flatten_strings(action)

    return str(action or "")


def turn1_score_access_compat_access_classes_from_effect_text(text):
    import re as _re

    t = turn1_score_access_compat_norm(text)
    classes = set()

    if not t:
        return classes

    opponent_only = any(
        marker in t
        for marker in [
            "your opponent shuffles",
            "your opponent draws",
            "your opponent reveals",
            "your opponent searches",
            "opponent's hand",
            "opponents hand",
            "their hand",
            "their deck",
        ]
    )
    self_access = any(
        marker in t
        for marker in [
            "your deck",
            "your hand",
            "put into your hand",
            "put onto your bench",
            "put it onto your bench",
            "draw a card",
            "draw cards",
            "you may draw",
        ]
    )

    if opponent_only and not self_access:
        classes.add("opponent_only")
        return classes

    # Restricted searches. These are target classes, not source-card classes.
    if "basic water energy" in t or "basic {w} energy" in t:
        classes.update(["energy", "basic_energy", "water_energy", "basic_water_energy"])
    elif "basic energy" in t:
        classes.update(["energy", "basic_energy"])
    elif _re.search(r"(search|reveal|choose|put).{0,180}energy", t):
        classes.add("energy")

    if _re.search(r"(search|look at|reveal|choose|put).{0,220}supporter", t):
        classes.update(["trainer", "supporter"])

    if _re.search(r"(search|look at|reveal|choose|put).{0,220}item", t):
        classes.update(["trainer", "item"])

    if _re.search(r"(search|look at|reveal|choose|put).{0,220}stadium", t):
        classes.update(["trainer", "stadium"])

    # Pokémon searches. Tie to access verbs; do not classify on source identity.
    if _re.search(r"(search|look at|reveal|choose|put|bench).{0,220}(basic pokemon|basic pokémon)", t):
        classes.update(["pokemon", "basic_pokemon"])
    elif _re.search(r"(search|look at|reveal|choose|put|bench).{0,220}(pokemon|pokémon)", t):
        classes.add("pokemon")
    elif "put onto your bench" in t or "put it onto your bench" in t:
        classes.add("pokemon")

    # Generic self draw is compatible with any goal because it can draw into
    # any card type. This must not save restricted searches like Shivery Chill.
    if _re.search(r"\bdraw\b.{0,120}\b(card|cards)\b", t) and "your opponent" not in t:
        classes.add("generic_draw")

    return classes


def turn1_score_access_compat_action_is_compatible(action, missing, st):
    # Only filter actions where we can classify the effect target class.
    text = turn1_score_access_compat_effect_text_for_action(action)
    access_classes = turn1_score_access_compat_access_classes_from_effect_text(text)

    if not access_classes:
        return True

    goal_classes = turn1_score_access_compat_goal_classes(missing, st)

    # If the goal is unclassified, do not block; keep the simulator usable.
    if not goal_classes:
        return True

    if "opponent_only" in access_classes:
        return False

    if "generic_draw" in access_classes:
        return True

    if access_classes.intersection(goal_classes):
        return True

    if "basic_pokemon" in access_classes and "pokemon" in goal_classes:
        return True

    if (
        {"basic_energy", "water_energy", "basic_water_energy", "fighting_energy", "grass_energy", "lightning_energy", "fire_energy", "psychic_energy", "darkness_energy", "metal_energy"}.intersection(access_classes)
        and "energy" in goal_classes
    ):
        return True

    try:
        label = action_label(action)
    except Exception:
        label = str(action)

    try:
        st.log.append(
            {
                "event": "blocked_score_time_incompatible_access",
                "action": label,
                "access_classes": sorted(access_classes),
                "goal_classes": sorted(goal_classes),
            }
        )
    except Exception:
        pass

    return False


def score_candidate_for_missing_targets(
    st: tf.SimState,
    missing: Sequence[GoalRequirement],
    going: str,
    enable_chain_search: bool,
) -> List[Tuple[float, Any, str]]:
    # TURN1_ACTION_BUDGET_GUARD
    if _turn1_action_budget_exhausted(st):
        return []
    scored = _ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_BEFORE_SCORE_ACCESS_COMPAT(
        st,
        missing,
        going,
        enable_chain_search,
    )

    filtered = []
    for score, action, target_norm in scored:
        if turn1_score_access_compat_action_is_compatible(action, missing, st):
            filtered.append((score, action, target_norm))

    return filtered




# ---------------------------------------------------------------------
# TURN1_BLOCK_UNVALIDATED_EFFECT_ACTIONS
# ---------------------------------------------------------------------
# Root fix:
# The Turn-1 goal finder sometimes appends ability/effect labels into the
# played line:
#
#   Ultra Ball -> Shivery Chill
#   Tatsugiri -> Attract Customers
#   Radiant Greninja -> Concealed Cards
#
# These labels are not physical cards being played from hand. Until an ability
# has an exact validated handler with source/precondition/target checks, it
# must not be credited as a valid access action.
#
# This wrapper converts successful trials into failures if the played line
# contains an action segment that is not an actual card name in the deck.

def _turn1_unvalidated_effect_guard_norm_name(value):
    import re as _re
    import unicodedata as _unicodedata

    s = str(value or "")
    s = _unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not _unicodedata.combining(ch))
    s = s.lower()
    s = s.replace("’", "'").replace("`", "'")
    s = _re.sub(r"\s+", " ", s)
    return s.strip()


def _turn1_unvalidated_effect_guard_card_name(card):
    try:
        return tf.card_name(card)
    except Exception:
        pass

    if isinstance(card, dict):
        ident = card.get("identity") or {}
        return (
            card.get("name")
            or card.get("card_name")
            or ident.get("name")
            or ident.get("canonical_name")
            or ""
        )

    return ""


def _turn1_unvalidated_effect_guard_find_deck_from_call(args, kwargs):
    values = list(kwargs.values()) + list(args)

    for value in values:
        if not isinstance(value, list):
            continue

        if len(value) < 1:
            continue

        sample = value[: min(10, len(value))]

        if not all(isinstance(x, dict) for x in sample):
            continue

        names = [_turn1_unvalidated_effect_guard_card_name(x) for x in sample]

        if any(names):
            return value

    return None


def _turn1_unvalidated_effect_guard_deck_card_name_norms(deck):
    names = set()

    if not deck:
        return names

    for card in deck:
        name = _turn1_unvalidated_effect_guard_card_name(card)

        if not name:
            continue

        names.add(_turn1_unvalidated_effect_guard_norm_name(name))

    return names


def _turn1_unvalidated_effect_guard_action_labels(line):
    raw = str(line or "").strip()

    if not raw or raw == "none":
        return []

    return [part.strip() for part in raw.split("->") if part.strip()]


def _turn1_unvalidated_effect_guard_actions(line, deck, reqs=None):
    card_names = _turn1_unvalidated_effect_guard_deck_card_name_norms(deck)

    if not card_names:
        return []

    blocked = []

    for action in _turn1_unvalidated_effect_guard_action_labels(line):
        action_norm = _turn1_unvalidated_effect_guard_norm_name(action)

        if not action_norm:
            continue

        # Physical card name: allowed. The action-goal compatibility filter handles
        # whether the physical card's search target is compatible with the goal.
        if action_norm in card_names:
            continue

        ok, reason = _turn1_effect_goal_compat_label_compatible_with_goal(action, deck, reqs or [])
        if ok:
            continue

        blocked.append(action)

    return blocked


_ORIG_SIMULATE_ONE_GOAL_TRIAL_BEFORE_UNVALIDATED_EFFECT_GUARD = simulate_one_goal_trial

def simulate_one_goal_trial(*args, **kwargs):
    result = _ORIG_SIMULATE_ONE_GOAL_TRIAL_BEFORE_UNVALIDATED_EFFECT_GUARD(*args, **kwargs)

    try:
        if not isinstance(result, dict):
            return result

        if not result.get("success"):
            return result

        line = result.get("line") or "none"

        if line == "none":
            return result

        deck = _turn1_unvalidated_effect_guard_find_deck_from_call(args, kwargs)
        reqs = _turn1_effect_goal_compat_find_reqs_from_call(args, kwargs)
        blocked = _turn1_unvalidated_effect_guard_actions(line, deck, reqs)

        if not blocked:
            return result

        result["success"] = False
        result["success_stage"] = "blocked_unvalidated_effect_action"
        result["blocked_unvalidated_effect_actions"] = blocked
        result["missing_requirements"] = [
            "Blocked unvalidated ability/effect action: " + ", ".join(blocked)
        ]

        return result

    except Exception as exc:
        # Do not crash the simulation from the safety guard.
        try:
            result["unvalidated_effect_guard_error"] = str(exc)
        except Exception:
            pass
        return result




# ---------------------------------------------------------------------
# TURN1_SEARCH_EFFECT_TARGET_FILTERS
# ---------------------------------------------------------------------
# Root fix:
# The Turn-1 goal finder was reusing single-target scoring/execution too broadly.
# It allowed search cards/effects to satisfy goals outside their legal target class:
#   - Buddy-Buddy Poffin used for Energy goals
#   - Irida used for Basic Water Energy goals
#   - Shivery Chill used for Pokemon goals
#
# This guard classifies the ACCESS TARGETS of each action/effect, then requires
# compatibility with the current missing goal target before the action can be
# scored. It also post-filters successful lines as a safety net.


def _turn1_action_goal_compat_norm(value):
    import re as _re
    import unicodedata as _unicodedata

    s = str(value or "")
    s = _unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not _unicodedata.combining(ch))
    s = s.lower().replace("’", "'").replace("`", "'")
    s = _re.sub(r"\s+", " ", s)
    return s.strip()


def _turn1_action_goal_compat_flatten_strings(obj, max_items=6000):
    out = []
    seen = set()

    def rec(x):
        if len(out) >= max_items:
            return
        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(x, str):
            if x.strip():
                out.append(x)
            return

        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k.strip():
                    out.append(k)
                rec(v)
            return

        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)

    rec(obj)
    return " ".join(out)


def _turn1_action_goal_compat_card_name(card):
    try:
        return tf.card_name(card)
    except Exception:
        pass

    if isinstance(card, dict):
        ident = card.get("identity") or {}
        return (
            card.get("name")
            or card.get("card_name")
            or ident.get("name")
            or ident.get("canonical_name")
            or ""
        )

    return ""


def _turn1_action_goal_compat_card_identity_blob(card):
    if not isinstance(card, dict):
        return ""

    ident = card.get("identity") or {}
    parts = []

    for source in [ident, card]:
        for key in ["name", "canonical_name", "supertype", "subtype", "subtypes", "types", "hp"]:
            val = source.get(key) if isinstance(source, dict) else None
            if isinstance(val, list):
                parts.extend(str(x) for x in val)
            elif val is not None:
                parts.append(str(val))

    return " ".join(parts)


def _turn1_action_goal_compat_card_hp(card):
    if not isinstance(card, dict):
        return None

    ident = card.get("identity") or {}
    for source in [card, ident]:
        for key in ["hp", "HP"]:
            val = source.get(key) if isinstance(source, dict) else None
            if val is None:
                continue
            try:
                return int(str(val).strip())
            except Exception:
                pass

    return None


def _turn1_action_goal_compat_goal_classes_for_card(card):
    blob = _turn1_action_goal_compat_norm(_turn1_action_goal_compat_card_identity_blob(card))
    name = _turn1_action_goal_compat_norm(_turn1_action_goal_compat_card_name(card))
    both = f"{blob} {name}"

    classes = set()

    if "pokemon" in both or "pokémon" in both:
        classes.add("pokemon")
        if "basic" in both:
            classes.add("basic_pokemon")
        if "water" in both:
            classes.add("water_pokemon")
            if "basic" in both:
                classes.add("basic_water_pokemon")
        if "fire" in both:
            classes.add("fire_pokemon")
        if "grass" in both:
            classes.add("grass_pokemon")
        if "lightning" in both:
            classes.add("lightning_pokemon")
        if "psychic" in both:
            classes.add("psychic_pokemon")
        if "fighting" in both:
            classes.add("fighting_pokemon")
        if "darkness" in both or "dark" in both:
            classes.add("darkness_pokemon")
        if "metal" in both:
            classes.add("metal_pokemon")
        if "dragon" in both:
            classes.add("dragon_pokemon")
        if "colorless" in both:
            classes.add("colorless_pokemon")

    if "energy" in both or "{w}" in both or "water energy" in both:
        classes.add("energy")
        if "basic" in both:
            classes.add("basic_energy")
        if "water" in both or "{w}" in both:
            classes.add("water_energy")
            if "basic" in both:
                classes.add("basic_water_energy")
        if "fire" in both or "{r}" in both:
            classes.add("fire_energy")
        if "grass" in both or "{g}" in both:
            classes.add("grass_energy")
        if "lightning" in both or "{l}" in both:
            classes.add("lightning_energy")
        if "psychic" in both or "{p}" in both:
            classes.add("psychic_energy")
        if "fighting" in both or "{f}" in both:
            classes.add("fighting_energy")
        if "darkness" in both or "{d}" in both:
            classes.add("darkness_energy")
        if "metal" in both or "{m}" in both:
            classes.add("metal_energy")

    if "supporter" in both:
        classes.add("supporter")
        classes.add("trainer")
    if "item" in both:
        classes.add("item")
        classes.add("trainer")
    if "stadium" in both:
        classes.add("stadium")
        classes.add("trainer")
    if "tool" in both:
        classes.add("tool")
        classes.add("trainer")
    if "trainer" in both:
        classes.add("trainer")

    return classes


def _turn1_action_goal_compat_goal_classes_from_norm(target_norm, candidate_cards):
    classes = set()
    target_norm = str(target_norm or "")

    for c in candidate_cards or []:
        try:
            if tf.target_matches(c, target_norm):
                classes.update(_turn1_action_goal_compat_goal_classes_for_card(c))
        except Exception:
            continue

    # Text fallback for goals like "Basic Water Energy".
    t = _turn1_action_goal_compat_norm(target_norm)
    if "pokemon" in t or "pokémon" in t:
        classes.add("pokemon")
        if "basic" in t:
            classes.add("basic_pokemon")
        if "water" in t:
            classes.add("water_pokemon")
            if "basic" in t:
                classes.add("basic_water_pokemon")

    if "energy" in t:
        classes.add("energy")
        if "basic" in t:
            classes.add("basic_energy")
        if "water" in t or "{w}" in t:
            classes.add("water_energy")
            if "basic" in t:
                classes.add("basic_water_energy")
        if "fighting" in t or "{f}" in t:
            classes.add("fighting_energy")

    if "supporter" in t:
        classes.add("supporter")
        classes.add("trainer")
    if "item" in t:
        classes.add("item")
        classes.add("trainer")
    if "stadium" in t:
        classes.add("stadium")
        classes.add("trainer")
    if "trainer" in t:
        classes.add("trainer")

    return classes


def _turn1_action_goal_compat_access_classes_from_text(text):
    import re as _re

    t = _turn1_action_goal_compat_norm(text)
    classes = set()

    # Opponent-only disruption is not self access.
    opponent_only = [
        "your opponent shuffles",
        "your opponent reveals",
        "your opponent draws",
        "opponent's hand",
        "their hand",
        "their deck",
    ]
    self_markers = [
        "your deck",
        "your hand",
        "put into your hand",
        "onto your bench",
        "put it onto your bench",
        "draw cards",
        "you may draw",
    ]
    if any(m in t for m in opponent_only) and not any(m in t for m in self_markers):
        return {"opponent_only"}

    # Energy targets. Keep typed/restricted classes specific.
    if _re.search(r"(search|reveal|choose|put|attach|find).{0,180}basic water energy", t):
        classes.add("basic_water_energy")
    elif _re.search(r"(search|reveal|choose|put|attach|find).{0,180}water energy", t):
        classes.add("water_energy")
    elif _re.search(r"(search|reveal|choose|put|attach|find).{0,180}basic energy", t):
        classes.add("basic_energy")
    elif _re.search(r"(search|reveal|choose|put|attach|find).{0,180}energy", t):
        classes.add("energy")

    # Pokemon targets. Restricted classes stay specific.
    if _re.search(r"(search|reveal|choose|put|bench|find).{0,220}basic water (pokemon|pokémon)", t):
        classes.add("basic_water_pokemon")
    elif _re.search(r"(search|reveal|choose|put|bench|find).{0,220}water (pokemon|pokémon)", t):
        classes.add("water_pokemon")
    elif _re.search(r"(search|reveal|choose|put|bench|find).{0,220}basic (pokemon|pokémon)", t):
        classes.add("basic_pokemon")
    elif _re.search(r"(search|reveal|choose|put|bench|find).{0,220}(pokemon|pokémon)", t):
        classes.add("pokemon")
    elif "put onto your bench" in t or "put it onto your bench" in t:
        classes.add("pokemon")

    # Trainer targets.
    if _re.search(r"(search|reveal|choose|put|find).{0,180}supporter", t):
        classes.add("supporter")
    if _re.search(r"(search|reveal|choose|put|find).{0,180}item", t):
        classes.add("item")
    if _re.search(r"(search|reveal|choose|put|find).{0,180}stadium", t):
        classes.add("stadium")
    if _re.search(r"(search|reveal|choose|put|find).{0,180}trainer", t):
        classes.add("trainer")

    # Generic self-draw can help any goal probabilistically.
    if _re.search(r"\b(draw|draws|drawn)\b.{0,80}\b(card|cards)\b", t) and "your opponent" not in t:
        classes.add("generic_draw")

    return classes


def _turn1_action_goal_compat_action_source_and_text(action):
    # Returns (label/source_name, text_blob). The text blob should describe the
    # effect's access target, not only the source card identity.
    if isinstance(action, dict) and action.get("_virtual_action"):
        va = str(action.get("_virtual_action") or "")
        if action.get("effect") is not None:
            return va, _turn1_action_goal_compat_flatten_strings(action.get("effect"))
        if action.get("source") is not None:
            return _turn1_action_goal_compat_card_name(action.get("source")), _turn1_action_goal_compat_flatten_strings(action.get("source"))
        if action.get("card") is not None:
            return _turn1_action_goal_compat_card_name(action.get("card")), _turn1_action_goal_compat_flatten_strings(action.get("card"))
        if action.get("search_card") is not None:
            return _turn1_action_goal_compat_card_name(action.get("search_card")), _turn1_action_goal_compat_flatten_strings(action.get("search_card"))
        return va, _turn1_action_goal_compat_flatten_strings(action)

    if isinstance(action, dict):
        return _turn1_action_goal_compat_card_name(action), _turn1_action_goal_compat_flatten_strings(action)

    return str(action), str(action)


def _turn1_action_goal_compat_target_cards_for_norm(target_norm, st):
    cards = []
    for attr in ["deck", "hand", "discard", "prizes", "bench"]:
        vals = getattr(st, attr, []) or []
        if isinstance(vals, list):
            cards.extend(vals)
    active = getattr(st, "active", None)
    if active is not None:
        cards.append(active)
    return [c for c in cards if isinstance(c, dict)]


def _turn1_action_goal_compat_action_matches_target_directly(action, target_norm):
    if isinstance(action, dict) and not action.get("_virtual_action"):
        try:
            return tf.target_matches(action, target_norm)
        except Exception:
            return False
    return False


def _turn1_action_goal_compat_effect_allows_target_constraints(text, target_cards):
    t = _turn1_action_goal_compat_norm(text)

    # Buddy-Buddy Poffin style HP restriction.
    if "70 hp or less" in t or "70 hp or fewer" in t:
        matched = False
        for card in target_cards or []:
            hp = _turn1_action_goal_compat_card_hp(card)
            if hp is None:
                continue
            matched = True
            if hp <= 70:
                return True
        return not matched

    return True


def _turn1_action_goal_compat_classes_compatible(action_classes, target_classes):
    if not action_classes:
        return True  # Unknown non-access action; do not block here.

    if "opponent_only" in action_classes:
        return False

    if "generic_draw" in action_classes:
        return True

    # Broad/unrestricted categories.
    if "pokemon" in action_classes and target_classes.intersection({
        "pokemon", "basic_pokemon", "water_pokemon", "basic_water_pokemon",
        "fire_pokemon", "grass_pokemon", "lightning_pokemon", "psychic_pokemon",
        "fighting_pokemon", "darkness_pokemon", "metal_pokemon", "dragon_pokemon",
        "colorless_pokemon",
    }):
        return True

    if "energy" in action_classes and target_classes.intersection({
        "energy", "basic_energy", "water_energy", "basic_water_energy", "fire_energy",
        "grass_energy", "lightning_energy", "psychic_energy", "fighting_energy",
        "darkness_energy", "metal_energy",
    }):
        return True

    if "trainer" in action_classes and target_classes.intersection({"trainer", "supporter", "item", "stadium", "tool"}):
        return True

    # Restricted exact/subclass matches.
    restricted_pairs = [
        ("basic_pokemon", {"basic_pokemon", "basic_water_pokemon"}),
        ("water_pokemon", {"water_pokemon", "basic_water_pokemon"}),
        ("basic_water_pokemon", {"basic_water_pokemon"}),
        ("basic_energy", {"basic_energy", "basic_water_energy"}),
        ("water_energy", {"water_energy", "basic_water_energy"}),
        ("basic_water_energy", {"basic_water_energy"}),
        ("supporter", {"supporter"}),
        ("item", {"item"}),
        ("stadium", {"stadium"}),
        ("tool", {"tool"}),
    ]

    for action_class, allowed_targets in restricted_pairs:
        if action_class in action_classes and target_classes.intersection(allowed_targets):
            return True

    return bool(action_classes.intersection(target_classes))


def _turn1_action_goal_compat_action_can_help_target(action, target_norm, st):
    if _turn1_action_goal_compat_action_matches_target_directly(action, target_norm):
        return True, "action_is_target_card"

    label, text = _turn1_action_goal_compat_action_source_and_text(action)
    target_cards = _turn1_action_goal_compat_target_cards_for_norm(target_norm, st)
    target_classes = _turn1_action_goal_compat_goal_classes_from_norm(target_norm, target_cards)
    action_classes = _turn1_action_goal_compat_access_classes_from_text(text)

    if not _turn1_action_goal_compat_effect_allows_target_constraints(text, [c for c in target_cards if tf.target_matches(c, target_norm)]):
        return False, f"effect_constraints_block:{label}"

    if _turn1_action_goal_compat_classes_compatible(action_classes, target_classes):
        return True, "compatible"

    return False, f"incompatible_search_target:{label}:action={sorted(action_classes)}:target={sorted(target_classes)}"


_ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_BEFORE_ACTION_GOAL_COMPAT = score_candidate_for_missing_targets

def score_candidate_for_missing_targets(st, missing, going, enable_chain_search):
    # TURN1_ACTION_BUDGET_GUARD
    if _turn1_action_budget_exhausted(st):
        return []
    rows = _ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_BEFORE_ACTION_GOAL_COMPAT(st, missing, going, enable_chain_search)
    filtered = []

    for score, action, target_norm in rows:
        ok, reason = _turn1_action_goal_compat_action_can_help_target(action, target_norm, st)
        if ok:
            filtered.append((score, action, target_norm))
        else:
            try:
                st.log.append({
                    "event": "blocked_incompatible_search_target",
                    "action": action_label(action),
                    "target_norm": target_norm,
                    "reason": reason,
                })
            except Exception:
                pass

    return filtered


def _turn1_action_goal_compat_goal_classes_for_reqs(reqs, deck):
    classes = set()
    for req in reqs or []:
        for opt in getattr(req, "options", []) or []:
            classes.update(_turn1_action_goal_compat_goal_classes_from_norm(getattr(opt, "norm", str(opt)), deck))
    return classes


def _turn1_action_goal_compat_deck_card_by_name(deck):
    out = {}
    for card in deck or []:
        name = _turn1_action_goal_compat_card_name(card)
        norm = _turn1_action_goal_compat_norm(name)
        if norm and norm not in out:
            out[norm] = card
    return out


def _turn1_action_goal_compat_action_labels_from_line(line):
    raw = str(line or "").strip()
    if not raw or raw == "none":
        return []
    return [p.strip() for p in raw.split("->") if p.strip()]


def _turn1_action_goal_compat_label_compatible_with_any_goal(label, deck, reqs):
    by_name = _turn1_action_goal_compat_deck_card_by_name(deck)
    norm = _turn1_action_goal_compat_norm(label)

    # If the action itself is one of the goal cards, keep it.
    action_card = by_name.get(norm)
    if action_card is not None:
        for req in reqs or []:
            for opt in getattr(req, "options", []) or []:
                try:
                    if card_matches_option(action_card, opt):
                        return True, "action_card_is_goal"
                except Exception:
                    pass

    # Find text for a physical card name or an ability/effect label.
    texts = []
    if action_card is not None:
        texts.append(_turn1_action_goal_compat_flatten_strings(action_card))
    else:
        # Ability/effect label: find cards whose text contains that label.
        for card in deck or []:
            blob = _turn1_action_goal_compat_flatten_strings(card)
            if norm and norm in _turn1_action_goal_compat_norm(blob):
                texts.append(blob)

    if not texts:
        return True, "unknown_action_label"

    action_classes = set()
    for txt in texts:
        action_classes.update(_turn1_action_goal_compat_access_classes_from_text(txt))

    goal_classes = _turn1_action_goal_compat_goal_classes_for_reqs(reqs, deck)

    if _turn1_action_goal_compat_classes_compatible(action_classes, goal_classes):
        return True, "compatible_with_goal"

    return False, f"line_action_incompatible:{label}:action={sorted(action_classes)}:goal={sorted(goal_classes)}"


_ORIG_SIMULATE_ONE_GOAL_TRIAL_BEFORE_ACTION_GOAL_COMPAT = simulate_one_goal_trial

def simulate_one_goal_trial(*args, **kwargs):
    result = _ORIG_SIMULATE_ONE_GOAL_TRIAL_BEFORE_ACTION_GOAL_COMPAT(*args, **kwargs)

    try:
        if not isinstance(result, dict) or not result.get("success"):
            return result

        line = result.get("line") or "none"
        if line == "none":
            return result

        deck = kwargs.get("deck") if "deck" in kwargs else (args[0] if len(args) > 0 else None)
        reqs = kwargs.get("reqs") if "reqs" in kwargs else (args[2] if len(args) > 2 else None)

        if not deck or not reqs:
            return result

        blocked = []
        for label in _turn1_action_goal_compat_action_labels_from_line(line):
            ok, reason = _turn1_action_goal_compat_label_compatible_with_any_goal(label, deck, reqs)
            if not ok:
                blocked.append({"action": label, "reason": reason})

        if blocked:
            result["success"] = False
            result["success_stage"] = "blocked_incompatible_search_target"
            result["blocked_incompatible_search_targets"] = blocked
            result["missing_requirements"] = [
                "Blocked incompatible search target: " + ", ".join(b["action"] for b in blocked)
            ]

        return result

    except Exception as exc:
        try:
            result["action_goal_compat_guard_error"] = str(exc)
        except Exception:
            pass
        return result




# ---------------------------------------------------------------------
# TURN1_VALIDATED_EFFECT_LABELS
# ---------------------------------------------------------------------
# Root fix for ability/effect labels in Turn-1 access lines.
#
# Earlier unvalidated-effect guard blocked every action label that was not a physical
# card name in the deck. That correctly stopped fake lines like:
#   Ultra Ball -> Shivery Chill
# for Pokemon goals, but it was too conservative for legitimate ability effects.
#
# Example legitimate case:
#   Goal: 4x Basic Water Energy
#   Chien-Pao ex / Shivery Chill searches up to 2 Basic Water Energy.
#
# This patch changes the unvalidated-effect guard from:
#   "block all non-card-name action labels"
# to:
#   "block non-card-name labels unless their owning effect text proves that
#    the effect can legally access the current goal class."
#
# It is broad, not a Chien-Pao-specific whitelist:
#   - Basic Pokemon search can help Pokemon goals, not Energy goals.
#   - Basic Water Energy search can help Basic Water Energy / Energy goals.
#   - Supporter search can help Supporter goals, not Pokemon/Energy goals.
#   - Generic self draw can help any goal.
#   - Opponent-only disruption is blocked.


def _turn1_effect_goal_compat_norm(value):
    import re as _re
    import unicodedata as _unicodedata

    s = str(value or "")
    s = _unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not _unicodedata.combining(ch))
    s = s.lower()
    s = s.replace("’", "'").replace("`", "'")
    s = _re.sub(r"\s+", " ", s)
    return s.strip()


def _turn1_effect_goal_compat_flatten_strings(obj, max_items=7000):
    out = []
    seen = set()

    def rec(x):
        if len(out) >= max_items:
            return

        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(x, str):
            if x.strip():
                out.append(x)
            return

        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k.strip():
                    out.append(k)
                rec(v)
            return

        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)
            return

    rec(obj)
    return " ".join(out)


def _turn1_effect_goal_compat_card_name(card):
    try:
        return tf.card_name(card)
    except Exception:
        pass

    if isinstance(card, dict):
        ident = card.get("identity") or {}
        return (
            card.get("name")
            or card.get("card_name")
            or ident.get("name")
            or ident.get("canonical_name")
            or ""
        )

    return ""


def _turn1_effect_goal_compat_card_identity_blob(card):
    if not isinstance(card, dict):
        return str(card or "")

    ident = card.get("identity") or {}
    parts = []

    for source in [ident, card]:
        if not isinstance(source, dict):
            continue
        for key in [
            "name", "canonical_name", "supertype", "subtype", "subtypes", "types",
            "trainerType", "energyType", "hp", "HP",
        ]:
            val = source.get(key)
            if isinstance(val, list):
                parts.extend(str(x) for x in val)
            elif val is not None:
                parts.append(str(val))

    name = _turn1_effect_goal_compat_card_name(card)
    if name:
        parts.append(name)

    return " ".join(parts)


def _turn1_effect_goal_compat_classes_for_card_or_text(value):
    blob = _turn1_effect_goal_compat_norm(
        _turn1_effect_goal_compat_card_identity_blob(value) if isinstance(value, dict) else str(value or "")
    )

    classes = set()

    # Pokemon card identity / goal text.
    if "pokemon" in blob or "pokémon" in blob:
        classes.add("pokemon")
        if "basic" in blob:
            classes.add("basic_pokemon")
        for typ in [
            "water", "fire", "grass", "lightning", "psychic", "fighting",
            "darkness", "metal", "dragon", "colorless",
        ]:
            if typ in blob:
                classes.add(f"{typ}_pokemon")
                if "basic" in blob:
                    classes.add(f"basic_{typ}_pokemon")

    # Energy card identity / goal text. Handles symbols like {W}, [W].
    if "energy" in blob or "{w}" in blob or "[w]" in blob:
        classes.add("energy")
        if "basic" in blob:
            classes.add("basic_energy")

        energy_markers = [
            ("water", ["water", "{w}", "[w]"]),
            ("fire", ["fire", "{r}", "[r]"]),
            ("grass", ["grass", "{g}", "[g]"]),
            ("lightning", ["lightning", "electric", "{l}", "[l]"]),
            ("psychic", ["psychic", "{p}", "[p]"]),
            ("fighting", ["fighting", "{f}", "[f]"]),
            ("darkness", ["darkness", "dark", "{d}", "[d]"]),
            ("metal", ["metal", "{m}", "[m]"]),
        ]

        for typ, markers in energy_markers:
            if any(m in blob for m in markers):
                classes.add(f"{typ}_energy")
                if "basic" in blob:
                    classes.add(f"basic_{typ}_energy")

    # Trainer card identity / goal text.
    if "trainer" in blob:
        classes.add("trainer")
    if "supporter" in blob:
        classes.update(["trainer", "supporter"])
    if "item" in blob:
        classes.update(["trainer", "item"])
    if "stadium" in blob:
        classes.update(["trainer", "stadium"])
    if "tool" in blob:
        classes.update(["trainer", "tool"])

    return classes


def _turn1_effect_goal_compat_find_reqs_from_call(args, kwargs):
    # Preferred explicit names first.
    for key in ["reqs", "requirements", "goal_reqs"]:
        val = kwargs.get(key)
        if isinstance(val, list) and val and all(hasattr(x, "options") for x in val[: min(5, len(val))]):
            return val

    # Positional scan. A requirements list is usually a list of GoalRequirement
    # objects with .options/.zone/.min_count attributes, not a list of card dicts.
    for val in list(args) + list(kwargs.values()):
        if not isinstance(val, list) or not val:
            continue
        sample = val[: min(5, len(val))]
        if all(hasattr(x, "options") for x in sample):
            return val

    return []


def _turn1_effect_goal_compat_goal_classes(reqs, deck):
    classes = set()

    for req in reqs or []:
        matched_any = False
        opts = list(getattr(req, "options", []) or [])
        label = str(getattr(req, "label", "") or "")

        for card in deck or []:
            if not isinstance(card, dict):
                continue
            for opt in opts:
                try:
                    if card_matches_option(card, opt):
                        classes.update(_turn1_effect_goal_compat_classes_for_card_or_text(card))
                        matched_any = True
                except Exception:
                    pass

        # Fallback when the matching card cannot be found in the loaded deck.
        if label:
            classes.update(_turn1_effect_goal_compat_classes_for_card_or_text(label))
        for opt in opts:
            classes.update(_turn1_effect_goal_compat_classes_for_card_or_text(str(opt)))

    return classes


def _turn1_effect_goal_compat_deck_card_names(deck):
    names = set()
    for card in deck or []:
        name = _turn1_effect_goal_compat_card_name(card)
        if name:
            names.add(_turn1_effect_goal_compat_norm(name))
    return names


def _turn1_effect_goal_compat_effect_texts_for_label(action_label, deck):
    action_norm = _turn1_effect_goal_compat_norm(action_label)
    if not action_norm:
        return []

    # Physical card names are not effect labels.
    if action_norm in _turn1_effect_goal_compat_deck_card_names(deck):
        return []

    texts = []

    def walk(obj):
        if isinstance(obj, dict):
            local_blob = _turn1_effect_goal_compat_flatten_strings(obj)
            if action_norm in _turn1_effect_goal_compat_norm(local_blob):
                texts.append(local_blob)
                return
            for v in obj.values():
                walk(v)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                walk(v)

    for card in deck or []:
        if not isinstance(card, dict):
            continue
        card_blob = _turn1_effect_goal_compat_flatten_strings(card)
        if action_norm not in _turn1_effect_goal_compat_norm(card_blob):
            continue

        before = len(texts)
        walk(card)
        if len(texts) == before:
            texts.append(card_blob)

    # Deduplicate while preserving order.
    out = []
    seen = set()
    for t in texts:
        n = _turn1_effect_goal_compat_norm(t)
        if n and n not in seen:
            seen.add(n)
            out.append(t)
    return out


def _turn1_effect_goal_compat_access_classes_from_effect_text(text):
    import re as _re

    t = _turn1_effect_goal_compat_norm(text)
    classes = set()

    if not t:
        return classes

    # Opponent-only disruption is not self-access.
    opponent_only = any(
        m in t
        for m in [
            "your opponent shuffles", "your opponent draws", "your opponent reveals",
            "your opponent searches", "opponent's hand", "opponents hand",
            "their hand", "their deck",
        ]
    )
    self_access = any(
        m in t
        for m in [
            "your deck", "your hand", "put into your hand", "onto your bench",
            "put it onto your bench", "draw a card", "draw cards", "you may draw",
        ]
    )
    if opponent_only and not self_access:
        return {"opponent_only"}

    # Energy access. Handles Basic {W} Energy / Basic [W] Energy.
    if (
        "basic water energy" in t
        or "basic {w} energy" in t
        or "basic [w] energy" in t
        or _re.search(r"basic\s*(\{w\}|\[w\])\s*energy", t)
    ):
        classes.update(["energy", "basic_energy", "water_energy", "basic_water_energy"])
    elif "water energy" in t or "{w} energy" in t or "[w] energy" in t:
        classes.update(["energy", "water_energy"])
    elif "basic energy" in t:
        classes.update(["energy", "basic_energy"])
    elif _re.search(r"(search|reveal|choose|put|attach|find).{0,220}energy", t):
        classes.add("energy")

    # Pokemon access. Tie to access verbs so source identity text does not count.
    if _re.search(r"(search|look at|reveal|choose|put|bench|find).{0,240}(basic water (pokemon|pokémon))", t):
        classes.update(["pokemon", "basic_pokemon", "water_pokemon", "basic_water_pokemon"])
    elif _re.search(r"(search|look at|reveal|choose|put|bench|find).{0,240}(water (pokemon|pokémon))", t):
        classes.update(["pokemon", "water_pokemon"])
    elif _re.search(r"(search|look at|reveal|choose|put|bench|find).{0,240}(basic (pokemon|pokémon))", t):
        classes.update(["pokemon", "basic_pokemon"])
    elif _re.search(r"(search|look at|reveal|choose|put|bench|find).{0,240}(pokemon|pokémon)", t):
        classes.add("pokemon")
    elif "put onto your bench" in t or "put it onto your bench" in t:
        classes.add("pokemon")

    # Trainer access.
    if _re.search(r"(search|look at|reveal|choose|put|find).{0,220}supporter", t):
        classes.update(["trainer", "supporter"])
    if _re.search(r"(search|look at|reveal|choose|put|find).{0,220}item", t):
        classes.update(["trainer", "item"])
    if _re.search(r"(search|look at|reveal|choose|put|find).{0,220}stadium", t):
        classes.update(["trainer", "stadium"])
    if _re.search(r"(search|look at|reveal|choose|put|find).{0,220}trainer", t):
        classes.add("trainer")

    # Generic self draw can help any goal. Only mark generic draw when this text
    # did not already classify as a restricted search.
    if not classes and _re.search(r"\b(draw|draws|drawn)\b.{0,120}\b(card|cards)\b", t) and "your opponent" not in t:
        classes.add("generic_draw")

    return classes


def _turn1_effect_goal_compat_classes_compatible(access_classes, goal_classes):
    if not access_classes:
        return False

    if "opponent_only" in access_classes:
        return False

    if "generic_draw" in access_classes:
        return True

    if access_classes.intersection(goal_classes):
        return True

    # Broad category/subclass compatibility.
    pokemon_goal = bool(goal_classes.intersection({
        "pokemon", "basic_pokemon", "water_pokemon", "basic_water_pokemon",
        "fire_pokemon", "grass_pokemon", "lightning_pokemon", "psychic_pokemon",
        "fighting_pokemon", "darkness_pokemon", "metal_pokemon", "dragon_pokemon", "colorless_pokemon",
    }))
    if "pokemon" in access_classes and pokemon_goal:
        return True
    if "basic_pokemon" in access_classes and pokemon_goal:
        return True

    energy_goal = bool(goal_classes.intersection({
        "energy", "basic_energy", "water_energy", "basic_water_energy",
        "fire_energy", "grass_energy", "lightning_energy", "psychic_energy",
        "fighting_energy", "darkness_energy", "metal_energy",
    }))
    if "energy" in access_classes and energy_goal:
        return True
    if {"basic_energy", "water_energy", "basic_water_energy"}.intersection(access_classes) and energy_goal:
        return True

    trainer_goal = bool(goal_classes.intersection({"trainer", "supporter", "item", "stadium", "tool"}))
    if "trainer" in access_classes and trainer_goal:
        return True

    return False


def _turn1_effect_goal_compat_label_compatible_with_goal(action_label, deck, reqs):
    texts = _turn1_effect_goal_compat_effect_texts_for_label(action_label, deck)

    # Not an effect label. Card names are handled elsewhere.
    if not texts:
        return True, "not_effect_label"

    goal_classes = _turn1_effect_goal_compat_goal_classes(reqs, deck)
    if not goal_classes:
        return False, "effect_label_but_goal_unclassified"

    access_classes = set()
    for t in texts:
        access_classes.update(_turn1_effect_goal_compat_access_classes_from_effect_text(t))

    if _turn1_effect_goal_compat_classes_compatible(access_classes, goal_classes):
        return True, f"validated_effect_access:{sorted(access_classes)}"

    return False, f"incompatible_effect_access:access={sorted(access_classes)}:goal={sorted(goal_classes)}"




# -----------------------------------------------------------------------------
# TURN1_USE_COMPILED_EFFECT_RUNTIME
# -----------------------------------------------------------------------------
# Older debugging patches added post-hoc effect-label blockers. Those were useful
# for finding bad rows, but they are too blunt once target_finder has a real
# compiled-effect runtime. Neutralize them so validated ability labels can be
# used when their compiled/source effect is legal for the target.

def _turn1_noop_filter_summary(*args, **kwargs):
    return {"enabled": False, "invalidated_successes": 0, "reason": "target_finder compiled-effect runtime is source of truth"}


def _turn1_neutralize_old_posthoc_effect_filters():
    names_to_noop = [
        "turn1_apply_opponent_only_filter",
        "turn1_apply_effect_name_compatibility_filter",
        "turn1_apply_pre_summary_effect_label_compatibility_guard",
        "turn1_v32_apply_prereturn_incompatible_effect_guard",
    ]
    for name in names_to_noop:
        if name in globals():
            globals()[name] = _turn1_noop_filter_summary

    # unvalidated-effect-guard blocked every non-card action label. That is no longer correct:
    # validated ability labels such as Shivery Chill should be legal when their
    # compiled effect can satisfy the target.
    if "_turn1_unvalidated_effect_guard_actions" in globals():
        globals()["_turn1_unvalidated_effect_guard_actions"] = lambda line, deck: []

    # effect-goal-compat had a validated-effect pass, but it was still not the source of truth.
    if "_turn1_effect_goal_compat_unvalidated_effect_actions" in globals():
        globals()["_turn1_effect_goal_compat_unvalidated_effect_actions"] = lambda line, deck, reqs=None: []


_turn1_neutralize_old_posthoc_effect_filters()




# ---------------------------------------------------------------------
# TURN1_FILTER_OPPONENT_DEPENDENT_SUCCESS_LINES
# ---------------------------------------------------------------------
# Safety net for chains that were already built by nested runtime code.
# If a successful line contains a card whose access effect depends on unmodeled
# opponent state, convert that trial to a failure before summaries/CSV.

def _turn1_opponent_dependent_filter_norm(value):
    import re as _re
    import unicodedata as _unicodedata

    s = str(value or "")
    s = _unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not _unicodedata.combining(ch))
    s = s.lower()
    s = s.replace("’", "'").replace("`", "'")
    s = _re.sub(r"\s+", " ", s)
    return s.strip()


def _turn1_opponent_dependent_filter_flatten_strings(obj, max_items=5000):
    out = []
    seen = set()

    def rec(x):
        if len(out) >= max_items:
            return

        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(x, str):
            if x.strip():
                out.append(x)
            return

        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k.strip():
                    out.append(k)
                rec(v)
            return

        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)

    rec(obj)
    return " ".join(out)


def _turn1_opponent_dependent_filter_card_name(card):
    try:
        return tf.card_name(card)
    except Exception:
        pass

    if isinstance(card, dict):
        ident = card.get("identity") or {}
        return (
            card.get("name")
            or card.get("card_name")
            or ident.get("name")
            or ident.get("canonical_name")
            or ""
        )

    return ""


def _turn1_is_opponent_state_dependent_access_card(card):
    blob = _turn1_opponent_dependent_filter_norm(_turn1_opponent_dependent_filter_flatten_strings(card))

    if not blob:
        return False

    access_words = [
        "draw",
        "search",
        "look at",
        "reveal",
        "put into your hand",
        "put them into your hand",
        "put it into your hand",
    ]

    opponent_state_markers = [
        "for each of your opponent",
        "opponent's benched pokemon",
        "opponents benched pokemon",
        "opponent's benched pokémon",
        "opponents benched pokémon",
        "your opponent's benched",
        "your opponents benched",
        "your opponent has in play",
        "your opponent's hand",
        "your opponents hand",
        "your opponent's deck",
        "your opponents deck",
        "your opponent's active",
        "your opponents active",
    ]

    return any(a in blob for a in access_words) and any(m in blob for m in opponent_state_markers)


def _turn1_opponent_dependent_filter_action_labels(line):
    raw = str(line or "").strip()

    if not raw or raw == "none":
        return []

    return [part.strip() for part in raw.split("->") if part.strip()]


def _turn1_opponent_dependent_filter_blocked_names_from_deck(deck):
    out = {}

    for card in deck or []:
        name = _turn1_opponent_dependent_filter_card_name(card)

        if not name:
            continue

        if _turn1_is_opponent_state_dependent_access_card(card):
            out[_turn1_opponent_dependent_filter_norm(name)] = name

    return out


def _turn1_apply_opponent_dependent_filter(results, deck):
    blocked_names = _turn1_opponent_dependent_filter_blocked_names_from_deck(deck)

    if not blocked_names:
        return {
            "enabled": True,
            "invalidated_successes": 0,
            "blocked_cards": [],
        }

    invalidated = 0

    for r in results:
        if not r.get("success"):
            continue

        line = r.get("line") or "none"

        if line == "none":
            continue

        used_blocked = []

        for action in _turn1_opponent_dependent_filter_action_labels(line):
            action_norm = _turn1_opponent_dependent_filter_norm(action)

            if action_norm in blocked_names:
                used_blocked.append(blocked_names[action_norm])

        if not used_blocked:
            continue

        r["success"] = False
        r["success_stage"] = "blocked_opponent_dependent_access"
        r["blocked_opponent_dependent_access_cards"] = used_blocked
        r["missing_requirements"] = [
            "Blocked opponent-dependent access card: " + ", ".join(used_blocked)
        ]

        invalidated += 1

    return {
        "enabled": True,
        "invalidated_successes": invalidated,
        "blocked_cards": sorted(set(blocked_names.values())),
    }




# ---------------------------------------------------------------------
# TURN1_COLLAPSE_ACTIVE_COMPILED_REPEAT_LINES
# ---------------------------------------------------------------------
# Lightweight fix.
#
# v41 correctly found Active compiled search routes, e.g.:
#   active = Chien-Pao ex
#   ability = Shivery Chill
#
# But the greedy executor represented "search up to 2" as:
#   Shivery Chill -> Shivery Chill
#
# That is illegal as two ability uses, but equivalent to one legal use when
# the compiled/source text allows "up to 2" cards.
#
# This patch normalizes completed trial results:
# - repeated Active compiled search labels from the same source/ability are
#   collapsed to one line label when repeat_count <= compiled/source amount.
# - if repeat_count exceeds the allowed amount, the trial is marked failed.
#
# It avoids the expensive v43 runtime wrapper, keeping hotpath-cache performance.

def _turn1_result_normalizer_norm(value):
    import re as _re
    import unicodedata as _unicodedata

    s = str(value or "")
    s = _unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not _unicodedata.combining(ch))
    s = s.lower().replace("’", "'").replace("`", "'")
    s = _re.sub(r"\s+", " ", s)
    return s.strip()


def _turn1_result_normalizer_flatten_strings(obj, max_items=8000):
    out = []
    seen = set()

    def rec(x):
        if len(out) >= max_items:
            return

        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(x, str):
            if x.strip():
                out.append(x)
            return

        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k.strip():
                    out.append(k)
                rec(v)
            return

        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)

    rec(obj)
    return " ".join(out)


def _turn1_result_normalizer_card_name(card):
    try:
        return tf.card_name(card)
    except Exception:
        pass

    if isinstance(card, dict):
        ident = card.get("identity") or {}
        return (
            card.get("name")
            or card.get("card_name")
            or ident.get("name")
            or ident.get("canonical_name")
            or ""
        )

    return str(card or "")


def _turn1_result_normalizer_find_card_by_name(deck, source_name):
    source_n = _turn1_result_normalizer_norm(source_name)

    for card in deck or []:
        if _turn1_result_normalizer_norm(_turn1_result_normalizer_card_name(card)) == source_n:
            return card

    # Fuzzy fallback: "Chien-Pao ex" vs "Chien-Pao ex PAL 61"
    for card in deck or []:
        name_n = _turn1_result_normalizer_norm(_turn1_result_normalizer_card_name(card))

        if source_n and (source_n in name_n or name_n in source_n):
            return card

    return None


def _turn1_result_normalizer_search_amount_from_card(card, ability_name):
    import re as _re

    if not card:
        return 1

    blob = _turn1_result_normalizer_norm(_turn1_result_normalizer_flatten_strings(card))
    ability_n = _turn1_result_normalizer_norm(ability_name)

    windows = [blob]

    if ability_n and ability_n in blob:
        idx = blob.find(ability_n)
        windows.insert(0, blob[idx:idx + 1200])

    for window in windows:
        m = _re.search(r"search[^.]{0,180}?up to\s+(\d+)", window)
        if m:
            return max(1, int(m.group(1)))

        m = _re.search(r"up to\s+(\d+)[^.]{0,140}?(?:card|cards|pokemon|energy)", window)
        if m:
            return max(1, int(m.group(1)))

    return 1


def _turn1_result_normalizer_split_line(line):
    raw = str(line or "").strip()

    if not raw or raw == "none":
        return []

    return [x.strip() for x in raw.split("->") if x.strip()]


def _turn1_result_normalizer_join_line(parts):
    parts = [p for p in parts if str(p).strip()]

    if not parts:
        return "none"

    return " -> ".join(parts)


def _turn1_result_normalizer_remove_extra_occurrences(parts, label, keep=1):
    out = []
    seen = 0
    label_n = _turn1_result_normalizer_norm(label)

    for part in parts:
        if _turn1_result_normalizer_norm(part) == label_n:
            seen += 1

            if seen > keep:
                continue

        out.append(part)

    return out


def _turn1_normalize_trial_result(result, deck):
    if not isinstance(result, dict):
        return result

    log = result.get("log") or []
    if not isinstance(log, list):
        return result

    active_events = []

    for ev in log:
        if not isinstance(ev, dict):
            continue

        if ev.get("event") != "active_compiled_search_selected_v41":
            continue

        source = ev.get("source") or result.get("active") or ""
        ability = ev.get("ability") or ev.get("name") or ""

        if not ability:
            continue

        active_events.append((source, ability))

    if not active_events:
        return result

    grouped = {}

    for source, ability in active_events:
        key = (_turn1_result_normalizer_norm(source), _turn1_result_normalizer_norm(ability))
        grouped.setdefault(key, {"source": source, "ability": ability, "count": 0})
        grouped[key]["count"] += 1

    changed = False
    blocked = []

    line_parts = _turn1_result_normalizer_split_line(result.get("line"))
    played_parts = _turn1_result_normalizer_split_line(result.get("played"))

    for item in grouped.values():
        source = item["source"]
        ability = item["ability"]
        count = item["count"]

        source_card = _turn1_result_normalizer_find_card_by_name(deck, source)
        amount = _turn1_result_normalizer_search_amount_from_card(source_card, ability)

        if count <= 1:
            continue

        if count <= amount:
            line_parts = _turn1_result_normalizer_remove_extra_occurrences(line_parts, ability, keep=1)
            played_parts = _turn1_result_normalizer_remove_extra_occurrences(played_parts, ability, keep=1)
            changed = True

            result.setdefault("normalization_events", []).append(
                {
                    "event": "collapsed_repeated_active_compiled_search",
                    "source": source,
                    "ability": ability,
                    "repeat_count": count,
                    "allowed_search_amount": amount,
                    "reason": "Represented as one legal compiled ability use instead of repeated uses.",
                }
            )
        else:
            blocked.append(
                {
                    "source": source,
                    "ability": ability,
                    "repeat_count": count,
                    "allowed_search_amount": amount,
                }
            )

    if blocked:
        result["success"] = False
        result["success_stage"] = "blocked_repeat_active_compiled_search"
        result["blocked_repeat_active_compiled_search"] = blocked
        result["missing_requirements"] = [
            "Blocked repeated Active compiled ability use beyond allowed search amount."
        ]
        return result

    if changed:
        result["line"] = _turn1_result_normalizer_join_line(line_parts)
        result["played"] = _turn1_result_normalizer_join_line(played_parts)

    return result


def _turn1_normalize_results(results, deck):
    changed = 0
    blocked = 0

    for r in results or []:
        before_success = bool(r.get("success")) if isinstance(r, dict) else False
        before_line = r.get("line") if isinstance(r, dict) else None

        _turn1_normalize_trial_result(r, deck)

        if isinstance(r, dict):
            if before_success and not r.get("success"):
                blocked += 1
            if before_line != r.get("line"):
                changed += 1

    return {
        "enabled": True,
        "changed_lines": changed,
        "blocked_trials": blocked,
    }



# ---------------------------------------------------------------------
# TURN1_DIRECT_ACTION_BUDGET_V56
# ---------------------------------------------------------------------
# Direct planner/execution fix, not a wrapper.
#
# The 100-trial Chien-Pao profile after v55 showed the real bug:
#   128 simulate_one_goal_trial calls
#   689,321 execute_action calls
#
# That means chain-search was executing internal actions far beyond the user
# supplied --max-actions value. This block gives the existing execute_action
# and scoring functions a shared per-state budget. Once a SimState has executed
# max_actions actions, further execution/scoring on that same state stops.

_TURN1_ACTION_BUDGET_MAX_ACTIONS = None
_TURN1_ACTION_BUDGET_COUNTS_BY_STATE = {}


def _turn1_set_action_budget_from_args(args):
    global _TURN1_ACTION_BUDGET_MAX_ACTIONS, _TURN1_ACTION_BUDGET_COUNTS_BY_STATE

    try:
        value = int(getattr(args, "max_actions", 0) or 0)
    except Exception:
        value = 0

    _TURN1_ACTION_BUDGET_MAX_ACTIONS = value if value > 0 else None
    _TURN1_ACTION_BUDGET_COUNTS_BY_STATE = {}


def _turn1_action_budget_state_key(st):
    return id(st)


def _turn1_mark_action_budget_exhausted(st):
    try:
        setattr(st, "_turn1_action_budget_exhausted_flag", True)
    except Exception:
        pass


def _turn1_action_budget_exhausted(st):
    try:
        if bool(getattr(st, "_turn1_action_budget_exhausted_flag", False)):
            return True
    except Exception:
        pass

    limit = _TURN1_ACTION_BUDGET_MAX_ACTIONS
    if not limit or limit <= 0:
        return False

    try:
        count = int(getattr(st, "_turn1_action_budget_count", 0) or 0)
    except Exception:
        try:
            count = int(_TURN1_ACTION_BUDGET_COUNTS_BY_STATE.get(_turn1_action_budget_state_key(st), 0) or 0)
        except Exception:
            count = 0

    if count >= limit:
        _turn1_mark_action_budget_exhausted(st)
        return True

    return False
def _turn1_action_budget_allows_next_action(st):
    limit = _TURN1_ACTION_BUDGET_MAX_ACTIONS
    if not limit or limit <= 0:
        return True

    if _turn1_action_budget_exhausted(st):
        return False

    try:
        count = int(getattr(st, "_turn1_action_budget_count", 0) or 0)
        setattr(st, "_turn1_action_budget_count", count + 1)
    except Exception:
        key = _turn1_action_budget_state_key(st)
        count = int(_TURN1_ACTION_BUDGET_COUNTS_BY_STATE.get(key, 0) or 0)
        _TURN1_ACTION_BUDGET_COUNTS_BY_STATE[key] = count + 1

    return True
def run_goal_scenario(args: argparse.Namespace, deck: List[Dict[str, Any]], reqs: Sequence[GoalRequirement], mode: str, going: str) -> Dict[str, Any]:
    # TURN1_INIT_SCENARIO_ACTION_BUDGET
    _turn1_set_action_budget_from_args(args)
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
    # TURN1_APPLY_OPPONENT_ONLY_TEXT_GUARD
    opponent_only_filter_summary = turn1_apply_opponent_only_filter(results, deck)

    # TURN1_APPLY_EXCLUDE_PLAYED_PATHS_V15
    exclusion_summary = turn1_apply_excluded_played_paths(
        results,
        getattr(args, "exclude_played", ""),
    )

    # TURN1_APPLY_EFFECT_NAME_COMPATIBILITY_GUARD_V29
    effect_name_compatibility_summary = turn1_apply_effect_name_compatibility_filter(
        results,
        deck,
        reqs,
    )

    # TURN1_APPLY_SEARCH_TARGET_TYPE_SYSTEM
    results = turn1_result_goal_filter_results(results, deck, reqs)

    # TURN1_APPLY_OPPONENT_DEPENDENT_FILTER
    opponent_dependent_filter_summary = _turn1_apply_opponent_dependent_filter(results, deck)

    # TURN1_APPLY_ACTIVE_REPEAT_NORMALIZATION
    active_repeat_normalization_summary = _turn1_normalize_results(results, deck)

    examples = [r for r in results if r["success"] and r["line"] != "none"][: args.example_lines]
    failures = [r for r in results if not r["success"]][: args.example_lines]
    return {
        "going": going,
        "summary": (turn1_apply_pre_summary_effect_label_compatibility_guard(results, deck, reqs), summarize_goal_trials(results))[1],
        "effect_name_compatibility_summary": effect_name_compatibility_summary,
        "played_exclusion_summary": exclusion_summary,
        "opponent_only_filter_summary": opponent_only_filter_summary,
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

    fieldnames = [
        "goal_name",
        "going",
        "starting_hand_draw",
        "played",
        "line",
        "count",
        "raw_percent",
        "conditional_on_not_natural_percent",
        "exact_weighted_percent_of_trials",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sc in scenarios:
            exact = sc.get("exact_plus_simulation") or {}
            context_rows = exact.get("line_context_contributions") or []

            if not context_rows:
                context_rows = sc.get("summary", {}).get("top_success_line_contexts") or []

            for row in context_rows:
                writer.writerow({
                    "goal_name": goal_name,
                    "going": sc.get("going"),
                    "starting_hand_draw": row.get("starting_hand_draw") or "—",
                    "played": row.get("played") or row.get("line") or "—",
                    "line": row.get("line"),
                    "count": row.get("count"),
                    "raw_percent": row.get("raw_percent_of_trials") or row.get("percent"),
                    "conditional_on_not_natural_percent": row.get("conditional_on_not_natural_percent"),
                    "exact_weighted_percent_of_trials": row.get("exact_weighted_percent_of_trials"),
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




# ---------------------------------------------------------------------
# TURN1_PARSE_GOAL_FILE_MIN_COUNT_WRAPPER_V19
# ---------------------------------------------------------------------
_ORIG_PARSE_GOAL_FILE_V19 = parse_goal_file

def parse_goal_file(path: str, default_zone: str = "accessed") -> Tuple[str, str, List[GoalRequirement]]:
    name, mode, reqs = _ORIG_PARSE_GOAL_FILE_V19(path, default_zone=default_zone)

    try:
        data = json.load(open(path, encoding="utf-8-sig"))
        rows = [
            row
            for row in (data.get("requirements") or data.get("goals") or [])
            if isinstance(row, dict)
        ]

        for req, row in zip(reqs, rows):
            try:
                setattr(req, "min_count", max(1, int(row.get("min_count") or getattr(req, "min_count", 1) or 1)))
            except Exception:
                setattr(req, "min_count", max(1, int(getattr(req, "min_count", 1) or 1)))
    except Exception:
        pass

    return name, mode, reqs




# ---------------------------------------------------------------------
# TURN1_MULTI_GOAL_SEARCH_EXECUTOR
# ---------------------------------------------------------------------
# Fixes multi-card / multi-copy goals that were still using the old
# single-target search executor.
#
# Before this patch, a search card was executed against one target_norm.
# So a card like Buddy-Buddy Poffin could be played for only "N's Zorua",
# even if it could also satisfy "Budew" in the same action.
#
# This patch makes search_deck actions goal-aware:
# - compute actual remaining deficits across all goal requirements
# - for a search effect with amount N, choose up to N cards that satisfy
#   those deficits
# - score search cards by how many goal deficits they can cover
# - stop naturally because the normal loop re-checks goal_satisfied after
#   each action

_ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_BEFORE_GOAL_SEARCH = score_candidate_for_missing_targets
_ORIG_EXECUTE_ACTION_BEFORE_GOAL_SEARCH = execute_action


def _turn1_goal_card_instance_key(c: Dict[str, Any]) -> Any:
    if not isinstance(c, dict):
        return id(c)
    return c.get("_instance_id") or id(c)


def _turn1_goal_requirement_current_count(req: GoalRequirement, st: tf.SimState, tracker: GoalTracker) -> int:
    pool = zone_cards(st, tracker, req.zone)
    seen = set()
    n = 0
    for c in pool:
        if not isinstance(c, dict):
            continue
        if not any(card_matches_option(c, opt) for opt in req.options):
            continue
        key = _turn1_goal_card_instance_key(c)
        if key in seen:
            continue
        seen.add(key)
        n += 1
    return n


def _turn1_goal_requirement_deficit(req: GoalRequirement, st: tf.SimState, tracker: GoalTracker) -> int:
    return max(0, int(getattr(req, "min_count", 1) or 1) - _turn1_goal_requirement_current_count(req, st, tracker))


def _turn1_missing_goal_requirements_with_deficits(reqs: Sequence[GoalRequirement], mode: str, st: tf.SimState, tracker: GoalTracker) -> List[Tuple[GoalRequirement, int]]:
    rows: List[Tuple[GoalRequirement, int]] = []
    for req in reqs:
        d = _turn1_goal_requirement_deficit(req, st, tracker)
        if d > 0:
            rows.append((req, d))

    if mode == "any":
        # For ANY goals, satisfying any one requirement is enough. Keep the
        # achievable requirements, sorted by smallest deficit first.
        rows.sort(key=lambda x: (x[1], x[0].label))
    return rows


def _turn1_search_filter_allows_for_action(filt: Dict[str, Any], card: Dict[str, Any], action_name: str) -> bool:
    # Ultra Ball's compiled filters vary. The target finder already special-cases
    # it as any Pokémon; keep that behavior here.
    if tf.norm(action_name) == "ultra ball":
        return tf.card_supertype(card) == "Pokémon"
    return tf.filter_allows_card(filt, card)


def _turn1_search_steps_for_card(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for eff in tf.iter_effects(card):
        if tf.effect_is_trivial_rule(eff):
            continue
        for step in tf.flatten_steps(eff):
            if isinstance(step, dict) and step.get("op") == "search_deck":
                out.append(step)
    return out




# ---------------------------------------------------------------------
# TURN1_SOURCE_TEXT_TARGET_FILTERS
# ---------------------------------------------------------------------
# Broad legality guard for source-text search restrictions.
#
# Fixes cases like Buddy-Buddy Poffin where a compiled search filter can be
# too vague (for example, only saying "Basic"), causing the goal executor to
# treat Energy as reachable. Printed/source text such as:
#   "Search your deck for up to 2 Basic Pokemon with 70 HP or less and put
#    them onto your Bench"
# means the searched targets must be Basic Pokemon with HP <= 70, never Energy.
#
# This is intentionally not a Buddy-Buddy-only special case. It enforces two
# broad classes of restrictions:
#   1. cards put onto the Bench must be Pokemon;
#   2. HP-or-less searches require Pokemon with HP at or below the limit.

def _turn1_source_text_filter_norm(value: Any) -> str:
    try:
        return tf.norm(value)
    except Exception:
        return str(value or "").lower().replace("pokémon", "pokemon").strip()


def _turn1_source_text_filter_flatten_strings(value: Any, depth: int = 0) -> str:
    if value is None or depth > 4:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts = []
        preferred = [
            "name", "card_name", "text", "raw_text", "source_text", "combined_text",
            "rules", "abilities_text", "attacks_text", "effect_text", "description",
            "filter", "card_filter", "selection", "target", "targets",
        ]
        for key in preferred:
            if key in value:
                parts.append(_turn1_source_text_filter_flatten_strings(value.get(key), depth + 1))
        for key, val in value.items():
            if key not in preferred:
                parts.append(_turn1_source_text_filter_flatten_strings(val, depth + 1))
        return " ".join(p for p in parts if p)
    if isinstance(value, (list, tuple, set)):
        return " ".join(_turn1_source_text_filter_flatten_strings(v, depth + 1) for v in value)
    return str(value)


def _turn1_source_text_filter_step_text(step: Any) -> str:
    parts = []
    if isinstance(step, dict):
        for key in ("text", "raw_text", "source_text", "effect_text", "description"):
            val = step.get(key)
            if val:
                parts.append(str(val))
        try:
            fn = getattr(tf, "step_text", None)
            if callable(fn):
                val = fn(step)
                if val:
                    parts.append(str(val))
        except Exception:
            pass
    parts.append(_turn1_source_text_filter_flatten_strings(step))
    return " ".join(p for p in parts if p)


def _turn1_source_text_filter_action_card_text(action_card: Any) -> str:
    parts = []
    if isinstance(action_card, dict):
        parts.append(_turn1_source_text_filter_flatten_strings(action_card))
        try:
            for eff in tf.iter_effects(action_card):
                fn = getattr(tf, "ability_text_blob", None)
                if callable(fn):
                    try:
                        parts.append(str(fn(eff)))
                    except Exception:
                        pass
                parts.append(_turn1_source_text_filter_flatten_strings(eff))
        except Exception:
            pass
    return " ".join(p for p in parts if p)


def _turn1_source_text_filter_blob(filt: Any) -> str:
    try:
        return str(tf.filter_text_blob(filt))
    except Exception:
        return _turn1_source_text_filter_flatten_strings(filt)


def _turn1_search_source_text_blob(filt: Any, action_name: str, action_card: Any = None, source_step: Any = None) -> str:
    # Prefer the actual search step/filter, but include the full action text as
    # fallback for under-compiled filters.
    parts = [
        str(action_name or ""),
        _turn1_source_text_filter_step_text(source_step),
        _turn1_source_text_filter_blob(filt),
        _turn1_source_text_filter_action_card_text(action_card),
    ]
    return _turn1_source_text_filter_norm(" ".join(p for p in parts if p))


def _turn1_card_hp_for_source_filter(card: Dict[str, Any]):
    vals = []
    if isinstance(card, dict):
        vals.extend([card.get("hp"), card.get("HP"), card.get("raw_hp")])
        for key in ("identity", "gameplay", "raw_card", "source"):
            obj = card.get(key)
            if isinstance(obj, dict):
                vals.extend([obj.get("hp"), obj.get("HP"), obj.get("raw_hp")])
    for val in vals:
        if val is None:
            continue
        m = re.search(r"\d+", str(val))
        if m:
            try:
                return int(m.group(0))
            except Exception:
                pass
    return None


def _turn1_source_filter_is_pokemon(card: Dict[str, Any]) -> bool:
    try:
        return _turn1_source_text_filter_norm(tf.card_supertype(card)) in {"pokemon", "pokémon"}
    except Exception:
        return False


def _turn1_source_filter_is_basic_pokemon(card: Dict[str, Any]) -> bool:
    try:
        return bool(tf.is_basic_pokemon(card))
    except Exception:
        if not _turn1_source_filter_is_pokemon(card):
            return False
        try:
            subs = {_turn1_source_text_filter_norm(x) for x in tf.card_subtypes(card)}
        except Exception:
            subs = set()
        try:
            name_n = _turn1_source_text_filter_norm(tf.card_name(card))
        except Exception:
            name_n = ""
        return "basic" in subs or name_n.startswith("basic ")


def _turn1_target_phrase_from_search_text(blob: str) -> str:
    b = _turn1_source_text_filter_norm(blob)
    patterns = [
        r"search (?:your|the) deck(?: and (?:your )?discard pile)? for (.*?)(?:, reveal| reveal| and reveal|,? and put|,? put| then shuffle| shuffle|\.|$)",
        r"look at the top \d+ cards? of your deck.*?reveal (.*?)(?: card| cards|,| and put| put|$)",
        r"choose (.*?)(?: from (?:your )?deck| from among them|,| and put| put|$)",
    ]
    for pat in patterns:
        m = re.search(pat, b)
        if not m:
            continue
        phrase = m.group(1)
        phrase = re.sub(r"^(up to|exactly)?\s*\d+\s+", "", phrase).strip()
        phrase = re.sub(r"^(a|an|any|one|two)\s+", "", phrase).strip()
        if phrase:
            return phrase
    return b


def _turn1_source_text_allows_card(
    filt: Dict[str, Any],
    card: Dict[str, Any],
    action_name: str,
    action_card: Any = None,
    source_step: Any = None,
) -> bool:
    blob = _turn1_search_source_text_blob(filt, action_name, action_card, source_step)
    if not blob:
        return True

    target = _turn1_target_phrase_from_search_text(blob)
    target_n = _turn1_source_text_filter_norm(target)

    # If the searched card is put onto the Bench, the target cannot be Energy
    # or a Trainer. This catches Buddy-Buddy Poffin and similar Bench-search
    # effects even when the structured compiler filter is vague.
    if "onto your bench" in blob or "to your bench" in blob:
        if not _turn1_source_filter_is_pokemon(card):
            return False

    hp_match = re.search(r"(\d+)\s*hp\s*or\s*less", target_n) or re.search(r"(\d+)\s*hp\s*or\s*less", blob)
    if hp_match:
        if not _turn1_source_filter_is_pokemon(card):
            return False
        hp = _turn1_card_hp_for_source_filter(card)
        if hp is None or hp > int(hp_match.group(1)):
            return False

    if "basic pokemon" in target_n or "basic pokémon" in target_n:
        if not _turn1_source_filter_is_basic_pokemon(card):
            return False

    return True



# ---------------------------------------------------------------------




# ---------------------------------------------------------------------
# TURN1_DIRECT_CAPACITY_AND_SCORE_CACHE_V53
# ---------------------------------------------------------------------
# Direct source edit, not a same-name wrapper stack.
#
# Fixes two confirmed runtime problems from the 100-trial Chien-Pao profile:
#   1. _turn1_card_goal_search_capacity used copy.deepcopy(st) thousands
#      of times. The capacity check only needs independent zone lists, not deep
#      copies of every card dictionary.
#   2. chain-search repeatedly asks for the same candidate scoring result for
#      equivalent states. Cache the score list by a semantic state key and drop
#      zero-score actions before later planner filters.

_TURN1_SCORE_CANDIDATE_CACHE = {}
_TURN1_SCORE_CANDIDATE_CACHE_MAX = 200000


def _turn1_copy_container_for_score_capacity(value):
    """Copy containers, but keep card dictionaries by reference.

    The capacity scorer simulates selected cards entering hand. It does not need
    to mutate card dictionaries, so deep-copying the whole SimState is wasteful.
    """
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    if isinstance(value, set):
        return set(value)
    if isinstance(value, dict):
        return dict(value)
    return value


def _turn1_light_state_copy_for_score_capacity(st):
    fake = copy.copy(st)

    # Known SimState zone/mutable attributes. Missing attributes are ignored so
    # this stays compatible if the dataclass changes.
    for attr in [
        'deck',
        'hand',
        'bench',
        'active',
        'discard',
        'lost_zone',
        'prizes',
        'events',
        'log',
        'played',
        'played_cards',
        'attached_energy',
        'supporter_played',
        'stadium',
    ]:
        if hasattr(st, attr):
            try:
                setattr(fake, attr, _turn1_copy_container_for_score_capacity(getattr(st, attr)))
            except Exception:
                pass

    return fake


def _turn1_score_cache_card_key(card):
    if card is None:
        return None

    if not isinstance(card, dict):
        return ('value', str(card))

    ident = card.get('identity') or {}
    key = (
        card.get('card_id')
        or card.get('representative_card_id')
        or card.get('id')
        or ident.get('card_id')
        or ident.get('id')
        or ident.get('canonical_id')
    )
    name = (
        card.get('name')
        or card.get('card_name')
        or ident.get('name')
        or ident.get('canonical_name')
    )
    set_code = card.get('set_code') or ident.get('set_code') or card.get('set') or ident.get('set')
    number = card.get('number') or card.get('collector_number') or ident.get('number') or ident.get('collector_number')

    if key:
        return ('card', str(key))
    return ('card', str(name or ''), str(set_code or ''), str(number or ''))


def _turn1_score_cache_freeze_value(value, depth=0):
    if depth > 4:
        return ('deep', type(value).__name__)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        # Card dictionaries are huge; represent them by identity only.
        if value.get('identity') or value.get('card_id') or value.get('name') or value.get('card_name'):
            return _turn1_score_cache_card_key(value)
        return ('dict', tuple(sorted((str(k), _turn1_score_cache_freeze_value(v, depth + 1)) for k, v in value.items())))

    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_turn1_score_cache_freeze_value(v, depth + 1) for v in value))

    if isinstance(value, set):
        try:
            return ('set', tuple(sorted(_turn1_score_cache_freeze_value(v, depth + 1) for v in value)))
        except Exception:
            return ('set', tuple(_turn1_score_cache_freeze_value(v, depth + 1) for v in value))

    if hasattr(value, '__dict__'):
        try:
            return (type(value).__name__, tuple(sorted((str(k), _turn1_score_cache_freeze_value(v, depth + 1)) for k, v in vars(value).items())))
        except Exception:
            return (type(value).__name__, repr(value))

    return (type(value).__name__, repr(value))


def _turn1_score_cache_zone_key(st, attr):
    if not hasattr(st, attr):
        return None
    value = getattr(st, attr)
    if isinstance(value, list):
        return tuple(_turn1_score_cache_card_key(x) for x in value)
    if isinstance(value, tuple):
        return tuple(_turn1_score_cache_card_key(x) for x in value)
    if isinstance(value, dict):
        return tuple(sorted((str(k), _turn1_score_cache_freeze_value(v)) for k, v in value.items()))
    return _turn1_score_cache_freeze_value(value)


def _turn1_score_cache_key(st, missing, going, enable_chain_search):
    zones = []
    for attr in [
        'deck',
        'hand',
        'bench',
        'active',
        'discard',
        'lost_zone',
        'prizes',
        'played',
        'played_cards',
        'supporter_played',
        'attached_energy',
    ]:
        zones.append((attr, _turn1_score_cache_zone_key(st, attr)))

    return (
        tuple(zones),
        _turn1_score_cache_freeze_value(missing),
        str(going),
        bool(enable_chain_search),
    )


def _turn1_cached_score_candidates(st, missing, going, enable_chain_search):
    try:
        key = _turn1_score_cache_key(st, missing, going, enable_chain_search)
        cached = _TURN1_SCORE_CANDIDATE_CACHE.get(key)
        if cached is not None:
            return cached
    except Exception:
        key = None

    scored = _ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_V27(st, missing, going, enable_chain_search)

    if key is not None and len(_TURN1_SCORE_CANDIDATE_CACHE) < _TURN1_SCORE_CANDIDATE_CACHE_MAX:
        _TURN1_SCORE_CANDIDATE_CACHE[key] = scored

    return scored

def _turn1_card_goal_search_capacity(
    st: tf.SimState,
    card: Dict[str, Any],
    reqs: Sequence[GoalRequirement],
    mode: str,
    tracker: GoalTracker,
) -> Tuple[int, int]:
    """Return (selected_count_possible, max_search_amount_seen) for goal-aware search steps."""
    if not isinstance(card, dict):
        return (0, 0)

    action_name = tf.card_name(card)
    total_possible = 0
    max_amt = 0

    # Simulate selection without mutating real state.
    fake_st = _turn1_light_state_copy_for_score_capacity(st)
    fake_tracker = GoalTracker()
    fake_tracker.mark(tracker.accessed)

    for step in _turn1_search_steps_for_card(card):
        filt = tf.extract_filter(step)
        amt = max(1, tf.search_amount(step))
        max_amt = max(max_amt, amt)
        selected = _turn1_goal_select_from_deck(
            fake_st,
            reqs=reqs,
            mode=mode,
            tracker=fake_tracker,
            filt=filt,
            amount=amt,
            action_name=action_name,
            action_card=card,
            source_step=step,
        )
        if selected:
            fake_st.hand.extend(selected)
            fake_tracker.mark(selected)
            total_possible += len(selected)

    return total_possible, max_amt


def _turn1_score_playable_card_for_goal(
    card: Dict[str, Any],
    st: tf.SimState,
    reqs: Sequence[GoalRequirement],
    mode: str,
    tracker: GoalTracker,
    going: str,
    enable_chain_search: bool,
) -> float:
    if not isinstance(card, dict):
        return -1.0
    if not tf.card_can_be_played_from_hand(card, going, st.supporter_used):
        return -1.0
    if tf.is_meowth_ex(card) and st.last_ditch_used:
        return -1.0

    # Cost check: protect goal pieces where possible using the monkey-patched
    # protected has_enough_discard_fodder / discard_fodder functions.
    target_norms = all_goal_norms(reqs)
    primary = target_norms[0] if target_norms else ""
    if not tf.has_enough_discard_fodder(st.hand, card, primary):
        return -1.0

    selected_possible, max_amt = _turn1_card_goal_search_capacity(st, card, reqs, mode, tracker)
    if selected_possible <= 0:
        # Fall back to old single-target scoring for pure draw / ability / special lines.
        best = -1.0
        for tn in target_norms:
            try:
                best = max(best, tf.score_playable_card(card, st, tn, going, enable_chain_search))
            except Exception:
                pass
        return best

    # Goal-aware search cards should beat single-target Ultra Ball when they cover
    # more missing pieces. Penalize discard costs slightly so Poffin/Poké Pad are
    # preferred when equally useful.
    cost = 0
    try:
        cost = int(tf.card_known_discard_cost(card) or 0)
    except Exception:
        cost = 0

    supporter_penalty = 75 if tf.is_supporter(card) else 0
    return 12000.0 + (1000.0 * selected_possible) + (10.0 * max_amt) - (150.0 * cost) - supporter_penalty


def score_candidate_for_missing_targets(
    st: tf.SimState,
    missing: Sequence[GoalRequirement],
    going: str,
    enable_chain_search: bool,
) -> List[Tuple[float, Any, str]]:
    # TURN1_ACTION_BUDGET_GUARD
    if _turn1_action_budget_exhausted(st):
        return []
    """Goal-aware action scoring.

    This overrides the old scorer that evaluated every card against one target_norm.
    Search cards are now scored by how many remaining goal deficits they can cover.
    """
    scored: List[Tuple[float, Any, str]] = []
    target_norms = [n for req in missing for n in [choose_primary_target_norm(req, st)] if n]
    if not target_norms:
        return scored

    tracker = GoalTracker()
    snapshot_accessed(tracker, st)

    playable = [c for c in list(st.hand) if tf.card_can_be_played_from_hand(c, going, st.supporter_used)]
    for c in playable:
        s = _turn1_score_playable_card_for_goal(c, st, missing, "all", tracker, going, enable_chain_search)
        if s > 0:
            scored.append((s, c, target_norms[0]))

    # Keep the old non-hand / ability candidates. They are still single-target,
    # but the main bug was hand-played multi-search cards.
    try:
        old_scored = _ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_BEFORE_GOAL_SEARCH(st, missing, going, enable_chain_search)
        scored.extend(old_scored)
    except Exception:
        pass

    return scored


def _turn1_execute_goal_search_card(
    st: tf.SimState,
    card: Dict[str, Any],
    rng: random.Random,
    reqs: Sequence[GoalRequirement],
    mode: str,
    tracker: GoalTracker,
    going: str,
    enable_chain_search: bool,
) -> bool:
    if not isinstance(card, dict) or card not in st.hand:
        return False

    search_steps = _turn1_search_steps_for_card(card)
    if not search_steps:
        return False

    action_name = tf.card_name(card)

    selected_possible, _max_amt = _turn1_card_goal_search_capacity(st, card, reqs, mode, tracker)
    if selected_possible <= 0:
        return False

    target_norms = all_goal_norms(reqs)
    primary = target_norms[0] if target_norms else ""

    if not tf.card_can_be_played_from_hand(card, going, st.supporter_used):
        return False
    if not tf.has_enough_discard_fodder(st.hand, card, primary):
        st.log.append({"event": "cannot_play_missing_discard_fodder", "card": action_name, "required_discards": tf.card_known_discard_cost(card)})
        return False

    # Pay discard costs before moving the card. This calls the monkey-patched
    # goal-protecting discard function installed earlier in this module.
    cost = int(tf.card_known_discard_cost(card) or 0)
    if cost > 0:
        try:
            tf.discard_fodder_cards(st, cost, primary, f"cost_for_{action_name}")
        except TypeError:
            tf.discard_fodder_cards(st.hand, card, primary)

    if card not in st.hand:
        # The cost function should not discard the played card, but be defensive.
        return False

    st.hand.remove(card)
    st.discard.append(card)
    if tf.is_supporter(card):
        st.supporter_used = True
    st.actions_used += 1
    st.line.append(action_name)
    stage = f"after_play_{action_name}"
    st.log.append({"event": "play_card", "card": action_name, "supporter_used": st.supporter_used, "goal_aware_search": True})

    found_any = False

    # Run through effects conservatively. We only override search_deck selection;
    # common draw/shuffle ops are still handled directly enough for this planner.
    for eff in tf.iter_effects(card):
        if tf.effect_is_trivial_rule(eff):
            continue
        for step in tf.flatten_steps(eff):
            if goal_satisfied(reqs, mode, st, tracker):
                return True
            if not isinstance(step, dict):
                continue
            op = step.get("op")
            if op in {"reference_global_rule", "register_usage_limit", "register_play_condition", "play_condition", "register_continuous_modifier", "register_trigger", "register_replacement_effect", "register_knockout_prize_rule"}:
                continue
            if op == "shuffle_deck":
                rng.shuffle(st.deck)
                st.log.append({"event": "shuffle_deck", "stage": stage})
                continue
            if op in {"discard_cards", "discard_card"}:
                # Required costs were already paid above. Non-required discard effects
                # are ignored for turn-1 access unless target_finder knows them specially.
                continue
            if op in {"draw_cards", "draw_cards_per_coin_heads"}:
                n = tf.draw_amount_from_step(step, counts=st.counts, coin_heads=st.coin_heads)
                if op == "draw_cards_per_coin_heads" and n == 0:
                    n = st.coin_heads
                tf.draw_cards(st, n, stage)
                tracker.mark(st.hand)
                continue
            if op == "draw_until_hand_size":
                target_size = tf.amount_value(step.get("target_hand_size"), default=len(st.hand), counts=st.counts)
                tf.draw_cards(st, max(0, target_size - len(st.hand)), stage)
                tracker.mark(st.hand)
                continue
            if op == "search_deck":
                filt = tf.extract_filter(step)
                amt = max(1, tf.search_amount(step))
                selected = _turn1_goal_select_from_deck(
                    st,
                    reqs=reqs,
                    mode=mode,
                    tracker=tracker,
                    filt=filt,
                    amount=amt,
                    action_name=action_name,
                    action_card=card,
                    source_step=step,
                )
                if selected:
                    st.hand.extend(selected)
                    tracker.mark(selected)
                    found_any = True
                    st.log.append({
                        "event": "goal_aware_search_deck_selected",
                        "stage": stage,
                        "source": action_name,
                        "selected": [tf.card_name(c) for c in selected],
                        "amount": amt,
                        "filter": filt,
                    })
                continue

    if found_any:
        st.found = True
        st.found_stage = stage
    return found_any


def execute_action(st: tf.SimState, action: Any, target_norm: str, rng: random.Random, going: str, enable_chain_search: bool) -> None:
    """Goal-aware action executor override.

    Normal signature is preserved. The current goal reqs/tracker are attached to
    state by simulate_one_goal_trial before calling execute_action.
    """
    reqs = list(getattr(st, "_turn1_goal_reqs", []) or [])
    mode = str(getattr(st, "_turn1_goal_mode", "all") or "all")
    tracker = getattr(st, "_turn1_goal_tracker", None)

    if reqs and tracker is not None and isinstance(action, dict) and not action.get("_virtual_action"):
        if _turn1_execute_goal_search_card(st, action, rng, reqs, mode, tracker, going, enable_chain_search):
            return

    return _ORIG_EXECUTE_ACTION_BEFORE_GOAL_SEARCH(st, action, target_norm, rng, going, enable_chain_search)




# ---------------------------------------------------------------------
# TURN1_MEOWTH_FILTER_FIX_V23
# ---------------------------------------------------------------------
# Fixes a bug introduced by the multi-goal search executor.
#
# Problem:
#   Meowth ex / Last-Ditch Catch searches for a Supporter, but the old goal-search executor treated
#   any compiled search_deck step as a goal-aware direct search if the older
#   compiled filter was too permissive. That allowed Meowth ex to appear in
#   lines for goals like 4x N's Zorua + Budew, even though Meowth only gets a
#   Supporter and should not directly fetch those Pokémon.
#
# Fix:
#   1. Never execute Meowth ex through the direct goal-aware search override.
#      Let the original target-finder Meowth handler run instead.
#   2. Make the goal-search executor's search filter gate stricter for Supporter/Trainer/Item/
#      Pokémon/Energy text so cards cannot search outside their printed filter.

_ORIG_SEARCH_FILTER_ALLOWS_FOR_ACTION_BEFORE_V23 = _turn1_search_filter_allows_for_action
_ORIG_SCORE_PLAYABLE_CARD_FOR_GOAL_BEFORE_V23 = _turn1_score_playable_card_for_goal
_ORIG_EXECUTE_GOAL_SEARCH_CARD_BEFORE_V23 = _turn1_execute_goal_search_card


def _turn1_goal_search_filter_norm_text(value: Any) -> str:
    try:
        if isinstance(value, dict):
            parts = []
            for key in ("raw_text", "source_text", "text", "name", "kind", "type", "subtype", "supertype"):
                if value.get(key) is not None:
                    parts.append(str(value.get(key)))
            return tf.norm(" ".join(parts))
        return tf.norm(str(value or ""))
    except Exception:
        return str(value or "").lower()


def _turn1_goal_search_filter_card_is_item(card: Dict[str, Any]) -> bool:
    try:
        name_blob = tf.norm(tf.card_name(card))
        subtype_blob = tf.norm(str(card.get("subtypes") or card.get("subtype") or ""))
        return bool(tf.is_trainer(card) and ("item" in subtype_blob or "item" in name_blob))
    except Exception:
        return False


def _turn1_goal_search_filter_card_is_stadium(card: Dict[str, Any]) -> bool:
    try:
        blob = tf.norm(str(card.get("subtypes") or card.get("subtype") or "") + " " + tf.card_name(card))
        return bool(tf.is_trainer(card) and "stadium" in blob)
    except Exception:
        return False


def _turn1_search_filter_allows_for_action(filt: Dict[str, Any], card: Dict[str, Any], action_name: str) -> bool:
    """Goal-aware search filter used by the goal-search executor.

    TURN1_V66_GOAL_SEARCH_OR_FILTERS

    Direct replacement for the old sequential raw-text gate. The old version
    returned as soon as it saw phrases like "basic energy", which broke OR
    filters such as Fighting Gong:

        Basic Fighting Energy card OR Basic Fighting Pokemon

    The executor must evaluate the full target expression and allow a card if
    it satisfies ANY printed/compiled search branch.
    """
    action_n = tf.norm(action_name)
    raw = _turn1_goal_search_filter_norm_text(filt)

    def _n(value: Any) -> str:
        try:
            return tf.norm(value)
        except Exception:
            return str(value or "").lower().strip()

    def _super(c: Dict[str, Any]) -> str:
        return _n(tf.card_supertype(c))

    def _subs(c: Dict[str, Any]) -> set:
        try:
            return {_n(x) for x in tf.card_subtypes(c)}
        except Exception:
            vals = []
            for obj in (c, c.get("identity") if isinstance(c, dict) else None):
                if isinstance(obj, dict):
                    v = obj.get("subtypes") or obj.get("subtype") or obj.get("trainerType")
                    if isinstance(v, list):
                        vals.extend(v)
                    elif v:
                        vals.append(v)
            return {_n(x) for x in vals}

    def _types(c: Dict[str, Any]) -> set:
        try:
            vals = list(tf.card_types(c))
        except Exception:
            vals = []
            for obj in (c, c.get("identity") if isinstance(c, dict) else None):
                if isinstance(obj, dict):
                    v = obj.get("types") or obj.get("type")
                    if isinstance(v, list):
                        vals.extend(v)
                    elif v:
                        vals.append(v)
        return {_n(x) for x in vals}

    type_aliases = {
        "grass": {"grass", "g"},
        "fire": {"fire", "r"},
        "water": {"water", "w"},
        "lightning": {"lightning", "electric", "l"},
        "psychic": {"psychic", "p"},
        "fighting": {"fighting", "f"},
        "darkness": {"darkness", "dark", "d"},
        "metal": {"metal", "steel", "m"},
        "colorless": {"colorless", "c"},
    }

    def _is_pokemon(c: Dict[str, Any]) -> bool:
        return _super(c) in {"pokemon", "pokémon"}

    def _is_trainer(c: Dict[str, Any]) -> bool:
        return _super(c) == "trainer"

    def _is_energy(c: Dict[str, Any]) -> bool:
        try:
            return bool(tf.is_energy(c))
        except Exception:
            return _super(c) == "energy"

    def _is_basic_pokemon(c: Dict[str, Any]) -> bool:
        try:
            return bool(tf.is_basic_pokemon(c))
        except Exception:
            return _is_pokemon(c) and "basic" in _subs(c)

    def _is_basic_energy(c: Dict[str, Any]) -> bool:
        if not _is_energy(c):
            return False
        return "basic" in _subs(c) or _n(tf.card_name(c)).startswith("basic ")

    def _has_type(c: Dict[str, Any], typ: str) -> bool:
        aliases = type_aliases.get(_n(typ), {_n(typ)})
        card_types = _types(c)
        name_n = _n(tf.card_name(c))
        return any(alias in card_types or alias in name_n for alias in aliases)

    def _mentions_typed(noun: str, typ: str) -> bool:
        aliases = type_aliases.get(_n(typ), {_n(typ)})
        noun_options = {noun, "pokémon" if noun == "pokemon" else noun}
        return any(f"{alias} {n}" in raw for alias in aliases for n in noun_options)

    def _trainer_kind(c: Dict[str, Any], kind: str) -> bool:
        if not _is_trainer(c):
            return False
        k = _n(kind)
        return k in _subs(c) or k in _n(tf.card_name(c))

    def _hp(c: Dict[str, Any]):
        vals = []
        if isinstance(c, dict):
            vals.extend([c.get("hp"), c.get("raw_hp")])
            for key in ("identity", "gameplay", "raw_card", "source"):
                obj = c.get(key)
                if isinstance(obj, dict):
                    vals.extend([obj.get("hp"), obj.get("raw_hp")])
        for v in vals:
            if v is None:
                continue
            m = re.search(r"\d+", str(v))
            if m:
                try:
                    return int(m.group(0))
                except Exception:
                    pass
        return None

    def _hp_ok(c: Dict[str, Any]) -> bool:
        m = re.search(r"(\d+)\s*hp\s*or\s*less", raw)
        if not m:
            return True
        h = _hp(c)
        return h is not None and h <= int(m.group(1))

    # Explicit known cards/one-off costs.
    if action_n == "ultra ball":
        return _is_pokemon(card)

    # Meowth ex / Last-Ditch Catch is not a direct goal search. It searches a
    # Supporter and may then play that Supporter through the original executor.
    if tf.is_meowth_ex({"name": action_name}) or action_n == "meowth ex" or "last ditch" in raw:
        return bool(tf.is_supporter(card))

    tests = []

    # Typed Energy / Pokémon branches. These intentionally accumulate instead
    # of returning early, because many cards print OR filters.
    for typ in type_aliases:
        if _mentions_typed("energy", typ):
            if "basic" in raw:
                tests.append(lambda c, typ=typ: _is_basic_energy(c) and _has_type(c, typ))
            else:
                tests.append(lambda c, typ=typ: _is_energy(c) and _has_type(c, typ))

        if _mentions_typed("pokemon", typ):
            if "basic" in raw:
                tests.append(lambda c, typ=typ: _is_basic_pokemon(c) and _has_type(c, typ) and _hp_ok(c))
            else:
                tests.append(lambda c, typ=typ: _is_pokemon(c) and _has_type(c, typ) and _hp_ok(c))

    # Generic Energy / Pokémon branches. Avoid letting Pokémon Tool text become
    # a generic Pokémon search.
    trainer_words = {"supporter", "item", "stadium", "tool", "pokemon tool", "pokémon tool"}

    if "basic energy" in raw and not any(_mentions_typed("energy", typ) for typ in type_aliases):
        tests.append(lambda c: _is_basic_energy(c))
    elif "energy" in raw and not any(_mentions_typed("energy", typ) for typ in type_aliases):
        tests.append(lambda c: _is_energy(c))

    if "pokemon ex" in raw or "pokémon ex" in raw:
        tests.append(lambda c: _is_pokemon(c) and "ex" in _n(tf.card_name(c)))

    has_typed_pokemon = any(_mentions_typed("pokemon", typ) for typ in type_aliases)
    if ("basic pokemon" in raw or "basic pokémon" in raw) and not has_typed_pokemon:
        tests.append(lambda c: _is_basic_pokemon(c) and _hp_ok(c))
    elif ("pokemon" in raw or "pokémon" in raw) and not has_typed_pokemon and not any(w in raw for w in trainer_words | {"energy"}):
        tests.append(lambda c: _is_pokemon(c) and _hp_ok(c))

    # Trainer branches. These also accumulate for filters like Irida-style
    # "Water Pokémon or Item" effects.
    if "supporter" in raw:
        tests.append(lambda c: bool(tf.is_supporter(c)))
    if "stadium" in raw:
        tests.append(lambda c: _turn1_goal_search_filter_card_is_stadium(c))
    if "item" in raw and "pokemon tool" not in raw and "pokémon tool" not in raw:
        tests.append(lambda c: _turn1_goal_search_filter_card_is_item(c))
    if "pokemon tool" in raw or "pokémon tool" in raw or "tool" in raw:
        tests.append(lambda c: _trainer_kind(c, "tool") or _trainer_kind(c, "pokemon tool") or _trainer_kind(c, "pokémon tool"))
    if "trainer" in raw and not any(w in raw for w in trainer_words):
        tests.append(lambda c: _is_trainer(c))

    if tests:
        return any(test(card) for test in tests)

    # Fallback to the original goal-search / target-finder compiled filter behavior.
    return _ORIG_SEARCH_FILTER_ALLOWS_FOR_ACTION_BEFORE_V23(filt, card, action_name)


def _turn1_original_single_target_score_for_goal(
    card: Dict[str, Any],
    st: tf.SimState,
    reqs: Sequence[GoalRequirement],
    going: str,
    enable_chain_search: bool,
) -> float:
    best = -1.0
    for tn in all_goal_norms(reqs):
        try:
            best = max(best, tf.score_playable_card(card, st, tn, going, enable_chain_search))
        except Exception:
            pass
    return best


def _turn1_score_playable_card_for_goal(
    card: Dict[str, Any],
    st: tf.SimState,
    reqs: Sequence[GoalRequirement],
    mode: str,
    tracker: GoalTracker,
    going: str,
    enable_chain_search: bool,
) -> float:
    # Meowth ex should not be scored as a direct multi-goal searcher. It only
    # searches Supporters. The original target-finder handler already models
    # its legal Supporter chain and correctly blocks it going first.
    try:
        if tf.is_meowth_ex(card):
            return _turn1_original_single_target_score_for_goal(card, st, reqs, going, enable_chain_search)
    except Exception:
        pass

    return _ORIG_SCORE_PLAYABLE_CARD_FOR_GOAL_BEFORE_V23(
        card,
        st,
        reqs,
        mode,
        tracker,
        going,
        enable_chain_search,
    )


def _turn1_execute_goal_search_card(
    st: tf.SimState,
    card: Dict[str, Any],
    rng: random.Random,
    reqs: Sequence[GoalRequirement],
    mode: str,
    tracker: GoalTracker,
    going: str,
    enable_chain_search: bool,
) -> bool:
    # Critical: do not let the goal-search executor directly turn Meowth's Supporter search into
    # missing goal Pokémon. Delegate to the original executor instead.
    try:
        if tf.is_meowth_ex(card):
            return False
    except Exception:
        pass

    return _ORIG_EXECUTE_GOAL_SEARCH_CARD_BEFORE_V23(
        st,
        card,
        rng,
        reqs,
        mode,
        tracker,
        going,
        enable_chain_search,
    )




# ---------------------------------------------------------------------
# TURN1_SELF_ONLY_SEARCH_GUARD
# ---------------------------------------------------------------------
# General fix for opponent-disruption cards being treated as self-search.
#
# The multi-goal executor must only credit search/draw-like steps that affect
# OUR deck/hand. Some compiled cards contain search/draw-looking operations
# derived from opponent-disruption text. Those should not become Turn-1 access
# paths for our goal cards.
#
# This is deliberately not a Special Red Card one-off. It filters by effect text:
# - self search: search your deck / your discard / your hand etc.
# - self draw: draw cards for you / each player draws / shuffle your hand and draw
# - opponent-only hand/deck/discard/bench/active text: not a self consistency tool
#
# The card still remains in the deck and can be drawn or used as discard fodder.
# It is only blocked from being chosen as a played action because of opponent-only
# compiled search/draw artifacts.

_TURN1_ORIG_SEARCH_STEPS_FOR_CARD_BEFORE_SELF_ONLY_GUARD = _turn1_search_steps_for_card
_TURN1_ORIG_SCORE_PLAYABLE_CARD_FOR_GOAL_BEFORE_SELF_ONLY_GUARD = _turn1_score_playable_card_for_goal
_TURN1_ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_BEFORE_SELF_ONLY_GUARD = score_candidate_for_missing_targets
_TURN1_ORIG_TF_CARD_DIRECTLY_SEARCHES_TARGET_BEFORE_SELF_ONLY_GUARD = tf.card_directly_searches_target
_TURN1_ORIG_TF_CARD_HAS_SEARCH_BEFORE_SELF_ONLY_GUARD = tf.card_has_search
_TURN1_ORIG_TF_CARD_DRAW_POWER_BEFORE_SELF_ONLY_GUARD = tf.card_draw_power

try:
    _TURN1_ORIG_TF_ABILITY_DIRECTLY_SEARCHES_TARGET_BEFORE_SELF_ONLY_GUARD = tf.ability_directly_searches_target
except Exception:
    _TURN1_ORIG_TF_ABILITY_DIRECTLY_SEARCHES_TARGET_BEFORE_SELF_ONLY_GUARD = None


def _turn1_self_only_search_add_text_parts(value: Any, parts: List[str], depth: int = 0) -> None:
    if value is None or depth > 5:
        return
    if isinstance(value, str):
        if value.strip():
            parts.append(value)
        return
    if isinstance(value, dict):
        for k, v in value.items():
            # Include keys too because compiler metadata can use keys like
            # opponent, target, recipient, destination, etc.
            if isinstance(k, str):
                parts.append(k)
            _turn1_self_only_search_add_text_parts(v, parts, depth + 1)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _turn1_self_only_search_add_text_parts(item, parts, depth + 1)
        return


def _turn1_self_only_search_norm_text(value: Any) -> str:
    parts: List[str] = []
    _turn1_self_only_search_add_text_parts(value, parts)
    raw = " ".join(parts)
    try:
        return tf.norm(raw)
    except Exception:
        return str(raw or "").lower()


def _turn1_self_only_search_step_text(step: Dict[str, Any], card: Optional[Dict[str, Any]] = None) -> str:
    parts: List[str] = []

    if isinstance(step, dict):
        # Prioritize human/printed text and filter text. Then include the rest
        # as metadata fallback.
        for key in (
            "source_text",
            "text",
            "raw_text",
            "effect_text",
            "filter",
            "recipient",
            "target",
            "from_zone",
            "to_zone",
            "owner",
            "player",
        ):
            if key in step:
                _turn1_self_only_search_add_text_parts(step.get(key), parts)
        _turn1_self_only_search_add_text_parts(step, parts)

    if isinstance(card, dict):
        for key in (
            "combined_text",
            "rules",
            "abilities_text",
            "attacks_text",
            "raw_abilities",
            "raw_attacks",
            "text",
            "source",
            "gameplay",
        ):
            if key in card:
                _turn1_self_only_search_add_text_parts(card.get(key), parts)

    return _turn1_self_only_search_norm_text(" ".join(parts))


def _turn1_text_mentions_opponent_zone(text: str) -> bool:
    t = str(text or "")
    opponent_hints = (
        "your opponent",
        "opponents",
        "opponent s",
        "opponent's",
        "opponent’s",
        "opponent hand",
        "opponent deck",
        "opponent discard",
        "opponent active",
        "opponent bench",
        "opponent benched",
    )
    return any(h in t for h in opponent_hints)


def _turn1_text_mentions_self_search(text: str) -> bool:
    t = str(text or "")
    self_search_hints = (
        "search your deck",
        "search your discard",
        "search your hand",
        "from your deck",
        "from your discard",
        "from your hand",
        "your deck for",
        "look at the top",
        "put it into your hand",
        "put them into your hand",
        "put those cards into your hand",
        "put that card into your hand",
        "put up to",
    )
    return any(h in t for h in self_search_hints)


def _turn1_text_mentions_self_draw(text: str) -> bool:
    t = str(text or "")
    if "each player" in t and "draw" in t:
        return True
    self_draw_hints = (
        "draw a card",
        "draw 1 card",
        "draw cards",
        "draw 2 cards",
        "draw 3 cards",
        "draw 4 cards",
        "draw 5 cards",
        "draw 6 cards",
        "draw 7 cards",
        "draw 8 cards",
        "until you have",
        "shuffle your hand into your deck and draw",
        "shuffle your hand into your deck then draw",
        "shuffle your hand into your deck. draw",
    )
    return any(h in t for h in self_draw_hints)


def _turn1_text_is_opponent_only_draw_or_search(text: str) -> bool:
    t = str(text or "")
    if not _turn1_text_mentions_opponent_zone(t):
        return False

    opponent_action_hints = (
        "opponent reveals",
        "opponent shuffle",
        "opponent shuffles",
        "opponent draw",
        "opponent draws",
        "opponent hand",
        "opponent's hand",
        "opponent’s hand",
        "opponent deck",
        "opponent's deck",
        "opponent’s deck",
        "opponent discard",
        "opponent's discard",
        "opponent’s discard",
        "look at your opponent",
        "choose a card from your opponent",
    )

    if not any(h in t for h in opponent_action_hints):
        return False

    # If the same printed effect also clearly benefits us, do not call it
    # opponent-only. Example: each player draw effects.
    if _turn1_text_mentions_self_search(t) or _turn1_text_mentions_self_draw(t):
        return False

    return True


def _turn1_is_self_search_step(step: Dict[str, Any], card: Optional[Dict[str, Any]] = None) -> bool:
    if not isinstance(step, dict):
        return False
    if step.get("op") not in {"search_deck", "choose_cards", "put_card_into_hand"}:
        return False

    text = _turn1_self_only_search_step_text(step, card)

    if _turn1_text_is_opponent_only_draw_or_search(text):
        return False

    if _turn1_text_mentions_opponent_zone(text) and not _turn1_text_mentions_self_search(text):
        return False

    if _turn1_text_mentions_self_search(text):
        return True

    # If the compiler gave a structured search_deck op with no opponent text,
    # allow it. This preserves normal compiled self-search effects that have
    # sparse metadata.
    return not _turn1_text_mentions_opponent_zone(text)


def _turn1_is_self_draw_step(step: Dict[str, Any], card: Optional[Dict[str, Any]] = None) -> bool:
    if not isinstance(step, dict):
        return False
    if step.get("op") not in {"draw_cards", "draw_cards_per_coin_heads", "draw_until_hand_size", "draw_until_hand_size_matches"}:
        return False

    text = _turn1_self_only_search_step_text(step, card)

    if _turn1_text_is_opponent_only_draw_or_search(text):
        return False

    if _turn1_text_mentions_opponent_zone(text) and not _turn1_text_mentions_self_draw(text):
        return False

    return True


def _turn1_self_search_steps_for_card(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(card, dict):
        return out

    for eff in tf.iter_effects(card):
        if tf.effect_is_trivial_rule(eff):
            continue
        for step in tf.flatten_steps(eff):
            if isinstance(step, dict) and step.get("op") == "search_deck":
                if _turn1_is_self_search_step(step, card):
                    out.append(step)
                else:
                    try:
                        # This log only fires later if the card is in an actual state;
                        # here we just keep filtering pure.
                        pass
                    except Exception:
                        pass
    return out


def _turn1_search_steps_for_card(card: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Only self-search steps can feed the multi-goal executor."""
    return _turn1_self_search_steps_for_card(card)


def _turn1_card_directly_searches_target_self_only(card: Dict[str, Any], target_norm: str, deck: Sequence[Dict[str, Any]]) -> bool:
    target_cards = [c for c in deck if tf.target_matches(c, target_norm)]
    if not target_cards:
        return False

    # Preserve known explicit card-specific searchers, but do not let
    # opponent-only compiled effects through the generic loop.
    try:
        if tf.card_specific_directly_searches_target(card, target_norm, deck):
            return True
    except Exception:
        pass

    for step in tf.flatten_steps(list(tf.iter_effects(card))):
        if not isinstance(step, dict):
            continue
        if step.get("op") in {"search_deck", "choose_cards", "put_card_into_hand"}:
            if not _turn1_is_self_search_step(step, card):
                continue
            filt = tf.extract_filter(step)
            if any(tf.filter_allows_card(filt, tc) for tc in target_cards):
                return True

    return False


def _turn1_card_has_self_search(card: Dict[str, Any]) -> bool:
    return bool(_turn1_self_search_steps_for_card(card))


def _turn1_card_self_draw_power(card: Dict[str, Any]) -> int:
    total = 0
    if not isinstance(card, dict):
        return 0

    for step in tf.flatten_steps(list(tf.iter_effects(card))):
        if not isinstance(step, dict):
            continue
        op = step.get("op")
        if op in {"draw_cards", "draw_cards_per_coin_heads"}:
            if _turn1_is_self_draw_step(step, card):
                total += tf.draw_amount_from_step(step, coin_heads=1)
        elif op == "draw_until_hand_size":
            if _turn1_is_self_draw_step(step, card):
                total += tf.amount_value(step.get("target_hand_size"), default=0)
        elif op == "draw_until_hand_size_matches":
            if _turn1_is_self_draw_step(step, card):
                total += 3

    return total


def _turn1_ability_directly_searches_target_self_only(effect: Dict[str, Any], target_norm: str, deck: Sequence[Dict[str, Any]]) -> bool:
    target_cards = [c for c in deck if tf.target_matches(c, target_norm)]
    if not target_cards:
        return False

    for step in tf.flatten_steps(effect):
        if not isinstance(step, dict):
            continue
        if step.get("op") in {"search_deck", "choose_cards", "put_card_into_hand"}:
            if not _turn1_is_self_search_step(step, None):
                continue
            filt = tf.extract_filter(step)
            if any(tf.filter_allows_card(filt, tc) for tc in target_cards):
                return True

    return False


# Patch the imported single-target module too, because goal_finder falls back
# to tf.score_playable_card for draw/special/chain lines.
tf.card_directly_searches_target = _turn1_card_directly_searches_target_self_only
tf.card_has_search = _turn1_card_has_self_search
tf.card_draw_power = _turn1_card_self_draw_power
if _TURN1_ORIG_TF_ABILITY_DIRECTLY_SEARCHES_TARGET_BEFORE_SELF_ONLY_GUARD is not None:
    tf.ability_directly_searches_target = _turn1_ability_directly_searches_target_self_only


def _turn1_action_source_card_for_self_only_guard(action: Any) -> Optional[Dict[str, Any]]:
    if isinstance(action, dict) and action.get("_virtual_action"):
        for key in ("card", "source", "search_card"):
            c = action.get(key)
            if isinstance(c, dict):
                return c
        return None
    if isinstance(action, dict):
        return action
    return None


def _turn1_is_opponent_only_consistency_action(action: Any) -> bool:
    card = _turn1_action_source_card_for_self_only_guard(action)
    if not isinstance(card, dict):
        return False

    any_access_like = False
    any_self_access = False
    any_opponent_only = False

    for eff in tf.iter_effects(card):
        if tf.effect_is_trivial_rule(eff):
            continue
        for step in tf.flatten_steps(eff):
            if not isinstance(step, dict):
                continue
            op = step.get("op")
            if op in {"search_deck", "choose_cards", "put_card_into_hand"}:
                any_access_like = True
                if _turn1_is_self_search_step(step, card):
                    any_self_access = True
                else:
                    any_opponent_only = True
            elif op in {"draw_cards", "draw_cards_per_coin_heads", "draw_until_hand_size", "draw_until_hand_size_matches"}:
                any_access_like = True
                if _turn1_is_self_draw_step(step, card):
                    any_self_access = True
                else:
                    any_opponent_only = True

    # If a card only has opponent-only access-like artifacts, it must not be
    # chosen as a played line. If it has no access-like ops, leave it alone so
    # explicit target_finder special cases can still handle it.
    return bool(any_access_like and any_opponent_only and not any_self_access)


def _turn1_score_playable_card_for_goal(
    card: Dict[str, Any],
    st: tf.SimState,
    reqs: Sequence[GoalRequirement],
    mode: str,
    tracker: GoalTracker,
    going: str,
    enable_chain_search: bool,
) -> float:
    if _turn1_is_opponent_only_consistency_action(card):
        return -1.0
    return _TURN1_ORIG_SCORE_PLAYABLE_CARD_FOR_GOAL_BEFORE_SELF_ONLY_GUARD(
        card,
        st,
        reqs,
        mode,
        tracker,
        going,
        enable_chain_search,
    )


def score_candidate_for_missing_targets(
    st: tf.SimState,
    missing: Sequence[GoalRequirement],
    going: str,
    enable_chain_search: bool,
) -> List[Tuple[float, Any, str]]:
    # TURN1_ACTION_BUDGET_GUARD
    if _turn1_action_budget_exhausted(st):
        return []
    scored = _TURN1_ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_BEFORE_SELF_ONLY_GUARD(
        st,
        missing,
        going,
        enable_chain_search,
    )

    filtered: List[Tuple[float, Any, str]] = []
    for score, action, target_norm in scored:
        if _turn1_is_opponent_only_consistency_action(action):
            try:
                st.log.append({
                    "event": "blocked_opponent_only_consistency_action",
                    "card": action_label(action),
                    "reason": "effect text affects opponent zones, not our deck/hand access",
                })
            except Exception:
                pass
            continue
        filtered.append((score, action, target_norm))

    return filtered




# ---------------------------------------------------------------------
# TURN1_ABILITY_REQUIREMENT_AND_FILTER_GUARD_V27
# ---------------------------------------------------------------------
# Root issue fixed here:
# The Turn-1 goal finder reuses broad single-target action scorers. Some
# abilities/effects that search a restricted class of cards, such as
# Tatsugiri's Attract Customers searching only Supporter cards, were being
# credited as if they could directly search any missing goal card.
#
# This guard is intentionally general:
# - active-only abilities require the source Pokemon to actually be Active
# - bench-only abilities require the source Pokemon to actually be on Bench
# - typed/restricted searches may only help compatible goal cards
#   e.g. Supporter search -> Supporter goal only
#        Basic Pokemon search -> Basic Pokemon goal only
#        Energy search -> Energy goal only
# - opponent-only disruption guards from earlier patches remain compatible

import re as _turn1_v27_re


def turn1_v27_flatten_text(obj: Any, limit: int = 20000) -> str:
    parts: List[str] = []

    def walk(x: Any) -> None:
        if sum(len(p) for p in parts) > limit:
            return
        if x is None:
            return
        if isinstance(x, str):
            parts.append(x)
            return
        if isinstance(x, (int, float, bool)):
            return
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str):
                    parts.append(k)
                walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                walk(v)
            return

    walk(obj)
    return " ".join(parts)


def turn1_v27_norm_blob(s: Any) -> str:
    text = str(s or "").lower()
    text = text.replace("pokémon", "pokemon")
    text = text.replace("’", "'")
    text = _turn1_v27_re.sub(r"\s+", " ", text)
    return text.strip()


def turn1_v27_card_blob(card: Any) -> str:
    return turn1_v27_norm_blob(turn1_v27_flatten_text(card))


def turn1_v27_action_source_and_effect(action: Any) -> Tuple[Any, Any, str]:
    if isinstance(action, dict):
        va = action.get("_virtual_action") or ""
        source = action.get("source") or action.get("card") or action.get("search_card")
        effect = action.get("effect") or action.get("compiled_effect") or action.get("ability")
        return source, effect, str(va)
    return action, action, "PlayCard"


def turn1_v27_source_in_list(source: Any, cards: Sequence[Any]) -> bool:
    if source is None:
        return False
    source_id = None
    try:
        source_id = source.get("_instance_id") if isinstance(source, dict) else None
    except Exception:
        source_id = None
    for c in cards or []:
        if c is source:
            return True
        if source_id and isinstance(c, dict) and c.get("_instance_id") == source_id:
            return True
    return False


def turn1_v27_requires_active(blob: str) -> bool:
    b = turn1_v27_norm_blob(blob)
    patterns = [
        "if this pokemon is in the active spot",
        "if this pokemon is your active pokemon",
        "if this pokemon is active",
        "while this pokemon is in the active spot",
        "as long as this pokemon is in the active spot",
        "this pokemon is in the active spot",
    ]
    return any(p in b for p in patterns)


def turn1_v27_requires_bench(blob: str) -> bool:
    b = turn1_v27_norm_blob(blob)
    patterns = [
        "if this pokemon is on your bench",
        "while this pokemon is on your bench",
        "as long as this pokemon is on your bench",
        "this pokemon is on your bench",
    ]
    return any(p in b for p in patterns)


def turn1_v27_position_allowed(st: Any, action: Any) -> bool:
    source, effect, va = turn1_v27_action_source_and_effect(action)
    blob = turn1_v27_norm_blob(turn1_v27_flatten_text(effect) + " " + turn1_v27_flatten_text(source))

    if not blob:
        return True

    # If an ability says Active Spot, benching the Pokemon from hand is not
    # enough. It must already be the actual Active Pokemon.
    if turn1_v27_requires_active(blob):
        return source is not None and st.active is not None and turn1_v27_source_in_list(source, [st.active])

    if turn1_v27_requires_bench(blob):
        return source is not None and turn1_v27_source_in_list(source, list(getattr(st, "bench", []) or []))

    return True


def turn1_v27_card_categories(card: Any) -> set:
    cats = set()
    if not isinstance(card, dict):
        return cats

    blob = turn1_v27_card_blob(card)

    identity = card.get("identity") or {}
    supertype = turn1_v27_norm_blob(identity.get("supertype") or card.get("supertype") or "")
    subtypes_raw = identity.get("subtypes") or card.get("subtypes") or card.get("subtype") or []
    if isinstance(subtypes_raw, str):
        sub_blob = turn1_v27_norm_blob(subtypes_raw)
    else:
        sub_blob = turn1_v27_norm_blob(" ".join(str(x) for x in subtypes_raw))

    if "pokemon" in supertype or "supertype pokemon" in blob:
        cats.add("pokemon")
    if "trainer" in supertype or "supertype trainer" in blob:
        cats.add("trainer")
    if "energy" in supertype or "supertype energy" in blob:
        cats.add("energy")

    try:
        if tf.is_basic_pokemon(card):
            cats.add("pokemon")
            cats.add("basic_pokemon")
    except Exception:
        pass

    if "basic" in sub_blob and "pokemon" in cats:
        cats.add("basic_pokemon")
    if "supporter" in sub_blob or "subtypes supporter" in blob or "supporter" in blob:
        cats.add("supporter")
        cats.add("trainer")
    if "item" in sub_blob or "subtypes item" in blob:
        cats.add("item")
        cats.add("trainer")
    if "stadium" in sub_blob or "subtypes stadium" in blob:
        cats.add("stadium")
        cats.add("trainer")
    if "tool" in sub_blob or "pokemon tool" in sub_blob or "subtypes tool" in blob:
        cats.add("tool")
        cats.add("trainer")
    if "basic" in sub_blob and "energy" in cats:
        cats.add("basic_energy")

    name = turn1_v27_norm_blob(tf.card_name(card))
    if "basic" in name and "energy" in name:
        cats.add("energy")
        cats.add("basic_energy")

    return cats


def turn1_v27_effect_search_categories(blob: str) -> set:
    """
    Infer the card class an effect is allowed to search/reveal/put into hand.
    Empty set means either not a restricted search, or parser cannot infer safely.
    """
    b = turn1_v27_norm_blob(blob)
    cats = set()

    # Only categorize effects that actually look like access/search effects.
    access_words = [
        "search your deck",
        "look at the top",
        "reveal",
        "put it into your hand",
        "put into your hand",
        "put them into your hand",
        "put onto your bench",
        "put them onto your bench",
    ]
    if not any(w in b for w in access_words):
        return cats

    # Opponent-only text should be handled by earlier guards, but never treat it
    # as our goal search here.
    if ("your opponent" in b or "opponent's" in b or "their hand" in b or "their deck" in b) and not ("your deck" in b or "your hand" in b):
        return {"opponent_only"}

    if "supporter" in b:
        cats.add("supporter")
    if "basic pokemon" in b or "basic pokemon card" in b:
        cats.add("basic_pokemon")
    elif "pokemon" in b and "opponent's pokemon" not in b and "your opponent's pokemon" not in b:
        cats.add("pokemon")
    if "basic energy" in b:
        cats.add("basic_energy")
        cats.add("energy")
    elif "energy" in b and "opponent" not in b:
        cats.add("energy")
    if "item" in b:
        cats.add("item")
        cats.add("trainer")
    if "stadium" in b:
        cats.add("stadium")
        cats.add("trainer")
    if "pokemon tool" in b or " tool card" in b:
        cats.add("tool")
        cats.add("trainer")
    if "trainer" in b:
        cats.add("trainer")

    return cats


def turn1_v27_categories_compatible(search_cats: set, card_cats: set) -> bool:
    if not search_cats:
        return True
    if "opponent_only" in search_cats:
        return False

    # Exact/specific categories first.
    if "supporter" in search_cats:
        return "supporter" in card_cats
    if "basic_pokemon" in search_cats:
        return "basic_pokemon" in card_cats
    if "pokemon" in search_cats:
        return "pokemon" in card_cats
    if "basic_energy" in search_cats:
        return "basic_energy" in card_cats
    if "energy" in search_cats:
        return "energy" in card_cats
    if "item" in search_cats:
        return "item" in card_cats
    if "stadium" in search_cats:
        return "stadium" in card_cats
    if "tool" in search_cats:
        return "tool" in card_cats
    if "trainer" in search_cats:
        return "trainer" in card_cats

    return True


# ---------------------------------------------------------------------
# TURN1_V62_PRECISE_SEARCH_FILTER_COMPAT
# ---------------------------------------------------------------------
# Direct fix, not a wrapper:
# Replace the old coarse category-only compatibility check with a precise
# check that asks: can this action's actual search filter select the target
# card that would satisfy the current goal?
#
# The old code could mark broad Pokémon search cards as incompatible with a
# named Pokémon goal. This is wrong for effects such as:
#   - Poké Pad: Pokémon without a Rule Box
#   - Fighting Gong: Basic Fighting Energy or Basic Fighting Pokémon
#
# The new flow:
#   1. Resolve target_norm to actual card(s) in the current state/deck pools.
#   2. Extract the action's real search steps/filters.
#   3. Use the same filter function used by execution:
#        _turn1_search_filter_allows_for_action(...)
#   4. Fall back to explicit text/category matching only when structured
#      filters are unavailable.


def _turn1_v62_norm(value: Any) -> str:
    try:
        return turn1_v27_norm_blob(str(value or ""))
    except Exception:
        return str(value or "").lower().strip()


def _turn1_v62_flatten_strings(obj: Any, max_items: int = 4000) -> str:
    out: List[str] = []
    seen = set()

    def rec(x: Any) -> None:
        if len(out) >= max_items:
            return
        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)

        if isinstance(x, str):
            if x.strip():
                out.append(x)
            return
        if isinstance(x, dict):
            for v in x.values():
                rec(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)
            return

    rec(obj)
    return " ".join(out)


def _turn1_v62_card_name(card: Any) -> str:
    try:
        return tf.card_name(card)
    except Exception:
        pass
    if isinstance(card, dict):
        ident = card.get("identity") or {}
        return str(card.get("name") or card.get("card_name") or ident.get("name") or ident.get("canonical_name") or "")
    return str(card or "")


def _turn1_v62_card_supertype(card: Any) -> str:
    try:
        return _turn1_v62_norm(tf.card_supertype(card))
    except Exception:
        pass
    if isinstance(card, dict):
        ident = card.get("identity") or {}
        return _turn1_v62_norm(card.get("supertype") or ident.get("supertype") or "")
    return ""


def _turn1_v62_card_subtypes(card: Any) -> set:
    vals = []
    if isinstance(card, dict):
        ident = card.get("identity") or {}
        for key in ["subtypes", "subtype"]:
            v = card.get(key) if key in card else ident.get(key)
            if isinstance(v, str):
                vals.append(v)
            elif isinstance(v, (list, tuple, set)):
                vals.extend(v)
    return {_turn1_v62_norm(v) for v in vals if str(v or "").strip()}


def _turn1_v62_card_types(card: Any) -> set:
    vals = []
    if isinstance(card, dict):
        ident = card.get("identity") or {}
        for key in ["types", "type"]:
            v = card.get(key) if key in card else ident.get(key)
            if isinstance(v, str):
                vals.append(v)
            elif isinstance(v, (list, tuple, set)):
                vals.extend(v)
    return {_turn1_v62_norm(v) for v in vals if str(v or "").strip()}


def _turn1_v62_card_hp(card: Any) -> int:
    if isinstance(card, dict):
        ident = card.get("identity") or {}
        for key in ["hp", "HP"]:
            v = card.get(key) if key in card else ident.get(key)
            try:
                return int(v)
            except Exception:
                pass
    return 0


def _turn1_v62_card_has_rule_box(card: Any) -> bool:
    blob = _turn1_v62_norm(_turn1_v62_flatten_strings(card))
    name = _turn1_v62_norm(_turn1_v62_card_name(card))
    # Conservative: cards with explicit Rule Box text or common rule-box suffixes.
    return (
        "rule box" in blob
        or " pokemon ex" in (" " + name)
        or name.endswith(" ex")
        or " pokemon v" in (" " + name)
        or name.endswith(" v")
        or "vstar" in name
        or "vmax" in name
    )


def _turn1_v62_target_cards_for_norm(st: Any, target_norm: str) -> List[Dict[str, Any]]:
    pools: List[Any] = []
    for zone in ["deck", "hand", "discard", "bench", "prizes"]:
        try:
            pools.extend(list(getattr(st, zone, []) or []))
        except Exception:
            pass
    try:
        if getattr(st, "active", None) is not None:
            pools.append(st.active)
    except Exception:
        pass

    out: List[Dict[str, Any]] = []
    seen = set()
    for c in pools:
        if not isinstance(c, dict):
            continue
        try:
            ok = tf.target_matches(c, target_norm)
        except Exception:
            ok = _turn1_v62_norm(_turn1_v62_card_name(c)) == _turn1_v62_norm(target_norm)
        if ok:
            key = id(c)
            if key not in seen:
                out.append(c)
                seen.add(key)
    return out


def _turn1_v62_action_search_steps(action: Any) -> List[Any]:
    steps: List[Any] = []
    seen = set()

    def add_step(step: Any) -> None:
        sid = id(step)
        if sid in seen:
            return
        seen.add(sid)
        steps.append(step)

    try:
        source, effect, _va = turn1_v27_action_source_and_effect(action)
    except Exception:
        source, effect = action, None

    for obj in [effect, source, action]:
        if obj is None:
            continue
        if isinstance(obj, dict):
            # The existing simulator function knows how compiled search steps are stored.
            try:
                for s in _turn1_search_steps_for_card(obj):
                    add_step(s)
            except Exception:
                pass
            # Some call paths pass the effect step itself rather than the whole card.
            try:
                filt = tf.extract_filter(obj)
                if filt:
                    add_step(obj)
            except Exception:
                pass
        elif isinstance(obj, (list, tuple)):
            for x in obj:
                if isinstance(x, dict):
                    try:
                        filt = tf.extract_filter(x)
                        if filt:
                            add_step(x)
                    except Exception:
                        pass
    return steps


def _turn1_v62_action_blob(action: Any) -> str:
    try:
        source, effect, _va = turn1_v27_action_source_and_effect(action)
    except Exception:
        source, effect = action, None
    return _turn1_v62_norm(_turn1_v62_flatten_strings(effect) + " " + _turn1_v62_flatten_strings(source) + " " + _turn1_v62_flatten_strings(action))


def _turn1_v62_text_allows_target(action_blob: str, target_card: Dict[str, Any]) -> bool:
    """Fallback for text not represented by structured filters."""
    s = _turn1_v62_norm(action_blob)
    if not s:
        return False

    stype = _turn1_v62_card_supertype(target_card)
    subtypes = _turn1_v62_card_subtypes(target_card)
    types = _turn1_v62_card_types(target_card)
    hp = _turn1_v62_card_hp(target_card)
    is_pokemon = stype == "pokemon"
    is_energy = stype == "energy"
    is_trainer = stype == "trainer"
    is_basic = "basic" in subtypes
    no_rule_box = not _turn1_v62_card_has_rule_box(target_card)

    # Explicit negative/positional categories.
    if "opponent" in s and "your opponent" in s and not any(x in s for x in ["your deck", "your hand"]):
        return False

    # Rule-box constrained Pokémon search: Poké Pad style.
    if "pokemon" in s or "pokémon" in s:
        if "doesn't have a rule box" in s or "does not have a rule box" in s or "without a rule box" in s:
            if is_pokemon and no_rule_box:
                return True
            return False

    # HP-limited Basic Pokémon search: Buddy-Buddy Poffin style.
    hp_match = re.search(r"hp\s*(?:is\s*)?(\d+)\s*or\s*less", s)
    if hp_match and ("basic pokemon" in s or "basic pokémon" in s):
        try:
            lim = int(hp_match.group(1))
        except Exception:
            lim = 0
        return bool(is_pokemon and is_basic and hp and hp <= lim)

    # Type-specific Basic Pokémon / Energy search: Fighting Gong style.
    pokemon_type_words = {
        "grass": "grass",
        "fire": "fire",
        "water": "water",
        "lightning": "lightning",
        "psychic": "psychic",
        "fighting": "fighting",
        "darkness": "darkness",
        "metal": "metal",
        "colorless": "colorless",
    }
    for word, typ in pokemon_type_words.items():
        if (f"basic {word} pokemon" in s or f"basic {word} pokémon" in s) and is_pokemon and is_basic and typ in types:
            return True
        if f"basic {word} energy" in s and is_energy and is_basic and typ in types:
            return True
        if (f"{word} pokemon" in s or f"{word} pokémon" in s) and is_pokemon and typ in types:
            return True
        if f"{word} energy" in s and is_energy and typ in types:
            return True

    # Generic categories.
    if ("basic pokemon" in s or "basic pokémon" in s) and is_pokemon and is_basic:
        return True
    if ("pokemon" in s or "pokémon" in s) and is_pokemon:
        return True
    if "basic energy" in s and is_energy and is_basic:
        return True
    if "energy" in s and is_energy:
        return True

    if is_trainer:
        if "supporter" in s and "supporter" in subtypes:
            return True
        if "item" in s and "item" in subtypes:
            return True
        if "stadium" in s and "stadium" in subtypes:
            return True
        if "tool" in s and ("tool" in subtypes or "pokemon tool" in subtypes or "pokémon tool" in subtypes):
            return True
        if "trainer" in s:
            return True

    return False


def turn1_v27_target_compatible_with_action_filter(st: Any, target_norm: str, action: Any) -> bool:
    source, effect, va = turn1_v27_action_source_and_effect(action)
    blob = turn1_v27_norm_blob(turn1_v27_flatten_text(effect) + " " + turn1_v27_flatten_text(source))
    search_cats = turn1_v27_effect_search_categories(blob)

    if "opponent_only" in search_cats:
        return False

    # Resolve the proposed target to actual card(s) and use the same search
    # filter machinery that execution uses. This is the important v62 change.
    matching_cards = _turn1_v62_target_cards_for_norm(st, target_norm)
    steps = _turn1_v62_action_search_steps(action)
    action_name = action_label(action)

    if matching_cards and steps:
        for target_card in matching_cards:
            for step in steps:
                try:
                    filt = tf.extract_filter(step)
                except Exception:
                    filt = {}
                try:
                    if _turn1_search_filter_allows_for_action(filt or {}, target_card, action_name):
                        return True
                except Exception:
                    pass
                # Some compiled filters are incomplete; use printed/compiled text fallback.
                try:
                    if _turn1_v62_text_allows_target(_turn1_v62_action_blob(action), target_card):
                        return True
                except Exception:
                    pass
        return False

    if matching_cards:
        action_blob = _turn1_v62_action_blob(action)
        for target_card in matching_cards:
            if _turn1_v62_text_allows_target(action_blob, target_card):
                return True
        if search_cats:
            return any(turn1_v27_categories_compatible(search_cats, turn1_v27_card_categories(c)) for c in matching_cards)
        return True

    if not search_cats:
        return True

    # Conservative textual fallback for unresolved targets.
    tn = turn1_v27_norm_blob(target_norm)
    if "supporter" in search_cats:
        return "supporter" in tn
    if "basic_energy" in search_cats:
        return "energy" in tn and "basic" in tn
    if "energy" in search_cats:
        return "energy" in tn
    if "pokemon" in search_cats or "basic_pokemon" in search_cats:
        return not any(x in tn for x in ["supporter", "energy", "stadium", "item", "tool"])

    return True

def turn1_v27_action_allowed_for_goal(st: Any, action: Any, target_norm: str) -> bool:
    if not turn1_v27_position_allowed(st, action):
        try:
            st.log.append({
                "event": "blocked_illegal_ability_position_v27",
                "action": action_label(action),
                "target_norm": target_norm,
            })
        except Exception:
            pass
        return False

    if not turn1_v27_target_compatible_with_action_filter(st, target_norm, action):
        try:
            st.log.append({
                "event": "blocked_incompatible_search_filter_v27",
                "action": action_label(action),
                "target_norm": target_norm,
            })
        except Exception:
            pass
        return False

    return True


_ORIG_SCORE_CANDIDATE_FOR_MISSING_TARGETS_V27 = score_candidate_for_missing_targets

def score_candidate_for_missing_targets(
    st: tf.SimState,
    missing: Sequence[GoalRequirement],
    going: str,
    enable_chain_search: bool,
) -> List[Tuple[float, Any, str]]:
    # TURN1_ACTION_BUDGET_GUARD
    if _turn1_action_budget_exhausted(st):
        return []
    scored = _turn1_cached_score_candidates(st, missing, going, enable_chain_search)
    filtered: List[Tuple[float, Any, str]] = []

    for score, action, target_norm in scored:
        # Zero-score actions are known not to help the current missing goal and
        # should not keep feeding the chain-search planner.
        try:
            if score <= 0:
                continue
        except Exception:
            pass

        if turn1_v27_action_allowed_for_goal(st, action, target_norm):
            filtered.append((score, action, target_norm))

    return filtered


_ORIG_EXECUTE_ACTION_V27 = execute_action

def execute_action(st: tf.SimState, action: Any, target_norm: str, rng: random.Random, going: str, enable_chain_search: bool) -> None:
    # Safety net. Scoring should already filter bad actions. If an older patch or
    # future path bypasses scoring, do not execute illegal/typed-incompatible access.
    if not turn1_v27_action_allowed_for_goal(st, action, target_norm):
        # Count it as a consumed consideration so the loop cannot repeatedly pick
        # the same blocked action forever, but do not append it to st.line.
        st.actions_used += 1
        return
    _ORIG_EXECUTE_ACTION_V27(st, action, target_norm, rng, going, enable_chain_search)




# ---------------------------------------------------------------------
# TURN1_SEARCH_TARGET_TYPE_SYSTEM
# ---------------------------------------------------------------------
# Root fix for search effects:
# Search/draw-like effects must only be allowed to access cards matching their
# printed target restriction.
#
# Examples fixed:
#   Buddy-Buddy Poffin -> Basic Pokémon only, never Energy.
#   Irida -> Water Pokémon + Item only, never Basic Water Energy directly.
#   Shivery Chill -> Basic Water Energy only, not Pokémon.
#   Attract Customers -> Supporter only, not Pokémon/Energy.
#
# This patch does three things:
#   1. Patches tf.filter_allows_card so compiled search filters are stricter.
#   2. Replaces older over-broad post filters with this typed validator.
#   3. Wraps simulate_one_goal_trial at the end so final success rows cannot
#      include incompatible search/effect actions.

import re as _turn1_result_goal_filter_re
import unicodedata as _turn1_result_goal_filter_unicodedata


def turn1_result_goal_filter_norm(value):
    s = str(value or "")
    s = _turn1_result_goal_filter_unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not _turn1_result_goal_filter_unicodedata.combining(ch))
    s = s.lower().replace("’", "'").replace("`", "'")
    s = _turn1_result_goal_filter_re.sub(r"\s+", " ", s)
    return s.strip()


def turn1_result_goal_filter_card_name(card):
    try:
        return tf.card_name(card)
    except Exception:
        pass
    if isinstance(card, dict):
        ident = card.get("identity") or {}
        return str(
            card.get("name")
            or card.get("card_name")
            or ident.get("name")
            or ident.get("canonical_name")
            or card.get("card_id")
            or ""
        )
    return ""


def turn1_result_goal_filter_identity(card):
    if not isinstance(card, dict):
        return {}
    ident = card.get("identity") or {}
    return ident if isinstance(ident, dict) else {}


def turn1_result_goal_filter_supertype(card):
    if not isinstance(card, dict):
        return ""
    ident = turn1_result_goal_filter_identity(card)
    return str(card.get("supertype") or ident.get("supertype") or "")


def turn1_result_goal_filter_subtypes(card):
    vals = []
    if isinstance(card, dict):
        ident = turn1_result_goal_filter_identity(card)
        for src in (card, ident):
            for key in ("subtypes", "subtype", "trainerType"):
                v = src.get(key)
                if isinstance(v, (list, tuple, set)):
                    vals.extend(str(x) for x in v)
                elif v:
                    vals.append(str(v))
    return vals


def turn1_result_goal_filter_types(card):
    vals = []
    if isinstance(card, dict):
        ident = turn1_result_goal_filter_identity(card)
        for src in (card, ident):
            v = src.get("types")
            if isinstance(v, (list, tuple, set)):
                vals.extend(str(x) for x in v)
            elif v:
                vals.append(str(v))
    return vals


def turn1_result_goal_filter_card_classes(card):
    """Strict identity-based classes. Do not inspect attack/effect text here."""
    classes = set()
    st = turn1_result_goal_filter_norm(turn1_result_goal_filter_supertype(card))
    subs = {turn1_result_goal_filter_norm(x) for x in turn1_result_goal_filter_subtypes(card)}
    types = {turn1_result_goal_filter_norm(x) for x in turn1_result_goal_filter_types(card)}
    name = turn1_result_goal_filter_norm(turn1_result_goal_filter_card_name(card))

    if "pokemon" in st or "pokémon" in st:
        classes.add("pokemon")
        if "basic" in subs:
            classes.add("basic_pokemon")
        if "ex" in subs or name.endswith(" ex") or " ex " in name:
            classes.add("pokemon_ex")
        if "v" in subs or name.endswith(" v") or " v " in name:
            classes.add("pokemon_v")
        for t in types:
            if t:
                classes.add(f"{t}_pokemon")
                if "basic" in subs:
                    classes.add(f"basic_{t}_pokemon")

    if "energy" in st:
        classes.add("energy")
        if "basic" in subs or "basic" in name:
            classes.add("basic_energy")
        for t in types:
            if t:
                classes.add(f"{t}_energy")
                if "basic" in subs or "basic" in name:
                    classes.add(f"basic_{t}_energy")
        # Fallback for energy cards whose type is only in the name.
        for t in ("water", "fighting", "fire", "grass", "lightning", "psychic", "darkness", "metal"):
            if t in name:
                classes.add(f"{t}_energy")
                if "basic" in name or "basic" in subs:
                    classes.add(f"basic_{t}_energy")

    if "trainer" in st:
        classes.add("trainer")
        for sub in subs:
            if "supporter" in sub:
                classes.update(["trainer", "supporter"])
            if "item" in sub or "tool" in sub or "pokemon tool" in sub or "pokémon tool" in sub:
                classes.update(["trainer", "item"])
            if "stadium" in sub:
                classes.update(["trainer", "stadium"])

    return classes


def turn1_result_goal_filter_class_implies(classes, wanted):
    if wanted in classes:
        return True
    # Specific card classes imply their broad parents.
    if wanted == "pokemon" and any(c.endswith("_pokemon") or c in {"basic_pokemon", "pokemon_ex", "pokemon_v"} for c in classes):
        return True
    if wanted == "energy" and any(c.endswith("_energy") or c in {"basic_energy"} for c in classes):
        return True
    if wanted == "trainer" and classes.intersection({"supporter", "item", "stadium"}):
        return True
    return False


def turn1_result_goal_filter_classes_from_text(text):
    t = turn1_result_goal_filter_norm(text)
    classes = set()

    if not t:
        return classes

    # Opponent-only hand/deck effects are not self access.
    opponent_only = (
        ("your opponent" in t or "opponent's" in t or "opponents" in t or "their hand" in t or "their deck" in t)
        and not any(x in t for x in ["your deck", "your hand", "put into your hand", "onto your bench", "you may draw", "draw cards"])
    )
    if opponent_only:
        classes.add("opponent_only")
        return classes

    # Energy searches.
    if "basic water energy" in t:
        classes.update(["energy", "basic_energy", "water_energy", "basic_water_energy"])
    elif "basic fighting energy" in t:
        classes.update(["energy", "basic_energy", "fighting_energy", "basic_fighting_energy"])
    elif "basic energy" in t:
        classes.update(["energy", "basic_energy"])
    elif _turn1_result_goal_filter_re.search(r"(search|choose|reveal|find|put).{0,180}energy", t):
        classes.add("energy")

    # Pokémon searches.
    if _turn1_result_goal_filter_re.search(r"(search|choose|reveal|find|put|bench).{0,220}(basic water pokemon|basic water pokémon)", t):
        classes.update(["pokemon", "basic_pokemon", "water_pokemon", "basic_water_pokemon"])
    elif _turn1_result_goal_filter_re.search(r"(search|choose|reveal|find|put|bench).{0,220}(water pokemon|water pokémon)", t):
        classes.update(["pokemon", "water_pokemon"])

    if _turn1_result_goal_filter_re.search(r"(search|choose|reveal|find|put|bench).{0,220}(basic fighting pokemon|basic fighting pokémon)", t):
        classes.update(["pokemon", "basic_pokemon", "fighting_pokemon", "basic_fighting_pokemon"])
    elif _turn1_result_goal_filter_re.search(r"(search|choose|reveal|find|put|bench).{0,220}(fighting pokemon|fighting pokémon)", t):
        classes.update(["pokemon", "fighting_pokemon"])

    if _turn1_result_goal_filter_re.search(r"(search|choose|reveal|find|put|bench).{0,220}(pokemon ex|pokémon ex)", t):
        classes.update(["pokemon", "pokemon_ex"])
    if _turn1_result_goal_filter_re.search(r"(search|choose|reveal|find|put|bench).{0,220}(basic pokemon|basic pokémon)", t):
        classes.update(["pokemon", "basic_pokemon"])
    elif _turn1_result_goal_filter_re.search(r"(search|choose|reveal|find|put|bench).{0,220}(pokemon|pokémon)", t):
        classes.add("pokemon")

    # Trainer searches.
    if _turn1_result_goal_filter_re.search(r"(search|choose|reveal|find|put|look at).{0,220}supporter", t):
        classes.update(["trainer", "supporter"])
    if _turn1_result_goal_filter_re.search(r"(search|choose|reveal|find|put|look at).{0,220}item", t):
        classes.update(["trainer", "item"])
    if _turn1_result_goal_filter_re.search(r"(search|choose|reveal|find|put|look at).{0,220}stadium", t):
        classes.update(["trainer", "stadium"])

    # Generic self-draw is allowed as general access but does not define a search target.
    if _turn1_result_goal_filter_re.search(r"\b(draw|draws|draw until)\b.{0,100}\b(card|cards)\b", t) and "your opponent" not in t:
        classes.add("generic_draw")

    return classes


def turn1_result_goal_filter_classes(filt):
    classes = set()
    if not isinstance(filt, dict):
        return classes

    # Structured compiler flags.
    if filt.get("basic_pokemon"):
        classes.update(["pokemon", "basic_pokemon"])
    elif filt.get("pokemon"):
        classes.add("pokemon")
    if filt.get("energy"):
        classes.add("energy")
    if filt.get("basic_energy"):
        classes.update(["energy", "basic_energy"])
    if filt.get("supporter"):
        classes.update(["trainer", "supporter"])
    if filt.get("item"):
        classes.update(["trainer", "item"])
    if filt.get("stadium"):
        classes.update(["trainer", "stadium"])

    # Raw text from compiled effect.
    raw_parts = []
    for key in ("raw_text", "text", "source_text", "printed", "name"):
        val = filt.get(key)
        if val:
            raw_parts.append(str(val))
    if raw_parts:
        classes.update(turn1_result_goal_filter_classes_from_text(" ".join(raw_parts)))

    return classes


def turn1_result_goal_filter_card_allowed_by_classes(card, allowed):
    if not allowed:
        return True
    if "opponent_only" in allowed:
        return False
    if "generic_draw" in allowed and len(allowed) == 1:
        return True

    cc = turn1_result_goal_filter_card_classes(card)

    # Remove broad classes from the search side when specific restrictions exist.
    specific = set(allowed)
    if any(x in specific for x in ["basic_pokemon", "water_pokemon", "basic_water_pokemon", "fighting_pokemon", "basic_fighting_pokemon", "pokemon_ex"]):
        specific.discard("pokemon")
    if any(x in specific for x in ["basic_energy", "water_energy", "basic_water_energy", "fighting_energy", "basic_fighting_energy"]):
        specific.discard("energy")
    if any(x in specific for x in ["supporter", "item", "stadium"]):
        specific.discard("trainer")
    specific.discard("generic_draw")

    if not specific:
        specific = set(allowed)

    return any(turn1_result_goal_filter_class_implies(cc, want) for want in specific)


# Patch target_finder's core filter check. This fixes many bad selections at the source.
_ORIG_TF_FILTER_ALLOWS_CARD_BEFORE_RESULT_GOAL_FILTER = getattr(tf, "filter_allows_card", None)

def turn1_result_goal_filter_allows_card(filt, card):
    classes = turn1_result_goal_filter_classes(filt)
    if classes:
        return turn1_result_goal_filter_card_allowed_by_classes(card, classes)
    if _ORIG_TF_FILTER_ALLOWS_CARD_BEFORE_RESULT_GOAL_FILTER is not None:
        return _ORIG_TF_FILTER_ALLOWS_CARD_BEFORE_RESULT_GOAL_FILTER(filt, card)
    return True

try:
    tf.filter_allows_card = turn1_result_goal_filter_allows_card
except Exception:
    pass


def turn1_result_goal_filter_flatten_strings(obj, max_items=7000):
    out = []
    seen = set()
    def rec(x):
        if len(out) >= max_items:
            return
        oid = id(x)
        if oid in seen:
            return
        seen.add(oid)
        if isinstance(x, str):
            if x.strip():
                out.append(x)
            return
        if isinstance(x, dict):
            for k, v in x.items():
                if isinstance(k, str) and k.strip():
                    out.append(k)
                rec(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                rec(v)
    rec(obj)
    return " ".join(out)


def turn1_result_goal_filter_action_labels(line):
    raw = str(line or "").strip()
    if not raw or raw == "none":
        return []
    return [p.strip() for p in raw.split("->") if p.strip()]


def turn1_result_goal_filter_deck_name_map(deck):
    out = {}
    for card in deck or []:
        name = turn1_result_goal_filter_card_name(card)
        n = turn1_result_goal_filter_norm(name)
        if n and n not in out:
            out[n] = card
    return out


def turn1_result_goal_filter_extract_steps_from_card_or_effect(obj):
    steps = []
    try:
        if isinstance(obj, dict):
            # Card-level meaningful steps.
            steps.extend(list(tf.meaningful_steps(obj)))
    except Exception:
        pass
    try:
        steps.extend(list(tf.flatten_steps(obj)))
    except Exception:
        pass
    return steps


def turn1_result_goal_filter_action_target_classes(action, deck):
    """What kind of card can this action directly access?"""
    action_norm = turn1_result_goal_filter_norm(action)
    name_map = turn1_result_goal_filter_deck_name_map(deck)

    # Physical card action.
    if action_norm in name_map:
        card = name_map[action_norm]
        cname = turn1_result_goal_filter_norm(turn1_result_goal_filter_card_name(card))
        classes = set()

        # Important explicit card policies.
        if cname == "buddy-buddy poffin" or cname == "buddy buddy poffin":
            return {"pokemon", "basic_pokemon"}
        if cname == "nest ball":
            return {"pokemon", "basic_pokemon"}
        if cname == "ultra ball":
            return {"pokemon"}
        if cname == "irida":
            return {"pokemon", "water_pokemon", "trainer", "item"}
        if cname == "cyrano":
            return {"pokemon", "pokemon_ex"}
        if cname == "fighting gong":
            return {"pokemon", "basic_pokemon", "fighting_pokemon", "basic_fighting_pokemon", "energy", "basic_energy", "fighting_energy", "basic_fighting_energy"}

        for step in turn1_result_goal_filter_extract_steps_from_card_or_effect(card):
            if not isinstance(step, dict):
                continue
            op = step.get("op")
            if op in {"search_deck", "choose_cards", "put_card_into_hand", "move_card", "move_cards"}:
                try:
                    filt = tf.extract_filter(step)
                except Exception:
                    filt = step.get("filter") or {}
                classes.update(turn1_result_goal_filter_classes(filt))
                classes.update(turn1_result_goal_filter_classes_from_text(turn1_result_goal_filter_flatten_strings(step)))
            elif op in {"draw_cards", "draw_cards_per_coin_heads", "draw_until_hand_size", "draw_until_hand_size_matches"}:
                classes.add("generic_draw")

        # If the card has no search/draw target class, allow it rather than making up a restriction.
        return classes

    # Ability/effect label. Find the owning effect/card text.
    blobs = []
    for card in deck or []:
        card_blob = turn1_result_goal_filter_flatten_strings(card)
        if action_norm and action_norm in turn1_result_goal_filter_norm(card_blob):
            blobs.append(card_blob)

    classes = set()
    for blob in blobs:
        classes.update(turn1_result_goal_filter_classes_from_text(blob))

    # If it is not a card name and not a known effect label, leave empty.
    return classes


def turn1_result_goal_filter_goal_classes(reqs, deck):
    classes = set()
    for req in reqs or []:
        for opt in getattr(req, "options", []) or []:
            matched = False
            for card in deck or []:
                try:
                    if card_matches_option(card, opt):
                        classes.update(turn1_result_goal_filter_card_classes(card))
                        matched = True
                except Exception:
                    pass
            if not matched:
                # Text fallback for goals like "Basic Water Energy" if deck match fails.
                classes.update(turn1_result_goal_filter_classes_from_text(getattr(opt, "raw", "")))
    return classes


def turn1_result_goal_filter_action_compatible_with_goal(action_classes, goal_classes):
    if not action_classes:
        # Unknown physical card actions are allowed elsewhere; unknown effect labels are handled by caller.
        return True
    if "opponent_only" in action_classes:
        return False
    if "generic_draw" in action_classes:
        return True

    # Remove broad classes when specific restrictions exist on the action.
    specific = set(action_classes)
    if any(x in specific for x in ["basic_pokemon", "water_pokemon", "basic_water_pokemon", "fighting_pokemon", "basic_fighting_pokemon", "pokemon_ex"]):
        specific.discard("pokemon")
    if any(x in specific for x in ["basic_energy", "water_energy", "basic_water_energy", "fighting_energy", "basic_fighting_energy"]):
        specific.discard("energy")
    if any(x in specific for x in ["supporter", "item", "stadium"]):
        specific.discard("trainer")
    specific.discard("generic_draw")

    if not specific:
        specific = set(action_classes)

    for allowed in specific:
        for goal in goal_classes:
            # Exact or subclass match.
            if allowed == goal:
                return True
            # Broad search can hit specific goal.
            if allowed == "pokemon" and turn1_result_goal_filter_class_implies({goal}, "pokemon"):
                return True
            if allowed == "energy" and turn1_result_goal_filter_class_implies({goal}, "energy"):
                return True
            if allowed == "trainer" and turn1_result_goal_filter_class_implies({goal}, "trainer"):
                return True
    return False


def turn1_direct_goal_filter_goal_target_norms(reqs, deck):
    # Return normalized concrete target names/card ids for the current goal requirements.
    norms = []
    seen = set()
    for req in reqs or []:
        for opt in getattr(req, "options", []) or []:
            raw_values = []
            for attr in ("name", "raw", "label", "target", "card_name"):
                try:
                    v = getattr(opt, attr, None)
                except Exception:
                    v = None
                if v:
                    raw_values.append(v)
            for card in deck or []:
                try:
                    if card_matches_option(card, opt):
                        raw_values.extend([
                            tf.card_name(card),
                            card.get("card_id") if isinstance(card, dict) else None,
                            card.get("representative_card_id") if isinstance(card, dict) else None,
                        ])
                except Exception:
                    pass
            for v in raw_values:
                n = turn1_result_goal_filter_norm(v)
                if n and n not in seen:
                    seen.add(n)
                    norms.append(n)
    return norms


def turn1_direct_goal_filter_action_reaches_goal_directly(action_card, deck, reqs) -> bool:
    # True when target_finder says this physical action can directly find a goal card.
    if not isinstance(action_card, dict):
        return False
    for target_norm in turn1_direct_goal_filter_goal_target_norms(reqs, deck):
        try:
            if tf.card_directly_searches_target(action_card, target_norm, deck):
                return True
        except TypeError:
            try:
                if tf.card_directly_searches_target(action_card, target_norm):
                    return True
            except Exception:
                pass
        except Exception:
            pass
    return False


def turn1_result_goal_filter_bad_actions_for_goal(line, deck, reqs):
    # TURN1_DEWRAP_DIRECT_GOAL_FILTER
    goal_classes = turn1_result_goal_filter_goal_classes(reqs, deck)
    if not goal_classes:
        return []

    name_map = turn1_result_goal_filter_deck_name_map(deck)
    bad = []

    for action in turn1_result_goal_filter_action_labels(line):
        action_norm = turn1_result_goal_filter_norm(action)
        action_card = name_map.get(action_norm)

        # Physical card action: if target_finder's direct search logic says it
        # can find a goal card, the post-filter must not invalidate the line.
        if action_card is not None and turn1_direct_goal_filter_action_reaches_goal_directly(action_card, deck, reqs):
            continue

        classes = turn1_result_goal_filter_action_target_classes(action, deck)

        # Unknown non-card action = unvalidated ability label. Block.
        if action_norm not in name_map and not classes:
            bad.append({"action": action, "reason": "unvalidated_effect_label", "classes": []})
            continue

        # Unknown physical card action: allow.
        if action_norm in name_map and not classes:
            continue

        if not turn1_result_goal_filter_action_compatible_with_goal(classes, goal_classes):
            bad.append({
                "action": action,
                "reason": "search_target_incompatible_with_goal",
                "classes": sorted(classes),
                "goal_classes": sorted(goal_classes),
            })

    return bad


def turn1_result_goal_filter_single_result(result, deck, reqs):
    if not isinstance(result, dict):
        return result
    if not result.get("success"):
        return result
    line = result.get("line") or "none"
    if line == "none":
        return result
    bad = turn1_result_goal_filter_bad_actions_for_goal(line, deck, reqs)
    if not bad:
        return result
    result["success"] = False
    result["success_stage"] = "blocked_search_target_type"
    result["blocked_search_target_type"] = bad
    result["missing_requirements"] = [
        "Blocked incompatible search target: " + ", ".join(b.get("action", "?") for b in bad)
    ]
    return result


def turn1_result_goal_filter_results(results, deck, reqs):
    for r in results or []:
        turn1_result_goal_filter_single_result(r, deck, reqs)
    return results


# Override old guard functions that were too broad or based on full card text.
# Wrap the trial simulator last. Prefer a pre-unvalidated-effect-guard base if present so the earlier
# "block all non-card labels" wrapper does not incorrectly kill valid Shivery Chill
# lines for Basic Water Energy goals.
_TURN1_BASE_SIMULATE_BEFORE_RESULT_GOAL_FILTER = globals().get("_ORIG_SIMULATE_ONE_GOAL_TRIAL_BEFORE_UNVALIDATED_EFFECT_GUARD") or globals().get("_ORIG_SIMULATE_ONE_GOAL_TRIAL_V32") or simulate_one_goal_trial

def simulate_one_goal_trial(*args, **kwargs):
    result = _TURN1_BASE_SIMULATE_BEFORE_RESULT_GOAL_FILTER(*args, **kwargs)
    deck = kwargs.get("deck")
    reqs = kwargs.get("reqs")
    if deck is None:
        for value in args:
            if isinstance(value, list) and value and isinstance(value[0], dict):
                deck = value
                break
    if reqs is None:
        for value in args:
            if isinstance(value, (list, tuple)) and value and hasattr(value[0], "options"):
                reqs = value
                break
    if deck is not None and reqs is not None:
        return turn1_result_goal_filter_single_result(result, deck, reqs)
    return result


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
    ap.add_argument("--exclude-played", default="", help="Comma-separated card names. Successful action paths that played any of these cards are counted as failures.")
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
            "played_exclusion_filter": getattr(args, "exclude_played", ""),
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



# ---------------------------------------------------------------------
# TURN1_ACTIVE_COMPILED_SEARCH_PERF_CACHE_V41_1
# ---------------------------------------------------------------------
# v0.41 added generic compiled-effect checks. Correct direction, but expensive:
# it can repeatedly flatten/scan the same card/effect text for every trial.
#
# This patch adds conservative memoization around known v41 helpers if they
# exist. It does not change simulator semantics.

try:
    from functools import lru_cache as _turn1_runtime_cache_wrapper_lru_cache
except Exception:
    _turn1_runtime_cache_wrapper_lru_cache = None


def _turn1_runtime_cache_card_key(card):
    if not isinstance(card, dict):
        return str(id(card))

    ident = card.get("identity") or {}

    return str(
        card.get("card_id")
        or card.get("representative_card_id")
        or card.get("id")
        or ident.get("card_id")
        or ident.get("id")
        or ident.get("name")
        or card.get("name")
        or id(card)
    )


def _turn1_install_runtime_cache_wrapper(name):
    if name not in globals():
        return False

    fn = globals().get(name)

    if not callable(fn):
        return False

    cache = {}

    def wrapped(*args, **kwargs):
        try:
            key_parts = [name]

            for arg in args:
                if isinstance(arg, dict):
                    key_parts.append(("card", _turn1_runtime_cache_card_key(arg)))
                elif isinstance(arg, list):
                    # For deck/list args, do not store whole list contents.
                    # Use card ids when it looks like a deck.
                    if arg and all(isinstance(x, dict) for x in arg[: min(5, len(arg))]):
                        key_parts.append(
                            (
                                "deck",
                                tuple(_turn1_runtime_cache_card_key(x) for x in arg),
                            )
                        )
                    else:
                        key_parts.append(("list", len(arg), id(arg)))
                elif isinstance(arg, (str, int, float, bool, type(None))):
                    key_parts.append(arg)
                else:
                    key_parts.append(("obj", id(arg)))

            for k, v in sorted(kwargs.items()):
                if isinstance(v, dict):
                    key_parts.append((k, "card", _turn1_runtime_cache_card_key(v)))
                elif isinstance(v, list):
                    if v and all(isinstance(x, dict) for x in v[: min(5, len(v))]):
                        key_parts.append(
                            (
                                k,
                                "deck",
                                tuple(_turn1_runtime_cache_card_key(x) for x in v),
                            )
                        )
                    else:
                        key_parts.append((k, "list", len(v), id(v)))
                elif isinstance(v, (str, int, float, bool, type(None))):
                    key_parts.append((k, v))
                else:
                    key_parts.append((k, "obj", id(v)))

            key = tuple(key_parts)

            if key in cache:
                return cache[key]

            result = fn(*args, **kwargs)

            # Keep cache bounded.
            if len(cache) < 50000:
                cache[key] = result

            return result

        except Exception:
            return fn(*args, **kwargs)

    wrapped.__name__ = getattr(fn, "__name__", name)
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    globals()[name] = wrapped
    return True


_turn1_runtime_cache_wrapper_wrapped_helpers = []

for _turn1_runtime_cache_wrapper_name in [
    # likely compiled-effect helpers
    "turn1_v41_flatten_strings",
    "turn1_v41_norm",
    "turn1_v41_effect_can_help_goal",
    "turn1_v41_search_effect_can_help_goal",
    "turn1_v41_card_has_active_compiled_search",
    "turn1_v41_active_compiled_search_candidates",
    "turn1_v41_compiled_search_filter_allows_goal",
    "turn1_v41_goal_classes",
    "turn1_v41_access_classes_from_effect",
    "turn1_v41_best_active_for_goal",
    # result-goal and later helper families, if present
    "turn1_result_goal_filter_goal_classes",
    "turn1_result_goal_filter_classes_from_text",
    "turn1_v38_goal_classes",
    "turn1_v38_access_classes_from_text",
]:
    try:
        if _turn1_install_runtime_cache_wrapper(_turn1_runtime_cache_wrapper_name):
            _turn1_runtime_cache_wrapper_wrapped_helpers.append(_turn1_runtime_cache_wrapper_name)
    except Exception:
        pass

try:
    TURN1_ACTIVE_COMPILED_SEARCH_PERF_CACHE_V41_1 = {
        "installed": True,
        "wrapped_helpers": list(_turn1_runtime_cache_wrapper_wrapped_helpers),
    }
except Exception:
    pass




# ---------------------------------------------------------------------
# TURN1_HOTPATH_CLASSIFICATION_CACHE
# ---------------------------------------------------------------------
# Performance fix after profiling.
#
# Hotspots found:
# - gf.score_candidate_for_missing_targets
# - gf._turn1_card_goal_search_capacity
# - tf.score_playable_card
# - tf.card_directly_searches_target
# - gf.turn1_opponent_only_filter_flatten_text
#
# The expensive part is repeated classification of the same cards/effects/text.
# This caches pure classification helpers. It intentionally does NOT cache
# stateful action execution or draw/deck mutation.

def _turn1_hotpath_cache_card_key(obj):
    if not isinstance(obj, dict):
        return None

    ident = obj.get("identity") or {}

    key = (
        obj.get("card_id")
        or obj.get("representative_card_id")
        or obj.get("id")
        or ident.get("card_id")
        or ident.get("id")
        or ident.get("canonical_id")
    )

    name = (
        obj.get("name")
        or obj.get("card_name")
        or ident.get("name")
        or ident.get("canonical_name")
    )

    set_code = (
        obj.get("set_code")
        or obj.get("set")
        or ident.get("set_code")
        or ident.get("set")
    )

    number = (
        obj.get("number")
        or obj.get("collector_number")
        or ident.get("number")
        or ident.get("collector_number")
    )

    if key:
        return ("card", str(key))

    # Only treat as a card if it has card-like fields. Goal requirement dicts
    # also have labels/names, so don't classify those as cards.
    card_like = any(
        k in obj
        for k in [
            "supertype",
            "subtypes",
            "types",
            "hp",
            "attacks",
            "abilities",
            "compiled_effects",
            "raw_card",
            "gameplay",
        ]
    )

    if card_like and name:
        return ("card", str(name), str(set_code or ""), str(number or ""))

    return None


def _turn1_hotpath_cache_safe_key(obj, depth=0):
    if depth > 4:
        return ("deep", id(obj))

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    card_key = _turn1_hotpath_cache_card_key(obj)
    if card_key is not None:
        return card_key

    if isinstance(obj, dict):
        # Goal requirements/options are small and should be value-cached.
        simple = {}

        for k, v in obj.items():
            if k in {
                "label",
                "name",
                "options",
                "zone",
                "min_count",
                "mode",
                "target",
                "target_norm",
                "supertype",
                "subtype",
                "subtypes",
                "type",
                "types",
                "hp_max",
                "max_hp",
                "basic_pokemon",
                "energy",
                "any_card",
                "any",
                "raw_text",
                "source_text",
                "text",
            }:
                simple[str(k)] = _turn1_hotpath_cache_safe_key(v, depth + 1)

        if simple:
            return ("dict", tuple(sorted(simple.items())))

        return ("dict_id", id(obj))

    if isinstance(obj, (list, tuple)):
        if len(obj) > 80:
            # Large mutable lists are usually deck/hand state. Avoid expensive
            # full value keys, but still allow repeated same-object cache hits.
            return (type(obj).__name__, "large", len(obj), id(obj))

        return (type(obj).__name__, tuple(_turn1_hotpath_cache_safe_key(x, depth + 1) for x in obj))

    if isinstance(obj, set):
        if len(obj) > 80:
            return ("set", "large", len(obj), id(obj))

        return ("set", tuple(sorted(_turn1_hotpath_cache_safe_key(x, depth + 1) for x in obj)))

    return ("obj", type(obj).__name__, id(obj))


def _turn1_wrap_hotpath_cached(owner, name, max_entries=200000):
    try:
        fn = getattr(owner, name)
    except Exception:
        return False

    if not callable(fn):
        return False

    if getattr(fn, "_turn1_hotpath_cached", False):
        return False

    cache = {}

    def wrapped(*args, **kwargs):
        try:
            key = (
                name,
                tuple(_turn1_hotpath_cache_safe_key(a) for a in args),
                tuple(sorted((str(k), _turn1_hotpath_cache_safe_key(v)) for k, v in kwargs.items())),
            )

            if key in cache:
                return cache[key]

            result = fn(*args, **kwargs)

            if len(cache) < max_entries:
                cache[key] = result

            return result

        except Exception:
            return fn(*args, **kwargs)

    wrapped.__name__ = getattr(fn, "__name__", name)
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    wrapped._turn1_hotpath_cached = True

    setattr(owner, name, wrapped)
    return True


_TURN1_HOTPATH_CACHED_HELPERS = []

# Goal-finder pure-ish classification hotpaths.
for _turn1_hotpath_cache_name in [
    "_turn1_card_goal_search_capacity",
    "turn1_opponent_only_filter_flatten_text",
    "card_matches_option",
    "requirement_satisfied",
    "goal_satisfied",
    "zone_cards",
    "turn1_result_goal_filter_single_result",
    "turn1_v41_card_has_active_compiled_search",
    "turn1_v41_search_effect_can_help_goal",
    "turn1_v41_compiled_search_filter_allows_goal",
    "turn1_v41_goal_classes",
    "turn1_v41_access_classes_from_effect",
]:
    try:
        if _turn1_wrap_hotpath_cached(globals(), _turn1_hotpath_cache_name):
            _TURN1_HOTPATH_CACHED_HELPERS.append("gf." + _turn1_hotpath_cache_name)
    except Exception:
        pass

# Target-finder pure classification hotpaths.
try:
    for _turn1_hotpath_cache_name in [
        "card_directly_searches_target",
        "filter_text_blob",
        "filter_allows_card",
        "compiled_effects",
        "ability_requirement_search_candidates",
        "search_amount",
        "card_name",
        "is_basic_pokemon",
        "is_energy",
    ]:
        try:
            if _turn1_wrap_hotpath_cached(tf, _turn1_hotpath_cache_name):
                _TURN1_HOTPATH_CACHED_HELPERS.append("tf." + _turn1_hotpath_cache_name)
        except Exception:
            pass
except Exception:
    pass

TURN1_HOTPATH_CLASSIFICATION_CACHE = {
    "installed": True,
    "cached_helpers": list(_TURN1_HOTPATH_CACHED_HELPERS),
}

# ---------------------------------------------------------------------

# TURN1_CLEAN_CANONICAL_GOAL_SELECTOR_V1
# Canonical source-bound goal selector.
#
# This replaces the patch-stack selector path with one readable rule:
#   candidate must match a missing goal option
#   AND the actual source card/effect must be able to select that concrete candidate.
#
# Regression targets:
#   Fighting Gong -> Solrock/Lunatone: allowed when candidate is Basic Fighting Pokemon
#   Poké Pad -> Wally's Compassion: blocked
#   Buddy-Buddy Poffin -> Energy: blocked
#   Shivery Chill / Earthen Vessel -> Basic Energy: allowed

def _turn1_clean_norm(value):
    try:
        return tf.norm(str(value or ""))
    except Exception:
        return str(value or "").lower().strip()


def _turn1_clean_card_name(card):
    try:
        return tf.card_name(card)
    except Exception:
        if isinstance(card, dict):
            return str(card.get("name") or card.get("card_name") or "")
        return ""


def _turn1_clean_card_supertype(card):
    try:
        return str(tf.card_supertype(card) or "")
    except Exception:
        if isinstance(card, dict):
            return str(card.get("supertype") or "")
        return ""


def _turn1_clean_card_types(card):
    try:
        return [_turn1_clean_norm(x) for x in tf.card_types(card)]
    except Exception:
        if isinstance(card, dict):
            value = card.get("types") or []
            if isinstance(value, str):
                value = [value]
            return [_turn1_clean_norm(x) for x in value]
        return []


def _turn1_clean_card_subtypes(card):
    try:
        return [_turn1_clean_norm(x) for x in tf.card_subtypes(card)]
    except Exception:
        if isinstance(card, dict):
            value = card.get("subtypes") or []
            if isinstance(value, str):
                value = [value]
            return [_turn1_clean_norm(x) for x in value]
        return []


def _turn1_clean_card_hp(card):
    if not isinstance(card, dict):
        return None
    raw = card.get("hp")
    try:
        return int(raw)
    except Exception:
        return None


def _turn1_clean_card_text_blob(card):
    parts = []
    def walk(value, depth=0):
        if value is None or depth > 5:
            return
        if isinstance(value, str):
            parts.append(value)
            return
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(k, str):
                    parts.append(k)
                walk(v, depth + 1)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                walk(item, depth + 1)
    walk(card)
    return " ".join(parts)


def _turn1_clean_source_text(action_card, filt, source_step):
    parts = []
    if isinstance(filt, dict):
        for key in ("raw_text", "source_text", "text"):
            if filt.get(key):
                parts.append(str(filt.get(key)))
    if isinstance(source_step, dict):
        for key in ("source_text", "raw_text", "text"):
            if source_step.get(key):
                parts.append(str(source_step.get(key)))
    if isinstance(action_card, dict):
        for key in ("combined_text", "raw_text", "text", "abilities_text", "attacks_text", "rules"):
            value = action_card.get(key)
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, list):
                parts.extend(str(x) for x in value)
    return _turn1_clean_norm(" ".join(parts))


def _turn1_clean_is_pokemon(card):
    return _turn1_clean_norm(_turn1_clean_card_supertype(card)) in {"pokemon", "pokémon"}


def _turn1_clean_is_trainer(card):
    return _turn1_clean_norm(_turn1_clean_card_supertype(card)) == "trainer"


def _turn1_clean_is_energy(card):
    return _turn1_clean_norm(_turn1_clean_card_supertype(card)) == "energy"


def _turn1_clean_is_basic(card):
    name = _turn1_clean_norm(_turn1_clean_card_name(card))
    return "basic" in _turn1_clean_card_subtypes(card) or name.startswith("basic ")


def _turn1_clean_has_type(card, type_name):
    t = _turn1_clean_norm(type_name)
    name = _turn1_clean_norm(_turn1_clean_card_name(card))
    return t in _turn1_clean_card_types(card) or f"basic {t} energy" in name


def _turn1_clean_has_rule_box(card):
    blob = _turn1_clean_norm(_turn1_clean_card_text_blob(card))
    name = _turn1_clean_norm(_turn1_clean_card_name(card))
    if "rule box" in blob:
        return True
    # Conservative rule-box approximation for Pokémon labels.
    return any(tok in name for tok in [" ex", "-ex", " vmax", " vstar", " v-union", " gx"])


def _turn1_clean_text_specific_allows(action_card, candidate, filt, action_name, source_step=None):
    text = _turn1_clean_source_text(action_card, filt, source_step)
    name = _turn1_clean_norm(action_name or _turn1_clean_card_name(action_card))

    # Fighting Gong:
    # Search your deck for a Basic Fighting Energy card or a Basic Fighting Pokemon.
    if "basic fighting energy" in text and "basic fighting pokemon" in text:
        if _turn1_clean_is_energy(candidate):
            return _turn1_clean_is_basic(candidate) and _turn1_clean_has_type(candidate, "fighting")
        if _turn1_clean_is_pokemon(candidate):
            return _turn1_clean_is_basic(candidate) and _turn1_clean_has_type(candidate, "fighting")
        return False

    # Buddy-Buddy Poffin:
    # up to 2 Basic Pokemon with 70 HP or less onto Bench.
    if "basic pokemon with 70 hp or less" in text or "basic pokémon with 70 hp or less" in text:
        hp = _turn1_clean_card_hp(candidate)
        return _turn1_clean_is_pokemon(candidate) and _turn1_clean_is_basic(candidate) and hp is not None and hp <= 70

    # Poké Pad:
    # Search for a Pokemon that doesn't have a Rule Box.
    if "doesn't have a rule box" in text or "does not have a rule box" in text:
        return _turn1_clean_is_pokemon(candidate) and not _turn1_clean_has_rule_box(candidate)

    # Shivery Chill / similar Basic Water Energy searches.
    if "basic water energy" in text:
        return _turn1_clean_is_energy(candidate) and _turn1_clean_is_basic(candidate) and _turn1_clean_has_type(candidate, "water")

    # Earthen Vessel / generic Basic Energy searches.
    if "basic energy" in text and "basic pokemon" not in text and "basic pokémon" not in text:
        return _turn1_clean_is_energy(candidate) and _turn1_clean_is_basic(candidate)

    # Generic Pokemon search.
    if "search your deck for a pokemon" in text or "search your deck for a pokémon" in text:
        return _turn1_clean_is_pokemon(candidate)

    return None


def _turn1_clean_source_can_select_candidate(action_card, candidate, filt, action_name, source_step=None):
    if not isinstance(candidate, dict):
        return False

    if action_card is None:
        return False

    # First apply the structured compiled filter.
    try:
        if not _turn1_search_filter_allows_for_action(filt or {}, candidate, action_name):
            return False
    except Exception:
        return False

    # Then apply explicit text-specific rules when recognizable.
    specific = _turn1_clean_text_specific_allows(action_card, candidate, filt or {}, action_name, source_step)
    if specific is not None:
        return bool(specific)

    # Then use the existing v67 printed/source-text guard.
    try:
        if "_turn1_source_text_allows_card" in globals():
            if not _turn1_source_text_allows_card(
                filt or {},
                candidate,
                action_name,
                action_card=action_card,
                source_step=source_step,
            ):
                return False
    except Exception:
        return False

    # Finally require target_finder's source-card/direct-search agreement.
    try:
        return bool(tf.card_directly_searches_target(
            action_card,
            tf.norm(_turn1_clean_card_name(candidate)),
            [candidate],
        ))
    except Exception:
        return False


def _turn1_goal_select_from_deck(
    st: tf.SimState,
    reqs: Sequence[GoalRequirement],
    mode: str,
    tracker: GoalTracker,
    filt: Dict[str, Any],
    amount: int,
    action_name: str,
    action_card=None,
    source_step=None,
) -> List[Dict[str, Any]]:
    # TURN1_CLEAN_CANONICAL_GOAL_SELECTOR_V1
    selected: List[Dict[str, Any]] = []
    if amount <= 0:
        return selected

    virtual_tracker = GoalTracker()
    virtual_tracker.mark(tracker.accessed)
    virtual_tracker.mark(selected)

    while len(selected) < amount:
        deficits = _turn1_missing_goal_requirements_with_deficits(reqs, mode, st, virtual_tracker)
        if not deficits:
            break

        if mode == "any":
            ordered_deficits = deficits
        else:
            ordered_deficits = sorted(deficits, key=lambda x: (-x[1], x[0].label))

        chosen_idx = None
        chosen_req = None

        for req, _deficit in ordered_deficits:
            for idx, candidate in enumerate(st.deck):
                if not isinstance(candidate, dict):
                    continue

                if not any(card_matches_option(candidate, opt) for opt in req.options):
                    continue

                if not _turn1_clean_source_can_select_candidate(
                    action_card,
                    candidate,
                    filt,
                    action_name,
                    source_step=source_step,
                ):
                    continue

                chosen_idx = idx
                chosen_req = req
                break

            if chosen_idx is not None:
                break

        if chosen_idx is None:
            break

        chosen = st.deck.pop(chosen_idx)
        selected.append(chosen)
        virtual_tracker.mark([chosen])

        if mode == "any" and chosen_req is not None:
            if _turn1_goal_requirement_deficit(chosen_req, st, virtual_tracker) <= 0:
                break

    return selected

if __name__ == "__main__":
    main()
