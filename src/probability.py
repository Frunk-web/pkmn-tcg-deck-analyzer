"""
Exact probability formulas for the Pokémon TCG probability app.

This module is intentionally pure math:
- no Streamlit
- no Pokémon TCG API calls
- no simulator state mutation

It handles:
- no-Basic opening hand probability
- mulligan distribution
- single-card opening/draw/prize probabilities
- legal-opening-hand-conditioned prize probabilities
- multi-card Turn 1 goal baselines such as X AND Y, X AND Y AND Z,
  and OR groups such as Riolu OR Makuhita.

Important distinction:
Prize probabilities here are NOT simply "6 random cards from the full 60-card deck."
They are conditioned on the player keeping a legal opening hand first. A legal
opening hand means the 7-card hand contains at least one Basic Pokémon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd


# ============================================================
# Core helpers
# ============================================================


def C(n: int, k: int) -> int:
    """Safe n choose k."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


# ============================================================
# Opening hand probabilities
# ============================================================


def p_no_basic_opening_7(deck_size: int, basic_count: int) -> float:
    non_basic_count = deck_size - basic_count
    if deck_size < 7:
        return 1.0
    if non_basic_count < 7:
        return 0.0
    return C(non_basic_count, 7) / C(deck_size, 7)


def p_card_in_random_7(deck_size: int, card_count: int) -> float:
    return 1 - C(deck_size - card_count, 7) / C(deck_size, 7)


def p_card_in_legal_opening_7(
    deck_size: int,
    basic_count: int,
    card_count: int,
    card_is_basic: bool,
) -> float:
    p_no_basic = p_no_basic_opening_7(deck_size, basic_count)
    p_legal = 1 - p_no_basic

    if p_legal <= 0:
        return float("nan")

    p_a = p_card_in_random_7(deck_size, card_count)

    if card_is_basic:
        p_a_and_no_basic = 0.0
    else:
        non_basic_count = deck_size - basic_count
        p_a_and_no_basic = (
            C(non_basic_count, 7) - C(non_basic_count - card_count, 7)
        ) / C(deck_size, 7)

    return (p_a - p_a_and_no_basic) / p_legal


def p_card_in_hand_after_turn_draw_given_legal_opening(
    deck_size: int,
    basic_count: int,
    card_count: int,
    card_is_basic: bool,
) -> float:
    p_no_basic = p_no_basic_opening_7(deck_size, basic_count)
    p_legal = 1 - p_no_basic

    if p_legal <= 0:
        return float("nan")

    p_no_target_opening = C(deck_size - card_count, 7) / C(deck_size, 7)

    if card_is_basic:
        p_no_target_and_no_basic = p_no_basic
    else:
        non_basic_count = deck_size - basic_count
        p_no_target_and_no_basic = C(non_basic_count - card_count, 7) / C(
            deck_size, 7
        )

    p_no_target_and_legal = p_no_target_opening - p_no_target_and_no_basic

    p_turn_draw_not_target_given_no_target_opening = (
        deck_size - 7 - card_count
    ) / (deck_size - 7)

    p_no_target_after_turn_draw_and_legal = (
        p_no_target_and_legal * p_turn_draw_not_target_given_no_target_opening
    )

    return 1 - (p_no_target_after_turn_draw_and_legal / p_legal)


def mulligan_distribution_exact(q: float, max_mulligans: int = 6) -> pd.DataFrame:
    rows = []

    for k in range(max_mulligans):
        rows.append(
            {
                "mulligans": str(k),
                "probability": (q**k) * (1 - q),
            }
        )

    rows.append(
        {
            "mulligans": f"{max_mulligans}+",
            "probability": q**max_mulligans,
        }
    )

    return pd.DataFrame(rows)


# ============================================================
# Raw prize probabilities
# These are kept for reference, but app code should prefer the
# true legal-hand-conditioned functions below.
# ============================================================


def p_raw_at_least_one_prized(
    deck_size: int, card_count: int, prize_count: int = 6
) -> float:
    return 1 - C(deck_size - card_count, prize_count) / C(deck_size, prize_count)


def raw_expected_prized(deck_size: int, card_count: int, prize_count: int = 6) -> float:
    return prize_count * card_count / deck_size


def p_raw_exactly_x_prized(
    deck_size: int,
    card_count: int,
    x: int,
    prize_count: int = 6,
) -> float:
    return (
        C(card_count, x)
        * C(deck_size - card_count, prize_count - x)
        / C(deck_size, prize_count)
    )


def p_raw_all_copies_prized(
    deck_size: int, card_count: int, prize_count: int = 6
) -> float:
    if card_count > prize_count:
        return 0.0

    return p_raw_exactly_x_prized(
        deck_size=deck_size,
        card_count=card_count,
        x=card_count,
        prize_count=prize_count,
    )


# ============================================================
# True game-start prize probabilities
# conditioned on keeping a legal opening hand
# ============================================================


def legal_opening_hand_count(deck_size: int, basic_count: int, hand_size: int = 7) -> int:
    """Number of legal opening hands.

    A legal opening hand contains at least one Basic Pokémon.
    """
    return C(deck_size, hand_size) - C(deck_size - basic_count, hand_size)


def legal_hand_target_count_weights(
    deck_size: int,
    basic_count: int,
    card_count: int,
    card_is_basic: bool,
    hand_size: int = 7,
) -> list[tuple[int, float]]:
    """Distribution of target copies in kept opening hand.

    Returns a list of (target_count_in_hand, probability_given_legal_hand).
    """
    legal_total = legal_opening_hand_count(
        deck_size=deck_size,
        basic_count=basic_count,
        hand_size=hand_size,
    )

    if legal_total <= 0:
        return []

    max_target_in_hand = min(card_count, hand_size)
    weights = []

    for j in range(max_target_in_hand + 1):
        if card_is_basic:
            # If target is a Basic Pokémon:
            # - j > 0 means the hand is automatically legal.
            # - j = 0 means the hand must contain another Basic Pokémon.
            if j > 0:
                legal_hands_with_j = C(card_count, j) * C(
                    deck_size - card_count, hand_size - j
                )
            else:
                other_cards = deck_size - card_count
                no_basic_cards = deck_size - basic_count
                legal_hands_with_j = C(other_cards, hand_size) - C(
                    no_basic_cards, hand_size
                )
        else:
            # Choose j copies of the non-Basic target, then remaining cards
            # must include at least one Basic Pokémon.
            non_basic_non_target_count = deck_size - basic_count - card_count
            legal_hands_with_j = C(card_count, j) * (
                C(deck_size - card_count, hand_size - j)
                - C(non_basic_non_target_count, hand_size - j)
            )

        if legal_hands_with_j > 0:
            weights.append((j, legal_hands_with_j / legal_total))

    return weights


def p_at_least_one_prized_after_legal_hand(
    deck_size: int,
    basic_count: int,
    card_count: int,
    card_is_basic: bool,
    prize_count: int = 6,
    hand_size: int = 7,
) -> float:
    """Probability at least one target copy is prized after legal opener."""
    deck_after_hand = deck_size - hand_size
    total = 0.0

    for target_in_hand, weight in legal_hand_target_count_weights(
        deck_size=deck_size,
        basic_count=basic_count,
        card_count=card_count,
        card_is_basic=card_is_basic,
        hand_size=hand_size,
    ):
        remaining_target = card_count - target_in_hand

        if remaining_target <= 0:
            prize_probability = 0.0
        else:
            prize_probability = 1 - C(
                deck_after_hand - remaining_target, prize_count
            ) / C(deck_after_hand, prize_count)

        total += weight * prize_probability

    return total


def expected_prized_after_legal_hand(
    deck_size: int,
    basic_count: int,
    card_count: int,
    card_is_basic: bool,
    prize_count: int = 6,
    hand_size: int = 7,
) -> float:
    """Expected target copies prized after legal opener."""
    deck_after_hand = deck_size - hand_size
    total = 0.0

    for target_in_hand, weight in legal_hand_target_count_weights(
        deck_size=deck_size,
        basic_count=basic_count,
        card_count=card_count,
        card_is_basic=card_is_basic,
        hand_size=hand_size,
    ):
        remaining_target = card_count - target_in_hand
        expected_given_hand = prize_count * remaining_target / deck_after_hand
        total += weight * expected_given_hand

    return total


def p_all_copies_prized_after_legal_hand(
    deck_size: int,
    basic_count: int,
    card_count: int,
    card_is_basic: bool,
    prize_count: int = 6,
    hand_size: int = 7,
) -> float:
    """Probability all target copies are prized after legal opener."""
    if card_count > prize_count:
        return 0.0

    deck_after_hand = deck_size - hand_size
    total = 0.0

    for target_in_hand, weight in legal_hand_target_count_weights(
        deck_size=deck_size,
        basic_count=basic_count,
        card_count=card_count,
        card_is_basic=card_is_basic,
        hand_size=hand_size,
    ):
        if target_in_hand > 0:
            prize_probability = 0.0
        else:
            prize_probability = C(
                deck_after_hand - card_count, prize_count - card_count
            ) / C(deck_after_hand, prize_count)

        total += weight * prize_probability

    return total


def p_still_prized_after_x_prizes_taken_after_legal_hand(
    deck_size: int,
    basic_count: int,
    card_count: int,
    card_is_basic: bool,
    prizes_taken: int,
    starting_prize_count: int = 6,
    hand_size: int = 7,
) -> float:
    """Probability at least one target is still in remaining prizes."""
    remaining_prize_count = starting_prize_count - prizes_taken

    if remaining_prize_count <= 0:
        return 0.0

    return p_at_least_one_prized_after_legal_hand(
        deck_size=deck_size,
        basic_count=basic_count,
        card_count=card_count,
        card_is_basic=card_is_basic,
        prize_count=remaining_prize_count,
        hand_size=hand_size,
    )


def p_all_copies_still_prized_after_x_prizes_taken_after_legal_hand(
    deck_size: int,
    basic_count: int,
    card_count: int,
    card_is_basic: bool,
    prizes_taken: int,
    starting_prize_count: int = 6,
    hand_size: int = 7,
) -> float:
    """Probability all target copies are still in remaining prizes."""
    remaining_prize_count = starting_prize_count - prizes_taken

    if remaining_prize_count <= 0:
        return 0.0

    if card_count > remaining_prize_count:
        return 0.0

    return p_all_copies_prized_after_legal_hand(
        deck_size=deck_size,
        basic_count=basic_count,
        card_count=card_count,
        card_is_basic=card_is_basic,
        prize_count=remaining_prize_count,
        hand_size=hand_size,
    )


# ============================================================
# Multi-card Turn 1 goal exact baselines
# ============================================================


@dataclass(frozen=True)
class GoalCardOption:
    """One card option inside a goal group.

    Example:
    - Riolu as a single-card group: one option with count=3.
    - Riolu OR Makuhita: two options in one group.

    The exact math assumes the physical card pools represented by options are
    disjoint. This is true for normal deck goals using different card IDs.
    """

    card_id: str
    count: int
    is_basic: bool = False


@dataclass(frozen=True)
class GoalGroup:
    """A requirement group.

    A goal succeeds when every group is satisfied. A group is satisfied when at
    least `min_count` cards from that group's options are present.

    Examples:
    - X AND Y AND Z = three groups, each with min_count=1.
    - X OR Y = one group with two options and min_count=1.
    """

    name: str
    options: tuple[GoalCardOption, ...]
    min_count: int = 1


@dataclass(frozen=True)
class _HandState:
    group_draws: tuple[int, ...]
    total_group_drawn: int
    total_basic_drawn_from_groups: int
    ways: int


def normalize_goal_groups(goal_groups: Iterable[dict[str, Any] | GoalGroup]) -> list[GoalGroup]:
    """Normalize user/script goal specs into GoalGroup objects.

    Accepted dict shape:
    {
        "name": "Riolu OR Makuhita",
        "min_count": 1,
        "options": [
            {"card_id": "me1-76", "count": 3, "is_basic": True},
            {"card_id": "me1-72", "count": 2, "is_basic": True},
        ],
    }
    """
    normalized: list[GoalGroup] = []

    for i, group in enumerate(goal_groups):
        if isinstance(group, GoalGroup):
            normalized.append(group)
            continue

        options = []
        for option in group.get("options", []):
            count = int(option.get("count", 0))
            if count <= 0:
                continue
            options.append(
                GoalCardOption(
                    card_id=str(option.get("card_id", "")),
                    count=count,
                    is_basic=bool(option.get("is_basic", False)),
                )
            )

        normalized.append(
            GoalGroup(
                name=str(group.get("name") or f"goal_group_{i + 1}"),
                options=tuple(options),
                min_count=int(group.get("min_count", 1)),
            )
        )

    return normalized


def _group_total_count(group: GoalGroup) -> int:
    return sum(option.count for option in group.options)


def _group_basic_count(group: GoalGroup) -> int:
    return sum(option.count for option in group.options if option.is_basic)


def _validate_goal_groups(deck_size: int, basic_count: int, groups: list[GoalGroup]) -> None:
    grouped_count = sum(_group_total_count(group) for group in groups)
    grouped_basic_count = sum(_group_basic_count(group) for group in groups)

    if grouped_count > deck_size:
        raise ValueError("Goal groups contain more cards than the deck size.")

    if grouped_basic_count > basic_count:
        raise ValueError("Goal groups contain more Basic Pokémon than basic_count.")

    for group in groups:
        if group.min_count < 0:
            raise ValueError(f"Goal group {group.name!r} has negative min_count.")
        if group.min_count > _group_total_count(group):
            raise ValueError(
                f"Goal group {group.name!r} requires {group.min_count} cards, "
                f"but only {_group_total_count(group)} are available."
            )


def _enumerate_group_states(
    groups: list[GoalGroup],
    max_cards: int,
) -> list[_HandState]:
    """Enumerate ways to draw counts from goal groups.

    Does not include cards outside goal groups. Other cards are handled by the
    caller so legality can account for other Basic Pokémon.
    """
    states: list[_HandState] = []

    def rec(
        group_index: int,
        group_draws: list[int],
        total_group_drawn: int,
        total_basic_drawn: int,
        ways: int,
    ) -> None:
        if total_group_drawn > max_cards:
            return

        if group_index == len(groups):
            states.append(
                _HandState(
                    group_draws=tuple(group_draws),
                    total_group_drawn=total_group_drawn,
                    total_basic_drawn_from_groups=total_basic_drawn,
                    ways=ways,
                )
            )
            return

        group = groups[group_index]
        group_basic = _group_basic_count(group)
        group_non_basic = _group_total_count(group) - group_basic
        max_group_draw = min(_group_total_count(group), max_cards - total_group_drawn)

        for basic_drawn in range(0, min(group_basic, max_group_draw) + 1):
            max_non_basic_drawn = min(group_non_basic, max_group_draw - basic_drawn)
            for non_basic_drawn in range(max_non_basic_drawn + 1):
                drawn = basic_drawn + non_basic_drawn
                group_ways = C(group_basic, basic_drawn) * C(
                    group_non_basic, non_basic_drawn
                )
                if group_ways <= 0:
                    continue

                group_draws.append(drawn)
                rec(
                    group_index + 1,
                    group_draws,
                    total_group_drawn + drawn,
                    total_basic_drawn + basic_drawn,
                    ways * group_ways,
                )
                group_draws.pop()

    rec(0, [], 0, 0, 1)
    return states


def _groups_satisfied(group_draws: tuple[int, ...], groups: list[GoalGroup]) -> bool:
    return all(drawn >= group.min_count for drawn, group in zip(group_draws, groups))


def count_legal_opening_hands_satisfying_goal_groups(
    deck_size: int,
    basic_count: int,
    goal_groups: Iterable[dict[str, Any] | GoalGroup],
    hand_size: int = 7,
) -> int:
    """Exact count of legal opening hands satisfying an AND of goal groups."""
    groups = normalize_goal_groups(goal_groups)
    _validate_goal_groups(deck_size, basic_count, groups)

    grouped_count = sum(_group_total_count(group) for group in groups)
    grouped_basic_count = sum(_group_basic_count(group) for group in groups)
    other_count = deck_size - grouped_count
    other_basic_count = basic_count - grouped_basic_count
    other_non_basic_count = other_count - other_basic_count

    total = 0

    for state in _enumerate_group_states(groups, hand_size):
        if not _groups_satisfied(state.group_draws, groups):
            continue

        remaining_slots = hand_size - state.total_group_drawn
        if remaining_slots < 0:
            continue

        for other_basic_drawn in range(min(other_basic_count, remaining_slots) + 1):
            other_non_basic_drawn = remaining_slots - other_basic_drawn
            if other_non_basic_drawn < 0 or other_non_basic_drawn > other_non_basic_count:
                continue

            total_basic_drawn = state.total_basic_drawn_from_groups + other_basic_drawn
            if total_basic_drawn <= 0:
                continue

            total += (
                state.ways
                * C(other_basic_count, other_basic_drawn)
                * C(other_non_basic_count, other_non_basic_drawn)
            )

    return total


def p_goal_in_legal_opening_hand(
    deck_size: int,
    basic_count: int,
    goal_groups: Iterable[dict[str, Any] | GoalGroup],
    hand_size: int = 7,
) -> float:
    """Exact P(goal satisfied in opening hand | opening hand is legal)."""
    legal_total = legal_opening_hand_count(deck_size, basic_count, hand_size)
    success_count = count_legal_opening_hands_satisfying_goal_groups(
        deck_size=deck_size,
        basic_count=basic_count,
        goal_groups=goal_groups,
        hand_size=hand_size,
    )
    return _safe_div(success_count, legal_total)


def p_goal_after_turn_draw_given_legal_opening(
    deck_size: int,
    basic_count: int,
    goal_groups: Iterable[dict[str, Any] | GoalGroup],
    hand_size: int = 7,
) -> float:
    """Exact P(goal satisfied after draw for turn | kept opening hand is legal).

    The opening 7 must already be legal. The turn draw may complete the goal,
    but it does not retroactively make an illegal opener legal.
    """
    groups = normalize_goal_groups(goal_groups)
    _validate_goal_groups(deck_size, basic_count, groups)

    legal_total = legal_opening_hand_count(deck_size, basic_count, hand_size)
    if legal_total <= 0:
        return float("nan")

    grouped_count = sum(_group_total_count(group) for group in groups)
    grouped_basic_count = sum(_group_basic_count(group) for group in groups)
    other_count = deck_size - grouped_count
    other_basic_count = basic_count - grouped_basic_count
    other_non_basic_count = other_count - other_basic_count

    total_ordered_success = 0
    total_ordered_legal = legal_total * (deck_size - hand_size)

    for state in _enumerate_group_states(groups, hand_size):
        remaining_slots = hand_size - state.total_group_drawn
        if remaining_slots < 0:
            continue

        for other_basic_drawn in range(min(other_basic_count, remaining_slots) + 1):
            other_non_basic_drawn = remaining_slots - other_basic_drawn
            if other_non_basic_drawn < 0 or other_non_basic_drawn > other_non_basic_count:
                continue

            total_basic_drawn = state.total_basic_drawn_from_groups + other_basic_drawn
            if total_basic_drawn <= 0:
                continue

            hand_ways = (
                state.ways
                * C(other_basic_count, other_basic_drawn)
                * C(other_non_basic_count, other_non_basic_drawn)
            )

            if hand_ways <= 0:
                continue

            if _groups_satisfied(state.group_draws, groups):
                total_ordered_success += hand_ways * (deck_size - hand_size)
                continue

            # Goal is not satisfied by opening hand. Count draw-card choices
            # that complete the goal.
            completing_draw_choices = 0
            for i, group in enumerate(groups):
                if state.group_draws[i] + 1 < group.min_count:
                    continue

                new_draws = list(state.group_draws)
                new_draws[i] += 1
                if not _groups_satisfied(tuple(new_draws), groups):
                    continue

                remaining_group_cards = _group_total_count(group) - state.group_draws[i]
                completing_draw_choices += remaining_group_cards

            total_ordered_success += hand_ways * completing_draw_choices

    return _safe_div(total_ordered_success, total_ordered_legal)


def goal_natural_probabilities_given_legal_opening(
    deck_size: int,
    basic_count: int,
    goal_groups: Iterable[dict[str, Any] | GoalGroup],
    hand_size: int = 7,
) -> dict[str, float]:
    """Exact natural baseline for a multi-card Turn 1 goal.

    Returns probabilities before any actions are simulated:
    - goal already satisfied in legal opening hand
    - goal satisfied after natural draw for turn
    - the draw-for-turn increment
    """
    opening = p_goal_in_legal_opening_hand(
        deck_size=deck_size,
        basic_count=basic_count,
        goal_groups=goal_groups,
        hand_size=hand_size,
    )
    after_draw = p_goal_after_turn_draw_given_legal_opening(
        deck_size=deck_size,
        basic_count=basic_count,
        goal_groups=goal_groups,
        hand_size=hand_size,
    )

    return {
        "opening_hand_goal_probability": opening,
        "opening_hand_goal_percent": round(100 * opening, 4),
        "after_turn_draw_goal_probability": after_draw,
        "after_turn_draw_goal_percent": round(100 * after_draw, 4),
        "draw_for_turn_increment_probability": after_draw - opening,
        "draw_for_turn_increment_percent": round(100 * (after_draw - opening), 4),
    }


def goal_group_all_prized_probabilities_after_legal_hand(
    deck_size: int,
    basic_count: int,
    goal_groups: Iterable[dict[str, Any] | GoalGroup],
    prize_count: int = 6,
    hand_size: int = 7,
) -> list[dict[str, Any]]:
    """Prize-lockout probabilities for each goal group.

    For an OR group such as Riolu OR Makuhita, this asks whether all copies of
    every option in the group are prized after the legal opening hand. That is
    the true prize lockout for that OR requirement.

    This is a conservative diagnostic for Turn 1 goals; it does not say the
    whole goal fails, only that this specific requirement is unavailable from
    the deck/prizes after setup.
    """
    groups = normalize_goal_groups(goal_groups)
    _validate_goal_groups(deck_size, basic_count, groups)

    results = []
    for group in groups:
        group_count = _group_total_count(group)
        group_is_basic = _group_basic_count(group) == group_count

        # If min_count > 1, "all copies prized" is still useful but not the only
        # lockout mode. Most current Turn 1 goal groups use min_count=1.
        probability = p_all_copies_prized_after_legal_hand(
            deck_size=deck_size,
            basic_count=basic_count,
            card_count=group_count,
            card_is_basic=group_is_basic,
            prize_count=prize_count,
            hand_size=hand_size,
        )

        results.append(
            {
                "group": group.name,
                "min_count": group.min_count,
                "total_copies": group_count,
                "all_group_copies_prized_probability": probability,
                "all_group_copies_prized_percent": round(100 * probability, 4),
            }
        )

    return results


# ============================================================
# Backwards-compatible aliases
# These keep older imports from breaking, but they are raw models.
# New app code should use the true legal-hand-conditioned functions.
# ============================================================


def p_at_least_one_prized(
    deck_size: int, card_count: int, prize_count: int = 6
) -> float:
    return p_raw_at_least_one_prized(deck_size, card_count, prize_count)


def expected_prized(deck_size: int, card_count: int, prize_count: int = 6) -> float:
    return raw_expected_prized(deck_size, card_count, prize_count)


def p_exactly_x_prized(
    deck_size: int,
    card_count: int,
    x: int,
    prize_count: int = 6,
) -> float:
    return p_raw_exactly_x_prized(deck_size, card_count, x, prize_count)


def p_all_copies_prized(deck_size: int, card_count: int, prize_count: int = 6) -> float:
    return p_raw_all_copies_prized(deck_size, card_count, prize_count)


def p_still_prized_after_x_prizes_taken(
    deck_size: int,
    card_count: int,
    prizes_taken: int,
    starting_prize_count: int = 6,
) -> float:
    remaining_prize_count = starting_prize_count - prizes_taken

    if remaining_prize_count <= 0:
        return 0.0

    return p_raw_at_least_one_prized(
        deck_size=deck_size,
        card_count=card_count,
        prize_count=remaining_prize_count,
    )


def p_all_copies_still_prized_after_x_prizes_taken(
    deck_size: int,
    card_count: int,
    prizes_taken: int,
    starting_prize_count: int = 6,
) -> float:
    remaining_prize_count = starting_prize_count - prizes_taken

    if remaining_prize_count <= 0:
        return 0.0

    if card_count > remaining_prize_count:
        return 0.0

    return p_raw_all_copies_prized(
        deck_size=deck_size,
        card_count=card_count,
        prize_count=remaining_prize_count,
    )


# ============================================================
# Custom starting-hand statement probabilities
# ============================================================


def _route_required_counts(route: Iterable[int]) -> dict[int, int]:
    """Return required copy counts by selected-card index for one AND route.

    Repeated indices are meaningful. Example: [0, 0, 1] means at least
    two copies of selected card 0 and at least one copy of selected card 1.
    """
    required: dict[int, int] = {}
    for i in route:
        i = int(i)
        required[i] = required.get(i, 0) + 1
    return required


def _dnf_statement_satisfied(
    selected_draws: tuple[int, ...],
    success_routes: Iterable[Iterable[int]],
) -> bool:
    """Return True if any AND-route is satisfied.

    A route is a collection of selected-card indices. The full statement is an
    OR of routes. Each route is an AND of required cards.

    Examples:
      X AND Y           -> [[0, 1]]
      X OR Y            -> [[0], [1]]
      X AND X           -> [[0, 0]] requiring 2 copies of X
      (X AND X) OR Y    -> [[0, 0], [1]]
    """
    for route in success_routes:
        required = _route_required_counts(route)
        if not required:
            continue
        if all(0 <= i < len(selected_draws) and selected_draws[i] >= needed for i, needed in required.items()):
            return True
    return False


def _enumerate_selected_draw_states(
    card_counts: list[int],
    card_is_basic: list[bool],
    hand_size: int,
) -> list[tuple[tuple[int, ...], int, int, int]]:
    """Enumerate selected-card draw states for an opening hand.

    Returns tuples:
      (selected_draw_counts, selected_total_drawn, selected_basic_drawn, ways)
    """
    states: list[tuple[tuple[int, ...], int, int, int]] = []

    def rec(index: int, draws: list[int], total_drawn: int, basic_drawn: int, ways: int) -> None:
        if total_drawn > hand_size:
            return
        if index == len(card_counts):
            states.append((tuple(draws), total_drawn, basic_drawn, ways))
            return

        max_draw = min(int(card_counts[index]), hand_size - total_drawn)
        for k in range(max_draw + 1):
            draw_ways = C(int(card_counts[index]), k)
            if draw_ways <= 0:
                continue
            draws.append(k)
            rec(
                index + 1,
                draws,
                total_drawn + k,
                basic_drawn + (k if card_is_basic[index] else 0),
                ways * draw_ways,
            )
            draws.pop()

    rec(0, [], 0, 0, 1)
    return states


def p_dnf_statement_given_legal_opening(
    deck_size: int,
    basic_count: int,
    card_counts: Iterable[int],
    card_is_basic: Iterable[bool],
    success_routes: Iterable[Iterable[int]],
    hand_size: int = 7,
    include_turn_draw: bool = False,
) -> float:
    """Exact probability for a custom starting-hand Boolean statement.

    The statement is represented in disjunctive normal form:
      OR of AND-routes.

    Examples:
      X AND Y           -> [[0, 1]]
      X OR Y            -> [[0], [1]]
      X AND X           -> [[0, 0]] requiring 2 copies of X
      (X AND Y) OR Z    -> [[0, 1], [2]]
      (A AND B) OR (C AND D) -> [[0, 1], [2, 3]]

    The probability is conditioned on keeping a legal opening hand, meaning the
    opening 7 contains at least one Basic Pokémon. If include_turn_draw=True,
    the opening 7 must still be legal first, then one natural card is drawn.

    This intentionally models only natural opening-hand/draw access. It does not
    include search effects, Supporters, Items, Abilities, or other turn actions.
    """
    counts = [int(c) for c in card_counts]
    basics = [bool(x) for x in card_is_basic]
    routes = [tuple(int(i) for i in route) for route in success_routes if tuple(route)]

    if len(counts) != len(basics):
        raise ValueError("card_counts and card_is_basic must have the same length.")
    if len(counts) > 8:
        raise ValueError("Custom starting-hand statements support up to 8 selected cards.")
    if not routes:
        return float("nan")
    if any(c < 0 for c in counts):
        raise ValueError("Card counts must be non-negative.")
    if sum(counts) > deck_size:
        raise ValueError("Selected card counts exceed deck size.")


    for route in routes:
        required = _route_required_counts(route)
        for i, needed in required.items():
            if i < 0 or i >= len(counts):
                raise ValueError(f"Route contains invalid selected-card index {i}.")
            if needed > counts[i]:
                raise ValueError(
                    f"Statement requires {needed} copies of selected card index {i}, "
                    f"but the deck only contains {counts[i]}."
                )

    legal_total = legal_opening_hand_count(deck_size, basic_count, hand_size)
    if legal_total <= 0:
        return float("nan")

    selected_total_count = sum(counts)
    selected_basic_count = sum(c for c, is_basic in zip(counts, basics) if is_basic)
    other_basic_count = basic_count - selected_basic_count
    other_non_basic_count = deck_size - basic_count - (selected_total_count - selected_basic_count)

    if other_basic_count < 0 or other_non_basic_count < 0:
        raise ValueError("Selected card counts/basic flags are inconsistent with deck totals.")

    states = _enumerate_selected_draw_states(counts, basics, hand_size)

    if not include_turn_draw:
        success_total = 0
        for selected_draws, selected_total_drawn, selected_basic_drawn, ways in states:
            if not _dnf_statement_satisfied(selected_draws, routes):
                continue
            remaining_slots = hand_size - selected_total_drawn
            for other_basic_drawn in range(min(other_basic_count, remaining_slots) + 1):
                other_non_basic_drawn = remaining_slots - other_basic_drawn
                if other_non_basic_drawn < 0 or other_non_basic_drawn > other_non_basic_count:
                    continue
                if selected_basic_drawn + other_basic_drawn <= 0:
                    continue
                success_total += (
                    ways
                    * C(other_basic_count, other_basic_drawn)
                    * C(other_non_basic_count, other_non_basic_drawn)
                )
        return _safe_div(success_total, legal_total)

    # Ordered model: choose a legal opening hand, then one draw from the remaining deck.
    total_ordered_legal = legal_total * (deck_size - hand_size)
    success_ordered = 0

    for selected_draws, selected_total_drawn, selected_basic_drawn, ways in states:
        remaining_slots = hand_size - selected_total_drawn
        for other_basic_drawn in range(min(other_basic_count, remaining_slots) + 1):
            other_non_basic_drawn = remaining_slots - other_basic_drawn
            if other_non_basic_drawn < 0 or other_non_basic_drawn > other_non_basic_count:
                continue
            if selected_basic_drawn + other_basic_drawn <= 0:
                continue

            hand_ways = (
                ways
                * C(other_basic_count, other_basic_drawn)
                * C(other_non_basic_count, other_non_basic_drawn)
            )
            if hand_ways <= 0:
                continue

            if _dnf_statement_satisfied(selected_draws, routes):
                success_ordered += hand_ways * (deck_size - hand_size)
                continue

            completing_draw_choices = 0
            for i, count in enumerate(counts):
                remaining_i = count - selected_draws[i]
                if remaining_i <= 0:
                    continue
                new_draws = list(selected_draws)
                new_draws[i] += 1
                if _dnf_statement_satisfied(tuple(new_draws), routes):
                    completing_draw_choices += remaining_i

            success_ordered += hand_ways * completing_draw_choices

    return _safe_div(success_ordered, total_ordered_legal)


def custom_starting_hand_statement_probabilities(
    deck_size: int,
    basic_count: int,
    card_counts: Iterable[int],
    card_is_basic: Iterable[bool],
    success_routes: Iterable[Iterable[int]],
    hand_size: int = 7,
) -> dict[str, float]:
    """Return opening-only and draw-for-turn exact probabilities."""
    opening = p_dnf_statement_given_legal_opening(
        deck_size=deck_size,
        basic_count=basic_count,
        card_counts=card_counts,
        card_is_basic=card_is_basic,
        success_routes=success_routes,
        hand_size=hand_size,
        include_turn_draw=False,
    )
    after_draw = p_dnf_statement_given_legal_opening(
        deck_size=deck_size,
        basic_count=basic_count,
        card_counts=card_counts,
        card_is_basic=card_is_basic,
        success_routes=success_routes,
        hand_size=hand_size,
        include_turn_draw=True,
    )
    return {
        "opening_hand_probability": opening,
        "opening_hand_percent": round(100 * opening, 4),
        "after_turn_draw_probability": after_draw,
        "after_turn_draw_percent": round(100 * after_draw, 4),
        "draw_for_turn_increment_probability": after_draw - opening,
        "draw_for_turn_increment_percent": round(100 * (after_draw - opening), 4),
    }
