"""
Explanation

This file contains the exact probability formulas for the app.

It does not use Streamlit and does not call the Pokémon TCG API.
It is pure math.

Main responsibilities:
- Calculate the probability of a no-Basic opening hand.
- Calculate the mulligan distribution.
- Calculate the probability of seeing each card in a random 7-card hand.
- Calculate the probability of seeing each card in a legal opening 7-card hand.
- Calculate the probability of seeing each card after drawing for turn.
- Calculate true game-start prize-card probabilities conditioned on keeping a legal opening hand:
  - probability at least one copy is prized
  - expected number of copies prized
  - probability all copies are prized
  - probability at least one copy is still prized after X prizes are taken
  - probability all copies are still prized after X prizes are taken

A legal opening hand means the 7-card hand contains at least one Basic Pokémon.

Important distinction:
The prize probabilities here are NOT just "6 random cards from the full 60-card deck."
They are conditioned on the player having kept a legal opening hand first. This matters
because Basic Pokémon are more likely to appear in the kept opening hand, and cards in
the opening hand cannot be prized.
"""

import math
import pandas as pd


def C(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


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
            C(non_basic_count, 7)
            - C(non_basic_count - card_count, 7)
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
        p_no_target_and_no_basic = C(non_basic_count - card_count, 7) / C(deck_size, 7)

    p_no_target_and_legal = p_no_target_opening - p_no_target_and_no_basic

    p_turn_draw_not_target_given_no_target_opening = (
        deck_size - 7 - card_count
    ) / (deck_size - 7)

    p_no_target_after_turn_draw_and_legal = (
        p_no_target_and_legal
        * p_turn_draw_not_target_given_no_target_opening
    )

    return 1 - (p_no_target_after_turn_draw_and_legal / p_legal)


def mulligan_distribution_exact(q: float, max_mulligans: int = 6) -> pd.DataFrame:
    rows = []

    for k in range(max_mulligans):
        rows.append(
            {
                "mulligans": str(k),
                "probability": (q ** k) * (1 - q),
            }
        )

    rows.append(
        {
            "mulligans": f"{max_mulligans}+",
            "probability": q ** max_mulligans,
        }
    )

    return pd.DataFrame(rows)


# ============================================================
# Raw prize probabilities
# These are kept for reference, but the app should use the
# true legal-hand-conditioned functions below.
# ============================================================

def p_raw_at_least_one_prized(deck_size: int, card_count: int, prize_count: int = 6) -> float:
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


def p_raw_all_copies_prized(deck_size: int, card_count: int, prize_count: int = 6) -> float:
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
    """
    Number of legal opening hands.
    A legal opening hand contains at least one Basic Pokémon.
    """

    return C(deck_size, hand_size) - C(deck_size - basic_count, hand_size)


def legal_hand_target_count_weights(
    deck_size: int,
    basic_count: int,
    card_count: int,
    card_is_basic: bool,
    hand_size: int = 7,
):
    """
    Returns a list of (target_count_in_hand, probability_given_legal_hand).

    This is the exact distribution of how many copies of a target card are in
    the kept opening hand, conditional on that hand being legal.

    This is the key step needed for true game-start prize probabilities.
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
            # If the target is a Basic Pokémon:
            # - If j > 0, the hand is automatically legal.
            # - If j = 0, the hand must contain at least one other Basic Pokémon.
            if j > 0:
                legal_hands_with_j = C(card_count, j) * C(deck_size - card_count, hand_size - j)
            else:
                other_cards = deck_size - card_count
                no_basic_cards = deck_size - basic_count
                legal_hands_with_j = C(other_cards, hand_size) - C(no_basic_cards, hand_size)

        else:
            # If the target is not a Basic Pokémon:
            # Choose j copies of the target, then the rest of the hand must
            # contain at least one Basic Pokémon.
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
    """
    True game-start probability that at least one copy of the target card is prized,
    after conditioning on keeping a legal opening hand.
    """

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
            prize_probability = (
                1 - C(deck_after_hand - remaining_target, prize_count)
                / C(deck_after_hand, prize_count)
            )

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
    """
    True expected number of copies prized after conditioning on a legal opening hand.
    """

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
    """
    True game-start probability that all copies of the target card are prized,
    after conditioning on keeping a legal opening hand.

    If any copy is in the opening hand, all copies cannot be prized.
    """

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
            prize_probability = (
                C(deck_after_hand - card_count, prize_count - card_count)
                / C(deck_after_hand, prize_count)
            )

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
    """
    True probability that at least one copy of the target card is still in the
    remaining prize cards after X prizes have been taken, conditioned on keeping
    a legal opening hand.

    Taking X random prizes from the original 6 means there are:
    starting_prize_count - prizes_taken
    prize cards remaining.
    """

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
    """
    True probability that all copies of the target card are still in the
    remaining prize cards after X prizes have been taken, conditioned on keeping
    a legal opening hand.

    If fewer prize cards remain than the number of copies of the card,
    the probability is 0.
    """

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
# Backwards-compatible aliases
# These keep older imports from breaking, but they are raw models.
# New app code should use the true legal-hand-conditioned functions.
# ============================================================

def p_at_least_one_prized(deck_size: int, card_count: int, prize_count: int = 6) -> float:
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