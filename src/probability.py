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
- Calculate prize-card probabilities:
  - probability at least one copy is prized
  - expected number of copies prized
  - probability all copies are prized

A legal opening hand means the 7-card hand contains at least one Basic Pokémon.
"""

import math
import pandas as pd


def C(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


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
        p_a_and_no_basic = (C(non_basic_count, 7) - C(non_basic_count - card_count, 7)) / C(
            deck_size, 7
        )

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

    p_turn_draw_not_target_given_no_target_opening = (deck_size - 7 - card_count) / (deck_size - 7)

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


def p_at_least_one_prized(deck_size: int, card_count: int, prize_count: int = 6) -> float:
    return 1 - C(deck_size - card_count, prize_count) / C(deck_size, prize_count)


def expected_prized(deck_size: int, card_count: int, prize_count: int = 6) -> float:
    return prize_count * card_count / deck_size


def p_exactly_x_prized(
    deck_size: int,
    card_count: int,
    x: int,
    prize_count: int = 6,
) -> float:
    return C(card_count, x) * C(deck_size - card_count, prize_count - x) / C(deck_size, prize_count)


def p_still_prized_after_x_prizes_taken(
    deck_size: int,
    card_count: int,
    prizes_taken: int,
    starting_prize_count: int = 6,
) -> float:
    """
    Probability that at least one copy of the card is still in the prize cards
    after the player has taken a certain number of prize cards.

    Example:
    - prizes_taken = 0 means the original 6 prizes.
    - prizes_taken = 1 means 5 prizes remain.
    - prizes_taken = 5 means 1 prize remains.

    This assumes the prizes taken are random with respect to the target card.
    """

    remaining_prizes = starting_prize_count - prizes_taken

    if remaining_prizes <= 0:
        return 0.0

    return 1 - C(deck_size - card_count, remaining_prizes) / C(deck_size, remaining_prizes)


def p_all_copies_prized(deck_size: int, card_count: int, prize_count: int = 6) -> float:
    if card_count > prize_count:
        return 0.0

    return p_exactly_x_prized(
        deck_size=deck_size,
        card_count=card_count,
        x=card_count,
        prize_count=prize_count,
    )
