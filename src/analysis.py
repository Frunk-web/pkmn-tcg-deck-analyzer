"""
Explanation

This file connects the parser, API metadata, and probability formulas.

It is the main analysis layer.

Main responsibilities:
- Parse the pasted decklist.
- Attach Pokémon TCG API metadata to each card.
- Count the deck size.
- Count the number of Basic Pokémon.
- Calculate mulligan probabilities.
- Calculate card odds for:
  - random 7-card hand
  - legal opening 7-card hand
  - hand after drawing for turn
- Calculate true game-start prize-card probabilities conditioned on keeping a legal opening hand.
- Calculate still-prized probabilities after prizes are taken.
- Include card image URLs in the output DataFrames.
- Build clean pandas DataFrames for the Streamlit app to display.
"""

import pandas as pd

from src.deck_parser import parse_decklist
from src.pokemon_api import attach_metadata
from src.probability import (
    p_no_basic_opening_7,
    p_card_in_random_7,
    p_card_in_legal_opening_7,
    p_card_in_hand_after_turn_draw_given_legal_opening,
    mulligan_distribution_exact,
    p_at_least_one_prized_after_legal_hand,
    expected_prized_after_legal_hand,
    p_all_copies_prized_after_legal_hand,
    p_still_prized_after_x_prizes_taken_after_legal_hand,
    p_all_copies_still_prized_after_x_prizes_taken_after_legal_hand,
)


def format_probability_table(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()

    for col in formatted.columns:
        if col.startswith("P_") or col == "probability" or col == "increase_from_turn_draw":
            formatted[col] = formatted[col].map(
                lambda x: "" if pd.isna(x) else (
                    f"{x:.4%}" if "all_copies" in col else f"{x:.2%}"
                )
            )

        if col.startswith("E_"):
            formatted[col] = formatted[col].map(
                lambda x: "" if pd.isna(x) else f"{x:.3f}"
            )

    return formatted


def analyze_deck_opening_hand(decklist_text: str, max_mulligans: int = 6):
    deck = parse_decklist(decklist_text)
    deck = attach_metadata(deck)

    deck_size = sum(card.count for card in deck)
    basic_count = sum(card.count for card in deck if card.is_basic_pokemon)

    if basic_count == 0:
        raise ValueError("No Basic Pokémon detected. Check the parsed card matching table.")

    q = p_no_basic_opening_7(deck_size, basic_count)
    mulligan_df = mulligan_distribution_exact(q, max_mulligans=max_mulligans)

    rows = []

    for card in deck:
        p_opening_7 = p_card_in_legal_opening_7(
            deck_size=deck_size,
            basic_count=basic_count,
            card_count=card.count,
            card_is_basic=card.is_basic_pokemon,
        )

        p_after_turn_draw = p_card_in_hand_after_turn_draw_given_legal_opening(
            deck_size=deck_size,
            basic_count=basic_count,
            card_count=card.count,
            card_is_basic=card.is_basic_pokemon,
        )

        naive_p_opening_7 = p_card_in_random_7(deck_size, card.count)

        # True game-start prize probabilities.
        # These are conditioned on first keeping a legal opening hand.
        p_prized = p_at_least_one_prized_after_legal_hand(
            deck_size=deck_size,
            basic_count=basic_count,
            card_count=card.count,
            card_is_basic=card.is_basic_pokemon,
            prize_count=6,
        )

        e_prized = expected_prized_after_legal_hand(
            deck_size=deck_size,
            basic_count=basic_count,
            card_count=card.count,
            card_is_basic=card.is_basic_pokemon,
            prize_count=6,
        )

        p_all_prized = p_all_copies_prized_after_legal_hand(
            deck_size=deck_size,
            basic_count=basic_count,
            card_count=card.count,
            card_is_basic=card.is_basic_pokemon,
            prize_count=6,
        )

        still_prized_values = {}
        all_copies_still_prized_values = {}

        for prizes_taken in range(1, 6):
            suffix = "prize_taken" if prizes_taken == 1 else "prizes_taken"

            still_prized_values[f"P_still_prized_after_{prizes_taken}_{suffix}"] = (
                p_still_prized_after_x_prizes_taken_after_legal_hand(
                    deck_size=deck_size,
                    basic_count=basic_count,
                    card_count=card.count,
                    card_is_basic=card.is_basic_pokemon,
                    prizes_taken=prizes_taken,
                    starting_prize_count=6,
                )
            )

            all_copies_still_prized_values[
                f"P_all_copies_still_prized_after_{prizes_taken}_{suffix}"
            ] = p_all_copies_still_prized_after_x_prizes_taken_after_legal_hand(
                deck_size=deck_size,
                basic_count=basic_count,
                card_count=card.count,
                card_is_basic=card.is_basic_pokemon,
                prizes_taken=prizes_taken,
                starting_prize_count=6,
            )

        rows.append(
            {
                "card": card.label,
                "name": card.name,
                "count": card.count,
                "section": card.section,
                "api_id": card.api_id,
                "supertype": card.supertype,
                "subtypes": ", ".join(card.subtypes or []),
                "is_basic_pokemon": card.is_basic_pokemon,
                "image_url": card.image_url,
                "image_large_url": card.image_large_url,

                "P_in_random_7_unconditioned": naive_p_opening_7,
                "P_in_legal_opening_7": p_opening_7,
                "P_in_hand_after_turn_draw": p_after_turn_draw,
                "increase_from_turn_draw": p_after_turn_draw - p_opening_7,

                "P_at_least_1_prized": p_prized,
                "E_prized": e_prized,
                "P_all_copies_prized": p_all_prized,

                **still_prized_values,
                **all_copies_still_prized_values,
            }
        )

    card_odds_df = pd.DataFrame(rows).sort_values(
        by="P_in_hand_after_turn_draw",
        ascending=False,
    )

    prize_cols = [
        "card",
        "name",
        "count",
        "section",
        "supertype",
        "image_url",
        "image_large_url",
        "P_at_least_1_prized",
        "E_prized",
        "P_all_copies_prized",
        "P_still_prized_after_1_prize_taken",
        "P_still_prized_after_2_prizes_taken",
        "P_still_prized_after_3_prizes_taken",
        "P_still_prized_after_4_prizes_taken",
        "P_still_prized_after_5_prizes_taken",
        "P_all_copies_still_prized_after_1_prize_taken",
        "P_all_copies_still_prized_after_2_prizes_taken",
        "P_all_copies_still_prized_after_3_prizes_taken",
        "P_all_copies_still_prized_after_4_prizes_taken",
        "P_all_copies_still_prized_after_5_prizes_taken",
    ]

    existing_prize_cols = [col for col in prize_cols if col in card_odds_df.columns]

    prize_df = card_odds_df[existing_prize_cols].sort_values(
        by="P_at_least_1_prized",
        ascending=False,
    )

    parsed_df = pd.DataFrame(
        [
            {
                "card": card.label,
                "count": card.count,
                "section": card.section,
                "api_id": card.api_id,
                "supertype": card.supertype,
                "subtypes": ", ".join(card.subtypes or []),
                "is_basic_pokemon": card.is_basic_pokemon,
                "image_url": card.image_url,
                "image_large_url": card.image_large_url,
            }
            for card in deck
        ]
    )

    summary = {
        "deck_size": deck_size,
        "basic_count": basic_count,
        "p_mulligan_one_hand": q,
        "p_opening_hand_legal": 1 - q,
        "expected_mulligans_before_legal_hand": q / (1 - q) if q < 1 else float("inf"),
    }

    return summary, mulligan_df, card_odds_df, prize_df, parsed_df, deck