"""
Explanation

This is the main Streamlit app file.

It controls the website interface.

Main responsibilities:
- Show the app title and description.
- Provide a text box where the user pastes a decklist.
- Provide sliders for display options.
- Run the deck analysis when the user clicks the Analyze button.
- Display:
  - deck summary
  - mulligan probabilities
  - opening-hand and turn-draw odds
  - prize-card probabilities
  - mulligan conditioning effect
  - parsed card matching table

Most of the actual logic is imported from the src/ folder.
"""

import streamlit as st

from src.analysis import analyze_deck_opening_hand, format_probability_table
from src.charts import (
    make_mulligan_chart,
    make_card_odds_chart,
    make_conditioning_effect_chart,
    make_prize_chart,
    make_all_copies_prized_chart,
)

EXAMPLE_DECKLIST = """Pokémon: 5
4 Teal Mask Ogerpon ex TWM 25
1 Latias ex SSP 76
4 Raging Bolt ex TEF 123
1 Fezandipiti ex ASC 142
1 Raging Bolt SCR 111

Trainer: 10
1 Energy Switch SVI 173
4 Crispin SCR 133
2 Lillie's Determination MEG 119
2 Dusk Ball SSP 175
2 Boss's Orders MEG 114
4 Ultra Ball MEG 131
4 Pokégear 3.0 SVI 186
1 Prime Catcher TEF 157
4 Tera Orb SSP 189
4 Bug Catching Set TWM 143

Energy: 3
13 Basic {G} Energy Energy 1
4 Basic {F} Energy Energy 14
4 Basic {L} Energy Energy 12

Total Cards: 60
"""


st.set_page_config(
    page_title="TCG Deck Analyzer",
    layout="wide",
)

st.title("TCG Deck Analyzer")
st.caption("Version 0.1 — exact mulligan, opening hand, turn draw, and prize probabilities")

st.markdown("""
Paste a 60-card decklist. This first version calculates:

- exact odds of mulliganing 0, 1, 2, 3, ... times
- exact odds of each card appearing in your legal opening 7
- exact odds of each card being in hand after drawing for turn
- exact odds of each card being prized

A legal opening hand means the 7-card hand contains at least one Basic Pokémon.
""")

decklist_text = st.text_area(
    "Paste decklist",
    value=EXAMPLE_DECKLIST,
    height=420,
)

max_mulligans = st.slider("Show mulligans up to", 3, 10, 6)

analyze = st.button("Analyze deck", type="primary")

if analyze:
    try:
        with st.spinner("Analyzing deck..."):
            summary, mulligan_df, card_odds_df, prize_df, parsed_df, deck = (
                analyze_deck_opening_hand(
                    decklist_text,
                    max_mulligans=max_mulligans,
                )
            )

        # Show all unique cards in every chart.
        top_n_cards = len(card_odds_df)

        st.subheader("Deck summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Deck size", summary["deck_size"])
        c2.metric("Basic Pokémon", summary["basic_count"])
        c3.metric("Mulligan chance", f"{summary['p_mulligan_one_hand']:.2%}")
        c4.metric("Expected mulligans", f"{summary['expected_mulligans_before_legal_hand']:.3f}")

        if summary["deck_size"] != 60:
            st.warning(f"Deck has {summary['deck_size']} cards, not 60.")

        st.subheader("Mulligan probabilities")
        st.plotly_chart(make_mulligan_chart(mulligan_df), use_container_width=True)
        st.dataframe(format_probability_table(mulligan_df), use_container_width=True)

        st.subheader("Opening hand and turn draw odds")
        st.plotly_chart(
            make_card_odds_chart(card_odds_df, top_n_cards=top_n_cards),
            use_container_width=True,
        )
        st.dataframe(format_probability_table(card_odds_df), use_container_width=True)

        st.subheader("Prize card probabilities")
        st.plotly_chart(
            make_prize_chart(prize_df, top_n_cards=top_n_cards),
            use_container_width=True,
        )

        st.subheader("Probability all copies are prized")
        st.plotly_chart(
            make_all_copies_prized_chart(prize_df, top_n_cards=top_n_cards),
            use_container_width=True,
        )

        st.subheader("Prize probability table")
        st.caption(
            "The 'still prized after X prizes taken' columns assume the prizes taken are random "
            "with respect to the target card."
        )
        st.dataframe(format_probability_table(prize_df), use_container_width=True)

        st.subheader("Mulligan conditioning effect")
        st.markdown("""
Positive means the card is more likely in a legal opening hand than in a random 7-card hand.
Basic Pokémon usually increase because no-Basic hands are thrown back.
""")
        st.plotly_chart(
            make_conditioning_effect_chart(card_odds_df, top_n_cards=top_n_cards),
            use_container_width=True,
        )

        st.subheader("Parsed card matching")
        st.dataframe(parsed_df, use_container_width=True)

    except Exception as e:
        st.error(str(e))
