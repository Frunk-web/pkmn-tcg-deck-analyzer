"""
Explanation

This file creates the Plotly charts shown in the Streamlit app.

It does not calculate probabilities. It only takes the DataFrames created by
analysis.py and turns them into visual charts.

Main responsibilities:
- Build the mulligan probability chart.
- Build the opening-hand versus after-turn-draw chart.
- Build the mulligan conditioning effect chart.
- Build the prize-risk chart.
"""

import pandas as pd
import plotly.express as px


def make_mulligan_chart(mulligan_df: pd.DataFrame):
    plot_df = mulligan_df.copy()
    plot_df["probability_pct"] = plot_df["probability"] * 100

    fig = px.bar(
        plot_df,
        x="mulligans",
        y="probability_pct",
        text="probability_pct",
        title="Exact mulligan distribution",
        labels={
            "mulligans": "Mulligans before legal hand",
            "probability_pct": "Probability (%)",
        },
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(yaxis_range=[0, max(5, plot_df["probability_pct"].max() * 1.15)])

    return fig


def make_card_odds_chart(card_odds_df: pd.DataFrame, top_n_cards: int = 25):
    top_cards = card_odds_df.head(top_n_cards).copy()

    plot_df = top_cards.melt(
        id_vars=["card"],
        value_vars=["P_in_legal_opening_7", "P_in_hand_after_turn_draw"],
        var_name="metric",
        value_name="probability",
    )

    plot_df["metric"] = plot_df["metric"].replace(
        {
            "P_in_legal_opening_7": "Legal opening 7",
            "P_in_hand_after_turn_draw": "After turn draw",
        }
    )

    plot_df["probability_pct"] = plot_df["probability"] * 100

    fig = px.bar(
        plot_df,
        y="card",
        x="probability_pct",
        color="metric",
        orientation="h",
        barmode="group",
        title=f"All card odds: opening 7 vs after drawing for turn",
        labels={
            "card": "Card",
            "probability_pct": "Probability (%)",
            "metric": "Hand state",
        },
    )

    fig.update_layout(
        height=max(600, top_n_cards * 32),
        yaxis={"categoryorder": "total ascending"},
    )

    return fig


def make_conditioning_effect_chart(card_odds_df: pd.DataFrame, top_n_cards: int = 25):
    plot_df = card_odds_df.copy()

    plot_df["conditioning_effect"] = (
        plot_df["P_in_legal_opening_7"] - plot_df["P_in_random_7_unconditioned"]
    )

    plot_df = plot_df.reindex(
        plot_df["conditioning_effect"].abs().sort_values(ascending=False).index
    ).head(top_n_cards)

    plot_df["conditioning_effect_pct"] = plot_df["conditioning_effect"] * 100

    fig = px.bar(
        plot_df.sort_values("conditioning_effect_pct", ascending=True),
        y="card",
        x="conditioning_effect_pct",
        orientation="h",
        title="How mulligan conditioning changes opening-hand odds",
        labels={
            "card": "Card",
            "conditioning_effect_pct": "Change vs random 7 (%)",
        },
    )

    fig.update_layout(height=max(600, top_n_cards * 32))

    return fig


def make_prize_chart(prize_df: pd.DataFrame, top_n_cards: int = 25):
    plot_df = prize_df.head(top_n_cards).copy()
    plot_df["prize_probability_pct"] = plot_df["P_at_least_1_prized"] * 100

    fig = px.bar(
        plot_df.sort_values("prize_probability_pct", ascending=True),
        y="card",
        x="prize_probability_pct",
        orientation="h",
        text="prize_probability_pct",
        title=f"At least 1 prized copy",
        labels={
            "card": "Card",
            "prize_probability_pct": "P(at least one copy prized) (%)",
        },
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(height=max(600, top_n_cards * 32))

    return fig


def make_all_copies_prized_chart(prize_df: pd.DataFrame, top_n_cards: int = 25):
    plot_df = prize_df.copy()

    # Usually only 1-ofs and 2-ofs have meaningful "all copies prized" odds,
    # so sort by that probability.
    plot_df = plot_df.sort_values("P_all_copies_prized", ascending=False).head(top_n_cards)

    plot_df["all_copies_prized_pct"] = plot_df["P_all_copies_prized"] * 100

    fig = px.bar(
        plot_df.sort_values("all_copies_prized_pct", ascending=True),
        y="card",
        x="all_copies_prized_pct",
        orientation="h",
        text="all_copies_prized_pct",
        title="Probability all copies are prized",
        labels={
            "card": "Card",
            "all_copies_prized_pct": "P(all copies prized) (%)",
        },
    )

    fig.update_traces(texttemplate="%{text:.4f}%", textposition="outside")
    fig.update_layout(height=max(600, top_n_cards * 32))

    return fig
