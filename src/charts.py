"""
Explanation

This file creates the Plotly charts shown in the Streamlit app.

It does not calculate probabilities. It only takes the DataFrames created by
analysis.py and turns them into visual charts.

Main responsibilities:
- Build deck composition charts.
- Build mulligan probability charts.
- Build opening-hand versus after-turn-draw charts.
- Build prize-card probability charts.
- Build all-copies-prized charts.
- Build prize-after-X-prizes heatmaps.
- Build diagnostic scatter plots.

All charts use a consistent dark visual style.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLOR_SEQUENCE = [
    "#60A5FA",
    "#A78BFA",
    "#34D399",
    "#FBBF24",
    "#F87171",
    "#22D3EE",
    "#FB7185",
    "#C084FC",
]

PLOTLY_TEMPLATE = "plotly_dark"


def apply_dark_layout(fig, height=None):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#E5E7EB"),
        title=dict(
            font=dict(size=22, color="#F8FAFC"),
            x=0.02,
            xanchor="left",
        ),
        margin=dict(l=20, r=30, t=70, b=40),
        legend=dict(
            bgcolor="rgba(15,23,42,0)",
            bordercolor="rgba(148,163,184,0.15)",
            font=dict(color="#CBD5E1"),
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=13,
            font_color="#F8FAFC",
        ),
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.15)",
        zerolinecolor="rgba(148,163,184,0.25)",
        title_font=dict(color="#CBD5E1"),
        tickfont=dict(color="#CBD5E1"),
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.10)",
        zerolinecolor="rgba(148,163,184,0.25)",
        title_font=dict(color="#CBD5E1"),
        tickfont=dict(color="#CBD5E1"),
    )

    if height is not None:
        fig.update_layout(height=height)

    return fig


def make_deck_composition_chart(parsed_df: pd.DataFrame):
    deck_comp = (
        parsed_df.groupby("supertype", dropna=False)["count"]
        .sum()
        .reset_index()
        .rename(columns={"supertype": "Card type", "count": "Cards"})
    )

    deck_comp["Card type"] = deck_comp["Card type"].fillna("Unknown")

    fig = px.pie(
        deck_comp,
        names="Card type",
        values="Cards",
        hole=0.55,
        title="Deck composition",
        color_discrete_sequence=COLOR_SEQUENCE,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Cards: %{value}<br>Share: %{percent}<extra></extra>",
    )

    apply_dark_layout(fig, height=430)

    fig.update_layout(
        showlegend=True,
        annotations=[
            dict(
                text="60<br>cards",
                x=0.5,
                y=0.5,
                font_size=24,
                font_color="#F8FAFC",
                showarrow=False,
            )
        ],
    )

    return fig


def make_mulligan_chart(mulligan_df: pd.DataFrame):
    plot_df = mulligan_df.copy()
    plot_df["probability_pct"] = plot_df["probability"] * 100

    fig = px.bar(
        plot_df,
        x="mulligans",
        y="probability_pct",
        text="probability_pct",
        title="Mulligan distribution",
        labels={
            "mulligans": "Mulligans before legal hand",
            "probability_pct": "Probability (%)",
        },
        color_discrete_sequence=["#60A5FA"],
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        marker_line_width=0,
        hovertemplate="<b>%{x} mulligans</b><br>Probability: %{y:.2f}%<extra></extra>",
    )

    apply_dark_layout(fig, height=430)

    fig.update_layout(
        yaxis_range=[0, max(5, plot_df["probability_pct"].max() * 1.18)],
        xaxis_title="Mulligans",
        yaxis_title="Probability (%)",
    )

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
        title="Opening-hand access",
        labels={
            "card": "Card",
            "probability_pct": "Probability (%)",
            "metric": "Hand state",
        },
        color_discrete_sequence=["#60A5FA", "#A78BFA"],
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
    )

    apply_dark_layout(fig, height=max(650, top_n_cards * 34))

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Probability (%)",
        yaxis_title="",
    )

    return fig


def make_conditioning_effect_chart(card_odds_df: pd.DataFrame, top_n_cards: int = 25):
    plot_df = card_odds_df.copy()

    plot_df["conditioning_effect"] = (
        plot_df["P_in_legal_opening_7"]
        - plot_df["P_in_random_7_unconditioned"]
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
        title="Mulligan conditioning effect",
        labels={
            "card": "Card",
            "conditioning_effect_pct": "Change vs random 7 (%)",
        },
        color="conditioning_effect_pct",
        color_continuous_scale=["#F87171", "#94A3B8", "#34D399"],
    )

    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Change: %{x:.2f}%<extra></extra>",
    )

    apply_dark_layout(fig, height=max(650, top_n_cards * 34))

    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_title="",
        xaxis_title="Change vs random 7 (%)",
    )

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
        title="At least 1 prized copy",
        labels={
            "card": "Card",
            "prize_probability_pct": "P(at least 1 copy prized) (%)",
        },
        color_discrete_sequence=["#FBBF24"],
    )

    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>P(at least 1 prized): %{x:.2f}%<extra></extra>",
    )

    apply_dark_layout(fig, height=max(650, top_n_cards * 34))

    fig.update_layout(
        yaxis_title="",
        xaxis_title="Probability (%)",
        xaxis_range=[0, max(10, plot_df["prize_probability_pct"].max() * 1.15)],
    )

    return fig


def make_all_copies_prized_chart(prize_df: pd.DataFrame, top_n_cards: int = 25):
    plot_df = prize_df.copy()
    plot_df = plot_df.sort_values("P_all_copies_prized", ascending=False).head(top_n_cards)
    plot_df["all_copies_prized_pct"] = plot_df["P_all_copies_prized"] * 100

    fig = px.bar(
        plot_df.sort_values("all_copies_prized_pct", ascending=True),
        y="card",
        x="all_copies_prized_pct",
        orientation="h",
        text="all_copies_prized_pct",
        title="All copies prized",
        labels={
            "card": "Card",
            "all_copies_prized_pct": "P(all copies prized) (%)",
        },
        color_discrete_sequence=["#FB7185"],
    )

    fig.update_traces(
        texttemplate="%{text:.4f}%",
        textposition="outside",
        marker_line_width=0,
        hovertemplate="<b>%{y}</b><br>P(all copies prized): %{x:.4f}%<extra></extra>",
    )

    apply_dark_layout(fig, height=max(650, top_n_cards * 34))

    fig.update_layout(
        yaxis_title="",
        xaxis_title="Probability (%)",
        xaxis_range=[0, max(0.05, plot_df["all_copies_prized_pct"].max() * 1.15)],
    )

    return fig


def make_prize_survival_heatmap(prize_df: pd.DataFrame):
    heatmap_cols = [
        "P_still_prized_after_1_prize_taken",
        "P_still_prized_after_2_prizes_taken",
        "P_still_prized_after_3_prizes_taken",
        "P_still_prized_after_4_prizes_taken",
        "P_still_prized_after_5_prizes_taken",
    ]

    available_cols = [col for col in heatmap_cols if col in prize_df.columns]

    if not available_cols:
        fig = go.Figure()
        fig.update_layout(title="Still-prized probability after prizes taken")
        return apply_dark_layout(fig, height=500)

    plot_df = prize_df.copy()
    plot_df = plot_df.sort_values("P_at_least_1_prized", ascending=True)

    z_values = plot_df[available_cols].values * 100
    y_labels = plot_df["card"].tolist()
    x_labels = ["After 1", "After 2", "After 3", "After 4", "After 5"]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels[: len(available_cols)],
            y=y_labels,
            colorscale=[
                [0.0, "#0F172A"],
                [0.35, "#1D4ED8"],
                [0.7, "#7C3AED"],
                [1.0, "#F59E0B"],
            ],
            colorbar=dict(
                title="Probability (%)",
                tickfont=dict(color="#CBD5E1"),
                titlefont=dict(color="#CBD5E1"),
            ),
            hovertemplate="<b>%{y}</b><br>%{x} prizes taken<br>Still prized: %{z:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Chance a card is still prized after prizes taken",
        xaxis_title="Prizes taken",
        yaxis_title="",
    )

    apply_dark_layout(fig, height=max(650, len(plot_df) * 30))

    return fig


def make_access_vs_prize_scatter(card_odds_df: pd.DataFrame):
    plot_df = card_odds_df.copy()
    plot_df["turn_draw_pct"] = plot_df["P_in_hand_after_turn_draw"] * 100
    plot_df["prize_pct"] = plot_df["P_at_least_1_prized"] * 100

    fig = px.scatter(
        plot_df,
        x="prize_pct",
        y="turn_draw_pct",
        size="count",
        color="supertype",
        hover_name="card",
        title="Access vs prize liability",
        labels={
            "prize_pct": "P(at least 1 prized) (%)",
            "turn_draw_pct": "P(after turn draw) (%)",
            "supertype": "Card type",
            "count": "Copies",
        },
        color_discrete_sequence=COLOR_SEQUENCE,
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=1, color="rgba(248,250,252,0.55)"),
            opacity=0.88,
        )
    )

    apply_dark_layout(fig, height=620)

    fig.update_layout(
        xaxis_title="Prize liability (%)",
        yaxis_title="Opening access after turn draw (%)",
    )

    return fig