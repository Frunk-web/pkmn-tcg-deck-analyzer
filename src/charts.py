"""
Explanation

This file creates the Plotly charts shown in the Streamlit app.

It does not calculate probabilities. It only takes the DataFrames created by
analysis.py and turns them into visual charts.

Main responsibilities:
- Build the deck composition chart.
- Build the mulligan probability chart.
- Build the opening-hand versus after-turn-draw chart.
- Build the mulligan conditioning effect chart.
- Build the at-least-1-prized chart.
- Build the all-copies-prized chart.
- Build the still-prized-after-prizes-taken heatmap.

Mobile improvements:
- Shortens long card names on chart axes while keeping full names in hover text.
- Uses smaller chart heights so mobile pages are less stretched.
- Uses percentage labels directly on bar charts.
- Places labels inside bars when there is room and outside when bars are small.
- Adds extra right margin so outside labels are not clipped.
- Keeps all charts compatible with horizontal scrolling from app.py.
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


def shorten_label(label: str, max_len: int = 24) -> str:
    label = str(label)

    if len(label) <= max_len:
        return label

    return label[: max_len - 1].rstrip() + "…"


def add_short_labels(
    df: pd.DataFrame,
    source_col: str = "card",
    max_len: int = 24,
) -> pd.DataFrame:
    plot_df = df.copy()
    plot_df["card_short"] = plot_df[source_col].map(
        lambda x: shorten_label(x, max_len=max_len)
    )
    return plot_df


def chart_height(
    row_count: int,
    base: int = 390,
    per_row: int = 26,
    max_height: int = 720,
) -> int:
    return min(max(base, row_count * per_row), max_height)


def percent_text(value: float, decimals: int = 2, signed: bool = False) -> str:
    if pd.isna(value):
        return ""

    if signed:
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.{decimals}f}%"

    return f"{value:.{decimals}f}%"


def padded_range(values, lower_floor=0, upper_floor=5, padding_factor=1.20):
    clean_values = [float(v) for v in values if pd.notna(v)]

    if not clean_values:
        return [lower_floor, upper_floor]

    min_value = min(clean_values)
    max_value = max(clean_values)

    if min_value >= 0:
        return [lower_floor, max(upper_floor, max_value * padding_factor)]

    max_abs = max(abs(min_value), abs(max_value))
    padding = max_abs * 0.25
    return [min_value - padding, max_value + padding]


def smart_text_positions(values, inside_threshold: float = 18.0):
    """
    For horizontal bar charts:
    - Put labels inside large bars.
    - Put labels outside small bars so they remain readable.
    """

    positions = []

    for value in values:
        if pd.isna(value):
            positions.append("outside")
        elif abs(float(value)) >= inside_threshold:
            positions.append("inside")
        else:
            positions.append("outside")

    return positions


def apply_dark_layout(fig, height=None, compact=True):
    title_size = 20 if compact else 23
    tick_size = 10 if compact else 12
    axis_title_size = 12 if compact else 14

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.35)",
        font=dict(color="#E5E7EB", size=12),
        title=dict(
            font=dict(size=title_size, color="#F8FAFC"),
            x=0.02,
            xanchor="left",
            y=0.97,
        ),
        margin=dict(l=8, r=72, t=58, b=40),
        legend=dict(
            bgcolor="rgba(15,23,42,0)",
            bordercolor="rgba(148,163,184,0.15)",
            font=dict(color="#CBD5E1", size=10),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            title_text="",
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_color="#F8FAFC",
        ),
        uniformtext_minsize=8,
        uniformtext_mode="show",
    )

    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.14)",
        zerolinecolor="rgba(148,163,184,0.24)",
        title_font=dict(color="#CBD5E1", size=axis_title_size),
        tickfont=dict(color="#CBD5E1", size=tick_size),
        automargin=True,
    )

    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.08)",
        zerolinecolor="rgba(148,163,184,0.24)",
        title_font=dict(color="#CBD5E1", size=axis_title_size),
        tickfont=dict(color="#CBD5E1", size=tick_size),
        automargin=True,
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
    total_cards = int(deck_comp["Cards"].sum())

    fig = px.pie(
        deck_comp,
        names="Card type",
        values="Cards",
        hole=0.58,
        title="Deck composition",
        color_discrete_sequence=COLOR_SEQUENCE,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>Cards: %{value}<br>Share: %{percent}<extra></extra>",
    )

    apply_dark_layout(fig, height=340, compact=True)

    fig.update_layout(
        showlegend=True,
        margin=dict(l=4, r=4, t=54, b=16),
        annotations=[
            dict(
                text=f"{total_cards}<br>cards",
                x=0.5,
                y=0.5,
                font_size=21,
                font_color="#F8FAFC",
                showarrow=False,
            )
        ],
    )

    return fig


def make_mulligan_chart(mulligan_df: pd.DataFrame):
    plot_df = mulligan_df.copy()
    plot_df["probability_pct"] = plot_df["probability"] * 100
    plot_df["probability_label"] = plot_df["probability_pct"].map(
        lambda x: percent_text(x, decimals=2)
    )

    fig = px.bar(
        plot_df,
        x="mulligans",
        y="probability_pct",
        text="probability_label",
        title="Mulligan distribution",
        labels={
            "mulligans": "Mulligans",
            "probability_pct": "Probability (%)",
        },
        color_discrete_sequence=["#60A5FA"],
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=11, color="#F8FAFC"),
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="<b>%{x} mulligans</b><br>Probability: %{text}<extra></extra>",
    )

    apply_dark_layout(fig, height=330, compact=True)

    fig.update_layout(
        yaxis_range=padded_range(plot_df["probability_pct"], upper_floor=5, padding_factor=1.18),
        xaxis_title="Mulligans",
        yaxis_title="Probability (%)",
        margin=dict(l=8, r=24, t=56, b=40),
    )

    return fig


def make_card_odds_chart(card_odds_df: pd.DataFrame, top_n_cards: int = 25):
    top_cards = card_odds_df.head(top_n_cards).copy()
    top_cards = add_short_labels(top_cards, max_len=24)

    plot_df = top_cards.melt(
        id_vars=["card", "card_short"],
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
    plot_df["probability_label"] = plot_df["probability_pct"].map(
        lambda x: percent_text(x, decimals=2)
    )

    fig = px.bar(
        plot_df,
        y="card_short",
        x="probability_pct",
        color="metric",
        orientation="h",
        barmode="group",
        text="probability_label",
        title="Opening-hand access",
        labels={
            "card_short": "",
            "probability_pct": "Probability (%)",
            "metric": "Hand state",
        },
        color_discrete_sequence=["#60A5FA", "#A78BFA"],
        custom_data=["card", "metric", "probability_label"],
    )

    for trace in fig.data:
        trace.textposition = smart_text_positions(trace.x, inside_threshold=22)
        trace.textfont = dict(size=9, color="#F8FAFC")
        trace.marker.line.width = 0
        trace.cliponaxis = False
        trace.hovertemplate = (
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}: %{customdata[2]}"
            "<extra></extra>"
        )

    apply_dark_layout(
        fig,
        height=chart_height(len(top_cards), base=430, per_row=30, max_height=760),
        compact=True,
    )

    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Probability (%)",
        yaxis_title="",
        legend_title_text="",
        margin=dict(l=4, r=96, t=78, b=48),
    )

    fig.update_xaxes(
        range=padded_range(plot_df["probability_pct"], upper_floor=10, padding_factor=1.30)
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

    plot_df = add_short_labels(plot_df, max_len=24)
    plot_df["conditioning_effect_pct"] = plot_df["conditioning_effect"] * 100
    plot_df["conditioning_effect_label"] = plot_df["conditioning_effect_pct"].map(
        lambda x: percent_text(x, decimals=2, signed=True)
    )

    plot_df = plot_df.sort_values("conditioning_effect_pct", ascending=True)

    fig = px.bar(
        plot_df,
        y="card_short",
        x="conditioning_effect_pct",
        orientation="h",
        text="conditioning_effect_label",
        title="Mulligan conditioning effect",
        labels={
            "card_short": "",
            "conditioning_effect_pct": "Change vs random 7 (%)",
        },
        color="conditioning_effect_pct",
        color_continuous_scale=["#F87171", "#94A3B8", "#34D399"],
        custom_data=["card", "conditioning_effect_label"],
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=9, color="#F8FAFC"),
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="<b>%{customdata[0]}</b><br>Change: %{customdata[1]}<extra></extra>",
    )

    apply_dark_layout(
        fig,
        height=chart_height(len(plot_df), base=430, per_row=28, max_height=760),
        compact=True,
    )

    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_title="",
        xaxis_title="Change vs random 7 (%)",
        margin=dict(l=4, r=98, t=56, b=48),
    )

    fig.update_xaxes(
        range=padded_range(plot_df["conditioning_effect_pct"], padding_factor=1.42)
    )

    return fig


def make_prize_chart(prize_df: pd.DataFrame, top_n_cards: int = 25):
    plot_df = prize_df.head(top_n_cards).copy()
    plot_df = add_short_labels(plot_df, max_len=24)
    plot_df["prize_probability_pct"] = plot_df["P_at_least_1_prized"] * 100
    plot_df["prize_probability_label"] = plot_df["prize_probability_pct"].map(
        lambda x: percent_text(x, decimals=2)
    )

    plot_df = plot_df.sort_values("prize_probability_pct", ascending=True)

    fig = px.bar(
        plot_df,
        y="card_short",
        x="prize_probability_pct",
        orientation="h",
        text="prize_probability_label",
        title="At least 1 prized copy",
        labels={
            "card_short": "",
            "prize_probability_pct": "Probability (%)",
        },
        color_discrete_sequence=["#FBBF24"],
        custom_data=["card", "prize_probability_label"],
    )

    fig.update_traces(
        textposition=smart_text_positions(plot_df["prize_probability_pct"], inside_threshold=18),
        textfont=dict(size=9, color="#F8FAFC"),
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="<b>%{customdata[0]}</b><br>P(at least 1 prized): %{customdata[1]}<extra></extra>",
    )

    apply_dark_layout(
        fig,
        height=chart_height(len(plot_df), base=430, per_row=28, max_height=760),
        compact=True,
    )

    fig.update_layout(
        yaxis_title="",
        xaxis_title="Probability (%)",
        xaxis_range=padded_range(
            plot_df["prize_probability_pct"],
            upper_floor=10,
            padding_factor=1.28,
        ),
        margin=dict(l=4, r=96, t=56, b=48),
    )

    return fig


def make_all_copies_prized_chart(prize_df: pd.DataFrame, top_n_cards: int = 25):
    plot_df = prize_df.copy()
    plot_df = plot_df.sort_values("P_all_copies_prized", ascending=False).head(top_n_cards)
    plot_df = add_short_labels(plot_df, max_len=24)
    plot_df["all_copies_prized_pct"] = plot_df["P_all_copies_prized"] * 100
    plot_df["all_copies_prized_label"] = plot_df["all_copies_prized_pct"].map(
        lambda x: percent_text(x, decimals=4)
    )

    plot_df = plot_df.sort_values("all_copies_prized_pct", ascending=True)

    fig = px.bar(
        plot_df,
        y="card_short",
        x="all_copies_prized_pct",
        orientation="h",
        text="all_copies_prized_label",
        title="All copies prized",
        labels={
            "card_short": "",
            "all_copies_prized_pct": "Probability (%)",
        },
        color_discrete_sequence=["#FB7185"],
        custom_data=["card", "all_copies_prized_label"],
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=9, color="#F8FAFC"),
        marker_line_width=0,
        cliponaxis=False,
        hovertemplate="<b>%{customdata[0]}</b><br>P(all copies prized): %{customdata[1]}<extra></extra>",
    )

    apply_dark_layout(
        fig,
        height=chart_height(len(plot_df), base=430, per_row=28, max_height=760),
        compact=True,
    )

    fig.update_layout(
        yaxis_title="",
        xaxis_title="Probability (%)",
        xaxis_range=padded_range(
            plot_df["all_copies_prized_pct"],
            upper_floor=0.05,
            padding_factor=1.65,
        ),
        margin=dict(l=4, r=118, t=56, b=48),
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
        return apply_dark_layout(fig, height=420, compact=True)

    plot_df = prize_df.copy()
    plot_df = plot_df.sort_values("P_at_least_1_prized", ascending=True)
    plot_df = add_short_labels(plot_df, max_len=24)

    z_values = plot_df[available_cols].values * 100
    y_labels = plot_df["card_short"].tolist()
    full_labels = plot_df["card"].tolist()
    x_labels = ["After 1", "After 2", "After 3", "After 4", "After 5"]

    text_values = [
        [percent_text(value, decimals=1) for value in row]
        for row in z_values
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=x_labels[: len(available_cols)],
            y=y_labels,
            text=text_values,
            texttemplate="%{text}",
            textfont=dict(size=9, color="#F8FAFC"),
            customdata=[[label] * len(available_cols) for label in full_labels],
            colorscale=[
                [0.0, "#0F172A"],
                [0.35, "#1D4ED8"],
                [0.7, "#7C3AED"],
                [1.0, "#F59E0B"],
            ],
            colorbar=dict(
                title=dict(
                    text="Prob. (%)",
                    font=dict(color="#CBD5E1", size=10),
                ),
                tickfont=dict(color="#CBD5E1", size=9),
                thickness=10,
                len=0.72,
            ),
            hovertemplate="<b>%{customdata}</b><br>%{x} prizes taken<br>Still prized: %{z:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Still prized after prizes taken",
        xaxis_title="Prizes taken",
        yaxis_title="",
    )

    apply_dark_layout(
        fig,
        height=chart_height(len(plot_df), base=430, per_row=24, max_height=700),
        compact=True,
    )

    fig.update_layout(
        margin=dict(l=4, r=4, t=56, b=56),
    )

    fig.update_xaxes(tickangle=-35)

    return fig