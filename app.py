"""Streamlit front end for the Pokémon TCG Consistency Lab.

This single-page app replaces the small default Streamlit multipage navigation
with a polished in-app workspace:
- Deck Dashboard: current mulligan, card access, prize, and gallery tools.
- Starting Hand Lab: exact custom Boolean statements on opening hands/draws.

The Starting Hand Lab is exact combinatorics only. It does not simulate Turn 1
actions, search effects, Supporters, Items, or Abilities.
"""

from __future__ import annotations

import html
import random
from typing import Iterable

import pandas as pd
import streamlit as st

from src.analysis import analyze_deck_opening_hand, format_probability_table
from src.charts import (
    make_all_copies_prized_chart,
    make_card_odds_chart,
    make_conditioning_effect_chart,
    make_deck_composition_chart,
    make_mulligan_chart,
    make_prize_chart,
    make_prize_survival_heatmap,
)
from src.probability import (
    custom_starting_hand_statement_probabilities,
    p_dnf_statement_given_legal_opening,
)

EXAMPLE_DECKLIST = """Pokémon: 8
1 Meowth ex POR 62
2 Solrock MEG 75
2 Makuhita MEG 72
1 Mega Zygarde ex POR 47
3 Mega Lucario ex MEG 77
2 Lunatone MEG 74
3 Riolu MEG 76
2 Hariyama MEG 73

Trainer: 14
3 Poké Pad ASC 198
4 Lillie's Determination MEG 119
2 Air Balloon BLK 79
2 Gravity Mountain SSP 177
1 Boss's Orders MEG 114
4 Fighting Gong MEG 116
3 Night Stretcher SFA 61
1 Switch MEG 130
4 Ultra Ball MEG 131
2 Wally's Compassion MEG 132
2 Judge POR 76
1 Unfair Stamp TWM 165
1 Core Memory POR 70
4 Premium Power Pro MEG 124

Energy: 1
10 Basic {F} Energy MEE 6

Total Cards: 60
"""
DISPLAY_COLUMN_NAMES = {
    "card": "Card",
    "name": "Name",
    "count": "Count",
    "section": "Deck section",
    "api_id": "API ID",
    "supertype": "Card type",
    "subtypes": "Subtypes",
    "is_basic_pokemon": "Basic Pokémon?",
    "image_url": "Image",
    "image_large_url": "Large image",
    "P_in_random_7_unconditioned": "P(in random 7)",
    "P_in_legal_opening_7": "P(in legal opening 7)",
    "P_in_hand_after_turn_draw": "Opening hand + draw",
    "increase_from_turn_draw": "Draw-for-turn gain",
    "P_at_least_1_prized": "P(at least 1 prized)",
    "E_prized": "Expected prized",
    "P_all_copies_prized": "P(all copies prized)",
    "P_still_prized_after_1_prize_taken": "P(≥1 still prized after 1 prize taken)",
    "P_still_prized_after_2_prizes_taken": "P(≥1 still prized after 2 prizes taken)",
    "P_still_prized_after_3_prizes_taken": "P(≥1 still prized after 3 prizes taken)",
    "P_still_prized_after_4_prizes_taken": "P(≥1 still prized after 4 prizes taken)",
    "P_still_prized_after_5_prizes_taken": "P(≥1 still prized after 5 prizes taken)",
    "P_all_copies_still_prized_after_1_prize_taken": "P(all still prized after 1 prize taken)",
    "P_all_copies_still_prized_after_2_prizes_taken": "P(all still prized after 2 prizes taken)",
    "P_all_copies_still_prized_after_3_prizes_taken": "P(all still prized after 3 prizes taken)",
    "P_all_copies_still_prized_after_4_prizes_taken": "P(all still prized after 4 prizes taken)",
    "P_all_copies_still_prized_after_5_prizes_taken": "P(all still prized after 5 prizes taken)",
}

CACHE_VERSION = "opening-hand-planner-v3-lucario-example"
PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}

st.set_page_config(
    page_title="Pokémon TCG Consistency Lab",
    page_icon="🎴",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg-card: rgba(18, 24, 38, 0.86);
            --bg-card-soft: rgba(30, 41, 59, 0.72);
            --border-soft: rgba(148, 163, 184, 0.22);
            --text-muted: rgba(226, 232, 240, 0.72);
            --accent: #7dd3fc;
            --accent-strong: #38bdf8;
            --good: #86efac;
        }
        html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            overflow-y: auto !important;
            height: auto !important;
        }
        .block-container {
            padding-top: 1.5rem;
            max-width: 1320px;
            padding-bottom: 5rem;
        }
        section[data-testid="stSidebar"] { width: 19rem !important; }
        [data-testid="stSidebarNav"] { display: none; }
        .hero {
            border: 1px solid var(--border-soft);
            border-radius: 28px;
            padding: 28px 30px;
            margin-bottom: 18px;
            background:
              radial-gradient(circle at top left, rgba(56, 189, 248, 0.24), transparent 35%),
              radial-gradient(circle at bottom right, rgba(168, 85, 247, 0.18), transparent 32%),
              linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(2, 6, 23, 0.95));
            box-shadow: 0 22px 55px rgba(0, 0, 0, 0.25);
        }
        .eyebrow {
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2.1rem, 4vw, 4.1rem);
            line-height: 1.0;
            letter-spacing: -0.055em;
        }
        .hero p {
            color: var(--text-muted);
            max-width: 840px;
            font-size: 1.05rem;
            margin-top: 14px;
            margin-bottom: 0;
        }
        .pill-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 18px; }
        .pill {
            border: 1px solid rgba(125, 211, 252, 0.24);
            background: rgba(14, 165, 233, 0.10);
            color: #e0f2fe;
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 0.86rem;
            font-weight: 700;
        }
        .metric-card {
            border: 1px solid var(--border-soft);
            background: var(--bg-card);
            border-radius: 22px;
            padding: 18px 20px;
            min-height: 112px;
            box-shadow: 0 14px 28px rgba(0,0,0,0.12);
        }
        .metric-label {
            color: var(--text-muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
        }
        .metric-value {
            font-size: 2.0rem;
            font-weight: 900;
            margin-top: 6px;
            letter-spacing: -0.03em;
        }
        .metric-note {
            color: var(--text-muted);
            font-size: 0.88rem;
            margin-top: 4px;
        }
        .section-card {
            border: 1px solid var(--border-soft);
            background: rgba(15, 23, 42, 0.62);
            border-radius: 24px;
            padding: 18px 20px;
            margin: 12px 0 18px 0;
        }
        .statement-box {
            border: 1px solid rgba(125, 211, 252, 0.28);
            border-radius: 22px;
            padding: 16px 18px;
            background: linear-gradient(135deg, rgba(14,165,233,0.13), rgba(99,102,241,0.11));
            font-size: 1.02rem;
            font-weight: 800;
        }

        .native-builder-shell {
            border: 1px solid rgba(125, 211, 252, 0.25);
            border-radius: 24px;
            overflow: hidden;
            margin: 12px 0 18px 0;
            background: linear-gradient(135deg, rgba(14,165,233,0.12), rgba(15,23,42,0.78));
        }
        .native-statement-panel { padding: 18px 20px; }
        .native-statement { display:flex; flex-wrap:wrap; align-items:center; gap:8px; min-height:44px; font-weight:900; }
        .statement-token {
            display:inline-flex; align-items:center; gap:8px; padding:7px 10px;
            border-radius:999px; border:1px solid rgba(148,163,184,0.28);
            background:rgba(15,23,42,0.72); color:#e2e8f0; font-size:0.86rem;
        }
        .statement-chip-img { width:25px; height:35px; object-fit:cover; border-radius:4px; }
        .statement-chip-placeholder { display:inline-grid; place-items:center; width:25px; height:35px; border-radius:4px; background:rgba(51,65,85,.8); }
        .native-empty-bracket {
            border:1px dashed rgba(148,163,184,.35); border-radius:18px; padding:22px;
            color:rgba(226,232,240,.72); text-align:center; font-weight:800;
            background:rgba(15,23,42,.38); margin-bottom:12px;
        }
        .native-card-label { font-size:.78rem; line-height:1.15; margin:.35rem 0 .5rem 0; min-height:44px; }
        .native-card-label span { color:rgba(226,232,240,.65); font-size:.72rem; }
        .success-way-intro {
            border: 1px solid rgba(125, 211, 252, 0.28);
            border-radius: 18px;
            padding: 12px 14px;
            background: rgba(14, 165, 233, 0.09);
            color: rgba(226, 232, 240, 0.84);
            font-size: 0.93rem;
            font-weight: 700;
            margin-bottom: 12px;
        }
        .success-way-divider {
            display:flex; align-items:center; gap:10px; margin: 18px 0 12px 0;
            color:#7dd3fc; font-weight:950; text-transform:uppercase; letter-spacing:.08em;
        }
        .success-way-divider:before, .success-way-divider:after {
            content:""; flex:1; height:1px; background:rgba(125,211,252,.22);
        }

        .how-to-card {
            border: 1px solid rgba(125, 211, 252, 0.24);
            border-radius: 22px;
            padding: 16px 18px;
            margin: 12px 0 18px 0;
            background: linear-gradient(135deg, rgba(14,165,233,0.12), rgba(30,41,59,0.45));
        }
        .how-to-title {
            color: #bae6fd;
            text-transform: uppercase;
            letter-spacing: .10em;
            font-weight: 950;
            font-size: .78rem;
            margin-bottom: 10px;
        }
        .how-to-steps { display:flex; flex-wrap:wrap; gap:10px; }
        .how-to-steps span {
            border:1px solid rgba(148,163,184,.24);
            background:rgba(15,23,42,.68);
            border-radius:999px;
            padding:8px 12px;
            font-weight:850;
            color:#e2e8f0;
        }
        .how-to-steps b {
            display:inline-grid; place-items:center;
            width:20px; height:20px;
            border-radius:50%;
            margin-right:6px;
            color:#082f49;
            background:#7dd3fc;
        }
        .how-to-example { color: var(--text-muted); margin-top: 10px; font-weight: 700; }
        .compact-placeholder { min-height:130px; }

        .small-muted { color: var(--text-muted); font-size: 0.92rem; }
        .card-grid {
            display: grid;
            grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
            gap: 16px;
            margin-top: 14px;
        }
        .card-tile {
            border: 1px solid var(--border-soft);
            background: var(--bg-card-soft);
            border-radius: 18px;
            padding: 12px;
            overflow: hidden;
        }
        .card-img {
            width: 100%;
            border-radius: 13px;
            display: block;
            margin-bottom: 10px;
        }
        .placeholder-card {
            aspect-ratio: 0.716;
            border: 1px dashed rgba(148, 163, 184, 0.35);
            border-radius: 13px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            font-weight: 800;
            text-align: center;
            margin-bottom: 10px;
        }
        .card-title { font-weight: 850; line-height: 1.2; }
        .card-meta { color: var(--text-muted); font-size: 0.86rem; margin-top: 4px; }
        div[data-testid="stTabs"] button p { font-size: 1.02rem; font-weight: 850; }
        div[data-testid="stTabs"] button { padding: 0.7rem 1.0rem; }
        div[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
        @media (max-width: 760px) {
            .hero { padding: 22px 18px; border-radius: 22px; }
            .card-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
        }

        .formula-bar {
            border: 1px solid rgba(125, 211, 252, 0.34);
            border-radius: 26px;
            padding: 18px 20px;
            background:
              radial-gradient(circle at left, rgba(14,165,233,0.20), transparent 42%),
              linear-gradient(135deg, rgba(15, 23, 42, 0.88), rgba(30, 41, 59, 0.64));
            font-size: 1.0rem;
            font-weight: 850;
            line-height: 2.2;
            margin: 12px 0 18px 0;
        }
        .formula-title {
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.75rem;
            font-weight: 900;
            margin-bottom: 8px;
        }
        .logic-token {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid rgba(148, 163, 184, 0.26);
            background: rgba(15, 23, 42, 0.88);
            border-radius: 999px;
            padding: 6px 10px;
            margin: 2px 4px;
            color: #e2e8f0;
            font-weight: 850;
            white-space: nowrap;
        }
        .operator-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            padding: 6px 10px;
            margin: 2px 5px;
            font-size: 0.78rem;
            font-weight: 950;
            letter-spacing: 0.08em;
            color: #082f49;
            background: linear-gradient(135deg, #7dd3fc, #bae6fd);
            box-shadow: 0 8px 18px rgba(14,165,233,0.18);
        }
        .bracket-chip {
            display: inline-flex;
            align-items: center;
            color: #bfdbfe;
            font-size: 1.35rem;
            font-weight: 950;
            margin: 0 1px;
        }
        .builder-card {
            border: 1px solid var(--border-soft);
            border-radius: 24px;
            padding: 16px;
            background: rgba(15, 23, 42, 0.56);
            margin-bottom: 14px;
        }
        .route-card {
            border: 1px solid rgba(125, 211, 252, 0.24);
            border-radius: 22px;
            padding: 14px 16px;
            background: linear-gradient(135deg, rgba(14, 165, 233, 0.10), rgba(99,102,241,0.08));
            margin-bottom: 14px;
        }
        .route-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #e0f2fe;
            font-size: 0.95rem;
            font-weight: 950;
            letter-spacing: -0.01em;
            margin-bottom: 8px;
        }
        .route-subtitle {
            color: var(--text-muted);
            font-size: 0.86rem;
            font-weight: 650;
        }
        .mini-card-strip {
            display: grid;
            grid-template-columns: repeat(var(--mini-cols), minmax(58px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .mini-card {
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 14px;
            background: rgba(2, 6, 23, 0.34);
            padding: 8px;
            min-height: 128px;
        }
        .mini-card img {
            width: 100%;
            border-radius: 10px;
            display: block;
            margin-bottom: 6px;
        }
        .mini-card-name {
            font-size: 0.74rem;
            color: #e2e8f0;
            line-height: 1.15;
            font-weight: 800;
        }
        .result-glow {
            border: 1px solid rgba(134, 239, 172, 0.30);
            border-radius: 28px;
            padding: 22px 24px;
            background:
              radial-gradient(circle at right top, rgba(134, 239, 172, 0.18), transparent 38%),
              linear-gradient(135deg, rgba(20, 83, 45, 0.25), rgba(15, 23, 42, 0.82));
            box-shadow: 0 18px 40px rgba(0,0,0,0.18);
            margin: 12px 0 18px 0;
        }
        .result-kicker {
            color: #bbf7d0;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-weight: 950;
        }
        .result-number {
            font-size: clamp(2.4rem, 6vw, 4.8rem);
            line-height: 1;
            letter-spacing: -0.06em;
            font-weight: 950;
            margin: 8px 0;
        }
        .result-copy { color: var(--text-muted); font-weight: 700; }
        .sortable-component {
            border: 1px solid rgba(148, 163, 184, 0.22) !important;
            border-radius: 18px !important;
            background: rgba(15, 23, 42, 0.50) !important;
            padding: 10px !important;
        }
        .sortable-container {
            border: 1px solid rgba(125, 211, 252, 0.22) !important;
            border-radius: 16px !important;
            background: rgba(2, 6, 23, 0.30) !important;
            min-height: 86px !important;
        }
        .sortable-item {
            border: 1px solid rgba(125, 211, 252, 0.30) !important;
            border-radius: 999px !important;
            background: rgba(14, 165, 233, 0.14) !important;
            color: #e0f2fe !important;
            font-weight: 850 !important;
            padding: 8px 12px !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


def pct(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:.{decimals}%}"


def pct_from_probability(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return f"{100 * value:.{decimals}f}%"


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{html.escape(label)}</div>
          <div class="metric-value">{html.escape(value)}</div>
          <div class="metric-note">{html.escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pretty_table(df: pd.DataFrame) -> pd.DataFrame:
    return format_probability_table(df).rename(columns=DISPLAY_COLUMN_NAMES)


def prize_column_names(prizes_taken: int):
    if prizes_taken == 0:
        return "P_at_least_1_prized", "P_all_copies_prized", "Initial prizes"
    suffix = "prize_taken" if prizes_taken == 1 else "prizes_taken"
    return (
        f"P_still_prized_after_{prizes_taken}_{suffix}",
        f"P_all_copies_still_prized_after_{prizes_taken}_{suffix}",
        f"After {prizes_taken} prize{'s' if prizes_taken != 1 else ''} taken",
    )


def show_chart(fig) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def render_card_tile_html(row: pd.Series, prizes_taken: int) -> str:
    at_least_col, all_col, context_label = prize_column_names(prizes_taken)
    image_url = row.get("image_large_url") or row.get("image_url")
    card_label = str(row.get("card", "Unknown card"))
    count = int(row.get("count", 0))
    turn_access = row.get("P_in_hand_after_turn_draw", float("nan"))
    at_least_prized = row.get(at_least_col, float("nan"))
    all_prized = row.get(all_col, float("nan"))
    safe_card_label = html.escape(card_label)
    safe_context_label = html.escape(context_label)

    if image_url:
        safe_image_url = html.escape(str(image_url), quote=True)
        image_html = f'<img class="card-img" src="{safe_image_url}" alt="{safe_card_label}">' 
    else:
        image_html = f'<div class="placeholder-card">{safe_card_label}<br>No image found</div>'

    return f"""
    <div class="card-tile">
      {image_html}
      <div class="card-meta">x{count}</div>
      <div class="card-title">{safe_card_label}</div>
      <div class="card-meta">Opening hand + draw: <b>{pct(turn_access, 2)}</b></div>
      <div class="card-meta">≥1 prized ({safe_context_label}): <b>{pct(at_least_prized, 2)}</b></div>
      <div class="card-meta">All prized ({safe_context_label}): <b>{pct(all_prized, 4)}</b></div>
    </div>
    """


def render_card_gallery(
    card_df: pd.DataFrame,
    prizes_taken: int,
    card_type_filter: str,
    sort_mode: str,
    columns_per_row: int,
) -> None:
    gallery_df = card_df.copy()
    if card_type_filter != "All":
        gallery_df = gallery_df[gallery_df["supertype"].fillna("Unknown") == card_type_filter].copy()

    at_least_col, all_col, _ = prize_column_names(prizes_taken)
    sort_map = {
        "Opening hand + draw": "P_in_hand_after_turn_draw",
        "At least 1 prized": at_least_col,
        "All copies prized": all_col,
        "Deck order / parsed order": None,
        "Card count": "count",
    }
    sort_col = sort_map.get(sort_mode)
    if sort_col and sort_col in gallery_df.columns:
        gallery_df = gallery_df.sort_values(sort_col, ascending=False)

    if gallery_df.empty:
        st.info("No cards match the selected gallery filters.")
        return

    safe_columns = max(1, min(int(columns_per_row), 6))
    cards_html = "\n".join(
        render_card_tile_html(pd.Series(row_dict), prizes_taken)
        for row_dict in gallery_df.to_dict("records")
    )
    st.markdown(
        f"""
        <div class="card-grid" style="--cols:{safe_columns}">
        {cards_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_analysis(decklist_text: str, max_mulligans: int, cache_version: str):
    return analyze_deck_opening_hand(decklist_text=decklist_text, max_mulligans=max_mulligans)


def init_session_state() -> None:
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "has_analyzed" not in st.session_state:
        st.session_state.has_analyzed = False


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Probability-driven deckbuilding</div>
          <h1>Pokémon TCG Consistency Lab</h1>
          <p>
            Exact opening-hand, mulligan, prize-card, and custom starting-hand probability tools
            for competitive Pokémon TCG deck tuning.
          </p>
          <div class="pill-row">
            <span class="pill">Exact legal-hand math</span>
            <span class="pill">Custom AND / OR statements</span>
            <span class="pill">Prize risk diagnostics</span>
            <span class="pill">Visual card gallery</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def deck_input_panel() -> None:
    input_expanded = not st.session_state.has_analyzed
    with st.expander("Deck setup", expanded=input_expanded):
        st.markdown(
            "Paste a Pokémon TCG Live, Limitless, or similar 60-card decklist. "
            "The same parsed deck powers every tool below."
        )
        decklist_text = st.text_area(
            "Decklist",
            value=EXAMPLE_DECKLIST,
            height=320,
            help="Paste a Pokémon TCG Live, Limitless, or similar decklist export.",
            key="main_decklist_text",
        )
        col_a, col_b, col_c = st.columns([1, 1, 1])
        with col_a:
            max_mulligans = st.slider(
                "Show mulligans up to",
                min_value=3,
                max_value=10,
                value=6,
                help="The final bucket is shown as X+ mulligans.",
                key="main_max_mulligans",
            )
        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)
            analyze = st.button("Analyze deck", type="primary", use_container_width=True)
        with col_c:
            st.markdown("<br>", unsafe_allow_html=True)
            clear = st.button("Clear", use_container_width=True)

        if clear:
            st.session_state.analysis_results = None
            st.session_state.has_analyzed = False
            st.rerun()

        if analyze:
            try:
                with st.spinner("Analyzing deck and fetching card images..."):
                    st.session_state.analysis_results = cached_analysis(
                        decklist_text, max_mulligans, CACHE_VERSION
                    )
                st.session_state.has_analyzed = True
                st.rerun()
            except Exception as exc:  # noqa: BLE001 - Streamlit should show parser/API errors.
                st.session_state.analysis_results = None
                st.session_state.has_analyzed = False
                st.error(str(exc))


def render_empty_state() -> None:
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Step 1", "Paste", "Paste a 60-card Pokémon TCG decklist.")
    with col2:
        metric_card("Step 2", "Analyze", "Run exact opening-hand and prize-card math.")
    with col3:
        metric_card("Step 3", "Build", "Create custom AND / OR starting-hand statements.")

    st.markdown(
        """
        <div class="section-card">
        <h3>What this site calculates</h3>
        <ul>
          <li>Probability of mulliganing 0, 1, 2, 3, ... times.</li>
          <li>Probability of seeing each card in a legal opening hand.</li>
          <li>Probability of seeing each card after drawing for turn.</li>
          <li>Custom statements like <b>(X AND Y) OR Z</b> with up to 8 selected cards.</li>
          <li>True game-start probability of at least one copy, or all copies, being prized.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_deck_dashboard(summary, mulligan_df, card_odds_df, parsed_df) -> None:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Deck size", str(summary["deck_size"]), "Total parsed cards")
    with c2:
        metric_card("Basic Pokémon", str(summary["basic_count"]), "Cards that make an opener legal")
    with c3:
        metric_card("Mulligan chance", f"{summary['p_mulligan_one_hand']:.2%}", "No Basic in one opening attempt")
    with c4:
        metric_card("Expected mulligans", f"{summary['expected_mulligans_before_legal_hand']:.3f}", "Before a legal opener")

    st.subheader("Deck overview")
    left, right = st.columns([0.9, 1.1])
    with left:
        show_chart(make_deck_composition_chart(parsed_df))
    with right:
        show_chart(make_mulligan_chart(mulligan_df))

    st.subheader("Most accessible cards")
    show_chart(make_card_odds_chart(card_odds_df, top_n_cards=len(card_odds_df)))

    st.subheader("Opening-hand probability table")
    hand_cols = [
        "card",
        "count",
        "supertype",
        "is_basic_pokemon",
        "P_in_random_7_unconditioned",
        "P_in_legal_opening_7",
        "P_in_hand_after_turn_draw",
        "increase_from_turn_draw",
    ]
    st.dataframe(pretty_table(card_odds_df[hand_cols]), use_container_width=True, hide_index=True)


def deck_card_options(deck) -> list[dict]:
    options = []
    for i, card in enumerate(deck):
        options.append(
            {
                "key": f"{i}:{card.key}",
                "label": card.label,
                "name": card.name,
                "count": card.count,
                "supertype": card.supertype or "Unknown",
                "is_basic": bool(card.is_basic_pokemon),
                "image_url": card.image_large_url or card.image_url,
            }
        )
    return options


def route_to_text(route_keys: Iterable[str], label_by_key: dict[str, str]) -> str:
    labels = [label_by_key[k] for k in route_keys if k in label_by_key]
    if not labels:
        return "—"
    if len(labels) == 1:
        return labels[0]
    return "(" + " AND ".join(labels) + ")"


def statement_to_text(routes: list[list[str]], label_by_key: dict[str, str]) -> str:
    pieces = [route_to_text(route, label_by_key) for route in routes if route]
    return " OR ".join(pieces) if pieces else "No statement built yet"


def render_selected_card_preview(selected: list[dict], card_odds_df: pd.DataFrame) -> None:
    if not selected:
        return

    odds_by_label = card_odds_df.set_index("card").to_dict("index")
    cards_html = []
    for card in selected:
        odds = odds_by_label.get(card["label"], {})
        opening = odds.get("P_in_legal_opening_7", float("nan"))
        after_draw = odds.get("P_in_hand_after_turn_draw", float("nan"))
        safe_label = html.escape(card["label"])
        image_url = card.get("image_url")
        if image_url:
            image_html = f'<img class="card-img" src="{html.escape(str(image_url), quote=True)}" alt="{safe_label}">'
        else:
            image_html = f'<div class="placeholder-card">{safe_label}<br>No image found</div>'
        cards_html.append(
            f"""
            <div class="card-tile">
              {image_html}
              <div class="card-meta">x{card['count']} • {html.escape(card['supertype'])}</div>
              <div class="card-title">{safe_label}</div>
              <div class="card-meta">Legal opening: <b>{pct(opening, 2)}</b></div>
              <div class="card-meta">Opening hand + draw: <b>{pct(after_draw, 2)}</b></div>
            </div>
            """
        )

    columns = min(max(len(selected), 1), 4)
    st.markdown(
        f"""
        <div class="card-grid" style="--cols:{columns}">
        {''.join(cards_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )




def html_logic_formula(routes: list[list[str]], label_by_key: dict[str, str]) -> str:
    if not routes:
        return '<span class="small-muted">No statement built yet</span>'
    route_html = []
    for route in routes:
        if not route:
            continue
        pieces = []
        for j, key in enumerate(route):
            if j > 0:
                pieces.append('<span class="operator-chip">AND</span>')
            pieces.append(f'<span class="logic-token">{html.escape(label_by_key.get(key, key))}</span>')
        route_html.append(
            '<span class="bracket-chip">(</span>'
            + ''.join(pieces)
            + '<span class="bracket-chip">)</span>'
        )
    if not route_html:
        return '<span class="small-muted">No statement built yet</span>'
    return '<span class="operator-chip">OR</span>'.join(route_html)


def render_formula_bar(title: str, routes: list[list[str]], label_by_key: dict[str, str]) -> None:
    st.markdown(
        f"""
        <div class="formula-bar">
          <div class="formula-title">{html.escape(title)}</div>
          {html_logic_formula(routes, label_by_key)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mini_card_strip(route: list[str], option_by_key: dict[str, dict]) -> None:
    if not route:
        st.markdown('<div class="small-muted">Drop or select cards for this bracket.</div>', unsafe_allow_html=True)
        return
    cards_html = []
    for key in route:
        card = option_by_key.get(key)
        if not card:
            continue
        safe_label = html.escape(card["label"])
        image_url = card.get("image_url")
        if image_url:
            img = f'<img src="{html.escape(str(image_url), quote=True)}" alt="{safe_label}">'
        else:
            img = f'<div class="placeholder-card" style="min-height:86px; aspect-ratio:0.716;">No image</div>'
        cards_html.append(
            f"""
            <div class="mini-card">
              {img}
              <div class="mini-card-name">{safe_label}</div>
            </div>
            """
        )
    cols = min(max(len(cards_html), 1), 4)
    st.markdown(
        f"""
        <div class="mini-card-strip" style="--mini-cols:{cols}">
          {''.join(cards_html)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_checkbox_builder(
    selected_keys: list[str],
    option_by_key: dict[str, dict],
    label_by_key: dict[str, str],
    route_count: int,
) -> list[list[str]]:
    st.caption(
        "Build the expression as bracketed routes. Each bracket is an AND group; the app joins brackets with OR."
    )
    routes: list[list[str]] = []
    route_cols = st.columns(2)
    for i in range(route_count):
        with route_cols[i % 2]:
            st.markdown(
                f"""
                <div class="route-card">
                  <div class="route-title">Bracket {i + 1}<span class="route-subtitle">AND group</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            default_route: list[str] = []
            if i == 0:
                default_route = selected_keys[: min(2, len(selected_keys))]
            elif i == 1 and len(selected_keys) >= 3:
                default_route = [selected_keys[2]]
            route = st.multiselect(
                f"Cards inside bracket {i + 1}",
                options=selected_keys,
                default=default_route,
                format_func=lambda key: label_by_key.get(key, key),
                key=f"route_{i + 1}",
                help="All cards selected here must be present. This bracket is wrapped in parentheses.",
            )
            render_mini_card_strip(route, option_by_key)
            if route:
                routes.append(route)

    render_formula_bar("Success statement", routes, label_by_key)
    return routes

def default_starting_hand_example_routes(options: list[dict]) -> list[list[str]]:
    """Create a friendly example Success Statement for the current deck.

    Prefer the Mega Lucario sample: (Lunatone AND Solrock) OR
    (Lunatone AND Fighting Gong). This demonstrates alternatives without
    needing probability knowledge.
    """
    if not options:
        return [[]]

    signature = "|".join(f"{opt.get('key')}:{opt.get('count')}" for opt in options)
    state_key = "starting_hand_builder_example_routes"
    sig_key = "starting_hand_builder_example_signature"

    if st.session_state.get(sig_key) == signature and state_key in st.session_state:
        return st.session_state[state_key]

    def find_key(name_part: str) -> str | None:
        needle = name_part.lower()
        for opt in options:
            label = f"{opt.get('label', '')} {opt.get('name', '')}".lower()
            if needle in label:
                return opt.get("key")
        return None

    lunatone = find_key("Lunatone")
    solrock = find_key("Solrock")
    fighting_gong = find_key("Fighting Gong")

    if lunatone and solrock and fighting_gong:
        routes = [[lunatone, solrock], [lunatone, fighting_gong]]
    else:
        candidates = [opt for opt in options if opt.get("supertype") != "Energy"] or options
        rng = random.Random(signature)
        sample_size = min(2, len(candidates))
        picked = rng.sample(candidates, sample_size) if sample_size else []
        routes = [[picked[0]["key"], picked[1]["key"]]] if len(picked) >= 2 else [[picked[0]["key"]]] if picked else [[]]

    st.session_state[sig_key] = signature
    st.session_state[state_key] = routes
    return routes

def render_statement_html(routes: list[list[str]], option_by_key: dict[str, dict]) -> str:
    """Render a visual DNF statement as image chips."""
    if not routes:
        return '<span class="empty-statement">Click cards from your deck to build a Success Statement.</span>'

    pieces: list[str] = []
    non_empty = [route for route in routes if route]
    for route_i, route in enumerate(non_empty):
        if route_i > 0:
            pieces.append('<span class="operator-chip">OR</span>')
        pieces.append('<span class="bracket-chip">(</span>')
        for card_i, key in enumerate(route):
            card = option_by_key.get(key)
            if not card:
                continue
            if card_i > 0:
                pieces.append('<span class="operator-chip">AND</span>')
            label = html.escape(str(card.get("label", key)))
            image_url = card.get("image_url")
            if image_url:
                img = f'<img class="statement-chip-img" src="{html.escape(str(image_url), quote=True)}" alt="{label}">'
            else:
                img = '<span class="statement-chip-placeholder">?</span>'
            pieces.append(f'<span class="statement-token">{img}<span>{label}</span></span>')
        pieces.append('<span class="bracket-chip">)</span>')
    return "".join(pieces)


def normalize_builder_routes(routes: list[list[str]], option_by_key: dict[str, dict]) -> list[list[str]]:
    clean: list[list[str]] = []
    seen_any = False
    for route in routes or []:
        if not isinstance(route, list):
            continue
        clean_route: list[str] = []
        per_card_counts: dict[str, int] = {}
        for key in route:
            key = str(key)
            if key not in option_by_key:
                continue
            max_copies = int(option_by_key[key].get("count", 0))
            current = per_card_counts.get(key, 0)
            if current >= max_copies:
                continue
            clean_route.append(key)
            per_card_counts[key] = current + 1
            seen_any = True
        clean.append(clean_route)
    if not clean:
        clean = [[]]
    if not seen_any and len(clean) == 0:
        clean = [[]]
    return clean


def current_builder_routes(options: list[dict], option_by_key: dict[str, dict]) -> list[list[str]]:
    signature = "|".join(f"{opt.get('key')}:{opt.get('count')}" for opt in options)
    sig_key = "starting_hand_click_builder_signature"
    routes_key = "starting_hand_click_builder_routes"
    active_key = "starting_hand_click_builder_active_route"

    if st.session_state.get(sig_key) != signature or routes_key not in st.session_state:
        st.session_state[sig_key] = signature
        st.session_state[routes_key] = default_starting_hand_example_routes(options)
        st.session_state[active_key] = 0

    routes = normalize_builder_routes(st.session_state.get(routes_key, [[]]), option_by_key)
    st.session_state[routes_key] = routes
    st.session_state[active_key] = max(0, min(int(st.session_state.get(active_key, 0)), len(routes) - 1))
    return routes


def set_builder_routes(routes: list[list[str]], option_by_key: dict[str, dict]) -> None:
    st.session_state["starting_hand_click_builder_routes"] = normalize_builder_routes(routes, option_by_key)


def add_card_to_active_route(key: str, option_by_key: dict[str, dict]) -> None:
    """Add one copy of a card to the active AND bracket.

    Repeated cards are intentionally allowed. For example, adding Ultra Ball
    twice means the bracket requires at least 2 Ultra Ball in the opening
    hand / opening hand plus draw.
    """
    routes = normalize_builder_routes(st.session_state.get("starting_hand_click_builder_routes", [[]]), option_by_key)
    active = max(0, min(int(st.session_state.get("starting_hand_click_builder_active_route", 0)), len(routes) - 1))

    current_required = routes[active].count(key)
    available_copies = int(option_by_key.get(key, {}).get("count", 0))
    if current_required >= available_copies:
        st.toast(f"You only play {available_copies} copies of this card.")
        return

    routes[active].append(key)
    set_builder_routes(routes, option_by_key)


def remove_card_from_route(route_index: int, key: str, option_by_key: dict[str, dict], occurrence_index: int | None = None) -> None:
    """Remove one copy of a card from a bracket.

    If occurrence_index is omitted, remove the first matching copy.
    """
    routes = normalize_builder_routes(st.session_state.get("starting_hand_click_builder_routes", [[]]), option_by_key)
    if 0 <= route_index < len(routes):
        if occurrence_index is not None and 0 <= occurrence_index < len(routes[route_index]):
            routes[route_index].pop(occurrence_index)
        else:
            for i, item in enumerate(routes[route_index]):
                if item == key:
                    routes[route_index].pop(i)
                    break
    set_builder_routes(routes, option_by_key)


def render_click_card_tile(card: dict, option_by_key: dict[str, dict], *, key_prefix: str) -> None:
    image_url = card.get("image_url")
    if image_url:
        st.image(image_url, use_container_width=True)
    else:
        st.markdown(
            f'<div class="placeholder-card compact-placeholder">{html.escape(card["label"])}<br>No image</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
        <div class="native-card-label">
          <b>{html.escape(card['label'])}</b><br>
          <span>x{card['count']} • {html.escape(card['supertype'])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Add one copy", key=f"{key_prefix}_add_{card['key']}", use_container_width=True):
        add_card_to_active_route(card["key"], option_by_key)
        st.rerun()


def render_starting_hand_lab(summary, card_odds_df, deck) -> None:
    st.markdown(
        """
        <div class="section-card">
          <h2>Starting Hand Planner</h2>
          <p class="small-muted">
            Build a Success Statement in plain English. Click cards from your deck to describe the starts you would be happy to see.
            A <b>success way</b> is one acceptable start. Put cards in the same way when you need all of them together. Add another way when either start is good enough.
            This page checks only your opening hand plus optional natural draw — it does not play Items, Supporters, Abilities, search effects, or Turn 1 actions.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    options = deck_card_options(deck)
    label_by_key = {option["key"]: option["label"] for option in options}
    option_by_key = {option["key"]: option for option in options}

    if not options:
        st.info("Analyze a deck first, then use the Starting Hand Planner.")
        return

    st.markdown(
        """
        <div class="builder-card">
          <b>Plain-English rule:</b>
          <span class="logic-token">one success way</span>
          <span class="operator-chip">=</span>
          <span class="logic-token">one kind of opening hand you would keep</span>
          <span class="operator-chip">OR</span>
          <span class="logic-token">another success way is another acceptable start</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    include_turn_draw = st.toggle(
        "Include natural draw for turn",
        value=True,
        help="When on, probability is for legal opening hand plus one natural draw. No card effects are included.",
    )

    st.markdown(
        """
        <div class="how-to-card">
          <div class="how-to-title">How to use this</div>
          <div class="how-to-steps">
            <span><b>1</b>Pick a start you would be happy with.</span>
            <span><b>2</b>Click cards from your deck to add them to that success way.</span>
            <span><b>3</b>Add another success way when a different start also works.</span>
          </div>
          <div class="how-to-example">
            Example: suppose your plan wants <b>Lunatone + Solrock</b>. Starting with both is great, but <b>Lunatone + Fighting Gong</b> may also be acceptable because Fighting Gong can find Solrock later.
            Build that as <b>(Lunatone AND Solrock) OR (Lunatone AND Fighting Gong)</b>.
          </div>
          <div class="how-to-example">
            If your hand matches more than one success way, it is still counted only once. You do not need probability formulas — just describe the hands you would be happy to open.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    routes = current_builder_routes(options, option_by_key)
    non_empty_routes = [route for route in routes if route]

    def score_and_render(routes_to_score: list[list[str]], *, compact: bool = False) -> bool:
        routes_to_score = [route for route in routes_to_score if route]
        if not routes_to_score:
            return False

        statement_keys: list[str] = []
        for route in routes_to_score:
            for key in route:
                if key not in statement_keys:
                    statement_keys.append(key)

        if len(statement_keys) > 8:
            st.error("This builder supports up to 8 unique card names in one statement. Repeated copies of the same card are allowed.")
            return False

        selected = [option_by_key[key] for key in statement_keys]
        selected_index = {key: i for i, key in enumerate(statement_keys)}
        route_indices = [[selected_index[key] for key in route if key in selected_index] for route in routes_to_score]
        route_indices = [route for route in route_indices if route]
        card_counts = [card["count"] for card in selected]
        card_is_basic = [card["is_basic"] for card in selected]

        try:
            result = custom_starting_hand_statement_probabilities(
                deck_size=summary["deck_size"],
                basic_count=summary["basic_count"],
                card_counts=card_counts,
                card_is_basic=card_is_basic,
                success_routes=route_indices,
            )
            selected_probability = p_dnf_statement_given_legal_opening(
                deck_size=summary["deck_size"],
                basic_count=summary["basic_count"],
                card_counts=card_counts,
                card_is_basic=card_is_basic,
                success_routes=route_indices,
                include_turn_draw=include_turn_draw,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            return False

        time_label = "opening hand + draw for turn" if include_turn_draw else "opening hand only"
        st.markdown(
            f"""
            <div class="result-glow" style="margin: 1rem 0 1.1rem 0;">
              <div class="result-kicker">Exact chance • after mulligans</div>
              <div class="result-number">{pct_from_probability(selected_probability)}</div>
              <div class="result-copy">Chance that your {html.escape(time_label)} matches the Success Statement.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not compact:
            m1, m2, m3 = st.columns(3)
            with m1:
                metric_card("Opening hand only", f"{result['opening_hand_percent']:.2f}%", "After mulligans")
            with m2:
                metric_card("Opening hand + draw", f"{result['after_turn_draw_percent']:.2f}%", "No card effects included")
            with m3:
                metric_card("Extra from draw", f"{result['draw_for_turn_increment_percent']:.2f}%", "How much the draw helps")
        return True

    score_and_render(non_empty_routes, compact=True)

    st.markdown(
        f"""
        <div class="native-builder-shell">
          <div class="native-statement-panel">
            <div class="result-kicker">Success Statement</div>
            <div class="native-statement">{render_statement_html(non_empty_routes, option_by_key)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.92, 1.65], gap="large")

    with left:
        st.markdown("### Your deck")
        st.caption("Click a card to add one copy to the highlighted success way. Click the same card again if your start needs two copies.")
        filter_text = st.text_input("Search cards", placeholder="Search cards in this deck...", label_visibility="collapsed")
        filtered_options = [
            option for option in options
            if filter_text.lower() in f"{option['label']} {option['name']} {option['supertype']}".lower()
        ]
        if not filtered_options:
            st.info("No cards match your search.")
        else:
            # A stable, native Streamlit grid. This is much more reliable than an iframe component.
            for start in range(0, len(filtered_options), 3):
                cols = st.columns(3)
                for col, card in zip(cols, filtered_options[start:start + 3]):
                    with col:
                        with st.container(border=True):
                            render_click_card_tile(card, option_by_key, key_prefix=f"tray_{start}")

    with right:
        st.markdown("### Build your success ways")
        st.caption("Cards in the same way must all show up. Different ways are alternatives.")

        for idx, route in enumerate(routes):
            is_active = idx == st.session_state.get("starting_hand_click_builder_active_route", 0)
            if idx > 0:
                st.markdown('<div class="success-way-divider">OR</div>', unsafe_allow_html=True)
            with st.container(border=True):
                header_cols = st.columns([1.4, 0.9, 0.9])
                with header_cols[0]:
                    st.markdown(f"#### {'✅ ' if is_active else ''}Success way {idx + 1}")
                    st.caption("Active: cards you click will be added here." if is_active else "All cards in this success way must be in your opener.")
                with header_cols[1]:
                    if st.button("Use this way", key=f"make_active_{idx}", disabled=is_active, use_container_width=True):
                        st.session_state["starting_hand_click_builder_active_route"] = idx
                        st.rerun()
                with header_cols[2]:
                    if st.button("Remove way", key=f"remove_route_{idx}", disabled=len(routes) <= 1, use_container_width=True):
                        routes.pop(idx)
                        if not routes:
                            routes = [[]]
                        st.session_state["starting_hand_click_builder_routes"] = routes
                        st.session_state["starting_hand_click_builder_active_route"] = max(0, min(idx, len(routes) - 1))
                        st.rerun()

                if not route:
                    st.markdown(
                        '<div class="native-empty-bracket">Click cards from your deck to add them to this success way.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    for card_idx, key in enumerate(route):
                        card = option_by_key[key]
                        card_cols = st.columns([0.18, 0.62, 0.20])
                        with card_cols[0]:
                            if card.get("image_url"):
                                st.image(card["image_url"], use_container_width=True)
                        with card_cols[1]:
                            st.markdown(f"**{html.escape(card['label'])}**  ")
                            st.caption(f"x{card['count']} • {card['supertype']}")
                            if card_idx < len(route) - 1:
                                st.markdown('<span class="operator-chip">AND</span>', unsafe_allow_html=True)
                        with card_cols[2]:
                            if st.button("Remove", key=f"remove_card_{idx}_{card_idx}_{key}", use_container_width=True):
                                remove_card_from_route(idx, key, option_by_key, occurrence_index=card_idx)
                                st.rerun()

        action_cols = st.columns([1, 1, 1])
        with action_cols[0]:
            if st.button("+ Add another way", use_container_width=True):
                routes.append([])
                st.session_state["starting_hand_click_builder_routes"] = routes
                st.session_state["starting_hand_click_builder_active_route"] = len(routes) - 1
                st.rerun()
        with action_cols[1]:
            if st.button("Clear statement", use_container_width=True):
                st.session_state["starting_hand_click_builder_routes"] = [[]]
                st.session_state["starting_hand_click_builder_active_route"] = 0
                st.rerun()
        with action_cols[2]:
            if st.button("Reset example", use_container_width=True):
                st.session_state["starting_hand_click_builder_routes"] = default_starting_hand_example_routes(options)
                st.session_state["starting_hand_click_builder_active_route"] = 0
                st.rerun()

    routes = normalize_builder_routes(st.session_state.get("starting_hand_click_builder_routes", [[]]), option_by_key)
    non_empty_routes = [route for route in routes if route]
    if not non_empty_routes:
        st.info("Click at least one card to calculate the statement.")
        return

    if not score_and_render(non_empty_routes, compact=False):
        return

    # Use only the cards that actually appear in the Success Statement.
    statement_keys: list[str] = []
    for route in non_empty_routes:
        for key in route:
            if key not in statement_keys:
                statement_keys.append(key)

    selected = [option_by_key[key] for key in statement_keys]
    selected_index = {key: i for i, key in enumerate(statement_keys)}
    route_rows = []
    card_counts = [card["count"] for card in selected]
    card_is_basic = [card["is_basic"] for card in selected]

    for i, route in enumerate(non_empty_routes, start=1):
        indices = [[selected_index[key] for key in route if key in selected_index]]
        opening_p = p_dnf_statement_given_legal_opening(
            deck_size=summary["deck_size"],
            basic_count=summary["basic_count"],
            card_counts=card_counts,
            card_is_basic=card_is_basic,
            success_routes=indices,
            include_turn_draw=False,
        )
        draw_p = p_dnf_statement_given_legal_opening(
            deck_size=summary["deck_size"],
            basic_count=summary["basic_count"],
            card_counts=card_counts,
            card_is_basic=card_is_basic,
            success_routes=indices,
            include_turn_draw=True,
        )
        route_rows.append(
            {
                "Success way": f"Way {i}",
                "Meaning": route_to_text(route, label_by_key),
                "Opening hand only": f"{opening_p:.2%}",
                "Opening hand + draw": f"{draw_p:.2%}",
            }
        )

    st.subheader("Compare each success way")
    st.caption("These individual chances can overlap. If one hand satisfies multiple success ways, the big result counts that hand once, not multiple times.")
    st.dataframe(pd.DataFrame(route_rows), use_container_width=True, hide_index=True)

    with st.expander("Cards used in this Success Statement", expanded=False):
        render_selected_card_preview(selected, card_odds_df)

def render_gallery(card_odds_df) -> None:
    st.subheader("Visual card probability gallery")
    st.caption("Each card shows opening + draw access, at least 1 prized copy, and all copies prized.")
    controls = st.columns([1, 1, 1, 1])
    with controls[0]:
        prizes_taken = st.slider("Prizes taken", min_value=0, max_value=5, value=0)
    card_types = ["All"] + sorted([x for x in card_odds_df["supertype"].dropna().unique().tolist()])
    with controls[1]:
        card_type_filter = st.selectbox("Card type", options=card_types, index=0)
    with controls[2]:
        sort_mode = st.selectbox(
            "Sort cards by",
            options=["Opening hand + draw", "At least 1 prized", "All copies prized", "Card count", "Deck order / parsed order"],
            index=1,
        )
    with controls[3]:
        columns_per_row = st.selectbox("Cards per row", options=[2, 3, 4, 5, 6], index=2)
    render_card_gallery(card_odds_df, prizes_taken, card_type_filter, sort_mode, columns_per_row)


def render_prize_map(prize_df) -> None:
    st.subheader("Prize-card probabilities")
    st.caption("Prize probabilities are conditioned on first keeping a legal opening hand.")
    show_chart(make_prize_chart(prize_df, top_n_cards=len(prize_df)))
    show_chart(make_all_copies_prized_chart(prize_df, top_n_cards=len(prize_df)))
    show_chart(make_prize_survival_heatmap(prize_df))
    st.subheader("Prize probability table")
    st.dataframe(
        pretty_table(prize_df.drop(columns=["image_url", "image_large_url"], errors="ignore")),
        use_container_width=True,
        hide_index=True,
    )


def render_diagnostics(card_odds_df, parsed_df) -> None:
    st.subheader("Diagnostics")
    st.caption("Verify card matching and see how legal-hand conditioning changes card access.")
    show_chart(make_conditioning_effect_chart(card_odds_df, top_n_cards=len(card_odds_df)))
    st.subheader("Parsed card matching")
    st.dataframe(parsed_df.rename(columns=DISPLAY_COLUMN_NAMES), use_container_width=True, hide_index=True)


def main() -> None:
    apply_custom_css()
    init_session_state()
    render_hero()
    deck_input_panel()

    if not st.session_state.has_analyzed or st.session_state.analysis_results is None:
        render_empty_state()
    else:
        summary, mulligan_df, card_odds_df, prize_df, parsed_df, deck = st.session_state.analysis_results
        if summary["deck_size"] != 60:
            st.warning(f"Deck has {summary['deck_size']} cards, not 60.")

        dashboard_tab, starting_tab, gallery_tab, prize_tab, diagnostics_tab = st.tabs(
            [
                "Deck Dashboard",
                "Starting Hand Planner",
                "Card Gallery",
                "Prize Map",
                "Diagnostics",
            ]
        )

        with dashboard_tab:
            render_deck_dashboard(summary, mulligan_df, card_odds_df, parsed_df)
        with starting_tab:
            render_starting_hand_lab(summary, card_odds_df, deck)
        with gallery_tab:
            render_gallery(card_odds_df)
        with prize_tab:
            render_prize_map(prize_df)
        with diagnostics_tab:
            render_diagnostics(card_odds_df, parsed_df)

    st.markdown(
        """
        <div class="small-muted" style="margin-top: 32px; padding-bottom: 18px;">
        Unofficial fan-made deck analysis tool. Not affiliated with, endorsed by, or sponsored by
        The Pokémon Company, Nintendo, Creatures, or Game Freak.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
