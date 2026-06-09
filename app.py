"""
Explanation

This is the main Streamlit app file.

It controls the website interface and overall user experience.

Main responsibilities:
- Apply custom dark-mode styling.
- Provide a sidebar where the user pastes a decklist.
- Run the deck analysis when the user clicks the Analyze button.
- Display a professional dashboard with:
  - summary metrics
  - deck composition
  - mulligan probabilities
  - opening-hand and turn-draw odds
  - prize-card probabilities
  - image-based card probability gallery
  - prize-after-X-prizes controls
  - parsed card matching diagnostics

Most of the actual probability and parsing logic is imported from the src/ folder.
"""

import html
import pandas as pd
import streamlit as st

from src.analysis import analyze_deck_opening_hand, format_probability_table
from src.charts import (
    make_deck_composition_chart,
    make_mulligan_chart,
    make_card_odds_chart,
    make_conditioning_effect_chart,
    make_prize_chart,
    make_all_copies_prized_chart,
    make_prize_survival_heatmap,
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
    "P_in_hand_after_turn_draw": "Turn 1 raw access",
    "increase_from_turn_draw": "Turn draw gain",
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


CACHE_VERSION = "card-gallery-v1"


def apply_custom_css():
    st.markdown(
        """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(31, 111, 235, 0.18), transparent 28rem),
            radial-gradient(circle at top right, rgba(124, 58, 237, 0.14), transparent 30rem),
            linear-gradient(180deg, #070A12 0%, #0B1020 45%, #090D16 100%);
        color: #F8FAFC;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1020 0%, #111827 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    [data-testid="stSidebar"] * {
        color: #E5E7EB;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    h1, h2, h3 {
        letter-spacing: -0.03em;
    }

    h1 {
        font-size: 3.3rem !important;
        line-height: 1.05 !important;
        margin-bottom: 0.35rem !important;
    }

    .hero-card {
        padding: 2rem 2.2rem;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 24px;
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.86)),
            radial-gradient(circle at top right, rgba(59, 130, 246, 0.25), transparent 18rem);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.4rem;
    }

    .hero-kicker {
        color: #93C5FD;
        text-transform: uppercase;
        font-weight: 800;
        letter-spacing: 0.14em;
        font-size: 0.82rem;
        margin-bottom: 0.6rem;
    }

    .hero-subtitle {
        color: #CBD5E1;
        font-size: 1.08rem;
        max-width: 920px;
        line-height: 1.65;
        margin-top: 0.5rem;
    }

    .feature-row {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }

    .feature-pill {
        border: 1px solid rgba(147, 197, 253, 0.24);
        background: rgba(15, 23, 42, 0.74);
        color: #DBEAFE;
        padding: 0.48rem 0.75rem;
        border-radius: 999px;
        font-size: 0.9rem;
    }

    .metric-card {
        padding: 1.15rem 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.72);
        box-shadow: 0 14px 38px rgba(0, 0, 0, 0.22);
        height: 100%;
    }

    .metric-label {
        color: #94A3B8;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 2rem;
        font-weight: 900;
        line-height: 1.1;
    }

    .metric-note {
        color: #CBD5E1;
        font-size: 0.86rem;
        margin-top: 0.4rem;
    }

    .section-card {
        padding: 1.2rem;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.58);
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 999px;
        color: #CBD5E1;
        padding: 0.4rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2563EB, #7C3AED) !important;
        color: white !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 18px;
        overflow: hidden;
    }

    .stButton > button {
        border-radius: 999px;
        border: 0;
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        color: white;
        font-weight: 800;
        padding: 0.7rem 1.25rem;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.28);
    }

    .stButton > button:hover {
        border: 0;
        filter: brightness(1.08);
        transform: translateY(-1px);
    }

    .gallery-card {
        position: relative;
        border-radius: 18px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.24);
        background: rgba(15, 23, 42, 0.78);
        box-shadow: 0 20px 48px rgba(0, 0, 0, 0.38);
        margin-bottom: 1rem;
        transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
    }

    .gallery-card:hover {
        transform: translateY(-3px);
        border-color: rgba(96, 165, 250, 0.68);
        box-shadow: 0 28px 70px rgba(37, 99, 235, 0.22);
    }

    .gallery-image-wrap {
        position: relative;
        background:
            radial-gradient(circle at top, rgba(96, 165, 250, 0.22), transparent 16rem),
            linear-gradient(180deg, #111827, #020617);
    }

    .gallery-image {
        width: 100%;
        display: block;
        border-radius: 16px;
    }

    .gallery-placeholder {
        height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        text-align: center;
        color: #CBD5E1;
        font-weight: 800;
        background:
            radial-gradient(circle at top, rgba(96, 165, 250, 0.22), transparent 16rem),
            linear-gradient(180deg, #111827, #020617);
    }

    .count-pill {
        position: absolute;
        top: 10px;
        right: 10px;
        background: rgba(2, 6, 23, 0.82);
        border: 1px solid rgba(248, 250, 252, 0.25);
        color: #F8FAFC;
        border-radius: 999px;
        padding: 0.28rem 0.55rem;
        font-size: 0.78rem;
        font-weight: 900;
        backdrop-filter: blur(10px);
    }

    .gallery-overlay {
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        padding: 0.7rem;
        background: linear-gradient(180deg, rgba(2, 6, 23, 0), rgba(2, 6, 23, 0.94) 38%, rgba(2, 6, 23, 0.98));
    }

    .gallery-title {
        color: #FFFFFF;
        font-weight: 900;
        font-size: 0.95rem;
        line-height: 1.18;
        margin-bottom: 0.45rem;
        text-shadow: 0 2px 12px rgba(0,0,0,0.65);
    }

    .prob-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.38rem;
    }

    .prob-box {
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 12px;
        padding: 0.42rem 0.35rem;
        background: rgba(15, 23, 42, 0.74);
        backdrop-filter: blur(10px);
        min-height: 54px;
    }

    .prob-label {
        color: #94A3B8;
        font-size: 0.62rem;
        line-height: 1.1;
        text-transform: uppercase;
        font-weight: 900;
        letter-spacing: 0.04em;
        margin-bottom: 0.18rem;
    }

    .prob-value {
        color: #F8FAFC;
        font-size: 0.92rem;
        font-weight: 950;
        line-height: 1.05;
    }

    .prob-value-hot {
        color: #FBBF24;
    }

    .prob-value-danger {
        color: #FB7185;
    }

    .prob-value-good {
        color: #60A5FA;
    }

    .small-muted {
        color: #94A3B8;
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .footer {
        margin-top: 3rem;
        color: #64748B;
        font-size: 0.82rem;
        border-top: 1px solid rgba(148, 163, 184, 0.14);
        padding-top: 1rem;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def pct(value, decimals=2):
    if pd.isna(value):
        return "—"
    return f"{value:.{decimals}%}"


def metric_card(label: str, value: str, note: str = ""):
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
        return (
            "P_at_least_1_prized",
            "P_all_copies_prized",
            "Initial prizes",
        )

    suffix = "prize_taken" if prizes_taken == 1 else "prizes_taken"

    return (
        f"P_still_prized_after_{prizes_taken}_{suffix}",
        f"P_all_copies_still_prized_after_{prizes_taken}_{suffix}",
        f"After {prizes_taken} prize{'s' if prizes_taken != 1 else ''} taken",
    )


def render_card_tile(row: pd.Series, prizes_taken: int):
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
        image_html = f'<img class="gallery-image" src="{html.escape(str(image_url))}" alt="{safe_card_label}">'
    else:
        image_html = f'<div class="gallery-placeholder">{safe_card_label}<br><span class="small-muted">No image found</span></div>'

    st.markdown(
        f"""
<div class="gallery-card">
    <div class="gallery-image-wrap">
        {image_html}
        <div class="count-pill">x{count}</div>
        <div class="gallery-overlay">
            <div class="gallery-title">{safe_card_label}</div>
            <div class="prob-grid">
                <div class="prob-box">
                    <div class="prob-label">Turn 1 raw</div>
                    <div class="prob-value prob-value-good">{pct(turn_access, 2)}</div>
                </div>
                <div class="prob-box">
                    <div class="prob-label">≥1 prized<br>{safe_context_label}</div>
                    <div class="prob-value prob-value-hot">{pct(at_least_prized, 2)}</div>
                </div>
                <div class="prob-box">
                    <div class="prob-label">All prized<br>{safe_context_label}</div>
                    <div class="prob-value prob-value-danger">{pct(all_prized, 4)}</div>
                </div>
            </div>
        </div>
    </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_card_gallery(
    card_df: pd.DataFrame,
    prizes_taken: int,
    card_type_filter: str,
    sort_mode: str,
    columns_per_row: int,
):
    gallery_df = card_df.copy()

    if card_type_filter != "All":
        gallery_df = gallery_df[gallery_df["supertype"].fillna("Unknown") == card_type_filter].copy()

    at_least_col, all_col, _ = prize_column_names(prizes_taken)

    sort_map = {
        "Turn 1 raw access": "P_in_hand_after_turn_draw",
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

    rows = gallery_df.to_dict("records")

    for start in range(0, len(rows), columns_per_row):
        cols = st.columns(columns_per_row)

        for col, row_dict in zip(cols, rows[start:start + columns_per_row]):
            with col:
                render_card_tile(pd.Series(row_dict), prizes_taken)


@st.cache_data(show_spinner=False)
def cached_analysis(decklist_text: str, max_mulligans: int, cache_version: str):
    return analyze_deck_opening_hand(
        decklist_text=decklist_text,
        max_mulligans=max_mulligans,
    )


st.set_page_config(
    page_title="PKMN TCG Deck Analyzer",
    page_icon="🃏",
    layout="wide",
)

apply_custom_css()

with st.sidebar:
    st.markdown("## 🃏 PKMN TCG Deck Analyzer")
    st.caption("Paste a decklist and get exact probability diagnostics.")

    decklist_text = st.text_area(
        "Decklist",
        value=EXAMPLE_DECKLIST,
        height=430,
        help="Paste a Pokémon TCG Live, Limitless, or similar decklist export.",
    )

    max_mulligans = st.slider(
        "Show mulligans up to",
        min_value=3,
        max_value=10,
        value=6,
        help="The final bucket is shown as X+ mulligans.",
    )

    analyze = st.button("Analyze deck", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        """
<div class="small-muted">
Current version uses exact probability formulas for opening hands and prizes.
Search-card reachability and card-swap optimization are planned future modules.
</div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    """
<div class="hero-card">
    <div class="hero-kicker">Probability-driven deckbuilding</div>
    <h1>PKMN TCG Deck Analyzer</h1>
    <div class="hero-subtitle">
        Analyze mulligans, opening hands, turn-draw consistency, and prize-card risk from a pasted decklist.
        Built for competitive deck tuning, prize mapping, and future full-deck optimization.
    </div>
    <div class="feature-row">
        <div class="feature-pill">Exact mulligan odds</div>
        <div class="feature-pill">Legal opening-hand conditioning</div>
        <div class="feature-pill">Visual card gallery</div>
        <div class="feature-pill">Prize mapping</div>
    </div>
</div>
    """,
    unsafe_allow_html=True,
)


if not analyze:
    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Step 1", "Paste", "Paste a 60-card Pokémon TCG decklist in the sidebar.")

    with col2:
        metric_card("Step 2", "Analyze", "Run exact opening-hand and prize-card probability calculations.")

    with col3:
        metric_card("Step 3", "Tune", "Use visual card overlays to spot consistency and prize-liability issues.")

    st.markdown(
        """
<div class="section-card">
<h3>What this version calculates</h3>
<ul>
<li>Probability of mulliganing 0, 1, 2, 3, ... times</li>
<li>Probability of seeing each card in a legal opening 7-card hand</li>
<li>Probability of seeing each card after drawing for turn</li>
<li>Probability of at least one copy being prized</li>
<li>Probability of all copies being prized</li>
<li>Probability of cards still being prized after taking prizes</li>
<li>Visual card-image gallery with probability overlays</li>
</ul>
</div>
        """,
        unsafe_allow_html=True,
    )

else:
    try:
        with st.spinner("Analyzing deck and fetching card images..."):
            summary, mulligan_df, card_odds_df, prize_df, parsed_df, deck = cached_analysis(
                decklist_text,
                max_mulligans,
                CACHE_VERSION,
            )

        top_n_cards = len(card_odds_df)

        if summary["deck_size"] != 60:
            st.warning(f"Deck has {summary['deck_size']} cards, not 60.")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card("Deck size", str(summary["deck_size"]), "Total parsed cards")
        with c2:
            metric_card("Basic Pokémon", str(summary["basic_count"]), "Cards that make an opening hand legal")
        with c3:
            metric_card("Mulligan chance", f"{summary['p_mulligan_one_hand']:.2%}", "Chance one opening attempt has no Basic")
        with c4:
            metric_card(
                "Expected mulligans",
                f"{summary['expected_mulligans_before_legal_hand']:.3f}",
                "Average mulligans before a legal hand",
            )

        overview_tab, gallery_tab, hand_tab, prize_tab, diagnostics_tab = st.tabs(
            ["Overview", "Card gallery", "Opening hands", "Prize map", "Diagnostics"]
        )

        with overview_tab:
            left, right = st.columns([0.9, 1.1])

            with left:
                st.plotly_chart(
                    make_deck_composition_chart(parsed_df),
                    use_container_width=True,
                )

            with right:
                st.plotly_chart(
                    make_mulligan_chart(mulligan_df),
                    use_container_width=True,
                )

            st.subheader("Mulligan probabilities")
            st.dataframe(
                pretty_table(mulligan_df),
                use_container_width=True,
                hide_index=True,
            )

        with gallery_tab:
            st.subheader("Visual card probability gallery")
            st.caption(
                "Each card image shows Turn 1 raw access, at least 1 prized copy, and all copies prized. "
                "Use the controls to view probabilities after prizes have been taken."
            )

            controls = st.columns([1, 1, 1, 1])

            with controls[0]:
                prizes_taken = st.slider(
                    "Prizes taken",
                    min_value=0,
                    max_value=5,
                    value=0,
                    help="Updates the prized-card overlays to show what remains after taking X random prizes.",
                )

            card_types = ["All"] + sorted(
                [x for x in card_odds_df["supertype"].dropna().unique().tolist()]
            )

            with controls[1]:
                card_type_filter = st.selectbox(
                    "Card type",
                    options=card_types,
                    index=0,
                )

            with controls[2]:
                sort_mode = st.selectbox(
                    "Sort cards by",
                    options=[
                        "Turn 1 raw access",
                        "At least 1 prized",
                        "All copies prized",
                        "Card count",
                        "Deck order / parsed order",
                    ],
                    index=1,
                )

            with controls[3]:
                columns_per_row = st.selectbox(
                    "Cards per row",
                    options=[3, 4, 5, 6],
                    index=1,
                )

            render_card_gallery(
                card_df=card_odds_df,
                prizes_taken=prizes_taken,
                card_type_filter=card_type_filter,
                sort_mode=sort_mode,
                columns_per_row=columns_per_row,
            )

        with hand_tab:
            st.plotly_chart(
                make_card_odds_chart(card_odds_df, top_n_cards=top_n_cards),
                use_container_width=True,
            )

            st.subheader("Opening-hand probability table")
            st.caption(
                "Legal opening 7 means the hand contains at least one Basic Pokémon. "
                "Turn 1 raw access means opening hand plus drawing for turn, without search effects."
            )
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
            st.dataframe(
                pretty_table(card_odds_df[hand_cols]),
                use_container_width=True,
                hide_index=True,
            )

        with prize_tab:
            st.plotly_chart(
                make_prize_chart(prize_df, top_n_cards=top_n_cards),
                use_container_width=True,
            )

            st.plotly_chart(
                make_all_copies_prized_chart(prize_df, top_n_cards=top_n_cards),
                use_container_width=True,
            )

            st.plotly_chart(
                make_prize_survival_heatmap(prize_df),
                use_container_width=True,
            )

            st.subheader("Prize probability table")
            st.caption(
                "The 'after X prizes taken' columns assume prizes taken are random with respect to the target card."
            )
            st.dataframe(
                pretty_table(prize_df.drop(columns=["image_url", "image_large_url"], errors="ignore")),
                use_container_width=True,
                hide_index=True,
            )

        with diagnostics_tab:
            st.plotly_chart(
                make_conditioning_effect_chart(card_odds_df, top_n_cards=top_n_cards),
                use_container_width=True,
            )

            st.subheader("Parsed card matching")
            st.caption(
                "This table shows how the app classified each parsed decklist entry and whether an API image was found."
            )
            st.dataframe(
                parsed_df.rename(columns=DISPLAY_COLUMN_NAMES),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as e:
        st.error(str(e))


st.markdown(
    """
<div class="footer">
Unofficial fan-made deck analysis tool. Not affiliated with, endorsed by, or sponsored by
The Pokémon Company, Nintendo, Creatures, or Game Freak.
</div>
    """,
    unsafe_allow_html=True,
)