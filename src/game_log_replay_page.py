from __future__ import annotations

import base64
import html
import textwrap
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from src.game_log.models import CardRef, GameState, PlayerState, PokemonInPlay
from src.game_log.parser import parse_battle_log, parse_card_refs
from src.game_log.reducer import build_replay_frames
from src.game_log.resolver import exported_id_to_api_card_id, image_url_for_card_ref


def _render_raw_html(raw_html: str) -> None:
    cleaned = textwrap.dedent(str(raw_html or "")).strip()
    cleaned = "\n".join(line.lstrip() for line in cleaned.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def _card_label(card: CardRef | None) -> str:
    if card is None:
        return "Empty"
    if card.unknown:
        return "Unknown card"
    if card.exported_id and card.name:
        return f"{card.name} [{card.exported_id}]"
    return card.name or card.exported_id or "Unknown card"


def _short_card_label(card: CardRef | None) -> str:
    label = _card_label(card)
    label = label.replace("Basic ", "")
    if len(label) > 34:
        return label[:31] + "..."
    return label


def _card_html(card: CardRef | None, *, small: bool = False) -> str:
    if card is None:
        return "<div class='glr-card glr-empty-card'>Empty</div>"

    label = html.escape(_card_label(card))
    short_label = html.escape(_short_card_label(card))
    img = image_url_for_card_ref(card)
    cls = "glr-card glr-card-small" if small else "glr-card"

    if img:
        return f"""
        <div class="{cls}" title="{label}">
          <img src="{html.escape(img)}" alt="{label}">
          <div class="glr-card-name">{short_label}</div>
        </div>
        """

    return f"""
    <div class="{cls} glr-no-image" title="{label}">
      <div class="glr-holo-sheen"></div>
      <div class="glr-card-name">{short_label}</div>
    </div>
    """


def _pokemon_html(pokemon: PokemonInPlay | None) -> str:
    if pokemon is None:
        return """
        <div class="glr-pokemon glr-empty-slot">
          <span>Empty</span>
        </div>
        """

    attached = "".join(_card_html(card, small=True) for card in pokemon.attached[:5])
    if len(pokemon.attached) > 5:
        attached += f"<span class='glr-more-chip'>+{len(pokemon.attached) - 5}</span>"

    stack = "".join(_card_html(card, small=True) for card in pokemon.evolution_stack[-2:])

    damage = int(pokemon.damage or 0)
    damage_badge = f'<div class="glr-damage-counter">{damage}</div>' if damage > 0 else ""

    instance_label = html.escape(getattr(pokemon, "copy_label", "") or "")
    instance_badge = (
        f'<div class="glr-instance-badge">{instance_label}</div>'
        if instance_label
        else ""
    )

    inferred = " glr-inferred" if getattr(pokemon, "inferred", False) else ""

    return f"""
    <div class="glr-pokemon{inferred}">
      <div class="glr-card-stack">
        {_card_html(pokemon.card)}
        {damage_badge}
        {instance_badge}
      </div>
      <div class="glr-pokemon-meta">
        <span>{damage} dmg</span>
        <span>{len(pokemon.attached)} energy</span>
      </div>
      <div class="glr-mini-row">{attached}</div>
      <div class="glr-mini-row">{stack}</div>
    </div>
    """


def _bench_html(player: PlayerState) -> str:
    cells = []

    for i in range(5):
        pokemon = player.bench[i] if i < len(player.bench) else None
        cells.append(
            f"""
            <div class="glr-bench-slot">
              <div class="glr-zone-label">Bench {i + 1}</div>
              {_pokemon_html(pokemon)}
            </div>
            """
        )

    return "<div class='glr-bench-row'>" + "".join(cells) + "</div>"


def _prizes_html(player: PlayerState) -> str:
    cells = []

    for i in range(player.starting_prize_count):
        if i < len(player.prizes_taken):
            prize = player.prizes_taken[i]
            if prize.unknown:
                cls = "glr-prize glr-prize-taken-unknown"
                label = "Taken"
            else:
                cls = "glr-prize glr-prize-taken-known"
                label = _short_card_label(prize)
        elif i < len(player.user_known_prizes):
            cls = "glr-prize glr-prize-remembered"
            label = _short_card_label(player.user_known_prizes[i])
        else:
            cls = "glr-prize glr-prize-unknown"
            label = "Unknown"

        cells.append(f"<div class='{cls}' title='{html.escape(label)}'>{html.escape(label)}</div>")

    return "<div class='glr-prize-grid'>" + "".join(cells) + "</div>"


def _card_counter_html(cards: list[CardRef], *, limit: int = 8) -> str:
    if not cards:
        return "<span class='glr-muted'>None known</span>"

    counts = Counter(_card_label(card) for card in cards)
    chips = []

    for label, count in counts.most_common(limit):
        text = f"{count}x {label}" if count > 1 else label
        chips.append(f"<span class='glr-chip' title='{html.escape(text)}'>{html.escape(text[:34])}</span>")

    if len(counts) > limit:
        chips.append(f"<span class='glr-chip glr-muted'>+{len(counts) - limit}</span>")

    return "<div class='glr-chip-row'>" + "".join(chips) + "</div>"


def _event_action_title(event) -> str:
    if event is None:
        return "Before game"

    labels = {
        "setup": "Setup",
        "opening_hand": "Opening hand",
        "turn_start": "Turn start",
        "draw_hidden": "Draw",
        "draw_revealed": "Draw",
        "draw_count": "Draw",
        "draw_and_play_to_bench": "Search and Bench",
        "play_to_active": "Play to Active",
        "play_to_bench": "Play to Bench",
        "play_stadium": "Play Stadium",
        "play_card": "Play Trainer",
        "attach_energy": "Attach Energy",
        "ability_or_attack": "Ability",
        "attack": "Attack",
        "place_damage_counters": "Place Damage Counters",
        "evolve": "Evolve",
        "knockout": "Knock Out",
        "promote_active": "Promote Active",
        "take_prize": "Take Prize",
        "add_to_hand_revealed": "Prize Revealed",
        "add_to_hand_hidden": "Prize to Hand",
        "discard_count": "Discard",
        "discard_from_play": "Discard from Play",
        "move_card": "Move Card",
        "game_end": "Game End",
    }

    return labels.get(getattr(event, "event_type", ""), getattr(event, "event_type", "Action"))


def _event_impact_class(event) -> str:
    if event is None:
        return ""

    event_type = getattr(event, "event_type", "")
    if event_type in {"attack", "knockout", "place_damage_counters"}:
        return " glr-action-impact"
    if event_type in {"play_card", "play_to_bench", "play_to_active", "evolve", "attach_energy"}:
        return " glr-action-play"
    if event_type in {"take_prize", "add_to_hand_revealed", "game_end"}:
        return " glr-action-prize"
    return ""


def _event_focus_html(event) -> str:
    if event is None:
        return """
        <div class="glr-action-panel">
          <div class="glr-zone-label">Current Action</div>
          <div class="glr-action-title">Before game</div>
          <div class="glr-action-text">Replay has not started yet.</div>
        </div>
        """

    cards = list(getattr(event, "cards", []) or [])[:5]
    card_strip = "".join(_card_html(card, small=True) for card in cards)

    raw = html.escape(getattr(event, "raw", "") or "")
    title = html.escape(_event_action_title(event))
    actor = html.escape(getattr(event, "actor", "") or "")

    actor_html = f'<div class="glr-action-actor">{actor}</div>' if actor else ""
    impact_class = _event_impact_class(event)

    return f"""
    <div class="glr-action-panel{impact_class}">
      <div class="glr-zone-label">Current Action</div>
      <div class="glr-action-title">{title}</div>
      {actor_html}
      <div class="glr-action-cards">{card_strip}</div>
      <div class="glr-action-text">{raw}</div>
    </div>
    """



_POKEMON_TYPE_TOKENS = {
    "grass",
    "fire",
    "water",
    "lightning",
    "psychic",
    "fighting",
    "darkness",
    "metal",
    "dragon",
    "colorless",
    "fairy",
}

_TYPE_ALIASES = {
    "electric": "lightning",
    "dark": "darkness",
    "steel": "metal",
    "normal": "colorless",
}

_FALLBACK_ACTIVE_TYPE_BY_NAME = {
    "raging bolt": "dragon",
    "teal mask ogerpon": "grass",
    "wellspring mask ogerpon": "water",
    "budew": "grass",
    "dreepy": "dragon",
    "drakloak": "dragon",
    "dragapult": "dragon",
    "meowth": "colorless",
    "mega kangaskhan": "colorless",
    "fezandipiti": "darkness",
    "duskull": "psychic",
    "dusclops": "psychic",
    "dusknoir": "psychic",
    "munkidori": "darkness",
}


def _normalize_pokemon_type(raw: object) -> str:
    value = str(raw or "").strip().lower()
    value = value.replace("'", "").replace('"', "")
    value = _TYPE_ALIASES.get(value, value)
    return value if value in _POKEMON_TYPE_TOKENS else ""


def _parse_type_cell(value: object) -> list[str]:
    if value is None:
        return []

    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return []

    # Handles cells like "['Grass']", "Grass", "Grass,Poison", etc.
    cleaned = (
        raw.replace("[", " ")
        .replace("]", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("{", " ")
        .replace("}", " ")
        .replace("'", " ")
        .replace('"', " ")
        .replace("|", ",")
        .replace(";", ",")
    )

    found: list[str] = []
    for piece in cleaned.replace("/", ",").split(","):
        token = _normalize_pokemon_type(piece)
        if token and token not in found:
            found.append(token)

    # Fallback: search inside string if delimiter parsing failed.
    if not found:
        lowered = cleaned.lower()
        for token in _POKEMON_TYPE_TOKENS:
            if token in lowered and token not in found:
                found.append(token)

    return found


@st.cache_data(show_spinner=False)
def _game_review_card_type_lookup() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "data" / "all_cards.csv",
        root / "data" / "cards.csv",
        root / "data" / "card_index.csv",
    ]

    for path in candidates:
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        lower_to_col = {str(col).lower(): col for col in df.columns}

        id_cols = [
            col
            for key, col in lower_to_col.items()
            if key in {"id", "card_id", "representative_card_id"}
        ]

        name_cols = [
            col
            for key, col in lower_to_col.items()
            if key in {"name", "card_name"}
        ]

        type_cols = [
            col
            for key, col in lower_to_col.items()
            if ("type" in key and "subtype" not in key and "supertype" not in key)
        ]

        if not type_cols:
            continue

        lookup: dict[str, str] = {}

        for row in df.to_dict("records"):
            card_type = ""

            for col in type_cols:
                parsed = _parse_type_cell(row.get(col))
                if parsed:
                    card_type = parsed[0]
                    break

            if not card_type:
                continue

            for col in id_cols:
                raw_id = str(row.get(col) or "").strip().lower()
                if raw_id and raw_id != "nan":
                    lookup[raw_id] = card_type

            for col in name_cols:
                raw_name = str(row.get(col) or "").strip().lower()
                if raw_name and raw_name != "nan":
                    lookup.setdefault(raw_name, card_type)

        if lookup:
            return lookup

    return {}


def _active_pokemon_type(player: PlayerState) -> str:
    active = getattr(player, "active", None)
    card = getattr(active, "card", None)

    if card is None:
        return "neutral"

    lookup = _game_review_card_type_lookup()

    exported_id = str(getattr(card, "exported_id", "") or "").strip()
    name = str(getattr(card, "name", "") or "").strip()

    keys = []
    if exported_id:
        keys.append(exported_id.lower())
        keys.append(exported_id.replace("_", "-").lower())
        try:
            keys.append(exported_id_to_api_card_id(exported_id).lower())
        except Exception:
            pass

    if name:
        keys.append(name.lower())

    for key in keys:
        if key in lookup:
            return lookup[key]

    lowered_name = name.lower()
    for name_piece, fallback_type in _FALLBACK_ACTIVE_TYPE_BY_NAME.items():
        if name_piece in lowered_name:
            return fallback_type

    return "neutral"


def _side_html(player: PlayerState, *, opponent: bool) -> str:
    active_type = _active_pokemon_type(player)
    theme_class = f"glr-theme-{active_type}"
    bg_image_html = _type_background_img_html(active_type)

    bench = f"""
    <div>
      <div class="glr-zone-label glr-main-label">Bench</div>
      {_bench_html(player)}
    </div>
    """

    active = f"""
    <div class="glr-active-wrap">
      <div class="glr-zone-label glr-main-label">Active Spot</div>
      {_pokemon_html(player.active)}
    </div>
    """

    play_area = bench + active if opponent else active + bench

    return f"""
    <section class="glr-side {'glr-opponent' if opponent else 'glr-player'} {theme_class}" data-active-type="{html.escape(active_type)}">
      {bg_image_html}
      <div class="glr-player-header">
        <div>
          <h3>{html.escape(player.name)}</h3>
          <div class="glr-player-subtitle">{'Opponent' if opponent else 'Player'}</div>
        </div>
        <div class="glr-stats">
          <span>{player.remaining_prize_count} prizes</span>
          <span>{len(player.bench)}/5 bench</span>
          <span>{len(player.hand_known)} hand</span>
          <span>{player.hand_unknown_count} hidden</span>
          <span>{len(player.discard)} discard</span>
        </div>
      </div>

      <div class="glr-side-grid">
        <main class="glr-play-area">
          {play_area}
        </main>

        <aside class="glr-side-zones">
          <div class="glr-zone-card glr-prize-zone">
            <div class="glr-zone-label">Prize Cards</div>
            {_prizes_html(player)}
          </div>

          <div class="glr-zone-card glr-hand-zone">
            <div class="glr-zone-title-row">
              <div class="glr-zone-label">Known Hand</div>
              <span>{len(player.hand_known)}</span>
            </div>
            {_card_counter_html(player.hand_known)}
          </div>

          <div class="glr-zone-card glr-discard-zone">
            <div class="glr-zone-title-row">
              <div class="glr-zone-label">Discard</div>
              <span>{len(player.discard)}</span>
            </div>
            {_card_counter_html(player.discard)}
          </div>
        </aside>
      </div>
    </section>
    """



@st.cache_data(show_spinner=False)


@st.cache_data(show_spinner=False)
def _basic_type_background_css(active_types: tuple[str, ...] = ()) -> str:
    """
    Always embed all basic type backgrounds.

    Earlier versions only generated CSS for the current frame's active types.
    That could make one player side fall back to the old procedural gradients.
    """
    file_by_type = {
        "grass": "basic_grass.png",
        "fire": "basic_fire.png",
        "water": "basic_water.png",
        "lightning": "basic_lightning.png",
        "psychic": "basic_psychic.png",
        "fighting": "basic_fighting.png",
        "darkness": "basic_darkness.png",
        "metal": "basic_metal.png",
        "dragon": "basic_dragon.png",
        "colorless": "basic_colorless.png",
        "fairy": "basic_fairy.png",
        "neutral": "basic_colorless.png",
    }

    css_parts: list[str] = []

    for type_name, filename in file_by_type.items():
        uri = _game_review_background_data_uri(filename)
        if not uri:
            continue

        css_parts.append(
            f"""
      .glr-board .glr-side.glr-theme-{type_name} {{
        --type-bg-image: url("{uri}");
        background:
          linear-gradient(180deg, rgba(2, 6, 23, 0.06), rgba(2, 6, 23, 0.42)),
          radial-gradient(circle at 50% 68%, rgba(255, 255, 255, 0.08), transparent 35%),
          var(--type-bg-image) !important;
        background-size: cover, cover, cover !important;
        background-position: center center, center center, center center !important;
        background-repeat: no-repeat, no-repeat, no-repeat !important;
        background-color: #020617 !important;
      }}
"""
        )

    css_parts.append(
        """
      /* FINAL BASIC BACKGROUND IMAGE OVERRIDE - always wins over procedural gradients */
      .glr-board .glr-side::before {
        display: none !important;
        opacity: 0 !important;
        background: none !important;
      }

      .glr-board .glr-side::after {
        content: "";
        position: absolute;
        inset: 0;
        background:
          radial-gradient(circle at 50% 72%, rgba(255,255,255,0.07), transparent 38%),
          linear-gradient(180deg, rgba(2,6,23,0.02), rgba(2,6,23,0.30)) !important;
        pointer-events: none;
        z-index: 0;
      }

      .glr-board .glr-side > * {
        position: relative;
        z-index: 1;
      }

      .glr-board .glr-side .glr-pokemon,
      .glr-board .glr-side .glr-hand-zone,
      .glr-board .glr-side .glr-discard-zone,
      .glr-board .glr-side .glr-prize-zone,
      .glr-board .glr-side .glr-zone-card {
        background: rgba(3, 7, 18, 0.72) !important;
        backdrop-filter: blur(5px);
      }

      .glr-board .glr-side .glr-empty-slot {
        background: rgba(3, 7, 18, 0.46) !important;
      }

      .glr-board .glr-side .glr-card {
        background: rgba(3, 7, 18, 0.62) !important;
      }

      .glr-board .glr-side .glr-card img {
        background: transparent !important;
      }
"""
    )

    return "\\n".join(css_parts)



@st.cache_data(show_spinner=False)
def _game_review_background_data_uri(filename: str) -> str:
    path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "game_review"
        / "backgrounds"
        / "basic"
        / filename
    )

    if not path.exists():
        return ""

    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""

    return f"data:image/png;base64,{encoded}"


def _type_background_filename(type_name: str) -> str:
    file_by_type = {
        "grass": "basic_grass.png",
        "fire": "basic_fire.png",
        "water": "basic_water.png",
        "lightning": "basic_lightning.png",
        "psychic": "basic_psychic.png",
        "fighting": "basic_fighting.png",
        "darkness": "basic_darkness.png",
        "metal": "basic_metal.png",
        "dragon": "basic_dragon.png",
        "colorless": "basic_colorless.png",
        "fairy": "basic_fairy.png",
        "neutral": "basic_colorless.png",
    }
    return file_by_type.get(str(type_name or "neutral").strip().lower(), "basic_colorless.png")


def _type_background_inline_style(type_name: str) -> str:
    uri = _game_review_background_data_uri(_type_background_filename(type_name))
    if not uri:
        return ""

    return (
        "background-image: "
        "linear-gradient(180deg, rgba(2,6,23,0.08), rgba(2,6,23,0.48)), "
        "radial-gradient(circle at 50% 68%, rgba(255,255,255,0.08), transparent 35%), "
        f"url('{uri}') !important; "
        "background-size: cover, cover, cover !important; "
        "background-position: center center, center center, center center !important; "
        "background-repeat: no-repeat, no-repeat, no-repeat !important; "
        "background-color: #020617 !important;"
    )



@st.cache_data(show_spinner=False)
def _game_review_background_data_uri_img_layer_v1(filename: str) -> str:
    path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "game_review"
        / "backgrounds"
        / "basic"
        / filename
    )

    if not path.exists():
        return ""

    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        return ""

    return f"data:image/png;base64,{encoded}"


def _type_background_filename(type_name: str) -> str:
    file_by_type = {
        "grass": "basic_grass.png",
        "fire": "basic_fire.png",
        "water": "basic_water.png",
        "lightning": "basic_lightning.png",
        "psychic": "basic_psychic.png",
        "fighting": "basic_fighting.png",
        "darkness": "basic_darkness.png",
        "metal": "basic_metal.png",
        "dragon": "basic_dragon.png",
        "colorless": "basic_colorless.png",
        "fairy": "basic_fairy.png",
        "neutral": "basic_colorless.png",
    }
    return file_by_type.get(str(type_name or "neutral").strip().lower(), "basic_colorless.png")


def _type_background_img_html(type_name: str) -> str:
    uri = _game_review_background_data_uri_img_layer_v1(_type_background_filename(type_name))
    if not uri:
        return ""

    safe_type = html.escape(str(type_name or "neutral"))
    return (
        f'<img class="glr-side-bg-img" '
        f'src="{html.escape(uri)}" '
        f'alt="" aria-hidden="true" data-bg-type="{safe_type}">'
    )


def _board_css(density: str, active_types: tuple[str, ...] = ()) -> str:
    cinematic = density == "Cinematic"

    card_max = 118 if cinematic else 86
    active_width = 162 if cinematic else 122
    pokemon_min = 168 if cinematic else 112
    side_zone = 230 if cinematic else 178
    bench_min = 108 if cinematic else 82
    board_gap = 12 if cinematic else 8
    basic_bg_css = _basic_type_background_css(active_types)

    return f"""
    <style>
      .glr-page-shell {{
        margin-top: -0.5rem;
      }}

      .glr-hero {{
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 24px;
        padding: 18px 20px;
        margin-bottom: 14px;
        background:
          radial-gradient(circle at 16% 15%, rgba(56, 189, 248, 0.28), transparent 28%),
          radial-gradient(circle at 78% 22%, rgba(168, 85, 247, 0.22), transparent 30%),
          linear-gradient(135deg, rgba(15, 23, 42, 0.98), rgba(30, 41, 59, 0.96));
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.38);
      }}

      .glr-hero::before {{
        content: "";
        position: absolute;
        inset: -40%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        transform: rotate(18deg);
        animation: glrShimmer 7s linear infinite;
      }}

      .glr-hero-inner {{
        position: relative;
        z-index: 1;
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: center;
      }}

      .glr-kicker {{
        color: #7dd3fc;
        font-size: 0.72rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        margin-bottom: 4px;
      }}

      .glr-hero h1 {{
        margin: 0;
        color: #ffffff;
        font-size: clamp(1.6rem, 3vw, 2.35rem);
        line-height: 1.02;
      }}

      .glr-hero p {{
        margin: 7px 0 0;
        color: #cbd5e1;
        max-width: 720px;
      }}

      .glr-hero-pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: flex-end;
      }}

      .glr-hero-pill {{
        border: 1px solid rgba(226, 232, 240, 0.28);
        border-radius: 999px;
        padding: 7px 10px;
        color: #e2e8f0;
        background: rgba(15,23,42,0.48);
        backdrop-filter: blur(14px);
        font-size: 0.72rem;
        font-weight: 800;
        white-space: nowrap;
      }}

      .glr-board {{
        display: flex;
        flex-direction: column;
        gap: {board_gap}px;
        padding: 10px;
        border-radius: 26px;
        background:
          radial-gradient(circle at 8% 8%, rgba(14, 165, 233, 0.32), transparent 24%),
          radial-gradient(circle at 95% 55%, rgba(168, 85, 247, 0.20), transparent 26%),
          radial-gradient(circle at 48% 102%, rgba(34,197,94,0.20), transparent 28%),
          linear-gradient(135deg, #020617, #0f172a 48%, #111827);
        box-shadow:
          0 28px 75px rgba(2, 6, 23, 0.50),
          inset 0 1px 0 rgba(255,255,255,0.08);
      }}

      .glr-side {{
        border: 1px solid rgba(226,232,240,0.26);
        border-radius: 20px;
        padding: 10px;
        box-shadow:
          0 18px 46px rgba(2, 6, 23, 0.22),
          inset 0 1px 0 rgba(255,255,255,0.58);
        animation: glrBoardIn 420ms ease both;
      }}

      .glr-opponent {{
        background:
          radial-gradient(circle at 12% 5%, rgba(59,130,246,0.10), transparent 26%),
          linear-gradient(135deg, #f8fafc, #eef2ff);
      }}

      .glr-player {{
        background:
          radial-gradient(circle at 88% 6%, rgba(20,184,166,0.13), transparent 30%),
          linear-gradient(135deg, #ecfeff, #f8fafc);
      }}

      .glr-player-header {{
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 6px;
      }}

      .glr-player-header h3 {{
        margin: 0;
        color: #020617;
        font-size: 0.98rem;
        letter-spacing: -0.02em;
      }}

      .glr-player-subtitle {{
        color: #64748b;
        font-size: 0.64rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.10em;
      }}

      .glr-stats {{
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 5px;
      }}

      .glr-stats span,
      .glr-chip {{
        border: 1px solid rgba(148, 163, 184, 0.55);
        background: rgba(255,255,255,0.80);
        color: #334155;
        border-radius: 999px;
        padding: 3px 7px;
        font-size: 0.60rem;
        font-weight: 750;
        box-shadow: 0 3px 10px rgba(15,23,42,0.05);
      }}

      .glr-side-grid {{
        display: grid;
        grid-template-columns: minmax(0, 1fr) {side_zone}px;
        gap: 8px;
      }}

      .glr-play-area {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}

      .glr-active-wrap {{
        width: {active_width}px;
        margin: 0 auto;
      }}

      .glr-bench-row {{
        display: grid;
        grid-template-columns: repeat(5, minmax({bench_min}px, 1fr));
        gap: 6px;
      }}

      .glr-pokemon {{
        position: relative;
        min-height: {pokemon_min}px;
        border: 1px solid rgba(148,163,184,0.52);
        border-radius: 16px;
        padding: 5px;
        background:
          linear-gradient(180deg, rgba(255,255,255,0.94), rgba(248,250,252,0.82));
        box-shadow:
          0 10px 28px rgba(15,23,42,0.10),
          inset 0 1px 0 rgba(255,255,255,0.88);
        transition: transform 180ms ease, box-shadow 180ms ease;
        animation: glrCardIn 260ms ease both;
      }}

      .glr-active-wrap .glr-pokemon {{
        box-shadow:
          0 0 0 2px rgba(59,130,246,0.18),
          0 14px 34px rgba(15,23,42,0.16);
        animation: glrActivePulse 2.8s ease-in-out infinite;
      }}

      .glr-pokemon:hover {{
        transform: translateY(-2px) scale(1.012);
        box-shadow: 0 18px 42px rgba(15,23,42,0.17);
      }}

      .glr-empty-slot {{
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94a3b8;
        border-style: dashed;
        background: rgba(255,255,255,0.42);
      }}

      .glr-empty-slot span {{
        font-size: 0.72rem;
      }}

      .glr-card-stack {{
        position: relative;
      }}

      .glr-card {{
        position: relative;
        border: 1px solid rgba(191,219,254,0.90);
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
        text-align: center;
        box-shadow: 0 4px 16px rgba(15,23,42,0.08);
      }}

      .glr-card img {{
        display: block;
        width: 100%;
        max-height: {card_max}px;
        object-fit: contain;
        background: #f8fafc;
      }}

      .glr-card-name {{
        padding: 2px 3px;
        color: #334155;
        font-size: 0.55rem;
        line-height: 1.08;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}

      .glr-card-small {{
        width: 42px;
        display: inline-block;
        vertical-align: top;
        margin: 1px;
      }}

      .glr-card-small img {{
        max-height: 54px;
      }}

      .glr-card-small .glr-card-name {{
        display: none;
      }}

      .glr-no-image,
      .glr-missing-image {{
        min-height: 70px;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
          radial-gradient(circle at 30% 15%, rgba(255,255,255,0.95), transparent 16%),
          linear-gradient(135deg, #dbeafe, #f0f9ff 45%, #fef3c7);
      }}

      .glr-holo-sheen {{
        position: absolute;
        inset: -80%;
        background: linear-gradient(100deg, transparent, rgba(255,255,255,0.40), transparent);
        animation: glrShimmer 4.5s linear infinite;
      }}

      .glr-empty-card {{
        min-height: 82px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94a3b8;
        border-style: dashed;
      }}

      .glr-instance-badge {{
        position: absolute;
        left: -7px;
        top: -7px;
        max-width: 94%;
        border-radius: 999px;
        padding: 2px 6px;
        background:
          linear-gradient(135deg, rgba(15,23,42,0.96), rgba(30,41,59,0.94));
        color: #ffffff;
        border: 2px solid rgba(255,255,255,0.92);
        box-shadow: 0 7px 16px rgba(15,23,42,0.28);
        font-weight: 950;
        font-size: 0.52rem;
        z-index: 6;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }}

      .glr-damage-counter {{
        position: absolute;
        right: -8px;
        top: -8px;
        width: 34px;
        height: 34px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
          radial-gradient(circle at 30% 25%, #fecaca, #ef4444 55%, #991b1b);
        color: #ffffff;
        border: 3px solid #ffffff;
        box-shadow:
          0 9px 18px rgba(15, 23, 42, 0.36),
          inset 0 1px 3px rgba(255, 255, 255, 0.45);
        font-weight: 950;
        font-size: 0.76rem;
        z-index: 7;
        animation: glrDamagePop 560ms ease both, glrDamagePulse 2.2s ease-in-out infinite;
      }}

      .glr-pokemon-meta {{
        display: flex;
        justify-content: space-between;
        margin-top: 3px;
        color: #64748b;
        font-size: 0.57rem;
        font-weight: 750;
      }}

      .glr-mini-row {{
        min-height: 3px;
        margin-top: 2px;
      }}

      .glr-more-chip {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 24px;
        height: 24px;
        border-radius: 999px;
        background: #0f172a;
        color: white;
        font-size: 0.6rem;
        font-weight: 900;
      }}

      .glr-zone-label {{
        margin: 2px 0 4px;
        color: #64748b;
        font-size: 0.56rem;
        font-weight: 950;
        text-transform: uppercase;
        letter-spacing: 0.11em;
      }}

      .glr-main-label {{
        text-align: center;
      }}

      .glr-spaced {{
        margin-top: 10px;
      }}

      .glr-prize-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(58px, 1fr));
        gap: 4px;
      }}

      .glr-prize {{
        min-height: 30px;
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        padding: 4px;
        font-size: 0.54rem;
        background: #f8fafc;
        color: #334155;
        overflow: hidden;
        text-overflow: ellipsis;
      }}

      .glr-prize-taken-known {{
        background: #dcfce7;
        border-color: #86efac;
        color: #166534;
      }}

      .glr-prize-taken-unknown {{
        background: #fef9c3;
        border-color: #fde68a;
        color: #854d0e;
      }}

      .glr-prize-remembered {{
        background: #e0f2fe;
        border-color: #7dd3fc;
      }}

      .glr-prize-unknown {{
        border-style: dashed;
        color: #94a3b8;
      }}

      .glr-chip-row {{
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
        max-height: 64px;
        overflow-y: auto;
        padding-right: 2px;
      }}

      .glr-muted {{
        color: #94a3b8;
      }}

      .glr-middle-row {{
        display: grid;
        grid-template-columns: 0.72fr 1.72fr 0.72fr;
        gap: 8px;
        align-items: stretch;
      }}

      .glr-middle-chip,
      .glr-stadium,
      .glr-action-panel {{
        border: 1px solid rgba(226,232,240,0.30);
        border-radius: 16px;
        padding: 9px;
        color: #e2e8f0;
        background:
          linear-gradient(135deg, rgba(15,23,42,0.76), rgba(30,41,59,0.58));
        box-shadow:
          0 14px 34px rgba(2,6,23,0.20),
          inset 0 1px 0 rgba(255,255,255,0.07);
        backdrop-filter: blur(18px);
      }}

      .glr-middle-chip,
      .glr-stadium {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
      }}

      .glr-middle-chip strong,
      .glr-stadium strong {{
        color: #ffffff;
      }}

      .glr-action-panel {{
        position: relative;
        overflow: hidden;
        min-height: 84px;
        text-align: center;
      }}

      .glr-action-panel::before {{
        content: "";
        position: absolute;
        inset: 0;
        background: radial-gradient(circle at 50% 0%, rgba(125,211,252,0.18), transparent 50%);
        opacity: 0.9;
        pointer-events: none;
      }}

      .glr-action-panel > * {{
        position: relative;
        z-index: 1;
      }}

      .glr-action-impact {{
        border-color: rgba(248,113,113,0.55);
        box-shadow:
          0 0 0 1px rgba(248,113,113,0.16),
          0 18px 42px rgba(127,29,29,0.22);
      }}

      .glr-action-play {{
        border-color: rgba(96,165,250,0.52);
      }}

      .glr-action-prize {{
        border-color: rgba(250,204,21,0.62);
      }}

      .glr-action-title {{
        color: #ffffff;
        font-weight: 950;
        font-size: 0.92rem;
        margin-bottom: 2px;
        letter-spacing: -0.01em;
      }}

      .glr-action-actor {{
        color: #bae6fd;
        font-weight: 850;
        font-size: 0.68rem;
        margin-bottom: 3px;
      }}

      .glr-action-cards {{
        display: flex;
        justify-content: center;
        align-items: flex-start;
        gap: 3px;
        min-height: 18px;
        margin: 2px 0;
      }}

      .glr-action-text {{
        color: #cbd5e1;
        font-size: 0.63rem;
        line-height: 1.18;
        max-height: 36px;
        overflow-y: auto;
      }}

      .glr-toolbar-card {{
        border: 1px solid rgba(148,163,184,0.24);
        border-radius: 18px;
        padding: 10px;
        margin: 10px 0 12px;
        background:
          linear-gradient(135deg, rgba(15,23,42,0.92), rgba(30,41,59,0.84));
        box-shadow: 0 18px 44px rgba(2,6,23,0.24);
      }}

      @keyframes glrShimmer {{
        0% {{ transform: translateX(-40%) rotate(18deg); }}
        100% {{ transform: translateX(40%) rotate(18deg); }}
      }}

      @keyframes glrBoardIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
      }}

      @keyframes glrCardIn {{
        from {{ opacity: 0; transform: translateY(6px) scale(0.985); }}
        to {{ opacity: 1; transform: translateY(0) scale(1); }}
      }}

      @keyframes glrActivePulse {{
        0%, 100% {{ box-shadow: 0 0 0 2px rgba(59,130,246,0.12), 0 14px 34px rgba(15,23,42,0.14); }}
        50% {{ box-shadow: 0 0 0 3px rgba(59,130,246,0.24), 0 18px 42px rgba(15,23,42,0.18); }}
      }}

      @keyframes glrDamagePop {{
        0% {{ transform: scale(0.65); opacity: 0; }}
        70% {{ transform: scale(1.12); opacity: 1; }}
        100% {{ transform: scale(1); }}
      }}

      @keyframes glrDamagePulse {{
        0%, 100% {{ filter: brightness(1); }}
        50% {{ filter: brightness(1.16); }}
      }}

      @media (max-width: 1050px) {{
        .glr-hero-inner {{
          flex-direction: column;
          align-items: flex-start;
        }}

        .glr-hero-pills {{
          justify-content: flex-start;
        }}

        .glr-side-grid {{
          grid-template-columns: 1fr;
        }}

        .glr-bench-row {{
          grid-template-columns: repeat(2, minmax(96px, 1fr));
        }}

        .glr-middle-row {{
          grid-template-columns: 1fr;
        }}

        .glr-active-wrap {{
          width: min({active_width}px, 100%);
        }}
      }}

      /* Compact right rail: prizes stay readable, hand/discard scroll cleanly */
      .glr-side-zones {{
        display: grid;
        grid-template-rows: auto minmax(62px, 0.8fr) minmax(70px, 1fr);
        gap: 6px;
        min-width: 0;
        max-height: 100%;
        overflow: hidden;
      }}

      .glr-zone-card {{
        min-width: 0;
        border: 1px solid rgba(148, 163, 184, 0.28);
        border-radius: 14px;
        padding: 6px;
        background: rgba(255, 255, 255, 0.48);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.62);
        overflow: hidden;
      }}

      .glr-zone-title-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 6px;
        margin-bottom: 3px;
      }}

      .glr-zone-title-row .glr-zone-label {{
        margin: 0;
      }}

      .glr-zone-title-row span {{
        border-radius: 999px;
        padding: 1px 6px;
        color: #475569;
        background: rgba(248,250,252,0.90);
        border: 1px solid rgba(148,163,184,0.36);
        font-size: 0.56rem;
        font-weight: 900;
      }}

      .glr-hand-zone .glr-chip-row,
      .glr-discard-zone .glr-chip-row {{
        max-height: 74px;
        overflow-y: auto;
        display: flex;
        flex-wrap: wrap;
        align-content: flex-start;
        gap: 3px;
        padding-right: 3px;
      }}

      .glr-hand-zone .glr-chip-row::-webkit-scrollbar,
      .glr-discard-zone .glr-chip-row::-webkit-scrollbar {{
        width: 5px;
      }}

      .glr-hand-zone .glr-chip-row::-webkit-scrollbar-thumb,
      .glr-discard-zone .glr-chip-row::-webkit-scrollbar-thumb {{
        background: rgba(100, 116, 139, 0.42);
        border-radius: 999px;
      }}

      .glr-hand-zone .glr-chip,
      .glr-discard-zone .glr-chip {{
        max-width: 100%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }}

      .glr-prize-zone {{
        overflow: visible;
      }}

      .glr-prize-zone .glr-prize-grid {{
        grid-template-columns: repeat(2, minmax(54px, 1fr));
      }}


      /* DARK TYPE-THEMED CINEMATIC BOARD V1 */
      .glr-board {{
        background:
          radial-gradient(circle at 8% 4%, rgba(59, 130, 246, 0.20), transparent 24%),
          radial-gradient(circle at 92% 46%, rgba(168, 85, 247, 0.18), transparent 28%),
          radial-gradient(circle at 45% 104%, rgba(20, 184, 166, 0.14), transparent 30%),
          linear-gradient(135deg, #020617 0%, #07111f 44%, #0f172a 100%) !important;
        border: 1px solid rgba(148, 163, 184, 0.20);
        box-shadow:
          0 30px 90px rgba(0, 0, 0, 0.54),
          inset 0 1px 0 rgba(255, 255, 255, 0.06);
      }}

      .glr-side {{
        --type-accent: #60a5fa;
        --type-accent-2: #22d3ee;
        --type-soft: rgba(96, 165, 250, 0.17);
        --type-glow: rgba(96, 165, 250, 0.38);

        position: relative;
        overflow: hidden;
        color: #e5eefc !important;
        background:
          radial-gradient(circle at 18% 12%, var(--type-soft), transparent 30%),
          radial-gradient(circle at 85% 22%, rgba(255, 255, 255, 0.055), transparent 24%),
          linear-gradient(145deg, rgba(4, 9, 20, 0.96), rgba(11, 18, 32, 0.97) 54%, rgba(3, 7, 18, 0.98)) !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.055),
          0 18px 55px rgba(0, 0, 0, 0.34),
          0 0 34px rgba(0, 0, 0, 0.22) !important;
        backdrop-filter: blur(14px);
        animation:
          glrTypeBoardIn 420ms ease-out both,
          glrTypeBoardFloat 8s ease-in-out infinite alternate;
      }}

      .glr-side::before {{
        content: "";
        position: absolute;
        inset: -35%;
        background:
          linear-gradient(
            115deg,
            transparent 0%,
            rgba(255, 255, 255, 0.035) 35%,
            var(--type-soft) 48%,
            rgba(255, 255, 255, 0.035) 58%,
            transparent 72%
          );
        transform: translateX(-68%) rotate(8deg);
        animation: glrTypeSweep 8s ease-in-out infinite;
        pointer-events: none;
        z-index: 0;
      }}

      .glr-side::after {{
        content: "";
        position: absolute;
        inset: 0;
        background:
          radial-gradient(circle at 50% 52%, rgba(255,255,255,0.025), transparent 55%);
        pointer-events: none;
        z-index: 0;
      }}

      .glr-side > * {{
        position: relative;
        z-index: 1;
      }}

      .glr-side .glr-player-header h3,
      .glr-side .glr-zone-label,
      .glr-side .glr-main-label {{
        color: #e5eefc !important;
      }}

      .glr-side .glr-player-subtitle,
      .glr-side .glr-muted {{
        color: #94a3b8 !important;
      }}

      .glr-side .glr-stats span,
      .glr-side .glr-chip,
      .glr-side .glr-zone-title-row span {{
        background: rgba(15, 23, 42, 0.70) !important;
        color: #e5eefc !important;
        border: 1px solid rgba(148, 163, 184, 0.22) !important;
        box-shadow:
          inset 0 1px 0 rgba(255,255,255,0.05),
          0 4px 12px rgba(0,0,0,0.18);
      }}

      .glr-side .glr-zone-card,
      .glr-side .glr-hand-zone,
      .glr-side .glr-discard-zone,
      .glr-side .glr-prize-zone {{
        background:
          linear-gradient(180deg, rgba(15, 23, 42, 0.58), rgba(2, 6, 23, 0.34)) !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        box-shadow:
          inset 0 1px 0 rgba(255, 255, 255, 0.045),
          0 8px 22px rgba(0, 0, 0, 0.20);
      }}

      .glr-side .glr-prize {{
        background: rgba(15, 23, 42, 0.56) !important;
        color: #cbd5e1 !important;
        border-color: rgba(148, 163, 184, 0.20) !important;
      }}

      .glr-side .glr-prize-taken-known {{
        background: rgba(34, 197, 94, 0.20) !important;
        border-color: rgba(74, 222, 128, 0.35) !important;
        color: #bbf7d0 !important;
      }}

      .glr-side .glr-prize-taken-unknown {{
        background: rgba(250, 204, 21, 0.18) !important;
        border-color: rgba(250, 204, 21, 0.35) !important;
        color: #fef08a !important;
      }}

      .glr-side .glr-prize-remembered {{
        background: rgba(56, 189, 248, 0.18) !important;
        border-color: rgba(125, 211, 252, 0.35) !important;
        color: #bae6fd !important;
      }}

      .glr-side .glr-pokemon {{
        background:
          linear-gradient(180deg, rgba(15, 23, 42, 0.72), rgba(2, 6, 23, 0.56)) !important;
        border: 1px solid rgba(148, 163, 184, 0.20) !important;
        box-shadow:
          inset 0 1px 0 rgba(255,255,255,0.055),
          0 10px 28px rgba(0,0,0,0.24);
        transition:
          transform 180ms ease,
          box-shadow 180ms ease,
          border-color 180ms ease,
          filter 180ms ease;
      }}

      .glr-side .glr-pokemon:hover {{
        transform: translateY(-3px) scale(1.018);
        border-color: color-mix(in srgb, var(--type-accent), white 12%) !important;
        box-shadow:
          inset 0 1px 0 rgba(255,255,255,0.07),
          0 18px 45px rgba(0,0,0,0.32),
          0 0 26px var(--type-glow);
        filter: saturate(1.06);
      }}

      .glr-side .glr-empty-slot {{
        color: rgba(203, 213, 225, 0.48) !important;
        background:
          linear-gradient(180deg, rgba(15,23,42,0.34), rgba(2,6,23,0.22)) !important;
        border: 1px dashed rgba(148,163,184,0.26) !important;
      }}

      .glr-side .glr-card {{
        background: rgba(248,250,252,0.97);
        box-shadow:
          0 8px 22px rgba(0,0,0,0.30),
          0 0 0 1px rgba(255,255,255,0.04);
        transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
      }}

      .glr-side .glr-card img {{
        filter: saturate(1.04) contrast(1.02);
      }}

      .glr-side .glr-card:hover {{
        transform: translateY(-2px) rotate(-0.4deg) scale(1.022);
        box-shadow:
          0 14px 32px rgba(0,0,0,0.38),
          0 0 22px var(--type-glow);
        filter: brightness(1.035);
      }}

      .glr-active-wrap .glr-pokemon {{
        border-color: color-mix(in srgb, var(--type-accent), white 18%) !important;
        box-shadow:
          inset 0 1px 0 rgba(255,255,255,0.075),
          0 0 0 1px rgba(255,255,255,0.04),
          0 0 34px var(--type-glow),
          0 18px 52px rgba(0,0,0,0.42) !important;
        animation: glrTypeActivePulse 2.3s ease-in-out infinite;
      }}

      .glr-active-wrap::after {{
        content: "";
        position: absolute;
        left: 50%;
        bottom: 0;
        width: 72%;
        height: 24px;
        transform: translateX(-50%);
        background: radial-gradient(circle, var(--type-glow), transparent 72%);
        filter: blur(14px);
        opacity: 0.85;
        pointer-events: none;
      }}

      .glr-instance-badge {{
        background:
          linear-gradient(135deg, rgba(2,6,23,0.95), rgba(15,23,42,0.90)) !important;
        border-color: rgba(255,255,255,0.82) !important;
      }}

      .glr-damage-counter {{
        background:
          radial-gradient(circle at 30% 24%, #fecaca, #ef4444 54%, #7f1d1d) !important;
        box-shadow:
          0 10px 20px rgba(127, 29, 29, 0.40),
          0 0 18px rgba(239, 68, 68, 0.38),
          inset 0 1px 3px rgba(255,255,255,0.45) !important;
        animation:
          glrTypeDamagePop 520ms ease-out both,
          glrTypeDamagePulse 1.65s ease-in-out infinite;
      }}

      .glr-middle-chip,
      .glr-stadium,
      .glr-action-panel {{
        background:
          radial-gradient(circle at 50% 0%, rgba(96,165,250,0.16), transparent 42%),
          linear-gradient(135deg, rgba(2,6,23,0.88), rgba(15,23,42,0.72)) !important;
        border-color: rgba(148,163,184,0.22) !important;
      }}

      .glr-action-panel {{
        animation: glrTypeActionGlow 2.8s ease-in-out infinite;
      }}

      .glr-action-impact {{
        border-color: rgba(248,113,113,0.56) !important;
        box-shadow:
          0 0 0 1px rgba(248,113,113,0.14),
          0 18px 46px rgba(127,29,29,0.26),
          inset 0 1px 0 rgba(255,255,255,0.06) !important;
      }}

      .glr-action-play {{
        border-color: rgba(96,165,250,0.54) !important;
      }}

      .glr-action-prize {{
        border-color: rgba(250,204,21,0.58) !important;
      }}

      .glr-theme-neutral {{
        --type-accent: #94a3b8;
        --type-accent-2: #64748b;
        --type-soft: rgba(148, 163, 184, 0.16);
        --type-glow: rgba(148, 163, 184, 0.28);
      }}

      .glr-theme-grass {{
        --type-accent: #22c55e;
        --type-accent-2: #84cc16;
        --type-soft: rgba(34, 197, 94, 0.20);
        --type-glow: rgba(74, 222, 128, 0.36);
      }}

      .glr-theme-fire {{
        --type-accent: #fb923c;
        --type-accent-2: #ef4444;
        --type-soft: rgba(249, 115, 22, 0.22);
        --type-glow: rgba(251, 146, 60, 0.40);
      }}

      .glr-theme-water {{
        --type-accent: #38bdf8;
        --type-accent-2: #2563eb;
        --type-soft: rgba(56, 189, 248, 0.21);
        --type-glow: rgba(96, 165, 250, 0.38);
      }}

      .glr-theme-lightning {{
        --type-accent: #facc15;
        --type-accent-2: #f59e0b;
        --type-soft: rgba(250, 204, 21, 0.20);
        --type-glow: rgba(250, 204, 21, 0.32);
      }}

      .glr-theme-psychic {{
        --type-accent: #c084fc;
        --type-accent-2: #ec4899;
        --type-soft: rgba(168, 85, 247, 0.22);
        --type-glow: rgba(217, 70, 239, 0.34);
      }}

      .glr-theme-fighting {{
        --type-accent: #f97316;
        --type-accent-2: #b45309;
        --type-soft: rgba(180, 83, 9, 0.21);
        --type-glow: rgba(217, 119, 6, 0.30);
      }}

      .glr-theme-darkness {{
        --type-accent: #64748b;
        --type-accent-2: #1e293b;
        --type-soft: rgba(71, 85, 105, 0.24);
        --type-glow: rgba(100, 116, 139, 0.30);
      }}

      .glr-theme-metal {{
        --type-accent: #cbd5e1;
        --type-accent-2: #64748b;
        --type-soft: rgba(148, 163, 184, 0.20);
        --type-glow: rgba(226, 232, 240, 0.28);
      }}

      .glr-theme-dragon {{
        --type-accent: #a78bfa;
        --type-accent-2: #f97316;
        --type-soft: rgba(124, 58, 237, 0.23);
        --type-glow: rgba(168, 85, 247, 0.35);
      }}

      .glr-theme-colorless {{
        --type-accent: #e2e8f0;
        --type-accent-2: #94a3b8;
        --type-soft: rgba(203, 213, 225, 0.17);
        --type-glow: rgba(226, 232, 240, 0.24);
      }}

      .glr-theme-fairy {{
        --type-accent: #f9a8d4;
        --type-accent-2: #f472b6;
        --type-soft: rgba(249, 168, 212, 0.19);
        --type-glow: rgba(244, 114, 182, 0.30);
      }}

      @keyframes glrTypeBoardIn {{
        from {{
          opacity: 0;
          transform: translateY(10px) scale(0.992);
        }}
        to {{
          opacity: 1;
          transform: translateY(0) scale(1);
        }}
      }}

      @keyframes glrTypeBoardFloat {{
        from {{ transform: translateY(0); }}
        to {{ transform: translateY(-2px); }}
      }}

      @keyframes glrTypeSweep {{
        0%, 16% {{
          transform: translateX(-68%) rotate(8deg);
          opacity: 0;
        }}
        28% {{
          opacity: 0.75;
        }}
        48% {{
          transform: translateX(66%) rotate(8deg);
          opacity: 0;
        }}
        100% {{
          transform: translateX(66%) rotate(8deg);
          opacity: 0;
        }}
      }}

      @keyframes glrTypeActivePulse {{
        0%, 100% {{
          filter: saturate(1);
          box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.075),
            0 0 0 1px rgba(255,255,255,0.04),
            0 0 24px var(--type-glow),
            0 18px 52px rgba(0,0,0,0.40);
        }}
        50% {{
          filter: saturate(1.14) brightness(1.04);
          box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.085),
            0 0 0 1px rgba(255,255,255,0.06),
            0 0 44px var(--type-glow),
            0 22px 62px rgba(0,0,0,0.48);
        }}
      }}

      @keyframes glrTypeDamagePop {{
        0% {{
          transform: scale(0.62);
          opacity: 0.4;
        }}
        68% {{
          transform: scale(1.18);
          opacity: 1;
        }}
        100% {{
          transform: scale(1);
        }}
      }}

      @keyframes glrTypeDamagePulse {{
        0%, 100% {{ filter: brightness(1); }}
        50% {{ filter: brightness(1.20); }}
      }}

      @keyframes glrTypeActionGlow {{
        0%, 100% {{
          box-shadow:
            0 14px 34px rgba(2,6,23,0.20),
            inset 0 1px 0 rgba(255,255,255,0.07);
        }}
        50% {{
          box-shadow:
            0 18px 44px rgba(2,6,23,0.28),
            0 0 24px rgba(96,165,250,0.14),
            inset 0 1px 0 rgba(255,255,255,0.08);
        }}
      }}

      @media (prefers-reduced-motion: reduce) {{
        .glr-side,
        .glr-side::before,
        .glr-active-wrap .glr-pokemon,
        .glr-damage-counter,
        .glr-action-panel {{
          animation: none !important;
        }}
      }}


      
      /* DARK TYPE-THEMED BOARD OVERRIDES V2 */
      .glr-board {{
        background:
          radial-gradient(circle at 14% 10%, rgba(59, 130, 246, 0.16), transparent 26%),
          radial-gradient(circle at 84% 18%, rgba(168, 85, 247, 0.14), transparent 28%),
          radial-gradient(circle at 50% 100%, rgba(255, 255, 255, 0.04), transparent 30%),
          linear-gradient(135deg, #020617 0%, #040916 34%, #08101d 100%) !important;
      }}

      .glr-side {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(255,255,255,0.06), transparent 34%),
          linear-gradient(180deg, rgba(10,14,28,0.98), rgba(4,8,18,1));
        --type-panel:
          linear-gradient(180deg, rgba(8, 12, 24, 0.96), rgba(4, 7, 16, 0.96));
        --type-card-shell:
          linear-gradient(180deg, rgba(10, 14, 26, 0.98), rgba(3, 6, 14, 0.98));
        --type-slot:
          linear-gradient(180deg, rgba(12,16,28,0.80), rgba(5,8,16,0.78));
        --type-outline: rgba(148,163,184,0.22);
        --type-highlight: rgba(255,255,255,0.06);

        background:
          var(--type-scene),
          linear-gradient(145deg, rgba(4, 9, 20, 0.995), rgba(8, 15, 30, 0.995) 55%, rgba(3, 7, 18, 0.995)) !important;
      }}

      .glr-side .glr-pokemon,
      .glr-side .glr-hand-zone,
      .glr-side .glr-discard-zone,
      .glr-side .glr-prize-zone,
      .glr-side .glr-zone-card {{
        background: var(--type-panel) !important;
        border: 1px solid var(--type-outline) !important;
        box-shadow:
          inset 0 1px 0 var(--type-highlight),
          0 8px 24px rgba(0,0,0,0.26) !important;
      }}

      .glr-side .glr-empty-slot {{
        background: var(--type-slot) !important;
        border: 1px dashed rgba(148,163,184,0.24) !important;
        color: rgba(203,213,225,0.65) !important;
      }}

      .glr-side .glr-card {{
        background: var(--type-card-shell) !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
        box-shadow:
          0 10px 26px rgba(0,0,0,0.34),
          inset 0 1px 0 rgba(255,255,255,0.04) !important;
      }}

      .glr-side .glr-card img {{
        background: transparent !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.30) !important;
        border-radius: 10px;
      }}

      .glr-side .glr-pokemon {{
        backdrop-filter: blur(2px);
      }}

      .glr-active-wrap .glr-pokemon {{
        box-shadow:
          0 0 0 1px rgba(255,255,255,0.06),
          0 0 22px rgba(255,255,255,0.10),
          0 18px 40px rgba(0,0,0,0.34) !important;
      }}

      .glr-action-panel {{
        background:
          linear-gradient(180deg, rgba(10,14,24,0.96), rgba(5,8,18,0.96)) !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
        box-shadow:
          inset 0 1px 0 rgba(255,255,255,0.04),
          0 10px 30px rgba(0,0,0,0.22) !important;
      }}

      .glr-side .glr-pokemon,
      .glr-side .glr-card,
      .glr-active-wrap .glr-pokemon {{
        animation: none !important;
      }}

      /* -------------------------------------------------- */
      /* TYPE THEMES                                        */
      /* -------------------------------------------------- */

      .glr-theme-grass {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(34,197,94,0.34), transparent 38%),
          radial-gradient(circle at 18% 12%, rgba(74,222,128,0.18), transparent 20%),
          repeating-linear-gradient(
            103deg,
            rgba(34,197,94,0.16) 0px,
            rgba(34,197,94,0.16) 3px,
            transparent 3px,
            transparent 18px
          ),
          linear-gradient(180deg, rgba(7,32,18,0.995), rgba(3,12,7,1));
        --type-panel:
          linear-gradient(180deg, rgba(7,28,16,0.96), rgba(4,12,8,0.96));
      }}

      .glr-theme-fire {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(249,115,22,0.34), transparent 40%),
          radial-gradient(circle at 80% 14%, rgba(239,68,68,0.18), transparent 22%),
          repeating-linear-gradient(
            125deg,
            rgba(251,146,60,0.14) 0px,
            rgba(251,146,60,0.14) 10px,
            transparent 10px,
            transparent 28px
          ),
          linear-gradient(180deg, rgba(36,10,6,0.995), rgba(14,4,5,1));
        --type-panel:
          linear-gradient(180deg, rgba(32,10,8,0.96), rgba(14,5,5,0.96));
      }}

      .glr-theme-water {{
        --type-scene:
          radial-gradient(circle at 52% 100%, rgba(56,189,248,0.34), transparent 38%),
          radial-gradient(circle at 14% 14%, rgba(147,197,253,0.16), transparent 24%),
          repeating-linear-gradient(
            0deg,
            rgba(56,189,248,0.10) 0px,
            rgba(56,189,248,0.10) 4px,
            transparent 4px,
            transparent 22px
          ),
          linear-gradient(180deg, rgba(4,18,38,0.995), rgba(1,8,24,1));
        --type-panel:
          linear-gradient(180deg, rgba(6,20,44,0.96), rgba(3,10,24,0.96));
      }}

      .glr-theme-lightning {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(250,204,21,0.30), transparent 34%),
          radial-gradient(circle at 84% 18%, rgba(253,224,71,0.18), transparent 18%),
          repeating-linear-gradient(
            122deg,
            transparent 0px,
            transparent 18px,
            rgba(250,204,21,0.18) 18px,
            rgba(250,204,21,0.18) 22px,
            transparent 22px,
            transparent 44px
          ),
          linear-gradient(180deg, rgba(28,22,4,0.995), rgba(10,8,2,1));
        --type-panel:
          linear-gradient(180deg, rgba(35,28,8,0.96), rgba(16,12,4,0.96));
      }}

      .glr-theme-psychic {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(217,70,239,0.28), transparent 34%),
          radial-gradient(circle at 50% 50%, rgba(192,132,252,0.12), transparent 22%),
          radial-gradient(circle at 50% 50%, transparent 0%, transparent 24%, rgba(192,132,252,0.10) 25%, transparent 28%),
          radial-gradient(circle at 74% 18%, rgba(236,72,153,0.14), transparent 18%),
          linear-gradient(180deg, rgba(24,6,34,0.995), rgba(10,3,20,1));
        --type-panel:
          linear-gradient(180deg, rgba(26,8,40,0.96), rgba(12,5,24,0.96));
      }}

      .glr-theme-fighting {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(234,88,12,0.26), transparent 34%),
          repeating-linear-gradient(
            138deg,
            rgba(180,83,9,0.12) 0px,
            rgba(180,83,9,0.12) 12px,
            transparent 12px,
            transparent 30px
          ),
          linear-gradient(180deg, rgba(32,16,6,0.995), rgba(12,6,2,1));
        --type-panel:
          linear-gradient(180deg, rgba(34,18,8,0.96), rgba(15,8,4,0.96));
      }}

      .glr-theme-darkness {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(71,85,105,0.22), transparent 32%),
          radial-gradient(circle at 78% 18%, rgba(30,41,59,0.24), transparent 24%),
          linear-gradient(180deg, rgba(6,8,14,0.998), rgba(1,3,8,1));
        --type-panel:
          linear-gradient(180deg, rgba(12,14,20,0.96), rgba(5,7,12,0.96));
      }}

      .glr-theme-metal {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(203,213,225,0.18), transparent 32%),
          repeating-linear-gradient(
            115deg,
            rgba(148,163,184,0.10) 0px,
            rgba(148,163,184,0.10) 6px,
            transparent 6px,
            transparent 18px
          ),
          linear-gradient(180deg, rgba(14,18,26,0.997), rgba(5,8,14,1));
        --type-panel:
          linear-gradient(180deg, rgba(22,26,34,0.96), rgba(10,12,18,0.96));
      }}

      .glr-theme-dragon {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(245,158,11,0.32), transparent 38%),
          radial-gradient(circle at 84% 18%, rgba(251,191,36,0.18), transparent 20%),
          repeating-linear-gradient(
            135deg,
            rgba(245,158,11,0.14) 0px,
            rgba(245,158,11,0.14) 12px,
            transparent 12px,
            transparent 30px
          ),
          linear-gradient(180deg, rgba(34,22,4,0.995), rgba(14,9,2,1));
        --type-panel:
          linear-gradient(180deg, rgba(42,28,6,0.96), rgba(18,12,4,0.96));
      }}

      .glr-theme-colorless {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(226,232,240,0.14), transparent 30%),
          linear-gradient(180deg, rgba(20,22,28,0.995), rgba(7,9,14,1));
        --type-panel:
          linear-gradient(180deg, rgba(25,28,34,0.96), rgba(10,12,18,0.96));
      }}

      .glr-theme-fairy {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(244,114,182,0.26), transparent 34%),
          radial-gradient(circle at 20% 16%, rgba(249,168,212,0.16), transparent 18%),
          linear-gradient(180deg, rgba(36,8,28,0.995), rgba(14,4,12,1));
        --type-panel:
          linear-gradient(180deg, rgba(40,10,28,0.96), rgba(16,6,14,0.96));
      }}

      .glr-theme-neutral {{
        --type-scene:
          radial-gradient(circle at 50% 100%, rgba(148,163,184,0.14), transparent 32%),
          linear-gradient(180deg, rgba(15,18,28,0.995), rgba(5,7,14,1));
        --type-panel:
          linear-gradient(180deg, rgba(18,22,32,0.96), rgba(8,10,16,0.96));
      }}
      /* END DARK TYPE-THEMED BOARD OVERRIDES V2 */


      {basic_bg_css}

      /* INLINE TYPE BACKGROUND IMAGE SAFETY OVERRIDE */
      .glr-board .glr-side::before {{
        display: none !important;
        opacity: 0 !important;
        background: none !important;
      }}

      .glr-board .glr-side::after {{
        background:
          radial-gradient(circle at 50% 72%, rgba(255,255,255,0.07), transparent 38%),
          linear-gradient(180deg, rgba(2,6,23,0.02), rgba(2,6,23,0.30)) !important;
      }}


      /* REAL BACKGROUND IMAGE LAYER V1 */
      .glr-board .glr-side {{
        position: relative !important;
        overflow: hidden !important;
        background: #020617 !important;
      }}

      .glr-board .glr-side-bg-img {{
        position: absolute !important;
        inset: 0 !important;
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        object-position: center center !important;
        z-index: 0 !important;
        opacity: 1 !important;
        filter: saturate(1.04) contrast(1.02) brightness(0.92) !important;
        pointer-events: none !important;
      }}

      .glr-board .glr-side::before {{
        display: none !important;
        opacity: 0 !important;
        background: none !important;
      }}

      .glr-board .glr-side::after {{
        content: "" !important;
        position: absolute !important;
        inset: 0 !important;
        z-index: 1 !important;
        background:
          radial-gradient(circle at 50% 72%, rgba(255,255,255,0.06), transparent 38%),
          linear-gradient(180deg, rgba(2,6,23,0.06), rgba(2,6,23,0.36)) !important;
        pointer-events: none !important;
      }}

      .glr-board .glr-side > *:not(.glr-side-bg-img) {{
        position: relative !important;
        z-index: 2 !important;
      }}

      .glr-board .glr-side .glr-pokemon,
      .glr-board .glr-side .glr-hand-zone,
      .glr-board .glr-side .glr-discard-zone,
      .glr-board .glr-side .glr-prize-zone,
      .glr-board .glr-side .glr-zone-card {{
        background: rgba(3, 7, 18, 0.72) !important;
        backdrop-filter: blur(5px);
      }}

      .glr-board .glr-side .glr-empty-slot {{
        background: rgba(3, 7, 18, 0.46) !important;
      }}

      .glr-board .glr-side .glr-card {{
        background: rgba(3, 7, 18, 0.58) !important;
      }}

      .glr-board .glr-side .glr-card img {{
        background: transparent !important;
        opacity: 1 !important;
        filter: none !important;
      }}

    </style>
    """


def _render_hero() -> None:
    _render_raw_html(
        """
        <div class="glr-page-shell">
          <div class="glr-hero">
            <div class="glr-hero-inner">
              <div>
                <div class="glr-kicker">Pokémon TCG Live Replay Studio</div>
                <h1>Game Review</h1>
                <p>Paste a battle log and replay the match as a polished, board-level timeline with visible actions, card images, damage counters, prizes, and uncertainty warnings.</p>
              </div>
              <div class="glr-hero-pills">
                <span class="glr-hero-pill">Board replay</span>
                <span class="glr-hero-pill">Autoplay</span>
                <span class="glr-hero-pill">Damage tracking</span>
                <span class="glr-hero-pill">Ambiguity-aware</span>
              </div>
            </div>
          </div>
        </div>
        """
    )


def _render_board_visual(state: GameState, *, density: str) -> None:

    player_names = [name for name in state.player_order if name in state.players]

    if not player_names:
        st.info("No players parsed yet.")
        return

    bottom = state.players[player_names[0]]
    top = state.players[player_names[1]] if len(player_names) > 1 else None

    active_types = {_active_pokemon_type(bottom)}
    if top is not None:
        active_types.add(_active_pokemon_type(top))

    _render_raw_html(_board_css(density, active_types=tuple(sorted(active_types))))

    active_types = {_active_pokemon_type(bottom)}
    if top is not None:
        active_types.add(_active_pokemon_type(top))

    stadium = _card_label(state.stadium) if state.stadium else "No Stadium"
    turn_player = state.turn_player or "Setup / unknown"
    action_panel = _event_focus_html(state.last_event)

    parts = ["<div class='glr-board'>"]

    if top is not None:
        parts.append(_side_html(top, opponent=True))

    parts.append(
        f"""
        <div class="glr-middle-row">
          <div class="glr-middle-chip">
            <div class="glr-zone-label">Turn</div>
            <strong>{html.escape(turn_player)}</strong>
          </div>
          {action_panel}
          <div class="glr-stadium">
            <div class="glr-zone-label">Stadium</div>
            <strong>{html.escape(stadium)}</strong>
          </div>
        </div>
        """
    )

    parts.append(_side_html(bottom, opponent=False))
    parts.append("</div>")

    _render_raw_html("".join(parts))


def _parse_known_prizes(text: str) -> dict[str, list[CardRef]]:
    out: dict[str, list[CardRef]] = {}
    current_player = ""

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.endswith(":") and not line.startswith("("):
            current_player = line[:-1].strip()
            out.setdefault(current_player, [])
            continue

        cards = parse_card_refs(line)
        if current_player and cards:
            out.setdefault(current_player, []).extend(cards)

    return out


def _events_df(events) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Step": event.index + 1,
                "Line": event.line_no,
                "Type": event.event_type,
                "Actor": event.actor,
                "Cards": ", ".join(card.display_name for card in event.cards),
                "Confidence": event.metadata.get("target_confidence", ""),
                "Raw": event.raw,
            }
            for event in events
        ]
    )


def _resolver_df(events) -> pd.DataFrame:
    rows = []

    for event in events:
        for card in event.cards:
            if card.exported_id:
                rows.append(
                    {
                        "Exported ID": card.exported_id,
                        "API-style ID": exported_id_to_api_card_id(card.exported_id),
                        "Name": card.name,
                        "Image URL": image_url_for_card_ref(card),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["Exported ID", "API-style ID", "Name", "Image URL"])

    return pd.DataFrame(rows).drop_duplicates()


def _replay_signature(raw_log: str, known_prizes_text: str) -> str:
    raw_log = raw_log or ""
    known_prizes_text = known_prizes_text or ""
    return f"{len(raw_log)}:{raw_log[:80]}:{raw_log[-80:]}:{len(known_prizes_text)}:{known_prizes_text[:80]}"


def _sync_replay_step(signature: str, frame_count: int) -> int:
    sig_key = "game_log_replay_signature"
    step_key = "game_log_replay_step"

    if st.session_state.get(sig_key) != signature:
        st.session_state[sig_key] = signature
        st.session_state[step_key] = 0
        st.session_state["game_log_replay_autoplay"] = False

    current = int(st.session_state.get(step_key, 0) or 0)
    current = max(0, min(current, max(0, frame_count - 1)))
    st.session_state[step_key] = current
    return current


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def _speed_to_delay_seconds(speed_label: str) -> float:
    speed_map = {
        "0.5x": 2.0,
        "1x": 1.0,
        "1.5x": 1.0 / 1.5,
        "2x": 0.5,
        "3x": 1.0 / 3.0,
    }
    return speed_map.get(speed_label, 1.0)


def _render_replay_controls(frame_count: int) -> int:
    step_key = "game_log_replay_step"
    autoplay_key = "game_log_replay_autoplay"
    speed_key = "game_log_replay_speed"

    current = int(st.session_state.get(step_key, 0) or 0)
    max_step = max(0, frame_count - 1)
    current = max(0, min(current, max_step))
    st.session_state[step_key] = current

    st.session_state.setdefault(speed_key, "1x")
    st.session_state.setdefault(autoplay_key, False)

    _render_raw_html("<div class='glr-toolbar-card'>")

    c1, c2, c3, c4, c5, c6, c7 = st.columns([0.8, 0.8, 0.9, 2.8, 0.8, 0.8, 0.9])

    with c1:
        if st.button("⏮ Start", use_container_width=True, disabled=current <= 0):
            st.session_state[autoplay_key] = False
            st.session_state[step_key] = 0
            _rerun()

    with c2:
        if st.button("◀ Back", use_container_width=True, disabled=current <= 0):
            st.session_state[autoplay_key] = False
            st.session_state[step_key] = max(0, current - 1)
            _rerun()

    with c3:
        is_playing = bool(st.session_state.get(autoplay_key, False))
        label = "⏸ Pause" if is_playing else "▶ Play"
        disabled = current >= max_step and not is_playing

        if st.button(label, use_container_width=True, disabled=disabled):
            if current >= max_step:
                st.session_state[step_key] = 0
                st.session_state[autoplay_key] = True
            else:
                st.session_state[autoplay_key] = not is_playing
            _rerun()

    with c4:
        new_step = st.slider(
            "Replay action",
            min_value=0,
            max_value=max_step,
            value=current,
            step=1,
            key="game_log_replay_slider",
        )
        if new_step != current:
            st.session_state[autoplay_key] = False
            st.session_state[step_key] = int(new_step)
            _rerun()

    with c5:
        if st.button("Next ▶", use_container_width=True, disabled=current >= max_step):
            st.session_state[autoplay_key] = False
            st.session_state[step_key] = min(max_step, current + 1)
            _rerun()

    with c6:
        if st.button("End ⏭", use_container_width=True, disabled=current >= max_step):
            st.session_state[autoplay_key] = False
            st.session_state[step_key] = max_step
            _rerun()

    with c7:
        st.selectbox(
            "Speed",
            options=["0.5x", "1x", "1.5x", "2x", "3x"],
            index=["0.5x", "1x", "1.5x", "2x", "3x"].index(st.session_state.get(speed_key, "1x")),
            key=speed_key,
            label_visibility="collapsed",
        )

    progress_pct = 0.0 if max_step == 0 else current / max_step

    try:
        st.progress(
            progress_pct,
            text=f"Action {current} / {max_step} · Speed {st.session_state.get(speed_key, '1x')}",
        )
    except TypeError:
        st.progress(progress_pct)
        st.caption(f"Action {current} / {max_step} · Speed {st.session_state.get(speed_key, '1x')}")

    _render_raw_html("</div>")

    if current >= max_step and st.session_state.get(autoplay_key):
        st.session_state[autoplay_key] = False

    return int(st.session_state.get(step_key, 0) or 0)


def _maybe_autoplay_advance(frame_count: int) -> None:
    step_key = "game_log_replay_step"
    autoplay_key = "game_log_replay_autoplay"
    speed_key = "game_log_replay_speed"

    if not st.session_state.get(autoplay_key, False):
        return

    max_step = max(0, frame_count - 1)
    current = int(st.session_state.get(step_key, 0) or 0)

    if current >= max_step:
        st.session_state[autoplay_key] = False
        return

    delay = _speed_to_delay_seconds(str(st.session_state.get(speed_key, "1x")))
    time.sleep(delay)

    st.session_state[step_key] = min(max_step, current + 1)
    _rerun()


def _load_demo_log_if_requested() -> None:
    fixture = Path("tests/fixtures/game_logs/bananahammer33_dragapult_test_log.txt")
    if not fixture.exists():
        return

    if st.button("Load demo replay", use_container_width=True):
        st.session_state["game_log_replay_text"] = fixture.read_text(encoding="utf-8")
        st.session_state["game_log_replay_autoplay"] = False
        st.session_state["game_log_replay_step"] = 0
        _rerun()


def render_game_log_replay_tab() -> None:
    _render_hero()

    with st.expander("Battle log export setting", expanded=False):
        st.markdown(
            """
            In Pokémon TCG Live, use:

            `Settings → Battle Log Settings → disable Hide card IDs from export`

            This gives card references like `(sv5_123) Raging Bolt ex`, which lets the replay resolve images.
            """
        )

    existing_log = str(st.session_state.get("game_log_replay_text", "") or "")
    input_expanded = not existing_log.strip()

    with st.expander("Battle log input", expanded=input_expanded):
        _load_demo_log_if_requested()

        raw_log = st.text_area(
            "Battle log",
            height=180,
            key="game_log_replay_text",
            placeholder="Paste exported battle log here...",
        )

        known_prizes_text = st.text_area(
            "Optional remembered prizes",
            height=80,
            key="game_log_known_prizes_text",
            placeholder="FrunkUke:\n(sv5_145) Ciphermaniac's Codebreaking\n(me1_131) Ultra Ball",
            help="Log-revealed prizes override remembered prizes if they disagree.",
        )

    if not raw_log.strip():
        st.info("Paste a battle log or load the demo replay to begin.")
        return

    events = parse_battle_log(raw_log)
    known_prizes = _parse_known_prizes(known_prizes_text)
    frames = build_replay_frames(events, known_prizes_by_player=known_prizes)

    if not frames:
        st.warning("No replay frames were built.")
        return

    signature = _replay_signature(raw_log, known_prizes_text)
    _sync_replay_step(signature, len(frames))

    ambiguity_count = sum(1 for event in events if event.metadata.get("ambiguous_target"))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Actions", len(events))
    m2.metric("Frames", len(frames))
    m3.metric("Players", len(frames[-1].state.players))
    m4.metric("Ambiguous targets", ambiguity_count)

    step = _render_replay_controls(len(frames))
    frame = frames[step]
    event = frame.event

    top_controls = st.columns([1.5, 1.2, 1.2])
    with top_controls[0]:
        view_mode = st.radio(
            "View",
            options=["Board visual", "Parsed events", "Card resolver debug"],
            horizontal=True,
            index=0,
            key="game_log_replay_view_mode",
        )

    with top_controls[1]:
        density = st.radio(
            "Board density",
            options=["Compact", "Cinematic"],
            horizontal=True,
            index=0,
            key="game_log_replay_density",
        )

    with top_controls[2]:
        if frame.state.winner:
            st.success(f"Winner: {frame.state.winner}")
        elif event is not None:
            st.caption(f"Line {event.line_no} · `{event.event_type}`")

    if event is not None and event.metadata.get("ambiguous_target"):
        st.warning(
            "Ambiguous target: the log did not identify the exact card copy. "
            f"Applied heuristic `{event.metadata.get('target_heuristic', 'default')}`."
        )
        with st.expander("Target ambiguity details", expanded=False):
            st.json(
                {
                    "reason": event.metadata.get("target_reason", ""),
                    "chosen": event.metadata.get("chosen_target", ""),
                    "candidates": event.metadata.get("candidate_targets", []),
                }
            )
    elif event is not None and event.metadata.get("target_confidence") == "missing":
        st.warning(event.metadata.get("target_reason", "Target could not be resolved."))

    if view_mode == "Board visual":
        _render_board_visual(frame.state, density=density)
    elif view_mode == "Parsed events":
        st.dataframe(_events_df(events), use_container_width=True, hide_index=True)
    else:
        st.dataframe(_resolver_df(events), use_container_width=True, hide_index=True)

    with st.expander("Current raw log line", expanded=False):
        if event is None:
            st.code("Step 0: empty board before applying the log.")
        else:
            st.code(event.raw)

    _maybe_autoplay_advance(len(frames))
