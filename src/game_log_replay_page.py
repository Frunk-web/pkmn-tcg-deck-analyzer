from __future__ import annotations

import html
import textwrap
from collections import Counter

import pandas as pd
import streamlit as st

from src.game_log.models import CardRef, GameState, PlayerState, PokemonInPlay
from src.game_log.parser import parse_battle_log, parse_card_refs
from src.game_log.reducer import build_replay_frames
from src.game_log.resolver import exported_id_to_api_card_id, image_url_for_card_ref, candidate_image_urls_for_card_ref


def _card_label(card: CardRef | None) -> str:
    if card is None:
        return "Empty"
    if card.unknown:
        return "Unknown card"
    if card.exported_id and card.name:
        return f"{card.name} [{card.exported_id}]"
    return card.name or card.exported_id or "Unknown card"



def _render_raw_html(raw_html: str) -> None:
    """
    Render HTML safely through Streamlit without Markdown treating indented
    triple-quoted HTML as a code block.
    """
    cleaned = textwrap.dedent(str(raw_html or "")).strip()
    cleaned = "\n".join(line.lstrip() for line in cleaned.splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)


def _card_html(card: CardRef | None, *, small: bool = False) -> str:
    if card is None:
        return "<div class='glr-card glr-empty-card'>Empty</div>"

    label = html.escape(_card_label(card))
    img = image_url_for_card_ref(card)
    cls = "glr-card glr-card-small" if small else "glr-card"

    if img:
        return f"""
        <div class="{cls}">
          <img src="{html.escape(img)}" alt="{label}">
          <div class="glr-card-name">{label}</div>
        </div>
        """

    return f"""
    <div class="{cls} glr-no-image">
      <div class="glr-card-name">{label}</div>
    </div>
    """


def _pokemon_html(pokemon: PokemonInPlay | None) -> str:
    if pokemon is None:
        return """
        <div class="glr-pokemon glr-empty-slot">
          Empty
        </div>
        """

    attached = "".join(_card_html(card, small=True) for card in pokemon.attached)
    stack = "".join(_card_html(card, small=True) for card in pokemon.evolution_stack)
    damage_badge = (
        f'<div class="glr-damage-counter">{int(pokemon.damage)}</div>'
        if int(pokemon.damage or 0) > 0
        else ""
    )

    return f"""
    <div class="glr-pokemon">
      <div class="glr-card-stack">
        {_card_html(pokemon.card)}
        {damage_badge}
      </div>
      <div class="glr-pokemon-meta">
        <span>{int(pokemon.damage)} damage</span>
        <span>{len(pokemon.attached)} attached</span>
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
                label = "Taken: unknown"
            else:
                cls = "glr-prize glr-prize-taken-known"
                label = "Taken: " + _card_label(prize)
        elif i < len(player.user_known_prizes):
            cls = "glr-prize glr-prize-remembered"
            label = "Remembered: " + _card_label(player.user_known_prizes[i])
        else:
            cls = "glr-prize glr-prize-unknown"
            label = "Unknown prize"

        cells.append(f"<div class='{cls}'>{html.escape(label)}</div>")

    return "<div class='glr-prize-grid'>" + "".join(cells) + "</div>"


def _card_counter_html(cards: list[CardRef], *, limit: int = 10) -> str:
    if not cards:
        return "<span class='glr-muted'>None known</span>"

    counts = Counter(_card_label(card) for card in cards)
    chips = []

    for label, count in counts.most_common(limit):
        text = f"{count}x {label}" if count > 1 else label
        chips.append(f"<span class='glr-chip'>{html.escape(text)}</span>")

    if len(counts) > limit:
        chips.append(f"<span class='glr-chip glr-muted'>+{len(counts) - limit} more</span>")

    return "<div class='glr-chip-row'>" + "".join(chips) + "</div>"


def _side_html(player: PlayerState, *, opponent: bool) -> str:
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
    <section class="glr-side {'glr-opponent' if opponent else 'glr-player'}">
      <div class="glr-player-header">
        <h3>{html.escape(player.name)}</h3>
        <div class="glr-stats">
          <span>Prizes left: {player.remaining_prize_count}</span>
          <span>Bench: {len(player.bench)}/5</span>
          <span>Known hand: {len(player.hand_known)}</span>
          <span>Unknown hand: {player.hand_unknown_count}</span>
          <span>Discard: {len(player.discard)}</span>
        </div>
      </div>

      <div class="glr-side-grid">
        <main class="glr-play-area">
          {play_area}
        </main>

        <aside class="glr-side-zones">
          <div class="glr-zone-label">Prize Cards</div>
          {_prizes_html(player)}

          <div class="glr-zone-label glr-spaced">Known Hand</div>
          {_card_counter_html(player.hand_known)}

          <div class="glr-zone-label glr-spaced">Discard</div>
          {_card_counter_html(player.discard)}
        </aside>
      </div>
    </section>
    """


def _board_css() -> str:
    return """
    <style>
      .glr-board {
        display: flex;
        flex-direction: column;
        gap: 14px;
        padding: 14px;
        border-radius: 24px;
        background:
          radial-gradient(circle at top left, rgba(59,130,246,0.25), transparent 26%),
          radial-gradient(circle at bottom right, rgba(34,197,94,0.20), transparent 28%),
          linear-gradient(135deg, #0f172a, #1e293b);
      }

      .glr-side {
        border: 1px solid rgba(226,232,240,0.45);
        border-radius: 20px;
        padding: 14px;
        box-shadow: 0 14px 35px rgba(15,23,42,0.22);
      }

      .glr-opponent {
        background: linear-gradient(135deg, #f8fafc, #eef2ff);
      }

      .glr-player {
        background: linear-gradient(135deg, #ecfeff, #f8fafc);
      }

      .glr-player-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 12px;
        margin-bottom: 10px;
      }

      .glr-player-header h3 {
        margin: 0;
        color: #0f172a;
        font-size: 1.1rem;
      }

      .glr-stats {
        display: flex;
        flex-wrap: wrap;
        justify-content: flex-end;
        gap: 6px;
      }

      .glr-stats span,
      .glr-chip {
        border: 1px solid #cbd5e1;
        background: rgba(255,255,255,0.92);
        color: #334155;
        border-radius: 999px;
        padding: 3px 8px;
        font-size: 0.72rem;
      }

      .glr-side-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 260px;
        gap: 12px;
      }

      .glr-play-area {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .glr-active-wrap {
        width: 190px;
        margin: 0 auto;
      }

      .glr-bench-row {
        display: grid;
        grid-template-columns: repeat(5, minmax(90px, 1fr));
        gap: 8px;
      }

      .glr-pokemon {
        min-height: 148px;
        border: 1px solid #cbd5e1;
        border-radius: 16px;
        padding: 7px;
        background: rgba(255,255,255,0.88);
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.55);
      }

      .glr-empty-slot {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94a3b8;
        border-style: dashed;
      }

      .glr-card {
        border: 1px solid #dbeafe;
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
        text-align: center;
      }

      .glr-card img {
        display: block;
        width: 100%;
        max-height: 154px;
        object-fit: contain;
        background: #f8fafc;
      }

      .glr-card-name {
        padding: 4px;
        color: #334155;
        font-size: 0.66rem;
        line-height: 1.2;
      }

      .glr-card-small {
        width: 48px;
        display: inline-block;
        vertical-align: top;
        margin: 2px;
      }

      .glr-card-small img {
        max-height: 64px;
      }

      .glr-card-small .glr-card-name {
        display: none;
      }

      .glr-no-image,
      .glr-missing-image {
        min-height: 75px;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
          linear-gradient(135deg, #f8fafc, #e2e8f0);
      }

      .glr-empty-card {
        min-height: 90px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #94a3b8;
        border-style: dashed;
      }

      .glr-card-stack {
        position: relative;
      }

      .glr-damage-counter {
        position: absolute;
        right: -10px;
        top: -10px;
        width: 42px;
        height: 42px;
        border-radius: 999px;
        display: flex;
        align-items: center;
        justify-content: center;
        background:
          radial-gradient(circle at 30% 25%, #fecaca, #ef4444 55%, #991b1b);
        color: #ffffff;
        border: 3px solid #ffffff;
        box-shadow:
          0 8px 18px rgba(15, 23, 42, 0.35),
          inset 0 1px 3px rgba(255, 255, 255, 0.45);
        font-weight: 950;
        font-size: 0.9rem;
        z-index: 5;
      }

      .glr-card-small + .glr-damage-counter {
        width: 28px;
        height: 28px;
        font-size: 0.65rem;
      }

      .glr-pokemon-meta {
        display: flex;
        justify-content: space-between;
        margin-top: 4px;
        color: #64748b;
        font-size: 0.68rem;
      }

      .glr-mini-row {
        min-height: 4px;
        margin-top: 3px;
      }

      .glr-zone-label {
        margin: 4px 0 6px;
        color: #64748b;
        font-size: 0.68rem;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 0.10em;
      }

      .glr-main-label {
        text-align: center;
      }

      .glr-spaced {
        margin-top: 12px;
      }

      .glr-middle-row {
        display: grid;
        grid-template-columns: 1fr 240px 1fr;
        gap: 12px;
        align-items: center;
      }

      .glr-middle-chip,
      .glr-stadium {
        border: 1px solid rgba(226,232,240,0.45);
        border-radius: 16px;
        padding: 12px;
        text-align: center;
        color: #e2e8f0;
        background: rgba(15,23,42,0.52);
      }

      .glr-stadium strong {
        color: #ffffff;
      }

      .glr-prize-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(70px, 1fr));
        gap: 6px;
      }

      .glr-prize {
        min-height: 42px;
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        padding: 7px;
        font-size: 0.67rem;
        background: #f8fafc;
        color: #334155;
      }

      .glr-prize-taken-known {
        background: #dcfce7;
        border-color: #86efac;
      }

      .glr-prize-taken-unknown {
        background: #fef9c3;
        border-color: #fde68a;
      }

      .glr-prize-remembered {
        background: #e0f2fe;
        border-color: #7dd3fc;
      }

      .glr-prize-unknown {
        border-style: dashed;
        color: #94a3b8;
      }

      .glr-side-zones {
        min-width: 0;
      }

      .glr-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
      }

      .glr-muted {
        color: #94a3b8;
      }

      @media (max-width: 1050px) {
        .glr-side-grid {
          grid-template-columns: 1fr;
        }

        .glr-bench-row {
          grid-template-columns: repeat(2, minmax(100px, 1fr));
        }

        .glr-middle-row {
          grid-template-columns: 1fr;
        }

        .glr-active-wrap {
          width: min(190px, 100%);
        }
      }
    </style>
    """


def _render_board_visual(state: GameState) -> None:
    _render_raw_html(_board_css())

    player_names = [name for name in state.player_order if name in state.players]

    if not player_names:
        st.info("No players parsed yet.")
        return

    bottom = state.players[player_names[0]]
    top = state.players[player_names[1]] if len(player_names) > 1 else None

    stadium = _card_label(state.stadium) if state.stadium else "No Stadium"
    turn_player = state.turn_player or "Setup / unknown"

    parts = ["<div class='glr-board'>"]

    if top is not None:
        parts.append(_side_html(top, opponent=True))

    parts.append(
        f"""
        <div class="glr-middle-row">
          <div class="glr-middle-chip">Turn: <strong>{html.escape(turn_player)}</strong></div>
          <div class="glr-stadium">
            <div class="glr-zone-label">Stadium</div>
            <strong>{html.escape(stadium)}</strong>
          </div>
          <div class="glr-middle-chip">Visible board state</div>
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

    current = int(st.session_state.get(step_key, 0) or 0)
    current = max(0, min(current, max(0, frame_count - 1)))
    st.session_state[step_key] = current
    return current


def _render_replay_controls(frame_count: int) -> int:
    step_key = "game_log_replay_step"
    current = int(st.session_state.get(step_key, 0) or 0)
    max_step = max(0, frame_count - 1)

    c1, c2, c3, c4, c5 = st.columns([0.85, 0.85, 3.0, 0.85, 0.85])

    with c1:
        if st.button("⏮ Start", use_container_width=True, disabled=current <= 0):
            st.session_state[step_key] = 0
            st.rerun()

    with c2:
        if st.button("◀ Back", use_container_width=True, disabled=current <= 0):
            st.session_state[step_key] = max(0, current - 1)
            st.rerun()

    with c3:
        new_step = st.slider(
            "Replay action",
            min_value=0,
            max_value=max_step,
            value=current,
            step=1,
            key="game_log_replay_slider",
        )
        if new_step != current:
            st.session_state[step_key] = int(new_step)
            st.rerun()

    with c4:
        if st.button("Next ▶", use_container_width=True, disabled=current >= max_step):
            st.session_state[step_key] = min(max_step, current + 1)
            st.rerun()

    with c5:
        if st.button("End ⏭", use_container_width=True, disabled=current >= max_step):
            st.session_state[step_key] = max_step
            st.rerun()

    return int(st.session_state.get(step_key, 0) or 0)


def render_game_log_replay_tab() -> None:
    st.markdown("## Game Review")
    st.caption(
        "Paste a Pokémon TCG Live battle log with card IDs enabled. "
        "The replay reconstructs visible board state. Hidden cards remain hidden."
    )

    with st.expander("Battle log export setting", expanded=False):
        st.markdown(
            """
            In Pokémon TCG Live:

            `Settings → Battle Log Settings → disable Hide card IDs from export`

            This gives card references like `(sv5_123) Raging Bolt ex`, which lets the replay resolve images.
            """
        )

    raw_log = st.text_area(
        "Battle log",
        height=245,
        key="game_log_replay_text",
        placeholder="Paste exported battle log here...",
    )

    known_prizes_text = st.text_area(
        "Optional remembered prizes",
        height=100,
        key="game_log_known_prizes_text",
        placeholder="FrunkUke:\n(sv5_145) Ciphermaniac's Codebreaking\n(me1_131) Ultra Ball",
        help="Log-revealed prizes override remembered prizes if they disagree.",
    )

    if not raw_log.strip():
        st.info("Paste a battle log to begin.")
        return

    events = parse_battle_log(raw_log)
    known_prizes = _parse_known_prizes(known_prizes_text)
    frames = build_replay_frames(events, known_prizes_by_player=known_prizes)

    if not frames:
        st.warning("No replay frames were built.")
        return

    signature = _replay_signature(raw_log, known_prizes_text)
    _sync_replay_step(signature, len(frames))

    m1, m2, m3 = st.columns(3)
    m1.metric("Parsed actions", len(events))
    m2.metric("Replay frames", len(frames))
    m3.metric("Players", len(frames[-1].state.players))

    step = _render_replay_controls(len(frames))
    frame = frames[step]
    event = frame.event

    if event is None:
        st.info("Step 0: empty board before applying the log.")
    else:
        st.markdown(f"**Action {step} of {len(frames) - 1} · line {event.line_no} · `{event.event_type}`**")
        st.code(event.raw)

    if frame.state.winner:
        st.success(f"Winner: {frame.state.winner}")

    view_mode = st.radio(
        "View",
        options=["Board visual", "Parsed events", "Card resolver debug"],
        horizontal=True,
        index=0,
        key="game_log_replay_view_mode",
    )

    if view_mode == "Board visual":
        _render_board_visual(frame.state)
    elif view_mode == "Parsed events":
        st.dataframe(_events_df(events), use_container_width=True, hide_index=True)
    else:
        st.dataframe(_resolver_df(events), use_container_width=True, hide_index=True)
