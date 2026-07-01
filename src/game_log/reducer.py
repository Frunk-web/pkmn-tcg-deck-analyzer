from __future__ import annotations

import copy
from typing import Any

from .models import CardRef, GameState, LogEvent, PlayerState, PokemonInPlay, ReplayFrame


Candidate = tuple[PlayerState, str, int | None, PokemonInPlay]


def _card_key(card: CardRef) -> str:
    return card.exported_id or card.name or card.display_name


def _same_card(a: CardRef, b: CardRef) -> bool:
    if a.exported_id and b.exported_id:
        return a.exported_id == b.exported_id
    return a.name == b.name


def _remove_one(cards: list[CardRef], target: CardRef) -> CardRef | None:
    for i, card in enumerate(cards):
        if _same_card(card, target):
            return cards.pop(i)
    return None


def _create_pokemon(player: PlayerState, card: CardRef, event_index: int, *, inferred: bool = False) -> PokemonInPlay:
    key = _card_key(card)
    player.pokemon_instance_counters[key] = player.pokemon_instance_counters.get(key, 0) + 1
    copy_number = player.pokemon_instance_counters[key]

    safe_player = "".join(ch for ch in player.name if ch.isalnum()).lower() or "player"
    safe_key = "".join(ch for ch in key if ch.isalnum()).lower() or "pokemon"

    return PokemonInPlay(
        card=card,
        instance_id=f"{safe_player}:{safe_key}:{copy_number}",
        copy_number=copy_number,
        created_event_index=event_index,
        inferred=inferred,
    )


def _candidate_label(candidate: Candidate) -> str:
    player, zone, idx, pokemon = candidate
    slot = "Active" if zone == "active" else f"Bench {int(idx) + 1 if idx is not None else '?'}"
    return (
        f"{player.name} {slot}: {pokemon.copy_label} "
        f"(attached={len(pokemon.attached)}, damage={int(pokemon.damage or 0)})"
    )


def _iter_candidates_for_player(player: PlayerState, target: CardRef, zone_hint: str = "") -> list[Candidate]:
    candidates: list[Candidate] = []

    if zone_hint in {"", "active"}:
        if player.active and _same_card(player.active.card, target):
            candidates.append((player, "active", None, player.active))

    if zone_hint in {"", "bench"}:
        for i, pokemon in enumerate(player.bench):
            if _same_card(pokemon.card, target):
                candidates.append((player, "bench", i, pokemon))

    return candidates


def _target_candidates(
    state: GameState,
    target: CardRef,
    *,
    default_player: PlayerState | None = None,
    owner_hint: str = "",
    zone_hint: str = "",
) -> list[Candidate]:
    if owner_hint and owner_hint in state.players:
        players = [state.players[owner_hint]]
    elif default_player is not None:
        players = [default_player]
    else:
        players = list(state.players.values())

    out: list[Candidate] = []
    for player in players:
        out.extend(_iter_candidates_for_player(player, target, zone_hint=zone_hint))

    return out


def _choose_candidate(candidates: list[Candidate], prefer: str) -> Candidate:
    def sort_key(candidate: Candidate) -> tuple:
        _, zone, idx, pokemon = candidate
        idx_value = -1 if idx is None else int(idx)
        return (
            0 if zone == "active" else 1,
            idx_value,
            pokemon.created_event_index,
        )

    if prefer == "active_first":
        return sorted(candidates, key=sort_key)[0]

    if prefer == "bench_oldest":
        bench = [c for c in candidates if c[1] == "bench"]
        return sorted(bench or candidates, key=lambda c: (c[3].created_event_index, c[2] or 0))[0]

    if prefer == "fewest_attached":
        return sorted(
            candidates,
            key=lambda c: (len(c[3].attached), c[3].created_event_index, c[2] or 0),
        )[0]

    if prefer == "most_attached":
        return sorted(
            candidates,
            key=lambda c: (-len(c[3].attached), c[3].created_event_index, c[2] or 0),
        )[0]

    if prefer == "highest_damage":
        return sorted(
            candidates,
            key=lambda c: (-int(c[3].damage or 0), c[3].created_event_index, c[2] or 0),
        )[0]

    return sorted(candidates, key=lambda c: (c[3].created_event_index, c[2] or 0))[0]


def _record_ambiguity(
    state: GameState,
    event: LogEvent,
    *,
    target: CardRef,
    candidates: list[Candidate],
    chosen: Candidate | None,
    reason: str,
    heuristic: str,
) -> None:
    info: dict[str, Any] = {
        "event_index": event.index,
        "line_no": event.line_no,
        "event_type": event.event_type,
        "reason": reason,
        "heuristic": heuristic,
        "target": target.display_name,
        "raw": event.raw,
        "candidates": [_candidate_label(c) for c in candidates],
        "chosen": _candidate_label(chosen) if chosen is not None else "",
    }

    event.metadata["ambiguous_target"] = True
    event.metadata["target_confidence"] = "inferred"
    event.metadata["target_reason"] = reason
    event.metadata["target_heuristic"] = heuristic
    event.metadata["candidate_targets"] = info["candidates"]
    event.metadata["chosen_target"] = info["chosen"]

    state.ambiguities.append(info)


def _resolve_target(
    state: GameState,
    event: LogEvent,
    target: CardRef,
    *,
    default_player: PlayerState | None = None,
    owner_hint: str = "",
    zone_hint: str = "",
    reason: str,
    prefer: str = "oldest",
) -> Candidate | None:
    candidates = _target_candidates(
        state,
        target,
        default_player=default_player,
        owner_hint=owner_hint,
        zone_hint=zone_hint,
    )

    if not candidates:
        event.metadata["target_confidence"] = "missing"
        event.metadata["target_reason"] = f"No matching Pokémon in play for {target.display_name}"
        return None

    if len(candidates) == 1:
        event.metadata["target_confidence"] = "exact"
        event.metadata["chosen_target"] = _candidate_label(candidates[0])
        return candidates[0]

    chosen = _choose_candidate(candidates, prefer=prefer)
    _record_ambiguity(
        state,
        event,
        target=target,
        candidates=candidates,
        chosen=chosen,
        reason=reason,
        heuristic=prefer,
    )
    return chosen


def _move_known_to_hand(player: PlayerState, cards: list[CardRef]) -> None:
    for card in cards:
        player.hand_known.append(card)


def _move_revealed_draws_to_hand(player: PlayerState, cards: list[CardRef]) -> None:
    if cards:
        player.hand_unknown_count = max(0, player.hand_unknown_count - len(cards))
    _move_known_to_hand(player, cards)


def _move_known_to_discard(player: PlayerState, cards: list[CardRef]) -> None:
    for card in cards:
        _remove_one(player.hand_known, card)
        player.discard.append(card)


def _play_pokemon(player: PlayerState, card: CardRef, zone: str, event: LogEvent, *, inferred: bool = False) -> PokemonInPlay:
    played_card = _remove_one(player.hand_known, card) or card
    pokemon = _create_pokemon(player, played_card, event.index, inferred=inferred)

    if zone == "active":
        if player.active is not None:
            player.bench.append(player.active)
        player.active = pokemon
    else:
        player.bench.append(pokemon)

    return pokemon


def _attach_energy(state: GameState, player: PlayerState, energy: CardRef, target: CardRef, event: LogEvent) -> None:
    _remove_one(player.hand_known, energy)

    owner_hint = event.metadata.get("target_owner") or player.name
    zone_hint = event.metadata.get("target_zone", "")

    resolved = _resolve_target(
        state,
        event,
        target,
        default_player=player,
        owner_hint=owner_hint,
        zone_hint=zone_hint,
        reason="Energy attachment target has multiple matching Pokémon.",
        prefer="fewest_attached",
    )

    if resolved is None:
        return

    _, _, _, pokemon = resolved
    pokemon.attached.append(energy)


def _discard_pokemon(player: PlayerState, pokemon: PokemonInPlay) -> None:
    player.discard.append(pokemon.card)
    player.discard.extend(pokemon.evolution_stack)
    player.discard.extend(pokemon.attached)


def _remove_pokemon_from_zone(player: PlayerState, zone: str, idx: int | None) -> None:
    if zone == "active":
        player.active = None
    elif zone == "bench" and idx is not None and 0 <= idx < len(player.bench):
        player.bench.pop(idx)


def _knockout(state: GameState, player: PlayerState, target: CardRef, event: LogEvent) -> None:
    owner_hint = event.metadata.get("target_owner") or player.name
    zone_hint = event.metadata.get("target_zone", "")

    resolved = _resolve_target(
        state,
        event,
        target,
        default_player=player,
        owner_hint=owner_hint,
        zone_hint=zone_hint,
        reason="Knock Out line matched multiple Pokémon copies.",
        prefer="highest_damage",
    )

    if resolved is None:
        return

    owner, zone, idx, pokemon = resolved
    _discard_pokemon(owner, pokemon)
    _remove_pokemon_from_zone(owner, zone, idx)


def _promote(state: GameState, player: PlayerState, target: CardRef, event: LogEvent) -> None:
    # The target is currently on the Bench, even though the line says it is now Active.
    resolved = _resolve_target(
        state,
        event,
        target,
        default_player=player,
        owner_hint=event.metadata.get("target_owner") or player.name,
        zone_hint="bench",
        reason="Promotion line matched multiple benched Pokémon copies.",
        prefer="most_attached",
    )

    if resolved is None:
        return

    owner, zone, idx, pokemon = resolved

    if owner.active is not None:
        owner.bench.append(owner.active)

    owner.active = pokemon

    if zone == "bench" and idx is not None and 0 <= idx < len(owner.bench):
        owner.bench.pop(idx)


def _evolve(state: GameState, player: PlayerState, base: CardRef, evolution: CardRef, event: LogEvent) -> None:
    resolved = _resolve_target(
        state,
        event,
        base,
        default_player=player,
        owner_hint=event.metadata.get("target_owner") or player.name,
        zone_hint=event.metadata.get("target_zone", ""),
        reason="Evolution line matched multiple possible base Pokémon.",
        prefer="oldest",
    )

    if resolved is None:
        # If the base was hidden from our reconstruction, create the evolved Pokémon.
        _play_pokemon(player, evolution, "bench", event, inferred=True)
        return

    _, _, _, pokemon = resolved
    _remove_one(player.hand_known, evolution)
    pokemon.evolution_stack.append(pokemon.card)
    pokemon.card = evolution


def _add_damage_to_target(
    state: GameState,
    event: LogEvent,
    target: CardRef,
    *,
    default_player: PlayerState | None = None,
    prefer: str = "active_first",
) -> None:
    resolved = _resolve_target(
        state,
        event,
        target,
        default_player=default_player,
        owner_hint=event.metadata.get("target_owner", ""),
        zone_hint=event.metadata.get("target_zone", ""),
        reason="Damage target matched multiple Pokémon copies.",
        prefer=prefer,
    )

    if resolved is None:
        return

    _, _, _, pokemon = resolved
    if event.amount:
        pokemon.damage += int(event.amount)


def _discard_attached_from_play(
    state: GameState,
    event: LogEvent,
    owner: PlayerState | None,
    cards: list[CardRef],
) -> None:
    if len(cards) < 2:
        if owner is not None:
            for card in cards:
                owner.discard.append(card)
        return

    discarded_card = cards[0]
    source_pokemon_card = cards[-1]

    resolved = _resolve_target(
        state,
        event,
        source_pokemon_card,
        default_player=owner,
        owner_hint=event.metadata.get("target_owner", ""),
        zone_hint="",
        reason="Discard-from-play source matched multiple Pokémon copies.",
        prefer="most_attached",
    )

    if resolved is not None:
        target_owner, _, _, pokemon = resolved
        _remove_one(pokemon.attached, discarded_card)
        target_owner.discard.append(discarded_card)
    elif owner is not None:
        owner.discard.append(discarded_card)


def _apply_known_prize_inputs(state: GameState, known_prizes_by_player: dict[str, list[CardRef]] | None) -> None:
    if not known_prizes_by_player:
        return

    for player_name, prizes in known_prizes_by_player.items():
        player = state.ensure_player(player_name)
        player.user_known_prizes = list(prizes)


def _apply_event(state: GameState, event: LogEvent) -> None:
    state.last_event = event

    actor = state.ensure_player(event.actor) if event.actor else None

    if event.turn_player:
        state.ensure_player(event.turn_player)

    if event.event_type == "turn_start":
        state.turn_player = event.actor
        return

    if event.event_type in {"coin_choice", "coin_winner", "turn_order_choice", "opening_hand"}:
        if actor is not None:
            state.ensure_player(actor.name)
        return

    if event.event_type == "draw_hidden" and actor is not None:
        actor.hand_unknown_count += event.amount or 1
        return

    if event.event_type == "draw_revealed" and actor is not None:
        _move_revealed_draws_to_hand(actor, event.cards)
        return

    if event.event_type == "draw_count" and actor is not None:
        if not event.cards and event.amount:
            actor.hand_unknown_count += int(event.amount)
        return

    if event.event_type == "draw_and_play_to_bench" and actor is not None:
        for card in event.cards:
            _play_pokemon(actor, card, "bench", event)
        return

    if event.event_type == "revealed_cards" and actor is not None:
        parent = event.metadata.get("parent_event_type", "")

        if parent == "draw_and_play_to_bench":
            for card in event.cards:
                _play_pokemon(actor, card, "bench", event)
        elif parent in {"opening_hand", "draw_count", "draw_revealed"}:
            _move_revealed_draws_to_hand(actor, event.cards)
        elif parent == "discard_count":
            _move_known_to_discard(actor, event.cards)
        elif parent == "discard_from_play":
            for card in event.cards:
                actor.discard.append(card)
        elif parent == "effect_detail" and "played" in event.raw and "Bench" in event.raw:
            for card in event.cards:
                _play_pokemon(actor, card, "bench", event)
        elif parent == "effect_detail" and "drew" in event.raw:
            _move_revealed_draws_to_hand(actor, event.cards)
        return

    if event.event_type == "effect_detail" and actor is not None:
        if event.cards and "played" in event.raw and "Bench" in event.raw:
            for card in event.cards:
                _play_pokemon(actor, card, "bench", event)
        elif event.cards and "drew" in event.raw:
            _move_revealed_draws_to_hand(actor, event.cards)
        return

    if event.event_type == "play_to_active" and actor is not None and event.cards:
        _play_pokemon(actor, event.cards[0], "active", event)
        return

    if event.event_type == "play_to_bench" and actor is not None and event.cards:
        _play_pokemon(actor, event.cards[0], "bench", event)
        return

    if event.event_type == "play_stadium" and actor is not None and event.cards:
        _remove_one(actor.hand_known, event.cards[0])
        state.stadium = event.cards[0]
        return

    if event.event_type == "play_card" and actor is not None and event.cards:
        _remove_one(actor.hand_known, event.cards[0])
        actor.discard.append(event.cards[0])
        return

    if event.event_type == "attach_energy" and actor is not None and len(event.cards) >= 2:
        _attach_energy(state, actor, event.cards[0], event.cards[1], event)
        return

    if event.event_type == "discard_from_play":
        _discard_attached_from_play(state, event, actor, event.cards)
        return

    if event.event_type == "evolve" and actor is not None and len(event.cards) >= 2:
        _evolve(state, actor, event.cards[0], event.cards[1], event)
        return

    if event.event_type == "attack" and len(event.cards) >= 2:
        _add_damage_to_target(state, event, event.cards[-1], default_player=None, prefer="active_first")
        return

    if event.event_type == "place_damage_counters" and event.cards:
        _add_damage_to_target(state, event, event.cards[-1], default_player=None, prefer="oldest")
        return

    if event.event_type == "knockout" and actor is not None and event.cards:
        _knockout(state, actor, event.cards[0], event)
        return

    if event.event_type == "promote_active" and actor is not None and event.cards:
        _promote(state, actor, event.cards[0], event)
        return

    if event.event_type == "retreat" and actor is not None and event.cards:
        resolved = _resolve_target(
            state,
            event,
            event.cards[0],
            default_player=actor,
            owner_hint=actor.name,
            zone_hint="active",
            reason="Retreat line matched multiple Pokémon copies.",
            prefer="active_first",
        )
        if resolved is not None:
            owner, zone, idx, pokemon = resolved
            if zone == "active":
                owner.bench.append(pokemon)
                owner.active = None
        return

    if event.event_type == "move_card" and actor is not None and event.cards:
        # Effects like Night Stretcher usually move a card from discard/deck to hand.
        # Do not remove from board unless a future parser proves the source zone.
        for card in event.cards:
            actor.hand_known.append(card)
        return

    if event.event_type == "take_prize" and actor is not None:
        for _ in range(event.amount or 1):
            actor.prizes_taken.append(CardRef(name="Unknown prize", unknown=True))
        return

    if event.event_type == "add_to_hand_revealed" and actor is not None and event.cards:
        card = event.cards[0]

        for i in range(len(actor.prizes_taken) - 1, -1, -1):
            if actor.prizes_taken[i].unknown:
                actor.prizes_taken[i] = card
                break

        actor.hand_known.append(card)
        return

    if event.event_type == "add_to_hand_hidden" and actor is not None:
        actor.hand_unknown_count += event.amount or 1
        return

    if event.event_type == "game_end":
        winner = event.metadata.get("winner") or event.actor
        if winner:
            state.winner = winner
        return


def build_replay_frames(
    events: list[LogEvent],
    known_prizes_by_player: dict[str, list[CardRef]] | None = None,
) -> list[ReplayFrame]:
    state = GameState()
    _apply_known_prize_inputs(state, known_prizes_by_player)

    frames = [ReplayFrame(step=0, event=None, state=copy.deepcopy(state))]

    for event in events:
        _apply_event(state, event)
        _apply_known_prize_inputs(state, known_prizes_by_player)
        frames.append(
            ReplayFrame(
                step=len(frames),
                event=event,
                state=copy.deepcopy(state),
            )
        )

    return frames
