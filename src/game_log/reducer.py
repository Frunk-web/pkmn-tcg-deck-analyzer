from __future__ import annotations

import copy

from .models import CardRef, GameState, LogEvent, PokemonInPlay, ReplayFrame
from .parser import parse_card_refs


def _same_card(a: CardRef, b: CardRef) -> bool:
    if a.exported_id and b.exported_id:
        return a.exported_id == b.exported_id
    return a.name == b.name


def _remove_one(cards: list[CardRef], target: CardRef) -> CardRef | None:
    for i, card in enumerate(cards):
        if _same_card(card, target):
            return cards.pop(i)
    return None


def _find_pokemon(player, target: CardRef) -> tuple[str, int | None, PokemonInPlay | None]:
    if player.active and _same_card(player.active.card, target):
        return "active", None, player.active

    for i, pokemon in enumerate(player.bench):
        if _same_card(pokemon.card, target):
            return "bench", i, pokemon

    return "", None, None


def _find_pokemon_any_player(state: GameState, target: CardRef) -> PokemonInPlay | None:
    for player in state.players.values():
        _, _, pokemon = _find_pokemon(player, target)
        if pokemon is not None:
            return pokemon
    return None


def _add_damage_to_target(state: GameState, target: CardRef, amount: int | None) -> None:
    pokemon = _find_pokemon_any_player(state, target)
    if pokemon is not None and amount:
        pokemon.damage += int(amount)


def _move_known_to_hand(player, cards: list[CardRef]) -> None:
    for card in cards:
        player.hand_known.append(card)


def _move_revealed_draws_to_hand(player, cards: list[CardRef]) -> None:
    # Some PTCG Live log lines first say "- 7 drawn cards." and then reveal
    # the exact bullet list. The count creates unknown cards; the bullet list
    # should replace those unknowns, not add on top of them.
    if cards:
        player.hand_unknown_count = max(0, player.hand_unknown_count - len(cards))
    _move_known_to_hand(player, cards)


def _move_known_to_discard(player, cards: list[CardRef]) -> None:
    for card in cards:
        _remove_one(player.hand_known, card)
        player.discard.append(card)


def _play_pokemon(player, card: CardRef, zone: str) -> None:
    _remove_one(player.hand_known, card)
    pokemon = PokemonInPlay(card=card)

    if zone == "active":
        if player.active is not None:
            player.bench.append(player.active)
        player.active = pokemon
    else:
        player.bench.append(pokemon)


def _attach_energy(player, energy: CardRef, target: CardRef) -> None:
    _remove_one(player.hand_known, energy)
    _, _, pokemon = _find_pokemon(player, target)
    if pokemon is not None:
        pokemon.attached.append(energy)


def _discard_pokemon(player, pokemon: PokemonInPlay) -> None:
    player.discard.append(pokemon.card)
    player.discard.extend(pokemon.evolution_stack)
    player.discard.extend(pokemon.attached)


def _knockout(player, target: CardRef) -> None:
    zone, idx, pokemon = _find_pokemon(player, target)
    if pokemon is None:
        return

    _discard_pokemon(player, pokemon)

    if zone == "active":
        player.active = None
    elif zone == "bench" and idx is not None:
        player.bench.pop(idx)


def _promote(player, target: CardRef) -> None:
    if player.active and _same_card(player.active.card, target):
        return

    for i, pokemon in enumerate(player.bench):
        if _same_card(pokemon.card, target):
            if player.active is not None:
                player.bench.append(player.active)
            player.active = player.bench.pop(i)
            return


def _evolve(player, base: CardRef, evolution: CardRef) -> None:
    _, _, pokemon = _find_pokemon(player, base)
    if pokemon is None:
        # If the base was hidden from our reconstruction, create the evolved Pokémon.
        _play_pokemon(player, evolution, "bench")
        return

    _remove_one(player.hand_known, evolution)
    pokemon.evolution_stack.append(pokemon.card)
    pokemon.card = evolution


def _apply_known_prize_inputs(state: GameState, known_prizes_by_player: dict[str, list[CardRef]] | None) -> None:
    if not known_prizes_by_player:
        return

    for player_name, prizes in known_prizes_by_player.items():
        player = state.ensure_player(player_name)
        player.user_known_prizes = list(prizes)


def _apply_event(state: GameState, event: LogEvent) -> None:
    state.last_event = event

    if event.actor:
        actor = state.ensure_player(event.actor)
    else:
        actor = None

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
        _move_known_to_hand(actor, event.cards)
        return

    if event.event_type == "draw_count" and actor is not None:
        # If the following bullet line reveals identities, revealed_cards will add them.
        if not event.cards and event.amount:
            actor.hand_unknown_count += int(event.amount)
        return

    if event.event_type == "revealed_cards" and actor is not None:
        parent = event.metadata.get("parent_event_type", "")

        if parent in {"opening_hand", "draw_count", "effect_detail"}:
            _move_revealed_draws_to_hand(actor, event.cards)
        elif parent == "discard_count":
            _move_known_to_discard(actor, event.cards)
        elif parent in {"ability_or_attack", "play_card"} and "played them to the Bench" in event.raw:
            for card in event.cards:
                _play_pokemon(actor, card, "bench")
        else:
            # Most bullet reveals are drawn cards, discarded cards, or damage details.
            # If unsure, preserve them as known text but do not mutate board.
            pass
        return

    if event.event_type == "play_to_active" and actor is not None and event.cards:
        _play_pokemon(actor, event.cards[0], "active")
        return

    if event.event_type == "play_to_bench" and actor is not None and event.cards:
        _play_pokemon(actor, event.cards[0], "bench")
        return

    if event.event_type == "play_stadium" and actor is not None and event.cards:
        _remove_one(actor.hand_known, event.cards[0])
        state.stadium = event.cards[0]
        return

    if event.event_type == "play_card" and actor is not None and event.cards:
        _remove_one(actor.hand_known, event.cards[0])
        # Trainers normally end up in discard after resolution.
        actor.discard.append(event.cards[0])
        return

    if event.event_type == "attach_energy" and actor is not None and len(event.cards) >= 2:
        _attach_energy(actor, event.cards[0], event.cards[1])
        return

    if event.event_type == "discard_from_play":
        # Usually an attached Energy or evolution stack piece.
        if actor is not None:
            for card in event.cards:
                actor.discard.append(card)
        return

    if event.event_type == "evolve" and actor is not None and len(event.cards) >= 2:
        _evolve(actor, event.cards[0], event.cards[1])
        return

    if event.event_type == "attack" and len(event.cards) >= 2:
        # Attack lines contain attacker card first and target card second.
        _add_damage_to_target(state, event.cards[-1], event.amount)
        return

    if event.event_type == "place_damage_counters" and event.cards:
        # Counter placement lines contain the target as the final card reference.
        _add_damage_to_target(state, event.cards[-1], event.amount)
        return

    if event.event_type == "knockout" and actor is not None and event.cards:
        _knockout(actor, event.cards[0])
        return

    if event.event_type == "promote_active" and actor is not None and event.cards:
        _promote(actor, event.cards[0])
        return

    if event.event_type == "retreat" and actor is not None and event.cards:
        if actor.active and _same_card(actor.active.card, event.cards[0]):
            actor.bench.append(actor.active)
            actor.active = None
        return

    if event.event_type == "take_prize" and actor is not None:
        for _ in range(event.amount or 1):
            actor.prizes_taken.append(CardRef(name="Unknown prize", unknown=True))
        return

    if event.event_type == "add_to_hand_revealed" and actor is not None and event.cards:
        card = event.cards[0]

        # Priority rule: log-revealed prize cards override user-entered remembered prizes.
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
