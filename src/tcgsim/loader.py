from __future__ import annotations

from typing import Any, Dict, Iterable, List
import json

from .state import CardInstance, GameState, PlayerState


def load_compiled_cards(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("compiled_cards", "cards"):
            if isinstance(payload.get(key), list):
                return payload[key]
        # Some tools emit a single card object.
        if "card_id" in payload and "compiled_effects" in payload:
            return [payload]
    raise ValueError(f"Could not find compiled card list in {path}")


def filter_complete_cards(cards: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in cards if c.get("parser", {}).get("status") == "complete"]


def build_card_index(cards: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for card in cards:
        cid = card.get("card_id") or card.get("representative_card_id")
        if cid:
            index[cid] = card
        for same_id in card.get("same_effect_card_ids", []) or []:
            index.setdefault(same_id, card)
    return index


def make_instance(card_def: Dict[str, Any], instance_id: str, owner: str, zone: str) -> CardInstance:
    return CardInstance(
        instance_id=instance_id,
        card_id=card_def.get("card_id", "unknown"),
        name=card_def.get("identity", {}).get("name", card_def.get("card_id", "unknown")),
        owner=owner,
        controller=owner,
        zone=zone,
        definition=card_def,
    )


def build_two_player_seed_state(card_defs: List[Dict[str, Any]], deck_size: int = 20, seed: int = 7) -> GameState:
    """Builds a tiny deterministic state for smoke tests, not a legal Pokémon TCG game."""
    if not card_defs:
        raise ValueError("No compiled complete cards available")
    cards: Dict[str, CardInstance] = {}
    players = {"p1": PlayerState("p1"), "p2": PlayerState("p2")}
    # Use the first deck_size cards for p1 and the next deck_size for p2, cycling if needed.
    pool = list(card_defs)
    while len(pool) < deck_size * 2:
        pool.extend(card_defs)
    for player_id, offset in (("p1", 0), ("p2", deck_size)):
        for i, card_def in enumerate(pool[offset : offset + deck_size]):
            iid = f"{player_id}-deck-{i+1:03d}"
            cards[iid] = make_instance(card_def, iid, player_id, "deck")
            players[player_id].deck.append(iid)
        # Put one Pokémon-ish card active if possible, otherwise top card.
        active_id = None
        for iid in list(players[player_id].deck):
            supertype = cards[iid].definition.get("identity", {}).get("supertype")
            if supertype == "Pokémon":
                active_id = iid
                break
        if active_id is None:
            active_id = players[player_id].deck[0]
        players[player_id].deck.remove(active_id)
        players[player_id].active = active_id
        cards[active_id].zone = "active"
        # Draw 5-card smoke-test hand.
        for _ in range(min(5, len(players[player_id].deck))):
            iid = players[player_id].deck.pop(0)
            players[player_id].hand.append(iid)
            cards[iid].zone = "hand"
    return GameState(players=players, cards=cards, rng_seed=seed)
