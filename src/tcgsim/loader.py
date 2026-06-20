from __future__ import annotations

from typing import Any, Dict, Iterable, List
import json
import gzip
from pathlib import Path

from .state import CardInstance, GameState, PlayerState


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

def _looks_like_card_record(x):
    if not isinstance(x, dict):
        return False
    if isinstance(x.get("parser"), dict):
        return True
    if isinstance(x.get("identity"), dict):
        return True
    return any(k in x for k in (
        "card_id",
        "representative_card_id",
        "same_effect_printings",
        "name",
        "card_name",
        "supertype",
        "compiled_effects",
        "effects",
    ))


def _extract_card_records(payload):
    if isinstance(payload, list):
        cards = [c for c in payload if _looks_like_card_record(c)]
        if cards:
            return cards

    if isinstance(payload, dict):
        for key in (
            "cards",
            "data",
            "records",
            "compiled_cards",
            "card_records",
            "items",
            "results",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                cards = [c for c in value if _looks_like_card_record(c)]
                if cards:
                    return cards

            if isinstance(value, dict):
                cards = [c for c in value.values() if _looks_like_card_record(c)]
                if cards:
                    return cards

        cards = [c for c in payload.values() if _looks_like_card_record(c)]
        if cards:
            return cards

    # Last-resort shallow recursive search for the largest card-looking collection.
    best = []

    def walk(x, depth=0):
        nonlocal best
        if depth > 5:
            return

        if isinstance(x, list):
            cards = [c for c in x if _looks_like_card_record(c)]
            if len(cards) > len(best):
                best = cards
            for item in x[:25]:
                walk(item, depth + 1)

        elif isinstance(x, dict):
            cards = [c for c in x.values() if _looks_like_card_record(c)]
            if len(cards) > len(best):
                best = cards
            for item in list(x.values())[:25]:
                walk(item, depth + 1)

    walk(payload)

    if best:
        return best

    if isinstance(payload, dict):
        raise ValueError(f"Could not extract card records from compiled payload. Top-level keys: {list(payload.keys())[:20]}")

    raise ValueError(f"Could not extract card records from compiled payload type: {type(payload).__name__}")

def load_compiled_cards(path):
    # TURN1_GZIP_COMPILED_SEMANTICS_NORMALIZED
    p = Path(path)
    read_path = p

    if not read_path.exists():
        gz_path = Path(str(p) + ".gz")
        if gz_path.exists():
            read_path = gz_path

    if not read_path.exists():
        raise FileNotFoundError(f"Compiled card semantics not found: {p} or {p}.gz")

    if read_path.suffix.lower() == ".gz":
        with gzip.open(read_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)
    else:
        with open(read_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

    return _extract_card_records(payload)


def filter_complete_cards(cards):
    # TURN1_FILTER_COMPLETE_CARDS_DICT_ONLY
    return [
        c for c in cards
        if isinstance(c, dict) and c.get("parser", {}).get("status") == "complete"
    ]
