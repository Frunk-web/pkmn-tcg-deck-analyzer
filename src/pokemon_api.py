"""
Explanation

This file handles communication with the Pokémon TCG API.

It takes parsed DeckCard objects and tries to match them to real Pokémon TCG API
cards using the card name, set code, and collector number.

Main responsibilities:
- Read the Pokémon TCG API key from Streamlit secrets or environment variables.
- Search the Pokémon TCG API.
- Match cards by name, set code, and collector number.
- Cache API responses locally so repeated analyses are faster.
- Add metadata to DeckCard objects, especially:
  - api_id
  - supertype
  - subtypes

The most important metadata for version 0.1 is whether a card is a Basic Pokémon,
because mulligan probabilities depend on the number of Basic Pokémon in the deck.
"""

import os
import json
import time
import hashlib
from typing import Optional, Dict, List

import requests
import streamlit as st

from src.deck_parser import DeckCard

POKEMON_TCG_API_BASE = "https://api.pokemontcg.io/v2"
CACHE_FILE = "pokemon_opening_hand_cache.json"


def get_api_key():
    return st.secrets.get("POKEMON_TCG_API_KEY", os.getenv("POKEMON_TCG_API_KEY"))


def load_cache() -> Dict[str, dict]:
    if not os.path.exists(CACHE_FILE):
        return {}

    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_cache(cache: Dict[str, dict]) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def make_cache_key(name: str, set_code: Optional[str], collector_number: Optional[str]) -> str:
    raw = f"{name}|{set_code or ''}|{collector_number or ''}".lower()
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def pokemon_tcg_search(query: str, page_size: int = 50) -> List[dict]:
    headers = {}

    api_key = get_api_key()
    if api_key:
        headers["X-Api-Key"] = api_key

    params = {
        "q": query,
        "pageSize": page_size,
        "orderBy": "-set.releaseDate",
    }

    url = f"{POKEMON_TCG_API_BASE}/cards"

    last_error = None

    for attempt in range(3):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=60)
            response.raise_for_status()
            return response.json().get("data", [])

        except requests.exceptions.RequestException as e:
            last_error = e
            wait_time = 2**attempt
            st.warning(f"API request failed for query `{query}`. Retrying in {wait_time}s...")
            time.sleep(wait_time)

    st.warning(f"API lookup failed for query `{query}`: {last_error}")
    return []


def card_matches(card: dict, set_code: Optional[str], collector_number: Optional[str]) -> bool:
    if not set_code and not collector_number:
        return True

    api_set = card.get("set", {}) or {}
    api_set_code = (api_set.get("ptcgoCode") or "").lower()
    api_set_id = (api_set.get("id") or "").lower()
    api_set_name = (api_set.get("name") or "").lower()
    api_number = (card.get("number") or "").lower()

    wanted_set = (set_code or "").lower()
    wanted_number = (collector_number or "").lower()

    set_ok = True
    number_ok = True

    if wanted_set and wanted_set != "energy":
        set_ok = (
            wanted_set == api_set_code or wanted_set == api_set_id or wanted_set in api_set_name
        )

    if wanted_number:
        number_ok = wanted_number == api_number

    return set_ok and number_ok


def fetch_card_metadata(card: DeckCard, cache: Dict[str, dict]) -> DeckCard:
    # Version 0.1 only needs API metadata to identify Basic Pokémon.
    # Trainer and Energy cards can never be Basic Pokémon, so we skip API calls for them.
    if card.section == "Trainer":
        card.api_id = None
        card.supertype = "Trainer"
        card.subtypes = []
        return card

    if card.section == "Energy":
        card.api_id = None
        card.supertype = "Energy"

        if card.name.lower().startswith("basic ") and card.name.lower().endswith(" energy"):
            card.subtypes = ["Basic"]
        else:
            card.subtypes = []

        return card

    key = make_cache_key(card.name, card.set_code, card.collector_number)

    if key in cache:
        api_card = cache[key]
    else:
        safe_name = card.name.replace('"', '\\"')
        queries = []

        if card.set_code and card.collector_number and card.set_code.lower() != "energy":
            queries.append(
                f'name:"{safe_name}" set.ptcgoCode:{card.set_code} number:{card.collector_number}'
            )

        if card.collector_number:
            queries.append(f'name:"{safe_name}" number:{card.collector_number}')

        queries.append(f'name:"{safe_name}"')

        api_card = None

        for query in queries:
            results = pokemon_tcg_search(query)

            exact_name_results = [
                r for r in results if r.get("name", "").lower().strip() == card.name.lower().strip()
            ]

            exact_name_results = exact_name_results or results

            filtered = [
                r
                for r in exact_name_results
                if card_matches(r, card.set_code, card.collector_number)
            ]

            if filtered:
                api_card = filtered[0]
                break

            if exact_name_results and api_card is None:
                api_card = exact_name_results[0]

        if api_card is None:
            st.warning(f"No API match found for {card.label}")
            return card

        cache[key] = api_card
        time.sleep(0.03)

    card.api_id = api_card.get("id")
    card.supertype = api_card.get("supertype")
    card.subtypes = api_card.get("subtypes", [])

    return card


def attach_metadata(deck):
    cache = load_cache()
    updated = []

    for card in deck:
        updated.append(fetch_card_metadata(card, cache))

    save_cache(cache)

    return updated
