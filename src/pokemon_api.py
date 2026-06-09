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
- Detect stale cached API responses that do not contain image URLs.
- Add metadata to DeckCard objects:
  - api_id
  - supertype
  - subtypes
  - image_url
  - image_large_url

The app uses API image URLs to display a visual card gallery with probability overlays.
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

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_cache(cache: Dict[str, dict]) -> None:
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def make_cache_key(name: str, set_code: Optional[str], collector_number: Optional[str]) -> str:
    raw = f"{name}|{set_code or ''}|{collector_number or ''}".lower()
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def normalize_set_aliases(set_code: Optional[str]) -> set:
    if not set_code:
        return set()

    wanted = set_code.lower().strip()
    aliases = {wanted, wanted.replace("-", "")}

    # Common promo export pattern:
    # PR-SV should match Pokémon TCG API set id "svp".
    if wanted.startswith("pr-"):
        promo_part = wanted.replace("pr-", "")
        aliases.add(f"{promo_part}p")

    return aliases


def api_card_has_image(api_card: dict) -> bool:
    images = api_card.get("images", {}) or {}
    return bool(images.get("small") or images.get("large"))


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

    wanted_aliases = normalize_set_aliases(set_code)
    wanted_number = (collector_number or "").lower()

    set_ok = True
    number_ok = True

    if wanted_aliases and (set_code or "").lower() != "energy":
        set_ok = (
            api_set_code in wanted_aliases
            or api_set_id in wanted_aliases
            or any(alias in api_set_name for alias in wanted_aliases)
        )

    if wanted_number:
        number_ok = wanted_number == api_number

    return set_ok and number_ok


def fallback_classify_card(card: DeckCard) -> DeckCard:
    """
    If the API cannot match a card, classify it from the decklist section.
    This keeps the probability calculations safe even if an image is unavailable.
    """

    card.api_id = None
    card.image_url = None
    card.image_large_url = None

    if card.section == "Trainer":
        card.supertype = "Trainer"
        card.subtypes = []
        return card

    if card.section == "Energy":
        card.supertype = "Energy"

        if card.name.lower().startswith("basic ") and card.name.lower().endswith(" energy"):
            card.subtypes = ["Basic"]
        else:
            card.subtypes = []

        return card

    return card


def attach_api_card_to_deck_card(card: DeckCard, api_card: dict) -> DeckCard:
    card.api_id = api_card.get("id")
    card.supertype = api_card.get("supertype")
    card.subtypes = api_card.get("subtypes", [])

    images = api_card.get("images", {}) or {}
    card.image_url = images.get("small")
    card.image_large_url = images.get("large") or images.get("small")

    return card


def build_queries(card: DeckCard) -> List[str]:
    safe_name = card.name.replace('"', '\\"')
    queries = []

    # Avoid strict set-code query for hyphenated promo codes like PR-SV,
    # because those often do not match ptcgoCode directly.
    if (
        card.set_code
        and card.collector_number
        and card.set_code.lower() != "energy"
        and "-" not in card.set_code
    ):
        queries.append(
            f'name:"{safe_name}" set.ptcgoCode:{card.set_code} number:{card.collector_number}'
        )

    # Basic Energy exports can produce fake set_code="Energy".
    # For those, name-only tends to be safer than forcing collector number.
    if card.collector_number and (card.set_code or "").lower() != "energy":
        queries.append(f'name:"{safe_name}" number:{card.collector_number}')

    queries.append(f'name:"{safe_name}"')

    return queries


def find_best_api_match(card: DeckCard) -> Optional[dict]:
    queries = build_queries(card)
    api_card = None

    for query in queries:
        results = pokemon_tcg_search(query)

        exact_name_results = [
            r for r in results
            if r.get("name", "").lower().strip() == card.name.lower().strip()
        ]

        exact_name_results = exact_name_results or results

        filtered = [
            r for r in exact_name_results
            if card_matches(r, card.set_code, card.collector_number)
        ]

        if filtered:
            image_matches = [r for r in filtered if api_card_has_image(r)]
            return image_matches[0] if image_matches else filtered[0]

        if exact_name_results and api_card is None:
            image_matches = [r for r in exact_name_results if api_card_has_image(r)]
            api_card = image_matches[0] if image_matches else exact_name_results[0]

    return api_card


def fetch_card_metadata(card: DeckCard, cache: Dict[str, dict]) -> DeckCard:
    key = make_cache_key(card.name, card.set_code, card.collector_number)

    if key in cache:
        cached_api_card = cache[key]

        # Durable cache validation:
        # Older cached entries may not have image URLs. If an image is missing,
        # treat the cache entry as stale and refetch it.
        if api_card_has_image(cached_api_card):
            return attach_api_card_to_deck_card(card, cached_api_card)

        del cache[key]

    api_card = find_best_api_match(card)

    if api_card is None:
        st.warning(f"No API match found for {card.label}")
        return fallback_classify_card(card)

    cache[key] = api_card
    time.sleep(0.03)

    return attach_api_card_to_deck_card(card, api_card)


def attach_metadata(deck):
    cache = load_cache()
    updated = []

    for card in deck:
        updated.append(fetch_card_metadata(card, cache))

    save_cache(cache)

    return updated