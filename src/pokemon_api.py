"""
Explanation
This file handles communication with the Pokémon TCG API.

It takes parsed DeckCard objects and tries to match them to real Pokémon TCG
API cards using the card name, set code, and collector number.

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

The app uses API image URLs to display a visual card gallery with probability
overlays.

Note on Basic Energy cards:
Some deck exports use pseudo-printings such as:
  Basic {G} Energy Energy 1
  Basic {L} Energy Energy 12
  Basic {F} Energy Energy 14

Those pseudo set codes are not reliable Pokémon TCG API identifiers. If the API
cannot resolve a Basic Energy printing, this module supplies a stable fallback
image from the Scarlet & Violet Energy set so the gallery does not show
"No image found" for vanilla Basic Energy cards.
"""

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st

from src.deck_parser import DeckCard
from src.card_index import get_card_index


POKEMON_TCG_API_BASE = "https://api.pokemontcg.io/v2"
CACHE_FILE = "pokemon_opening_hand_cache.json"


# Stable fallback images for vanilla Basic Energy cards.
# These are intentionally keyed by Energy type rather than collector number,
# because exported pseudo-printings like "Energy 12" are not stable API IDs.
BASIC_ENERGY_FALLBACK_IMAGES: Dict[str, Tuple[str, str, str]] = {
    "grass": ("sve-1", "https://images.pokemontcg.io/sve/1.png", "https://images.pokemontcg.io/sve/1_hires.png"),
    "fire": ("sve-2", "https://images.pokemontcg.io/sve/2.png", "https://images.pokemontcg.io/sve/2_hires.png"),
    "water": ("sve-3", "https://images.pokemontcg.io/sve/3.png", "https://images.pokemontcg.io/sve/3_hires.png"),
    "lightning": ("sve-4", "https://images.pokemontcg.io/sve/4.png", "https://images.pokemontcg.io/sve/4_hires.png"),
    "psychic": ("sve-5", "https://images.pokemontcg.io/sve/5.png", "https://images.pokemontcg.io/sve/5_hires.png"),
    "fighting": ("sve-6", "https://images.pokemontcg.io/sve/6.png", "https://images.pokemontcg.io/sve/6_hires.png"),
    "darkness": ("sve-7", "https://images.pokemontcg.io/sve/7.png", "https://images.pokemontcg.io/sve/7_hires.png"),
    "metal": ("sve-8", "https://images.pokemontcg.io/sve/8.png", "https://images.pokemontcg.io/sve/8_hires.png"),
}


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


def is_basic_energy_card(card: DeckCard) -> bool:
    name = (card.name or "").lower().strip()
    return name.startswith("basic ") and name.endswith(" energy")


def infer_basic_energy_type(card_name: str) -> Optional[str]:
    name = (card_name or "").lower()

    for energy_type in BASIC_ENERGY_FALLBACK_IMAGES:
        if energy_type in name:
            return energy_type

    return None


def apply_basic_energy_fallback_metadata(card: DeckCard) -> DeckCard:
    """Attach safe local metadata and fallback images for vanilla Basic Energy."""
    energy_type = infer_basic_energy_type(card.name)

    card.supertype = "Energy"
    card.subtypes = ["Basic"]

    if energy_type is None:
        return card

    fallback_id, small_url, large_url = BASIC_ENERGY_FALLBACK_IMAGES[energy_type]

    # Preserve a real API id if one was found. Otherwise use the known fallback id.
    card.api_id = card.api_id or fallback_id
    card.image_url = card.image_url or small_url
    card.image_large_url = card.image_large_url or large_url or small_url

    return card


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

    # Basic Energy exports can produce fake set_code="Energy".
    # Do not enforce set matching for that pseudo set.
    if wanted_aliases and (set_code or "").lower() != "energy":
        # STRICT_SET_CODE_MATCH_V1
        # Do not allow substring matches against set names.
        # Example bug: deck code "CRI" matched "Crimson Invasion",
        # causing Kakuna CRI 2 to display sm4-2.
        set_ok = (
            api_set_code in wanted_aliases
            or api_set_id in wanted_aliases
        )

    if wanted_number:
        number_ok = wanted_number == api_number

    return set_ok and number_ok


def fallback_classify_card(card: DeckCard) -> DeckCard:
    """
    If the API cannot match a card, classify it from the decklist section.

    This keeps the probability calculations safe even if a card image is
    unavailable. Basic Energy gets a special fallback image so the gallery still
    renders a real card image for vanilla Energy cards.
    """
    card.api_id = None
    card.image_url = None
    card.image_large_url = None

    if card.section == "Trainer":
        card.supertype = "Trainer"
        card.subtypes = []
        return card

    if card.section == "Energy":
        if is_basic_energy_card(card):
            return apply_basic_energy_fallback_metadata(card)

        card.supertype = "Energy"
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

    # If the API/cache entry is missing images for a Basic Energy, keep the
    # useful metadata but supply a stable fallback image.
    if is_basic_energy_card(card) and not (card.image_url or card.image_large_url):
        card = apply_basic_energy_fallback_metadata(card)

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

    # STRICT_EXACT_PRINT_IMAGES_V2
    # If the decklist provided set/number identity, do not accept another card
    # with the same name just because it has an image.
    exact_print_requested = bool(
        (card.set_code or card.collector_number)
        and (card.set_code or "").lower() != "energy"
    )

    api_card = None

    for query in queries:
        results = pokemon_tcg_search(query)

        exact_name_results = [
            r
            for r in results
            if r.get("name", "").lower().strip() == card.name.lower().strip()
        ]
        exact_name_results = exact_name_results or results

        filtered = [
            r
            for r in exact_name_results
            if card_matches(r, card.set_code, card.collector_number)
        ]

        if filtered:
            image_matches = [r for r in filtered if api_card_has_image(r)]
            return image_matches[0] if image_matches else filtered[0]

        # Name-only fallback is allowed only for ambiguous/name-only deck rows.
        if exact_name_results and api_card is None and not exact_print_requested:
            image_matches = [r for r in exact_name_results if api_card_has_image(r)]
            api_card = image_matches[0] if image_matches else exact_name_results[0]

    return api_card

def try_attach_local_card_index_metadata(card: DeckCard) -> Optional[DeckCard]:
    """Attach metadata from the prebuilt local index before using the API.

    The web deployment should hit this path for normal deck analysis. If the
    index is missing or a card cannot be found locally, callers fall back to the
    existing API/cache behavior.
    """

    if get_card_index is None:
        return None

    try:
        matched = get_card_index().attach_metadata(card)
    except Exception:
        return None

    if matched is None:
        return None

    if is_basic_energy_card(matched) and not (matched.image_url or matched.image_large_url):
        matched = apply_basic_energy_fallback_metadata(matched)

    return matched



def fetch_card_metadata(card: DeckCard, cache: Dict[str, dict]) -> DeckCard:
    key = make_cache_key(card.name, card.set_code, card.collector_number)

    exact_print_requested = bool(
        (card.set_code or card.collector_number)
        and (card.set_code or "").lower() != "energy"
    )

    if key in cache:
        cached_api_card = cache[key]

        # STRICT_EXACT_PRINT_IMAGES_V2
        # Old cache entries may contain a same-name-but-wrong-print card.
        # If an exact print was requested, invalidate mismatched cache rows.
        if exact_print_requested and not card_matches(
            cached_api_card,
            card.set_code,
            card.collector_number,
        ):
            del cache[key]
        elif api_card_has_image(cached_api_card):
            return attach_api_card_to_deck_card(card, cached_api_card)
        else:
            del cache[key]

    api_card = find_best_api_match(card)

    if api_card is None:
        st.warning(f"No API match found for {card.label}")
        return fallback_classify_card(card)

    cache[key] = api_card
    time.sleep(0.03)
    return attach_api_card_to_deck_card(card, api_card)

def attach_metadata(deck):
    """Attach card metadata using the preloaded local card index first.

    If a card is not present in the local index, fall back to the existing
    API/cache path. This keeps current behavior but removes API/cache work for
    cards already present in data/all_cards.csv.
    """
    cache = load_cache()
    try:
        card_index = get_card_index()
    except Exception:
        card_index = None

    updated = []
    for card in deck:
        indexed_card = None
        if card_index is not None:
            try:
                indexed_card = card_index.attach_metadata(card)
            except Exception:
                indexed_card = None
        if indexed_card is not None:
            updated.append(indexed_card)
        else:
            updated.append(fetch_card_metadata(card, cache))

    save_cache(cache)
    return updated
