"""
Explanation

This script downloads Pokémon TCG card data from the Pokémon TCG API and saves it
as a dataframe file.

Main responsibilities:
- Fetch cards from the Pokémon TCG API.
- Keep useful metadata and card text fields.
- Flatten rules, abilities, and attacks into readable text columns.
- Save the full card dataframe to data/all_cards.csv.
- Save a smaller text-focused dataframe to data/all_card_text.csv.

Run from project root:

python scripts/download_all_cards.py

Optional examples:

python scripts/download_all_cards.py --standard-only
python scripts/download_all_cards.py --max-cards 100
python scripts/download_all_cards.py --standard-only --max-cards 100
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests


POKEMON_TCG_API_BASE = "https://api.pokemontcg.io/v2"
OUTPUT_DIR = Path("data")
FULL_OUTPUT_PATH = OUTPUT_DIR / "all_cards.csv"
TEXT_OUTPUT_PATH = OUTPUT_DIR / "all_card_text.csv"


def get_api_key() -> Optional[str]:
    return os.getenv("POKEMON_TCG_API_KEY")


def pokemon_headers() -> Dict[str, str]:
    headers = {}

    api_key = get_api_key()
    if api_key:
        headers["X-Api-Key"] = api_key

    return headers


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_cards(
    query: Optional[str],
    page_size: int = 250,
    max_cards: Optional[int] = None,
    sleep_seconds: float = 0.15,
) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    page = 1

    while True:
        params = {
            "page": page,
            "pageSize": page_size,
            "orderBy": "-set.releaseDate",
        }

        if query:
            params["q"] = query

        response = requests.get(
            f"{POKEMON_TCG_API_BASE}/cards",
            headers=pokemon_headers(),
            params=params,
            timeout=120,
        )

        if response.status_code != 200:
            print("Request failed.")
            print(f"URL: {response.url}")
            print(f"Status: {response.status_code}")
            print(response.text)
            response.raise_for_status()

        payload = response.json()
        page_cards = payload.get("data", [])

        if not page_cards:
            break

        cards.extend(page_cards)

        print(f"Fetched page {page}: {len(page_cards)} cards. Total: {len(cards)}")

        if max_cards is not None and len(cards) >= max_cards:
            return cards[:max_cards]

        if len(page_cards) < page_size:
            break

        page += 1
        time.sleep(sleep_seconds)

    return cards


def join_list(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return " | ".join(str(x) for x in value)

    return str(value)


def json_text(value: Any) -> str:
    if value is None:
        return ""

    return json.dumps(value, ensure_ascii=False)


def flatten_abilities(card: Dict[str, Any]) -> str:
    abilities = card.get("abilities") or []

    parts = []

    for ability in abilities:
        name = ability.get("name") or ""
        ability_type = ability.get("type") or ""
        text = ability.get("text") or ""

        parts.append(f"{name} [{ability_type}]: {text}".strip())

    return " | ".join(parts)


def flatten_attacks(card: Dict[str, Any]) -> str:
    attacks = card.get("attacks") or []

    parts = []

    for attack in attacks:
        name = attack.get("name") or ""
        cost = join_list(attack.get("cost") or [])
        converted_cost = attack.get("convertedEnergyCost")
        damage = attack.get("damage") or ""
        text = attack.get("text") or ""

        parts.append(
            f"{name} | Cost: {cost} | Converted cost: {converted_cost} | "
            f"Damage: {damage} | Text: {text}"
        )

    return " | ".join(parts)


def combined_text(card: Dict[str, Any]) -> str:
    text_parts = []

    rules = card.get("rules") or []
    if rules:
        text_parts.append("Rules: " + " ".join(rules))

    abilities = flatten_abilities(card)
    if abilities:
        text_parts.append("Abilities: " + abilities)

    attacks = flatten_attacks(card)
    if attacks:
        text_parts.append("Attacks: " + attacks)

    return "\n".join(text_parts)


def card_to_row(card: Dict[str, Any]) -> Dict[str, Any]:
    card_set = card.get("set") or {}
    legalities = card.get("legalities") or {}
    images = card.get("images") or {}

    return {
        "card_id": card.get("id"),
        "name": card.get("name"),
        "supertype": card.get("supertype"),
        "subtypes": join_list(card.get("subtypes")),
        "types": join_list(card.get("types")),
        "hp": card.get("hp"),
        "evolves_from": card.get("evolvesFrom"),
        "rules": join_list(card.get("rules")),
        "abilities_text": flatten_abilities(card),
        "attacks_text": flatten_attacks(card),
        "combined_text": combined_text(card),
        "set_id": card_set.get("id"),
        "set_name": card_set.get("name"),
        "set_series": card_set.get("series"),
        "set_release_date": card_set.get("releaseDate"),
        "number": card.get("number"),
        "rarity": card.get("rarity"),
        "regulation_mark": card.get("regulationMark"),
        "legal_standard": legalities.get("standard"),
        "legal_expanded": legalities.get("expanded"),
        "legal_unlimited": legalities.get("unlimited"),
        "image_small": images.get("small"),
        "image_large": images.get("large"),
        "raw_rules_json": json_text(card.get("rules")),
        "raw_abilities_json": json_text(card.get("abilities")),
        "raw_attacks_json": json_text(card.get("attacks")),
        "raw_card_json": json_text(card),
    }


def build_dataframes(cards: List[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [card_to_row(card) for card in cards]

    full_df = pd.DataFrame(rows)

    text_columns = [
        "card_id",
        "name",
        "supertype",
        "subtypes",
        "types",
        "hp",
        "rules",
        "abilities_text",
        "attacks_text",
        "combined_text",
        "set_id",
        "set_name",
        "number",
        "regulation_mark",
        "legal_standard",
        "image_small",
        "image_large",
    ]

    existing_text_columns = [col for col in text_columns if col in full_df.columns]
    text_df = full_df[existing_text_columns].copy()

    return full_df, text_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Pokémon TCG cards and save them as CSV files."
    )

    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="Only download cards legal in Standard.",
    )

    parser.add_argument(
        "--max-cards",
        type=int,
        default=None,
        help="Optional limit for testing.",
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.15,
        help="Delay between API pages.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_output_dir()

    query = "legalities.standard:Legal" if args.standard_only else None

    print(f"Using query: {query}")

    cards = fetch_cards(
        query=query,
        max_cards=args.max_cards,
        sleep_seconds=args.sleep_seconds,
    )

    full_df, text_df = build_dataframes(cards)

    full_df.to_csv(FULL_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    text_df.to_csv(TEXT_OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print()
    print("Done.")
    print(f"Cards downloaded: {len(full_df)}")
    print(f"Saved full dataframe: {FULL_OUTPUT_PATH}")
    print(f"Saved text dataframe: {TEXT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()