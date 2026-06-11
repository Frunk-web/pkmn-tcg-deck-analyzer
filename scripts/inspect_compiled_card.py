from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from tcgsim import load_compiled_cards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiled", default="data/compiled_cards/auto/compiled_cards_all.json")
    ap.add_argument("--card-id")
    ap.add_argument("--name")
    args = ap.parse_args()

    cards = load_compiled_cards(args.compiled)
    matches = []
    for card in cards:
        name = card.get("identity", {}).get("name", "")
        card_id = card.get("card_id", "")
        if args.card_id and args.card_id.lower() == card_id.lower():
            matches.append(card)
        elif args.name and args.name.lower() in name.lower():
            matches.append(card)
    for card in matches[:10]:
        print(json.dumps({
            "card_id": card.get("card_id"),
            "name": card.get("identity", {}).get("name"),
            "parser_status": card.get("parser", {}).get("status"),
            "compiled_effects": card.get("compiled_effects", []),
            "unparsed_text": card.get("parser", {}).get("unparsed_text", []),
        }, indent=2, ensure_ascii=False))
    if not matches:
        print("No matches found.")


if __name__ == "__main__":
    main()
