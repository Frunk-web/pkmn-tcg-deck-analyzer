from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tcgcompiler import default_template_engine  # noqa: E402


EXAMPLES = [
    "Heal 80 damage from your Active Pokémon that has 3 or more Energy attached.",
    "Remove 3 damage counters from each of your Benched Pokémon.",
    "Search your deck for up to 2 basic Energy cards, reveal them, and put them into your hand. Shuffle your deck afterward.",
    "Look at the top 7 cards of your deck. You may reveal a Pokémon you find there and put it into your hand. Shuffle the other cards back into your deck.",
    "Draw 3 cards.",
    "Attach a basic Energy card from your hand to 1 of your Benched Pokémon.",
    "Move a basic Energy from 1 of your Pokémon to another of your Pokémon.",
    "Switch in 1 of your opponent's Benched Pokémon to the Active Spot.",
    "During your opponent's next turn, prevent all damage from and effects of attacks done to this Pokémon.",
    "The Retreat Cost of each Pokémon in play is Colorless more.",
    "Don't apply Weakness and Resistance.",
]


def main() -> None:
    engine = default_template_engine()
    rows = []
    for text in EXAMPLES:
        match = engine.match_first(text)
        rows.append(
            {
                "text": text,
                "matched": match is not None,
                "family": None if match is None else match.family,
                "template_id": None if match is None else match.template_id,
                "ops": [] if match is None else [s.get("op") for s in match.steps],
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if not all(r["matched"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
