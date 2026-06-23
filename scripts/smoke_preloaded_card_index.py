from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.card_index import get_card_index


def main() -> None:
    index = get_card_index()
    print("Card index summary:")
    for key, value in index.summary().items():
        print(f"  {key}: {value}")

    print("\nSample lookups:")
    for name in ["N's Zorua", "N's Zoroark ex", "Ultra Ball", "Secret Box", "Poké Pad"]:
        record = index.find_raw(name)
        if record is None:
            print(f"  {name}: MISSING")
        else:
            print(
                f"  {name}: {record.card_id or 'NO_ID'} | "
                f"{record.supertype or 'NO_SUPERTYPE'} | "
                f"{', '.join(record.subtypes) if record.subtypes else 'NO_SUBTYPES'}"
            )


if __name__ == "__main__":
    main()
