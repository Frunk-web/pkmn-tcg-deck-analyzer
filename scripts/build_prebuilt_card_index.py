from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.card_index import DEFAULT_ALL_CARDS_CSV, DEFAULT_PREBUILT_INDEX, write_prebuilt_index


def main() -> None:
    if not DEFAULT_ALL_CARDS_CSV.exists():
        raise SystemExit(
            f"Missing {DEFAULT_ALL_CARDS_CSV}. Build or copy data/all_cards.csv locally first."
        )

    summary = write_prebuilt_index(DEFAULT_PREBUILT_INDEX, DEFAULT_ALL_CARDS_CSV)

    print("Built prebuilt card index:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if summary["record_count"] <= 0:
        raise SystemExit("Prebuilt index has zero records; refusing to continue.")

    print()
    print("Commit this file for web speed:")
    print(f"  git add -f {DEFAULT_PREBUILT_INDEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
