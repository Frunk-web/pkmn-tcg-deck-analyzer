from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.card_index import DEFAULT_PREBUILT_INDEX, get_card_index


def main() -> None:
    idx = get_card_index()
    summary = idx.summary()

    print("Card index summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if summary["raw_records"] <= 0:
        raise SystemExit("FAIL: card index has zero raw records.")

    samples = [
        "N's Zorua",
        "N's Zoroark ex",
        "Ultra Ball",
        "Secret Box",
        "Poké Pad",
        "Basic Grass Energy",
    ]

    missing = []
    print()
    print("Sample lookups:")
    for name in samples:
        record = idx.find_raw(name)
        if record is None:
            missing.append(name)
            print(f"  MISSING: {name}")
        else:
            print(f"  OK: {name} -> {record.card_id or '(no id)'} | {record.supertype}")

    if missing:
        raise SystemExit(f"FAIL: missing sample lookups: {missing}")

    if DEFAULT_PREBUILT_INDEX.exists():
        print()
        print(f"Prebuilt artifact present: {DEFAULT_PREBUILT_INDEX}")
    else:
        print()
        print("WARNING: prebuilt artifact missing; local CSV fallback is being used.")

    print()
    print("PASS: web-fast card index smoke test succeeded.")


if __name__ == "__main__":
    main()
