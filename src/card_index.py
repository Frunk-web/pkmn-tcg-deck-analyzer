"""Fast local Pokémon TCG card index.

The web app should not rebuild the index from ``data/all_cards.csv`` on every
cold start. This module first tries to load a compact prebuilt artifact:

    data/card_index_prebuilt.json.gz

If that artifact is missing, it falls back to building the same index from the
local CSV. The fallback is useful for development, but the deployed web app
should commit the prebuilt artifact for fast startup/analyze times.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PREBUILT_INDEX = PROJECT_ROOT / "data" / "card_index_prebuilt.json.gz"
DEFAULT_ALL_CARDS_CSV = PROJECT_ROOT / "data" / "all_cards.csv"

INDEX_SCHEMA_VERSION = 1


def normalize_card_name(value: Any) -> str:
    """Normalize a card name for stable dictionary lookup."""

    text = str(value or "")
    text = text.replace("’", "'").replace("`", "'").replace("\u2019", "'")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _first_nonempty(mapping: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return None


def _json_loads_maybe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value

    text = str(value).strip()
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    loaded = _json_loads_maybe(value)
    if isinstance(loaded, list):
        return [str(x).strip() for x in loaded if str(x).strip()]

    text = str(value).strip()
    if not text:
        return []

    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def _raw_json_from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("raw_json", "raw", "card_json", "json"):
        loaded = _json_loads_maybe(row.get(key))
        if isinstance(loaded, dict):
            return loaded
    return {}


@dataclass(frozen=True)
class CardRecord:
    card_id: str
    name: str
    name_norm: str
    supertype: Optional[str] = None
    subtypes: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    hp: Optional[str] = None
    set_id: Optional[str] = None
    set_ptcgo_code: Optional[str] = None
    set_name: Optional[str] = None
    number: Optional[str] = None
    image_url: Optional[str] = None
    image_large_url: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CardRecord":
        return cls(
            card_id=str(data.get("card_id") or ""),
            name=str(data.get("name") or ""),
            name_norm=str(data.get("name_norm") or normalize_card_name(data.get("name"))),
            supertype=data.get("supertype") or None,
            subtypes=tuple(_parse_list(data.get("subtypes"))),
            types=tuple(_parse_list(data.get("types"))),
            hp=str(data.get("hp") or "").strip() or None,
            set_id=str(data.get("set_id") or "").strip() or None,
            set_ptcgo_code=str(data.get("set_ptcgo_code") or "").strip() or None,
            set_name=str(data.get("set_name") or "").strip() or None,
            number=str(data.get("number") or "").strip() or None,
            image_url=str(data.get("image_url") or "").strip() or None,
            image_large_url=str(data.get("image_large_url") or "").strip() or None,
        )


class CardIndex:
    """Read-only card metadata lookup maps."""

    def __init__(
        self,
        raw_records: Sequence[CardRecord],
        source_paths: Sequence[Path] = (),
        source_kind: str = "memory",
    ) -> None:
        self.raw_records = list(raw_records)
        self.source_paths = tuple(source_paths)
        self.source_kind = source_kind

        self.raw_by_id: Dict[str, CardRecord] = {
            card.card_id: card for card in self.raw_records if card.card_id
        }

        self.raw_by_name_norm: Dict[str, List[CardRecord]] = {}
        for card in self.raw_records:
            if card.name_norm:
                self.raw_by_name_norm.setdefault(card.name_norm, []).append(card)

    @classmethod
    def from_prebuilt(cls, path: Path = DEFAULT_PREBUILT_INDEX) -> "CardIndex":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            payload = json.load(f)

        version = int(payload.get("schema_version") or 0)
        if version != INDEX_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported card index schema version {version}; expected {INDEX_SCHEMA_VERSION}."
            )

        records = [CardRecord.from_dict(item) for item in payload.get("records", [])]
        return cls(raw_records=records, source_paths=[path], source_kind="prebuilt")

    @classmethod
    def from_csv(cls, path: Path = DEFAULT_ALL_CARDS_CSV) -> "CardIndex":
        return cls(
            raw_records=load_records_from_all_cards_csv(path),
            source_paths=[path],
            source_kind="csv",
        )

    @classmethod
    def load(
        cls,
        prebuilt_path: Path = DEFAULT_PREBUILT_INDEX,
        all_cards_csv: Path = DEFAULT_ALL_CARDS_CSV,
    ) -> "CardIndex":
        if prebuilt_path.exists():
            return cls.from_prebuilt(prebuilt_path)

        if all_cards_csv.exists():
            return cls.from_csv(all_cards_csv)

        return cls(raw_records=[], source_paths=[], source_kind="empty")

    def summary(self) -> Dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "raw_records": len(self.raw_records),
            "raw_names": len(self.raw_by_name_norm),
            "source_paths": [str(p) for p in self.source_paths],
        }

    @staticmethod
    def _set_aliases(set_code: Optional[str]) -> set[str]:
        if not set_code:
            return set()

        wanted = str(set_code).lower().strip()
        aliases = {wanted, wanted.replace("-", "")}

        # Common promo export pattern: PR-SV should match Pokémon TCG API set id svp.
        if wanted.startswith("pr-"):
            promo_part = wanted.replace("pr-", "")
            aliases.add(f"{promo_part}p")

        return aliases

    @staticmethod
    def _record_matches_print(
        record: CardRecord,
        set_code: Optional[str],
        collector_number: Optional[str],
    ) -> bool:
        wanted_number = str(collector_number or "").lower().strip()
        wanted_aliases = CardIndex._set_aliases(set_code)

        number_ok = True
        if wanted_number:
            number_ok = wanted_number == str(record.number or "").lower().strip()

        set_ok = True
        if wanted_aliases and str(set_code or "").lower() != "energy":
            record_values = {
                str(record.set_id or "").lower().strip(),
                str(record.set_ptcgo_code or "").lower().strip(),
                str(record.set_name or "").lower().strip(),
            }
            set_ok = bool(wanted_aliases & record_values) or any(
                alias and alias in str(record.set_name or "").lower()
                for alias in wanted_aliases
            )

        return set_ok and number_ok

    def find_raw(
        self,
        name: str,
        set_code: Optional[str] = None,
        collector_number: Optional[str] = None,
    ) -> Optional[CardRecord]:
        candidates = self.raw_by_name_norm.get(normalize_card_name(name), [])
        if not candidates:
            return None

        print_matches = [
            card
            for card in candidates
            if self._record_matches_print(card, set_code, collector_number)
        ]
        if print_matches:
            image_matches = [c for c in print_matches if c.image_url or c.image_large_url]
            return image_matches[0] if image_matches else print_matches[0]

        # If the exact print is not in the local index, still use an exact-name record
        # so card type/basic-Pokémon metadata remains local and fast.
        image_matches = [c for c in candidates if c.image_url or c.image_large_url]
        return image_matches[0] if image_matches else candidates[0]

    def attach_metadata(self, deck_card: Any) -> Optional[Any]:
        """Attach local metadata to a DeckCard-like object.

        Returns the same object when a local match is found. Returns ``None`` so
        callers can fall back to the API/cache path when no local match exists.
        """

        record = self.find_raw(
            getattr(deck_card, "name", ""),
            getattr(deck_card, "set_code", None),
            getattr(deck_card, "collector_number", None),
        )
        if record is None:
            return None

        deck_card.api_id = record.card_id or getattr(deck_card, "api_id", None)
        deck_card.supertype = record.supertype or getattr(deck_card, "supertype", None)
        deck_card.subtypes = list(record.subtypes) if record.subtypes else (
            getattr(deck_card, "subtypes", None) or []
        )
        deck_card.image_url = record.image_url or getattr(deck_card, "image_url", None)
        deck_card.image_large_url = (
            record.image_large_url
            or record.image_url
            or getattr(deck_card, "image_large_url", None)
        )
        return deck_card


def load_records_from_all_cards_csv(path: Path = DEFAULT_ALL_CARDS_CSV) -> List[CardRecord]:
    records: List[CardRecord] = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            raw = _raw_json_from_row(row)
            raw_set = raw.get("set") if isinstance(raw.get("set"), dict) else {}
            raw_images = raw.get("images") if isinstance(raw.get("images"), dict) else {}

            card_id = str(
                _first_nonempty(row, ("card_id", "id")) or raw.get("id") or ""
            ).strip()

            name = str(
                _first_nonempty(row, ("name", "card_name")) or raw.get("name") or ""
            ).strip()
            if not name:
                continue

            supertype = (
                _first_nonempty(row, ("supertype", "card_supertype"))
                or raw.get("supertype")
            )
            subtypes = _parse_list(
                _first_nonempty(row, ("subtypes", "subtypes_json")) or raw.get("subtypes")
            )
            types = _parse_list(
                _first_nonempty(row, ("types", "types_json")) or raw.get("types")
            )

            image_url = (
                _first_nonempty(
                    row,
                    ("image_url", "image_small", "images_small", "small_image_url"),
                )
                or raw_images.get("small")
            )
            image_large_url = (
                _first_nonempty(
                    row,
                    ("image_large_url", "image_large", "images_large", "large_image_url"),
                )
                or raw_images.get("large")
                or image_url
            )

            records.append(
                CardRecord(
                    card_id=card_id,
                    name=name,
                    name_norm=normalize_card_name(name),
                    supertype=str(supertype).strip() if supertype else None,
                    subtypes=tuple(subtypes),
                    types=tuple(types),
                    hp=str(_first_nonempty(row, ("hp",)) or raw.get("hp") or "").strip() or None,
                    set_id=str(
                        _first_nonempty(row, ("set_id", "set.id"))
                        or raw_set.get("id")
                        or ""
                    ).strip() or None,
                    set_ptcgo_code=str(
                        _first_nonempty(
                            row,
                            ("set_ptcgo_code", "set_ptcgoCode", "ptcgoCode"),
                        )
                        or raw_set.get("ptcgoCode")
                        or ""
                    ).strip() or None,
                    set_name=str(
                        _first_nonempty(row, ("set_name", "set.name"))
                        or raw_set.get("name")
                        or ""
                    ).strip() or None,
                    number=str(
                        _first_nonempty(row, ("number", "collector_number"))
                        or raw.get("number")
                        or ""
                    ).strip() or None,
                    image_url=str(image_url).strip() if image_url else None,
                    image_large_url=str(image_large_url).strip() if image_large_url else None,
                )
            )

    return records


def write_prebuilt_index(
    output_path: Path = DEFAULT_PREBUILT_INDEX,
    all_cards_csv: Path = DEFAULT_ALL_CARDS_CSV,
) -> Dict[str, Any]:
    """Build and write the compact web-fast index artifact."""

    records = load_records_from_all_cards_csv(all_cards_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "source": str(all_cards_csv),
        "record_count": len(records),
        "records": [asdict(record) for record in records],
    }

    with gzip.open(output_path, "wt", encoding="utf-8", compresslevel=6) as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))

    return {
        "output_path": str(output_path),
        "source": str(all_cards_csv),
        "record_count": len(records),
        "size_bytes": output_path.stat().st_size,
    }


@lru_cache(maxsize=1)
def get_card_index() -> CardIndex:
    """Load the local card index once per Python process."""

    return CardIndex.load()


def clear_card_index_cache() -> None:
    """Clear the process-local index cache after rebuilding data files."""

    get_card_index.cache_clear()
