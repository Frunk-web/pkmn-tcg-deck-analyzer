"""Preloaded local Pokémon TCG card index.

This module builds a read-only in-memory index from the local card data files
under ``data/``. It is intentionally independent of Streamlit so CLI scripts,
unit tests, and the frontend can all share the same cached object.
"""
from __future__ import annotations

import csv
import gzip
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALL_CARDS_CSV = PROJECT_ROOT / "data" / "all_cards.csv"
DEFAULT_COMPILED_CARD_PATHS = (
    PROJECT_ROOT / "data" / "compiled_cards" / "auto" / "compiled_cards_all.turn1_semantics.json.gz",
    PROJECT_ROOT / "data" / "compiled_cards" / "auto" / "compiled_cards_all.json.gz",
    PROJECT_ROOT / "data" / "compiled_cards" / "compiled_cards_all.json.gz",
    PROJECT_ROOT / "data" / "compiled_cards_all.json.gz",
    PROJECT_ROOT / "data" / "compiled_cards_all.json",
)


def normalize_card_name(value: Any) -> str:
    """Normalize a card name for dictionary lookup."""
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


def _load_json_path(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_compiled_cards(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return
    for key in ("cards", "compiled_cards", "data", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
            return
    # Some historical outputs are dicts keyed by card id/name.
    for value in payload.values():
        if isinstance(value, dict):
            yield value


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
    raw: Dict[str, Any] | None = None


class CardIndex:
    """Read-only lookup maps for raw and compiled card data."""

    def __init__(
        self,
        raw_records: Sequence[CardRecord],
        compiled_cards: Sequence[Dict[str, Any]],
        source_paths: Sequence[Path],
    ) -> None:
        self.raw_records = list(raw_records)
        self.compiled_cards = list(compiled_cards)
        self.source_paths = tuple(source_paths)

        self.raw_by_id: Dict[str, CardRecord] = {
            card.card_id: card for card in self.raw_records if card.card_id
        }
        self.raw_by_name_norm: Dict[str, List[CardRecord]] = {}
        for card in self.raw_records:
            self.raw_by_name_norm.setdefault(card.name_norm, []).append(card)

        self.compiled_by_id: Dict[str, Dict[str, Any]] = {}
        self.compiled_by_name_norm: Dict[str, List[Dict[str, Any]]] = {}
        for card in self.compiled_cards:
            card_id = str(_first_nonempty(card, ("card_id", "id")) or "")
            name = str(_first_nonempty(card, ("name", "card_name")) or "")
            identity = card.get("identity") if isinstance(card.get("identity"), dict) else {}
            if not name:
                name = str(identity.get("name") or "")
            if card_id:
                self.compiled_by_id[card_id] = card
            name_norm = normalize_card_name(name)
            if name_norm:
                self.compiled_by_name_norm.setdefault(name_norm, []).append(card)

    @classmethod
    def from_files(
        cls,
        all_cards_csv: Path = DEFAULT_ALL_CARDS_CSV,
        compiled_paths: Sequence[Path] = DEFAULT_COMPILED_CARD_PATHS,
    ) -> "CardIndex":
        source_paths: List[Path] = []
        raw_records: List[CardRecord] = []
        if all_cards_csv.exists():
            raw_records = cls._load_all_cards_csv(all_cards_csv)
            source_paths.append(all_cards_csv)

        compiled_cards: List[Dict[str, Any]] = []
        for path in compiled_paths:
            if not path.exists():
                continue
            payload = _load_json_path(path)
            compiled_cards = list(_iter_compiled_cards(payload))
            source_paths.append(path)
            break

        return cls(raw_records=raw_records, compiled_cards=compiled_cards, source_paths=source_paths)

    @staticmethod
    def _load_all_cards_csv(path: Path) -> List[CardRecord]:
        records: List[CardRecord] = []
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw = _raw_json_from_row(row)
                raw_set = raw.get("set") if isinstance(raw.get("set"), dict) else {}
                raw_images = raw.get("images") if isinstance(raw.get("images"), dict) else {}

                card_id = str(
                    _first_nonempty(row, ("card_id", "id"))
                    or raw.get("id")
                    or ""
                ).strip()
                name = str(_first_nonempty(row, ("name", "card_name")) or raw.get("name") or "").strip()
                if not name:
                    continue

                supertype = _first_nonempty(row, ("supertype", "card_supertype")) or raw.get("supertype")
                subtypes = _parse_list(_first_nonempty(row, ("subtypes", "subtypes_json")) or raw.get("subtypes"))
                types = _parse_list(_first_nonempty(row, ("types", "types_json")) or raw.get("types"))

                image_url = (
                    _first_nonempty(row, ("image_url", "image_small", "images_small", "small_image_url"))
                    or raw_images.get("small")
                )
                image_large_url = (
                    _first_nonempty(row, ("image_large_url", "image_large", "images_large", "large_image_url"))
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
                        set_id=str(_first_nonempty(row, ("set_id", "set.id")) or raw_set.get("id") or "").strip() or None,
                        set_ptcgo_code=str(
                            _first_nonempty(row, ("set_ptcgo_code", "set_ptcgoCode", "ptcgoCode"))
                            or raw_set.get("ptcgoCode")
                            or ""
                        ).strip() or None,
                        set_name=str(_first_nonempty(row, ("set_name", "set.name")) or raw_set.get("name") or "").strip() or None,
                        number=str(_first_nonempty(row, ("number", "collector_number")) or raw.get("number") or "").strip() or None,
                        image_url=str(image_url).strip() if image_url else None,
                        image_large_url=str(image_large_url).strip() if image_large_url else None,
                        raw=raw or dict(row),
                    )
                )
        return records

    def summary(self) -> Dict[str, Any]:
        return {
            "raw_records": len(self.raw_records),
            "raw_names": len(self.raw_by_name_norm),
            "compiled_cards": len(self.compiled_cards),
            "compiled_names": len(self.compiled_by_name_norm),
            "source_paths": [str(p) for p in self.source_paths],
        }

    @staticmethod
    def _set_aliases(set_code: Optional[str]) -> set[str]:
        if not set_code:
            return set()
        wanted = str(set_code).lower().strip()
        aliases = {wanted, wanted.replace("-", "")}
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
        # Deck exports can use set_code="Energy" for pseudo Basic Energy printings.
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

        # If the exact print is not in the local CSV, still use an exact-name
        # record so card type/basic-Pokémon metadata remains local and fast.
        image_matches = [c for c in candidates if c.image_url or c.image_large_url]
        return image_matches[0] if image_matches else candidates[0]

    def find_compiled(self, name: str) -> Optional[Dict[str, Any]]:
        candidates = self.compiled_by_name_norm.get(normalize_card_name(name), [])
        return candidates[0] if candidates else None

    def attach_metadata(self, deck_card: Any) -> Optional[Any]:
        """Attach local metadata to a DeckCard-like object.

        Returns the same object when a local match is found, otherwise ``None`` so
        callers can fall back to the existing API/cache path.
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
        deck_card.subtypes = list(record.subtypes) if record.subtypes else (getattr(deck_card, "subtypes", None) or [])
        deck_card.image_url = record.image_url or getattr(deck_card, "image_url", None)
        deck_card.image_large_url = record.image_large_url or record.image_url or getattr(deck_card, "image_large_url", None)
        return deck_card


@lru_cache(maxsize=1)
def get_card_index() -> CardIndex:
    """Load the local card index once per Python process."""
    return CardIndex.from_files()


def clear_card_index_cache() -> None:
    """Clear the process-local index cache, useful after rebuilding data files."""
    get_card_index.cache_clear()
