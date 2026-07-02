from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import CardRef

try:
    from src.pokemon_api import BASIC_ENERGY_FALLBACK_IMAGES, infer_basic_energy_type
except Exception:
    BASIC_ENERGY_FALLBACK_IMAGES = {
        "grass": ("sve-1", "https://images.pokemontcg.io/sve/1.png", "https://images.pokemontcg.io/sve/1_hires.png"),
        "fire": ("sve-2", "https://images.pokemontcg.io/sve/2.png", "https://images.pokemontcg.io/sve/2_hires.png"),
        "water": ("sve-3", "https://images.pokemontcg.io/sve/3.png", "https://images.pokemontcg.io/sve/3_hires.png"),
        "lightning": ("sve-4", "https://images.pokemontcg.io/sve/4.png", "https://images.pokemontcg.io/sve/4_hires.png"),
        "psychic": ("sve-5", "https://images.pokemontcg.io/sve/5.png", "https://images.pokemontcg.io/sve/5_hires.png"),
        "fighting": ("sve-6", "https://images.pokemontcg.io/sve/6.png", "https://images.pokemontcg.io/sve/6_hires.png"),
        "darkness": ("sve-7", "https://images.pokemontcg.io/sve/7.png", "https://images.pokemontcg.io/sve/7_hires.png"),
        "metal": ("sve-8", "https://images.pokemontcg.io/sve/8.png", "https://images.pokemontcg.io/sve/8_hires.png"),
    }

    def infer_basic_energy_type(card_name: str) -> str | None:
        lowered = str(card_name or "").lower()
        for energy_type in BASIC_ENERGY_FALLBACK_IMAGES:
            if energy_type in lowered:
                return energy_type
        return None


@dataclass(frozen=True)
class ResolvedCardMetadata:
    card_id: str = ""
    name: str = ""
    supertype: str = ""
    subtypes: tuple[str, ...] = ()
    types: tuple[str, ...] = ()
    image_url: str = ""
    image_large_url: str = ""


_SPECIAL_SET_ID_OVERRIDES = {
    "sv6-5": "sv6pt5",
    "sv8-5": "sv8pt5",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _norm(value: Any) -> str:
    return _clean(value).lower()


def _norm_col(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(value))


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for value in values:
        clean = _clean(value)
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)

    return out


def _split_exported_id(exported_id: str) -> tuple[str, str]:
    raw = _clean(exported_id)
    if "_" not in raw:
        return raw, ""

    set_id, number = raw.rsplit("_", 1)
    return set_id.strip(), number.strip()


def _api_set_id_for_live_set_id(set_id: str) -> str:
    clean = _clean(set_id)
    if not clean:
        return ""

    if clean in _SPECIAL_SET_ID_OVERRIDES:
        return _SPECIAL_SET_ID_OVERRIDES[clean]

    match = re.fullmatch(r"([a-z]+\d+)-(\d+)", clean, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}pt{match.group(2)}"

    return clean


def exported_id_to_api_card_id(exported_id: str) -> str:
    set_id, number = _split_exported_id(exported_id)
    if not set_id or not number:
        return _clean(exported_id)

    return f"{_api_set_id_for_live_set_id(set_id)}-{number}"


def exported_id_to_image_url(exported_id: str, *, hires: bool = False) -> str:
    set_id, number = _split_exported_id(exported_id)
    if not set_id or not number:
        return ""

    api_set_id = _api_set_id_for_live_set_id(set_id)
    suffix = "_hires" if hires else ""
    return f"https://images.pokemontcg.io/{api_set_id}/{number}{suffix}.png"


def _id_variants(raw_id: str) -> list[str]:
    raw = _clean(raw_id)
    if not raw:
        return []

    variants = [raw, raw.replace("_", "-")]

    if "_" in raw:
        set_id, number = _split_exported_id(raw)
        api_set_id = _api_set_id_for_live_set_id(set_id)
        if set_id and number:
            variants.extend(
                [
                    f"{set_id}-{number}",
                    f"{set_id}_{number}",
                    f"{api_set_id}-{number}",
                    f"{api_set_id}_{number}",
                ]
            )

    if "-" in raw:
        set_id, number = raw.rsplit("-", 1)
        api_set_id = _api_set_id_for_live_set_id(set_id)
        if set_id and number:
            variants.extend(
                [
                    f"{set_id}-{number}",
                    f"{set_id}_{number}",
                    f"{api_set_id}-{number}",
                    f"{api_set_id}_{number}",
                ]
            )

            if "pt" in set_id:
                live_style = re.sub(r"([a-z]+\d+)pt(\d+)$", r"\1-\2", set_id, flags=re.IGNORECASE)
                variants.extend([f"{live_style}-{number}", f"{live_style}_{number}"])

    return _dedupe_keep_order([_norm(v) for v in variants])


def _parse_listish(value: Any) -> tuple[str, ...]:
    raw = _clean(value)
    if not raw or raw.lower() == "nan":
        return ()

    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, list):
                return tuple(_clean(x) for x in parsed if _clean(x))
        except Exception:
            pass

    pieces = re.split(r"[|,;/]", raw)
    return tuple(_clean(x) for x in pieces if _clean(x))


def _raw_json_from_row(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("raw_json", "raw", "json"):
        raw = _clean(row.get(key))
        if not raw or not raw.startswith("{"):
            continue
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _first_nonempty(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    norm_to_key = {_norm_col(k): k for k in row.keys()}

    for key in keys:
        real_key = norm_to_key.get(_norm_col(key))
        if real_key is None:
            continue

        value = _clean(row.get(real_key))
        if value and value.lower() != "nan":
            return value

    return ""


def _metadata_from_row(row: dict[str, Any]) -> ResolvedCardMetadata:
    raw = _raw_json_from_row(row)
    raw_set = raw.get("set") if isinstance(raw.get("set"), dict) else {}
    raw_images = raw.get("images") if isinstance(raw.get("images"), dict) else {}

    card_id = (
        _first_nonempty(row, ("card_id", "id", "representative_card_id"))
        or _clean(raw.get("id"))
    )

    name = _first_nonempty(row, ("name", "card_name")) or _clean(raw.get("name"))

    supertype = _first_nonempty(row, ("supertype", "card_supertype")) or _clean(raw.get("supertype"))

    subtypes_raw = (
        _first_nonempty(row, ("subtypes", "subtype"))
        or raw.get("subtypes")
    )

    types_raw = (
        _first_nonempty(row, ("types", "type", "pokemon_types"))
        or raw.get("types")
    )

    image_url = (
        _first_nonempty(row, ("image_url", "image_small", "images_small", "small_image_url"))
        or _clean(raw_images.get("small"))
    )

    image_large_url = (
        _first_nonempty(row, ("image_large_url", "image_large", "images_large", "large_image_url"))
        or _clean(raw_images.get("large"))
        or image_url
    )

    return ResolvedCardMetadata(
        card_id=card_id,
        name=name,
        supertype=supertype,
        subtypes=_parse_listish(subtypes_raw),
        types=_parse_listish(types_raw),
        image_url=image_url,
        image_large_url=image_large_url,
    )


@lru_cache(maxsize=1)
def _metadata_lookup() -> tuple[dict[str, ResolvedCardMetadata], dict[str, list[ResolvedCardMetadata]]]:
    root = _repo_root()
    data_dir = root / "data"

    candidate_files: list[Path] = []

    preferred_files = [
        data_dir / "game_review_card_metadata.csv",
        data_dir / "all_cards.csv",
        data_dir / "cards.csv",
        data_dir / "card_index.csv",
    ]

    for path in preferred_files:
        if path.exists():
            candidate_files.append(path)

    if data_dir.exists():
        for path in sorted(data_dir.glob("*.csv")):
            if path not in candidate_files:
                candidate_files.append(path)

    by_id: dict[str, ResolvedCardMetadata] = {}
    by_name: dict[str, list[ResolvedCardMetadata]] = {}

    for path in candidate_files:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue

                for row in reader:
                    metadata = _metadata_from_row(row)
                    if not metadata.card_id and not metadata.name:
                        continue

                    id_values = [metadata.card_id]

                    set_id = _first_nonempty(row, ("set_id", "set"))
                    number = _first_nonempty(row, ("number", "card_number", "collector_number"))

                    if not set_id:
                        raw = _raw_json_from_row(row)
                        raw_set = raw.get("set") if isinstance(raw.get("set"), dict) else {}
                        set_id = _clean(raw_set.get("id"))

                    if set_id and number:
                        api_set = _api_set_id_for_live_set_id(set_id)
                        id_values.extend(
                            [
                                f"{set_id}-{number}",
                                f"{set_id}_{number}",
                                f"{api_set}-{number}",
                                f"{api_set}_{number}",
                            ]
                        )

                    for raw_id in id_values:
                        for key in _id_variants(raw_id):
                            existing = by_id.get(key)

                            # Prefer exact records with images, matching gallery behavior.
                            if existing is None:
                                by_id[key] = metadata
                            elif not (existing.image_url or existing.image_large_url) and (
                                metadata.image_url or metadata.image_large_url
                            ):
                                by_id[key] = metadata

                    if metadata.name:
                        by_name.setdefault(_norm(metadata.name), []).append(metadata)

        except Exception:
            continue

    return by_id, by_name


def _basic_energy_metadata(card_name: str) -> ResolvedCardMetadata | None:
    energy_type = infer_basic_energy_type(card_name)
    if not energy_type:
        return None

    fallback = BASIC_ENERGY_FALLBACK_IMAGES.get(energy_type)
    if not fallback:
        return None

    card_id, small_url, large_url = fallback

    return ResolvedCardMetadata(
        card_id=card_id,
        name=f"Basic {energy_type.title()} Energy",
        supertype="Energy",
        subtypes=("Basic",),
        types=(),
        image_url=small_url,
        image_large_url=large_url or small_url,
    )


def resolve_card_metadata(card: CardRef | None) -> ResolvedCardMetadata | None:
    if card is None or getattr(card, "unknown", False):
        return None

    name = _clean(getattr(card, "name", ""))
    exported_id = _clean(getattr(card, "exported_id", ""))

    energy_metadata = _basic_energy_metadata(name)
    if energy_metadata is not None:
        return energy_metadata

    by_id, by_name = _metadata_lookup()

    exact_keys: list[str] = []

    exact_keys.extend(_id_variants(exported_id))
    exact_keys.extend(_id_variants(exported_id_to_api_card_id(exported_id)))

    for key in exact_keys:
        metadata = by_id.get(key)
        if metadata is not None:
            return metadata

    # Final fallback only when exact log identity does not exist in local/API metadata.
    # Keep this after exact-ID resolution so alternate prints do not outrank logs.
    if name:
        candidates = by_name.get(_norm(name), [])
        image_matches = [x for x in candidates if x.image_url or x.image_large_url]
        if image_matches:
            return image_matches[0]
        if candidates:
            return candidates[0]

    return None


def candidate_image_urls_for_card_ref(card: CardRef | None) -> list[str]:
    metadata = resolve_card_metadata(card)
    if metadata is None:
        return []

    return _dedupe_keep_order([metadata.image_large_url, metadata.image_url])


def best_image_url_for_card_ref(card: CardRef | None) -> str:
    urls = candidate_image_urls_for_card_ref(card)
    return urls[0] if urls else ""


def image_url_for_card_ref(card: CardRef | None) -> str:
    return best_image_url_for_card_ref(card)


def types_for_card_ref(card: CardRef | None) -> tuple[str, ...]:
    metadata = resolve_card_metadata(card)
    if metadata is None:
        return ()
    return metadata.types
