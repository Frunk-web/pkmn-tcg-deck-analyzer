from __future__ import annotations

import csv
import json
import re
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

    # Generic PTCGL special-set normalization:
    #   sv6-5  -> sv6pt5
    #   sv8-5  -> sv8pt5
    #   me2-5  -> me2pt5
    #   zsv10-5 -> zsv10pt5
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
        parts = raw.rsplit("-", 1)
        if len(parts) == 2:
            set_id, number = parts
            api_set_id = _api_set_id_for_live_set_id(set_id)

            variants.extend(
                [
                    f"{set_id}-{number}",
                    f"{set_id}_{number}",
                    f"{api_set_id}-{number}",
                    f"{api_set_id}_{number}",
                ]
            )

            if "pt" in set_id:
                variants.append(f"{set_id.replace('pt', '-')}-{number}")
                variants.append(f"{set_id.replace('pt', '-')}_{number}")

    return _dedupe_keep_order([_norm(v) for v in variants])


def _looks_like_image_url(value: Any) -> bool:
    raw = _clean(value)
    lowered = raw.lower()

    return lowered.startswith("http") and (
        ".png" in lowered
        or ".jpg" in lowered
        or ".jpeg" in lowered
        or ".webp" in lowered
        or "images." in lowered
    )


def _extract_urls_from_value(value: Any) -> list[str]:
    raw = _clean(value)
    if not raw:
        return []

    urls: list[str] = []

    if _looks_like_image_url(raw):
        urls.append(raw)

    if raw.startswith("{") and raw.endswith("}"):
        try:
            payload = json.loads(raw)
        except Exception:
            payload = None

        if isinstance(payload, dict):
            images = payload.get("images") if isinstance(payload.get("images"), dict) else payload
            for key in ("large", "small", "image_large_url", "image_url", "url"):
                if _looks_like_image_url(images.get(key)):
                    urls.append(_clean(images.get(key)))

    for found in re.findall(r"https?://[^\s,'\"\]\}]+", raw):
        if _looks_like_image_url(found):
            urls.append(found)

    return _dedupe_keep_order(urls)


def _extract_image_urls_from_row(row: dict[str, Any]) -> list[str]:
    large_urls: list[str] = []
    small_urls: list[str] = []
    other_urls: list[str] = []

    raw_json = row.get("raw_json") or row.get("raw") or row.get("json")
    if raw_json:
        large_urls.extend(_extract_urls_from_value(raw_json))

    for col, value in row.items():
        col_norm = _norm_col(col)
        urls = _extract_urls_from_value(value)
        if not urls:
            continue

        if "large" in col_norm or "hires" in col_norm or "highres" in col_norm:
            large_urls.extend(urls)
        elif "small" in col_norm:
            small_urls.extend(urls)
        elif "image" in col_norm or "url" in col_norm:
            other_urls.extend(urls)

    # Card Gallery prefers large image, then small image.
    return _dedupe_keep_order(large_urls + small_urls + other_urls)


@lru_cache(maxsize=1)
def _gallery_image_lookup() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    root = _repo_root()
    data_dir = root / "data"

    candidate_files: list[Path] = []

    preferred_files = [
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

    by_id: dict[str, list[str]] = {}
    by_name: dict[str, list[str]] = {}

    for path in candidate_files:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue

                norm_to_col = {_norm_col(col): col for col in reader.fieldnames}

                id_cols = [
                    col
                    for norm_col, col in norm_to_col.items()
                    if norm_col in {"id", "cardid", "representativecardid"}
                    or norm_col.endswith("cardid")
                ]

                name_cols = [
                    col
                    for norm_col, col in norm_to_col.items()
                    if norm_col in {"name", "cardname"}
                ]

                set_cols = [
                    col
                    for norm_col, col in norm_to_col.items()
                    if norm_col in {"setid", "set"}
                ]

                number_cols = [
                    col
                    for norm_col, col in norm_to_col.items()
                    if norm_col in {"number", "cardnumber", "collectornumber"}
                ]

                for row in reader:
                    urls = _extract_image_urls_from_row(row)
                    if not urls:
                        continue

                    row_id_values: list[str] = []

                    for col in id_cols:
                        row_id_values.append(_clean(row.get(col)))

                    for set_col in set_cols:
                        for number_col in number_cols:
                            set_id = _clean(row.get(set_col))
                            number = _clean(row.get(number_col))
                            if set_id and number:
                                row_id_values.append(f"{set_id}-{number}")
                                row_id_values.append(f"{set_id}_{number}")
                                row_id_values.append(f"{_api_set_id_for_live_set_id(set_id)}-{number}")
                                row_id_values.append(f"{_api_set_id_for_live_set_id(set_id)}_{number}")

                    for raw_id in row_id_values:
                        for key in _id_variants(raw_id):
                            by_id[key] = _dedupe_keep_order(by_id.get(key, []) + urls)

                    for col in name_cols:
                        name_key = _norm(row.get(col))
                        if name_key and name_key != "nan":
                            by_name[name_key] = _dedupe_keep_order(by_name.get(name_key, []) + urls)

        except Exception:
            continue

    return by_id, by_name


def _basic_energy_urls_by_name(card_name: str) -> list[str]:
    energy_type = infer_basic_energy_type(card_name)
    if not energy_type:
        return []

    fallback = BASIC_ENERGY_FALLBACK_IMAGES.get(energy_type)
    if not fallback:
        return []

    _, small_url, large_url = fallback
    return _dedupe_keep_order([large_url, small_url])



def _scrydex_convention_urls(exported_id: str) -> list[str]:
    api_card_id = exported_id_to_api_card_id(exported_id)
    if not api_card_id or "-" not in api_card_id:
        return []

    return [
        f"https://images.scrydex.com/pokemon/{api_card_id}/large",
        f"https://images.scrydex.com/pokemon/{api_card_id}/small",
    ]


def _pokemon_tcg_convention_urls(exported_id: str) -> list[str]:
    set_id, number = _split_exported_id(exported_id)
    if not set_id or not number:
        return []

    api_set_id = _api_set_id_for_live_set_id(set_id)

    urls = [
        f"https://images.pokemontcg.io/{api_set_id}/{number}_hires.png",
        f"https://images.pokemontcg.io/{api_set_id}/{number}.png",
    ]

    if api_set_id != set_id:
        urls.extend(
            [
                f"https://images.pokemontcg.io/{set_id}/{number}_hires.png",
                f"https://images.pokemontcg.io/{set_id}/{number}.png",
            ]
        )

    return _dedupe_keep_order(urls)


def candidate_image_urls_for_card_ref(card: CardRef | None) -> list[str]:
    if card is None or getattr(card, "unknown", False):
        return []

    exported_id = _clean(getattr(card, "exported_id", ""))
    name = _clean(getattr(card, "name", ""))

    # Same Basic Energy behavior as the gallery metadata path.
    # Basic Energy export IDs in PTCG Live can be pseudo IDs like ec_15 / mee_1,
    # so card name is the canonical identity for these.
    energy_urls = _basic_energy_urls_by_name(name)
    if energy_urls:
        return energy_urls

    by_id, by_name = _gallery_image_lookup()

    api_card_id = exported_id_to_api_card_id(exported_id)

    exact_urls: list[str] = []

    # Exact identity from the log must win over same-name matches.
    # Example: sv6_129 must resolve as sv6-129 Drakloak, not another Drakloak print.
    for key in _id_variants(exported_id):
        exact_urls.extend(by_id.get(key, []))

    for key in _id_variants(api_card_id):
        exact_urls.extend(by_id.get(key, []))

    exact_urls.extend(_scrydex_convention_urls(exported_id))
    exact_urls.extend(_pokemon_tcg_convention_urls(exported_id))
    exact_urls = _dedupe_keep_order(exact_urls)

    if exact_urls:
        return exact_urls

    # Last-resort fallback only when the log has no resolvable exact card ID.
    # This can pick a same-name print, so it must never outrank exact identity.
    name_urls: list[str] = []
    if name:
        name_urls.extend(by_name.get(_norm(name), []))

    return _dedupe_keep_order(name_urls)


def best_image_url_for_card_ref(card: CardRef | None) -> str:
    urls = candidate_image_urls_for_card_ref(card)
    return urls[0] if urls else ""


def image_url_for_card_ref(card: CardRef | None) -> str:
    return best_image_url_for_card_ref(card)
