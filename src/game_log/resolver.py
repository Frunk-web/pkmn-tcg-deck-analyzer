from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

from .models import CardRef


_LIVE_SET_ID_TO_API_SET_ID = {
    # PTCG Live exports special sets with hyphen notation; the public API uses pt notation.
    "sv6-5": "sv6pt5",
    "sv8-5": "sv8pt5",
}


_BASIC_ENERGY_NAME_FALLBACKS = {
    "basic grass energy": [
        "https://images.pokemontcg.io/sv2/278.png",
        "https://images.pokemontcg.io/sv2/278_hires.png",
    ],
    "basic fire energy": [
        "https://images.pokemontcg.io/sv3/230.png",
        "https://images.pokemontcg.io/sv3/230_hires.png",
    ],
    "basic water energy": [
        "https://images.pokemontcg.io/sv2/279.png",
        "https://images.pokemontcg.io/sv2/279_hires.png",
    ],
    "basic lightning energy": [
        "https://images.pokemontcg.io/sv1/257.png",
        "https://images.pokemontcg.io/sv1/257_hires.png",
    ],
    "basic psychic energy": [
        "https://images.pokemontcg.io/sv3pt5/207.png",
        "https://images.pokemontcg.io/sv3pt5/207_hires.png",
    ],
    "basic fighting energy": [
        "https://images.pokemontcg.io/sv1/258.png",
        "https://images.pokemontcg.io/sv1/258_hires.png",
    ],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _clean(value: object) -> str:
    return str(value or "").strip()


def _norm_key(value: object) -> str:
    return _clean(value).lower()


def _norm_col(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", _clean(value).lower())


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out = []

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
    mapped = _LIVE_SET_ID_TO_API_SET_ID.get(clean)
    if mapped:
        return mapped

    # Generic special-set normalization, e.g. sv6-5 -> sv6pt5.
    m = re.fullmatch(r"(sv\d+)-(\d+)", clean)
    if m:
        return f"{m.group(1)}pt{m.group(2)}"

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


def _looks_like_image_url(value: object) -> bool:
    raw = _clean(value)
    lowered = raw.lower()
    return lowered.startswith("http") and (
        ".png" in lowered
        or ".jpg" in lowered
        or ".jpeg" in lowered
        or ".webp" in lowered
        or "images." in lowered
    )


def _extract_urls_from_value(value: object) -> list[str]:
    raw = _clean(value)
    if not raw:
        return []

    urls: list[str] = []

    if _looks_like_image_url(raw):
        urls.append(raw)

    # Some CSV exports store image payloads as JSON-ish strings.
    if raw.startswith("{") and raw.endswith("}"):
        try:
            payload = json.loads(raw)
        except Exception:
            payload = None

        if isinstance(payload, dict):
            for key in ("small", "large", "url", "image", "image_url"):
                if _looks_like_image_url(payload.get(key)):
                    urls.append(_clean(payload.get(key)))

    # Last-resort URL extraction from a string blob.
    for found in re.findall(r"https?://[^\s,'\"\]\}]+", raw):
        if _looks_like_image_url(found):
            urls.append(found)

    return _dedupe_keep_order(urls)


def _extract_image_urls_from_row(row: dict[str, str]) -> list[str]:
    small_urls: list[str] = []
    large_urls: list[str] = []
    other_urls: list[str] = []

    for col, value in row.items():
        col_norm = _norm_col(col)
        urls = _extract_urls_from_value(value)
        if not urls:
            continue

        if "small" in col_norm:
            small_urls.extend(urls)
        elif "large" in col_norm or "hires" in col_norm or "highres" in col_norm:
            large_urls.extend(urls)
        elif "image" in col_norm or "url" in col_norm:
            other_urls.extend(urls)

    # Prefer smaller images for replay performance; browser can still fall back.
    return _dedupe_keep_order(small_urls + large_urls + other_urls)


@lru_cache(maxsize=1)
def _local_gallery_image_lookup() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Build an image lookup from local card/gallery data.

    This is global and data-driven. It avoids hard-coding one set like ME3.
    """
    root = _repo_root()
    data_dir = root / "data"

    candidate_files: list[Path] = []
    preferred = [
        data_dir / "all_cards.csv",
        data_dir / "cards.csv",
        data_dir / "card_index.csv",
    ]

    for path in preferred:
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
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue

                norm_to_col = {_norm_col(col): col for col in reader.fieldnames}

                id_cols = [
                    col
                    for norm, col in norm_to_col.items()
                    if norm in {"id", "cardid", "representativecardid"}
                    or norm.endswith("cardid")
                ]

                name_cols = [
                    col
                    for norm, col in norm_to_col.items()
                    if norm in {"name", "cardname"}
                ]

                for row in reader:
                    urls = _extract_image_urls_from_row(row)
                    if not urls:
                        continue

                    for col in id_cols:
                        key = _norm_key(row.get(col))
                        if key and key != "nan":
                            by_id[key] = _dedupe_keep_order(by_id.get(key, []) + urls)

                    for col in name_cols:
                        key = _norm_key(row.get(col))
                        if key and key != "nan":
                            by_name[key] = _dedupe_keep_order(by_name.get(key, []) + urls)

        except Exception:
            continue

    return by_id, by_name


def _scrydex_candidates(api_card_id: str) -> list[str]:
    clean = _clean(api_card_id)
    if not clean or "-" not in clean:
        return []

    return [
        f"https://images.scrydex.com/pokemon/{clean}/small",
        f"https://images.scrydex.com/pokemon/{clean}/large",
    ]


def _pokemontcg_candidates(set_id: str, number: str) -> list[str]:
    clean_set = _clean(set_id)
    clean_number = _clean(number)
    if not clean_set or not clean_number:
        return []

    return [
        f"https://images.pokemontcg.io/{clean_set}/{clean_number}.png",
        f"https://images.pokemontcg.io/{clean_set}/{clean_number}_hires.png",
    ]


def candidate_image_urls_for_card_ref(card: CardRef | None) -> list[str]:
    if card is None or getattr(card, "unknown", False):
        return []

    exported_id = _clean(getattr(card, "exported_id", ""))
    name = _clean(getattr(card, "name", ""))

    set_id, number = _split_exported_id(exported_id)
    api_set_id = _api_set_id_for_live_set_id(set_id) if set_id else ""
    api_card_id = f"{api_set_id}-{number}" if api_set_id and number else ""

    by_id, by_name = _local_gallery_image_lookup()

    urls: list[str] = []

    # 1. ScryDex first for all normal card IDs.
    # Reason: pokemontcg.io can return a card-back placeholder with HTTP 200,
    # which means browser onerror never fires. This is not set-specific.
    if api_card_id and not api_card_id.startswith("mee-"):
        urls.extend(_scrydex_candidates(api_card_id))

    # 2. Local/gallery data, globally by exact IDs and names.
    for key in _dedupe_keep_order([exported_id, exported_id.replace("_", "-"), api_card_id]):
        urls.extend(by_id.get(_norm_key(key), []))

    if name:
        urls.extend(by_name.get(_norm_key(name), []))

    # 3. Semantic fallback for PTCGL's non-public basic-energy export IDs.
    if name:
        urls.extend(_BASIC_ENERGY_NAME_FALLBACKS.get(_norm_key(name), []))

    # 4. PokémonTCG convention fallbacks.
    if api_set_id and number:
        urls.extend(_pokemontcg_candidates(api_set_id, number))

    if set_id and number and api_set_id != set_id:
        urls.extend(_pokemontcg_candidates(set_id, number))

    # 5. ScryDex fallback again after convention in case something was skipped.
    if api_card_id and not api_card_id.startswith("mee-"):
        urls.extend(_scrydex_candidates(api_card_id))

    return _dedupe_keep_order(urls)


def best_image_url_for_card_ref(card: CardRef | None) -> str:
    urls = candidate_image_urls_for_card_ref(card)
    return urls[0] if urls else ""


def image_url_for_card_ref(card: CardRef | None) -> str:
    return best_image_url_for_card_ref(card)
