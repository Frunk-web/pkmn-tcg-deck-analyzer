from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
import urllib.request
from typing import Any

from .models import CardRef


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _split_exported_id(exported_id: str) -> tuple[str, str]:
    raw = str(exported_id or "").strip()
    if "_" not in raw:
        return raw, ""
    set_id, number = raw.rsplit("_", 1)
    return set_id.strip(), number.strip()


def ptcgl_set_id_to_public_set_id(set_id: str) -> str:
    """
    Convert PTCG Live export set IDs to public Pokémon TCG API-style set IDs
    when the naming convention differs.

    Example:
      sv6-5 -> sv6pt5
      sv8-5 -> sv8pt5
    """
    raw = str(set_id or "").strip()

    m = re.fullmatch(r"(sv\d+)-5", raw)
    if m:
        return f"{m.group(1)}pt5"

    m = re.fullmatch(r"(swsh\d+)-5", raw)
    if m:
        return f"{m.group(1)}pt5"

    return raw


def exported_id_to_api_card_id(exported_id: str) -> str:
    set_id, number = _split_exported_id(exported_id)
    if not set_id or not number:
        return str(exported_id or "").strip()

    public_set_id = ptcgl_set_id_to_public_set_id(set_id)
    return f"{public_set_id}-{number}"


def _candidate_set_ids(set_id: str) -> list[str]:
    mapped = ptcgl_set_id_to_public_set_id(set_id)

    out = []
    for value in [mapped, set_id]:
        if value and value not in out:
            out.append(value)

    return out


def candidate_image_urls_for_exported_id(exported_id: str, *, hires: bool = True) -> list[str]:
    set_id, number = _split_exported_id(exported_id)
    if not set_id or not number:
        return []

    suffix = "_hires" if hires else ""

    urls = []
    for public_set_id in _candidate_set_ids(set_id):
        urls.append(f"https://images.pokemontcg.io/{public_set_id}/{number}{suffix}.png")

    return urls


def exported_id_to_image_url(exported_id: str, *, hires: bool = True) -> str:
    urls = candidate_image_urls_for_exported_id(exported_id, hires=hires)
    return urls[0] if urls else ""


def _normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _possible_local_card_files() -> list[Path]:
    root = _repo_root()
    return [
        root / "data" / "all_cards.csv",
        root / "data" / "cards.csv",
        root / "data" / "card_index.csv",
    ]


@lru_cache(maxsize=1)
def _local_card_rows() -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except Exception:
        return []

    for path in _possible_local_card_files():
        if not path.exists():
            continue

        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        rows = []
        for row in df.to_dict("records"):
            rows.append({str(k): v for k, v in row.items()})
        return rows

    return []


def _row_value(row: dict[str, Any], candidates: list[str]) -> str:
    lower_map = {k.lower(): k for k in row.keys()}

    for candidate in candidates:
        key = lower_map.get(candidate.lower())
        if key is not None:
            val = row.get(key)
            if val is not None and str(val).strip() and str(val).lower() != "nan":
                return str(val).strip()

    return ""


def _row_image_url(row: dict[str, Any]) -> str:
    preferred = [
        "image_large_url",
        "images.large",
        "images_large",
        "large_image_url",
        "image_url",
        "images.small",
        "images_small",
        "small_image_url",
    ]

    direct = _row_value(row, preferred)
    if direct.startswith("http"):
        return direct

    for key, value in row.items():
        key_l = str(key).lower()
        val = str(value or "").strip()
        if "image" in key_l and val.startswith("http"):
            return val

    return ""


def _local_image_url_for_card(card: CardRef) -> str:
    if card is None or card.unknown:
        return ""

    set_id, number = _split_exported_id(card.exported_id)
    api_ids = []

    if set_id and number:
        api_ids.append(f"{ptcgl_set_id_to_public_set_id(set_id)}-{number}")
        api_ids.append(f"{set_id}-{number}")

    api_ids = [x.lower() for x in dict.fromkeys(api_ids)]

    card_name = _normalize_text(card.name)

    rows = _local_card_rows()
    if not rows:
        return ""

    id_cols = ["id", "card_id", "representative_card_id"]
    set_cols = ["set.id", "set_id", "set"]
    number_cols = ["number", "collector_number"]
    name_cols = ["name", "card_name"]

    # Best: exact local card ID match.
    for row in rows:
        row_id = _normalize_text(_row_value(row, id_cols))
        if row_id and row_id in api_ids:
            img = _row_image_url(row)
            if img:
                return img

    # Good: same set + number + name.
    public_set_id = ptcgl_set_id_to_public_set_id(set_id)
    for row in rows:
        row_set = _normalize_text(_row_value(row, set_cols))
        row_number = _normalize_text(_row_value(row, number_cols))
        row_name = _normalize_text(_row_value(row, name_cols))

        if (
            row_name == card_name
            and row_number == _normalize_text(number)
            and row_set in {_normalize_text(set_id), _normalize_text(public_set_id)}
        ):
            img = _row_image_url(row)
            if img:
                return img

    # Fallback: exact name match. This may pick a different printing, but it is
    # better than showing a card back for cards missing from public image paths.
    for row in rows:
        row_name = _normalize_text(_row_value(row, name_cols))
        if row_name == card_name:
            img = _row_image_url(row)
            if img:
                return img

    return ""


def candidate_image_urls_for_card_ref(card: CardRef | None) -> list[str]:
    if card is None or card.unknown or not card.exported_id:
        return []

    out: list[str] = []

    local = _local_image_url_for_card(card)
    if local:
        out.append(local)

    for url in candidate_image_urls_for_exported_id(card.exported_id):
        if url not in out:
            out.append(url)

    return out


@lru_cache(maxsize=4096)
def _image_url_exists(url: str) -> bool:
    """
    Return True when an image URL appears loadable.

    This avoids rendering broken first-choice URLs and showing blank/card-back
    placeholders when a later fallback URL would have worked.
    """
    raw = str(url or "").strip()
    if not raw.startswith("http"):
        return False

    try:
        req = urllib.request.Request(
            raw,
            method="HEAD",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            status = getattr(response, "status", 200)
            content_type = response.headers.get("content-type", "")
            return 200 <= int(status) < 400 and "image" in content_type.lower()
    except Exception:
        pass

    # Some image hosts reject HEAD. Fall back to a tiny GET.
    try:
        req = urllib.request.Request(
            raw,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Range": "bytes=0-128",
            },
        )
        with urllib.request.urlopen(req, timeout=4) as response:
            status = getattr(response, "status", 200)
            content_type = response.headers.get("content-type", "")
            return 200 <= int(status) < 400 and "image" in content_type.lower()
    except Exception:
        return False


def best_image_url_for_card_ref(card: CardRef | None) -> str:
    """
    Return the best known image URL without server-side rejection.

    Important: the Card Gallery succeeds by letting the browser load known image
    URLs directly. Server-side HEAD/GET checks can fail on deployed hosts even
    when the image works in the browser, so Game Review should use the same
    gallery-style behavior.
    """
    urls = candidate_image_urls_for_card_ref(card)
    return urls[0] if urls else ""


def image_url_for_card_ref(card: CardRef | None) -> str:
    return best_image_url_for_card_ref(card)
