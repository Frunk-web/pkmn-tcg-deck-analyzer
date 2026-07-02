from __future__ import annotations

from .models import CardRef


_LIVE_SET_ID_TO_API_SET_ID = {
    "sv6-5": "sv6pt5",
    "sv8-5": "sv8pt5",
}


# These PTCG Live energy IDs are not normal PokémonTCG API card IDs.
# Use stable public Energy print images that work in normal browser rendering.
_EXPORTED_ID_IMAGE_OVERRIDES = {
    "mee_1": [
        "https://images.pokemontcg.io/sv2/278.png",
        "https://images.pokemontcg.io/sv2/278_hires.png",
    ],
    "mee_2": [
        "https://images.pokemontcg.io/sv3/230.png",
        "https://images.pokemontcg.io/sv3/230_hires.png",
    ],
    "mee_3": [
        "https://images.pokemontcg.io/sv2/279.png",
        "https://images.pokemontcg.io/sv2/279_hires.png",
    ],
    "mee_4": [
        "https://images.pokemontcg.io/sv1/257.png",
        "https://images.pokemontcg.io/sv1/257_hires.png",
    ],
    "mee_5": [
        "https://images.pokemontcg.io/sv3pt5/207.png",
        "https://images.pokemontcg.io/sv3pt5/207_hires.png",
    ],
    "mee_6": [
        "https://images.pokemontcg.io/sv1/258.png",
        "https://images.pokemontcg.io/sv1/258_hires.png",
    ],
}


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen = set()
    out = []

    for value in values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        out.append(clean)

    return out


def _split_exported_id(exported_id: str) -> tuple[str, str]:
    raw = str(exported_id or "").strip()
    if "_" not in raw:
        return raw, ""

    set_id, number = raw.rsplit("_", 1)
    return set_id.strip(), number.strip()


def _api_set_id_for_live_set_id(set_id: str) -> str:
    clean = str(set_id or "").strip()
    return _LIVE_SET_ID_TO_API_SET_ID.get(clean, clean)


def exported_id_to_api_card_id(exported_id: str) -> str:
    set_id, number = _split_exported_id(exported_id)
    if not set_id or not number:
        return str(exported_id or "").strip()

    return f"{_api_set_id_for_live_set_id(set_id)}-{number}"


def exported_id_to_image_url(exported_id: str, *, hires: bool = False) -> str:
    set_id, number = _split_exported_id(exported_id)
    if not set_id or not number:
        return ""

    api_set_id = _api_set_id_for_live_set_id(set_id)
    suffix = "_hires" if hires else ""
    return f"https://images.pokemontcg.io/{api_set_id}/{number}{suffix}.png"


def candidate_image_urls_for_card_ref(card: CardRef | None) -> list[str]:
    if card is None or getattr(card, "unknown", False):
        return []

    exported_id = str(getattr(card, "exported_id", "") or "").strip()
    if not exported_id:
        return []

    exported_key = exported_id.lower()
    set_id, number = _split_exported_id(exported_id)

    urls: list[str] = []

    # 1. Explicit known-good overrides first.
    urls.extend(_EXPORTED_ID_IMAGE_OVERRIDES.get(exported_key, []))

    if set_id and number:
        api_set_id = _api_set_id_for_live_set_id(set_id)
        api_card_id = f"{api_set_id}-{number}"

        # 2. PokémonTCG image convention.
        # Prefer hires first for newer/cached gallery-style card images.
        urls.append(f"https://images.pokemontcg.io/{api_set_id}/{number}_hires.png")
        urls.append(f"https://images.pokemontcg.io/{api_set_id}/{number}.png")

        # 3. Original live set ID convention, useful if a weird set ID exists as-is.
        if api_set_id != set_id:
            urls.append(f"https://images.pokemontcg.io/{set_id}/{number}_hires.png")
            urls.append(f"https://images.pokemontcg.io/{set_id}/{number}.png")

        # 4. ScryDex fallback for newer or weird ME cards.
        urls.append(f"https://images.scrydex.com/pokemon/{api_card_id}/large")
        urls.append(f"https://images.scrydex.com/pokemon/{api_card_id}/small")

    return _dedupe_keep_order(urls)


def best_image_url_for_card_ref(card: CardRef | None) -> str:
    urls = candidate_image_urls_for_card_ref(card)
    return urls[0] if urls else ""


def image_url_for_card_ref(card: CardRef | None) -> str:
    return best_image_url_for_card_ref(card)
