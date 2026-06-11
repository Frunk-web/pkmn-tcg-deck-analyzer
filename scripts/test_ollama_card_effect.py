"""
Explanation

This script tests whether a local Ollama model can extract structured Pokémon TCG
card effects from card text.

This version intentionally avoids hardcoding interpretation rules like:
- "Tera Pokémon means subtype Tera"
- "Supporter card means Trainer + Supporter"

Instead, it asks the model to extract what the card text literally says and stores
the original raw text with each effect.

The goal is to test whether the model can infer card relationships without us
feeding it too many assumptions.

Run from project root:

python scripts/test_ollama_card_effect.py
"""

import json
import re
import ollama


MODEL = "qwen3:4b"


EXAMPLE_CARDS = [
    {
        "name": "Tera Orb",
        "supertype": "Trainer",
        "subtypes": ["Item"],
        "text": "Search your deck for a Tera Pokémon, reveal it, and put it into your hand. Then, shuffle your deck.",
    },
    {
        "name": "Ultra Ball",
        "supertype": "Trainer",
        "subtypes": ["Item"],
        "text": "You can use this card only if you discard 2 other cards from your hand. Search your deck for a Pokémon, reveal it, and put it into your hand. Then, shuffle your deck.",
    },
    {
        "name": "Nest Ball",
        "supertype": "Trainer",
        "subtypes": ["Item"],
        "text": "Search your deck for a Basic Pokémon and put it onto your Bench. Then, shuffle your deck.",
    },
    {
        "name": "Pokégear 3.0",
        "supertype": "Trainer",
        "subtypes": ["Item"],
        "text": "Look at the top 7 cards of your deck. You may reveal a Supporter card you find there and put it into your hand. Shuffle the other cards back into your deck.",
    },
    {
        "name": "Professor's Research",
        "supertype": "Trainer",
        "subtypes": ["Supporter"],
        "text": "Discard your hand and draw 7 cards.",
    },
    {
        "name": "Buddy-Buddy Poffin",
        "supertype": "Trainer",
        "subtypes": ["Item"],
        "text": "Search your deck for up to 2 Basic Pokémon with 70 HP or less and put them onto your Bench. Then, shuffle your deck.",
    },
]


REQUIRED_TOP_LEVEL_KEYS = {
    "card_name",
    "has_relevant_effects",
    "effects",
    "overall_confidence",
    "needs_review",
    "review_notes",
}

REQUIRED_EFFECT_KEYS = {
    "effect_type",
    "source_zone",
    "destination_zone",
    "target_text",
    "target",
    "quantity",
    "look_at_count",
    "draw_count",
    "requires_discard",
    "discard_count",
    "puts_card_in_hand",
    "puts_card_on_bench",
    "shuffles_deck",
    "confidence",
    "needs_review",
    "notes",
    "raw_text",
}


def empty_target() -> dict:
    return {
        "supertype": None,
        "subtypes_include": [],
        "subtypes_exclude": [],
        "types_include": [],
        "types_exclude": [],
        "name_exact": None,
        "name_contains": [],
        "hp_max": None,
        "hp_min": None,
        "custom_text": None,
    }


def normalize_json_text(text: str) -> str:
    """
    Some local models still wrap JSON in markdown fences.
    This removes fences if present.
    """

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text.strip()).strip()

    return text


def build_prompt(card: dict) -> str:
    return f"""
You are extracting structured effects from Pokémon TCG card text.

Return only valid JSON. No markdown. No explanation.

Do not rely on external card knowledge. Use only the supplied text.

Your job is to identify effects that affect card access, deck search, drawing,
looking at cards, moving cards between zones, or recovering cards.

Use this exact JSON structure:

{{
  "card_name": "string",
  "has_relevant_effects": true,
  "effects": [
    {{
      "effect_type": "search | draw | look_top_cards | recover_from_discard | move_card | attach_energy | other",
      "source_zone": "deck | hand | discard | top_deck | prize | play | unknown | null",
      "destination_zone": "hand | bench | active | discard | deck | attached | play | unknown | null",
      "target_text": "literal phrase describing the target, or null",
      "target": {{
        "supertype": "Pokémon | Trainer | Energy | null",
        "subtypes_include": [],
        "subtypes_exclude": [],
        "types_include": [],
        "types_exclude": [],
        "name_exact": null,
        "name_contains": [],
        "hp_max": null,
        "hp_min": null,
        "custom_text": null
      }},
      "quantity": null,
      "look_at_count": null,
      "draw_count": null,
      "requires_discard": false,
      "discard_count": null,
      "puts_card_in_hand": false,
      "puts_card_on_bench": false,
      "shuffles_deck": false,
      "confidence": 0.0,
      "needs_review": false,
      "notes": "brief explanation",
      "raw_text": "exact text sentence or clause supporting this effect"
    }}
  ],
  "overall_confidence": 0.0,
  "needs_review": false,
  "review_notes": ""
}}

Guidelines:
- Prefer one effect object per meaningful gameplay effect.
- Do not create a separate effect only for shuffling. Instead set shuffles_deck=true
  on the related search/look effect.
- If the text says the player searches the deck, effect_type should usually be "search".
- If the text says the player draws cards, effect_type should usually be "draw".
- If the text says the player looks at the top N cards, effect_type should usually be "look_top_cards".
- If the effect puts a card into hand, set puts_card_in_hand=true.
- If the effect puts a Pokémon onto the Bench, set puts_card_on_bench=true.
- If there is a discard cost or discard requirement, set requires_discard=true.
- Use null when a field is unknown or not applicable.
- Use confidence from 0.0 to 1.0.
- Set needs_review=true if the target, cost, or effect is ambiguous.
- Keep target_text close to the exact card wording.
- Preserve the exact relevant card text in raw_text.

Card:
{json.dumps(card, indent=2, ensure_ascii=False)}
"""


def extract_card_effect(card: dict) -> dict:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You extract Pokémon TCG card text into strict JSON. "
                    "Return only JSON supported by the supplied card text."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(card),
            },
        ],
        options={
            "temperature": 0,
        },
        format="json",
    )

    content = response["message"]["content"]
    content = normalize_json_text(content)
    return json.loads(content)


def validate_and_clean_result(result: dict, card: dict) -> dict:
    """
    Makes the local model output safer for testing.

    This is not full production validation yet.
    It just:
    - restores missing top-level keys
    - restores missing effect keys
    - removes separate shuffle-only effects
    - forces card_name to match the input card
    """

    cleaned = {}

    for key in REQUIRED_TOP_LEVEL_KEYS:
        cleaned[key] = result.get(key)

    cleaned["card_name"] = card["name"]
    cleaned["has_relevant_effects"] = bool(
        result.get("has_relevant_effects", result.get("has_effect", False))
    )
    cleaned["overall_confidence"] = result.get("overall_confidence", 0.0)
    cleaned["needs_review"] = bool(result.get("needs_review", False))
    cleaned["review_notes"] = result.get("review_notes", "")

    effects = result.get("effects", [])
    if not isinstance(effects, list):
        effects = []

    cleaned_effects = []

    for effect in effects:
        if not isinstance(effect, dict):
            continue

        # Do not keep shuffle-only as a separate effect.
        if (
            effect.get("effect_type") in {"shuffle", "other"}
            and effect.get("shuffles_deck") is True
            and not effect.get("puts_card_in_hand")
            and not effect.get("puts_card_on_bench")
            and effect.get("destination_zone") == "deck"
        ):
            if cleaned_effects:
                cleaned_effects[-1]["shuffles_deck"] = True
            continue

        cleaned_effect = {}

        for key in REQUIRED_EFFECT_KEYS:
            cleaned_effect[key] = effect.get(key)

        cleaned_effect["effect_type"] = cleaned_effect["effect_type"] or "other"
        cleaned_effect["source_zone"] = cleaned_effect["source_zone"]
        cleaned_effect["destination_zone"] = cleaned_effect["destination_zone"]
        cleaned_effect["target_text"] = cleaned_effect["target_text"]
        cleaned_effect["target"] = cleaned_effect["target"] or empty_target()

        # Make sure target has all expected keys.
        base_target = empty_target()
        base_target.update(cleaned_effect["target"])
        cleaned_effect["target"] = base_target

        cleaned_effect["requires_discard"] = bool(cleaned_effect["requires_discard"])
        cleaned_effect["puts_card_in_hand"] = bool(cleaned_effect["puts_card_in_hand"])
        cleaned_effect["puts_card_on_bench"] = bool(cleaned_effect["puts_card_on_bench"])
        cleaned_effect["shuffles_deck"] = bool(cleaned_effect["shuffles_deck"])
        cleaned_effect["confidence"] = cleaned_effect["confidence"] or 0.0
        cleaned_effect["needs_review"] = bool(cleaned_effect["needs_review"])
        cleaned_effect["notes"] = cleaned_effect["notes"] or ""
        cleaned_effect["raw_text"] = cleaned_effect["raw_text"] or card.get("text", "")

        cleaned_effects.append(cleaned_effect)

    # If any effect mentions shuffling in raw text, attach it to the effect.
    full_text_lower = card.get("text", "").lower()
    if "shuffle your deck" in full_text_lower or "shuffle" in full_text_lower:
        for effect in cleaned_effects:
            if effect["effect_type"] in {"search", "look_top_cards"}:
                effect["shuffles_deck"] = True

    cleaned["effects"] = cleaned_effects
    cleaned["has_relevant_effects"] = len(cleaned_effects) > 0

    if cleaned_effects and cleaned["overall_confidence"] == 0.0:
        confidences = [
            effect.get("confidence", 0.0)
            for effect in cleaned_effects
            if isinstance(effect.get("confidence", 0.0), (int, float))
        ]
        cleaned["overall_confidence"] = max(confidences) if confidences else 0.0

    return cleaned


def main():
    for card in EXAMPLE_CARDS:
        print("=" * 80)
        print(f"Testing: {card['name']}")
        print("=" * 80)

        try:
            raw_result = extract_card_effect(card)
            cleaned_result = validate_and_clean_result(raw_result, card)
            print(json.dumps(cleaned_result, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"ERROR: {e}")

        print()


if __name__ == "__main__":
    main()