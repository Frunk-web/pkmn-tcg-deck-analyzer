import json
from pathlib import Path

p = Path("data/compiled_cards/auto/compiled_cards_all.json")
raw = json.loads(p.read_text(encoding="utf-8"))

def find_card_lists(obj, path="root"):
    found = []
    if isinstance(obj, list):
        card_like = [
            x for x in obj
            if isinstance(x, dict) and (
                "compiled_effects" in x
                or "card_id" in x
                or "name" in x
                or "unparsed_effects" in x
            )
        ]
        if card_like:
            found.append((path, obj, len(card_like), len(obj)))
        for i, x in enumerate(obj[:20]):
            found.extend(find_card_lists(x, f"{path}[{i}]"))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            found.extend(find_card_lists(v, f"{path}.{k}"))
    return found

candidates = find_card_lists(raw)
if not candidates:
    raise SystemExit("No card-like list found.")

path, cards, card_like_count, list_len = max(candidates, key=lambda x: x[3])
print("using_card_list_path:", path)
print("list_len:", list_len)
print("card_like_count:", card_like_count)

def has_semantic_marker(obj):
    if isinstance(obj, str):
        return obj == "semantic_ir_marker"
    if isinstance(obj, dict):
        if obj.get("op") == "semantic_ir_marker":
            return True
        return any(has_semantic_marker(v) for v in obj.values())
    if isinstance(obj, list):
        return any(has_semantic_marker(x) for x in obj)
    return False

def has_nonempty_unparsed(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if "unparsed" in str(k).lower():
                if isinstance(v, (list, dict)) and len(v) > 0:
                    return True
                if isinstance(v, str) and v.strip():
                    return True
            if has_nonempty_unparsed(v):
                return True
    elif isinstance(obj, list):
        return any(has_nonempty_unparsed(x) for x in obj)
    return False

cards_total = 0
complete_cards = 0
partial_cards = 0
cards_with_semantic_ir = 0
complete_cards_with_semantic_ir = 0
effects_total = 0
effects_with_semantic_ir = 0

for card in cards:
    if not isinstance(card, dict):
        continue

    cards_total += 1
    card_has_semantic = has_semantic_marker(card)
    card_is_complete = not has_nonempty_unparsed(card)

    if card_is_complete:
        complete_cards += 1
    else:
        partial_cards += 1

    if card_has_semantic:
        cards_with_semantic_ir += 1
        if card_is_complete:
            complete_cards_with_semantic_ir += 1

    effects = card.get("compiled_effects", [])
    if isinstance(effects, list):
        for effect in effects:
            effects_total += 1
            if has_semantic_marker(effect):
                effects_with_semantic_ir += 1

precise_complete_estimate = complete_cards - complete_cards_with_semantic_ir

print({
    "cards_total": cards_total,
    "complete_cards_by_unparsed_check": complete_cards,
    "partial_cards_by_unparsed_check": partial_cards,
    "cards_with_semantic_ir_marker": cards_with_semantic_ir,
    "complete_cards_with_semantic_ir_marker": complete_cards_with_semantic_ir,
    "precise_or_legacy_complete_cards_estimate": precise_complete_estimate,
    "semantic_ir_complete_cards_estimate": complete_cards_with_semantic_ir,
    "effects_total": effects_total,
    "effects_with_semantic_ir_marker": effects_with_semantic_ir,
    "precise_complete_rate_estimate": round(precise_complete_estimate / cards_total, 4),
    "semantic_or_precise_complete_rate": round(complete_cards / cards_total, 4),
})
