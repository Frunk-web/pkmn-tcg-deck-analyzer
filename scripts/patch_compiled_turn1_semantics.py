from __future__ import annotations

"""
Generic Turn-1 runtime semantics metadata patcher.

This script does not change the original compiled steps. It adds a
`turn1_runtime` block that a planner/simulator can consume without guessing from
card names. It is intentionally pattern-based and text-grounded:

- draw N cards = random draw from current deck
- search deck = deterministic selection from deck subject to filter/prizes
- search + shuffle + put on top = topdeck setup for later draws
- attach Basic Energy from hand + "if attached, draw" = conditional draw
- ability requirements such as Active Spot / Bench / play from hand to Bench
- once-per-turn and named/shared ability limits
- Supporter one-per-turn and going-first restriction metadata

It is safe to run repeatedly: it overwrites/merges the metadata but does not
mutate original compiled steps.
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

VERSION = "turn1_generic_semantics_v0_1"


# -----------------------------
# Normalization helpers
# -----------------------------

def norm_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def lower(x: Any) -> str:
    return norm_text(x).lower()


def source(effect: Dict[str, Any]) -> Dict[str, Any]:
    return effect.get("source") if isinstance(effect.get("source"), dict) else {}


def source_name(effect: Dict[str, Any]) -> str:
    return norm_text(source(effect).get("name") or "")


def source_text(effect: Dict[str, Any]) -> str:
    return norm_text(source(effect).get("text") or "")


def effect_blob(effect: Dict[str, Any]) -> str:
    parts = [
        source_name(effect),
        source(effect).get("ability_type") or "",
        source_text(effect),
        effect.get("kind") or "",
    ]
    for step in effect.get("steps") or []:
        if isinstance(step, dict):
            parts.append(step.get("source_text") or "")
            parts.append(step.get("op") or "")
    return norm_text("\n".join(str(p or "") for p in parts))


def card_name(card: Dict[str, Any]) -> str:
    for k in ("name", "card_name"):
        if card.get(k):
            return norm_text(card.get(k))
    return ""


def card_supertype(card: Dict[str, Any]) -> str:
    return norm_text(card.get("supertype") or card.get("card_supertype") or "")


def card_subtypes(card: Dict[str, Any]) -> List[str]:
    raw = card.get("subtypes") or card.get("card_subtypes") or []
    if isinstance(raw, str):
        # Some CSV-derived JSON fields store list-like text.
        return [s.strip(" '\"[]") for s in re.split(r"[,;]", raw) if s.strip(" '\"[]")]
    if isinstance(raw, list):
        return [norm_text(x) for x in raw]
    return []


def is_supporter_card(card: Dict[str, Any], effect: Dict[str, Any]) -> bool:
    subs = {s.lower() for s in card_subtypes(card)}
    if "supporter" in subs:
        return True
    txt = lower(source_text(effect))
    # The global rule alone is not the action effect, but it identifies the card
    # as Supporter if card metadata is missing.
    if "you may play only 1 supporter card during your turn" in txt:
        return True
    return False


def steps(effect: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [s for s in (effect.get("steps") or []) if isinstance(s, dict)]


def step_ops(effect: Dict[str, Any]) -> List[str]:
    return [str(s.get("op") or "") for s in steps(effect)]


def amount_value(x: Any, default: Optional[int] = None) -> Optional[int]:
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, dict):
        if x.get("mode") == "exact" and isinstance(x.get("value"), int):
            return int(x["value"])
        if isinstance(x.get("value"), int):
            return int(x["value"])
    if isinstance(x, str):
        m = re.search(r"\d+", x)
        if m:
            return int(m.group(0))
    return default


def first_amount_from_step(step: Dict[str, Any], default: int = 1) -> int:
    for key in ("amount", "count", "number", "n"):
        if key in step:
            val = amount_value(step.get(key), None)
            if val is not None:
                return val
    return default


# -----------------------------
# Runtime metadata helpers
# -----------------------------

def runtime(effect: Dict[str, Any]) -> Dict[str, Any]:
    rt = effect.setdefault("turn1_runtime", {})
    rt.setdefault("version", VERSION)
    rt.setdefault("patches", [])
    return rt


def add_patch(effect: Dict[str, Any], patch_name: str, payload: Dict[str, Any]) -> None:
    rt = runtime(effect)
    # Preserve existing keys unless this patch deliberately updates them.
    for k, v in payload.items():
        if isinstance(v, dict) and isinstance(rt.get(k), dict):
            merged = dict(rt[k])
            merged.update(v)
            rt[k] = merged
        elif isinstance(v, list) and isinstance(rt.get(k), list):
            # For runtime_steps, replace rather than append if this patch gives a
            # more exact sequence. Other lists are unioned by repr.
            if k == "runtime_steps":
                rt[k] = v
            else:
                seen = {json.dumps(x, sort_keys=True, ensure_ascii=False) for x in rt[k]}
                for item in v:
                    s = json.dumps(item, sort_keys=True, ensure_ascii=False)
                    if s not in seen:
                        rt[k].append(item)
                        seen.add(s)
        else:
            rt[k] = v
    if patch_name not in rt["patches"]:
        rt["patches"].append(patch_name)
    rt["runtime_support"] = rt.get("runtime_support") or "structured_from_compiled_text"


def add_playability(effect: Dict[str, Any], key: str, value: Any) -> None:
    rt = runtime(effect)
    rt.setdefault("playability", {})[key] = value


def add_usage_limit(effect: Dict[str, Any], payload: Dict[str, Any]) -> None:
    rt = runtime(effect)
    rt.setdefault("usage_limit", {}).update(payload)


def add_note(effect: Dict[str, Any], note: str) -> None:
    rt = runtime(effect)
    rt.setdefault("notes", [])
    if note not in rt["notes"]:
        rt["notes"].append(note)


# -----------------------------
# Text pattern inference
# -----------------------------

def infer_ability_playability(effect: Dict[str, Any]) -> Dict[str, Any]:
    txt_l = lower(source_text(effect))
    out: Dict[str, Any] = {}
    if "active spot" in txt_l:
        # Most ability text says "if this Pokémon is in the Active Spot".
        out["requires_source_zone"] = "active"
    elif "on your bench" in txt_l and "as long as" not in txt_l:
        out["requires_source_zone"] = "bench"
    elif effect.get("kind") == "ability_activated":
        out["requires_source_zone"] = "in_play"

    if "when you play this pokémon from your hand onto your bench" in txt_l or "when you play this pokemon from your hand onto your bench" in txt_l:
        out["trigger"] = {
            "event": "play_from_hand_to_bench",
            "source_card": "this_pokemon",
        }
        out["requires_source_zone"] = "bench"
    return out


def infer_usage_limit(effect: Dict[str, Any]) -> Dict[str, Any]:
    txt = source_text(effect)
    txt_l = txt.lower()
    name = source_name(effect)
    out: Dict[str, Any] = {}

    if "once during your turn" in txt_l:
        out.update({"scope": "turn", "max_uses": 1})
        if name:
            out.setdefault("shared_name", name)

    # You can't use more than 1 Run Errand Ability each turn.
    m = re.search(r"You can'?t use more than 1 ([A-Za-z0-9 '\-]+?) Ability each turn", txt, flags=re.I)
    if m:
        out.update({"scope": "turn", "max_uses": 1, "shared_name": norm_text(m.group(1))})

    # You can't use more than 1 Ability that has "Last-Ditch" in its name each turn.
    m = re.search(r"more than 1 Ability that has [\"“]([^\"”]+)[\"”] in its name each turn", txt, flags=re.I)
    if m:
        out.update({"scope": "turn", "max_uses": 1, "shared_name_prefix": norm_text(m.group(1))})

    return out


def infer_energy_filter_from_text(txt: str) -> Dict[str, Any]:
    txt_l = txt.lower()
    filt: Dict[str, Any] = {"supertype": "Energy"}
    if "basic" in txt_l:
        filt["basic"] = True
    for typ in ["grass", "fire", "water", "lightning", "psychic", "fighting", "darkness", "metal"]:
        if typ in txt_l:
            filt["energy_type"] = typ.capitalize()
            break
    return filt


def infer_search_filter_from_text(txt: str, step_filter: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    txt_l = txt.lower()
    if step_filter and not step_filter.get("from_text"):
        return dict(step_filter)
    filt: Dict[str, Any] = {}
    if "any card" in txt_l or "for 2 cards" in txt_l or "for a card" in txt_l:
        filt["any_card"] = True
    if "supporter" in txt_l:
        filt.update({"supertype": "Trainer", "subtype": "Supporter"})
    elif "item card" in txt_l:
        filt.update({"supertype": "Trainer", "subtype": "Item"})
    elif "stadium" in txt_l:
        filt.update({"supertype": "Trainer", "subtype": "Stadium"})
    elif "basic pokémon" in txt_l or "basic pokemon" in txt_l:
        filt.update({"supertype": "Pokémon", "stage": "Basic"})
    elif "pokémon" in txt_l or "pokemon" in txt_l:
        filt.update({"supertype": "Pokémon"})
    elif "energy" in txt_l:
        filt.update(infer_energy_filter_from_text(txt))
    return filt or (dict(step_filter or {}) if step_filter else {"from_text": True})


def is_topdeck_setup(effect: Dict[str, Any]) -> bool:
    txt_l = lower(source_text(effect))
    ops = set(step_ops(effect))
    return (
        "search your deck" in txt_l
        and "shuffle your deck" in txt_l
        and "on top" in txt_l
        and ("any order" in txt_l or "in any order" in txt_l)
        and "search_deck" in ops
        and "shuffle_deck" in ops
    )


def topdeck_setup_count(effect: Dict[str, Any]) -> int:
    for step in steps(effect):
        if step.get("op") == "search_deck":
            amt = amount_value(step.get("amount"), None)
            if amt is not None:
                return amt
    m = re.search(r"for (\d+) cards?", lower(source_text(effect)))
    return int(m.group(1)) if m else 1


# -----------------------------
# Effect patching
# -----------------------------

def patch_effect(card: Dict[str, Any], effect: Dict[str, Any], counts: Counter, examples: List[Dict[str, Any]]) -> None:
    name = source_name(effect)
    txt = source_text(effect)
    txt_l = txt.lower()
    kind = str(effect.get("kind") or "")
    ops = step_ops(effect)
    flat_steps = steps(effect)

    before = set(runtime(effect).get("patches", [])) if "turn1_runtime" in effect else set()

    # Baseline classification.
    if kind == "ability_activated":
        add_patch(effect, "generic_ability_classification", {
            "effect_type": "ability",
            "action_name": name or "Ability",
            "does_not_end_turn": True,
        })
        for k, v in infer_ability_playability(effect).items():
            if k == "trigger":
                runtime(effect)["trigger"] = v
            else:
                add_playability(effect, k, v)
        ul = infer_usage_limit(effect)
        if ul:
            add_usage_limit(effect, ul)
        counts["generic_ability_classification"] += 1

    # Supporters / Supporter global rule.
    if is_supporter_card(card, effect) or "you may play only 1 supporter card during your turn" in txt_l:
        add_patch(effect, "supporter_turn_rule", {
            "effect_type": "supporter" if kind == "trainer_rule" else runtime(effect).get("effect_type", "supporter_rule"),
            "supporter": True,
            "playability": {
                "blocked_on_first_player_first_turn": True,
                "requires_supporter_not_used": True,
            },
            "usage_limit": {"scope": "turn", "max_uses": 1, "shared_rule": "supporter"},
        })
        counts["supporter_turn_rule"] += 1

    # Generic draw effects: draw N is random from current deck unless topdeck_setup exists.
    draw_runtime_steps: List[Dict[str, Any]] = []
    for step in flat_steps:
        if step.get("op") == "draw_cards":
            n = first_amount_from_step(step, default=1)
            draw_runtime_steps.append({
                "op": "draw_cards",
                "amount": n,
                "draw_behavior": "random_from_current_deck_unless_deck_order_known",
                "random": True,
            })
    if draw_runtime_steps:
        add_patch(effect, "generic_random_draw", {
            "draw_behavior": "random_from_current_deck_unless_deck_order_known",
            "runtime_draw_steps": draw_runtime_steps,
        })
        counts["generic_random_draw"] += 1

    # Search effects: deterministic choice from deck, not a random hit.
    search_runtime_steps: List[Dict[str, Any]] = []
    for step in flat_steps:
        if step.get("op") == "search_deck":
            amt = amount_value(step.get("amount"), None) or first_amount_from_step(step, 1)
            search_runtime_steps.append({
                "op": "search_deck",
                "amount": amt,
                "filter": infer_search_filter_from_text(txt, step.get("filter") if isinstance(step.get("filter"), dict) else None),
                "destination": step.get("destination") or "hand",
                "reveal": bool(step.get("reveal", False)),
                "choice_behavior": "player_selects_valid_cards_from_current_deck",
                "random": False,
            })
    if search_runtime_steps:
        add_patch(effect, "generic_deck_search", {
            "search_behavior": "deterministic_choice_from_current_deck_subject_to_filter",
            "runtime_search_steps": search_runtime_steps,
        })
        counts["generic_deck_search"] += 1

    # Topdeck setup for later draw effects (e.g. Ciphermaniac-like text). This is intentionally generic.
    if is_topdeck_setup(effect):
        n = topdeck_setup_count(effect)
        add_patch(effect, "generic_topdeck_setup", {
            "topdeck_setup": {
                "cards": n,
                "selected_from_deck": True,
                "shuffle_before_placing": True,
                "ordering": "self_choice",
                "future_draws_can_be_deterministic_for_n_cards": n,
            },
            "runtime_steps": [
                {"op": "search_deck", "amount": n, "filter": {"any_card": True}, "destination": "temporary.selection", "random": False},
                {"op": "shuffle_deck"},
                {"op": "move_cards", "source": "temporary.selection", "destination": "deck.top", "ordering": "self_choice"},
            ],
        })
        counts["generic_topdeck_setup"] += 1

    # Attach from hand, then conditional draw if attachment succeeded.
    has_attach = any(s.get("op") == "attach_card" for s in flat_steps)
    has_draw = any(s.get("op") == "draw_cards" for s in flat_steps)
    if has_attach and has_draw and "if you attached" in txt_l:
        attach_step = next((s for s in flat_steps if s.get("op") == "attach_card"), {})
        draw_step = next((s for s in flat_steps if s.get("op") == "draw_cards"), {})
        draw_n = first_amount_from_step(draw_step, 1)
        add_patch(effect, "conditional_attach_then_draw", {
            "conditional_execution": True,
            "runtime_steps": [
                {
                    "op": "attach_card",
                    "source_zone": "hand",
                    "target": "source_pokemon",
                    "filter": infer_energy_filter_from_text(txt),
                    "required_for_followup": True,
                    "result_id": "attach_succeeded",
                },
                {
                    "op": "conditional",
                    "condition_ref": "attach_succeeded",
                    "then": [{"op": "draw_cards", "amount": draw_n, "random": True}],
                },
            ],
        })
        counts["conditional_attach_then_draw"] += 1

    # Bench-from-hand trigger.
    if "when you play this pokémon from your hand onto your bench" in txt_l or "when you play this pokemon from your hand onto your bench" in txt_l:
        rt_steps = []
        # Preserve search if present.
        if search_runtime_steps:
            rt_steps.extend(search_runtime_steps)
            if "shuffle_deck" in ops:
                rt_steps.append({"op": "shuffle_deck"})
        add_patch(effect, "bench_from_hand_trigger", {
            "effect_type": "triggered_ability",
            "trigger": {"event": "play_from_hand_to_bench", "source_card": "this_pokemon"},
            "does_not_end_turn": True,
            "runtime_steps": rt_steps or runtime(effect).get("runtime_steps", []),
        })
        counts["bench_from_hand_trigger"] += 1

    # Semantic fallback handling: if we added structured runtime, mark it as usable by runtime; otherwise warn.
    if any(s.get("op") == "semantic_ir_marker" for s in flat_steps):
        rt = runtime(effect)
        if len(rt.get("patches", [])) > 1 or any(p != "semantic_ir_marker_notice" for p in rt.get("patches", [])):
            add_patch(effect, "semantic_ir_structured_override", {
                "semantic_ir_marker_present": True,
                "runtime_support": "structured_override_from_text_patterns",
                "exact_runtime_support": True,
            })
            counts["semantic_ir_structured_override"] += 1
        else:
            add_patch(effect, "semantic_ir_marker_notice", {
                "semantic_ir_marker_present": True,
                "runtime_support": "not_executable_without_more_structure",
                "exact_runtime_support": False,
            })
            counts["semantic_ir_marker_notice"] += 1

    after = set(runtime(effect).get("patches", [])) if "turn1_runtime" in effect else set()
    added = sorted(after - before)
    if added:
        examples.append({
            "card": card_name(card),
            "effect": name,
            "kind": kind,
            "patches_added": added,
            "ops": ",".join(ops),
            "text": txt[:220],
        })


def get_cards(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        for key in ("compiled_cards", "cards", "data"):
            if isinstance(raw.get(key), list):
                return raw[key]
    if isinstance(raw, list):
        return raw
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--examples", type=int, default=80)
    args = ap.parse_args()

    raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cards = get_cards(raw)
    counts: Counter = Counter()
    examples: List[Dict[str, Any]] = []

    for card in cards:
        if not isinstance(card, dict):
            continue
        for effect in card.get("compiled_effects", []) or []:
            if isinstance(effect, dict):
                patch_effect(card, effect, counts, examples)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    # High-level audit by patch name.
    effects_with_runtime = 0
    semantic_runtime_overrides = 0
    for card in cards:
        for effect in card.get("compiled_effects", []) or []:
            if isinstance(effect, dict) and effect.get("turn1_runtime"):
                effects_with_runtime += 1
                if effect["turn1_runtime"].get("runtime_support") == "structured_override_from_text_patterns":
                    semantic_runtime_overrides += 1

    print(json.dumps({
        "version": VERSION,
        "input": args.input,
        "output": args.output,
        "cards": len(cards),
        "effects_with_turn1_runtime": effects_with_runtime,
        "semantic_runtime_overrides": semantic_runtime_overrides,
        "patched_effect_counts": dict(counts),
        "examples": examples[: max(0, args.examples)],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
