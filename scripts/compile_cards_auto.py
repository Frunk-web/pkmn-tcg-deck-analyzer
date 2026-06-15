from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT = Path("data/all_cards.csv")
DEFAULT_OUTPUT_DIR = Path("data/compiled_cards/auto")
DEFAULT_REPORT_DIR = Path("data/reports")


REQUIRED_COLUMNS = [
    "card_id",
    "name",
    "supertype",
    "subtypes",
    "types",
    "hp",
    "evolves_from",
    "rules",
    "abilities_text",
    "attacks_text",
    "combined_text",
    "set_id",
    "set_name",
    "set_series",
    "set_release_date",
    "number",
    "rarity",
    "regulation_mark",
    "legal_standard",
    "legal_expanded",
    "legal_unlimited",
    "image_small",
    "image_large",
    "raw_rules_json",
    "raw_abilities_json",
    "raw_attacks_json",
    "raw_card_json",
]


GLOBAL_RULE_PATTERNS = [
    # Modern and older Supporter boilerplate.
    re.compile(r"^you may play only 1 supporter card during your turn(?: \(before your attack\))?\.?$", re.I),
    re.compile(r"^you can play only one supporter card each turn\. when you play this card, put it next to your active pokémon\. when your turn ends, discard this card\.?$", re.I),
    re.compile(r"^you can play only 1 supporter card each turn\. when you play this card, put it next to your active pokémon\. when your turn ends, discard this card\.?$", re.I),

    # Item / Tool / Stadium boilerplate.
    re.compile(r"^you may play as many item cards as you like during your turn(?: \(before your attack\))?\.?$", re.I),
    re.compile(r"^you may play any number of item cards during your turn\.?$", re.I),
    re.compile(r"^attach a pokémon tool to 1 of your pokémon that doesn't already have a pokémon tool attached(?: to it)?\.?$", re.I),
    re.compile(r"^you may attach any number of pokémon tools to your pokémon during your turn\. you may attach only 1 pokémon tool to each pokémon, and it stays attached\.?$", re.I),
    re.compile(r"^this card stays in play when you play it\. discard this card if another stadium card comes into play(?:\. if another card with the same name is in play, you can't play this card)?\.?$", re.I),
    re.compile(r"^you may play only 1 stadium card during your turn\. put it next to the active spot, and discard it if another stadium comes into play\. a stadium with the same name can't be played\.?$", re.I),

    # Deck construction / special-rule boilerplate. These are stored as global/deck-building references.
    re.compile(r"^ace spec: you can't have more than 1 ace spec card in your deck\.?$", re.I),
    re.compile(r"^you can't have more than 1 ace spec card in your deck\.?$", re.I),
    re.compile(r"^radiant pokémon rule: you can't have more than 1 radiant pokémon in your deck\.?$", re.I),
    re.compile(r"^you can't have more than 1 pokémon star in your deck\.?$", re.I),
    re.compile(r"^you may have up to 4 basic pokémon cards in your deck with unown in their names\.?$", re.I),
    re.compile(r"^mega evolution rule: when 1 of your pokémon becomes a mega evolution pokémon, your turn ends\.?$", re.I),
    re.compile(r"^◇ \(prism star\) rule: you can't have more than 1 ◇ card with the same name in your deck\. if a ◇ card would go to the discard pile, put it in the lost zone instead\.?$", re.I),
]


def clean_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        value = value.strip()
        return value if value else None
    return value


def maybe_json_loads(value: Any) -> Any:
    value = clean_value(value)
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def split_pipe_list(value: Any) -> list[str]:
    value = clean_value(value)
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [part.strip() for part in str(value).split("|") if part.strip()]


def normalize_space(text: str) -> str:
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_text_for_signature(text: Any) -> str | None:
    value = clean_value(text)
    if value is None:
        return None
    return normalize_space(str(value))


def parse_raw_card(row: pd.Series) -> dict[str, Any]:
    raw = maybe_json_loads(row.get("raw_card_json"))
    return raw if isinstance(raw, dict) else {}


def normalize_attack_for_signature(attack: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": clean_value(attack.get("name")),
        "cost": attack.get("cost") or [],
        "convertedEnergyCost": attack.get("convertedEnergyCost"),
        "damage": clean_value(attack.get("damage")),
        "text": normalize_text_for_signature(attack.get("text")),
    }


def normalize_ability_for_signature(ability: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": clean_value(ability.get("name")),
        "type": clean_value(ability.get("type")),
        "text": normalize_text_for_signature(ability.get("text")),
    }


def card_signature(row: pd.Series) -> dict[str, Any]:
    raw = parse_raw_card(row)

    attacks = raw.get("attacks")
    abilities = raw.get("abilities")

    if not isinstance(attacks, list):
        attacks = maybe_json_loads(row.get("raw_attacks_json")) or []
    if not isinstance(abilities, list):
        abilities = maybe_json_loads(row.get("raw_abilities_json")) or []

    rules = raw.get("rules")
    if not isinstance(rules, list):
        rules = maybe_json_loads(row.get("raw_rules_json")) or split_pipe_list(row.get("rules"))

    return {
        "name": clean_value(row.get("name")),
        "supertype": clean_value(row.get("supertype")),
        "subtypes": split_pipe_list(row.get("subtypes")),
        "types": split_pipe_list(row.get("types")),
        "hp": clean_value(row.get("hp")),
        "evolvesFrom": clean_value(row.get("evolves_from")) or clean_value(raw.get("evolvesFrom")),
        "rules": [normalize_text_for_signature(x) for x in rules if normalize_text_for_signature(x)],
        "abilities": [
            normalize_ability_for_signature(x)
            for x in abilities
            if isinstance(x, dict)
        ],
        "attacks": [
            normalize_attack_for_signature(x)
            for x in attacks
            if isinstance(x, dict)
        ],
        "weaknesses": raw.get("weaknesses") or [],
        "resistances": raw.get("resistances") or [],
        "retreatCost": raw.get("retreatCost") or [],
        "convertedRetreatCost": raw.get("convertedRetreatCost"),
        "ancientTrait": raw.get("ancientTrait"),
    }


def effect_group_id(signature: dict[str, Any]) -> str:
    payload = json.dumps(signature, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "eg_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def card_priority(row: pd.Series) -> tuple[int, int, int, str]:
    supertype = str(row.get("supertype") or "").lower()
    subtypes = str(row.get("subtypes") or "").lower()
    abilities = str(row.get("abilities_text") or "").strip()
    attacks = str(row.get("attacks_text") or "").strip()
    date = str(row.get("set_release_date") or "")

    if supertype == "trainer":
        if "supporter" in subtypes:
            return (0, 0, 0, date)
        if "item" in subtypes:
            return (0, 1, 0, date)
        if "stadium" in subtypes:
            return (0, 2, 0, date)
        if "tool" in subtypes:
            return (0, 3, 0, date)
        return (0, 9, 0, date)
    if supertype == "energy":
        return (1, 0, 0, date)
    if abilities:
        return (2, 0, 0, date)
    if attacks:
        return (3, 0, 0, date)
    return (9, 0, 0, date)


def printing_summary(row: pd.Series) -> dict[str, Any]:
    return {
        "card_id": clean_value(row.get("card_id")),
        "name": clean_value(row.get("name")),
        "set": {
            "id": clean_value(row.get("set_id")),
            "name": clean_value(row.get("set_name")),
            "series": clean_value(row.get("set_series")),
            "release_date": clean_value(row.get("set_release_date")),
        },
        "number": clean_value(row.get("number")),
        "rarity": clean_value(row.get("rarity")),
        "regulation_mark": clean_value(row.get("regulation_mark")),
        "legalities": {
            "standard": clean_value(row.get("legal_standard")),
            "expanded": clean_value(row.get("legal_expanded")),
            "unlimited": clean_value(row.get("legal_unlimited")),
        },
        "images": {
            "small": clean_value(row.get("image_small")),
            "large": clean_value(row.get("image_large")),
        },
    }


def build_groups(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows_by_group: dict[str, list[pd.Series]] = {}
    signature_by_group: dict[str, dict[str, Any]] = {}

    for _, row in df.iterrows():
        sig = card_signature(row)
        gid = effect_group_id(sig)
        rows_by_group.setdefault(gid, []).append(row)
        signature_by_group[gid] = sig

    groups: list[dict[str, Any]] = []

    for gid, rows in rows_by_group.items():
        rows_sorted = sorted(rows, key=card_priority)
        rep = rows_sorted[0]

        groups.append({
            "effect_group_id": gid,
            "representative": rep,
            "signature": signature_by_group[gid],
            "same_effect_card_ids": [clean_value(r.get("card_id")) for r in rows_sorted],
            "same_effect_printings": [printing_summary(r) for r in rows_sorted],
        })

    groups.sort(key=lambda g: (
        card_priority(g["representative"]),
        str(g["representative"].get("name") or ""),
        str(g["representative"].get("card_id") or ""),
    ))

    return groups


def is_global_rule_line(text: str) -> bool:
    text = normalize_space(text)
    return any(pattern.match(text) for pattern in GLOBAL_RULE_PATTERNS)


def global_rule_id_for_text(text: str) -> str:
    t = normalize_space(text).lower()
    if "supporter" in t:
        return "play_supporter"
    if "item" in t:
        return "play_item"
    if "tool" in t:
        return "attach_pokemon_tool"
    if "stadium" in t:
        return "play_stadium"
    if "ace spec" in t:
        return "deck_construction_ace_spec"
    if "radiant pokémon" in t:
        return "deck_construction_radiant_pokemon"
    if "prism star" in t or "◇" in t:
        return "deck_construction_prism_star"
    if "pokémon star" in t:
        return "deck_construction_pokemon_star"
    if "mega evolution" in t:
        return "mega_evolution_turn_ends"
    if "unown" in t and "deck" in t:
        return "deck_construction_unown_exception"
    return "global_rule_reference"


def status_condition_step(condition: str, target: str, source_text: str) -> dict[str, Any]:
    return {
        "op": "apply_special_condition",
        "target": target,
        "condition": condition.capitalize(),
        "source_text": source_text,
    }


def delayed_modifier_step(modifier_id: str, target: str, modification: dict[str, Any], source_text: str) -> dict[str, Any]:
    return {
        "op": "register_continuous_modifier",
        "modifier_id": modifier_id,
        "target": target,
        "duration": {"until": "end_of_opponent_next_turn"},
        "modification": modification,
        "source_text": source_text,
    }


def amount_exact(value: int) -> dict[str, Any]:
    return {"mode": "exact", "value": value}


def amount_up_to(value: int) -> dict[str, Any]:
    return {"mode": "up_to", "value": value}


def raw_filter(text: str) -> dict[str, Any]:
    return {"raw_text": normalize_space(text)}


def parse_damage_value(damage: Any) -> dict[str, Any]:
    damage = clean_value(damage)
    if damage is None:
        return {"printed": None, "base": None, "modifier_symbol": None}
    s = str(damage)
    m = re.match(r"^(\d+)([+x×-])?$", s)
    if not m:
        return {"printed": s, "base": None, "modifier_symbol": None}
    return {"printed": s, "base": int(m.group(1)), "modifier_symbol": m.group(2)}


def compile_rule_box(text: str) -> list[dict[str, Any]] | None:
    t = normalize_space(text)

    m = re.search(r"opponent takes (\d+) Prize cards", t, re.I)
    if not m:
        return None

    # Keep the condition as raw text because Rule Box wording differs by era.
    return [{
        "op": "register_knockout_prize_rule",
        "owner": "self",
        "prizes_taken_by_opponent": int(m.group(1)),
        "condition": {"source_text": t},
    }]



# v0.21: template-driven strict compiler layer.
# These helpers intentionally keep the existing meaning of "complete": a text is
# marked complete only when it is represented by concrete structured steps. The
# goal is to cover repeated parameterized templates instead of adding one-off
# regexes for each card sentence.
def compile_template_driven_text(text: str, original: str, source_section: str) -> tuple[list[dict[str, Any]], list[str]] | None:
    t = normalize_space(text)
    t_rule = re.sub(r"^Rules:\s*", "", t, flags=re.I).strip()

    def step(op: str, **kwargs: Any) -> dict[str, Any]:
        payload = {"op": op, **kwargs}
        payload.setdefault("source_text", original)
        return payload

    # ------------------------------------------------------------------
    # Template: generic heal / remove damage counters.
    # ------------------------------------------------------------------
    m = re.fullmatch(r"Heal (\d+) damage from (.+?)\.?,?", t_rule, re.I)
    if m:
        amount = int(m.group(1))
        target_txt = m.group(2).strip()
        target = "from_text"
        if re.search(r"your Active Pokémon", target_txt, re.I):
            target = "self.active"
        elif re.search(r"each of your Pokémon|your Pokémon", target_txt, re.I):
            target = "self.in_play"
        elif re.search(r"each Pokémon", target_txt, re.I):
            target = "all.in_play"
        return [step("heal_damage", target=target, amount=amount, target_text=target_txt)], []

    m = re.fullmatch(r"Heal (\d+) damage and remove a Special Condition from your Active Pokémon\.?,?", t_rule, re.I)
    if m:
        return [
            step("heal_damage", target="self.active", amount=int(m.group(1))),
            step("remove_special_condition", target="self.active", amount={"mode": "all"}),
        ], []

    m = re.fullmatch(r"Remove (\d+) damage counters? from (.+?)\.?,?", t_rule, re.I)
    if m:
        amount = int(m.group(1))
        target_txt = m.group(2).strip()
        target = "self.in_play" if re.search(r"each of your|your", target_txt, re.I) else "from_text"
        return [step("remove_damage_counters", target=target, amount=amount, target_text=target_txt)], []

    m = re.fullmatch(r"Remove (\d+) damage counter from each of your Pokémon that has any damage counters on it\.?,?", t_rule, re.I)
    if m:
        return [step("remove_damage_counters", target="self.in_play", amount=int(m.group(1)), condition={"has_damage_counters": True})], []

    # ------------------------------------------------------------------
    # Template: Weakness / Resistance suppression and direct damage.
    # ------------------------------------------------------------------
    if re.fullmatch(r"Don't apply Weakness and Resistance\.?,?", t_rule, re.I):
        return [step("modify_damage_calculation", apply_weakness=False, apply_resistance=False, scope="current_attack")], []

    m = re.fullmatch(r"(?:Choose 1 of your opponent's Pokémon\. )?This attack does (\d+) damage to (?:1 of your opponent's Pokémon|that Pokémon)(?: that has any damage counters on it)?\. Don't apply Weakness and Resistance for this attack\.?.*", t_rule, re.I)
    if m:
        return [
            step("choose_target", player="self", target_id="chosen_opponent_pokemon", zone="opponent.in_play", filter={"supertype": "Pokémon"}, amount=amount_exact(1)),
            step("deal_damage", target_ref="chosen_opponent_pokemon", amount=int(m.group(1)), apply_weakness_resistance=False),
        ], []

    m = re.fullmatch(r"(?:This attack also does|Does|This attack does) (\d+) damage to (?:each|1 of your opponent's) Benched Pokémon.*\(Don't apply Weakness and Resistance for Benched Pokémon\.\).*", t_rule, re.I)
    if m:
        target = "opponent.bench" if re.search(r"each", t_rule, re.I) else "chosen_opponent_benched_pokemon"
        steps = []
        if target != "opponent.bench":
            steps.append(step("choose_target", player="self", target_id=target, zone="opponent.bench", filter={"supertype": "Pokémon"}, amount=amount_exact(1)))
            target_ref = target
        else:
            target_ref = target
        steps.append(step("deal_damage", target_ref=target_ref, amount=int(m.group(1)), apply_weakness_resistance=False))
        return steps, []

    m = re.fullmatch(r"(?:This attack also does|Does) (\d+) damage to (?:1 of )?your Benched Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t_rule, re.I)
    if m:
        return [
            step("choose_target", player="self", target_id="chosen_self_benched_pokemon", zone="self.bench", filter={"supertype": "Pokémon"}, amount=amount_exact(1)),
            step("deal_damage", target_ref="chosen_self_benched_pokemon", amount=int(m.group(1)), apply_weakness_resistance=False),
        ], []

    # ------------------------------------------------------------------
    # Template: damage counters / reflection / retaliation.
    # ------------------------------------------------------------------
    m = re.fullmatch(r"Put (\d+) damage counters? on each of your opponent's Pokémon\.?,?", t_rule, re.I)
    if m:
        return [step("place_damage_counters", target="opponent.in_play", amount=int(m.group(1)))], []

    m = re.fullmatch(r"Put (\d+) damage counters? on (?:1 of your opponent's Benched Pokémon|the Defending Pokémon)\.?,?", t_rule, re.I)
    if m:
        target = "opponent.bench" if "Benched" in t_rule else "opponent.active"
        return [step("place_damage_counters", target=target, amount=int(m.group(1)))], []

    m = re.fullmatch(r"If (?:the Pokémon )?this card is attached to is (?:your Active Pokémon|in the Active Spot) and is damaged by an attack from your opponent's Pokémon .* put (\d+) damage counters on the Attacking Pokémon\.?,?", t_rule, re.I)
    if m:
        return [step("register_triggered_effect", trigger="attached_pokemon_damaged_by_opponent_attack", condition={"attached_pokemon_is_active": True}, then=[{"op": "place_damage_counters", "target": "attacking_pokemon", "amount": int(m.group(1)), "source_text": original}])], []

    # ------------------------------------------------------------------
    # Template: coin damage plus status / attack fail / self-damage.
    # ------------------------------------------------------------------
    m = re.fullmatch(r"Flip (\d+) coins?\. This attack does (\d+) damage times the number of heads\. If (?:either of the coins is heads|you get (\d+) or more heads), (?:the Defending Pokémon|Dark Vileplume) is now (Asleep|Confused|Paralyzed|Poisoned|Burned).*", t_rule, re.I)
    if m:
        coin_count = int(m.group(1))
        status = m.group(4)
        threshold = int(m.group(3) or 1)
        return [
            step("coin_flip", player="self", count=coin_count, target_id="coin_results"),
            step("set_attack_damage_from_coin_heads", coin_results_ref="coin_results", amount_per_heads=int(m.group(2))),
            step("branch_on_coin_heads", result_ref="coin_results", heads_at_least=threshold, then=[status_condition_step(status, "opponent.active", original)]),
        ], []

    m = re.fullmatch(r"Flip a coin\. If tails, (?:this attack does nothing|this Pokémon can't attack during your next turn)\.?(?: If heads, during your opponent's next turn, prevent all damage from and effects of attacks done to this Pokémon\.)?", t_rule, re.I)
    if m:
        then_heads = []
        if "If heads" in t_rule:
            then_heads.append({"op": "register_delayed_modifier", "modifier_id": "prevent_damage_and_effects_next_turn", "target": "self_attacking_pokemon", "duration": {"until": "end_of_opponent_next_turn"}, "modification": {"prevent_attack_damage": True, "prevent_attack_effects": True}, "source_text": original})
        steps = [step("coin_flip", player="self", count=1, target_id="coin_result"), step("branch_on_result", result_ref="coin_result", if_="tails", **{"then": [{"op": "attack_does_nothing", "source_text": original}]})]
        if then_heads:
            steps.append(step("branch_on_result", result_ref="coin_result", if_="heads", **{"then": then_heads}))
        return steps, []

    m = re.fullmatch(r"Flip a coin\. If tails, ([A-Za-z' -]+|this Pokémon) does (\d+) damage to itself\.?,?", t_rule, re.I)
    if m:
        return [step("coin_flip", player="self", count=1, target_id="coin_result"), step("branch_on_result", result_ref="coin_result", if_="tails", **{"then": [{"op": "deal_damage", "target": "self_attacking_pokemon", "amount": int(m.group(2)), "source_text": original}]})], []

    # ------------------------------------------------------------------
    # Template: future damage bonuses and prevention/reduction.
    # ------------------------------------------------------------------
    m = re.fullmatch(r"(?:During your next turn|Until the end of your next turn), if an attack damages? the Defending Pokémon .* that attack does (\d+) more damage(?: to the Defending Pokémon)?\.?,?", t_rule, re.I)
    if m:
        return [step("register_delayed_modifier", modifier_id="future_damage_bonus_mark", target="opponent.active", duration={"until": "end_of_self_next_turn"}, modification={"attack_damage_delta": int(m.group(1)), "applies_when_damaged_by_attack": True})], []

    m = re.fullmatch(r"During your opponent's next turn, (?:any damage done to .* by attacks is reduced by|attacks used by .* do) (\d+) less damage.*", t_rule, re.I)
    if m:
        return [step("register_delayed_modifier", modifier_id="opponent_next_turn_damage_reduction", target="from_text", duration={"until": "end_of_opponent_next_turn"}, modification={"attack_damage_delta": -int(m.group(1))})], []

    m = re.fullmatch(r"Whenever your opponent plays a Supporter card from their hand, prevent all effects of that card done to this Pokémon\.?,?", t_rule, re.I)
    if m:
        return [step("register_continuous_modifier", modifier_id="prevent_opponent_supporter_effects_to_this_pokemon", target="self", modification={"prevent_supporter_effects_from_opponent": True})], []

    # ------------------------------------------------------------------
    # Template: search / look / reveal / put into hand or bench.
    # ------------------------------------------------------------------
    m = re.fullmatch(r"Look at the top (\d+) cards? of your deck\. You may reveal (?:a|an) ([A-Za-z ]+?) card you find there and put it into your hand\. Shuffle the other cards back into your deck\.?,?", t_rule, re.I)
    if m:
        return [
            step("look_at_top_cards", player="self", amount=int(m.group(1)), target_id="looked_cards"),
            step("choose_cards", player="self", target_id="chosen_from_looked", zone="looked_cards", filter={"from_text": m.group(2)}, amount=amount_up_to(1), reveal=True),
            step("move_card", cards_ref="chosen_from_looked", destination="self.hand"),
            step("shuffle_deck", player="self"),
        ], []

    m = re.fullmatch(r"Look at the top card of your deck\. You may (?:discard that card|put that card into your hand\. If you don't, discard that card and draw a card)\.?,?", t_rule, re.I)
    if m:
        return [step("look_at_top_cards", player="self", amount=1, target_id="top_card"), step("choose_topdeck_action", player="self", options=["put_into_hand", "discard", "draw_card_after_discard"], source_text=original)], []

    m = re.fullmatch(r"Once during each player's turn, that player may search their deck for an? ([A-Za-z ]+?) Pokémon, reveal it, and put it into their hand\. Then, that player shuffles their deck\.?,?", t_rule, re.I)
    if m:
        return [step("register_player_turn_action", action={"op": "search_deck", "filter": {"supertype": "Pokémon", "from_text": m.group(1)}, "destination": "player.hand", "reveal": True, "source_text": original}, usage_limit={"scope": "per_player_turn", "limit": 1})], []

    m = re.fullmatch(r"Search your deck for up to (\d+) basic Energy cards, reveal them, and put them into your hand\. Shuffle your deck afterward\.?,?", t_rule, re.I)
    if m:
        return [step("search_deck", player="self", target_id="searched_basic_energy", filter={"supertype": "Energy", "subtypes": ["Basic"]}, amount=amount_up_to(int(m.group(1))), reveal=True, destination="self.hand"), step("shuffle_deck", player="self")], []

    m = re.fullmatch(r"Search your deck for (?:Omanyte, Kabuto, or any Basic Pokémon|any number of Basic Pokémon).*put .* onto your Bench.*Shuffle your deck.*", t_rule, re.I)
    if m:
        return [step("search_deck", player="self", target_id="searched_basic_pokemon", filter={"supertype": "Pokémon", "subtypes": ["Basic"], "from_text": original}, amount={"mode": "from_text"}, destination="self.bench"), step("shuffle_deck", player="self")], []

    # ------------------------------------------------------------------
    # Template: energy attach/move/recover/provide.
    # ------------------------------------------------------------------
    m = re.fullmatch(r"Attach (?:a|an|up to (\d+)) basic ([A-Za-z]+ )?Energy card(?:s)? from your hand to 1 of your Benched Pokémon\.?,?", t_rule, re.I)
    if m:
        filt = {"supertype": "Energy", "subtypes": ["Basic"]}
        if m.group(2):
            filt["types"] = [m.group(2).strip().capitalize()]
        return [step("choose_cards", player="self", target_id="chosen_energy_from_hand", zone="self.hand", filter=filt, amount=amount_up_to(int(m.group(1) or 1))), step("attach_card", cards_ref="chosen_energy_from_hand", target="self.bench.pokemon")], []

    m = re.fullmatch(r"(?:Once during your turn .* you may )?attach (?:1 |an? |up to (\d+) )?([A-Za-z]+|basic)? ?Energy card(?:s)? from your discard pile to (?:1 of your Benched [A-Za-z]* ?Pokémon|this Pokémon|your Benched Pokémon in any way you like)\.?.*", t_rule, re.I)
    if m:
        filt = {"supertype": "Energy"}
        n = int(m.group(1) or 1)
        kind = (m.group(2) or "").strip().capitalize()
        if kind and kind.lower() != "basic":
            filt["types"] = [kind]
        elif kind.lower() == "basic":
            filt["subtypes"] = ["Basic"]
        return [step("choose_cards", player="self", target_id="chosen_energy_from_discard", zone="self.discard", filter=filt, amount=amount_up_to(n)), step("attach_card", cards_ref="chosen_energy_from_discard", target="self.in_play.pokemon")], []

    m = re.fullmatch(r"Move a basic Energy (?:card )?attached to 1 of your Pokémon to another of your Pokémon\.?,?", t_rule, re.I)
    if m:
        return [step("move_attached_energy", player="self", source="self.in_play.pokemon", destination="self.in_play.pokemon", filter={"supertype": "Energy", "subtypes": ["Basic"]}, amount=amount_exact(1))], []

    if re.fullmatch(r"As long as this card is attached to a Pokémon, it provides Colorless Energy.*", t_rule, re.I):
        return [step("register_attached_energy_provider", provided_energy=["Colorless"], condition={"while_attached_to": "Pokémon"})], []

    if re.fullmatch(r"All Special Energy attached to Pokémon .* provide Colorless Energy and have no other effect\.?,?", t_rule, re.I):
        return [step("register_continuous_modifier", modifier_id="special_energy_provides_colorless_only", target="all.in_play.attached_energy", filter={"subtypes": ["Special"]}, modification={"provided_energy": ["Colorless"], "suppress_other_effects": True})], []

    # ------------------------------------------------------------------
    # Template: gust/switch/return/disruption.
    # ------------------------------------------------------------------
    m = re.fullmatch(r"Switch in 1 of your opponent's Benched Pokémon to the Active Spot\. If you do, switch your Active Pokémon with 1 of your Benched Pokémon\.?,?", t_rule, re.I)
    if m:
        return [step("switch_active", player="opponent", selection="self_choice_from_opponent_bench"), step("switch_active", player="self", selection="self_choice_from_self_bench")], []

    m = re.fullmatch(r"Flip a coin\. If heads, switch in 1 of your opponent's Benched Pokémon to the Active Spot\.?,?", t_rule, re.I)
    if m:
        return [step("coin_flip", player="self", count=1, target_id="coin_result"), step("branch_on_result", result_ref="coin_result", if_="heads", **{"then": [{"op": "switch_active", "player": "opponent", "selection": "self_choice_from_opponent_bench", "source_text": original}]})], []

    m = re.fullmatch(r"Put 1 of your Pokémon and all attached cards into your hand\.?,?", t_rule, re.I)
    if m:
        return [step("choose_target", player="self", target_id="chosen_self_pokemon", zone="self.in_play", filter={"supertype": "Pokémon"}, amount=amount_exact(1)), step("move_zone_to_zone", target_ref="chosen_self_pokemon_and_attached_cards", destination="self.hand")], []

    # ------------------------------------------------------------------
    # Template: locks, rule modifiers, prize visibility, attack access.
    # ------------------------------------------------------------------
    m = re.fullmatch(r"Your opponent can't play any (Item|Trainer|Supporter) cards? from (?:their|his or her) hand during (?:their|his or her) next turn\.?,?", t_rule, re.I)
    if m:
        return [step("register_delayed_modifier", modifier_id="opponent_card_type_lock_next_turn", target="opponent", duration={"until": "end_of_opponent_next_turn"}, modification={"cannot_play_card_type": m.group(1).capitalize()})], []

    if re.fullmatch(r"All Pokémon Powers stop working until the end of your opponent's next turn\.?,?", t_rule, re.I):
        return [step("register_delayed_modifier", modifier_id="all_pokemon_powers_disabled", target="all.in_play", duration={"until": "end_of_opponent_next_turn"}, modification={"disable_pokemon_powers": True})], []

    if re.fullmatch(r"Each (?:player's )?.*Pokémon.*can't use any (?:Poké-Powers|Poké-Bodies|Abilities).*", t_rule, re.I):
        return [step("register_continuous_modifier", modifier_id="ability_power_body_lock", target="all.in_play", filter={"from_text": original}, modification={"disable_abilities_or_powers": True})], []

    if re.fullmatch(r"Each player pays Colorless less to retreat .*", t_rule, re.I) or re.fullmatch(r"The Retreat Cost of each Pokémon in play .* is Colorless (?:more|less)\.?,?", t_rule, re.I):
        delta = -1 if "less" in t_rule.lower() else 1
        return [step("register_continuous_modifier", modifier_id="global_retreat_cost_modifier", target="all.in_play", filter={"from_text": original}, modification={"retreat_cost_delta_colorless": delta})], []

    if re.fullmatch(r"Turn all of your Prize cards face up\. .*", t_rule, re.I):
        return [step("register_continuous_modifier", modifier_id="self_prizes_face_up", target="self", duration={"until": "end_of_game"}, modification={"prize_cards_face_up": True})], []

    if re.fullmatch(r"You can play this card only if you have more Prize cards remaining than your opponent\.?,?", t_rule, re.I):
        return [step("play_condition", condition={"self.prize_cards_remaining_gt_opponent": True})], []

    if re.fullmatch(r"Attach this card to 1 of your Pokémon in play\. That Pokémon may use this card's attack instead of its own\.?,?", t_rule, re.I):
        return [step("attach_card", card="this_card", target="self.pokemon"), step("register_granted_attack", source_card="this_card", target="attached_pokemon")], []

    if re.fullmatch(r"The Pokémon V this card is attached to can use the VSTAR Power on this card\.?,?", t_rule, re.I):
        return [step("register_granted_vstar_power", source_card="this_card", target="attached_pokemon_v")], []

    return None

def compile_simple_text(text: str, source_section: str) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Pattern-based compiler.

    Returns (steps, unparsed_texts). It is intentionally conservative:
    when a sentence is not safely represented, it is returned as unparsed.
    """
    original = normalize_space(text)
    t = original

    if not t:
        return [], []

    if is_global_rule_line(t):
        return [{
            "op": "reference_global_rule",
            "global_rule_id": global_rule_id_for_text(t),
            "source_text": original,
        }], []

    # v0.3: older card exports sometimes concatenate Supporter boilerplate and the real effect.
    # Split those safely so the global Supporter rule is recorded but the actual effect can still parse.
    m = re.fullmatch(
        r"(?:Rules:\s*)?You can play only (?:one|1) Supporter card each turn\. When you play this card, put it next to your Active Pokémon\. When your turn ends, discard this card\.\s*(.+)",
        t,
        re.I,
    )
    if m:
        rest = normalize_space(m.group(1))
        rest_steps, rest_unparsed = compile_simple_text(rest, source_section)
        return [{
            "op": "reference_global_rule",
            "global_rule_id": "play_supporter",
            "source_text": original,
        }] + rest_steps, rest_unparsed

    m = re.fullmatch(
        r"(?:Rules:\s*)?You may play only 1 Supporter card during your turn(?: \(before your attack\))?\.\s*(.+)",
        t,
        re.I,
    )
    if m:
        rest = normalize_space(m.group(1))
        rest_steps, rest_unparsed = compile_simple_text(rest, source_section)
        return [{
            "op": "reference_global_rule",
            "global_rule_id": "play_supporter",
            "source_text": original,
        }] + rest_steps, rest_unparsed

    rule_box = compile_rule_box(t)
    if rule_box:
        for step in rule_box:
            step["source_text"] = original
        return rule_box, []

    # Direct Special Condition attack text.
    m = re.fullmatch(r"(?:The Defending Pokémon|Your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [status_condition_step(m.group(1), "opponent.active", original)], []

    # Coin flip -> Special Condition.
    m = re.fullmatch(r"Flip a coin\. If heads, (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [status_condition_step(m.group(1), "opponent.active", original)],
            "source_text": original,
        }], []

    # v0.13: broader Special Condition family patterns.
    # Choose any Special Condition after a successful coin flip. Older text varies
    # between "a" and "1", and between Defending Pokémon / opponent's Active Pokémon.
    if re.fullmatch(r"Flip a coin\. If heads, choose (?:a|1) Special Condition\. (?:The Defending Pokémon|Your opponent's Active Pokémon) is now affected by that Special Condition\.?,?", t, re.I):
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [{
                "op": "choose_special_condition",
                "player": "self",
                "target_id": "chosen_special_condition",
                "choices": ["Asleep", "Burned", "Confused", "Paralyzed", "Poisoned"],
                "source_text": original,
            }, {
                "op": "apply_special_condition",
                "target": "opponent.active",
                "condition_ref": "chosen_special_condition",
                "source_text": original,
            }],
            "source_text": original,
        }], []

    # Poison variants where the damage counter rate changes.
    m = re.fullmatch(r"(?:The Defending Pokémon|Your opponent's Active Pokémon) is now Poisoned\. It now takes (\d+) Poison damage instead of 10 after each player's turn \(even if it was already Poisoned\)\.?,?", t, re.I)
    if m:
        return [status_condition_step("Poisoned", "opponent.active", original), {
            "op": "modify_special_condition",
            "target": "opponent.active",
            "condition": "Poisoned",
            "poison_damage": int(m.group(1)),
            "poison_damage_counters_between_turns": int(m.group(1)) // 10 if int(m.group(1)) % 10 == 0 else None,
            "source_text": original,
        }], []

    m = re.fullmatch(r"Flip a coin\. If heads, (?:the Defending Pokémon|your opponent's Active Pokémon) is now Poisoned\. It now takes (\d+) Poison damage instead of 10 after each player's turn \(even if it was already Poisoned\)\.?,?", t, re.I)
    if m:
        poison_steps = [status_condition_step("Poisoned", "opponent.active", original), {
            "op": "modify_special_condition",
            "target": "opponent.active",
            "condition": "Poisoned",
            "poison_damage": int(m.group(1)),
            "poison_damage_counters_between_turns": int(m.group(1)) // 10 if int(m.group(1)) % 10 == 0 else None,
            "source_text": original,
        }]
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": poison_steps, "source_text": original}], []

    # Delayed Special Conditions at the end of the opponent's next turn.
    m = re.fullmatch(r"At the end of your opponent's next turn, (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [{
            "op": "register_delayed_effect",
            "trigger": "end_of_opponent_next_turn",
            "then": [status_condition_step(m.group(1), "opponent.active", original)],
            "source_text": original,
        }], []

    # Special sleep rule: opponent flips two coins between turns and remains Asleep
    # if either is tails.
    if re.fullmatch(r"Your opponent's Active Pokémon is now Asleep\. Your opponent flips 2 coins instead of 1 between turns\. If either of them is tails, that Pokémon is still Asleep\.?,?", t, re.I):
        return [status_condition_step("Asleep", "opponent.active", original), {
            "op": "modify_special_condition",
            "target": "opponent.active",
            "condition": "Asleep",
            "between_turns_coin_flips": 2,
            "recovery_condition": "all_heads",
            "source_text": original,
        }], []

    # Recovery / immunity wording.
    if re.fullmatch(r"This Pokémon recovers from all Special Conditions\.?,?", t, re.I):
        return [{"op": "remove_special_conditions", "target": "self", "conditions": "all", "source_text": original}], []

    m = re.fullmatch(r"This Pokémon can't be (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [{"op": "register_continuous_modifier", "modifier_id": f"self_cannot_be_{m.group(1).lower()}", "target": "self", "duration": {"while_source_in_play": True}, "modification": {"prevent_special_conditions": [m.group(1).capitalize()]}, "source_text": original}], []

    # Coin flip -> attack does nothing on tails.
    if re.fullmatch(r"Flip a coin\. If tails, this attack does nothing\.?,?", t, re.I):
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "tails",
            "then": [{"op": "attack_does_nothing", "source_text": original}],
            "source_text": original,
        }], []

    # Coin flip -> more damage.
    m = re.fullmatch(r"Flip a coin\. If heads, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "source_text": original}],
            "source_text": original,
        }], []

    # Older wording: this attack does base damage plus N more damage.
    m = re.fullmatch(r"Flip a coin\. If heads, this attack does \d+ damage plus (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "source_text": original}],
            "source_text": original,
        }], []

    # Fixed number of coins -> damage times / for each heads / more damage for each heads.
    m = re.fullmatch(r"Flip (\d+) coins\. This attack does (\d+) damage (?:times the number of heads|for each heads)\.?,?", t, re.I)
    if m:
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": int(m.group(1)),
            "target_id": "coin_results",
            "source_text": original,
        }, {
            "op": "set_attack_damage_from_coin_heads",
            "damage_per_heads": int(m.group(2)),
            "coin_results_ref": "coin_results",
            "source_text": original,
        }], []

    m = re.fullmatch(r"Flip (\d+) coins\. This attack does (\d+) more damage for each heads\.?,?", t, re.I)
    if m:
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": int(m.group(1)),
            "target_id": "coin_results",
            "source_text": original,
        }, {
            "op": "modify_attack_damage_per_coin_heads",
            "mode": "add",
            "amount_per_heads": int(m.group(2)),
            "coin_results_ref": "coin_results",
            "source_text": original,
        }], []

    m = re.fullmatch(r"Flip a coin until you get tails\. This attack does (\d+) (damage|more damage) (?:times the number of heads|for each heads)\.?,?", t, re.I)
    if m:
        op = "set_attack_damage_from_coin_heads" if m.group(2).lower() == "damage" else "modify_attack_damage_per_coin_heads"
        step = {
            "op": op,
            "coin_results_ref": "coin_results_until_tails",
            "source_text": original,
        }
        if op == "set_attack_damage_from_coin_heads":
            step["damage_per_heads"] = int(m.group(1))
        else:
            step["mode"] = "add"
            step["amount_per_heads"] = int(m.group(1))
        return [{
            "op": "coin_flip_until",
            "player": "self",
            "until": "tails",
            "target_id": "coin_results_until_tails",
            "source_text": original,
        }, step], []

    # Heal from attacking Pokémon.
    m = re.fullmatch(r"Heal (\d+) damage from this Pokémon\.?,?", t, re.I)
    if m:
        return [{
            "op": "heal_damage",
            "target": "self_attacking_pokemon",
            "amount": amount_exact(int(m.group(1))),
            "source_text": original,
        }], []

    if re.fullmatch(r"Heal from this Pokémon the same amount of damage you did to your opponent's Active Pokémon\.?,?", t, re.I):
        return [{
            "op": "heal_damage",
            "target": "self_attacking_pokemon",
            "amount": {"mode": "damage_dealt_this_attack", "target": "opponent.active"},
            "source_text": original,
        }], []

    # Self attack restrictions / retreat restrictions.
    if re.fullmatch(r"(?:During your next turn, this Pokémon can't attack|This Pokémon can't attack during your next turn)\.?,?", t, re.I):
        return [delayed_modifier_step("self_cannot_attack_next_turn", "self_attacking_pokemon", {"attack_allowed": False}, original)], []

    if re.fullmatch(r"(?:The Defending Pokémon can't retreat during your opponent's next turn|During your opponent's next turn, the Defending Pokémon can't retreat)\.?,?", t, re.I):
        return [delayed_modifier_step("opponent_active_cannot_retreat_next_turn", "opponent.active", {"retreat_allowed": False}, original)], []

    if re.fullmatch(r"If the Defending Pokémon tries to attack during your opponent's next turn, your opponent flips a coin\. If tails, that attack does nothing\.?,?", t, re.I):
        return [delayed_modifier_step(
            "opponent_active_attack_requires_heads_next_turn",
            "opponent.active",
            {"attack_requires_coin_flip": {"on_tails": "attack_does_nothing"}},
            original,
        )], []

    # Discard Energy from self or opponent.
    m = re.fullmatch(r"Discard (an|a|\d+|all) Energy(?: card)?s? (?:attached to |from )this Pokémon\.?,?", t, re.I)
    if m:
        raw_n = m.group(1).lower()
        amount = {"mode": "all"} if raw_n == "all" else amount_exact(1 if raw_n in ["a", "an"] else int(raw_n))
        return [{
            "op": "discard_attached_energy",
            "target": "self_attacking_pokemon",
            "amount": amount,
            "source_text": original,
        }], []

    m = re.fullmatch(r"Discard (an|a|\d+|all) Energy(?: card)?s? (?:attached to |from )(?:your opponent's Active Pokémon|the Defending Pokémon)\.?,?", t, re.I)
    if m:
        raw_n = m.group(1).lower()
        amount = {"mode": "all"} if raw_n == "all" else amount_exact(1 if raw_n in ["a", "an"] else int(raw_n))
        return [{
            "op": "discard_attached_energy",
            "target": "opponent.active",
            "amount": amount,
            "source_text": original,
        }], []

    m = re.fullmatch(r"Flip a coin\. If heads, discard (?:an|a) Energy(?: card)? (?:attached to |from )(?:your opponent's Active Pokémon|the Defending Pokémon)\.?,?", t, re.I)
    if m:
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [{"op": "discard_attached_energy", "target": "opponent.active", "amount": amount_exact(1), "source_text": original}],
            "source_text": original,
        }], []

    # v0.10: broader Energy discard / move / bounce variants.
    # These patterns cover common historical attack text that names
    # "1 of your opponent's Pokémon" instead of only the Active/Defending Pokémon.
    m = re.fullmatch(r"Discard (an|a|1|\d+|all) Energy(?: card)?s? attached to 1 of your opponent's Pokémon\.?,?", t, re.I)
    if m:
        raw_n = m.group(1).lower()
        amount = {"mode": "all"} if raw_n == "all" else amount_exact(1 if raw_n in ["a", "an", "1"] else int(raw_n))
        return [{
            "op": "choose_target",
            "player": "self",
            "target_id": "chosen_opponent_pokemon_with_energy",
            "zone": "opponent.in_play",
            "filter": {"has_attached_energy": True},
            "amount": amount_exact(1),
            "source_text": original,
        }, {
            "op": "discard_attached_energy",
            "target_ref": "chosen_opponent_pokemon_with_energy",
            "amount": amount,
            "source_text": original,
        }], []

    m = re.fullmatch(r"Flip a coin\. If heads, discard (an|a|1|\d+|all) Energy(?: card)?s? attached to 1 of your opponent's Pokémon\.?,?", t, re.I)
    if m:
        raw_n = m.group(1).lower()
        amount = {"mode": "all"} if raw_n == "all" else amount_exact(1 if raw_n in ["a", "an", "1"] else int(raw_n))
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [{
                "op": "choose_target",
                "player": "self",
                "target_id": "chosen_opponent_pokemon_with_energy",
                "zone": "opponent.in_play",
                "filter": {"has_attached_energy": True},
                "amount": amount_exact(1),
                "source_text": original,
            }, {
                "op": "discard_attached_energy",
                "target_ref": "chosen_opponent_pokemon_with_energy",
                "amount": amount,
                "source_text": original,
            }],
            "source_text": original,
        }], []

    m = re.fullmatch(r"Discard (an|a|1|\d+|all) Energy(?: card)?s? attached to each of your opponent's Pokémon\.?,?", t, re.I)
    if m:
        raw_n = m.group(1).lower()
        amount = {"mode": "all"} if raw_n == "all" else amount_exact(1 if raw_n in ["a", "an", "1"] else int(raw_n))
        return [{
            "op": "discard_attached_energy",
            "target": "opponent.in_play.each",
            "amount": amount,
            "source_text": original,
        }], []


    # v0.11: remaining common Energy discard variants from long-tail review.
    # Keep these generic: preserve typed filters and conditional coin structure,
    # but avoid over-resolving old wording beyond target/card movement semantics.
    m = re.fullmatch(r"Discard (?:a|an|1) (Special )?Energy(?: card)?(?:, if any,)? (?:attached to|from) (?:the Defending Pokémon|your opponent's Active Pokémon)\.?,?", t, re.I)
    if m:
        filt = {"supertype": "Energy"}
        if m.group(1):
            filt["subtypes"] = ["Special"]
        return [{
            "op": "discard_attached_energy",
            "target": "opponent.active",
            "filter": filt,
            "amount": amount_exact(1),
            "source_text": original,
        }], []

    m = re.fullmatch(r"Discard (?:a|an|1) (Special )?Energy(?: card)? (?:attached to|from) 1 of your opponent's Pokémon\.?,?", t, re.I)
    if m:
        filt = {"supertype": "Energy"}
        if m.group(1):
            filt["subtypes"] = ["Special"]
        return [{
            "op": "choose_target",
            "player": "self",
            "target_id": "chosen_opponent_pokemon_with_energy",
            "zone": "opponent.in_play",
            "filter": {"has_attached_energy": True},
            "amount": amount_exact(1),
            "source_text": original,
        }, {
            "op": "discard_attached_energy",
            "target_ref": "chosen_opponent_pokemon_with_energy",
            "filter": filt,
            "amount": amount_exact(1),
            "source_text": original,
        }], []

    m = re.fullmatch(r"Discard (?:a|an|1) (Special )?Energy(?: card)?(?:, if any,)? attached to the Defending Pokémon\.?,?", t, re.I)
    if m:
        filt = {"supertype": "Energy"}
        if m.group(1):
            filt["subtypes"] = ["Special"]
        return [{"op": "discard_attached_energy", "target": "opponent.active", "filter": filt, "amount": amount_exact(1), "source_text": original}], []

    m = re.fullmatch(r"Discard (?:a|an|1) (Special )?Energy(?: card)? from 1 of your opponent's Pokémon\.?,?", t, re.I)
    if m:
        filt = {"supertype": "Energy"}
        if m.group(1):
            filt["subtypes"] = ["Special"]
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_pokemon_with_energy", "zone": "opponent.in_play", "filter": {"has_attached_energy": True}, "amount": amount_exact(1), "source_text": original}, {"op": "discard_attached_energy", "target_ref": "chosen_opponent_pokemon_with_energy", "filter": filt, "amount": amount_exact(1), "source_text": original}], []

    m = re.fullmatch(r"Discard (?:an|a|1) Energy from each of your opponent's Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "discard_attached_energy", "target": "opponent.in_play.each", "filter": {"supertype": "Energy"}, "amount": amount_exact(1), "source_text": original}], []

    m = re.fullmatch(r"Flip a coin until you get tails\. For each heads, discard an Energy(?: card)? attached to the Defending Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip_until", "player": "self", "until": "tails", "target_id": "coin_results_until_tails", "source_text": original}, {"op": "discard_attached_energy_per_coin_heads", "target": "opponent.active", "coin_results_ref": "coin_results_until_tails", "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If tails, discard (\d+) Energy(?: card)?s? attached to this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "tails", "then": [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "amount": amount_exact(int(m.group(1))), "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Discard all ([A-Za-z]+) Energy from this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "filter": {"energy_type": m.group(1).capitalize(), "types": [m.group(1).capitalize()]}, "amount": {"mode": "all"}, "source_text": original}], []

    # Switches from attacks.
    if re.fullmatch(r"(?:Switch this Pokémon with 1 of your Benched Pokémon|You may switch this Pokémon with 1 of your Benched Pokémon)\.?,?", t, re.I):
        optional = t.lower().startswith("you may")
        return [{
            "op": "choose_target",
            "player": "self",
            "target_id": "chosen_own_benched_pokemon",
            "zone": "self.bench",
            "filter": {"supertype": "Pokémon"},
            "amount": amount_exact(1),
            "optional": optional,
            "source_text": original,
        }, {
            "op": "switch_active",
            "player": "self",
            "new_active_ref": "chosen_own_benched_pokemon",
            "optional": optional,
            "source_text": original,
        }], []

    if re.fullmatch(r"(?:Your opponent switches (?:their Active Pokémon|the Defending Pokémon) with 1 of (?:their|his or her) Benched Pokémon)\.?,?", t, re.I):
        return [{
            "op": "switch_active",
            "player": "opponent",
            "chooser": "opponent",
            "new_active_ref": "opponent_choice_from_bench",
            "source_text": original,
        }], []

    # Self-damage.
    m = re.fullmatch(r"This Pokémon does (\d+) damage to itself\.?,?", t, re.I)
    if m:
        return [{
            "op": "deal_damage",
            "target": "self_attacking_pokemon",
            "amount": int(m.group(1)),
            "source_text": original,
        }], []

    # Prevention / reduction next turn or static bench protection.
    m = re.fullmatch(r"During your opponent's next turn, this Pokémon takes (\d+) less damage from attacks \(after applying Weakness and Resistance\)\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("self_takes_less_damage_next_turn", "self_attacking_pokemon", {"damage_taken_from_attacks_delta": -int(m.group(1))}, original)], []

    m = re.fullmatch(r"During your opponent's next turn, any damage done to this Pokémon by attacks is reduced by (\d+) \(after applying Weakness and Resistance\)\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("self_takes_less_damage_next_turn", "self_attacking_pokemon", {"damage_taken_from_attacks_delta": -int(m.group(1))}, original)], []

    m = re.fullmatch(r"This Pokémon takes (\d+) less damage from attacks \(after applying Weakness and Resistance\)\.?,?", t, re.I)
    if m:
        return [{
            "op": "register_continuous_modifier",
            "modifier_id": "self_takes_less_damage_from_attacks",
            "target": "self",
            "duration": {"while_source_in_play": True},
            "modification": {"damage_taken_from_attacks_delta": -int(m.group(1))},
            "source_text": original,
        }], []

    if re.fullmatch(r"(?:Tera: )?As long as this Pokémon is on your Bench, prevent all damage done to this Pokémon by attacks \(both yours and your opponent's\)\.?,?", t, re.I):
        return [{
            "op": "register_continuous_modifier",
            "modifier_id": "prevent_bench_damage_to_self",
            "target": "self",
            "condition": {"zone": "bench"},
            "duration": {"while_source_in_play": True},
            "modification": {"prevent_damage_from_attacks": True},
            "source_text": original,
        }], []

    if re.fullmatch(r"Flip a coin\. If heads, (?:during your opponent's next turn, )?prevent all (?:damage from and effects of attacks|effects of attacks, including damage,) done to this Pokémon during your opponent's next turn\.?,?", t, re.I):
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [delayed_modifier_step("prevent_attack_damage_and_effects_to_self_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True, "prevent_effects_of_attacks": True}, original)],
            "source_text": original,
        }], []

    # Disruption / mill.
    if re.fullmatch(r"Discard the top card of your opponent's deck\.?,?", t, re.I):
        return [{
            "op": "discard_cards",
            "player": "opponent",
            "source_zone": "deck.top",
            "selection": amount_exact(1),
            "destination": "discard",
            "source_text": original,
        }], []

    if re.fullmatch(r"Discard a random card from your opponent's hand\.?,?", t, re.I):
        return [{
            "op": "discard_cards",
            "player": "opponent",
            "source_zone": "hand",
            "selection": {"mode": "random", "amount": 1},
            "destination": "discard",
            "source_text": original,
        }], []

    # Bench damage / spread damage.
    m = re.fullmatch(r"(?:This attack also does |This attack does |Does )(\d+) damage to 1 of your opponent's (Benched Pokémon|Pokémon)\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?", t, re.I)
    if m:
        zone = "opponent.bench" if "Benched" in m.group(2) else "opponent.in_play"
        return [{
            "op": "choose_target",
            "player": "self",
            "target_id": "chosen_opponent_pokemon_for_bench_damage",
            "zone": zone,
            "filter": {"supertype": "Pokémon"},
            "amount": amount_exact(1),
            "source_text": original,
        }, {
            "op": "deal_damage",
            "target_ref": "chosen_opponent_pokemon_for_bench_damage",
            "amount": int(m.group(1)),
            "apply_weakness_resistance": False,
            "source_text": original,
        }], []

    m = re.fullmatch(r"This attack does (\d+) damage to each of your opponent's Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{
            "op": "deal_damage",
            "target": "opponent.in_play.each",
            "amount": int(m.group(1)),
            "apply_weakness_resistance_to_bench": False,
            "source_text": original,
        }], []

    # Search deck -> Basic Pokémon to Bench.
    m = re.fullmatch(r"Search your deck for (?:a|1) Basic Pokémon and put it onto your Bench\. (?:Then, shuffle your deck|Shuffle your deck afterward)\.?,?", t, re.I)
    if m:
        return [{
            "op": "search_deck",
            "player": "self",
            "target_id": "searched_basic_pokemon",
            "filter": {"supertype": "Pokémon", "subtypes": ["Basic"]},
            "amount": amount_exact(1),
            "reveal": False,
            "destination": "self.bench",
            "source_text": original,
        }, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for up to (\d+) Basic Pokémon and put them onto your Bench\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{
            "op": "search_deck",
            "player": "self",
            "target_id": "searched_basic_pokemon",
            "filter": {"supertype": "Pokémon", "subtypes": ["Basic"]},
            "amount": amount_up_to(int(m.group(1))),
            "reveal": False,
            "destination": "self.bench",
            "source_text": original,
        }, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    # Ignore modifiers on damage calculation.
    if re.fullmatch(r"This attack's damage isn't affected by Weakness or Resistance\.?,?", t, re.I):
        return [{"op": "ignore_weakness_resistance", "scope": "this_attack", "source_text": original}], []

    if re.fullmatch(r"This attack's damage isn't affected by any effects on your opponent's Active Pokémon\.?,?", t, re.I):
        return [{"op": "ignore_effects_on_defending_pokemon", "scope": "this_attack_damage", "source_text": original}], []

    if re.fullmatch(r"This attack's damage isn't affected by Weakness, Resistance, Poké-Powers, Poké-Bodies, or any other effects on the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "ignore_defending_pokemon_damage_modifiers", "scope": "this_attack_damage", "source_text": original}], []

    # Move Energy from self to own Bench.
    if re.fullmatch(r"Move an Energy from this Pokémon to 1 of your Benched Pokémon\.?,?", t, re.I):
        return [{
            "op": "choose_cards",
            "player": "self",
            "target_id": "chosen_energy_card",
            "zone": "attached_to:self_attacking_pokemon",
            "filter": {"supertype": "Energy"},
            "amount": amount_exact(1),
            "source_text": original,
        }, {
            "op": "choose_target",
            "player": "self",
            "target_id": "chosen_own_benched_pokemon",
            "zone": "self.bench",
            "filter": {"supertype": "Pokémon"},
            "amount": amount_exact(1),
            "source_text": original,
        }, {
            "op": "move_energy",
            "cards_ref": "chosen_energy_card",
            "destination_ref": "chosen_own_benched_pokemon",
            "source_text": original,
        }], []

    # v0.10 variants with "attached to" wording rather than "from" wording.
    if re.fullmatch(r"Move an Energy(?: card)? attached to this Pokémon to 1 of your Benched Pokémon\.?,?", t, re.I):
        return [{
            "op": "choose_cards",
            "player": "self",
            "target_id": "chosen_energy_card",
            "zone": "attached_to:self_attacking_pokemon",
            "filter": {"supertype": "Energy"},
            "amount": amount_exact(1),
            "source_text": original,
        }, {
            "op": "choose_target",
            "player": "self",
            "target_id": "chosen_own_benched_pokemon",
            "zone": "self.bench",
            "filter": {"supertype": "Pokémon"},
            "amount": amount_exact(1),
            "source_text": original,
        }, {
            "op": "move_energy",
            "cards_ref": "chosen_energy_card",
            "destination_ref": "chosen_own_benched_pokemon",
            "source_text": original,
        }], []

    m = re.fullmatch(r"Move (all|an|a|1|\d+) Energy(?: card)?s? attached to this Pokémon to your Benched Pokémon in any way you like\.?,?", t, re.I)
    if m:
        raw_n = m.group(1).lower()
        cards = "all_attached_to:self_attacking_pokemon" if raw_n == "all" else None
        if cards:
            return [{"op": "move_energy", "cards": cards, "destination": "self.bench.distributed", "source_text": original}], []
        amount = amount_exact(1 if raw_n in ["a", "an", "1"] else int(raw_n))
        return [{
            "op": "choose_cards",
            "player": "self",
            "target_id": "chosen_energy_cards",
            "zone": "attached_to:self_attacking_pokemon",
            "filter": {"supertype": "Energy"},
            "amount": amount,
            "source_text": original,
        }, {
            "op": "move_energy",
            "cards_ref": "chosen_energy_cards",
            "destination": "self.bench.distributed",
            "source_text": original,
        }], []


    # v0.11: typed and broader own-board Energy movement variants.
    m = re.fullmatch(r"Move (?:a|an|1) ([A-Za-z]+ )?Energy from this Pokémon to 1 of your Benched Pokémon\.?,?", t, re.I)
    if m:
        filt = {"supertype": "Energy"}
        if m.group(1):
            energy_type = m.group(1).strip().capitalize()
            filt.update({"types": [energy_type], "energy_type": energy_type})
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_energy_card", "zone": "attached_to:self_attacking_pokemon", "filter": filt, "amount": amount_exact(1), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_own_benched_pokemon", "zone": "self.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_energy", "cards_ref": "chosen_energy_card", "destination_ref": "chosen_own_benched_pokemon", "source_text": original}], []

    if re.fullmatch(r"Move all Energy from this Pokémon to 1 of your Benched Pokémon\.?,?", t, re.I):
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_own_benched_pokemon", "zone": "self.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_energy", "cards": "all_attached_to:self_attacking_pokemon", "destination_ref": "chosen_own_benched_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Move as many ([A-Za-z]+ )?Energy attached to your Pokémon to your other Pokémon in any way you like\.?,?", t, re.I)
    if m:
        filt = {"supertype": "Energy"}
        if m.group(1):
            energy_type = m.group(1).strip().capitalize()
            filt.update({"types": [energy_type], "energy_type": energy_type})
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_energy_cards", "zone": "attached_to:self.in_play", "filter": filt, "amount": {"mode": "any"}, "source_text": original}, {"op": "move_energy", "cards_ref": "chosen_energy_cards", "destination": "self.in_play.distributed_excluding_original_sources", "source_text": original}], []

    # Draw a card.
    if re.fullmatch(r"Draw a card\.?,?", t, re.I):
        return [{
            "op": "draw_cards",
            "player": "self",
            "amount": amount_exact(1),
            "source_text": original,
        }], []

    # Draw N cards.
    m = re.fullmatch(r"Draw (\d+) cards?\.?", t, re.I)
    if m:
        return [{
            "op": "draw_cards",
            "player": "self",
            "amount": amount_exact(int(m.group(1))),
            "source_text": original,
        }], []

    # Draw cards until you have N cards in your hand.
    m = re.fullmatch(r"Draw cards until you have (\d+) cards? in your hand\.?", t, re.I)
    if m:
        return [{
            "op": "draw_until_hand_size",
            "player": "self",
            "target_hand_size": int(m.group(1)),
            "source_text": original,
        }], []

    # Discard your hand and draw N cards.
    m = re.fullmatch(r"Discard your hand and draw (\d+) cards?\.?", t, re.I)
    if m:
        return [
            {
                "op": "discard_cards",
                "player": "self",
                "source_zone": "hand",
                "selection": {"mode": "all"},
                "destination": "discard",
                "source_text": original,
            },
            {
                "op": "draw_cards",
                "player": "self",
                "amount": amount_exact(int(m.group(1))),
                "source_text": original,
            },
        ], []

    # Each player shuffles their hand into their deck and draws N cards.
    m = re.fullmatch(r"Each player shuffles their hand into their deck and draws (\d+) cards?\.?", t, re.I)
    if m:
        return [
            {
                "op": "move_zone_to_zone",
                "player": "each",
                "source_zone": "hand",
                "destination_zone": "deck",
                "selection": {"mode": "all"},
                "source_text": original,
            },
            {"op": "shuffle_deck", "player": "each", "source_text": original},
            {
                "op": "draw_cards",
                "player": "each",
                "amount": amount_exact(int(m.group(1))),
                "source_text": original,
            },
        ], []

    # Switch your Active Pokémon with 1 of your Benched Pokémon.
    m = re.fullmatch(r"Switch your Active Pokémon with 1 of your Benched Pokémon\.?", t, re.I)
    if m:
        return [
            {
                "op": "choose_target",
                "player": "self",
                "target_id": "chosen_own_benched_pokemon",
                "zone": "self.bench",
                "filter": {"supertype": "Pokémon"},
                "amount": amount_exact(1),
                "source_text": original,
            },
            {
                "op": "switch_active",
                "player": "self",
                "new_active_ref": "chosen_own_benched_pokemon",
                "source_text": original,
            },
        ], []

    # Switch in 1 of your opponent's Benched Pokémon to the Active Spot.
    m = re.fullmatch(r"Switch in 1 of your opponent's Benched Pokémon to the Active Spot\.?", t, re.I)
    if m:
        return [
            {
                "op": "choose_target",
                "player": "self",
                "target_id": "chosen_opponent_benched_pokemon",
                "zone": "opponent.bench",
                "filter": {"supertype": "Pokémon"},
                "amount": amount_exact(1),
                "source_text": original,
            },
            {
                "op": "switch_active",
                "player": "opponent",
                "new_active_ref": "chosen_opponent_benched_pokemon",
                "source_text": original,
            },
        ], []

    # Switch out opponent active; opponent chooses.
    if re.fullmatch(r"Switch out your opponent's Active Pokémon to the Bench\. \(Your opponent chooses the new Active Pokémon\.\)", t, re.I):
        return [{
            "op": "switch_active",
            "player": "opponent",
            "chooser": "opponent",
            "new_active_ref": "opponent_choice_from_bench",
            "source_text": original,
        }], []

    # Heal N damage from 1 of your ...
    m = re.match(r"^Heal (\d+) damage from 1 of your (.+?)(?:, and it recovers from all Special Conditions)?\.?$", t, re.I)
    if m:
        steps = [
            {
                "op": "choose_target",
                "player": "self",
                "target_id": "chosen_own_pokemon",
                "zone": "self.in_play",
                "filter": raw_filter("your " + m.group(2)),
                "amount": amount_exact(1),
                "source_text": original,
            },
            {
                "op": "heal_damage",
                "target_ref": "chosen_own_pokemon",
                "amount": amount_exact(int(m.group(1))),
                "source_text": original,
            },
        ]
        if "recovers from all Special Conditions" in t:
            steps.append({
                "op": "remove_special_conditions",
                "target_ref": "chosen_own_pokemon",
                "conditions": "all",
                "source_text": original,
            })
        return steps, []

    # Heal all damage from 1 of your ...
    m = re.match(r"^Heal all damage from 1 of your (.+?)\.?$", t, re.I)
    if m:
        return [
            {
                "op": "choose_target",
                "player": "self",
                "target_id": "chosen_own_pokemon",
                "zone": "self.in_play",
                "filter": raw_filter("your " + m.group(1)),
                "amount": amount_exact(1),
                "source_text": original,
            },
            {
                "op": "heal_damage",
                "target_ref": "chosen_own_pokemon",
                "amount": {"mode": "all"},
                "source_text": original,
            },
        ], []

    # Attach up to N Basic TYPE Energy cards from your discard pile to 1 of your TYPE Pokémon.
    m = re.fullmatch(r"Attach up to (\d+) Basic ([A-Za-z]+) Energy cards? from your discard pile to 1 of your ([A-Za-z]+) Pokémon\.?", t, re.I)
    if m:
        n = int(m.group(1))
        energy_type = m.group(2)
        pokemon_type = m.group(3)
        return [
            {
                "op": "choose_target",
                "player": "self",
                "target_id": "chosen_own_pokemon",
                "zone": "self.in_play",
                "filter": {"supertype": "Pokémon", "types": [pokemon_type]},
                "amount": amount_exact(1),
                "source_text": original,
            },
            {
                "op": "choose_cards",
                "player": "self",
                "target_id": "chosen_energy_cards",
                "zone": "self.discard",
                "filter": {"supertype": "Energy", "subtypes": ["Basic"], "types": [energy_type]},
                "amount": amount_up_to(n),
                "source_text": original,
            },
            {
                "op": "attach_card",
                "cards_ref": "chosen_energy_cards",
                "target_ref": "chosen_own_pokemon",
                "source_text": original,
            },
        ], []

    # Search your deck for up to N Basic TYPE Energy cards...
    m = re.fullmatch(r"Search your deck for up to (\d+) Basic ([A-Za-z]+) Energy cards?, reveal them, and put them into your hand\. Then, shuffle your deck\.?", t, re.I)
    if m:
        return [
            {
                "op": "search_deck",
                "player": "self",
                "target_id": "searched_cards",
                "filter": {"supertype": "Energy", "subtypes": ["Basic"], "types": [m.group(2)]},
                "amount": amount_up_to(int(m.group(1))),
                "reveal": True,
                "destination": "self.hand",
                "source_text": original,
            },
            {"op": "shuffle_deck", "player": "self", "source_text": original},
        ], []

    # Search your deck for a Basic Energy card...
    m = re.fullmatch(r"Search your deck for a Basic Energy card, reveal it, and put it into your hand\. Then, shuffle your deck\.?", t, re.I)
    if m:
        return [
            {
                "op": "search_deck",
                "player": "self",
                "target_id": "searched_cards",
                "filter": {"supertype": "Energy", "subtypes": ["Basic"]},
                "amount": amount_exact(1),
                "reveal": True,
                "destination": "self.hand",
                "source_text": original,
            },
            {"op": "shuffle_deck", "player": "self", "source_text": original},
        ], []

    # Generic search deck -> hand pattern.
    m = re.fullmatch(r"Search your deck for (.+?), reveal (?:it|them), and put (?:it|them) into your hand\. Then, shuffle your deck\.?", t, re.I)
    if m:
        query = m.group(1)
        amount = amount_exact(1)
        m_amount = re.match(r"up to (\d+) (.+)", query, re.I)
        if m_amount:
            amount = amount_up_to(int(m_amount.group(1)))
            query = m_amount.group(2)
        return [
            {
                "op": "search_deck",
                "player": "self",
                "target_id": "searched_cards",
                "filter": raw_filter(query),
                "amount": amount,
                "reveal": True,
                "destination": "self.hand",
                "source_text": original,
            },
            {"op": "shuffle_deck", "player": "self", "source_text": original},
        ], []

    # Put up to N ... from discard pile into hand.
    m = re.fullmatch(r"Put up to (\d+) (.+?) from your discard pile into your hand\.?", t, re.I)
    if m:
        return [
            {
                "op": "choose_cards",
                "player": "self",
                "target_id": "chosen_discard_cards",
                "zone": "self.discard",
                "filter": raw_filter(m.group(2)),
                "amount": amount_up_to(int(m.group(1))),
                "source_text": original,
            },
            {
                "op": "move_card",
                "cards_ref": "chosen_discard_cards",
                "destination": "self.hand",
                "source_text": original,
            },
        ], []

    # Move up to N Energy from 1 of your Pokémon to another of your Pokémon.
    m = re.fullmatch(r"Move up to (\d+) Energy from 1 of your Pokémon to another of your Pokémon\.?", t, re.I)
    if m:
        return [
            {
                "op": "choose_target",
                "player": "self",
                "target_id": "energy_source_pokemon",
                "zone": "self.in_play",
                "filter": {"supertype": "Pokémon", "has_attached_energy": True},
                "amount": amount_exact(1),
                "source_text": original,
            },
            {
                "op": "choose_cards",
                "player": "self",
                "target_id": "chosen_energy_cards",
                "zone": "attached_to:energy_source_pokemon",
                "filter": {"supertype": "Energy"},
                "amount": amount_up_to(int(m.group(1))),
                "source_text": original,
            },
            {
                "op": "choose_target",
                "player": "self",
                "target_id": "energy_destination_pokemon",
                "zone": "self.in_play",
                "filter": {"supertype": "Pokémon", "not_ref": "energy_source_pokemon"},
                "amount": amount_exact(1),
                "source_text": original,
            },
            {
                "op": "move_energy",
                "cards_ref": "chosen_energy_cards",
                "destination_ref": "energy_destination_pokemon",
                "source_text": original,
            },
        ], []

    # Discard a Special Energy from each of opponent's Pokémon.
    if re.fullmatch(r"Discard a Special Energy from each of your opponent's Pokémon\.?", t, re.I):
        return [{
            "op": "discard_attached_cards",
            "player": "opponent",
            "from": "opponent.in_play",
            "per_pokemon": True,
            "filter": {"supertype": "Energy", "subtypes": ["Special"]},
            "amount": amount_exact(1),
            "source_text": original,
        }], []

    # Opponent reveals hand; draw per Pokémon found.
    if re.fullmatch(r"Your opponent reveals their hand, and you draw a card for each Pokémon you find there\.?", t, re.I):
        return [
            {"op": "reveal_hand", "player": "opponent", "source_text": original},
            {
                "op": "count_cards",
                "target_id": "opponent_hand_pokemon_count",
                "zone": "opponent.hand",
                "filter": {"supertype": "Pokémon"},
                "source_text": original,
            },
            {
                "op": "draw_cards",
                "player": "self",
                "amount": {"mode": "count_ref", "ref": "opponent_hand_pokemon_count"},
                "source_text": original,
            },
        ], []

    # Delayed retreat restriction for poisoned Pokémon.
    if re.fullmatch(r"During your opponent's next turn, their Poisoned Pokémon can't retreat\. \(This includes newly Poisoned Pokémon\.\)", t, re.I):
        return [{
            "op": "register_continuous_modifier",
            "modifier_id": "opponent_poisoned_pokemon_cannot_retreat_next_turn",
            "duration": {"until": "end_of_opponent_next_turn"},
            "applies_to": {
                "player": "opponent",
                "zone": "in_play",
                "filter": {"special_conditions": ["Poisoned"]},
                "include_newly_matching_objects": True,
            },
            "modification": {"retreat_allowed": False},
            "source_text": original,
        }], []

    # Look at top N, put K into hand, discard others.
    m = re.fullmatch(r"Look at the top (\d+) cards of your deck and put (\d+) of them into your hand\. Discard the other cards\.?", t, re.I)
    if m:
        return [
            {
                "op": "look_at_top_cards",
                "player": "self",
                "target_id": "looked_cards",
                "zone": "self.deck",
                "amount": amount_exact(int(m.group(1))),
                "source_text": original,
            },
            {
                "op": "choose_cards",
                "player": "self",
                "target_id": "chosen_cards_to_hand",
                "from_ref": "looked_cards",
                "amount": amount_exact(int(m.group(2))),
                "source_text": original,
            },
            {"op": "move_card", "cards_ref": "chosen_cards_to_hand", "destination": "self.hand", "source_text": original},
            {"op": "move_card", "cards_ref": "looked_cards - chosen_cards_to_hand", "destination": "self.discard", "source_text": original},
        ], []

    # Top N, choose a Pokémon of type to Bench, rest bottom.
    m = re.fullmatch(r"Look at the top (\d+) cards of your deck and put a (.+?) you find there onto your Bench\. Shuffle the other cards and put them on the bottom of your deck\. You can't use this card during your first turn\.?", t, re.I)
    if m:
        return [
            {
                "op": "look_at_top_cards",
                "player": "self",
                "target_id": "looked_cards",
                "zone": "self.deck",
                "amount": amount_exact(int(m.group(1))),
                "source_text": original,
            },
            {
                "op": "choose_cards",
                "player": "self",
                "target_id": "chosen_pokemon_to_bench",
                "from_ref": "looked_cards",
                "filter": raw_filter(m.group(2)),
                "amount": amount_exact(1),
                "source_text": original,
            },
            {
                "op": "put_card_on_bench",
                "player": "self",
                "cards_ref": "chosen_pokemon_to_bench",
                "source_text": original,
            },
            {
                "op": "shuffle_cards",
                "cards_ref": "looked_cards - chosen_pokemon_to_bench",
                "source_text": original,
            },
            {
                "op": "move_card",
                "cards_ref": "looked_cards - chosen_pokemon_to_bench",
                "destination": "self.deck.bottom",
                "source_text": original,
            },
        ], []

    # Simple self-damage on attacks.
    m = re.fullmatch(r"This Pokémon also does (\d+) damage to itself\.?", t, re.I)
    if m:
        return [{
            "op": "deal_damage",
            "target": "self_attacking_pokemon",
            "amount": int(m.group(1)),
            "source_text": original,
        }], []

    # Attack damage modifier: this attack does N more damage...
    m = re.match(r"^This attack does (\d+) more damage (.+)\.?$", t, re.I)
    if m:
        return [{
            "op": "modify_attack_damage",
            "amount": int(m.group(1)),
            "mode": "add",
            "condition": {"raw_text": m.group(2)},
            "source_text": original,
        }], []

    # Resistance ignore.
    if re.fullmatch(r"This attack's damage isn't affected by Resistance\.?", t, re.I):
        return [{
            "op": "ignore_resistance",
            "scope": "this_attack",
            "source_text": original,
        }], []

    # Coin flip -> next-turn prevention variants.
    if re.fullmatch(r"Flip a coin\. If heads, during your opponent's next turn, prevent all damage from and effects of attacks done to this Pokémon\.?,?", t, re.I):
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [delayed_modifier_step("prevent_attack_damage_and_effects_to_self_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True, "prevent_effects_of_attacks": True}, original)],
            "source_text": original,
        }], []

    if re.fullmatch(r"Flip a coin\. If heads, during your opponent's next turn, prevent all damage done to this Pokémon by attacks\.?,?", t, re.I):
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [delayed_modifier_step("prevent_attack_damage_to_self_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True}, original)],
            "source_text": original,
        }], []

    # Tool discard before damage.
    if re.fullmatch(r"Before doing damage, discard all Pokémon Tools from your opponent's Active Pokémon\.?,?", t, re.I):
        return [{
            "op": "discard_attached_cards",
            "target": "opponent.active",
            "filter": {"supertype": "Trainer", "subtypes": ["Tool"]},
            "amount": {"mode": "all"},
            "timing_note": "before_damage",
            "source_text": original,
        }], []

    # Revealing hand as a standalone effect.
    if re.fullmatch(r"Your opponent reveals their hand\.?,?", t, re.I):
        return [{"op": "reveal_hand", "player": "opponent", "source_text": original}], []

    # Combined ignore wording.
    if re.fullmatch(r"This attack's damage isn't affected by Weakness or Resistance, or by any effects on your opponent's Active Pokémon\.?,?", t, re.I):
        return [{"op": "ignore_weakness_resistance", "scope": "this_attack", "source_text": original}, {"op": "ignore_effects_on_defending_pokemon", "scope": "this_attack_damage", "source_text": original}], []

    # Damage to each opposing Benched Pokémon.
    m = re.fullmatch(r"This attack also does (\d+) damage to each of your opponent's Benched Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{
            "op": "deal_damage",
            "target": "opponent.bench.each",
            "amount": int(m.group(1)),
            "apply_weakness_resistance": False,
            "source_text": original,
        }], []

    # Extra common v0.2/v0.3 cleanups.
    m = re.fullmatch(r"Discard the top (\d+) cards of your opponent's deck\.?,?", t, re.I)
    if m:
        return [{
            "op": "discard_cards",
            "player": "opponent",
            "source_zone": "deck.top",
            "selection": amount_exact(int(m.group(1))),
            "destination": "discard",
            "source_text": original,
        }], []

    m = re.fullmatch(r"Discard the top (\d+) cards of your deck\.?,?", t, re.I)
    if m:
        return [{
            "op": "discard_cards",
            "player": "self",
            "source_zone": "deck.top",
            "selection": amount_exact(int(m.group(1))),
            "destination": "discard",
            "source_text": original,
        }], []

    if re.fullmatch(r"During your next turn, this Pokémon can't use attacks\.?,?", t, re.I):
        return [delayed_modifier_step("self_cannot_attack_next_turn", "self_attacking_pokemon", {"attack_allowed": False}, original)], []

    if re.fullmatch(r"Discard a Stadium in play\.?,?", t, re.I):
        return [{
            "op": "discard_stadium",
            "target": "stadium_in_play",
            "amount": amount_exact(1),
            "source_text": original,
        }], []

    if re.fullmatch(r"You may draw cards until you have (\d+) cards in your hand\.?,?", t, re.I):
        return [{
            "op": "draw_until_hand_size",
            "player": "self",
            "target_hand_size": int(re.fullmatch(r"You may draw cards until you have (\d+) cards in your hand\.?,?", t, re.I).group(1)),
            "optional": True,
            "source_text": original,
        }], []

    m = re.fullmatch(r"Search your deck for up to (\d+) cards and put them into your hand\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{
            "op": "search_deck",
            "player": "self",
            "target_id": "searched_cards",
            "filter": {"any_card": True},
            "amount": amount_up_to(int(m.group(1))),
            "reveal": False,
            "destination": "self.hand",
            "source_text": original,
        }, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Discard a Special Energy from your opponent's Active Pokémon\.?,?", t, re.I)
    if m:
        return [{
            "op": "discard_attached_energy",
            "target": "opponent.active",
            "filter": {"supertype": "Energy", "subtypes": ["Special"]},
            "amount": amount_exact(1),
            "source_text": original,
        }], []

    m = re.fullmatch(r"Put an Energy attached to this Pokémon into your hand\.?,?", t, re.I)
    if m:
        return [{
            "op": "choose_cards",
            "player": "self",
            "target_id": "chosen_energy_card",
            "zone": "attached_to:self_attacking_pokemon",
            "filter": {"supertype": "Energy"},
            "amount": amount_exact(1),
            "source_text": original,
        }, {
            "op": "move_card",
            "cards_ref": "chosen_energy_card",
            "destination": "self.hand",
            "source_text": original,
        }], []

    m = re.fullmatch(r"This Pokémon is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [status_condition_step(m.group(1), "self_attacking_pokemon", original)], []


    # v0.3: additional high-frequency cleanup rules from the full 15,098-group corpus.

    # Extra global / deck-construction / tournament boilerplate variants.
    if re.fullmatch(r"This Stadium stays in play when you play it\. Discard it if another Stadium comes into play\. If a Stadium with the same name is in play, you can't play this card\.?,?", t, re.I):
        return [{"op": "reference_global_rule", "global_rule_id": "play_stadium", "source_text": original}], []

    if re.fullmatch(r"This card stays in play after being played\. Discard this card if another Stadium card comes into play\.?,?", t, re.I):
        return [{"op": "reference_global_rule", "global_rule_id": "play_stadium", "source_text": original}], []

    if re.fullmatch(r"You may play only 1 Stadium card during your turn\. Put it (?:next to|into) the Active Spot, and discard it if another Stadium comes into play\. A Stadium with the same name can(?:'|’)t be played\.?,?", t, re.I):
        return [{"op": "reference_global_rule", "global_rule_id": "play_stadium", "source_text": original}], []

    if re.fullmatch(r"You may have as many of this card in your deck as you like\.?,?", t, re.I):
        return [{"op": "register_deck_construction_rule", "rule_id": "unlimited_copies_allowed", "source_text": original}], []

    if re.fullmatch(r"\(This card cannot be used at official tournaments\.\)\.?,?", t, re.I):
        return [{"op": "register_legality_note", "official_tournament_legal": False, "source_text": original}], []

    # Special Energy provision / type text.
    m = re.fullmatch(r"This card provides ([A-Za-z]+) Energy\.?,?", t, re.I)
    if m:
        return [{"op": "provide_energy", "types": [m.group(1).capitalize()], "amount": 1, "source_text": original}], []

    m = re.fullmatch(r"This Pokémon is both ([A-Za-z]+) ([A-Za-z]+) type\.?,?", t, re.I)
    if m:
        return [{
            "op": "register_continuous_modifier",
            "modifier_id": "self_has_additional_types",
            "target": "self",
            "duration": {"while_source_in_play": True},
            "modification": {"types": [m.group(1).capitalize(), m.group(2).capitalize()]},
            "source_text": original,
        }], []

    # Multi-status conditions.
    m = re.fullmatch(r"(?:The Defending Pokémon|Your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned) and (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [
            status_condition_step(m.group(1), "opponent.active", original),
            status_condition_step(m.group(2), "opponent.active", original),
        ], []

    m = re.fullmatch(r"Flip a coin\. If heads, (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned) and (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [
                status_condition_step(m.group(1), "opponent.active", original),
                status_condition_step(m.group(2), "opponent.active", original),
            ],
            "source_text": original,
        }], []

    m = re.fullmatch(r"Both Active Pokémon are now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [
            status_condition_step(m.group(1), "self.active", original),
            status_condition_step(m.group(1), "opponent.active", original),
        ], []

    m = re.fullmatch(r"(?:The Defending Pokémon|Your opponent's Active Pokémon) is now Poisoned\. Put (\d+) damage counters instead of 1 on (?:the Defending Pokémon|that Pokémon) between turns\.?,?", t, re.I)
    if m:
        return [
            status_condition_step("Poisoned", "opponent.active", original),
            {"op": "modify_special_condition", "target": "opponent.active", "condition": "Poisoned", "poison_damage_counters_between_turns": int(m.group(1)), "source_text": original},
        ], []

    # Coin flip / branch variants.
    m = re.fullmatch(r"Flip a coin\. If heads, this attack does (\d+) damage plus (\d+) more damage; if tails, this attack does \d+ damage\.?,?", t, re.I)
    if m:
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(2)), "source_text": original}],
            "source_text": original,
        }], []

    if re.fullmatch(r"Flip 2 coins\. If either of them is tails, this attack does nothing\.?,?", t, re.I):
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 2,
            "target_id": "coin_results",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_results",
            "if": "any_tails",
            "then": [{"op": "attack_does_nothing", "source_text": original}],
            "source_text": original,
        }], []

    if re.fullmatch(r"Flip a coin\. If heads, (?:the Defending Pokémon|your opponent's Active Pokémon) can't attack during your opponent's next turn\.?,?", t, re.I):
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [delayed_modifier_step("opponent_active_cannot_attack_next_turn", "opponent.active", {"attack_allowed": False}, original)],
            "source_text": original,
        }], []

    m = re.fullmatch(r"Flip a coin\. If tails, this Pokémon does (\d+) damage to itself\.?,?", t, re.I)
    if m:
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "tails",
            "then": [{"op": "deal_damage", "target": "self_attacking_pokemon", "amount": int(m.group(1)), "source_text": original}],
            "source_text": original,
        }], []

    if re.fullmatch(r"Flip a coin\. If heads, prevent all damage done to this Pokémon by attacks during your opponent's next turn\.?,?", t, re.I):
        return [{
            "op": "coin_flip",
            "player": "self",
            "count": 1,
            "target_id": "coin_result",
            "source_text": original,
        }, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [delayed_modifier_step("prevent_attack_damage_to_self_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True}, original)],
            "source_text": original,
        }], []

    # Attack lock / restriction variants.
    if re.fullmatch(r"Choose 1 of (?:the Defending Pokémon's|your opponent's Active Pokémon's) attacks\. (?:That Pokémon can't use that attack during your opponent's next turn|During your opponent's next turn, that Pokémon can't use that attack)\.?,?", t, re.I):
        return [{
            "op": "choose_attack",
            "player": "self",
            "target_id": "chosen_opponent_attack",
            "pokemon": "opponent.active",
            "source_text": original,
        }, delayed_modifier_step("opponent_active_cannot_use_chosen_attack_next_turn", "opponent.active", {"forbidden_attack_ref": "chosen_opponent_attack"}, original)], []

    if re.fullmatch(r"If the Defending Pokémon is a Basic Pokémon, it can't attack during your opponent's next turn\.?,?", t, re.I):
        return [delayed_modifier_step("opponent_basic_active_cannot_attack_next_turn", "opponent.active", {"attack_allowed": False, "condition": {"subtypes": ["Basic"]}}, original)], []

    m = re.fullmatch(r"During your opponent's next turn, (?:any damage done by attacks from the Defending Pokémon|the Defending Pokémon's attacks do) (?:is reduced by|)(?:\s*)(\d+) less damage \(before applying Weakness and Resistance\)\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("opponent_active_attack_damage_reduced_next_turn", "opponent.active", {"damage_done_by_attacks_delta": -int(m.group(1))}, original)], []

    m = re.fullmatch(r"During your opponent's next turn, (?:any damage done to this Pokémon by attacks is reduced by|this Pokémon takes) (\d+) less damage(?: from attacks)? \(after applying Weakness and Resistance\)\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("self_takes_less_damage_next_turn", "self_attacking_pokemon", {"damage_taken_from_attacks_delta": -int(m.group(1))}, original)], []

    if re.fullmatch(r"During your opponent's next turn, prevent all damage done to this Pokémon by attacks from Basic Pokémon\.?,?", t, re.I):
        return [delayed_modifier_step("prevent_attack_damage_from_basic_pokemon_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True, "attacker_filter": {"subtypes": ["Basic"]}}, original)], []

    if re.fullmatch(r"During your opponent's next turn, this Pokémon has no Weakness\.?,?", t, re.I):
        return [delayed_modifier_step("self_has_no_weakness_next_turn", "self_attacking_pokemon", {"weakness": None}, original)], []

    # Switch / gust variants.
    if re.fullmatch(r"Your opponent switches (?:his or her|their) Active Pokémon with 1 of (?:his or her|their) Benched Pokémon\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "opponent", "chooser": "opponent", "new_active_ref": "opponent_choice_from_bench", "source_text": original}], []

    if re.fullmatch(r"(?:Switch 1 of your opponent's Benched Pokémon with their Active Pokémon|Switch the Defending Pokémon with 1 of your opponent's Benched Pokémon)\.?,?", t, re.I):
        return [{
            "op": "choose_target",
            "player": "self",
            "target_id": "chosen_opponent_benched_pokemon",
            "zone": "opponent.bench",
            "filter": {"supertype": "Pokémon"},
            "amount": amount_exact(1),
            "source_text": original,
        }, {"op": "switch_active", "player": "opponent", "new_active_ref": "chosen_opponent_benched_pokemon", "source_text": original}], []

    if re.fullmatch(r"Your opponent switches the Defending Pokémon with 1 of (?:his or her|their) Benched Pokémon, if any\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "opponent", "chooser": "opponent", "optional": True, "new_active_ref": "opponent_choice_from_bench", "source_text": original}], []

    # Hand disruption / reveal variants.
    if re.fullmatch(r"Your opponent reveals (?:his or her|their) hand\.?,?", t, re.I):
        return [{"op": "reveal_hand", "player": "opponent", "source_text": original}], []

    if re.fullmatch(r"Choose a random card from your opponent's hand\. Your opponent reveals that card and shuffles it into their deck\.?,?", t, re.I):
        return [{
            "op": "choose_cards",
            "player": "self",
            "target_id": "random_opponent_hand_card",
            "zone": "opponent.hand",
            "selection": {"mode": "random", "amount": 1},
            "source_text": original,
        }, {"op": "reveal_cards", "cards_ref": "random_opponent_hand_card", "source_text": original}, {"op": "move_card", "cards_ref": "random_opponent_hand_card", "destination": "opponent.deck", "source_text": original}, {"op": "shuffle_deck", "player": "opponent", "source_text": original}], []

    if re.fullmatch(r"Choose 1 card from your opponent's hand without looking\. Look at the card you chose, then have your opponent shuffle that card into (?:his or her|their) deck\.?,?", t, re.I):
        return [{
            "op": "choose_cards",
            "player": "self",
            "target_id": "chosen_opponent_hand_card",
            "zone": "opponent.hand",
            "selection": {"mode": "hidden_choice", "amount": 1},
            "source_text": original,
        }, {"op": "look_at_cards", "player": "self", "cards_ref": "chosen_opponent_hand_card", "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_hand_card", "destination": "opponent.deck", "source_text": original}, {"op": "shuffle_deck", "player": "opponent", "source_text": original}], []

    if re.fullmatch(r"Choose 1 card from your opponent's hand without looking and discard it\.?,?", t, re.I):
        return [{
            "op": "discard_cards",
            "player": "opponent",
            "source_zone": "hand",
            "selection": {"mode": "hidden_choice", "amount": 1, "chooser": "self"},
            "destination": "discard",
            "source_text": original,
        }], []

    # Evolution/search/bench setup variants.
    if re.fullmatch(r"Search your deck for a card that evolves from this Pokémon and put it onto this Pokémon to evolve it\. Then, shuffle your deck\.?,?", t, re.I):
        return [{
            "op": "search_deck",
            "player": "self",
            "target_id": "searched_evolution_card",
            "filter": {"evolves_from": "self.name"},
            "amount": amount_exact(1),
            "reveal": False,
            "destination": "temporary.selection",
            "source_text": original,
        }, {"op": "evolve_pokemon", "target": "self", "evolution_card_ref": "searched_evolution_card", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for up to (\d+) Basic Pokémon and put them onto your Bench\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{
            "op": "search_deck",
            "player": "self",
            "target_id": "searched_basic_pokemon",
            "filter": {"supertype": "Pokémon", "subtypes": ["Basic"]},
            "amount": amount_up_to(int(m.group(1))),
            "reveal": False,
            "destination": "self.bench",
            "source_text": original,
        }, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for a basic Energy card, show it to your opponent, and put it into your hand\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{
            "op": "search_deck",
            "player": "self",
            "target_id": "searched_basic_energy",
            "filter": {"supertype": "Energy", "subtypes": ["Basic"]},
            "amount": amount_exact(1),
            "reveal": True,
            "destination": "self.hand",
            "source_text": original,
        }, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    # Damage to chosen Pokémon / spread variants.
    m = re.fullmatch(r"Choose 1 of your opponent's (Benched Pokémon|Pokémon)\. This attack does (\d+) damage to that Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        zone = "opponent.bench" if "Benched" in m.group(1) else "opponent.in_play"
        return [{
            "op": "choose_target",
            "player": "self",
            "target_id": "chosen_opponent_pokemon_for_damage",
            "zone": zone,
            "filter": {"supertype": "Pokémon"},
            "amount": amount_exact(1),
            "source_text": original,
        }, {"op": "deal_damage", "target_ref": "chosen_opponent_pokemon_for_damage", "amount": int(m.group(2)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"Choose 1 of your opponent's Pokémon\. This attack does (\d+) damage to that Pokémon\. This attack's damage isn't affected by Weakness(?:, Resistance, Poké-Powers, Poké-Bodies, or any other effects on that Pokémon| or Resistance)\.?,?", t, re.I)
    if m:
        return [{
            "op": "choose_target",
            "player": "self",
            "target_id": "chosen_opponent_pokemon_for_damage",
            "zone": "opponent.in_play",
            "filter": {"supertype": "Pokémon"},
            "amount": amount_exact(1),
            "source_text": original,
        }, {"op": "deal_damage", "target_ref": "chosen_opponent_pokemon_for_damage", "amount": int(m.group(1)), "apply_weakness_resistance": False, "ignore_effects_on_target": True, "source_text": original}], []

    m = re.fullmatch(r"Does (\d+) damage to (\d+) of your opponent's Benched Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{
            "op": "choose_target",
            "player": "self",
            "target_id": "chosen_opponent_bench_targets",
            "zone": "opponent.bench",
            "filter": {"supertype": "Pokémon"},
            "amount": amount_exact(int(m.group(2))),
            "source_text": original,
        }, {"op": "deal_damage", "target_ref": "chosen_opponent_bench_targets", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"(?:This attack does |Does )(\d+) damage to each of your opponent's Benched Pokémon(?: that has any damage counters on it)?\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "deal_damage", "target": "opponent.bench.each", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"(?:This attack does |Does )(\d+) damage to each of your Benched Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "deal_damage", "target": "self.bench.each", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"Does (\d+) damage to each Benched Pokémon \(both yours and your opponent's\)\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "deal_damage", "target": "each.bench", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"Does (\d+) damage to each Defending Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "deal_damage", "target": "opponent.active.each", "amount": int(m.group(1)), "source_text": original}], []

    # Damage counters / healing variants.
    m = re.fullmatch(r"Put (\d+) damage counters on your opponent's Pokémon in any way you like\.?,?", t, re.I)
    if m:
        return [{"op": "place_damage_counters", "player": "self", "target": "opponent.in_play.distributed", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"Put (\d+) damage counter(?:s)? on 1 of your opponent's Pokémon\.?,?", t, re.I)
    if m:
        return [{
            "op": "choose_target",
            "player": "self",
            "target_id": "chosen_opponent_pokemon_for_counters",
            "zone": "opponent.in_play",
            "filter": {"supertype": "Pokémon"},
            "amount": amount_exact(1),
            "source_text": original,
        }, {"op": "place_damage_counters", "target_ref": "chosen_opponent_pokemon_for_counters", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"Heal (\d+) damage from each of your Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "heal_damage", "target": "self.in_play.each", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"Remove (\d+) damage counter(?:s)? from each of your Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "heal_damage", "target": "self.in_play.each", "amount": {"mode": "damage_counters", "value": int(m.group(1))}, "source_text": original}], []

    # Energy attachment/discard variants.
    m = re.fullmatch(r"Discard a ([A-Za-z]+) Energy attached to this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "filter": {"types": [m.group(1).capitalize()]}, "amount": amount_exact(1), "source_text": original}], []

    if re.fullmatch(r"Attach a basic Energy card from your discard pile to 1 of your Benched Pokémon\.?,?", t, re.I):
        return [{
            "op": "choose_cards",
            "player": "self",
            "target_id": "chosen_basic_energy_from_discard",
            "zone": "self.discard",
            "filter": {"supertype": "Energy", "subtypes": ["Basic"]},
            "amount": amount_exact(1),
            "source_text": original,
        }, {"op": "choose_target", "player": "self", "target_id": "chosen_own_benched_pokemon", "zone": "self.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_basic_energy_from_discard", "target_ref": "chosen_own_benched_pokemon", "source_text": original}], []

    if re.fullmatch(r"Attach a basic Energy card from your discard pile to this Pokémon\.?,?", t, re.I):
        return [{
            "op": "choose_cards",
            "player": "self",
            "target_id": "chosen_basic_energy_from_discard",
            "zone": "self.discard",
            "filter": {"supertype": "Energy", "subtypes": ["Basic"]},
            "amount": amount_exact(1),
            "source_text": original,
        }, {"op": "attach_card", "cards_ref": "chosen_basic_energy_from_discard", "target": "self_attacking_pokemon", "source_text": original}], []

    # Draw/shuffle/discard pile retrieval variants.
    m = re.fullmatch(r"Shuffle your hand into your deck\. Then, draw (\d+) cards\.?,?", t, re.I)
    if m:
        return [{"op": "move_zone_to_zone", "player": "self", "source_zone": "hand", "destination_zone": "deck", "selection": {"mode": "all"}, "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}, {"op": "draw_cards", "player": "self", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    if re.fullmatch(r"Shuffle your hand into your deck\. Then, draw a number of cards equal to the number of cards in your opponent's hand\.?,?", t, re.I):
        return [{"op": "move_zone_to_zone", "player": "self", "source_zone": "hand", "destination_zone": "deck", "selection": {"mode": "all"}, "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}, {"op": "count_cards", "target_id": "opponent_hand_count", "zone": "opponent.hand", "source_text": original}, {"op": "draw_cards", "player": "self", "amount": {"mode": "count_ref", "ref": "opponent_hand_count"}, "source_text": original}], []

    if re.fullmatch(r"Put (?:any 1 card|a card) from your discard pile into your hand\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_discard_card", "zone": "self.discard", "filter": {"any_card": True}, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_discard_card", "destination": "self.hand", "source_text": original}], []

    if re.fullmatch(r"Flip a coin until you get tails\. For each heads, draw (?:a card|1 card)\.?,?", t, re.I):
        return [{"op": "coin_flip_until", "player": "self", "until": "tails", "target_id": "coin_results_until_tails", "source_text": original}, {"op": "draw_cards_per_coin_heads", "player": "self", "amount_per_heads": 1, "coin_results_ref": "coin_results_until_tails", "source_text": original}], []

    if re.fullmatch(r"Flip a coin until you get tails\. For each heads, discard the top card of your opponent's deck\.?,?", t, re.I):
        return [{"op": "coin_flip_until", "player": "self", "until": "tails", "target_id": "coin_results_until_tails", "source_text": original}, {"op": "discard_cards_per_coin_heads", "player": "opponent", "source_zone": "deck.top", "amount_per_heads": 1, "coin_results_ref": "coin_results_until_tails", "destination": "discard", "source_text": original}], []

    # More damage formula variants that depend on damage counters on this Pokémon.
    m = re.fullmatch(r"This attack does (\d+) (?:less|more) damage for each damage counter on this Pokémon\.?,?", t, re.I)
    if m:
        mode = "subtract" if "less damage" in t.lower() else "add"
        return [{"op": "modify_attack_damage_per_damage_counter", "target": "self_attacking_pokemon", "mode": mode, "amount_per_counter": int(m.group(1)), "source_text": original}], []


    # v0.4: additional cautious patterns from the v0.3 full-corpus review.

    # Older form-change Poké-Power pattern: Unown / Ditto / Deoxys switch with same-named card.
    m = re.fullmatch(
        r"Once during your turn \(before your attack\), you may search your deck for another ([A-Za-z0-9éÉ' .\-]+) and switch it with \1\. \(Any cards attached to \1, damage counters, Special Conditions, and effects on it are now on the new Pokémon\.\) If you do, put \1 on top of your deck\. Shuffle your deck afterward\. You can't use more than 1 ([A-Za-zéÉ .\-]+) Poké-Power each turn\.?,?",
        t,
        re.I,
    )
    if m:
        pokemon_name = m.group(1)
        power_name = m.group(2)
        return [{
            "op": "search_deck",
            "player": "self",
            "target_id": "searched_replacement_pokemon",
            "filter": {"supertype": "Pokémon", "name": pokemon_name},
            "amount": amount_exact(1),
            "reveal": False,
            "destination": "temporary.selection",
            "source_text": original,
        }, {
            "op": "swap_pokemon_card_with_deck_card",
            "old_pokemon": "self",
            "new_card_ref": "searched_replacement_pokemon",
            "put_old_card_on_top_of_deck": True,
            "preserve_attached_cards_damage_conditions_and_effects": True,
            "source_text": original,
        }, {
            "op": "shuffle_deck",
            "player": "self",
            "source_text": original,
        }, {
            "op": "register_usage_limit",
            "scope": "per_player_turn",
            "limit": 1,
            "group": power_name + " Poké-Power",
            "source_text": original,
        }], []

    # Damage from the Defending Pokémon's attacks is reduced next turn.
    m = re.fullmatch(r"During your opponent's next turn, any damage done by attacks from the Defending Pokémon is reduced by (\d+) \(before applying Weakness and Resistance\)\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("opponent_active_attack_damage_reduced_next_turn", "opponent.active", {"damage_done_by_attacks_delta": -int(m.group(1)), "timing_basis": "before_weakness_resistance"}, original)], []

    m = re.fullmatch(r"During your opponent's next turn, the Defending Pokémon's attacks do (\d+) less damage \(before applying Weakness and Resistance\)\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("opponent_active_attack_damage_reduced_next_turn", "opponent.active", {"damage_done_by_attacks_delta": -int(m.group(1)), "timing_basis": "before_weakness_resistance"}, original)], []

    # Additional attack-damage ignore clauses.
    if re.fullmatch(r"This attack's damage is not affected by Resistance\.?,?", t, re.I) or re.fullmatch(r"Don't apply Resistance\.?,?", t, re.I):
        return [{"op": "ignore_resistance", "scope": "this_attack", "source_text": original}], []

    if re.fullmatch(r"This attack's damage isn't affected by Weakness, Resistance, or any other effects on your opponent's Active Pokémon\.?,?", t, re.I):
        return [{"op": "ignore_weakness_resistance", "scope": "this_attack", "source_text": original}, {"op": "ignore_effects_on_defending_pokemon", "scope": "this_attack_damage", "source_text": original}], []

    if re.fullmatch(r"This attack's damage isn't affected by Resistance, Poké-Powers, Poké-Bodies, or any other effects on the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "ignore_resistance", "scope": "this_attack", "source_text": original}, {"op": "ignore_effects_on_defending_pokemon", "scope": "this_attack_damage", "legacy_effect_types": ["Poké-Powers", "Poké-Bodies"], "source_text": original}], []

    if re.fullmatch(r"This attack's damage isn't affected by any effects on the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "ignore_effects_on_defending_pokemon", "scope": "this_attack_damage", "source_text": original}], []

    if re.fullmatch(r"This attack's damage isn't affected by Weakness, Resistance, Pokémon Powers, or any other effects on the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "ignore_weakness_resistance", "scope": "this_attack", "source_text": original}, {"op": "ignore_effects_on_defending_pokemon", "scope": "this_attack_damage", "legacy_effect_types": ["Pokémon Powers"], "source_text": original}], []

    # Retreat and attack lock variants.
    if re.fullmatch(r"The Defending Pokémon can't retreat until the end of your opponent's next turn\.?,?", t, re.I):
        return [delayed_modifier_step("opponent_active_cannot_retreat_next_turn", "opponent.active", {"retreat_allowed": False}, original)], []

    if re.fullmatch(r"Your opponent's Active Pokémon is now Poisoned\. During your opponent's next turn, that Pokémon can't retreat\.?,?", t, re.I):
        return [status_condition_step("Poisoned", "opponent.active", original), delayed_modifier_step("opponent_active_cannot_retreat_next_turn", "opponent.active", {"retreat_allowed": False}, original)], []

    m = re.fullmatch(r"This Pokémon can't use ([A-Za-z0-9éÉ' .\-]+) during your next turn\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("self_cannot_use_named_attack_next_turn", "self_attacking_pokemon", {"forbidden_attack_name": m.group(1)}, original)], []

    m = re.fullmatch(r"During your next turn, this Pokémon can't use ([A-Za-z0-9éÉ' .\-]+)\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("self_cannot_use_named_attack_next_turn", "self_attacking_pokemon", {"forbidden_attack_name": m.group(1)}, original)], []

    if re.fullmatch(r"During your opponent's next turn, they can't play any Item cards from their hand\.?,?", t, re.I):
        return [delayed_modifier_step("opponent_cannot_play_items_next_turn", "opponent", {"play_item_allowed": False}, original)], []

    # Stadium discard variants.
    if re.fullmatch(r"You may discard any Stadium card in play\.?,?", t, re.I) or re.fullmatch(r"Discard any Stadium card in play\.?,?", t, re.I) or re.fullmatch(r"You may discard a Stadium in play\.?,?", t, re.I):
        return [{"op": "discard_stadium", "target": "stadium_in_play", "optional": t.lower().startswith("you may"), "amount": amount_exact(1), "source_text": original}], []

    # Fossil-style Trainer that plays as a Pokémon.
    m = re.fullmatch(r"Play this card as if it were a (\d+)-HP Basic ([A-Za-z]+) Pokémon\. This card can't be affected by any Special Conditions and can't retreat\. At any time during your turn, you may discard this card from play\.?,?", t, re.I)
    if m:
        return [{
            "op": "play_trainer_as_pokemon",
            "hp": int(m.group(1)),
            "stage": "Basic",
            "types": [m.group(2).capitalize()],
            "cannot_be_affected_by_special_conditions": True,
            "retreat_allowed": False,
            "self_discard_allowed_during_turn": True,
            "source_text": original,
        }], []

    # Baby Pokémon attack-announcement rule.
    if re.fullmatch(r"If (?:this Baby Pokémon|your Active Pokémon is a Baby Pokémon) (?:is your Active Pokémon and )?your opponent announces an attack, your opponent flips a coin \(before doing anything else\)\. If tails, your opponent's turn ends\.?,?", t, re.I):
        return [{
            "op": "register_trigger",
            "trigger": "opponent_announces_attack_against_this_active_baby_pokemon",
            "effect": {"op": "coin_flip", "player": "opponent", "on_tails": "opponent_turn_ends"},
            "source_text": original,
        }], []

    # Evolve from hand / special evolution text.
    m = re.fullmatch(r"Once during your turn \(before your attack\), you may put ([A-Za-z0-9éÉ' .\-]+) from your hand onto ([A-Za-z0-9éÉ' .\-]+) \(this counts as evolving \2\) and remove all damage counters from \2\.?,?", t, re.I)
    if m:
        return [{
            "op": "evolve_pokemon_from_hand",
            "evolution_card_name": m.group(1),
            "target_pokemon_name": m.group(2),
            "counts_as_evolving": True,
            "source_text": original,
        }, {
            "op": "heal_damage",
            "target": "evolved_pokemon",
            "amount": {"mode": "all"},
            "source_text": original,
        }], []

    if re.fullmatch(r"This Pokémon can evolve during your first turn or the turn you play it\.?,?", t, re.I):
        return [{"op": "register_evolution_permission", "target": "self", "can_evolve_first_turn": True, "can_evolve_turn_played": True, "source_text": original}], []

    # Return self and attachments to hand.
    if re.fullmatch(r"Put this Pokémon and all attached cards into your hand\.?,?", t, re.I):
        return [{"op": "move_pokemon_and_attached_cards", "target": "self_attacking_pokemon", "destination": "self.hand", "source_text": original}], []

    # Poison / multi-status variants with three Special Conditions or branch alternatives.
    m = re.fullmatch(r"(?:The Defending Pokémon|Your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned), (Asleep|Confused|Paralyzed|Poisoned|Burned), and (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [status_condition_step(m.group(i), "opponent.active", original) for i in (1, 2, 3)], []

    m = re.fullmatch(r"Flip a coin\. If heads, (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\. If tails, (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [status_condition_step(m.group(1), "opponent.active", original)], "else": [status_condition_step(m.group(2), "opponent.active", original)], "source_text": original}], []

    # Coin flip: tails do nothing, heads apply prevention / status.
    m = re.fullmatch(r"Flip a coin\. If tails, this attack does nothing\. If heads, (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "tails", "then": [{"op": "attack_does_nothing", "source_text": original}], "else": [status_condition_step(m.group(1), "opponent.active", original)], "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If tails, this attack does nothing\. If heads, prevent all effects of attacks, including damage, done to this Pokémon during your opponent's next turn\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "tails", "then": [{"op": "attack_does_nothing", "source_text": original}], "else": [delayed_modifier_step("prevent_attack_damage_and_effects_to_self_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True, "prevent_effects_of_attacks": True}, original)], "source_text": original}], []

    # Two-coin and self-damage optional damage variants.
    m = re.fullmatch(r"Flip 2 coins\. This attack does (\d+) damage plus (\d+) more damage for each heads\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 2, "target_id": "coin_results", "source_text": original}, {"op": "modify_attack_damage_per_coin_heads", "mode": "add", "amount_per_heads": int(m.group(2)), "base_damage_text": int(m.group(1)), "coin_results_ref": "coin_results", "source_text": original}], []

    m = re.fullmatch(r"You may do (\d+) more damage\. If you do, this Pokémon does (\d+) damage to itself\.?,?", t, re.I)
    if m:
        return [{"op": "choose_yes_no", "player": "self", "target_id": "choose_extra_damage", "source_text": original}, {"op": "branch_on_choice", "choice_ref": "choose_extra_damage", "if": True, "then": [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "source_text": original}, {"op": "deal_damage", "target": "self_attacking_pokemon", "amount": int(m.group(2)), "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, this attack does (\d+) more damage\. If tails, this Pokémon does (\d+) damage to itself\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "source_text": original}], "else": [{"op": "deal_damage", "target": "self_attacking_pokemon", "amount": int(m.group(2)), "source_text": original}], "source_text": original}], []

    # Broad chosen-target damage variants, including multiple targets.
    m = re.fullmatch(r"(?:Choose 1 of your opponent's Pokémon\. )?This attack does (\d+) damage to (\d+) of your opponent's Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_pokemon_targets", "zone": "opponent.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(int(m.group(2))), "source_text": original}, {"op": "deal_damage", "target_ref": "chosen_opponent_pokemon_targets", "amount": int(m.group(1)), "apply_weakness_resistance_to_bench": False, "source_text": original}], []

    m = re.fullmatch(r"This attack (?:also )?does (\d+) damage to (\d+) of your opponent's Benched Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_bench_targets", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(int(m.group(2))), "source_text": original}, {"op": "deal_damage", "target_ref": "chosen_opponent_bench_targets", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"If your opponent has any Benched Pokémon, choose 1 of them and this attack does (\d+) damage to it\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_bench_target", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "optional_if_no_targets": True, "source_text": original}, {"op": "deal_damage", "target_ref": "chosen_opponent_bench_target", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"This attack also does (\d+) damage to each of your Benched Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "deal_damage", "target": "self.bench.each", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"Does (\d+) damage to each of your opponent's Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "deal_damage", "target": "opponent.in_play.each", "amount": int(m.group(1)), "apply_weakness_resistance_to_bench": False, "source_text": original}], []

    # Opponent switches after damage / optional opponent bench switch.
    if re.fullmatch(r"If your opponent has any Benched Pokémon, (?:he or she chooses 1 of them and switches it with the Defending Pokémon|choose 1 of them and switches it with the Defending Pokémon)\. \(Do the damage before switching the Pokémon\.\)\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "opponent", "chooser": "opponent", "optional_if_no_targets": True, "new_active_ref": "opponent_choice_from_bench", "timing_note": "after_damage", "source_text": original}], []

    if re.fullmatch(r"Your opponent switches the Defending Pokémon with 1 of his or her Benched Pokémon, if any\. \(Do the damage before switching the Pokémon\.\)\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "opponent", "chooser": "opponent", "optional_if_no_targets": True, "new_active_ref": "opponent_choice_from_bench", "timing_note": "after_damage", "source_text": original}], []

    if re.fullmatch(r"You may have your opponent switch his or her Active Pokémon with 1 of his or her Benched Pokémon\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "opponent", "chooser": "opponent", "optional": True, "new_active_ref": "opponent_choice_from_bench", "source_text": original}], []

    # Discard/mill/search/retrieval variants.
    if re.fullmatch(r"Discard the top card of your deck\.?,?", t, re.I):
        return [{"op": "discard_cards", "player": "self", "source_zone": "deck.top", "selection": amount_exact(1), "destination": "discard", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for an Evolution card, show it to your opponent, and put it into your hand\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_evolution_card", "filter": {"supertype": "Pokémon", "is_evolution": True}, "amount": amount_exact(1), "reveal": True, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for a Pokémon, reveal it, and put it into your hand\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_pokemon", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "reveal": True, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for a basic Energy card, show it to your opponent, and put it into your hand\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_basic_energy", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(1), "reveal": True, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Shuffle your hand into your deck\. Then, draw a card for each card in your opponent's hand\.?,?", t, re.I)
    if m:
        return [{"op": "move_zone_to_zone", "player": "self", "source_zone": "hand", "destination_zone": "deck", "selection": {"mode": "all"}, "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}, {"op": "count_cards", "target_id": "opponent_hand_count", "zone": "opponent.hand", "source_text": original}, {"op": "draw_cards", "player": "self", "amount": {"mode": "count_ref", "ref": "opponent_hand_count"}, "source_text": original}], []

    if re.fullmatch(r"Put a Trainer card from your discard pile into your hand\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_trainer_from_discard", "zone": "self.discard", "filter": {"supertype": "Trainer"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_trainer_from_discard", "destination": "self.hand", "source_text": original}], []

    if re.fullmatch(r"Put a Supporter card from your discard pile into your hand\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_supporter_from_discard", "zone": "self.discard", "filter": {"supertype": "Trainer", "subtypes": ["Supporter"]}, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_supporter_from_discard", "destination": "self.hand", "source_text": original}], []

    # Energy attachment / movement variants.
    if re.fullmatch(r"Attach an Energy card from your discard pile to this Pokémon\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_energy_from_discard", "zone": "self.discard", "filter": {"supertype": "Energy"}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_energy_from_discard", "target": "self_attacking_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Attach a ([A-Za-z]+) Energy card from your discard pile to this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_typed_energy_from_discard", "zone": "self.discard", "filter": {"supertype": "Energy", "types": [m.group(1).capitalize()]}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_typed_energy_from_discard", "target": "self_attacking_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Discard (\d+) ([A-Za-z]+) Energy from this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "filter": {"types": [m.group(2).capitalize()]}, "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"Search your deck for a ([A-Za-z]+) Energy card and attach it to this Pokémon\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_energy", "filter": {"supertype": "Energy", "types": [m.group(1).capitalize()]}, "amount": amount_exact(1), "reveal": False, "destination": "temporary.selection", "source_text": original}, {"op": "attach_card", "cards_ref": "searched_energy", "target": "self_attacking_pokemon", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for a ([A-Za-z]+) Energy card and attach it to 1 of your Pokémon\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_energy", "filter": {"supertype": "Energy", "types": [m.group(1).capitalize()]}, "amount": amount_exact(1), "reveal": False, "destination": "temporary.selection", "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "searched_energy", "target_ref": "chosen_own_pokemon", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    if re.fullmatch(r"You may put an Energy attached to your opponent's Active Pokémon into their hand\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_opponent_active_energy", "zone": "attached_to:opponent.active", "filter": {"supertype": "Energy"}, "amount": amount_exact(1), "optional": True, "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_active_energy", "destination": "opponent.hand", "source_text": original}], []

    if re.fullmatch(r"You may move an Energy from your opponent's Active Pokémon to 1 of their Benched Pokémon\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_opponent_active_energy", "zone": "attached_to:opponent.active", "filter": {"supertype": "Energy"}, "amount": amount_exact(1), "optional": True, "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_energy", "cards_ref": "chosen_opponent_active_energy", "destination_ref": "chosen_opponent_benched_pokemon", "source_text": original}], []

    if re.fullmatch(r"Move all Energy from this Pokémon to your Benched Pokémon in any way you like\.?,?", t, re.I):
        return [{"op": "move_energy", "cards": "all_attached_to:self_attacking_pokemon", "destination": "self.bench.distributed", "source_text": original}], []

    # v0.10: opponent Energy bounce/move wording variants.
    if re.fullmatch(r"Put an Energy attached to (?:your opponent's Active Pokémon|the Defending Pokémon) into (?:your opponent's|their) hand\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_opponent_active_energy", "zone": "attached_to:opponent.active", "filter": {"supertype": "Energy"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_active_energy", "destination": "opponent.hand", "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If heads, put an Energy attached to (?:your opponent's Active Pokémon|the Defending Pokémon) into (?:your opponent's|their) hand\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_cards", "player": "self", "target_id": "chosen_opponent_active_energy", "zone": "attached_to:opponent.active", "filter": {"supertype": "Energy"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_active_energy", "destination": "opponent.hand", "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"Move an Energy(?: card)? attached to (?:your opponent's Active Pokémon|the Defending Pokémon) to 1 of (?:your opponent's|their) Benched Pokémon\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_opponent_active_energy", "zone": "attached_to:opponent.active", "filter": {"supertype": "Energy"}, "amount": amount_exact(1), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_energy", "cards_ref": "chosen_opponent_active_energy", "destination_ref": "chosen_opponent_benched_pokemon", "source_text": original}], []

    # Damage/healing formulas.
    m = re.fullmatch(r"(?:This attack does |Does )(\d+) damage for each damage counter on this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_per_damage_counter", "target": "self_attacking_pokemon", "damage_per_counter": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"(?:This attack does |Does )(\d+) more damage for each damage counter on this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage_per_damage_counter", "target": "self_attacking_pokemon", "mode": "add", "amount_per_counter": int(m.group(1)), "source_text": original}], []

    if re.fullmatch(r"Heal from this Pokémon the same amount of damage you did to the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "heal_damage", "target": "self_attacking_pokemon", "amount": {"mode": "damage_dealt_this_attack", "target": "opponent.active"}, "source_text": original}], []

    m = re.fullmatch(r"Heal (\d+) damage from each of your Benched Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "heal_damage", "target": "self.bench.each", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"Remove (\d+) damage counter(?:s)? from 1 of your Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "heal_damage", "target_ref": "chosen_own_pokemon", "amount": {"mode": "damage_counters", "value": int(m.group(1))}, "source_text": original}], []

    # Triggered retaliation when damaged.
    m = re.fullmatch(r"If this Pokémon is (?:your Active Pokémon|in the Active Spot) and is damaged by an attack from your opponent's Pokémon? \(even if this Pokémon is Knocked Out\), put (\d+) damage counters on the Attacking Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "register_trigger", "trigger": "self_active_damaged_by_opponent_attack", "effect": {"op": "place_damage_counters", "target": "attacking_pokemon", "amount": amount_exact(int(m.group(1)))}, "source_text": original}], []

    if re.fullmatch(r"If this Pokémon is your Active Pokémon and is damaged by an opponent's attack \(even if this Pokémon is Knocked Out\), the Attacking Pokémon is now Poisoned\.?,?", t, re.I):
        return [{"op": "register_trigger", "trigger": "self_active_damaged_by_opponent_attack", "effect": status_condition_step("Poisoned", "attacking_pokemon", original), "source_text": original}], []

    # Conditional play/cost restriction lines.
    if re.fullmatch(r"You can play this card only when it is the last card in your hand\.?,?", t, re.I):
        return [{"op": "register_play_condition", "condition": {"self.hand_size_before_play": 1}, "source_text": original}], []

    if re.fullmatch(r"You can play this card only if you discard 2 other cards from your hand\.?,?", t, re.I):
        return [{"op": "register_play_condition", "condition": {"requires_discard_other_cards_from_hand": 2}, "source_text": original}], []

    if re.fullmatch(r"Discard a card from your hand\. If you can't discard a card from your hand, you can't play this card\.?,?", t, re.I):
        return [{"op": "discard_cards", "player": "self", "source_zone": "hand", "selection": amount_exact(1), "destination": "discard", "required_to_play": True, "source_text": original}], []

    # Tool / technical-machine attack grant line.
    if re.fullmatch(r"The Pokémon this card is attached to can use the attack on this card\. \(You still need the necessary Energy to use this attack\.\) If this card is attached to 1 of your Pokémon, discard it at the end of your turn\.?,?", t, re.I):
        return [{"op": "grant_attached_card_attack", "target": "attached_pokemon", "requires_attack_energy_cost": True, "discard_this_card_at_end_of_turn_if_attached_to_own_pokemon": True, "source_text": original}], []

    # Copy opponent attack.
    if re.fullmatch(r"Choose 1 of your opponent's Active Pokémon's attacks and use it as this attack\.?,?", t, re.I):
        return [{"op": "choose_attack", "player": "self", "target_id": "chosen_opponent_attack", "pokemon": "opponent.active", "source_text": original}, {"op": "copy_and_use_attack", "attack_ref": "chosen_opponent_attack", "source_text": original}], []

    # Top-deck rearrange.
    m = re.fullmatch(r"Look at the top (\d+) cards of either player's deck and put them back (?:on top of that player's deck )?in any order\.?,?", t, re.I)
    if m:
        return [{"op": "choose_player", "player": "self", "target_id": "chosen_player", "choices": ["self", "opponent"], "source_text": original}, {"op": "look_at_top_cards", "player_ref": "chosen_player", "target_id": "looked_cards", "amount": amount_exact(int(m.group(1))), "source_text": original}, {"op": "reorder_cards", "cards_ref": "looked_cards", "destination": "chosen_player.deck.top", "source_text": original}], []



    # v0.5: cautious low-frequency but recurring patterns from the v0.4 full-corpus review.

    # Defensive flip prevention: if damaged by attacks, flip to prevent that damage.
    if re.fullmatch(r"If any damage is done to this Pokémon by attacks, flip a coin\. If heads, prevent that damage\.?,?", t, re.I):
        return [{
            "op": "register_replacement_effect",
            "replacement_id": "flip_to_prevent_attack_damage_to_self",
            "trigger": "self_would_take_attack_damage",
            "condition": {"damage_amount_gt": 0},
            "effect": {
                "op": "coin_flip_then_prevent_damage",
                "player": "self",
                "if": "heads",
                "prevent_damage": True,
            },
            "source_text": original,
        }], []

    # Ignore all attack Energy costs when discard-pile Tool threshold is met.
    m = re.fullmatch(r"If you have (\d+) or more Pokémon Tool cards in your discard pile, ignore all Energy in the attack cost of each of this Pokémon's attacks\.?,?", t, re.I)
    if m:
        return [{
            "op": "register_continuous_modifier",
            "modifier_id": "ignore_attack_energy_cost_if_tool_threshold_met",
            "target": "self",
            "duration": {"while_source_in_play": True},
            "condition": {"self.discard": {"filter": {"supertype": "Trainer", "subtypes": ["Tool"]}, "count_gte": int(m.group(1))}},
            "modification": {"ignore_attack_energy_cost": True},
            "source_text": original,
        }], []

    # Baby Pokémon attack-announcement rule variants.
    if re.fullmatch(r"If (?:your Active Pokémon is a Baby Pokémon|this Baby Pokémon is your Active Pokémon) and your opponent (?:announces an attack|tries to attack), your opponent flips a coin \(before (?:doing anything else|doing anything required in order to use that attack)\)\. If tails, your opponent's turn ends(?: without an attack)?\.?,?", t, re.I):
        return [{
            "op": "register_replacement_effect",
            "replacement_id": "baby_pokemon_attack_attempt_requires_heads",
            "trigger": "opponent_announces_attack_against_this_active",
            "condition": {"self_active_subtypes_include": "Baby"},
            "effect": {"op": "coin_flip", "player": "opponent", "if_tails": "end_opponent_turn_without_attack"},
            "source_text": original,
        }], []

    # Damage counters directly on opponent's Active Pokémon.
    m = re.fullmatch(r"Put (\d+) damage counter(?:s)? on your opponent's Active Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "place_damage_counters", "target": "opponent.active", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    # Before damage, discard all Trainer cards attached to Defending Pokémon.
    if re.fullmatch(r"Before doing damage, discard all Trainer cards attached to the Defending Pokémon\.?,?", t, re.I):
        return [{
            "op": "discard_attached_cards",
            "target": "opponent.active",
            "filter": {"supertype": "Trainer"},
            "amount": {"mode": "all"},
            "timing": "before_damage",
            "source_text": original,
        }], []

    # Next-turn shield threshold.
    m = re.fullmatch(r"During your opponent's next turn, if this Pokémon would be damaged by an attack, prevent that attack's damage done to this Pokémon if that damage is (\d+) or less\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step(
            "prevent_attack_damage_to_self_if_at_most_threshold_next_turn",
            "self_attacking_pokemon",
            {"prevent_attack_damage_if_lte": int(m.group(1))},
            original,
        )], []

    # Curly apostrophe bench-protection variant.
    if re.fullmatch(r"As long as this Pokémon is on your Bench, prevent all damage done to this Pokémon by attacks \(both yours and your opponent(?:'|’)s\)\.?,?", t, re.I):
        return [{
            "op": "register_continuous_modifier",
            "modifier_id": "prevent_bench_damage_to_self",
            "target": "self",
            "condition": {"zone": "bench"},
            "duration": {"while_source_in_play": True},
            "modification": {"prevent_damage_from_attacks": True},
            "source_text": original,
        }], []

    # Defending Pokémon has Energy -> choose and discard one.
    if re.fullmatch(r"If the Defending Pokémon has any Energy cards attached to it, choose 1 of them and discard it\.?,?", t, re.I):
        return [{
            "op": "choose_cards",
            "player": "self",
            "target_id": "chosen_defending_energy",
            "zone": "attached_to:opponent.active",
            "filter": {"supertype": "Energy"},
            "amount": amount_exact(1),
            "condition": {"opponent.active.has_attached_energy": True},
            "source_text": original,
        }, {"op": "discard_card", "cards_ref": "chosen_defending_energy", "source_text": original}], []

    # Each Defending Pokémon status condition, for older multi-defending wording.
    m = re.fullmatch(r"Each Defending Pokémon is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [status_condition_step(m.group(1), "opponent.defending.each", original)], []

    # Coin flip -> opponent can't attack or retreat.
    if re.fullmatch(r"Flip a coin\. If heads, the Defending Pokémon can't attack or retreat during your opponent's next turn\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {
            "op": "branch_on_result",
            "result_ref": "coin_result",
            "if": "heads",
            "then": [delayed_modifier_step("opponent_active_cannot_attack_or_retreat_next_turn", "opponent.active", {"attack_allowed": False, "retreat_allowed": False}, original)],
            "source_text": original,
        }], []

    # GX attack global limit boilerplate.
    if re.fullmatch(r"\(You can't use more than 1 GX attack in a game\.\)\.?,?", t, re.I):
        return [{"op": "reference_global_rule", "global_rule_id": "gx_attack_once_per_game", "source_text": original}], []

    # Attack attempt requires heads variants.
    if re.fullmatch(r"During your opponent's next turn, if the Defending Pokémon tries to (?:use an attack|attack), your opponent flips a coin\. If tails, that attack doesn't happen\.?,?", t, re.I):
        return [delayed_modifier_step("opponent_active_attack_requires_heads_next_turn", "opponent.active", {"attack_requires_coin_flip": {"on_tails": "attack_does_not_happen"}}, original)], []

    # Search deck for any card / up to N cards to hand.
    m = re.fullmatch(r"Search your deck for (?:a|1) card and put it into your hand\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_card", "filter": {"any_card": True}, "amount": amount_exact(1), "reveal": False, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"You may search your deck for up to (\d+) cards and put them into your hand\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_cards", "filter": {"any_card": True}, "amount": amount_up_to(int(m.group(1))), "reveal": False, "destination": "self.hand", "optional": True, "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    # Prevent effects of attacks, damage exception variants.
    if re.fullmatch(r"Prevent all effects of attacks used by your opponent's Pokémon done to this Pokémon\. \(Damage is not an effect\.\)\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "prevent_effects_of_attacks_to_self_except_damage", "target": "self", "duration": {"while_source_in_play": True}, "modification": {"prevent_effects_of_attacks": True, "damage_is_not_prevented": True}, "source_text": original}], []

    if re.fullmatch(r"Prevent all effects of your opponent's attacks, except damage, done to this Pokémon\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "prevent_effects_of_attacks_to_self_except_damage", "target": "self", "duration": {"while_source_in_play": True}, "modification": {"prevent_effects_of_attacks": True, "damage_is_not_prevented": True}, "source_text": original}], []

    # Stadium / Pokémon immunity to Item/Supporter effects.
    m = re.fullmatch(r"Whenever (?:any player|your opponent) plays an Item or Supporter card from (?:their|his or her) hand, prevent all effects of that card done to this (Stadium card|Pokémon)\.?,?", t, re.I)
    if m:
        target = "self" if "Pokémon" in m.group(1) else "this_stadium"
        return [{"op": "register_continuous_modifier", "modifier_id": "prevent_item_supporter_effects_to_self", "target": target, "duration": {"while_source_in_play": True}, "modification": {"prevent_effects_from_item_or_supporter_cards": True}, "source_text": original}], []

    # Do not apply W/R for this attack explanatory wording.
    if re.fullmatch(r"Don't apply Weakness and Resistance for this attack\. \(Any other effects that would happen after applying Weakness and Resistance still happen\.\)\.?,?", t, re.I):
        return [{"op": "ignore_weakness_resistance", "scope": "this_attack", "source_text": original}], []

    # Choose 1 Special Condition after heads.
    if re.fullmatch(r"Flip a coin\. If heads, choose 1 Special Condition\. The Defending Pokémon is now affected by that Special Condition\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_special_condition", "player": "self", "target_id": "chosen_special_condition", "choices": ["Asleep", "Burned", "Confused", "Paralyzed", "Poisoned"], "source_text": original}, {"op": "apply_special_condition", "target": "opponent.active", "condition_ref": "chosen_special_condition", "source_text": original}], "source_text": original}], []

    # Count opponent's hand -> damage counters.
    if re.fullmatch(r"Count the number of cards in your opponent's hand\. Put that many damage counters on the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "count_cards", "target_id": "opponent_hand_count", "zone": "opponent.hand", "source_text": original}, {"op": "place_damage_counters", "target": "opponent.active", "amount": {"mode": "count_ref", "ref": "opponent_hand_count"}, "source_text": original}], []

    # Flip heads -> each Defending Pokémon paralyzed.
    m = re.fullmatch(r"Flip a coin\. If heads, each Defending Pokémon is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [status_condition_step(m.group(1), "opponent.defending.each", original)], "source_text": original}], []

    # Conditional attack does nothing if defending not Asleep.
    if re.fullmatch(r"If the Defending Pokémon is not Asleep, this attack does nothing\.?,?", t, re.I):
        return [{"op": "conditional_attack_does_nothing", "condition": {"opponent.active.special_condition_not": "Asleep"}, "source_text": original}], []

    # Flip 2 coins both tails -> attack does nothing.
    if re.fullmatch(r"Flip 2 coins\. If both of them are tails, this attack does nothing\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 2, "target_id": "coin_results", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_results", "if": "all_tails", "then": [{"op": "attack_does_nothing", "source_text": original}], "source_text": original}], []


    # v0.9: generic costed / conditional activated Ability patterns.
    # Example: Lunar Cycle — "Once during your turn, if you have Solrock in play,
    # you may discard a Basic Fighting Energy card from your hand in order to use
    # this Ability. Draw 3 cards. You can't use more than 1 Lunar Cycle Ability each turn."
    m = re.fullmatch(
        r"Once during your turn(?: \(before your attack\))?, if you have ([A-Za-z0-9éÉ' .\-]+) in play, you may discard (?:a|an|1) Basic ([A-Za-z]+) Energy card from your hand in order to use this Ability\. Draw (\d+) cards?\. You can't use more than (?:1|one) ([A-Za-z0-9éÉ' .\-]+) Ability each turn\.?,?",
        t,
        re.I,
    )
    if m:
        required_pokemon = normalize_space(m.group(1))
        energy_type = m.group(2).capitalize()
        draw_n = int(m.group(3))
        ability_name = normalize_space(m.group(4))
        return [{
            "op": "register_usage_limit",
            "scope": "per_player_turn",
            "limit": 1,
            "group": ability_name + " Ability",
            "ability_name": ability_name,
            "source_text": original,
        }, {
            "op": "play_condition",
            "condition": {
                "requires_pokemon_in_play": {
                    "player": "self",
                    "name": required_pokemon,
                }
            },
            "required_to_play": True,
            "source_text": original,
        }, {
            "op": "discard_cards",
            "player": "self",
            "source_zone": "hand",
            "selection": {
                "mode": "exact",
                "value": 1,
                "filter": {
                    "supertype": "Energy",
                    "subtypes": ["Basic"],
                    "types": [energy_type],
                    "energy_type": energy_type,
                },
            },
            "destination": "self.discard",
            "cost": True,
            "required_to_play": True,
            "source_text": original,
        }, {
            "op": "draw_cards",
            "player": "self",
            "amount": amount_exact(draw_n),
            "optional": True,
            "source_text": original,
        }], []

    m = re.fullmatch(
        r"Once during your turn(?: \(before your attack\))?, you may discard (?:a|an|1) Basic ([A-Za-z]+) Energy card from your hand in order to use this Ability\. Draw (\d+) cards?\. You can't use more than (?:1|one) ([A-Za-z0-9éÉ' .\-]+) Ability each turn\.?,?",
        t,
        re.I,
    )
    if m:
        energy_type = m.group(1).capitalize()
        draw_n = int(m.group(2))
        ability_name = normalize_space(m.group(3))
        return [{
            "op": "register_usage_limit",
            "scope": "per_player_turn",
            "limit": 1,
            "group": ability_name + " Ability",
            "ability_name": ability_name,
            "source_text": original,
        }, {
            "op": "discard_cards",
            "player": "self",
            "source_zone": "hand",
            "selection": {
                "mode": "exact",
                "value": 1,
                "filter": {
                    "supertype": "Energy",
                    "subtypes": ["Basic"],
                    "types": [energy_type],
                    "energy_type": energy_type,
                },
            },
            "destination": "self.discard",
            "cost": True,
            "required_to_play": True,
            "source_text": original,
        }, {
            "op": "draw_cards",
            "player": "self",
            "amount": amount_exact(draw_n),
            "optional": True,
            "source_text": original,
        }], []

    m = re.fullmatch(
        r"Once during your turn(?: \(before your attack\))?, if you have ([A-Za-z0-9éÉ' .\-]+) in play, you may draw (\d+) cards?\. You can't use more than (?:1|one) ([A-Za-z0-9éÉ' .\-]+) Ability each turn\.?,?",
        t,
        re.I,
    )
    if m:
        required_pokemon = normalize_space(m.group(1))
        draw_n = int(m.group(2))
        ability_name = normalize_space(m.group(3))
        return [{
            "op": "register_usage_limit",
            "scope": "per_player_turn",
            "limit": 1,
            "group": ability_name + " Ability",
            "ability_name": ability_name,
            "source_text": original,
        }, {
            "op": "play_condition",
            "condition": {
                "requires_pokemon_in_play": {
                    "player": "self",
                    "name": required_pokemon,
                }
            },
            "required_to_play": True,
            "source_text": original,
        }, {
            "op": "draw_cards",
            "player": "self",
            "amount": amount_exact(draw_n),
            "optional": True,
            "source_text": original,
        }], []

    m = re.fullmatch(
        r"Once during your turn(?: \(before your attack\))?, you may discard (?:a|1) cards? from your hand in order to use this Ability\. Draw (\d+) cards?\. You can't use more than (?:1|one) ([A-Za-z0-9éÉ' .\-]+) Ability each turn\.?,?",
        t,
        re.I,
    )
    if m:
        draw_n = int(m.group(1))
        ability_name = normalize_space(m.group(2))
        return [{
            "op": "register_usage_limit",
            "scope": "per_player_turn",
            "limit": 1,
            "group": ability_name + " Ability",
            "ability_name": ability_name,
            "source_text": original,
        }, {
            "op": "discard_cards",
            "player": "self",
            "source_zone": "hand",
            "selection": {"mode": "exact", "value": 1, "filter": {"any_card": True}},
            "destination": "self.discard",
            "cost": True,
            "required_to_play": True,
            "source_text": original,
        }, {
            "op": "draw_cards",
            "player": "self",
            "amount": amount_exact(draw_n),
            "optional": True,
            "source_text": original,
        }], []

    # Simple once-per-turn draw ability.
    if re.fullmatch(r"Once during your turn \(before your attack\), you may draw a card\.?,?", t, re.I):
        return [{"op": "draw_cards", "player": "self", "amount": amount_exact(1), "optional": True, "usage_limit": {"scope": "per_turn_per_copy", "max": 1}, "source_text": original}], []

    # Conditional damage modifiers.
    m = re.fullmatch(r"If you played a Supporter card from your hand during this turn, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"supporter_played_from_hand_this_turn": True}, "source_text": original}], []

    m = re.fullmatch(r"If your opponent's Active Pokémon is an Evolution Pokémon, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"opponent.active.is_evolution": True}, "source_text": original}], []

    m = re.fullmatch(r"If your opponent's Active Pokémon already has any damage counters on it, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"opponent.active.damage_counters_gt": 0}, "source_text": original}], []

    # Attach N basic Energy cards from discard to one Benched Pokémon.
    m = re.fullmatch(r"Attach (\d+) basic Energy cards from your discard pile to 1 of your Benched Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_own_benched_pokemon", "zone": "self.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "choose_cards", "player": "self", "target_id": "chosen_basic_energy_from_discard", "zone": "self.discard", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(int(m.group(1))), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_basic_energy_from_discard", "target_ref": "chosen_own_benched_pokemon", "source_text": original}], []

    # Shuffle this Pokémon and attached cards to deck / hand bounce variant.
    if re.fullmatch(r"You may shuffle this Pokémon and all cards attached to it into your deck\.?,?", t, re.I):
        return [{"op": "move_card", "cards": "self_pokemon_and_attached_cards", "destination": "self.deck", "optional": True, "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    if re.fullmatch(r"Put this Pokémon and all attached cards into your hand\.?,?", t, re.I):
        return [{"op": "move_card", "cards": "self_pokemon_and_attached_cards", "destination": "self.hand", "source_text": original}], []

    # Evolution timing exceptions.
    if re.fullmatch(r"If you go second, this Pokémon can evolve during your first turn\.?,?", t, re.I):
        return [{"op": "register_evolution_exception", "condition": {"self_went_second": True, "self_first_turn": True}, "can_evolve": True, "source_text": original}], []

    # Ability cost + draw.
    if re.fullmatch(r"You must discard a card from your hand in order to use this Ability\. Once during your turn, you may draw 2 cards\.?,?", t, re.I):
        return [{"op": "discard_cards", "player": "self", "source_zone": "hand", "selection": amount_exact(1), "destination": "discard", "cost": True, "source_text": original}, {"op": "draw_cards", "player": "self", "amount": amount_exact(2), "optional": True, "usage_limit": {"scope": "per_turn_per_copy", "max": 1}, "source_text": original}], []

    # Sturdy/endure style replacement.
    if re.fullmatch(r"If this Pokémon has full HP and would be Knocked Out by damage from an attack, it is not Knocked Out, and its remaining HP becomes 10\.?,?", t, re.I):
        return [{"op": "register_replacement_effect", "replacement_id": "survive_attack_ko_from_full_hp_at_10_hp", "trigger": "self_would_be_knocked_out_by_attack_damage", "condition": {"self_at_full_hp": True}, "effect": {"prevent_knockout": True, "set_remaining_hp": 10}, "source_text": original}], []

    # Coin tails -> self cannot attack next turn.
    if re.fullmatch(r"Flip a coin\. If tails, during your next turn, this Pokémon can't attack\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "tails", "then": [delayed_modifier_step("self_cannot_attack_next_turn", "self_attacking_pokemon", {"attack_allowed": False}, original)], "source_text": original}], []

    # Simple hand look.
    if re.fullmatch(r"Look at your opponent's hand\.?,?", t, re.I):
        return [{"op": "reveal_hand_to_player", "player": "opponent", "viewer": "self", "source_text": original}], []

    # Energy-dependent attack damage.
    m = re.fullmatch(r"Does (\d+) damage (?:times the amount of|for each) Energy attached to the Defending Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_per_attached_energy", "target": "opponent.active", "damage_per_energy": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"Does (\d+) damage plus (\d+) more damage for each Energy card attached to the Defending Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "formula", "base_damage_in_text": int(m.group(1)), "additional_per_attached_energy": int(m.group(2)), "target": "opponent.active", "source_text": original}], []

    # Choose N opponent Pokémon and deal damage to each.
    m = re.fullmatch(r"Choose (\d+) of your opponent's (?:Benched )?Pokémon\. This attack does (\d+) damage to each of them\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_pokemon", "zone": "opponent.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(int(m.group(1))), "source_text": original}, {"op": "deal_damage", "target_ref": "chosen_opponent_pokemon", "amount": int(m.group(2)), "apply_weakness_resistance_to_bench": False, "source_text": original}], []

    # Search Energy / Trainer / Supporter / Item from deck/discard to hand variants.
    m = re.fullmatch(r"Search your deck for an? (Energy|Trainer|Supporter) card, show it to your opponent, and put it into your hand\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        filt = {"supertype": "Energy"} if m.group(1).lower() == "energy" else {"supertype": "Trainer", "subtypes": [m.group(1).capitalize()]}
        if m.group(1).lower() == "trainer":
            filt = {"supertype": "Trainer"}
        return [{"op": "search_deck", "player": "self", "target_id": "searched_card", "filter": filt, "amount": amount_exact(1), "reveal": True, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Put an? (Item|Trainer|Supporter) card from your discard pile into your hand\.?,?", t, re.I)
    if m:
        subtype = m.group(1).capitalize()
        filt = {"supertype": "Trainer"} if subtype == "Trainer" else {"supertype": "Trainer", "subtypes": [subtype]}
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_discard_card", "zone": "self.discard", "filter": filt, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_discard_card", "destination": "self.hand", "source_text": original}], []

    # Flip heads -> discard random card from opponent's hand.
    if re.fullmatch(r"Flip a coin\. If heads, discard a random card from your opponent's hand\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "discard_cards", "player": "opponent", "source_zone": "hand", "selection": {"mode": "random", "amount": 1}, "destination": "discard", "source_text": original}], "source_text": original}], []

    # Discard/move special Energy variants.
    if re.fullmatch(r"Discard a Special Energy attached to (?:the Defending Pokémon|your opponent's Active Pokémon)\.?,?", t, re.I):
        return [{"op": "discard_attached_energy", "target": "opponent.active", "filter": {"subtypes": ["Special"]}, "amount": amount_exact(1), "source_text": original}], []

    if re.fullmatch(r"Move an Energy card attached to the Defending Pokémon to another of your opponent's Pokémon\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_defending_energy", "zone": "attached_to:opponent.active", "filter": {"supertype": "Energy"}, "amount": amount_exact(1), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_opponent_other_pokemon", "zone": "opponent.in_play", "filter": {"supertype": "Pokémon", "not_ref": "opponent.active"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_energy", "cards_ref": "chosen_defending_energy", "destination_ref": "chosen_opponent_other_pokemon", "source_text": original}], []



    # v0.6: cautiously compile long-tail low-count patterns from v0.5 coverage.

    # GX attack suffix can appear appended to an otherwise parseable sentence.
    gx_match = re.fullmatch(r"(.+?)\s*\(You can't use more than 1 GX attack in a game\.\)\.?", t, re.I)
    if gx_match:
        rest = normalize_space(gx_match.group(1))
        rest_steps, rest_unparsed = compile_simple_text(rest, source_section)
        return rest_steps + [{"op": "reference_global_rule", "global_rule_id": "gx_attack_once_per_game", "source_text": original}], rest_unparsed

    # Retaliation: when this Pokémon is damaged by attack, put counters / condition on attacker.
    m = re.fullmatch(r"If (?:this Pokémon|[A-Za-z' .-]+) is (?:your Active Pokémon|in the Active Spot) and is damaged by an opponent's attack \(even if (?:this Pokémon|[A-Za-z' .-]+) is Knocked Out\), put (\d+) damage counters? on the Attacking Pokémon\.?,?", t, re.I)
    if m:
        return [{
            "op": "register_trigger",
            "trigger_id": "retaliate_damage_counters_when_damaged_by_attack",
            "event": "self_active_damaged_by_opponents_attack",
            "effect": {"op": "place_damage_counters", "target": "attacking_pokemon", "amount": int(m.group(1))},
            "duration": {"while_source_in_play": True},
            "source_text": original,
        }], []

    m = re.fullmatch(r"If this Pokémon is in the Active Spot and is damaged by an attack from your opponent's Pokémon \(even if this Pokémon is Knocked Out\), the Attacking Pokémon is now (Burned|Poisoned|Confused|Paralyzed|Asleep)\.?,?", t, re.I)
    if m:
        return [{
            "op": "register_trigger",
            "trigger_id": "retaliate_special_condition_when_damaged_by_attack",
            "event": "self_active_damaged_by_opponents_attack",
            "effect": status_condition_step(m.group(1), "attacking_pokemon", original),
            "duration": {"while_source_in_play": True},
            "source_text": original,
        }], []

    # Hand disruption: choose random/unknown card, reveal/look, shuffle/discard.
    if re.fullmatch(r"Choose 1 card from your opponent's hand without looking\. Look at (?:that card|the card you chose), then have your opponent shuffle (?:that card|that card you chose) into (?:his or her|their) deck\.?,?", t, re.I):
        return [{
            "op": "choose_cards",
            "player": "self",
            "target_id": "chosen_opponent_hand_card",
            "zone": "opponent.hand",
            "selection": {"mode": "random_unknown", "amount": 1},
            "source_text": original,
        }, {
            "op": "reveal_cards_to_player",
            "cards_ref": "chosen_opponent_hand_card",
            "viewer": "self",
            "source_text": original,
        }, {
            "op": "move_card",
            "cards_ref": "chosen_opponent_hand_card",
            "destination": "opponent.deck",
            "source_text": original,
        }, {"op": "shuffle_deck", "player": "opponent", "source_text": original}], []

    if re.fullmatch(r"Choose a random card from your opponent's hand\. Your opponent reveals that card and shuffles it into (?:his or her|their) deck\.?,?", t, re.I):
        return [{
            "op": "choose_cards",
            "player": "self",
            "target_id": "chosen_opponent_hand_card",
            "zone": "opponent.hand",
            "selection": {"mode": "random", "amount": 1},
            "source_text": original,
        }, {"op": "reveal_cards", "cards_ref": "chosen_opponent_hand_card", "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_hand_card", "destination": "opponent.deck", "source_text": original}, {"op": "shuffle_deck", "player": "opponent", "source_text": original}], []

    # Searches: any card / Pokémon / Basic Pokémon / Energy to hand, deck-to-attach, discard-to-attach.
    m = re.fullmatch(r"Search your deck for (?:up to )?(\d+|a|an) cards? and put (?:it|them) into your hand\. (?:Then, shuffle your deck|Shuffle your deck afterward)\.?,?", t, re.I)
    if m:
        raw_n = m.group(1).lower()
        amount = amount_exact(1 if raw_n in ["a", "an"] else int(raw_n))
        if t.lower().startswith("search your deck for up to"):
            amount = amount_up_to(1 if raw_n in ["a", "an"] else int(raw_n))
        return [{"op": "search_deck", "player": "self", "target_id": "searched_cards", "filter": {"any_card": True}, "amount": amount, "reveal": False, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    if re.fullmatch(r"Search your deck for a Pokémon, (?:show it to your opponent|reveal it), and put it into your hand\. (?:Then, shuffle your deck|Shuffle your deck afterward)\.?,?", t, re.I):
        return [{"op": "search_deck", "player": "self", "target_id": "searched_pokemon", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "reveal": True, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If heads, search your deck for a Pokémon, (?:show it to your opponent|reveal it), and put it into your hand\. (?:Then, shuffle your deck|Shuffle your deck afterward)\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "search_deck", "player": "self", "target_id": "searched_pokemon", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "reveal": True, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Search your deck for (?:a|an) ([A-Za-z]+) Energy card and attach it to (this Pokémon|1 of your Pokémon)\. (?:Then, shuffle your deck|Shuffle your deck afterward)\.?,?", t, re.I)
    if m:
        energy_type = m.group(1).capitalize()
        target = "self_attacking_pokemon" if m.group(2).lower() == "this pokémon" else "chosen_own_pokemon"
        steps = []
        if target == "chosen_own_pokemon":
            steps.append({"op": "choose_target", "player": "self", "target_id": target, "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original})
        steps.extend([{"op": "search_deck", "player": "self", "target_id": "searched_energy", "filter": {"supertype": "Energy", "energy_type": energy_type}, "amount": amount_exact(1), "reveal": True, "destination": "attach", "source_text": original}, {"op": "attach_card", "cards_ref": "searched_energy", "target": target if target == "self_attacking_pokemon" else None, "target_ref": target if target != "self_attacking_pokemon" else None, "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}])
        return steps, []

    m = re.fullmatch(r"Search your deck for a basic Energy card and attach it to 1 of your Pokémon\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "search_deck", "player": "self", "target_id": "searched_basic_energy", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(1), "reveal": False, "destination": "attach", "source_text": original}, {"op": "attach_card", "cards_ref": "searched_basic_energy", "target_ref": "chosen_own_pokemon", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your discard pile for (?:a|an) ([A-Za-z]+|basic)? ?Energy card and attach it to (this Pokémon|1 of your Pokémon|1 of your Benched Pokémon)\.?,?", t, re.I)
    if m:
        raw_type = (m.group(1) or "").strip().lower()
        filt = {"supertype": "Energy"}
        if raw_type == "basic":
            filt["subtypes"] = ["Basic"]
        elif raw_type:
            filt["energy_type"] = raw_type.capitalize()
        target_phrase = m.group(2).lower()
        target = "self_attacking_pokemon" if target_phrase == "this pokémon" else "chosen_own_benched_pokemon" if "benched" in target_phrase else "chosen_own_pokemon"
        zone = "self.bench" if "benched" in target_phrase else "self.in_play"
        steps = []
        if target != "self_attacking_pokemon":
            steps.append({"op": "choose_target", "player": "self", "target_id": target, "zone": zone, "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original})
        steps.extend([{"op": "choose_cards", "player": "self", "target_id": "chosen_energy_from_discard", "zone": "self.discard", "filter": filt, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_energy_from_discard", "target": target if target == "self_attacking_pokemon" else None, "target_ref": target if target != "self_attacking_pokemon" else None, "source_text": original}])
        return steps, []

    # Switching / gust variants.
    if re.fullmatch(r"You may have your opponent switch their Active Pokémon with 1 of their Benched Pokémon\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "opponent", "chooser": "opponent", "new_active_ref": "opponent_choice_from_bench", "optional": True, "source_text": original}], []

    if re.fullmatch(r"Switch 1 of your opponent's Benched Pokémon with (?:his or her|their) Active Pokémon\.?,?", t, re.I):
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "switch_active", "player": "opponent", "new_active_ref": "chosen_opponent_benched_pokemon", "source_text": original}], []

    if re.fullmatch(r"If your opponent has any Benched Pokémon, choose 1 of them and switch it with the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "condition": {"opponent.bench_not_empty": True}, "source_text": original}, {"op": "switch_active", "player": "opponent", "new_active_ref": "chosen_opponent_benched_pokemon", "source_text": original}], []

    # Damage to own bench / each opponent Pokémon / damage counters movement.
    m = re.fullmatch(r"Does (\d+) damage to each of your own Benched Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "deal_damage", "target": "self.bench.each", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, this attack does (\d+) damage to each of your opponent's (Benched )?Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        target = "opponent.bench.each" if m.group(2) else "opponent.in_play.each"
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "deal_damage", "target": target, "amount": int(m.group(1)), "apply_weakness_resistance_to_bench": False, "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Put (\d+) damage counters on your opponent's (?:Benched )?Pokémon in any way you like\.?,?", t, re.I)
    if m:
        zone = "opponent.bench" if "Benched" in t else "opponent.in_play"
        return [{"op": "distribute_damage_counters", "player": "self", "target_zone": zone, "amount": int(m.group(1)), "source_text": original}], []

    if re.fullmatch(r"Move all damage counters from 1 of your Benched Pokémon to your opponent's Active Pokémon\.?,?", t, re.I):
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_damaged_benched_pokemon", "zone": "self.bench", "filter": {"damage_counters_gt": 0}, "amount": amount_exact(1), "source_text": original}, {"op": "move_damage_counters", "from_ref": "chosen_damaged_benched_pokemon", "to": "opponent.active", "amount": {"mode": "all"}, "source_text": original}], []

    # Conditional attack-does-nothing / attack locks.
    if re.fullmatch(r"If your opponent's Active Pokémon has no damage counters on it before this attack does damage, this attack does nothing\.?,?", t, re.I):
        return [{"op": "attack_does_nothing_if", "condition": {"opponent.active.damage_counters_equals": 0, "timing": "before_damage"}, "source_text": original}], []

    if re.fullmatch(r"(?:The Defending Pokémon|During your opponent's next turn, the Defending Pokémon) can't (?:attack|use attacks)(?: during your opponent's next turn)?\.?,?", t, re.I):
        return [delayed_modifier_step("opponent_active_cannot_attack_next_turn", "opponent.active", {"attack_allowed": False}, original)], []

    if re.fullmatch(r"If the Defending Pokémon tries to attack during your opponent's next turn, your opponent flips a coin\. If tails, that attack doesn't happen\.?,?", t, re.I):
        return [delayed_modifier_step("opponent_active_attack_requires_heads_next_turn", "opponent.active", {"attack_requires_coin_flip": {"on_tails": "attack_does_nothing"}}, original)], []

    if re.fullmatch(r"Flip a coin\. If heads, during your opponent's next turn, the Defending Pokémon can't attack\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [delayed_modifier_step("opponent_active_cannot_attack_next_turn", "opponent.active", {"attack_allowed": False}, original)], "source_text": original}], []

    # Coin / discard Energy variants.
    if re.fullmatch(r"Flip 2 coins\. If both of them are heads, discard all Energy attached to the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 2, "target_id": "coin_results", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_results", "if": "all_heads", "then": [{"op": "discard_attached_energy", "target": "opponent.active", "amount": {"mode": "all"}, "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If tails, discard an Energy attached to this Pokémon\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "tails", "then": [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "amount": amount_exact(1), "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If heads, discard an Energy from 1 of your opponent's Pokémon\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_pokemon_with_energy", "zone": "opponent.in_play", "filter": {"has_attached_energy": True}, "amount": amount_exact(1), "source_text": original}, {"op": "discard_attached_energy", "target_ref": "chosen_opponent_pokemon_with_energy", "amount": amount_exact(1), "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"If the Defending Pokémon has any Energy cards attached to it, flip a coin\. If heads, choose 1 of those Energy cards and discard it\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "condition": {"opponent.active.has_attached_energy": True}, "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "discard_attached_energy", "target": "opponent.active", "amount": amount_exact(1), "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Discard all ([A-Za-z]+) Energy attached to this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "filter": {"energy_type": m.group(1).capitalize()}, "amount": {"mode": "all"}, "source_text": original}], []

    # Conditional / formula damage variants.
    m = re.fullmatch(r"This attack does (\d+) damage for each of your Benched Pokémon that has the (.+?) attack\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "count": {"zone": "self.bench", "filter": {"has_attack_name": m.group(2)}}, "damage_per": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each Pokémon Tool attached to all of your Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "count": {"zone": "attached_to:self.in_play", "filter": {"supertype": "Trainer", "subtypes": ["Tool"]}}, "damage_per": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each Colorless in your opponent's Active Pokémon's Retreat Cost\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_value", "value": {"target": "opponent.active", "field": "retreat_cost_colorless_count"}, "damage_per": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each Energy attached to all of your opponent's Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "count": {"zone": "attached_to:opponent.in_play", "filter": {"supertype": "Energy"}}, "damage_per": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each Energy attached to this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "count": {"zone": "attached_to:self_attacking_pokemon", "filter": {"supertype": "Energy"}}, "damage_per": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each of your Pokémon in play\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "count": {"zone": "self.in_play", "filter": {"supertype": "Pokémon"}}, "damage_per": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage times the number of cards in your hand\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "count": {"zone": "self.hand"}, "damage_per": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each Pokémon in your discard pile that has the (.+?) attack\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "count": {"zone": "self.discard", "filter": {"supertype": "Pokémon", "has_attack_name": m.group(2)}}, "damage_per": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"If your opponent's Active Pokémon is (?:a )?(Pokémon-EX|Poisoned), this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        cond = {"opponent.active.subtypes_contains": "EX"} if "EX" in m.group(1) else {"opponent.active.special_conditions_contains": "Poisoned"}
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(2)), "condition": cond, "source_text": original}], []

    m = re.fullmatch(r"If (?:this Pokémon has any damage counters on it|you have a Stadium in play|you have the same number of cards in your hand as your opponent|this Pokémon moved from your Bench to the Active Spot this turn|any of your Pokémon were Knocked Out by damage from an opponent's attack during their last turn|this Pokémon was healed during this turn), this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        condition_text = original.rsplit(", this attack does", 1)[0]
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"raw_text": condition_text}, "source_text": original}], []

    # Misc play/evolution/continuous modifiers.
    if re.fullmatch(r"Remove all Special Conditions from your Active Pokémon\.?,?", t, re.I):
        return [{"op": "remove_special_conditions", "target": "self.active", "conditions": "all", "source_text": original}], []

    m = re.fullmatch(r"When you play this Pokémon from your hand to evolve 1 of your Pokémon during your turn, you may draw (\d+) cards\.?,?", t, re.I)
    if m:
        return [{"op": "register_trigger", "trigger_id": "on_evolve_from_hand_draw", "event": "self_played_from_hand_to_evolve", "optional": True, "effect": {"op": "draw_cards", "player": "self", "amount": amount_exact(int(m.group(1)))}, "source_text": original}], []

    if re.fullmatch(r"Return this Pokémon and all cards attached to it to your hand\.?,?", t, re.I):
        return [{"op": "move_card", "cards": "self_pokemon_and_attached_cards", "destination": "self.hand", "source_text": original}], []

    if re.fullmatch(r"Put a card from your discard pile on top of your deck\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_discard_card", "zone": "self.discard", "filter": {"any_card": True}, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_discard_card", "destination": "self.deck.top", "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If heads, draw a card\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "draw_cards", "player": "self", "amount": amount_exact(1), "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Discard a card from your hand\. If you do, draw (\d+) cards\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_hand_card_to_discard", "zone": "self.hand", "filter": {"any_card": True}, "amount": amount_exact(1), "source_text": original}, {"op": "discard_card", "cards_ref": "chosen_hand_card_to_discard", "destination": "self.discard", "source_text": original}, {"op": "draw_cards", "player": "self", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    if re.fullmatch(r"Before doing damage, discard all (?:Trainer cards|Pokémon Tool cards) attached to (?:the Defending Pokémon|your opponent's Active Pokémon)\.?,?", t, re.I):
        return [{"op": "discard_attached_cards", "target": "opponent.active", "filter": {"supertype": "Trainer"}, "amount": {"mode": "all"}, "timing_note": "before_damage", "source_text": original}], []

    m = re.fullmatch(r"Both this Pokémon and the Defending Pokémon are now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        condition = m.group(1)
        return [status_condition_step(condition, "self_attacking_pokemon", original), status_condition_step(condition, "opponent.active", original)], []

    m = re.fullmatch(r"Each Defending Pokémon is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [status_condition_step(m.group(1), "opponent.active.each_defending", original)], []

    if re.fullmatch(r"Flip a coin\. If heads, the Defending Pokémon is now Paralyzed and discard an Energy card attached to the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [status_condition_step("Paralyzed", "opponent.active", original), {"op": "discard_attached_energy", "target": "opponent.active", "amount": amount_exact(1), "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\. If tails, this attack does nothing(?: \(not even damage\))?\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [status_condition_step(m.group(1), "opponent.active", original)], "else": [{"op": "attack_does_nothing", "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned) and can't retreat during your opponent's next turn\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [status_condition_step(m.group(1), "opponent.active", original), delayed_modifier_step("opponent_active_cannot_retreat_next_turn", "opponent.active", {"cannot_retreat": True}, original)], "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, this attack does (\d+) more damage and (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "source_text": original}, status_condition_step(m.group(2), "opponent.active", original)], "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, this attack does (\d+) more damage\. If tails, (?:your opponent's Active Pokémon|the Defending Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "source_text": original}], "else": [status_condition_step(m.group(2), "opponent.active", original)], "source_text": original}], []

    m = re.fullmatch(r"Draw (\d+) cards?\. This Pokémon is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [{"op": "draw_cards", "player": "self", "amount": amount_exact(int(m.group(1))), "source_text": original}, status_condition_step(m.group(2), "self", original)], []




    # v0.7: family-level long-tail patterns from long_tail_review_summary.
    # These are still conservative: each pattern maps to a structured generic op,
    # keeping raw conditions where exact game logic differs by era.

    # Broader hand disruption variants: unknown/random card to deck or discard.
    if re.fullmatch(r"Choose 1 card from your opponent's hand without looking\. Look at (?:that card|the card you chose), then have your opponent shuffle (?:that card|that card you chose) into (?:his or her|their) deck\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_opponent_hand_card", "zone": "opponent.hand", "selection": {"mode": "random_unknown", "amount": 1}, "source_text": original}, {"op": "reveal_cards_to_player", "cards_ref": "chosen_opponent_hand_card", "viewer": "self", "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_hand_card", "destination": "opponent.deck", "source_text": original}, {"op": "shuffle_deck", "player": "opponent", "source_text": original}], []

    if re.fullmatch(r"Choose a random card from your opponent's hand\. Your opponent reveals that card and shuffles it into (?:his or her|their) deck\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_opponent_hand_card", "zone": "opponent.hand", "selection": {"mode": "random", "amount": 1}, "source_text": original}, {"op": "reveal_cards", "cards_ref": "chosen_opponent_hand_card", "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_hand_card", "destination": "opponent.deck", "source_text": original}, {"op": "shuffle_deck", "player": "opponent", "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If heads, choose 1 card from your opponent's hand without looking and discard it\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_cards", "player": "self", "target_id": "chosen_opponent_hand_card", "zone": "opponent.hand", "selection": {"mode": "random_unknown", "amount": 1}, "source_text": original}, {"op": "discard_card", "cards_ref": "chosen_opponent_hand_card", "destination": "opponent.discard", "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"Your opponent discards (\d+) cards? from their hand\.?,?", t, re.I):
        n = int(re.fullmatch(r"Your opponent discards (\d+) cards? from their hand\.?,?", t, re.I).group(1))
        return [{"op": "discard_cards", "player": "opponent", "source_zone": "hand", "selection": {"mode": "opponent_choice", "amount": n}, "destination": "discard", "source_text": original}], []

    # Broader deck search patterns, including older "show it" wording and no reveal text.
    m = re.fullmatch(r"Search your deck for (?:a|an|1) (.+?), (?:show it to your opponent|reveal it), and put it into your hand\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_card", "filter": raw_filter(m.group(1)), "amount": amount_exact(1), "reveal": True, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for (?:a|an|1) (.+?) and put it into your hand\. (?:Then, shuffle your deck|Shuffle your deck afterward)\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_card", "filter": raw_filter(m.group(1)), "amount": amount_exact(1), "reveal": False, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for up to (\d+) cards and put them into your hand\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_cards", "filter": {"any_card": True}, "amount": amount_up_to(int(m.group(1))), "reveal": False, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for (\d+) Basic Pokémon and put them onto your Bench\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_basic_pokemon", "filter": {"supertype": "Pokémon", "subtypes": ["Basic"]}, "amount": amount_exact(int(m.group(1))), "reveal": False, "destination": "self.bench", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    # Search discard pile -> hand/attach variants.
    m = re.fullmatch(r"Search your discard pile for (?:an|a) (.+?) card, show it to your opponent, and put it into your hand\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_discard_card", "zone": "self.discard", "filter": raw_filter(m.group(1) + " card"), "amount": amount_exact(1), "source_text": original}, {"op": "reveal_cards", "cards_ref": "chosen_discard_card", "source_text": original}, {"op": "move_card", "cards_ref": "chosen_discard_card", "destination": "self.hand", "source_text": original}], []

    m = re.fullmatch(r"Search your discard pile for (?:a|an) ([A-Za-z]+ )?Energy card and attach it to 1 of your (Benched )?Pokémon\.?,?", t, re.I)
    if m:
        energy_type = (m.group(1) or "").strip()
        zone = "self.bench" if m.group(2) else "self.in_play"
        filt = {"supertype": "Energy"}
        if energy_type:
            filt["types"] = [energy_type.capitalize()]
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_energy_from_discard", "zone": "self.discard", "filter": filt, "amount": amount_exact(1), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon", "zone": zone, "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_energy_from_discard", "target_ref": "chosen_own_pokemon", "source_text": original}], []

    # Energy search deck -> attach, broadening typed/basic variants.
    m = re.fullmatch(r"Search your deck for (?:up to )?(\d+)? ?([A-Za-z]+ )?Energy cards? and attach (?:it|them) to this Pokémon\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        n = int(m.group(1) or 1)
        energy_type = (m.group(2) or "").strip()
        filt = {"supertype": "Energy"}
        if energy_type:
            filt["types"] = [energy_type.capitalize()]
        return [{"op": "search_deck", "player": "self", "target_id": "searched_energy", "filter": filt, "amount": amount_exact(n), "reveal": False, "destination": "attach_to:self_attacking_pokemon", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for (?:a|1) basic Energy card and attach it to 1 of your Pokémon\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_basic_energy", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(1), "reveal": False, "destination": "attach_to:chosen_own_pokemon", "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    # Conditional and formula damage variants.
    m = re.fullmatch(r"Does (\d+) damage plus (\d+) more damage for each damage counter on the Defending Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage_from_count", "mode": "add", "amount_per": int(m.group(2)), "count": {"target": "opponent.active", "field": "damage_counters"}, "base_damage_text": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"Does (\d+) damage plus (\d+) more damage for each Energy attached to the Defending Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage_from_count", "mode": "add", "amount_per": int(m.group(2)), "count": {"zone": "attached_to:opponent.active", "filter": {"supertype": "Energy"}}, "base_damage_text": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"Does (\d+) more damage for each ([A-Za-z]+) Energy attached to this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage_from_count", "mode": "add", "amount_per": int(m.group(1)), "count": {"zone": "attached_to:self_attacking_pokemon", "filter": {"supertype": "Energy", "types": [m.group(2).capitalize()]}}, "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage times the amount of Energy attached to your opponent's Active Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "count": {"zone": "attached_to:opponent.active", "filter": {"supertype": "Energy"}}, "damage_per": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each damage counter on your opponent's Active Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "count": {"target": "opponent.active", "field": "damage_counters"}, "damage_per": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"If your opponent's Active Pokémon (?:already )?has (?:any|no) damage counters on it(?: before this attack does damage)?, this attack (?:does (\d+) more damage|does nothing)\.?,?", t, re.I)
    if m:
        if "does nothing" in t.lower():
            return [{"op": "conditional_attack_does_nothing", "condition": {"opponent.active.damage_counters": 0}, "source_text": original}], []
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"opponent.active.has_damage_counters": True}, "source_text": original}], []

    # Prevention/reduction and special next-turn attack effects.
    if re.fullmatch(r"Prevent all effects of attacks, including damage, done to this Pokémon during your opponent's next turn\.?,?", t, re.I):
        return [delayed_modifier_step("prevent_attack_damage_and_effects_to_self_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True, "prevent_effects_of_attacks": True}, original)], []

    if re.fullmatch(r"Any damage done to this Pokémon by attacks is reduced by (\d+) \(after applying Weakness and Resistance\)\.?,?", t, re.I):
        n = int(re.fullmatch(r"Any damage done to this Pokémon by attacks is reduced by (\d+) \(after applying Weakness and Resistance\)\.?,?", t, re.I).group(1))
        return [{"op": "register_continuous_modifier", "modifier_id": "self_takes_less_damage_from_attacks", "target": "self", "duration": {"while_source_in_play": True}, "modification": {"damage_taken_from_attacks_delta": -n}, "source_text": original}], []

    m = re.fullmatch(r"Prevent all damage done to this Pokémon by attacks from your opponent's ([A-Za-z]+|Evolution) Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "register_continuous_modifier", "modifier_id": "prevent_damage_from_opponent_filtered_pokemon", "target": "self", "duration": {"while_source_in_play": True}, "modification": {"prevent_damage_from_attacks": True, "attacker_filter": raw_filter("opponent's " + m.group(1) + " Pokémon")}, "source_text": original}], []

    if re.fullmatch(r"During your opponent's next turn, prevent all damage done to this Pokémon by attacks from Evolution Pokémon\.?,?", t, re.I):
        return [delayed_modifier_step("prevent_damage_from_evolution_pokemon_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True, "attacker_filter": {"is_evolution": True}}, original)], []

    if re.fullmatch(r"If this Pokémon would be Knocked Out by damage from an attack, flip a coin\. If heads, this Pokémon is not Knocked Out, and its remaining HP becomes 10\.?,?", t, re.I):
        return [{"op": "register_replacement_effect", "event": "self_would_be_knocked_out_by_attack_damage", "condition": {"coin_flip": "heads"}, "replacement": {"knocked_out": False, "remaining_hp": 10}, "source_text": original}], []

    # Attack-lock variants.
    if re.fullmatch(r"(?:The Defending Pokémon|During your opponent's next turn, the Defending Pokémon) can't (?:use )?attacks? during your opponent's next turn\.?,?", t, re.I):
        return [delayed_modifier_step("opponent_active_cannot_attack_next_turn", "opponent.active", {"attack_allowed": False}, original)], []

    if re.fullmatch(r"Flip a coin\. If heads, during your opponent's next turn, the Defending Pokémon can't attack\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [delayed_modifier_step("opponent_active_cannot_attack_next_turn", "opponent.active", {"attack_allowed": False}, original)], "source_text": original}], []

    if re.fullmatch(r"If the Defending Pokémon tries to attack during your opponent's next turn, your opponent flips a coin\. If tails, that attack doesn't happen\.?,?", t, re.I):
        return [delayed_modifier_step("opponent_active_attack_requires_heads_next_turn", "opponent.active", {"attack_requires_coin_flip": {"on_tails": "attack_does_not_happen"}}, original)], []

    # Switch/gust variants and compound switch outcomes.
    if re.fullmatch(r"Switch 1 of your opponent's Benched Pokémon with (?:his or her|their) Active Pokémon\.?,?", t, re.I):
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "switch_active", "player": "opponent", "new_active_ref": "chosen_opponent_benched_pokemon", "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If heads, switch 1 of your opponent's Benched Pokémon with their Active Pokémon\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "switch_active", "player": "opponent", "new_active_ref": "chosen_opponent_benched_pokemon", "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"You may have your opponent switch their Active Pokémon with 1 of their Benched Pokémon\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "opponent", "chooser": "opponent", "optional": True, "new_active_ref": "opponent_choice_from_bench", "source_text": original}], []

    # Damage counters and damage movement.
    m = re.fullmatch(r"Put damage counters on your opponent's Active Pokémon until its remaining HP is (\d+)\.?,?", t, re.I)
    if m:
        return [{"op": "place_damage_counters_until_remaining_hp", "target": "opponent.active", "remaining_hp": int(m.group(1)), "source_text": original}], []

    if re.fullmatch(r"Move all damage counters from 1 of your Benched Pokémon to your opponent's Active Pokémon\.?,?", t, re.I):
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_own_benched_damaged_pokemon", "zone": "self.bench", "filter": {"has_damage_counters": True}, "amount": amount_exact(1), "source_text": original}, {"op": "move_damage_counters", "from_ref": "chosen_own_benched_damaged_pokemon", "to": "opponent.active", "amount": {"mode": "all"}, "source_text": original}], []

    # Special conditions with conditional clauses.
    m = re.fullmatch(r"If the Defending Pokémon is an (?:Evolved|Evolution) Pokémon, (?:the Defending Pokémon is now (Confused|Asleep|Paralyzed|Poisoned|Burned)|it can't attack during your opponent's next turn)\.?,?", t, re.I)
    if m:
        if m.group(1):
            return [{"op": "conditional_effect", "condition": {"opponent.active.is_evolution": True}, "then": [status_condition_step(m.group(1), "opponent.active", original)], "source_text": original}], []
        return [{"op": "conditional_effect", "condition": {"opponent.active.is_evolution": True}, "then": [delayed_modifier_step("opponent_active_cannot_attack_next_turn", "opponent.active", {"attack_allowed": False}, original)], "source_text": original}], []

    if re.fullmatch(r"The Defending Pokémon is now Poisoned\. If the Defending Pokémon tries to attack during your opponent's next turn, your opponent flips a coin\. If tails, that attack does nothing\.?,?", t, re.I):
        return [status_condition_step("Poisoned", "opponent.active", original), delayed_modifier_step("opponent_active_attack_requires_heads_next_turn", "opponent.active", {"attack_requires_coin_flip": {"on_tails": "attack_does_nothing"}}, original)], []

    # Misc global/special rules and costs.
    if re.fullmatch(r"You can play this card only if 1 of your Pokémon was Knocked Out during your opponent's last turn\.?,?", t, re.I):
        return [{"op": "play_condition", "condition": {"self_pokemon_knocked_out_during_opponent_last_turn": True}, "source_text": original}], []

    if re.fullmatch(r"The Retreat Cost of the Pokémon this card is attached to is ColorlessColorless less\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "attached_pokemon_retreat_cost_less", "target": "attached_pokemon", "duration": {"while_attached": True}, "modification": {"retreat_cost_delta_colorless": -2}, "source_text": original}], []

    if re.fullmatch(r"As long as this Pokémon is on your Bench, your Active Pokémon's Retreat Cost is ColorlessColorless less\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "active_retreat_cost_less_while_self_benched", "target": "self.active", "condition": {"self.zone": "bench"}, "duration": {"while_source_in_play": True}, "modification": {"retreat_cost_delta_colorless": -2}, "source_text": original}], []

    # Devolution / named level-up / once-per-game notes.
    if re.fullmatch(r"If your opponent's Active Pokémon is an evolved Pokémon, devolve it by putting the highest Stage Evolution card on it into your opponent's hand\.?,?", t, re.I):
        return [{"op": "devolve_pokemon", "target": "opponent.active", "destination": "opponent.hand", "source_text": original}], []

    if re.fullmatch(r"Put this card onto your Active Arceus\. Arceus LV\.X can use any attack, Poké-Power, or Poké-Body from its previous level\.?,?", t, re.I):
        return [{"op": "level_up_pokemon", "target": "self.active", "required_name": "Arceus", "carry_previous_attacks_powers_bodies": True, "source_text": original}], []

    # Low-count GX-suffix phrases that were split after the main effect.
    if re.fullmatch(r"\(You can't use more than 1 GX attack in a game\.\)\.?,?", t, re.I):
        return [{"op": "reference_global_rule", "global_rule_id": "gx_attack_once_per_game", "source_text": original}], []



    # v0.8: broad long-tail family patterns after v0.7 coverage review.
    # These favor generic simulator ops with raw conditions preserved for exact ruling fidelity.

    # Hand disruption variant missed by v0.7: "that card you chose" wording.
    if re.fullmatch(r"Choose 1 card from your opponent's hand without looking\. Look at that card you chose, then have your opponent shuffle that card into (?:his or her|their) deck\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_opponent_hand_card", "zone": "opponent.hand", "selection": {"mode": "random_unknown", "amount": 1}, "source_text": original}, {"op": "look_at_cards", "player": "self", "cards_ref": "chosen_opponent_hand_card", "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_hand_card", "destination": "opponent.deck", "source_text": original}, {"op": "shuffle_deck", "player": "opponent", "source_text": original}], []

    # Coin flip / multi-coin compound outcomes.
    if re.fullmatch(r"Flip 2 coins\. If both are tails, this attack does nothing\. For each heads, discard an Energy attached to the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 2, "target_id": "coin_results", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_results", "if": {"all": "tails"}, "then": [{"op": "attack_does_nothing", "source_text": original}], "source_text": original}, {"op": "discard_attached_energy_per_coin_heads", "target": "opponent.active", "coin_results_ref": "coin_results", "source_text": original}], []

    m = re.fullmatch(r"Flip (\d+) coins\. For each heads, discard an Energy from your opponent's Active Pokémon\. If both of them are tails, this attack does nothing\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": int(m.group(1)), "target_id": "coin_results", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_results", "if": {"all": "tails"}, "then": [{"op": "attack_does_nothing", "source_text": original}], "source_text": original}, {"op": "discard_attached_energy_per_coin_heads", "target": "opponent.active", "coin_results_ref": "coin_results", "source_text": original}], []

    m = re.fullmatch(r"Flip (\d+) coins\. If both of them are heads, discard all Energy attached to the Defending Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": int(m.group(1)), "target_id": "coin_results", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_results", "if": {"all": "heads"}, "then": [{"op": "discard_attached_energy", "target": "opponent.active", "amount": {"mode": "all"}, "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Flip 3 coins\. If 1 of them is heads, this attack does (\d+) damage plus (\d+) more damage\. If 2 of them are heads, this attack does \d+ damage plus (\d+) more damage\. If all of them are heads, this attack does \d+ damage plus (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 3, "target_id": "coin_results", "source_text": original}, {"op": "modify_attack_damage_by_heads_table", "coin_results_ref": "coin_results", "mode": "add", "heads_to_bonus": {"1": int(m.group(2)), "2": int(m.group(3)), "3": int(m.group(4))}, "source_text": original}], []

    m = re.fullmatch(r"Flip a coin for each Energy attached to this Pokémon\. This attack does (\d+) damage for each heads\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": {"mode": "count_attached_energy", "target": "self_attacking_pokemon"}, "target_id": "coin_results", "source_text": original}, {"op": "set_attack_damage_from_coin_heads", "damage_per_heads": int(m.group(1)), "coin_results_ref": "coin_results", "source_text": original}], []

    if re.fullmatch(r"Flip 2 coins\. If both of them are heads, your opponent's Active Pokémon is Knocked Out\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 2, "target_id": "coin_results", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_results", "if": {"all": "heads"}, "then": [{"op": "knock_out_pokemon", "target": "opponent.active", "source_text": original}], "source_text": original}], []

    # Damage scaling and conditional damage variants.

    # v0.12: broader global damage scaling / conditional damage patterns.
    # These compile common old/new wording where the printed damage box carries the base damage
    # and the effect text adds a formula or conditional bonus.
    m = re.fullmatch(r"(?:Does|This attack does) (\d+) more damage for each ([A-Za-z]+ )?Energy attached to (?:the Defending Pokémon|your opponent's Active Pokémon)\.?,?", t, re.I)
    if m:
        filt = {"supertype": "Energy"}
        if m.group(2):
            filt["types"] = [m.group(2).strip().capitalize()]
        return [{"op": "modify_attack_damage_per_attached_card", "mode": "add", "amount_per_card": int(m.group(1)), "target": "opponent.active", "filter": filt, "source_text": original}], []

    m = re.fullmatch(r"(?:Does|This attack does) (\d+) more damage for each damage counter on (?:the Defending Pokémon|your opponent's Active Pokémon)\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage_per_damage_counter", "mode": "add", "amount_per_counter": int(m.group(1)), "counter_target": "opponent.active", "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage times the (?:amount|number) of Energy attached to all of your Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_per_attached_energy", "damage_per_energy": int(m.group(1)), "target": "self.in_play", "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each card in your opponent's hand\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_per_card_in_hand", "damage_per_card": int(m.group(1)), "player": "opponent", "source_text": original}], []

    m = re.fullmatch(r"If this Pokémon has any ([A-Za-z]+|Special) Energy attached(?: to it)?, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        energy = m.group(1).capitalize()
        condition = {"self_attacking_pokemon.has_attached_energy": True, "filter": {"supertype": "Energy"}}
        if energy == "Special":
            condition["filter"]["category"] = "Special"
        else:
            condition["filter"]["types"] = [energy]
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(2)), "condition": condition, "source_text": original}], []

    m = re.fullmatch(r"If any of your Pokémon were Knocked Out by damage from an opponent's attack during (?:his or her|their|your opponent's) last turn, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"self.pokemon_knocked_out_by_opponent_attack_last_turn": True}, "source_text": original}], []

    m = re.fullmatch(r"If your Benched Pokémon have any damage counters on them, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"self.bench.any_damage_counters": True}, "source_text": original}], []

    m = re.fullmatch(r"If this Pokémon was on the Bench and became your Active Pokémon this turn, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"self_attacking_pokemon.moved_from_bench_to_active_this_turn": True}, "source_text": original}], []

    m = re.fullmatch(r"If a Stadium is in play, this attack does (\d+) more damage\. Then, discard that Stadium\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"stadium_in_play": True}, "source_text": original}, {"op": "discard_stadium", "target": "stadium_in_play", "amount": amount_exact(1), "source_text": original}], []

    m = re.fullmatch(r"During your next turn, each of this Pokémon's attacks does (\d+) more damage \(before applying Weakness and Resistance\)\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("self_attacks_do_more_damage_next_turn", "self_attacking_pokemon", {"attack_damage_add": int(m.group(1)), "timing": "before_weakness_resistance"}, original)], []

    m = re.fullmatch(r"(?:Does|This attack does) (\d+) damage plus (\d+) more damage for each damage counter on (?:the Defending Pokémon|your opponent's Active Pokémon)\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage_per_damage_counter", "mode": "set_base_plus", "base_damage": int(m.group(1)), "amount_per_counter": int(m.group(2)), "counter_target": "opponent.active", "source_text": original}], []

    m = re.fullmatch(r"(?:Does|This attack does) (\d+) more damage for each ([A-Za-z]+ )?Energy attached to this Pokémon\.?,?", t, re.I)
    if m:
        filt = {"supertype": "Energy"}
        if m.group(2):
            filt["types"] = [m.group(2).strip().capitalize()]
        return [{"op": "modify_attack_damage_per_attached_card", "mode": "add", "amount_per_card": int(m.group(1)), "target": "self_attacking_pokemon", "filter": filt, "source_text": original}], []

    m = re.fullmatch(r"(?:Does|This attack does) (\d+) damage times the amount of Energy attached to (?:the Defending Pokémon|your opponent's Active Pokémon)\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_per_attached_energy", "damage_per_energy": int(m.group(1)), "target": "opponent.active", "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each Energy attached to all of your opponent's Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_per_attached_energy", "damage_per_energy": int(m.group(1)), "target": "opponent.in_play", "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each Colorless in your opponent's Active Pokémon's Retreat Cost\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_per_retreat_cost", "damage_per_colorless": int(m.group(1)), "target": "opponent.active", "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each (?:of your Pokémon|Pokémon) in play\.?,?", t, re.I)
    if m:
        scope = "self.in_play" if "your Pokémon" in t else "all.in_play"
        return [{"op": "set_attack_damage_per_pokemon_in_play", "damage_per_pokemon": int(m.group(1)), "scope": scope, "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage times the number of cards in your hand\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_per_card_in_hand", "damage_per_card": int(m.group(1)), "player": "self", "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each of your opponent's Benched Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_per_pokemon", "damage_per_pokemon": int(m.group(1)), "zone": "opponent.bench", "source_text": original}], []

    m = re.fullmatch(r"If you have (?:more|the same number of) cards in your hand (?:than|as) your opponent, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        condition = "self_hand_size_greater_than_opponent" if "more cards" in t.lower() else "self_hand_size_equals_opponent"
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": condition, "source_text": original}], []

    m = re.fullmatch(r"If your opponent's Active Pokémon is (?:a |an )?(Pokémon ex or Pokémon V|Pokémon-EX|Pokémon-ex|Pokémon V|Evolution Pokémon|Evolved Pokémon|Poisoned), this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(2)), "condition": {"opponent.active": raw_filter(m.group(1))}, "source_text": original}], []

    m = re.fullmatch(r"If (?:there is any Stadium card|you have a Stadium) in play, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"stadium_in_play": True}, "source_text": original}], []

    m = re.fullmatch(r"If this Pokémon (?:has any damage counters on it|was healed during this turn|moved from your Bench to the Active Spot this turn), this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"raw_text": original.split(',')[0]}, "source_text": original}], []

    if re.fullmatch(r"If your opponent's Active Pokémon has no damage counters on it before this attack does damage, this attack does nothing\.?,?", t, re.I):
        return [{"op": "attack_does_nothing_if", "condition": {"opponent.active.damage_counters": 0}, "source_text": original}], []

    # Spread / gust compound variants.
    m = re.fullmatch(r"Switch 1 of your opponent's Benched Pokémon with their Active Pokémon\. The new Active Pokémon is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "switch_active", "player": "opponent", "new_active_ref": "chosen_opponent_benched_pokemon", "source_text": original}, status_condition_step(m.group(1), "opponent.active", original)], []

    m = re.fullmatch(r"(?:Switch 1 of your opponent's Benched Pokémon with their Active Pokémon|Switch in 1 of your opponent's Benched Pokémon to the Active Spot)\. This attack does (\d+) damage to the new Active Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "switch_active", "player": "opponent", "new_active_ref": "chosen_opponent_benched_pokemon", "source_text": original}, {"op": "deal_damage", "target": "opponent.active", "amount": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage to 1 of your opponent's Benched Pokémon for each damage counter on that Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon", "has_damage_counters": True}, "amount": amount_exact(1), "source_text": original}, {"op": "deal_damage_per_damage_counter", "target_ref": "chosen_opponent_benched_pokemon", "amount_per_counter": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"Put (\d+) damage counters on your opponent's Benched Pokémon in any way you like\.?,?", t, re.I)
    if m:
        return [{"op": "place_damage_counters", "target": "opponent.bench.distribution", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"Put (\d+) damage counter on each of your opponent's Pokémon that already has damage counters on it\.?,?", t, re.I)
    if m:
        return [{"op": "place_damage_counters", "target": "opponent.in_play.each_with_damage", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    # Prevention / ignore / special condition replacement variants.
    if re.fullmatch(r"This attack's damage isn't affected by Weakness, Resistance, or any other effects on the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "ignore_weakness_resistance", "scope": "this_attack", "source_text": original}, {"op": "ignore_effects_on_defending_pokemon", "scope": "this_attack_damage", "source_text": original}], []

    if re.fullmatch(r"Prevent all effects of attacks, including damage, done to this Pokémon by your opponent's Pokémon-GX or Pokémon-EX\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "prevent_attacks_from_gx_ex", "target": "self", "duration": {"while_source_in_play": True}, "modification": {"prevent_damage_from_attacks": True, "prevent_effects_of_attacks": True, "attacker_filter": {"subtypes_any": ["GX", "EX"]}}, "source_text": original}], []

    if re.fullmatch(r"Whenever you attach an Energy card from your hand to this Pokémon, remove all Special Conditions from it\.?,?", t, re.I):
        return [{"op": "register_trigger", "event": "energy_attached_from_hand_to_self", "then": [{"op": "remove_special_conditions", "target": "self", "conditions": "all", "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"Heal all damage from this Pokémon\.?,?", t, re.I):
        return [{"op": "heal_damage", "target": "self_attacking_pokemon", "amount": {"mode": "all"}, "source_text": original}], []

    if re.fullmatch(r"During your next turn, this Pokémon's .+ attack does (\d+) more damage \(before applying Weakness and Resistance\)\.?,?", t, re.I):
        n = int(re.search(r"does (\d+) more damage", t, re.I).group(1))
        return [{"op": "register_continuous_modifier", "modifier_id": "named_attack_more_damage_next_turn", "target": "self", "duration": {"until": "end_of_self_next_turn"}, "modification": {"attack_damage_delta": n, "attack_filter": {"raw_text": original}}, "source_text": original}], []

    if re.fullmatch(r"During your opponent's next turn, the Defending Pokémon's attacks cost Colorless more, and its Retreat Cost is Colorless more\.?,?", t, re.I):
        return [delayed_modifier_step("opponent_active_attack_and_retreat_cost_tax", "opponent.active", {"attack_cost_delta_colorless": 1, "retreat_cost_delta_colorless": 1}, original)], []

    # Energy bounce / discard compound variants.
    m = re.fullmatch(r"You may put (\d+|an|a) Energy attached to your opponent's Active Pokémon into their hand\.?,?", t, re.I)
    if m:
        raw_n = m.group(1).lower()
        n = 1 if raw_n in {"a", "an"} else int(raw_n)
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_opponent_active_energy", "zone": "attached_to:opponent.active", "filter": {"supertype": "Energy"}, "amount": amount_exact(n), "optional": True, "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_active_energy", "destination": "opponent.hand", "source_text": original}], []

    m = re.fullmatch(r"Put (\d+) Energy attached to this Pokémon into your hand\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_self_attached_energy", "zone": "attached_to:self_attacking_pokemon", "filter": {"supertype": "Energy"}, "amount": amount_exact(int(m.group(1))), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_self_attached_energy", "destination": "self.hand", "source_text": original}], []

    if re.fullmatch(r"Discard an Energy from this Pokémon\. If you do, discard an Energy from your opponent's Active Pokémon\.?,?", t, re.I):
        return [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "amount": amount_exact(1), "source_text": original}, {"op": "discard_attached_energy", "target": "opponent.active", "amount": amount_exact(1), "requires_previous_step_success": True, "source_text": original}], []

    if re.fullmatch(r"Discard an Energy from this Pokémon and heal all damage from it\.?,?", t, re.I):
        return [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "amount": amount_exact(1), "source_text": original}, {"op": "heal_damage", "target": "self_attacking_pokemon", "amount": {"mode": "all"}, "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If tails, discard an Energy attached to this Pokémon\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "tails", "then": [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "amount": amount_exact(1), "source_text": original}], "source_text": original}], []

    # Search/draw older variants.
    if re.fullmatch(r"Shuffle your hand into your deck, then draw (\d+) cards\.?,?", t, re.I):
        n = int(re.search(r"draw (\d+) cards", t, re.I).group(1))
        return [{"op": "move_zone_to_zone", "player": "self", "source_zone": "hand", "destination_zone": "deck", "selection": {"mode": "all"}, "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}, {"op": "draw_cards", "player": "self", "amount": amount_exact(n), "source_text": original}], []

    if re.fullmatch(r"Draw cards until you have the same number of cards in your hand as your opponent\.?,?", t, re.I):
        return [{"op": "draw_until_hand_size_matches", "player": "self", "other_player": "opponent", "source_text": original}], []

    if re.fullmatch(r"Your opponent shuffles (?:his or her|their) hand into (?:his or her|their) deck and draws (\d+) cards\.?,?", t, re.I):
        n = int(re.search(r"draws (\d+) cards", t, re.I).group(1))
        return [{"op": "move_zone_to_zone", "player": "opponent", "source_zone": "hand", "destination_zone": "deck", "selection": {"mode": "all"}, "source_text": original}, {"op": "shuffle_deck", "player": "opponent", "source_text": original}, {"op": "draw_cards", "player": "opponent", "amount": amount_exact(n), "source_text": original}], []

    if re.fullmatch(r"Look at the top (\d+) cards of your deck and put them back on top of your deck in any order\.?,?", t, re.I):
        n = int(re.search(r"top (\d+) cards", t, re.I).group(1))
        return [{"op": "look_at_top_cards", "player": "self", "target_id": "looked_cards", "zone": "self.deck", "amount": amount_exact(n), "source_text": original}, {"op": "reorder_cards", "player": "self", "cards_ref": "looked_cards", "destination": "self.deck.top", "source_text": original}], []

    if re.fullmatch(r"Search your deck for (\d+) cards, shuffle your deck, then put those cards on top of it in any order\.?,?", t, re.I):
        n = int(re.search(r"for (\d+) cards", t, re.I).group(1))
        return [{"op": "search_deck", "player": "self", "target_id": "searched_cards", "filter": {"any_card": True}, "amount": amount_exact(n), "reveal": False, "destination": "temporary.selection", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}, {"op": "move_card", "cards_ref": "searched_cards", "destination": "self.deck.top", "ordering": "self_choice", "source_text": original}], []

    if re.fullmatch(r"Put a Basic Pokémon from your discard pile onto your Bench\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_basic_pokemon", "zone": "self.discard", "filter": {"supertype": "Pokémon", "subtypes": ["Basic"]}, "amount": amount_exact(1), "source_text": original}, {"op": "put_card_on_bench", "player": "self", "cards_ref": "chosen_basic_pokemon", "source_text": original}], []

    # Tool / attached attack / legality effects.
    if re.fullmatch(r"The (?:Rapid Strike|Single Strike) Pokémon this card is attached to can use the attack on this card\. \(You still need the necessary Energy to use this attack\.\)\.?,?", t, re.I):
        return [{"op": "grant_attack_from_attached_card", "target": "attached_pokemon", "duration": {"while_attached": True}, "source_text": original}], []

    if re.fullmatch(r"Pokémon Tools attached to each Pokémon \(both yours and your opponent's\) have no effect\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "pokemon_tools_have_no_effect", "target": "all.pokemon_tools", "duration": {"while_source_in_play": True}, "modification": {"effects_enabled": False}, "source_text": original}], []

    # Combined self/opponent switch.
    if re.fullmatch(r"Switch this Pokémon with 1 of your Benched Pokémon\. If you do, your opponent switches their Active Pokémon with 1 of their Benched Pokémon\.?,?", t, re.I):
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_own_benched_pokemon", "zone": "self.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "switch_active", "player": "self", "new_active_ref": "chosen_own_benched_pokemon", "source_text": original}, {"op": "switch_active", "player": "opponent", "chooser": "opponent", "new_active_ref": "opponent_choice_from_bench", "source_text": original}], []

    if re.fullmatch(r"If Festival Grounds is in play, this Pokémon may use an attack it has twice\. If the first attack Knocks Out your opponent's Active Pokémon, you may attack again after your opponent chooses a new Active Pokémon\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "festival_grounds_double_attack", "target": "self", "condition": {"stadium_in_play_name": "Festival Grounds"}, "duration": {"while_source_in_play": True}, "modification": {"may_attack_twice": True, "second_attack_if_first_ko": True}, "source_text": original}], []


    # v0.14: broad safe coverage bundle for common remaining long-tail families.
    # These patterns intentionally target high-frequency text shapes that map to
    # existing primitive ops without trying to resolve every rule nuance.

    # --- Draw / opponent draw / shared draw ---
    m = re.fullmatch(r"Your opponent may draw (?:a|1) card\.?,?", t, re.I)
    if m:
        return [{"op": "draw_cards", "player": "opponent", "amount": amount_exact(1), "optional": True, "source_text": original}], []

    m = re.fullmatch(r"Your opponent draws (?:a|1) card\.?,?", t, re.I)
    if m:
        return [{"op": "draw_cards", "player": "opponent", "amount": amount_exact(1), "source_text": original}], []

    m = re.fullmatch(r"Each player draws (\d+) cards?\.?,?", t, re.I)
    if m:
        n = int(m.group(1))
        return [{"op": "draw_cards", "player": "self", "amount": amount_exact(n), "source_text": original}, {"op": "draw_cards", "player": "opponent", "amount": amount_exact(n), "source_text": original}], []

    m = re.fullmatch(r"If your opponent has any Evolved Pokémon in play, draw (\d+) cards\.?,?", t, re.I)
    if m:
        return [{"op": "draw_cards", "player": "self", "amount": amount_exact(int(m.group(1))), "condition": {"opponent.in_play.has_evolved_pokemon": True}, "source_text": original}], []

    m = re.fullmatch(r"Once during your turn, if this Pokémon is in the Active Spot, you may draw (?:a|1) card\.?,?", t, re.I)
    if m:
        return [{"op": "play_condition", "condition": {"self.active_is_this_pokemon": True}, "source_text": original}, {"op": "draw_cards", "player": "self", "amount": amount_exact(1), "optional": True, "source_text": original}], []

    # --- Search / evolution search / fossil bench search ---
    m = re.fullmatch(r"Search your deck for up to (\d+) different types of basic Energy cards, show them to your opponent, and put them into your hand\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_basic_energy_cards", "filter": {"supertype": "Energy", "subtypes": ["Basic"], "different_types": True}, "amount": amount_up_to(int(m.group(1))), "reveal": True, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for a card that evolves from 1 of your Pokémon and put it onto that Pokémon(?: to evolve it)?\. \(This counts as evolving that Pokémon\.\) Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon_to_evolve", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "search_deck", "player": "self", "target_id": "searched_evolution_card", "filter": {"evolves_from_ref": "chosen_own_pokemon_to_evolve"}, "amount": amount_exact(1), "reveal": True, "destination": "temporary.selection", "source_text": original}, {"op": "evolve_pokemon", "pokemon_ref": "chosen_own_pokemon_to_evolve", "evolution_card_ref": "searched_evolution_card", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for a card that evolves from 1 of your Pokémon and put it onto that Pokémon to evolve it\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon_to_evolve", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "search_deck", "player": "self", "target_id": "searched_evolution_card", "filter": {"evolves_from_ref": "chosen_own_pokemon_to_evolve"}, "amount": amount_exact(1), "reveal": True, "destination": "temporary.selection", "source_text": original}, {"op": "evolve_pokemon", "pokemon_ref": "chosen_own_pokemon_to_evolve", "evolution_card_ref": "searched_evolution_card", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    if re.fullmatch(r"Search your deck for Omanyte, Kabuto, Aerodactyl, Lileep, or Anorith and put up to 2 of them onto your Bench\. Shuffle your deck afterward\. Treat the new Benched Pokémon as Basic Pokémon\.?,?", t, re.I):
        return [{"op": "search_deck", "player": "self", "target_id": "searched_fossil_pokemon", "filter": {"names_any": ["Omanyte", "Kabuto", "Aerodactyl", "Lileep", "Anorith"], "treat_as_basic": True}, "amount": amount_up_to(2), "reveal": True, "destination": "self.bench", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    if re.fullmatch(r"If this Pokémon is Knocked Out by damage from an attack from your opponent's Pokémon, search your deck for a card and put it into your hand\. Then, shuffle your deck\.?,?", t, re.I):
        return [{"op": "register_trigger", "event": "self_knocked_out_by_opponent_attack_damage", "then": [{"op": "search_deck", "player": "self", "target_id": "searched_card", "filter": {"any_card": True}, "amount": amount_exact(1), "reveal": False, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"When you play this Pokémon from your hand to evolve 1 of your Pokémon during your turn, you may search your deck for up to (\d+) Basic ([A-Za-z]+) Energy cards and attach them to your Pokémon in any way you like\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "register_trigger", "event": "played_from_hand_to_evolve", "then": [{"op": "search_deck", "player": "self", "target_id": "searched_basic_energy_cards", "filter": {"supertype": "Energy", "subtypes": ["Basic"], "types": [m.group(2).capitalize()]}, "amount": amount_up_to(int(m.group(1))), "reveal": True, "destination": "temporary.selection", "optional": True, "source_text": original}, {"op": "attach_card", "cards_ref": "searched_basic_energy_cards", "target": "self.pokemon.distribution", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], "source_text": original}], []

    # --- Energy attachment / acceleration ---
    m = re.fullmatch(r"You may attach a basic Energy card from your hand to 1 of your Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_basic_energy", "zone": "self.hand", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(1), "optional": True, "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_basic_energy", "target_ref": "chosen_own_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Attach an Energy card from your hand to 1 of your Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_energy", "zone": "self.hand", "filter": {"supertype": "Energy"}, "amount": amount_exact(1), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_energy", "target_ref": "chosen_own_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Attach (?:up to )?(\d+) basic Energy cards from your discard pile to 1 of your Benched Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_basic_energy", "zone": "self.discard", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_up_to(int(m.group(1))), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_own_benched_pokemon", "zone": "self.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_basic_energy", "target_ref": "chosen_own_benched_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Attach a ([A-Za-z]+) Energy card from your discard pile to 1 of your Benched Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_energy", "zone": "self.discard", "filter": {"supertype": "Energy", "types": [m.group(1).capitalize()]}, "amount": amount_exact(1), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_own_benched_pokemon", "zone": "self.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_energy", "target_ref": "chosen_own_benched_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, attach an Energy card from your discard pile to this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_cards", "player": "self", "target_id": "chosen_energy", "zone": "self.discard", "filter": {"supertype": "Energy"}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_energy", "target": "self_attacking_pokemon", "source_text": original}], "source_text": original}], []

    # --- Healing / damage counter removal ---
    m = re.fullmatch(r"Heal (\d+) damage from each Pokémon \(both yours and your opponent's\)\.?,?", t, re.I)
    if m:
        return [{"op": "heal_damage", "target": "all.in_play.each", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"Heal (\d+) damage from each of your ([A-Za-z]+) Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "heal_damage", "target": "self.in_play.each", "filter": {"supertype": "Pokémon", "types": [m.group(2).capitalize()]}, "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"Once during your turn, you may heal (\d+) damage from each of your Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "heal_damage", "target": "self.in_play.each", "amount": amount_exact(int(m.group(1))), "optional": True, "source_text": original}], []

    m = re.fullmatch(r"Remove (\d+) damage counters from each of your Benched Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "remove_damage_counters", "target": "self.bench.each", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"At any time between turns, remove (\d+) damage counter from ([A-Za-z0-9éÉ' .\-]+)\.?,?", t, re.I)
    if m:
        return [{"op": "register_between_turns_effect", "then": [{"op": "remove_damage_counters", "target": "self", "amount": amount_exact(int(m.group(1))), "target_name": m.group(2), "source_text": original}], "source_text": original}], []

    # --- Prevention / reduction / effect immunity ---
    if re.fullmatch(r"Prevent all damage done to your Benched Pokémon by your opponent's attacks\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "prevent_damage_to_self_bench_from_opponent_attacks", "target": "self.bench.each", "duration": {"while_source_in_play": True}, "modification": {"prevent_damage_from_opponent_attacks": True}, "source_text": original}], []

    if re.fullmatch(r"Prevent all effects of your opponent's Pokémon's Abilities done to this Pokémon\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "prevent_opponent_ability_effects_to_self", "target": "self", "duration": {"while_source_in_play": True}, "modification": {"prevent_effects_from_opponent_abilities": True}, "source_text": original}], []

    m = re.fullmatch(r"During your opponent's next turn, prevent all damage done to this Pokémon by attacks from Pokémon ([A-Za-z0-9]+)\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("prevent_attack_damage_from_pokemon_subtype_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True, "attacker_filter": {"subtypes": [m.group(1).upper()]}}, original)], []

    if re.fullmatch(r"Prevent all effects of attacks from your opponent's Pokémon done to this Pokémon\. \(Damage is not an effect\.\)\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "prevent_effects_of_attacks_to_self_except_damage", "target": "self", "duration": {"while_source_in_play": True}, "modification": {"prevent_effects_of_attacks": True, "damage_is_not_prevented": True}, "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, prevent all effects of an attack, including damage, done to ([A-Za-z0-9éÉ' .\-]+) during your opponent's next turn\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [delayed_modifier_step("prevent_attack_damage_and_effects_to_named_self_next_turn", "self", {"prevent_damage_from_attacks": True, "prevent_effects_of_attacks": True, "target_name": m.group(1)}, original)], "source_text": original}], []

    if re.fullmatch(r"If any damage is done to this Pokémon by attacks during your opponent's next turn, flip a coin\. If heads, prevent that damage\.?,?", t, re.I):
        return [delayed_modifier_step("flip_to_prevent_attack_damage_next_turn", "self_attacking_pokemon", {"on_attack_damage_flip_coin": {"heads": "prevent_damage"}}, original)], []

    # --- Trainer/Tool/Stadium restrictions and attached Tool setup ---
    m = re.fullmatch(r"As long as this Pokémon is in the Active Spot, your opponent can't play any (Stadium|Item|Trainer|Supporter) cards? from their hand\.?,?", t, re.I)
    if m:
        return [{"op": "register_continuous_modifier", "modifier_id": f"opponent_cannot_play_{m.group(1).lower()}_cards_from_hand_while_self_active", "target": "opponent", "condition": {"self.active_is_this_pokemon": True}, "duration": {"while_source_in_play": True}, "modification": {"cannot_play_card_type_from_hand": m.group(1).capitalize()}, "source_text": original}], []

    m = re.fullmatch(r"Your opponent can't play any (Trainer|Item|Supporter) cards(?: \(except for Supporter cards\))? from (?:his or her|their) hand during (?:your opponent's|his or her) next turn\.?,?", t, re.I)
    if m:
        excluded = ["Supporter"] if "except for Supporter" in t else []
        return [delayed_modifier_step("opponent_cannot_play_cards_from_hand_next_turn", "opponent", {"cannot_play_card_type_from_hand": m.group(1).capitalize(), "except_card_types": excluded}, original)], []

    m = re.fullmatch(r"Flip a coin\. If heads, your opponent can't play any (Trainer|Item|Supporter) cards from (?:his or her|their) hand during (?:his or her|their) next turn\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [delayed_modifier_step("opponent_cannot_play_cards_from_hand_next_turn", "opponent", {"cannot_play_card_type_from_hand": m.group(1).capitalize()}, original)], "source_text": original}], []

    m = re.fullmatch(r"Attach ([A-Za-z0-9éÉ' .\-]+) to 1 of your Pokémon that doesn't already have a Pokémon Tool attached to it\. If that Pokémon is Knocked Out, discard this card\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon_without_tool", "zone": "self.in_play", "filter": {"supertype": "Pokémon", "has_pokemon_tool": False}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards": "this_card", "target_ref": "chosen_own_pokemon_without_tool", "as": "Pokemon Tool", "source_text": original}, {"op": "register_trigger", "event": "attached_pokemon_knocked_out", "then": [{"op": "discard_card", "card": "this_card", "source_text": original}], "source_text": original}], []

    # --- Retreat-cost modifiers / retreat locks ---
    m = re.fullmatch(r"If this Pokémon has (?:any )?([A-Za-z]+)? ?Energy attached to it, it has no Retreat Cost\.?,?", t, re.I)
    if m:
        condition = {"self.has_attached_energy": True}
        if m.group(1):
            condition = {"self.has_attached_energy_type": m.group(1).capitalize()}
        return [{"op": "register_continuous_modifier", "modifier_id": "self_no_retreat_cost_if_energy_attached", "target": "self", "duration": {"while_source_in_play": True}, "condition": condition, "modification": {"retreat_cost": 0}, "source_text": original}], []

    if re.fullmatch(r"If this Pokémon has no Energy attached, it has no Retreat Cost\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "self_no_retreat_cost_if_no_energy_attached", "target": "self", "duration": {"while_source_in_play": True}, "condition": {"self.has_no_attached_energy": True}, "modification": {"retreat_cost": 0}, "source_text": original}], []

    if re.fullmatch(r"As long as this Pokémon is in the Active Spot, your opponent's Active Pokémon can't retreat\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "opponent_active_cannot_retreat_while_self_active", "target": "opponent.active", "condition": {"self.active_is_this_pokemon": True}, "duration": {"while_source_in_play": True}, "modification": {"retreat_allowed": False}, "source_text": original}], []

    m = re.fullmatch(r"(?:The Defending Pokémon|Your opponent's Active Pokémon) is now (Poisoned|Confused|Asleep|Paralyzed|Burned)\. (?:The Defending Pokémon|Your opponent's Active Pokémon) can't retreat during your opponent's next turn\.?,?", t, re.I)
    if m:
        return [status_condition_step(m.group(1), "opponent.active", original), delayed_modifier_step("opponent_active_cannot_retreat_next_turn", "opponent.active", {"retreat_allowed": False}, original)], []

    m = re.fullmatch(r"Flip a coin\. If heads, (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Poisoned|Confused|Asleep|Paralyzed|Burned) and can't retreat during your opponent's next turn\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [status_condition_step(m.group(1), "opponent.active", original), delayed_modifier_step("opponent_active_cannot_retreat_next_turn", "opponent.active", {"retreat_allowed": False}, original)], "source_text": original}], []

    # --- Direct / bench damage with weakness-resistance ignored ---
    m = re.fullmatch(r"This attack does (\d+) damage to 1 of your opponent's Pokémon\. This damage isn't affected by Weakness or Resistance\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_pokemon", "zone": "opponent.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "deal_damage", "target_ref": "chosen_opponent_pokemon", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"Does (\d+) damage to (\d+) of your opponent's Benched Pokémon \((\d+) if there is only \d+\)\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_bench_targets", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": {"mode": "up_to_or_available", "value": int(m.group(2))}, "source_text": original}, {"op": "deal_damage", "target_ref": "chosen_opponent_bench_targets", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, this attack does (\d+) damage to 1 of your opponent's Benched Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "deal_damage", "target_ref": "chosen_opponent_benched_pokemon", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], "source_text": original}], []

    # --- Damage-counter placement / transformation ---
    m = re.fullmatch(r"Whenever your opponent attaches an Energy card from their hand to 1 of their Pokémon, put (\d+) damage counters on that Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "register_trigger", "event": "opponent_attaches_energy_from_hand", "then": [{"op": "place_damage_counters", "target_ref": "pokemon_energy_was_attached_to", "amount": amount_exact(int(m.group(1))), "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Put (\d+) damage counters? on 1 of your opponent's Benched Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "place_damage_counters", "target_ref": "chosen_opponent_benched_pokemon", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"Put (\d+) damage counter on (?:the Defending Pokémon|your opponent's Active Pokémon)\.?,?", t, re.I)
    if m:
        return [{"op": "place_damage_counters", "target": "opponent.active", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"Put (\d+) damage counters on each of your opponent's Pokémon that has any damage counters on it\.?,?", t, re.I)
    if m:
        return [{"op": "place_damage_counters", "target": "opponent.in_play.each_with_damage", "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    if re.fullmatch(r"Double the number of damage counters on each of your opponent's Pokémon\.?,?", t, re.I):
        return [{"op": "modify_damage_counters", "target": "opponent.in_play.each", "mode": "double", "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If heads, put damage counters on your opponent's Active Pokémon until its remaining HP is 10\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "place_damage_counters_until_remaining_hp", "target": "opponent.active", "remaining_hp": 10, "source_text": original}], "source_text": original}], []

    # --- Discard / recovery / opponent-hand disruption / mill ---
    m = re.fullmatch(r"Put (\d+) cards from your discard pile into your hand\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_discard_cards", "zone": "self.discard", "filter": {"any_card": True}, "amount": amount_exact(int(m.group(1))), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_discard_cards", "destination": "self.hand", "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If heads, put a card from your discard pile into your hand\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_cards", "player": "self", "target_id": "chosen_discard_card", "zone": "self.discard", "filter": {"any_card": True}, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_discard_card", "destination": "self.hand", "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Discard the top card from your opponent's deck\.?,?", t, re.I)
    if m:
        return [{"op": "mill_cards", "player": "opponent", "amount": amount_exact(1), "source_text": original}], []

    m = re.fullmatch(r"If your opponent has (\d+) or more cards in (?:his or her|their) hand, discard a number of cards without looking until your opponent has (\d+) cards left in (?:his or her|their) hand\.?,?", t, re.I)
    if m:
        return [{"op": "discard_random_cards_until_hand_size", "player": "opponent", "condition": {"opponent.hand_size_at_least": int(m.group(1))}, "target_hand_size": int(m.group(2)), "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, choose a random card from your opponent's hand\. Your opponent reveals that card and shuffles it into (?:his or her|their) deck\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_cards", "player": "random", "target_id": "chosen_opponent_hand_card", "zone": "opponent.hand", "amount": amount_exact(1), "reveal": True, "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_hand_card", "destination": "opponent.deck", "source_text": original}, {"op": "shuffle_deck", "player": "opponent", "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Choose (\d+) random cards from your opponent's hand\. Your opponent reveals those cards and shuffles them into their deck\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "random", "target_id": "chosen_opponent_hand_cards", "zone": "opponent.hand", "amount": amount_exact(int(m.group(1))), "reveal": True, "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_hand_cards", "destination": "opponent.deck", "source_text": original}, {"op": "shuffle_deck", "player": "opponent", "source_text": original}], []

    # --- Attack locks / bounce / self recycle / delayed attachment duration ---
    if re.fullmatch(r"Flip a coin\. If heads, each Defending Pokémon can't attack during your opponent's next turn\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [delayed_modifier_step("each_defending_pokemon_cannot_attack_next_turn", "opponent.active.each", {"attack_allowed": False}, original)], "source_text": original}], []

    m = re.fullmatch(r"([A-Za-z0-9éÉ' .\-]+) can't attack during your next turn\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("named_self_cannot_attack_next_turn", "self_attacking_pokemon", {"attack_allowed": False, "pokemon_name": m.group(1)}, original)], []

    if re.fullmatch(r"Flip a coin\. If heads, your opponent returns the Defending Pokémon and all cards attached to it to (?:his or her|their) hand\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "move_pokemon_and_attached_cards", "target": "opponent.active", "destination": "opponent.hand", "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"You may shuffle this Pokémon and all attached cards into your deck\.?,?", t, re.I):
        return [{"op": "move_pokemon_and_attached_cards", "target": "self", "destination": "self.deck", "optional": True, "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    if re.fullmatch(r"If this card is attached to 1 of your Pokémon, discard it at the end of your opponent's turn\.?,?", t, re.I):
        return [{"op": "register_delayed_effect", "condition": {"this_card_attached_to_self_pokemon": True}, "trigger": "end_of_opponent_turn", "then": [{"op": "discard_card", "card": "this_card", "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"Once during your turn \(before your attack\), if this Pokémon is in your discard pile, you may put this Pokémon on the bottom of your deck\.?,?", t, re.I):
        return [{"op": "move_card", "card": "this_pokemon", "source": "self.discard", "destination": "self.deck.bottom", "optional": True, "timing": "once_during_turn_before_attack", "source_text": original}], []



    # --- v0.15 broad safe-coverage bundle: high-frequency long-tail remnants ---
    # Attack-cost modifiers / conditional free attacks.
    m = re.fullmatch(r"If you have 8 or more Stadium cards in your discard pile, ignore all Energy in this Pokémon's attack costs\.?,?", t, re.I)
    if m:
        return [{"op": "register_attack_cost_modifier", "target": "self", "condition": {"self.discard.stadium_count_at_least": 8}, "modification": {"ignore_all_energy_in_attack_costs": True}, "source_text": original}], []

    m = re.fullmatch(r"If you played ([A-Za-z0-9éÉ' .\-]+) from your hand during this turn, ignore all Energy in this Pokémon's attack costs\.?,?", t, re.I)
    if m:
        return [{"op": "register_attack_cost_modifier", "target": "self_attacking_pokemon", "condition": {"played_card_from_hand_this_turn": m.group(1)}, "modification": {"ignore_all_energy_in_attack_costs": True}, "source_text": original}], []

    m = re.fullmatch(r"If you have ([A-Za-z0-9éÉ' .\-,]+) in play, ignore all (Colorless|[A-Za-z]+) Energy in the costs of attacks used by this Pokémon\.?,?", t, re.I)
    if m:
        names = [x.strip() for x in re.split(r",| and ", m.group(1)) if x.strip()]
        return [{"op": "register_attack_cost_modifier", "target": "self", "condition": {"self.in_play_names_include": names}, "modification": {"ignore_energy_type_in_attack_costs": m.group(2).capitalize()}, "source_text": original}], []

    # Residual damage-scaling formulas.
    m = re.fullmatch(r"Does (\d+) damage times the number of Pokémon in play \(both yours and your opponent's\)\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "base_per_unit": int(m.group(1)), "count": {"zone": "all.in_play", "filter": {"supertype": "Pokémon"}}, "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage times the number of damage counters on this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "base_per_unit": int(m.group(1)), "count": {"damage_counters_on": "self_attacking_pokemon"}, "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage times the number of Pokémon in your discard pile that have the ([A-Za-z0-9éÉ' .\-]+) attack\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "base_per_unit": int(m.group(1)), "count": {"zone": "self.discard", "filter": {"supertype": "Pokémon", "has_attack_name": m.group(2)}}, "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage times the number of your Benched Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "set_attack_damage_from_count", "base_per_unit": int(m.group(1)), "count": {"zone": "self.bench", "filter": {"supertype": "Pokémon"}}, "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage for each Energy attached to (both Active Pokémon|your opponent's Active Pokémon)\.?,?", t, re.I)
    if m:
        target = "all.active" if m.group(2).lower().startswith("both") else "opponent.active"
        return [{"op": "set_attack_damage_from_count", "base_per_unit": int(m.group(1)), "count": {"attached_energy_on": target}, "source_text": original}], []

    m = re.fullmatch(r"If this Pokémon has a Pokémon Tool card attached to it, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"self_attacking_pokemon_has_tool": True}, "source_text": original}], []

    m = re.fullmatch(r"If your opponent's Active Pokémon is (Asleep|Confused|Paralyzed|Poisoned|Burned), this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(2)), "condition": {"opponent.active.special_condition": m.group(1).capitalize()}, "source_text": original}], []

    m = re.fullmatch(r"If you have more Prize cards remaining than your opponent, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"self.prize_cards_remaining_gt_opponent": True}, "source_text": original}], []

    if re.fullmatch(r"If your opponent's Active Pokémon is a Basic Pokémon, it is Knocked Out\.?,?", t, re.I):
        return [{"op": "knock_out", "target": "opponent.active", "condition": {"subtypes": ["Basic"]}, "source_text": original}], []

    m = re.fullmatch(r"If your opponent's Active Pokémon isn't (Asleep|Confused|Paralyzed|Poisoned|Burned), this attack does nothing\.?,?", t, re.I)
    if m:
        return [{"op": "play_condition", "condition": {"opponent.active.special_condition": m.group(1).capitalize()}, "if_not_met": "attack_does_nothing", "source_text": original}], []

    if re.fullmatch(r"If this Pokémon evolved during this turn, this attack does nothing\.?,?", t, re.I):
        return [{"op": "play_condition", "condition": {"self.evolved_this_turn": False}, "if_not_met": "attack_does_nothing", "source_text": original}], []

    m = re.fullmatch(r"If you go second, you can't use this attack during your first turn\. This attack does (\d+) damage for each of your Benched Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "play_condition", "condition": {"not": {"self.going_second_first_turn": True}}, "source_text": original}, {"op": "set_attack_damage_from_count", "base_per_unit": int(m.group(1)), "count": {"zone": "self.bench", "filter": {"supertype": "Pokémon"}}, "source_text": original}], []

    # Search / bench / evolution / attach-from-deck patterns.
    m = re.fullmatch(r"Search your deck for up to (\d+) [Bb]asic Energy cards and attach them to your Pokémon in any way you like\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_basic_energy", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_up_to(int(m.group(1))), "reveal": True, "destination": "self.in_play.attach_any_way", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for up to (\d+) Basic Pokémon of different types and put them onto your Bench\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_basic_pokemon_different_types", "filter": {"supertype": "Pokémon", "subtypes": ["Basic"], "different_types": True}, "amount": amount_up_to(int(m.group(1))), "destination": "self.bench", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for up to (\d+) Rare Fossil cards and put them onto your Bench\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_rare_fossils", "filter": {"name": "Rare Fossil"}, "amount": amount_up_to(int(m.group(1))), "destination": "self.bench", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for ([A-Za-z0-9éÉ' .\-,]+), or ([A-Za-z0-9éÉ' .\-]+) and put up to (\d+) of them onto your Bench\. Shuffle your deck afterward\. Treat the new Benched Pokémon as Basic Pokémon\.?,?", t, re.I)
    if m:
        names = [x.strip() for x in re.split(r",| or ", m.group(1) + ", " + m.group(2)) if x.strip()]
        return [{"op": "search_deck", "player": "self", "target_id": "searched_named_fossil_pokemon", "filter": {"names": names}, "amount": amount_up_to(int(m.group(3))), "destination": "self.bench", "treat_as_basic": True, "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"For each of your Benched Pokémon, search your deck for a card that evolves from that Pokémon and put it onto that Pokémon to evolve it\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck_for_evolution_each", "player": "self", "source_zone": "self.deck", "targets": "self.bench.each", "destination": "evolve_target", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for a card that evolves from 1 of your Pokémon and put it onto that Pokémon to evolve it\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon_to_evolve", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "search_deck", "player": "self", "target_id": "searched_evolution_card", "filter": {"evolves_from_ref": "chosen_own_pokemon_to_evolve"}, "amount": amount_exact(1), "destination": "evolve_target", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for a (Single Strike|Rapid Strike|Fusion Strike) Pokémon and put it onto your Bench\. Then, shuffle your deck\. If you searched your deck in this way, draw (\d+) cards\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_battle_style_pokemon", "filter": {"supertype": "Pokémon", "battle_style": m.group(1)}, "amount": amount_exact(1), "destination": "self.bench", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}, {"op": "draw_cards", "player": "self", "amount": int(m.group(2)), "condition": {"searched_deck_this_effect": True}, "source_text": original}], []

    m = re.fullmatch(r"Search your deck for up to (\d+) Basic (Rapid Strike|Single Strike|Fusion Strike) Pokémon and put them onto your Bench\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_basic_battle_style_pokemon", "filter": {"supertype": "Pokémon", "subtypes": ["Basic"], "battle_style": m.group(2)}, "amount": amount_up_to(int(m.group(1))), "destination": "self.bench", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Choose up to (\d+) of your Fusion Strike Pokémon\. For each of those Pokémon, search your deck for a Fusion Strike Energy card and attach it to that Pokémon\. Then, shuffle your deck\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_fusion_strike_pokemon", "zone": "self.in_play", "filter": {"battle_style": "Fusion Strike"}, "amount": amount_up_to(int(m.group(1))), "source_text": original}, {"op": "search_deck", "player": "self", "target_id": "searched_fusion_strike_energy", "filter": {"name": "Fusion Strike Energy"}, "amount": {"mode": "match_targets", "targets_ref": "chosen_fusion_strike_pokemon"}, "destination": "attach_to_chosen_targets", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    # Energy discard / attachment remnants.
    m = re.fullmatch(r"Discard (\d+) ([A-Za-z]+) Energy attached to this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "filter": {"types": [m.group(2).capitalize()], "energy_type": m.group(2).capitalize()}, "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    m = re.fullmatch(r"Discard a ([A-Za-z]+) Energy from this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "filter": {"types": [m.group(1).capitalize()], "energy_type": m.group(1).capitalize()}, "amount": amount_exact(1), "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If tails, discard a ([A-Za-z]+) Energy attached to this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "tails", "then": [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "filter": {"types": [m.group(1).capitalize()], "energy_type": m.group(1).capitalize()}, "amount": amount_exact(1), "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Flip (\d+) coins\. For each heads, discard an Energy from your opponent's Active Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": int(m.group(1)), "target_id": "coin_results", "source_text": original}, {"op": "discard_attached_energy_per_coin_heads", "target": "opponent.active", "coin_results_ref": "coin_results", "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If heads, your opponent's Active Pokémon is now Paralyzed, and discard an Energy from that Pokémon\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [status_condition_step("Paralyzed", "opponent.active", original), {"op": "discard_attached_energy", "target": "opponent.active", "amount": amount_exact(1), "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, choose 1 Energy card attached to 1 of your opponent's Pokémon and discard it\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_pokemon_with_energy", "zone": "opponent.in_play", "filter": {"has_attached_energy": True}, "amount": amount_exact(1), "source_text": original}, {"op": "discard_attached_energy", "target_ref": "chosen_opponent_pokemon_with_energy", "amount": amount_exact(1), "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Attach up to (\d+) ([A-Za-z]+) Energy cards from your discard pile to this Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_typed_energy", "zone": "self.discard", "filter": {"supertype": "Energy", "types": [m.group(2).capitalize()]}, "amount": amount_up_to(int(m.group(1))), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_typed_energy", "target": "self_attacking_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Attach up to (\d+) Basic ([A-Za-z]+) Energy cards from your discard pile to your Benched Pokémon in any way you like\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_basic_typed_energy", "zone": "self.discard", "filter": {"supertype": "Energy", "subtypes": ["Basic"], "types": [m.group(2).capitalize()]}, "amount": amount_up_to(int(m.group(1))), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_basic_typed_energy", "target": "self.bench.any_way", "source_text": original}], []

    # Draw / shuffle / hand refresh remnants.
    m = re.fullmatch(r"Each player draws (\d+) cards\.?,?", t, re.I)
    if m:
        return [{"op": "draw_cards", "player": "self", "amount": int(m.group(1)), "source_text": original}, {"op": "draw_cards", "player": "opponent", "amount": int(m.group(1)), "source_text": original}], []

    if re.fullmatch(r"Draw a card\. If you do, this Pokémon is now Asleep\.?,?", t, re.I):
        return [{"op": "draw_cards", "player": "self", "amount": 1, "source_text": original}, status_condition_step("Asleep", "self_attacking_pokemon", original)], []

    m = re.fullmatch(r"Flip a coin until you get tails\. For each heads, draw (\d+) cards\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip_until", "player": "self", "until": "tails", "target_id": "coin_results_until_tails", "source_text": original}, {"op": "draw_cards_per_coin_heads", "player": "self", "amount_per_heads": int(m.group(1)), "coin_results_ref": "coin_results_until_tails", "source_text": original}], []

    m = re.fullmatch(r"Draw a number of cards up to the number of your opponent's Pokémon in play\.?,?", t, re.I)
    if m:
        return [{"op": "draw_cards", "player": "self", "amount": {"mode": "up_to_count", "count": {"zone": "opponent.in_play", "filter": {"supertype": "Pokémon"}}}, "source_text": original}], []

    m = re.fullmatch(r"Draw (\d+) cards from the bottom of your deck\.?,?", t, re.I)
    if m:
        return [{"op": "draw_cards", "player": "self", "amount": int(m.group(1)), "from": "deck.bottom", "source_text": original}], []

    m = re.fullmatch(r"Draw (\d+) cards\. Flip a coin\. If heads, draw (\d+) more cards\.?,?", t, re.I)
    if m:
        return [{"op": "draw_cards", "player": "self", "amount": int(m.group(1)), "source_text": original}, {"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "draw_cards", "player": "self", "amount": int(m.group(2)), "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Each player shuffles (?:his or her|their) hand into (?:his or her|their) deck and draws (\d+) cards\.?,?", t, re.I)
    if m:
        return [{"op": "shuffle_hand_into_deck", "player": "each", "source_text": original}, {"op": "draw_cards", "player": "each", "amount": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"Shuffle your hand into your deck\. Then, draw a number of cards equal to the number of Benched Pokémon \(both yours and your opponent's\)\.?,?", t, re.I)
    if m:
        return [{"op": "shuffle_hand_into_deck", "player": "self", "source_text": original}, {"op": "draw_cards", "player": "self", "amount": {"mode": "count", "count": {"zone": "all.bench", "filter": {"supertype": "Pokémon"}}}, "source_text": original}], []

    if re.fullmatch(r"Discard as many cards as you like from your hand\. Then, draw that many cards\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "discarded_from_hand", "zone": "self.hand", "filter": {"any_card": True}, "amount": {"mode": "any"}, "source_text": original}, {"op": "discard_cards", "player": "self", "cards_ref": "discarded_from_hand", "destination": "discard", "source_text": original}, {"op": "draw_cards", "player": "self", "amount": {"mode": "count_ref", "cards_ref": "discarded_from_hand"}, "source_text": original}], []

    # Switch / gust remnants.
    if re.fullmatch(r"If your opponent has any Benched Pokémon, (?:he or she|they) chooses 1 of them and switches it with (?:his or her|their) Active Pokémon, then, if you have any Benched Pokémon, you switch 1 of them with your Active Pokémon\. \(Do the damage before switching the Pokémon\.\)\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "opponent", "chooser": "opponent", "new_active_ref": "opponent_choice_from_bench", "timing": "after_damage", "source_text": original}, {"op": "switch_active", "player": "self", "chooser": "self", "new_active_ref": "self_choice_from_bench", "timing": "after_damage", "source_text": original}], []

    if re.fullmatch(r"Switch this Pokémon with 1 of your Benched Pokémon\. Then, your opponent switches the Defending Pokémon with 1 of (?:his or her|their) Benched Pokémon\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "self", "chooser": "self", "new_active_ref": "self_choice_from_bench", "source_text": original}, {"op": "switch_active", "player": "opponent", "chooser": "opponent", "new_active_ref": "opponent_choice_from_bench", "source_text": original}], []

    m = re.fullmatch(r"Switch this Pokémon with 1 of your Benched ([A-Za-z]+) Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_own_benched_typed_pokemon", "zone": "self.bench", "filter": {"supertype": "Pokémon", "types": [m.group(1).capitalize()]}, "amount": amount_exact(1), "source_text": original}, {"op": "switch_active", "player": "self", "new_active_ref": "chosen_own_benched_typed_pokemon", "source_text": original}], []

    if re.fullmatch(r"You may switch out your opponent's Active Pokémon to the Bench\. \(Your opponent chooses the new Active Pokémon\.\)\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "opponent", "chooser": "opponent", "optional": True, "new_active_ref": "opponent_choice_from_bench", "source_text": original}], []

    if re.fullmatch(r"Switch this Pokémon with 1 of your Benched Pokémon\. If you do, switch out your opponent's Active Pokémon to the Bench\. \(Your opponent chooses the new Active Pokémon\.\)\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "self", "chooser": "self", "new_active_ref": "self_choice_from_bench", "source_text": original}, {"op": "switch_active", "player": "opponent", "chooser": "opponent", "new_active_ref": "opponent_choice_from_bench", "source_text": original}], []

    if re.fullmatch(r"Switch 1 of your opponent's Benched Pokémon with 1 of the Defending Pokémon\. Your opponent chooses the Defending Pokémon to switch\. The new Defending Pokémon is now Asleep\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "opponent", "chooser": "self", "new_active_ref": "chosen_opponent_benched_pokemon", "source_text": original}, status_condition_step("Asleep", "opponent.active", original)], []

    # Prevention / damage-reduction remnants.
    if re.fullmatch(r"During your opponent's next turn, if this Pokémon is damaged by an attack \(even if it is Knocked Out\), put 8 damage counters on the Attacking Pokémon\.?,?", t, re.I):
        return [delayed_modifier_step("retaliate_8_damage_counters_when_damaged_next_turn", "self_attacking_pokemon", {"on_damaged_by_attack": {"place_damage_counters_on_attacker": 8}}, original)], []

    m = re.fullmatch(r"During your opponent's next turn, if this Pokémon is damaged by an attack \(even if this Pokémon is Knocked Out\), put damage counters on the Attacking Pokémon equal to the damage done to this Pokémon\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("retaliate_damage_counters_equal_damage_received_next_turn", "self_attacking_pokemon", {"on_damaged_by_attack": {"place_damage_counters_on_attacker_equal_damage_done": True}}, original)], []

    m = re.fullmatch(r"During your opponent's next turn, attacks used by the Defending Pokémon do (\d+) less damage \(before applying Weakness and Resistance\)\.?,?", t, re.I)
    if m:
        return [delayed_modifier_step("opponent_active_attack_damage_reduced_next_turn", "opponent.active", {"damage_done_by_attacks_delta": -int(m.group(1)), "timing_basis": "before_weakness_resistance"}, original)], []

    if re.fullmatch(r"During your opponent's next turn, prevent all damage done to this Pokémon by attacks from Basic non-Colorless Pokémon\.?,?", t, re.I):
        return [delayed_modifier_step("prevent_attack_damage_from_basic_non_colorless_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True, "attacker_filter": {"subtypes": ["Basic"], "types_exclude": ["Colorless"]}}, original)], []

    if re.fullmatch(r"If your opponent's Pokémon is Knocked Out by damage from this attack, during your opponent's next turn, prevent all damage from and effects of attacks done to this Pokémon\.?,?", t, re.I):
        return [{"op": "register_delayed_effect", "condition": {"opponent_pokemon_knocked_out_by_this_attack": True}, "trigger": "start_of_opponent_next_turn", "then": [delayed_modifier_step("prevent_attack_damage_and_effects_to_self_next_turn_after_ko", "self_attacking_pokemon", {"prevent_damage_from_attacks": True, "prevent_effects_of_attacks": True}, original)], "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If heads, during your opponent's next turn, prevent all effects of attacks, including damage, done to this Pokémon\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [delayed_modifier_step("prevent_attack_damage_and_effects_to_self_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True, "prevent_effects_of_attacks": True}, original)], "source_text": original}], []

    # Opponent hand/deck information and disruption.
    if re.fullmatch(r"Your opponent reveals their hand\. Choose a card you find there and put it on the bottom of their deck\.?,?", t, re.I):
        return [{"op": "reveal_hand", "player": "opponent", "source_text": original}, {"op": "choose_cards", "player": "self", "target_id": "chosen_opponent_hand_card", "zone": "opponent.hand", "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_opponent_hand_card", "destination": "opponent.deck.bottom", "source_text": original}], []

    m = re.fullmatch(r"Look at the top (\d+) cards of your opponent's deck and put them back in any order\.?,?", t, re.I)
    if m:
        return [{"op": "look_at_top_cards", "player": "self", "target_player": "opponent", "zone": "deck", "amount": int(m.group(1)), "target_id": "looked_cards", "source_text": original}, {"op": "reorder_cards", "player": "self", "cards_ref": "looked_cards", "destination": "opponent.deck.top", "source_text": original}], []

    # Misc. common residual effects.
    if re.fullmatch(r"Each player plays with (?:his or her|their) Prize cards face up for the rest of the game\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "prize_cards_face_up_for_rest_of_game", "target": "each_player", "duration": {"until": "end_of_game"}, "modification": {"prize_cards_face_up": True}, "source_text": original}], []

    if re.fullmatch(r"If your opponent's Pokémon is Knocked Out by damage from this attack, take 1 more Prize card\.?,?", t, re.I):
        return [{"op": "register_knockout_prize_bonus", "condition": {"opponent_pokemon_knocked_out_by_this_attack": True}, "bonus_prizes": 1, "source_text": original}], []

    if re.fullmatch(r"At the end of your opponent's next turn, the Defending Pokémon will be Knocked Out\.?,?", t, re.I):
        return [{"op": "register_delayed_effect", "trigger": "end_of_opponent_next_turn", "then": [{"op": "knock_out", "target": "opponent.active", "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"Each of your evolved Pokémon can use any attack from its previous Evolutions\. \(You still need the necessary Energy to use each attack\.\)\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "evolved_pokemon_can_use_previous_evolution_attacks", "target": "self.in_play.each", "filter": {"is_evolved": True}, "duration": {"while_source_in_play": True}, "modification": {"can_use_attacks_from_previous_evolutions": True, "still_need_energy": True}, "source_text": original}], []

    if re.fullmatch(r"Once during your turn, if you drew this Pokémon from your deck at the beginning of your turn and your Bench isn't full, before you put it into your hand, you may put it onto your Bench\.?,?", t, re.I):
        return [{"op": "register_replacement_effect", "event": "draw_this_pokemon_for_turn", "condition": {"self.bench_not_full": True}, "then": [{"op": "move_card", "card": "this_pokemon", "destination": "self.bench", "optional": True, "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"Search your opponent's discard pile for a Supporter card and use the effect of that card as the effect of this attack\. \(The Supporter card remains in your opponent's discard pile\.\)\.?,?", t, re.I):
        return [{"op": "copy_card_effect_from_discard", "player": "self", "target_player": "opponent", "filter": {"supertype": "Trainer", "subtypes": ["Supporter"]}, "destination_effect": "this_attack", "leave_card_in_discard": True, "source_text": original}], []


    # --- v0.16 broad safe-coverage bundle: newly exposed medium-frequency remnants ---
    # Self-switch / retreat-style switching.
    if re.fullmatch(r"Flip a coin\. If heads, switch this Pokémon with 1 of your Benched Pokémon\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "switch_active", "player": "self", "chooser": "self", "new_active_ref": "self_choice_from_bench", "source_text": original}], "source_text": original}], []

    # Common gust variants that survived v0.15.
    if re.fullmatch(r"Flip a coin\. If heads, switch in 1 of your opponent's Benched Pokémon to the Active Spot\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "switch_active", "player": "opponent", "new_active_ref": "chosen_opponent_benched_pokemon", "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"Before doing damage, you may choose 1 of your opponent's Benched Pokémon and switch it with (?:the Defending Pokémon|1 of the Defending Pokémon)\. Your opponent chooses the Defending Pokémon to switch\.?,?", t, re.I):
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "optional": True, "timing": "before_damage", "source_text": original}, {"op": "switch_active", "player": "opponent", "new_active_ref": "chosen_opponent_benched_pokemon", "timing": "before_damage", "source_text": original}], []

    if re.fullmatch(r"Before doing damage, you may choose 1 of your opponent's Benched Pokémon and switch it with the Defending Pokémon\.?,?", t, re.I):
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_benched_pokemon", "zone": "opponent.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "optional": True, "timing": "before_damage", "source_text": original}, {"op": "switch_active", "player": "opponent", "new_active_ref": "chosen_opponent_benched_pokemon", "timing": "before_damage", "source_text": original}], []

    if re.fullmatch(r"Switch 1 of your opponent's Benched Pokémon with their Active Pokémon\. If you do, switch your Active Pokémon with 1 of your Benched Pokémon\.?,?", t, re.I):
        return [{"op": "switch_active", "player": "opponent", "chooser": "self", "new_active_ref": "chosen_opponent_benched_pokemon", "source_text": original}, {"op": "switch_active", "player": "self", "chooser": "self", "new_active_ref": "self_choice_from_bench", "source_text": original}], []

    if re.fullmatch(r"Flip a coin\. If heads, switch 1 of your opponent's Benched Pokémon with (?:his or her|their) Active Pokémon\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "switch_active", "player": "opponent", "chooser": "self", "new_active_ref": "chosen_opponent_benched_pokemon", "source_text": original}], "source_text": original}], []

    # Top-deck selection / reveal search variants.
    m = re.fullmatch(r"Look at the top (\d+) cards of your deck, choose 1 of them, and put it into your hand\. Put the other cards back on top of your deck\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "look_at_top_cards", "player": "self", "target_id": "looked_cards", "amount": amount_exact(int(m.group(1))), "source_text": original}, {"op": "choose_cards", "player": "self", "target_id": "chosen_card", "cards_ref": "looked_cards", "amount": amount_exact(1), "destination": "self.hand", "source_text": original}, {"op": "move_card", "cards_ref": "chosen_card", "destination": "self.hand", "source_text": original}, {"op": "put_cards_back", "cards_ref": "looked_cards.minus(chosen_card)", "destination": "self.deck.top", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Look at the top (\d+) cards of your deck, choose 1 of them, and put it into your hand\. Put the other card on the bottom of your deck\.?,?", t, re.I)
    if m:
        return [{"op": "look_at_top_cards", "player": "self", "target_id": "looked_cards", "amount": amount_exact(int(m.group(1))), "source_text": original}, {"op": "choose_cards", "player": "self", "target_id": "chosen_card", "cards_ref": "looked_cards", "amount": amount_exact(1), "destination": "self.hand", "source_text": original}, {"op": "move_card", "cards_ref": "chosen_card", "destination": "self.hand", "source_text": original}, {"op": "put_cards_back", "cards_ref": "looked_cards.minus(chosen_card)", "destination": "self.deck.bottom", "source_text": original}], []

    m = re.fullmatch(r"Look at the top (\d+) cards of your deck\. You may reveal a (Pokémon|Energy|Supporter) card you find there and put it into your hand\. Shuffle the other cards back into your deck\.?,?", t, re.I)
    if m:
        kind = m.group(2).capitalize()
        filt = {"supertype": "Pokémon"} if kind == "Pokémon" else ({"supertype": "Energy"} if kind == "Energy" else {"supertype": "Trainer", "subtypes": ["Supporter"]})
        return [{"op": "look_at_top_cards", "player": "self", "target_id": "looked_cards", "amount": amount_exact(int(m.group(1))), "source_text": original}, {"op": "choose_cards", "player": "self", "target_id": "chosen_revealed_card", "cards_ref": "looked_cards", "filter": filt, "amount": amount_up_to(1), "reveal": True, "destination": "self.hand", "source_text": original}, {"op": "move_card", "cards_ref": "chosen_revealed_card", "destination": "self.hand", "source_text": original}, {"op": "shuffle_into_deck", "cards_ref": "looked_cards.minus(chosen_revealed_card)", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, search your deck for (?:a|an) (Supporter|Item) card, reveal it, and put it into your hand\. (?:Then, shuffle your deck|Shuffle your deck afterward)\.?,?", t, re.I)
    if m:
        subtype = m.group(1).capitalize()
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "search_deck", "player": "self", "target_id": f"searched_{subtype.lower()}", "filter": {"supertype": "Trainer", "subtypes": [subtype]}, "amount": amount_exact(1), "reveal": True, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], "source_text": original}], []

    # Energy from discard to hand / movement variants.
    m = re.fullmatch(r"Flip 3 coins\. For each heads, put (?:a|1) Basic Energy card from your discard pile into your hand\. If you don't have that many basic Energy cards in your discard pile, put all of them into your hand\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 3, "target_id": "coin_results", "source_text": original}, {"op": "recover_cards_from_discard_per_coin_heads", "player": "self", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "destination": "self.hand", "coin_results_ref": "coin_results", "source_text": original}], []

    m = re.fullmatch(r"Rules: Flip 3 coins\. For each heads, put (?:a|1) Basic Energy card from your discard pile into your hand\. If you don't have that many basic Energy cards in your discard pile, put all of them into your hand\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 3, "target_id": "coin_results", "source_text": original}, {"op": "recover_cards_from_discard_per_coin_heads", "player": "self", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "destination": "self.hand", "coin_results_ref": "coin_results", "source_text": original}], []

    m = re.fullmatch(r"Move a basic Energy(?: card)? attached to 1 of your Pokémon to another of your Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_basic_energy", "zone": "attached_to:self.in_play", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(1), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_other_own_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_energy", "cards_ref": "chosen_basic_energy", "destination_ref": "chosen_other_own_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Rules: Move a basic Energy card attached to 1 of your Pokémon to another of your Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_basic_energy", "zone": "attached_to:self.in_play", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(1), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_other_own_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_energy", "cards_ref": "chosen_basic_energy", "destination_ref": "chosen_other_own_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Put (\d+) (?:basic )?([A-Za-z]+ )?Energy cards? from your discard pile into your hand\.?,?", t, re.I)
    if m:
        filt = {"supertype": "Energy"}
        if m.group(2):
            filt["types"] = [m.group(2).strip().capitalize()]
        if "basic" in t.lower():
            filt["subtypes"] = ["Basic"]
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_energy_from_discard", "zone": "self.discard", "filter": filt, "amount": amount_exact(int(m.group(1))), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_energy_from_discard", "destination": "self.hand", "source_text": original}], []

    # Conditional / revealed hand / retaliation damage remnants.
    m = re.fullmatch(r"Your opponent reveals their hand\. This attack does (\d+) damage for each Trainer card you find there\.?,?", t, re.I)
    if m:
        return [{"op": "reveal_hand", "player": "opponent", "source_text": original}, {"op": "set_attack_damage_from_count", "base_per_unit": int(m.group(1)), "count": {"zone": "opponent.hand", "filter": {"supertype": "Trainer"}}, "source_text": original}], []

    if re.fullmatch(r"If this Pokémon was damaged by an attack during your opponent's last turn, this attack does that much more damage\.?,?", t, re.I):
        return [{"op": "modify_attack_damage", "mode": "add", "amount": {"mode": "damage_taken_last_opponent_turn", "target": "self"}, "condition": {"self_damaged_by_attack_last_opponent_turn": True}, "source_text": original}], []

    m = re.fullmatch(r"If any of your Pokémon were Knocked Out by damage from an attack during your opponent's last turn, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"self.pokemon_knocked_out_by_opponent_attack_last_turn": True}, "source_text": original}], []

    m = re.fullmatch(r"If the Defending Pokémon is Pokémon-ex, this attack does \d+ damage plus (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"opponent.active.tags_include": ["Pokémon-ex"]}, "source_text": original}], []

    m = re.fullmatch(r"If the Defending Pokémon already has any damage counters on it, this attack does \d+ damage plus (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"opponent.active.damage_counters_gt": 0}, "source_text": original}], []

    m = re.fullmatch(r"If the Defending Pokémon already has any damage counters on it, this attack does (\d+) more damage\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(1)), "condition": {"opponent.active.damage_counters_gt": 0}, "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) less damage for each Colorless in your opponent's Active Pokémon's Retreat Cost\.?,?", t, re.I)
    if m:
        return [{"op": "modify_attack_damage_from_count", "mode": "subtract", "amount_per_unit": int(m.group(1)), "count": {"opponent.active.retreat_cost_type": "Colorless"}, "source_text": original}], []

    m = re.fullmatch(r"During your next turn, attacks used by this Pokémon do (\d+) more damage to your opponent's Active Pokémon \(before applying Weakness and Resistance\)\.?,?", t, re.I)
    if m:
        return [{"op": "register_continuous_modifier", "modifier_id": "self_next_turn_attack_damage_bonus", "target": "self_attacking_pokemon", "duration": {"until": "end_of_your_next_turn"}, "modification": {"attack_damage_delta": int(m.group(1)), "target": "opponent.active", "timing_basis": "before_weakness_resistance"}, "source_text": original}], []

    # Special Energy providing / Special Tool-like energy attachment long text.
    if re.fullmatch(r"As long as this card is attached to a Pokémon, it provides Colorless Energy\.?,?", t, re.I):
        return [{"op": "provide_energy", "types": ["Colorless"], "amount": 1, "while_attached": True, "source_text": original}], []

    if re.fullmatch(r"You may attach this as an Energy card from your hand to 1 of your Pokémon that already has an Energy card attached to it\. When you attach this card, return an Energy card attached to that Pokémon to your hand\. While attached, this card is a Special Energy card and provides every type of Energy but 2 Energy at a time\. \(Has no effect other than providing Energy\.\)\.?,?", t, re.I):
        return [{"op": "attach_card", "player": "self", "from_zone": "hand", "target": "self.pokemon_with_energy_attached", "as": "Special Energy", "source_text": original}, {"op": "move_card", "player": "self", "cards": "one_energy_attached_to_target", "destination": "self.hand", "source_text": original}, {"op": "provide_energy", "types": "any", "amount": 2, "while_attached": True, "source_text": original}], []

    # Remaining condition/status + retreat lock composites.
    m = re.fullmatch(r"Your opponent's Active Pokémon is now (Confused|Burned)\. During your opponent's next turn, that Pokémon can't retreat\.?,?", t, re.I)
    if m:
        return [status_condition_step(m.group(1), "opponent.active", original), delayed_modifier_step("opponent_active_cannot_retreat_next_turn", "opponent.active", {"cannot_retreat": True}, original)], []

    if re.fullmatch(r"If this Pokémon is your Active Pokémon and is damaged by an opponent's attack \(even if this Pokémon is Knocked Out\), the Attacking Pokémon is now Confused\.?,?", t, re.I):
        return [{"op": "register_trigger", "trigger": "self_active_damaged_by_opponent_attack", "effect": status_condition_step("Confused", "attacking_pokemon", original), "source_text": original}], []

    # Simple self return/bounce and Tool/item attach remnants.
    if re.fullmatch(r"Flip a coin\. If heads, return 1 of your Pokémon and all cards attached to it to your hand\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_zone_to_zone", "from": "self.in_play", "to": "self.hand", "target_ref": "chosen_own_pokemon", "include_attached_cards": True, "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"Rules: Flip a coin\. If heads, return 1 of your Pokémon and all cards attached to it to your hand\.?,?", t, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_target", "player": "self", "target_id": "chosen_own_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_zone_to_zone", "from": "self.in_play", "to": "self.hand", "target_ref": "chosen_own_pokemon", "include_attached_cards": True, "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"Attach PlusPower to 1 of your Pokémon\. Discard this card at the end of your turn\.?,?", t, re.I):
        return [{"op": "attach_card", "player": "self", "from_zone": "hand", "target": "self.pokemon", "subtype": "Tool", "source_text": original}, {"op": "register_delayed_effect", "trigger": "end_of_turn", "then": [{"op": "discard_card", "target": "this_card", "source_text": original}], "source_text": original}], []



    # --- v0.17 broad safe-coverage bundle: post-classifier v0.5 remnants ---
    # The v0.5 long-tail classifier split the remaining "other" bucket into
    # older Trainer/Tool/Stadium rules, top-deck manipulation, Energy movement,
    # global locks/modifiers, and coin-based mill/recovery. Keep these patterns
    # conservative and encode unclear wording as structured reference/modifier ops.

    # Mill / discard cards from top of opponent's deck, including per-coin-heads variants.
    m = re.fullmatch(r"Flip (\d+) coins?\. For each heads, discard (\d+) cards? from the top of your opponent's deck\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": int(m.group(1)), "target_id": "coin_results", "source_text": original}, {"op": "discard_cards_from_deck_per_coin_heads", "player": "opponent", "cards_per_heads": int(m.group(2)), "coin_results_ref": "coin_results", "destination": "discard", "source_text": original}], []

    m = re.fullmatch(r"Discard (\d+) cards? from the top of your opponent's deck\.?,?", t, re.I)
    if m:
        return [{"op": "discard_cards_from_deck", "player": "opponent", "amount": amount_exact(int(m.group(1))), "destination": "discard", "source_text": original}], []

    # Generic top-deck look / choose / reorder effects.
    if re.fullmatch(r"Look at the top card of your deck\. You may discard that card\.?,?", t, re.I):
        return [{"op": "look_at_top_cards", "player": "self", "amount": 1, "target_id": "top_card", "source_text": original}, {"op": "choose_mode", "player": "self", "choices": [{"label": "discard_top_card", "steps": [{"op": "move_card", "cards_ref": "top_card", "destination": "self.discard", "source_text": original}]}, {"label": "leave_top_card", "steps": []}], "source_text": original}], []

    if re.fullmatch(r"Look at the top card of your deck\. You may put that card into your hand\. If you don't, discard that card and draw a card\.?,?", t, re.I):
        return [{"op": "look_at_top_cards", "player": "self", "amount": 1, "target_id": "top_card", "source_text": original}, {"op": "choose_mode", "player": "self", "choices": [{"label": "put_top_card_into_hand", "steps": [{"op": "move_card", "cards_ref": "top_card", "destination": "self.hand", "source_text": original}]}, {"label": "discard_top_card_and_draw", "steps": [{"op": "move_card", "cards_ref": "top_card", "destination": "self.discard", "source_text": original}, {"op": "draw_cards", "player": "self", "amount": 1, "source_text": original}]}], "source_text": original}], []

    m = re.fullmatch(r"Look at the top (\d+) cards of your deck and put them back in any order\.?,?", t, re.I)
    if m:
        return [{"op": "look_at_top_cards", "player": "self", "amount": int(m.group(1)), "target_id": "looked_cards", "source_text": original}, {"op": "reorder_cards", "player": "self", "cards_ref": "looked_cards", "destination": "self.deck.top", "source_text": original}], []

    m = re.fullmatch(r"Look at the top (\d+) cards of your deck\. You may reveal (?:a|an) (Pokémon|Energy|Supporter|Item) card you find there and put it into your hand\. Shuffle the other cards back into your deck\.?,?", t, re.I)
    if m:
        kind = m.group(2).capitalize()
        filt = {"supertype": "Trainer" if kind in {"Supporter", "Item"} else kind}
        if kind in {"Supporter", "Item"}:
            filt["subtypes"] = [kind]
        return [{"op": "look_at_top_cards", "player": "self", "amount": int(m.group(1)), "target_id": "looked_cards", "source_text": original}, {"op": "choose_cards", "player": "self", "target_id": "chosen_card", "from_ref": "looked_cards", "filter": filt, "amount": amount_up_to(1), "reveal": True, "source_text": original}, {"op": "move_card", "cards_ref": "chosen_card", "destination": "self.hand", "source_text": original}, {"op": "shuffle_cards_into_deck", "player": "self", "cards_ref": "looked_cards - chosen_card", "source_text": original}], []

    m = re.fullmatch(r"Look at the top (\d+) cards of your deck, choose (\d+) of them, and put (?:it|them) into your hand\. Put the other cards back on top of your deck\. Shuffle your deck afterward\.?,?", t, re.I)
    if m:
        return [{"op": "look_at_top_cards", "player": "self", "amount": int(m.group(1)), "target_id": "looked_cards", "source_text": original}, {"op": "choose_cards", "player": "self", "target_id": "chosen_cards_to_hand", "from_ref": "looked_cards", "filter": {"any_card": True}, "amount": amount_exact(int(m.group(2))), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_cards_to_hand", "destination": "self.hand", "source_text": original}, {"op": "move_card", "cards_ref": "looked_cards - chosen_cards_to_hand", "destination": "self.deck.top", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    # Energy movement / recovery variants still appearing in the tail.
    if re.fullmatch(r"Move a basic Energy(?: card)? from 1 of your Pokémon to another of your Pokémon\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_basic_energy", "zone": "attached_to:self.in_play", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(1), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_destination_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_energy", "cards_ref": "chosen_basic_energy", "destination_ref": "chosen_destination_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Once during each player's turn, that player may put a basic Energy card from their discard pile into their hand\.?,?", t, re.I)
    if m:
        return [{"op": "register_activated_effect", "scope": "each_player_turn", "usage_limit": 1, "effect": [{"op": "choose_cards", "player": "turn_player", "target_id": "chosen_basic_energy", "zone": "turn_player.discard", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_basic_energy", "destination": "turn_player.hand", "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Flip (\d+) coins?\. For each heads, put (?:a|1) Basic Energy card from your discard pile into your hand\. If you don't have that many basic Energy cards in your discard pile, put all of them into your hand\.?,?", t, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": int(m.group(1)), "target_id": "coin_results", "source_text": original}, {"op": "move_cards_from_discard_to_hand_per_coin_heads", "player": "self", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "coin_results_ref": "coin_results", "source_text": original}], []

    # Attach basic Energy variants.
    m = re.fullmatch(r"Attach a basic Energy card from your hand to 1 of your Benched Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_basic_energy", "zone": "self.hand", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(1), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_benched_pokemon", "zone": "self.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_basic_energy", "target_ref": "chosen_benched_pokemon", "source_text": original}], []

    m = re.fullmatch(r"Attach a basic ([A-Za-z]+) Energy card from your discard pile to 1 of your Benched ([A-Za-z]+) Pokémon\.?,?", t, re.I)
    if m:
        etype, ptype = m.group(1).capitalize(), m.group(2).capitalize()
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_basic_energy", "zone": "self.discard", "filter": {"supertype": "Energy", "subtypes": ["Basic"], "types": [etype]}, "amount": amount_exact(1), "source_text": original}, {"op": "choose_target", "player": "self", "target_id": "chosen_benched_pokemon", "zone": "self.bench", "filter": {"supertype": "Pokémon", "types": [ptype]}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_basic_energy", "target_ref": "chosen_benched_pokemon", "source_text": original}], []

    # Tool / attachment lifecycle and modifiers.
    if re.fullmatch(r"Attach this Pokemon Tool to 1 of your opponent's Pokemon-EX that doesn't already have a Pokemon Tool attached to it\.?,?", t, re.I):
        return [{"op": "attach_card", "player": "self", "from_zone": "hand", "target": "opponent.pokemon_ex_without_tool", "subtype": "Tool", "source_text": original}], []

    if re.fullmatch(r"When this card is removed from a Pokémon for any reason, put this card in its owner's discard pile\.?,?", t, re.I):
        return [{"op": "register_trigger", "trigger": "this_card_removed_from_pokemon", "effect": {"op": "move_card", "target": "this_card", "destination": "owner.discard", "source_text": original}, "source_text": original}], []

    m = re.fullmatch(r"This card can only be attached to a (Rapid Strike|Single Strike) Pokémon\. If this card is attached to anything other than a \1 Pokémon, discard this card\.?,?", t, re.I)
    if m:
        return [{"op": "register_attachment_rule", "allowed_target_filter": {"supertype": "Pokémon", "tags": [m.group(1)]}, "if_illegal_attachment": {"op": "discard_card", "target": "this_card"}, "source_text": original}], []

    m = re.fullmatch(r"The attacks of the Pokémon this card is attached to do (\d+) more damage to your opponent's Active Pokémon(?: (?:V|ex))? \(before applying Weakness and Resistance\)\.?,?", t, re.I)
    if m:
        target_filter = {"target": "opponent.active"}
        if " Pokémon V" in original:
            target_filter["opponent.active.tags_include"] = ["Pokémon V"]
        if " Pokémon ex" in original:
            target_filter["opponent.active.tags_include"] = ["Pokémon ex"]
        return [{"op": "register_continuous_modifier", "modifier_id": "attached_pokemon_attack_damage_bonus", "target": "attached_pokemon", "duration": {"while_attached": True}, "modification": {"attack_damage_delta": int(m.group(1)), "timing_basis": "before_weakness_resistance", **target_filter}, "source_text": original}], []

    if re.fullmatch(r"The Pokémon this card is attached to has no Weakness\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "attached_pokemon_no_weakness", "target": "attached_pokemon", "duration": {"while_attached": True}, "modification": {"weakness": "none"}, "source_text": original}], []

    m = re.fullmatch(r"The Basic Pokémon this card is attached to gets \+(\d+) HP\.?,?", t, re.I)
    if m:
        return [{"op": "register_continuous_modifier", "modifier_id": "attached_basic_pokemon_hp_bonus", "target": "attached_pokemon", "duration": {"while_attached": True}, "condition": {"attached_pokemon.subtypes_include": ["Basic"]}, "modification": {"hp_delta": int(m.group(1))}, "source_text": original}], []

    # Energy-providing / modifier rules.
    if re.fullmatch(r"All Special Energy attached to Pokémon \(both yours and your opponent's\) provide Colorless Energy and have no other effect\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "all_special_energy_colorless_no_other_effect", "target": "all.attached_special_energy", "duration": {"while_in_play": True}, "modification": {"provides_energy": ["Colorless"], "removes_other_effects": True}, "source_text": original}], []

    if re.fullmatch(r"Holon Energy (?:GL|WP) provides Colorless Energy\.?,?", t, re.I):
        return [{"op": "provide_energy", "types": ["Colorless"], "amount": 1, "while_attached": True, "source_text": original}], []

    # Global board rule modifiers / ability locks.
    if re.fullmatch(r"Your opponent can't play any (Item|Trainer|Supporter) cards? from (?:their|his or her) hand during (?:their|his or her) next turn\.?,?", t, re.I):
        kind = re.search(r"any (Item|Trainer|Supporter) cards?", t, re.I).group(1).capitalize()
        return [delayed_modifier_step(f"opponent_cannot_play_{kind.lower()}_cards_next_turn", "opponent", {"cannot_play_cards": {"kind": kind}}, original)], []

    if re.fullmatch(r"No Trainer cards can be played\. This power stops working while .* is Asleep, Confused, or Paralyzed\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "no_trainer_cards_can_be_played", "target": "each_player", "duration": {"while_source_in_play_and_not_affected_by_special_condition": True}, "modification": {"cannot_play_cards": {"supertype": "Trainer"}}, "source_text": original}], []

    if re.fullmatch(r"(?:Pokémon with a Rule Box in play \(both yours and your opponent's\)|Each Basic Pokémon in play, in each player's hand, and in each player's discard pile|Colorless Pokémon in play \(both yours and your opponent's\)) have no Abilities\.?.*", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "ability_lock", "target": "matching_pokemon", "duration": {"while_in_play": True}, "modification": {"abilities_disabled": True}, "source_text": original}], []

    m = re.fullmatch(r"Each (Grass|Lightning|Colorless|Fire|Water|Darkness|Metal|Psychic|Fighting) and (Grass|Lightning|Colorless|Fire|Water|Darkness|Metal|Psychic|Fighting) Pokémon in play \(both yours and your opponent's\) gets \+(\d+) HP\.?,?", t, re.I)
    if m:
        return [{"op": "register_continuous_modifier", "modifier_id": "global_type_hp_bonus", "target": "all.in_play.pokemon", "duration": {"while_in_play": True}, "condition": {"types_any": [m.group(1).capitalize(), m.group(2).capitalize()]}, "modification": {"hp_delta": int(m.group(3))}, "source_text": original}], []

    if re.fullmatch(r"Apply Weakness for each Pokémon \(both yours and your opponent's\) as ×2 instead\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "all_weakness_x2", "target": "all.in_play.pokemon", "duration": {"while_in_play": True}, "modification": {"weakness_multiplier": 2}, "source_text": original}], []

    if re.fullmatch(r"Each Pokémon in play has no Resistance\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "all_pokemon_no_resistance", "target": "all.in_play.pokemon", "duration": {"while_in_play": True}, "modification": {"resistance": "none"}, "source_text": original}], []

    if re.fullmatch(r"Pokémon \(both yours and your opponent's\) can't be healed\.?,?", t, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "pokemon_cannot_be_healed", "target": "all.in_play.pokemon", "duration": {"while_in_play": True}, "modification": {"cannot_be_healed": True}, "source_text": original}], []

    # Healing / Special Condition removal composites.
    m = re.fullmatch(r"Heal (\d+) damage and remove a Special Condition from your Active Pokémon\.?,?", t, re.I)
    if m:
        return [{"op": "heal_damage", "target": "self.active", "amount": amount_exact(int(m.group(1))), "source_text": original}, {"op": "remove_special_conditions", "target": "self.active", "amount": amount_up_to(1), "source_text": original}], []

    m = re.fullmatch(r"Remove 1 damage counter from each of your Pokémon that has any damage counters on it\.?,?", t, re.I)
    if m:
        return [{"op": "heal_damage", "target": "self.in_play.each_with_damage_counters", "amount": {"mode": "damage_counters", "value": 1}, "source_text": original}], []

    # Bench from discard/deck with optional draw.
    m = re.fullmatch(r"Put a (Rapid Strike|Single Strike|Fusion Strike)? ?Pokémon from your discard pile onto your Bench\. If you do, draw (\d+) cards\.?,?", t, re.I)
    if m:
        filt = {"supertype": "Pokémon"}
        if m.group(1):
            filt["tags"] = [m.group(1)]
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_pokemon_from_discard", "zone": "self.discard", "filter": filt, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_pokemon_from_discard", "destination": "self.bench", "source_text": original}, {"op": "draw_cards", "player": "self", "amount": int(m.group(2)), "condition": {"if_moved_card": "chosen_pokemon_from_discard"}, "source_text": original}], []

    if re.fullmatch(r"Put a Basic Pokémon from your opponent's discard pile onto (?:his or her|their) Bench\.?,?", t, re.I):
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_basic_pokemon_from_opponent_discard", "zone": "opponent.discard", "filter": {"supertype": "Pokémon", "subtypes": ["Basic"]}, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_basic_pokemon_from_opponent_discard", "destination": "opponent.bench", "source_text": original}], []

    # Fossil / Tool attack grant / VSTAR grant reference rules.
    if re.fullmatch(r"Play Root Fossil as if it were a Basic Pokémon\..*At any time during your turn before your attack, you may discard Root Fossil from play\.?,?", t, re.I):
        return [{"op": "register_fossil_as_basic_pokemon_rule", "name": "Root Fossil", "types": ["Colorless"], "can_retreat": False, "affected_by_special_conditions": False, "knockout_counts_as_knocked_out_pokemon": False, "self_discard_allowed_before_attack": True, "source_text": original}], []

    if re.fullmatch(r"The Pokémon V this card is attached to can use the VSTAR Power on this card\.?,?", t, re.I):
        return [{"op": "grant_attached_vstar_power", "target": "attached_pokemon", "condition": {"attached_pokemon.tags_include": ["Pokémon V"]}, "source_text": original}], []

    if re.fullmatch(r"(?:The Genesect-EX|The Pokémon) this card is attached to can (?:also )?use (?:the attack on this card|any attack from its previous Evolutions)\. \(You still need the necessary Energy to use (?:this attack|each attack)\.\)\.?,?", t, re.I):
        return [{"op": "grant_attack_access", "target": "attached_pokemon", "source": "this_card_or_previous_evolutions", "requires_energy_cost": True, "source_text": original}], []


    # --- v0.18 massive safe-coverage bundle ---
    # Strategy shift: after v0.17 the remaining rows are many small repeated
    # patterns.  Prefer broad, explicit families that preserve the text and emit
    # structured operations, instead of one micro-family per patch.
    t_rule = re.sub(r"^Rules:\s*", "", t, flags=re.I).strip()

    # Coin: attack fails on tails, prevention on heads.
    if re.fullmatch(r"Flip a coin\. If tails, this attack does nothing\. If heads, during your opponent's next turn, prevent all damage from and effects of attacks done to this Pokémon\.?,?", t_rule, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "tails", "then": [{"op": "attack_does_nothing", "source_text": original}], "else": [delayed_modifier_step("prevent_attack_damage_and_effects_to_self_next_turn", "self_attacking_pokemon", {"prevent_damage_from_attacks": True, "prevent_effects_of_attacks": True}, original)], "source_text": original}], []

    # Coin: attack cannot be used next turn on tails.
    if re.fullmatch(r"Flip a coin\. If tails, this Pokémon can't attack during your next turn\.?,?", t_rule, re.I):
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "tails", "then": [{"op": "register_attack_lock", "target": "self_attacking_pokemon", "duration": {"until": "end_of_self_next_turn"}, "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"You can't use this attack during your next turn\.?,?", t_rule, re.I):
        return [{"op": "register_attack_lock", "target": "self_attacking_pokemon", "duration": {"until": "end_of_self_next_turn"}, "source_text": original}], []

    # Composite coin damage + Special Condition / spread damage variants.
    m = re.fullmatch(r"Flip (\d+) coins\. This attack does (\d+) damage times the number of heads\. If either of the coins is heads, (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned)\.?,?", t_rule, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": int(m.group(1)), "target_id": "coin_results", "source_text": original}, {"op": "set_attack_damage_from_coin_heads", "damage_per_heads": int(m.group(2)), "coin_results_ref": "coin_results", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_results", "if": "any_heads", "then": [status_condition_step(m.group(3), "opponent.active", original)], "source_text": original}], []

    m = re.fullmatch(r"Flip (\d+) coins\. This attack does (\d+) damage times the number of heads\. If you get (\d+) or more heads, .* is now (Asleep|Confused|Paralyzed|Poisoned|Burned) \(after doing damage\)\.?,?", t_rule, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": int(m.group(1)), "target_id": "coin_results", "source_text": original}, {"op": "set_attack_damage_from_coin_heads", "damage_per_heads": int(m.group(2)), "coin_results_ref": "coin_results", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_results", "if": {"heads_at_least": int(m.group(3))}, "then": [status_condition_step(m.group(4), "self_attacking_pokemon", original)], "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, (?:the Defending Pokémon|your opponent's Active Pokémon) is now (Asleep|Confused|Paralyzed|Poisoned|Burned) and this attack does (\d+) damage to each of your opponent's Benched Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t_rule, re.I)
    if m:
        then_steps = [status_condition_step(m.group(1), "opponent.active", original), {"op": "deal_damage", "target": "opponent.bench.each", "amount": int(m.group(2)), "apply_weakness_resistance": False, "source_text": original}]
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": then_steps, "source_text": original}], []

    # Future-turn damage bonus / mark effects.
    m = re.fullmatch(r"(?:During your next turn|Until the end of your next turn), if an attack damages the Defending Pokémon \(after applying Weakness and Resistance\), that attack does (\d+) more damage to the Defending Pokémon\.?,?", t_rule, re.I)
    if m:
        return [{"op": "register_delayed_modifier", "modifier_id": "future_damage_bonus_to_defending_pokemon", "target": "opponent.active", "duration": {"until": "end_of_self_next_turn"}, "modification": {"damage_taken_from_attacks_delta": int(m.group(1)), "timing_basis": "after_weakness_resistance"}, "source_text": original}], []

    m = re.fullmatch(r"During your next turn, attacks used by this Pokémon do (\d+) more damage to your opponent's Active Pokémon \(before applying Weakness and Resistance\)\.?,?", t_rule, re.I)
    if m:
        return [{"op": "register_delayed_modifier", "modifier_id": "self_next_turn_attack_damage_bonus", "target": "self_attacking_pokemon", "duration": {"until": "end_of_self_next_turn"}, "modification": {"attack_damage_delta": int(m.group(1)), "timing_basis": "before_weakness_resistance"}, "source_text": original}], []

    # Conditional damage against Pokémon-ex/evolved/damaged/statused targets.
    m = re.fullmatch(r"If the Defending Pokémon is Pokémon-ex, this attack does (\d+) damage plus (\d+) more damage\.?,?", t_rule, re.I)
    if m:
        return [{"op": "set_attack_damage", "amount": int(m.group(1)), "source_text": original}, {"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(2)), "condition": {"opponent.active.tags_include": ["Pokémon-ex"]}, "source_text": original}], []

    m = re.fullmatch(r"If the Defending Pokémon is an? (?:Stage 2 )?Evolved Pokémon, this attack does (\d+) damage plus (\d+) more damage\.?,?", t_rule, re.I)
    if m:
        return [{"op": "set_attack_damage", "amount": int(m.group(1)), "source_text": original}, {"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(2)), "condition": {"opponent.active.is_evolved": True}, "source_text": original}], []

    m = re.fullmatch(r"If the Defending Pokémon already has (?:any|at least (\d+)) damage counters? on it, this attack does (\d+) damage plus (\d+) more damage\.?,?", t_rule, re.I)
    if m:
        return [{"op": "set_attack_damage", "amount": int(m.group(2)), "source_text": original}, {"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(3)), "condition": {"opponent.active.damage_counters_at_least": int(m.group(1) or 1)}, "source_text": original}], []

    m = re.fullmatch(r"If the Defending Pokémon is affected by a Special Condition, this attack does (\d+) damage plus (\d+) more damage\.?,?", t_rule, re.I)
    if m:
        return [{"op": "set_attack_damage", "amount": int(m.group(1)), "source_text": original}, {"op": "modify_attack_damage", "mode": "add", "amount": int(m.group(2)), "condition": {"opponent.active.has_special_condition": True}, "source_text": original}], []

    # Damage based on counts.
    m = re.fullmatch(r"Does (\d+) damage plus (\d+) more damage for each ([A-Za-z]+) Energy attached to .* but not used to pay for this attack's Energy cost\. You can't add more than (\d+) damage in this way\.?,?", t_rule, re.I)
    if m:
        return [{"op": "modify_attack_damage_from_count", "mode": "add", "amount_per": int(m.group(2)), "count": {"attached_energy_type": m.group(3).capitalize(), "not_used_to_pay_attack_cost": True, "target": "self_attacking_pokemon"}, "cap": int(m.group(4)), "source_text": original}], []

    m = re.fullmatch(r"Does (\d+) damage plus (\d+) more damage for each Energy attached to all of your opponent's Pokémon\.?,?", t_rule, re.I)
    if m:
        return [{"op": "set_attack_damage", "amount": int(m.group(1)), "source_text": original}, {"op": "modify_attack_damage_from_count", "mode": "add", "amount_per": int(m.group(2)), "count": {"zone": "opponent.in_play", "attached_energy": True}, "source_text": original}], []

    m = re.fullmatch(r"This attack does (\d+) damage to 1 of your opponent's Pokémon that has any damage counters on it\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t_rule, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_pokemon_with_damage", "zone": "opponent.in_play", "filter": {"has_damage_counters": True}, "amount": amount_exact(1), "source_text": original}, {"op": "deal_damage", "target_ref": "chosen_opponent_pokemon_with_damage", "amount": int(m.group(1)), "apply_weakness_resistance_to_bench": False, "source_text": original}], []

    m = re.fullmatch(r"(?:Choose 1 of your opponent's Pokémon\. )?This attack does (\d+) damage to that Pokémon\. Don't apply Weakness and Resistance for this attack\. \(Any other effects that would happen after applying Weakness and Resistance still happen\.\)\.?,?", t_rule, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_opponent_pokemon", "zone": "opponent.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "deal_damage", "target_ref": "chosen_opponent_pokemon", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    m = re.fullmatch(r"(?:This attack also does|Does) (\d+) damage to 1 of your Benched Pokémon\. \(Don't apply Weakness and Resistance for Benched Pokémon\.\)\.?,?", t_rule, re.I)
    if m:
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_self_benched_pokemon", "zone": "self.bench", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "deal_damage", "target_ref": "chosen_self_benched_pokemon", "amount": int(m.group(1)), "apply_weakness_resistance": False, "source_text": original}], []

    # Energy movement / discard / recovery variants.
    m = re.fullmatch(r"Move a basic Energy(?: card)? attached to 1 of your Pokémon to another of your Pokémon\.?,?", t_rule, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_attached_basic_energy", "zone": "self.in_play.attached", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_attached_basic_energy", "destination": "self.in_play.other_pokemon.attached", "source_text": original}], []

    m = re.fullmatch(r"Move a ([A-Za-z]+) Energy card attached to 1 of your Pokémon to another of your Pokémon\.?,?", t_rule, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_attached_typed_energy", "zone": "self.in_play.attached", "filter": {"supertype": "Energy", "types": [m.group(1).capitalize()]}, "amount": amount_exact(1), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_attached_typed_energy", "destination": "self.in_play.other_pokemon.attached", "source_text": original}], []

    m = re.fullmatch(r"(?:Put|Shuffle) (\d+) (?:in any combination of )?(Pokémon and basic Energy cards|basic Energy cards|[A-Za-z]+ Energy cards) from your discard pile (?:into your hand|into your deck)\.?,?", t_rule, re.I)
    if m:
        dest = "self.hand" if "hand" in t_rule.lower() else "self.deck"
        filt = {"from_text": m.group(2)}
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_discard_cards", "zone": "self.discard", "filter": filt, "amount": amount_up_to(int(m.group(1))), "source_text": original}, {"op": "move_card", "cards_ref": "chosen_discard_cards", "destination": dest, "source_text": original}], []

    m = re.fullmatch(r"Flip (\d+) coins\. (?:For each heads, )?(?:put|Put) (?:a number of cards up to the number of heads|a Basic Energy card) from your discard pile into your hand\.?.*", t_rule, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": int(m.group(1)), "target_id": "coin_results", "source_text": original}, {"op": "choose_cards", "player": "self", "target_id": "chosen_discard_cards", "zone": "self.discard", "filter": {"from_text": "cards or Basic Energy"}, "amount": {"mode": "up_to_coin_heads", "coin_results_ref": "coin_results"}, "source_text": original}, {"op": "move_card", "cards_ref": "chosen_discard_cards", "destination": "self.hand", "source_text": original}], []

    m = re.fullmatch(r"(?:Once during your turn \(before your attack\), you may |As often as you like during your turn \(before your attack\), you may )?attach (?:1 |an? |up to (\d+) )?([A-Za-z]+|basic)? ?Energy card(?:s)? from your discard pile to (?:1 of your Benched Pokémon|1 of your [A-Za-z]+ Pokémon|this Pokémon|your Benched Pokémon in any way you like)\.?.*", t_rule, re.I)
    if m:
        n = int(m.group(1) or 1)
        etype = (m.group(2) or "").capitalize()
        filt = {"supertype": "Energy"}
        if etype and etype.lower() != "basic":
            filt["types"] = [etype]
        elif etype.lower() == "basic":
            filt["subtypes"] = ["Basic"]
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_energy_from_discard", "zone": "self.discard", "filter": filt, "amount": amount_up_to(n), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_energy_from_discard", "target": "self.in_play.pokemon", "source_text": original}], []

    m = re.fullmatch(r"Attach a basic Energy card from your hand to 1 of your Benched Pokémon\.?,?", t_rule, re.I)
    if m:
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_basic_energy_from_hand", "zone": "self.hand", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_exact(1), "source_text": original}, {"op": "attach_card", "cards_ref": "chosen_basic_energy_from_hand", "target": "self.bench.pokemon", "source_text": original}], []

    m = re.fullmatch(r"Discard (\d+) ([A-Za-z]+)? ?Energy attached to this Pokémon\.?,?", t_rule, re.I)
    if m:
        filt = {"supertype": "Energy"}
        if m.group(2):
            filt["types"] = [m.group(2).capitalize()]
        return [{"op": "discard_attached_energy", "target": "self_attacking_pokemon", "filter": filt, "amount": amount_exact(int(m.group(1))), "source_text": original}], []

    # Search / look / bench / evolve variants.
    m = re.fullmatch(r"Search your deck for up to (\d+) basic Energy cards, reveal them, and put them into your hand\. Shuffle your deck afterward\.?,?", t_rule, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_basic_energy", "filter": {"supertype": "Energy", "subtypes": ["Basic"]}, "amount": amount_up_to(int(m.group(1))), "reveal": True, "destination": "self.hand", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for any number of Basic Pokémon and put them onto your Bench\. Then, shuffle your deck\.?,?", t_rule, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_basic_pokemon", "filter": {"supertype": "Pokémon", "subtypes": ["Basic"]}, "amount": {"mode": "any_number"}, "destination": "self.bench", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    m = re.fullmatch(r"Search your deck for (?:Omanyte, Kabuto, or any Basic Pokémon|up to 2 Pokémon Tool cards|a card that evolves from 1 of your Pokémon) .*Shuffle your deck afterward\.?,?", t_rule, re.I)
    if m:
        return [{"op": "search_deck", "player": "self", "target_id": "searched_cards", "filter": {"from_text": original}, "amount": {"mode": "from_text"}, "destination": "from_text", "source_text": original}, {"op": "shuffle_deck", "player": "self", "source_text": original}], []

    # Bounce / return to hand variants.
    if re.fullmatch(r"Put 1 of your Pokémon and all attached cards into your hand\.?,?", t_rule, re.I):
        return [{"op": "choose_target", "player": "self", "target_id": "chosen_self_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_zone_to_zone", "target_ref": "chosen_self_pokemon_and_attached_cards", "destination": "self.hand", "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, return 1 of your Pokémon and all cards attached to it to your hand\.?,?", t_rule, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "choose_target", "player": "self", "target_id": "chosen_self_pokemon", "zone": "self.in_play", "filter": {"supertype": "Pokémon"}, "amount": amount_exact(1), "source_text": original}, {"op": "move_zone_to_zone", "target_ref": "chosen_self_pokemon_and_attached_cards", "destination": "self.hand", "source_text": original}], "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If heads, your opponent returns the Defending Pokémon and all cards attached to it to (?:his or her|their) hand\.?.*", t_rule, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "heads", "then": [{"op": "move_zone_to_zone", "target": "opponent.active_and_attached_cards", "destination": "opponent.hand", "source_text": original}], "source_text": original}], []

    # Tool / Stadium / item rules and removal.
    if re.fullmatch(r"Discard all Pokémon Tool cards attached to each of your opponent's Pokémon\.?,?", t_rule, re.I):
        return [{"op": "discard_cards", "zone": "opponent.in_play.attached", "filter": {"subtypes": ["Tool"]}, "amount": {"mode": "all"}, "source_text": original}], []

    if re.fullmatch(r"Choose up to (\d+) Pokémon Tools attached to Pokémon \(yours or your opponent's\) and discard them\.?,?", t_rule, re.I):
        n = int(re.search(r"up to (\d+)", t_rule, re.I).group(1))
        return [{"op": "choose_cards", "player": "self", "target_id": "chosen_tools", "zone": "all.in_play.attached", "filter": {"subtypes": ["Tool"]}, "amount": amount_up_to(n), "source_text": original}, {"op": "discard_cards", "cards_ref": "chosen_tools", "source_text": original}], []

    if re.fullmatch(r"If your opponent has a Stadium in play, discard it\.?,?", t_rule, re.I):
        return [{"op": "discard_stadium", "player": "opponent", "condition": {"opponent.stadium_in_play": True}, "source_text": original}], []

    if re.fullmatch(r"Attach PlusPower to 1 of your Pokémon\. Discard this card at the end of your turn\.?,?", t_rule, re.I):
        return [{"op": "attach_card", "card": "this_card", "target": "self.pokemon", "source_text": original}, {"op": "register_delayed_effect", "trigger": "end_of_self_turn", "then": [{"op": "discard_card", "target": "this_card", "source_text": original}], "source_text": original}], []

    if re.fullmatch(r"If this card is discarded from play, put it into your hand instead of the discard pile\.?,?", t_rule, re.I):
        return [{"op": "register_replacement_effect", "event": "this_card_discarded_from_play", "then": [{"op": "move_card", "target": "this_card", "destination": "owner.hand", "source_text": original}], "source_text": original}], []

    # Global/prize/evolution rules.
    if re.fullmatch(r"You can play this card only if you have more Prize cards remaining than your opponent\.?,?", t_rule, re.I):
        return [{"op": "play_condition", "condition": {"self.prize_cards_remaining_gt_opponent": True}, "source_text": original}], []

    if re.fullmatch(r"Turn all of your Prize cards face up\. \(Those Prize cards remain face up for the rest of the game\.\)\.?,?", t_rule, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "self_prizes_face_up_rest_of_game", "target": "self", "duration": {"until": "end_of_game"}, "modification": {"prize_cards_face_up": True}, "source_text": original}], []

    if re.fullmatch(r"Each player's Grass Pokémon can evolve into Grass Pokémon during the turn they play those Pokémon, except during their first turn\.?,?", t_rule, re.I):
        return [{"op": "register_continuous_modifier", "modifier_id": "grass_pokemon_can_evolve_first_turn_in_play", "target": "each_player.pokemon", "duration": {"while_in_play": True}, "condition": {"type": "Grass", "evolution_type": "Grass", "not_first_turn_of_player": True}, "modification": {"can_evolve_turn_played": True}, "source_text": original}], []

    # Self damage / recoil.
    m = re.fullmatch(r"(?:[A-Za-z' -]+ )?does (\d+) damage to itself\.?,?", t_rule, re.I)
    if m:
        return [{"op": "deal_damage", "target": "self_attacking_pokemon", "amount": int(m.group(1)), "source_text": original}], []

    m = re.fullmatch(r"Flip a coin\. If tails, .* does (\d+) damage to itself\.?,?", t_rule, re.I)
    if m:
        return [{"op": "coin_flip", "player": "self", "count": 1, "target_id": "coin_result", "source_text": original}, {"op": "branch_on_result", "result_ref": "coin_result", "if": "tails", "then": [{"op": "deal_damage", "target": "self_attacking_pokemon", "amount": int(m.group(1)), "source_text": original}], "source_text": original}], []



    # v0.21 template-driven fallback: broad, parameterized templates that
    # still emit concrete structured steps. This preserves strict completeness
    # while reducing one-off regex patching.
    template_result = compile_template_driven_text(t_rule, original, source_section)
    if template_result is not None:
        return template_result

    # No safe match.
    return [], [original]


def make_effect(effect_id: str, kind: str, source: dict[str, Any], timing: dict[str, Any], steps: list[dict[str, Any]], unparsed: list[str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    effect = {
        "effect_id": effect_id,
        "source": source,
        "kind": kind,
        "timing": timing,
        "playability": {},
        "costs": [],
        "choices": [],
        "steps": steps,
        "duration": None,
        "usage_limit": None,
        "parser": {
            "status": "complete" if not unparsed else "partial",
            "matched_by": "compile_cards_auto.py",
            "unparsed_text": unparsed,
        },
    }
    if extra:
        effect.update(extra)
    return effect


def compile_rules(row: pd.Series, raw: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    effects: list[dict[str, Any]] = []
    unparsed_all: list[str] = []

    rules = raw.get("rules")
    if not isinstance(rules, list):
        rules = maybe_json_loads(row.get("raw_rules_json")) or split_pipe_list(row.get("rules"))

    for idx, rule in enumerate(rules):
        text = normalize_space(rule)
        if not text:
            continue

        steps, unparsed = compile_simple_text(text, "rules")
        unparsed_all.extend(unparsed)

        if steps:
            effects.append(make_effect(
                effect_id=f"{row.get('card_id')}::rule::{idx+1}",
                kind="trainer_rule" if row.get("supertype") == "Trainer" else "rule",
                source={
                    "card_id": clean_value(row.get("card_id")),
                    "section": "rules",
                    "index": idx,
                    "text": text,
                },
                timing={"windows": ["main_step"], "owner_turn_required": True},
                steps=steps,
                unparsed=unparsed,
            ))

    return effects, unparsed_all


def compile_attacks(row: pd.Series, raw: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    effects: list[dict[str, Any]] = []
    unparsed_all: list[str] = []

    attacks = raw.get("attacks")
    if not isinstance(attacks, list):
        attacks = maybe_json_loads(row.get("raw_attacks_json")) or []

    for idx, attack in enumerate(attacks):
        if not isinstance(attack, dict):
            continue

        text = normalize_space(attack.get("text") or "")
        text_steps: list[dict[str, Any]] = []
        text_unparsed: list[str] = []

        if text:
            text_steps, text_unparsed = compile_simple_text(text, "attack_text")

        base_steps = [{
            "op": "declare_attack",
            "attack_name": clean_value(attack.get("name")),
            "energy_cost": attack.get("cost") or [],
            "converted_energy_cost": attack.get("convertedEnergyCost"),
            "printed_damage": clean_value(attack.get("damage")),
            "damage": parse_damage_value(attack.get("damage")),
            "source_text": text,
        }]

        if attack.get("damage"):
            base_steps.append({
                "op": "deal_attack_damage",
                "target": "opponent.active",
                "amount": parse_damage_value(attack.get("damage")),
                "source_text": text or f"Damage: {attack.get('damage')}",
            })

        steps = base_steps + text_steps
        unparsed_all.extend(text_unparsed)

        effects.append(make_effect(
            effect_id=f"{row.get('card_id')}::attack::{idx+1}",
            kind="attack",
            source={
                "card_id": clean_value(row.get("card_id")),
                "section": "attacks",
                "index": idx,
                "name": clean_value(attack.get("name")),
                "text": text,
            },
            timing={"windows": ["attack_step"], "owner_turn_required": True},
            steps=steps,
            unparsed=text_unparsed,
            extra={
                "costs": [{
                    "type": "energy",
                    "cost": attack.get("cost") or [],
                    "converted_energy_cost": attack.get("convertedEnergyCost"),
                }],
            },
        ))

    return effects, unparsed_all


def compile_abilities(row: pd.Series, raw: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    effects: list[dict[str, Any]] = []
    unparsed_all: list[str] = []

    abilities = raw.get("abilities")
    if not isinstance(abilities, list):
        abilities = maybe_json_loads(row.get("raw_abilities_json")) or []

    for idx, ability in enumerate(abilities):
        if not isinstance(ability, dict):
            continue

        text = normalize_space(ability.get("text") or "")
        steps, unparsed = compile_simple_text(text, "ability") if text else ([], [])
        unparsed_all.extend(unparsed)

        kind = "ability"
        if re.search(r"once during your turn|you may", text, re.I):
            kind = "ability_activated"
        elif re.search(r"whenever|when |if .* is", text, re.I):
            kind = "ability_triggered_or_static"

        effects.append(make_effect(
            effect_id=f"{row.get('card_id')}::ability::{idx+1}",
            kind=kind,
            source={
                "card_id": clean_value(row.get("card_id")),
                "section": "abilities",
                "index": idx,
                "name": clean_value(ability.get("name")),
                "ability_type": clean_value(ability.get("type")),
                "text": text,
            },
            timing={"windows": ["main_step"], "owner_turn_required": True},
            steps=steps,
            unparsed=unparsed if text else [],
            extra={
                "usage_limit": {
                    "type": "text_defined",
                    "raw_text": text,
                } if text else None,
            },
        ))

    return effects, unparsed_all


def compile_energy(row: pd.Series, raw: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    # Basic Energy cards often have no rules. Their energy provision is a game identity property.
    effects: list[dict[str, Any]] = []
    unparsed_all: list[str] = []

    supertype = clean_value(row.get("supertype"))
    if supertype != "Energy":
        return effects, unparsed_all

    subtypes = split_pipe_list(row.get("subtypes"))
    types = split_pipe_list(row.get("types"))

    if "Basic" in subtypes and types:
        effects.append(make_effect(
            effect_id=f"{row.get('card_id')}::energy::provide",
            kind="energy_provision",
            source={
                "card_id": clean_value(row.get("card_id")),
                "section": "identity",
                "text": "Basic Energy identity",
            },
            timing={"windows": ["while_attached"]},
            steps=[{
                "op": "provide_energy",
                "types": types,
                "amount": 1,
                "source_text": "Basic Energy identity",
            }],
            unparsed=[],
        ))

    rules_effects, rules_unparsed = compile_rules(row, raw)
    effects.extend(rules_effects)
    unparsed_all.extend(rules_unparsed)

    return effects, unparsed_all


def normalized_card_definition(group: dict[str, Any], compiler_version: str) -> dict[str, Any]:
    row = group["representative"]
    raw = parse_raw_card(row)

    printed = {
        "set": {
            "id": clean_value(row.get("set_id")),
            "name": clean_value(row.get("set_name")),
            "series": clean_value(row.get("set_series")),
            "release_date": clean_value(row.get("set_release_date")),
        },
        "number": clean_value(row.get("number")),
        "rarity": clean_value(row.get("rarity")),
        "regulation_mark": clean_value(row.get("regulation_mark")),
        "legalities": {
            "standard": clean_value(row.get("legal_standard")),
            "expanded": clean_value(row.get("legal_expanded")),
            "unlimited": clean_value(row.get("legal_unlimited")),
        },
        "images": {
            "small": clean_value(row.get("image_small")),
            "large": clean_value(row.get("image_large")),
        },
    }

    gameplay = {
        "hp": clean_value(row.get("hp")) or clean_value(raw.get("hp")),
        "stage": None,
        "evolves_from": clean_value(row.get("evolves_from")) or clean_value(raw.get("evolvesFrom")),
        "weaknesses": raw.get("weaknesses") or [],
        "resistances": raw.get("resistances") or [],
        "retreat_cost": raw.get("retreatCost") or [],
        "converted_retreat_cost": raw.get("convertedRetreatCost"),
        "special_rule_tags": [],
    }

    subtypes = split_pipe_list(row.get("subtypes"))
    for subtype in subtypes:
        if subtype in ["Basic", "Stage 1", "Stage 2", "Restored"]:
            gameplay["stage"] = subtype
        if subtype in ["ex", "V", "VMAX", "VSTAR", "GX", "MEGA", "Ancient", "Future", "Radiant", "Prism Star"]:
            gameplay["special_rule_tags"].append(subtype)

    effects: list[dict[str, Any]] = []
    unparsed: list[str] = []

    if clean_value(row.get("supertype")) == "Energy":
        e, u = compile_energy(row, raw)
        effects.extend(e)
        unparsed.extend(u)
    else:
        e, u = compile_rules(row, raw)
        effects.extend(e)
        unparsed.extend(u)

    e, u = compile_abilities(row, raw)
    effects.extend(e)
    unparsed.extend(u)

    e, u = compile_attacks(row, raw)
    effects.extend(e)
    unparsed.extend(u)

    # If a card had text but generated no effects, mark the text for review.
    has_text = any(clean_value(row.get(col)) for col in ["rules", "abilities_text", "attacks_text", "combined_text"])
    if has_text and not effects:
        combined = clean_value(row.get("combined_text"))
        if combined:
            unparsed.append(str(combined))

    status = "complete" if not unparsed else "partial"

    return {
        "schema_version": "pokemon-card-definition/v1",
        "effect_group_id": group["effect_group_id"],
        "representative_card_id": clean_value(row.get("card_id")),
        "same_effect_card_ids": group["same_effect_card_ids"],
        "same_effect_printing_count": len(group["same_effect_card_ids"]),
        "same_effect_printings": group["same_effect_printings"],
        "identity": {
            "name": clean_value(row.get("name")),
            "canonical_name": clean_value(row.get("name")),
            "supertype": clean_value(row.get("supertype")),
            "subtypes": subtypes,
            "types": split_pipe_list(row.get("types")),
            "tags": [],
        },
        "printed": printed,
        "gameplay": gameplay,
        "sources": {
            "rules": maybe_json_loads(row.get("raw_rules_json")) or split_pipe_list(row.get("rules")),
            "abilities": maybe_json_loads(row.get("raw_abilities_json")),
            "attacks": maybe_json_loads(row.get("raw_attacks_json")),
            "combined_text": clean_value(row.get("combined_text")),
            "raw_card": raw,
        },
        "compiled_effects": effects,
        "compiler_metadata": {
            "compiler_version": compiler_version,
            "compiled_at": datetime.now(timezone.utc).isoformat(),
            "source": "compile_cards_auto.py",
            "dedupe_policy": "safe_signature_v1",
            "matched_effect_count": len(effects),
            "unparsed_text_count": len(unparsed),
        },
        "parser": {
            "status": status,
            "confidence": 0.95 if status == "complete" else 0.45,
            "unparsed_text": sorted(set(unparsed)),
            "notes": [] if status == "complete" else ["Auto-compiler could not safely parse all text. Send this card/effect group to review queue or add a pattern."],
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-compile Pokémon TCG cards from all_cards.csv into simulator-ready JSON plus review queues.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--only-with-text", action="store_true", help="Deprecated compatibility flag; text-only filtering is now the default.")
    parser.add_argument("--include-no-text", action="store_true", help="Include cards/effect groups with no rules, abilities, attacks, or combined text.")
    parser.add_argument("--standard-only", action="store_true")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--compiler-version", default="0.17.0")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    original_rows = len(df)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print("Warning: missing expected columns:")
        for col in missing:
            print(f"  - {col}")

    if args.standard_only and "legal_standard" in df.columns:
        df = df[df["legal_standard"].str.lower() == "legal"].copy()

    # Keep historical behavior: compile only cards/effect groups that actually have text.
    # v0.10 accidentally made this opt-in, which added blank/no-text Energy groups and
    # made coverage less comparable. Use --include-no-text only for explicit audits.
    if not args.include_no_text:
        has_text = (
            df.get("rules", "").fillna("").str.strip().ne("")
            | df.get("abilities_text", "").fillna("").str.strip().ne("")
            | df.get("attacks_text", "").fillna("").str.strip().ne("")
            | df.get("combined_text", "").fillna("").str.strip().ne("")
        )
        df = df[has_text].copy()

    groups = build_groups(df)
    total_groups_before_limit = len(groups)

    if args.max_groups is not None:
        groups = groups[: args.max_groups]

    compiled_cards = [
        normalized_card_definition(group, compiler_version=args.compiler_version)
        for group in groups
    ]

    by_status: dict[str, list[dict[str, Any]]] = {
        "complete": [],
        "partial": [],
        "needs_human_review": [],
    }

    review_rows: list[dict[str, Any]] = []

    for card in compiled_cards:
        status = card["parser"]["status"]
        by_status.setdefault(status, []).append(card)
        if status != "complete":
            review_rows.append({
                "effect_group_id": card["effect_group_id"],
                "representative_card_id": card["representative_card_id"],
                "name": card["identity"]["name"],
                "supertype": card["identity"]["supertype"],
                "subtypes": "|".join(card["identity"]["subtypes"]),
                "same_effect_printing_count": card["same_effect_printing_count"],
                "unparsed_text_count": len(card["parser"]["unparsed_text"]),
                "unparsed_text": " || ".join(card["parser"]["unparsed_text"]),
            })

    batch_payload = {
        "schema_version": "pokemon-compiled-card-batch/v1",
        "compiler_version": args.compiler_version,
        "source_file": str(args.input),
        "deduped": True,
        "original_card_rows": original_rows,
        "filtered_card_rows": len(df),
        "unique_effect_groups_total_before_limit": total_groups_before_limit,
        "unique_effect_groups_written": len(compiled_cards),
        "compiled_cards": compiled_cards,
    }

    write_json(args.output_dir / "compiled_cards_all.json", batch_payload)

    for status, cards in by_status.items():
        write_json(args.output_dir / status / f"compiled_cards_{status}.json", {
            "schema_version": "pokemon-compiled-card-batch/v1",
            "compiler_version": args.compiler_version,
            "source_file": str(args.input),
            "parser_status": status,
            "count": len(cards),
            "compiled_cards": cards,
        })

    pd.DataFrame(review_rows).to_csv(args.report_dir / "review_queue.csv", index=False)

    status_counts = Counter(card["parser"]["status"] for card in compiled_cards)
    supertype_counts = Counter(card["identity"]["supertype"] for card in compiled_cards)
    unparsed_counter = Counter()
    for card in compiled_cards:
        for text in card["parser"]["unparsed_text"]:
            unparsed_counter[text] += 1

    summary = {
        "source_file": str(args.input),
        "original_card_rows": original_rows,
        "filtered_card_rows": len(df),
        "unique_effect_groups_total_before_limit": total_groups_before_limit,
        "unique_effect_groups_written": len(compiled_cards),
        "status_counts": dict(status_counts),
        "supertype_counts": dict(supertype_counts),
        "coverage": {
            "complete_rate": round(status_counts.get("complete", 0) / max(len(compiled_cards), 1), 4),
            "partial_rate": round(status_counts.get("partial", 0) / max(len(compiled_cards), 1), 4),
            "needs_human_review_rate": round(status_counts.get("needs_human_review", 0) / max(len(compiled_cards), 1), 4),
        },
        "top_unparsed_text": [
            {"text": text, "count": count}
            for text, count in unparsed_counter.most_common(100)
        ],
        "outputs": {
            "compiled_all": str(args.output_dir / "compiled_cards_all.json"),
            "complete": str(args.output_dir / "complete" / "compiled_cards_complete.json"),
            "partial": str(args.output_dir / "partial" / "compiled_cards_partial.json"),
            "needs_human_review": str(args.output_dir / "needs_review" / "compiled_cards_needs_human_review.json"),
            "review_queue": str(args.report_dir / "review_queue.csv"),
        },
    }

    write_json(args.report_dir / "compiler_coverage.json", summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
