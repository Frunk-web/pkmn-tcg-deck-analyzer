from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Pattern

from .models import Step, TemplateMatch

Builder = Callable[[re.Match[str], str], list[Step]]


@dataclass(frozen=True)
class TextTemplate:
    template_id: str
    family: str
    pattern: Pattern[str]
    builder: Builder
    confidence: float = 0.9
    executable: bool = True
    priority: int = 100
    notes: tuple[str, ...] = ()

    def try_match(self, text: str) -> TemplateMatch | None:
        m = self.pattern.search(text.strip())
        if not m:
            return None
        return TemplateMatch(
            family=self.family,
            template_id=self.template_id,
            source_text=text,
            steps=self.builder(m, text),
            confidence=self.confidence,
            executable=self.executable,
            notes=list(self.notes),
        )


def _amount(m: re.Match[str], default: int | None = None) -> int | str | None:
    value = m.groupdict().get("amount")
    if value is None:
        return default
    if value.lower() in {"all", "any"}:
        return value.lower()
    try:
        return int(value)
    except ValueError:
        return value


def build_heal_damage(m: re.Match[str], text: str) -> list[Step]:
    amount = _amount(m)
    target = m.groupdict().get("target") or "self.active_or_specified"
    steps: list[Step] = [
        {
            "op": "heal_damage",
            "target": target,
            "amount": {"mode": "exact", "value": amount},
            "source_text": text,
        }
    ]
    if re.search(r"remove (a |all )?special condition", text, re.I):
        steps.append({"op": "remove_special_condition", "target": target, "amount": "all", "source_text": text})
    return steps


def build_remove_damage_counters(m: re.Match[str], text: str) -> list[Step]:
    amount = _amount(m)
    target = m.groupdict().get("target") or "self.active_or_specified"
    return [
        {
            "op": "remove_damage_counters",
            "target": target,
            "amount": {"mode": "exact", "value": amount},
            "source_text": text,
        }
    ]


def build_damage_counters(m: re.Match[str], text: str) -> list[Step]:
    amount = _amount(m)
    target = m.groupdict().get("target") or "opponent.active_or_specified"
    return [
        {
            "op": "place_damage_counters",
            "target": target,
            "amount": {"mode": "exact", "value": amount},
            "source_text": text,
        }
    ]


def build_search_deck(m: re.Match[str], text: str) -> list[Step]:
    amount = _amount(m, default="up_to")
    card_filter = (m.groupdict().get("filter") or "card").strip()
    destination = "hand"
    if re.search(r"put (it|them|as many of them|up to \d+ of them) onto your bench", text, re.I):
        destination = "bench"
    elif re.search(r"attach (it|them|those cards|.*) to", text, re.I):
        destination = "attached"
    return [
        {
            "op": "search_deck",
            "player": "self",
            "selection": {
                "mode": "up_to" if re.search(r"up to|any number|as many", text, re.I) else "exact",
                "value": amount,
                "filter": {"text_filter": card_filter},
            },
            "destination": f"self.{destination}",
            "reveal": bool(re.search(r"reveal|show", text, re.I)),
            "source_text": text,
        },
        {"op": "shuffle_deck", "player": "self", "source_text": text},
    ]


def build_topdeck(m: re.Match[str], text: str) -> list[Step]:
    amount = _amount(m, default=1)
    steps: list[Step] = [
        {
            "op": "look_at_top_cards",
            "player": "self",
            "amount": {"mode": "exact", "value": amount},
            "source_text": text,
        }
    ]
    if re.search(r"put (it|one|1|that card|a .* you find there) into your hand", text, re.I):
        steps.append(
            {
                "op": "choose_cards",
                "player": "self",
                "source_zone": "viewed_cards",
                "destination": "self.hand",
                "selection": {"mode": "up_to", "value": 1},
                "source_text": text,
            }
        )
    if re.search(r"shuffle", text, re.I):
        steps.append({"op": "shuffle_deck", "player": "self", "source_text": text})
    return steps


def build_draw_cards(m: re.Match[str], text: str) -> list[Step]:
    amount = _amount(m)
    player = "both" if re.search(r"each player|both players", text, re.I) else "self"
    return [
        {
            "op": "draw_cards",
            "player": player,
            "amount": {"mode": "exact", "value": amount},
            "source_text": text,
        }
    ]


def build_attach_energy(m: re.Match[str], text: str) -> list[Step]:
    amount = _amount(m, default=1)
    source_zone = "discard" if re.search(r"from your discard pile", text, re.I) else "hand"
    target = "self.bench" if re.search(r"benched", text, re.I) else "self.pokemon"
    energy_type = m.groupdict().get("energy_type") or "Energy"
    return [
        {
            "op": "attach_card",
            "player": "self",
            "source_zone": f"self.{source_zone}",
            "target": target,
            "selection": {
                "mode": "up_to" if re.search(r"up to", text, re.I) else "exact",
                "value": amount,
                "filter": {"supertype": "Energy", "text_filter": energy_type.strip()},
            },
            "source_text": text,
        }
    ]


def build_move_energy(m: re.Match[str], text: str) -> list[Step]:
    amount = _amount(m, default=1)
    return [
        {
            "op": "move_attached_energy",
            "player": "self",
            "amount": {"mode": "up_to" if str(amount).lower() in {"any", "all"} else "exact", "value": amount},
            "source": "self.pokemon",
            "destination": "self.pokemon",
            "source_text": text,
        }
    ]


def build_switch(m: re.Match[str], text: str) -> list[Step]:
    opponent = bool(re.search(r"opponent", text, re.I))
    steps: list[Step] = []
    if opponent:
        steps.append({"op": "switch_active", "player": "opponent", "source_text": text})
    if re.search(r"switch (this pokémon|your active|your active pokémon|this pokemon)", text, re.I):
        steps.append({"op": "switch_active", "player": "self", "source_text": text})
    if not steps:
        steps.append({"op": "switch_active", "player": "opponent" if opponent else "self", "source_text": text})
    return steps


def build_prevention(m: re.Match[str], text: str) -> list[Step]:
    return [
        {
            "op": "register_prevention_effect",
            "scope": "attacks",
            "duration": "opponent_next_turn" if re.search(r"opponent'?s next turn", text, re.I) else "text_defined",
            "prevents_damage": bool(re.search(r"prevent all damage|prevent that damage", text, re.I)),
            "prevents_effects": bool(re.search(r"effects", text, re.I)),
            "source_text": text,
        }
    ]


def build_special_condition(m: re.Match[str], text: str) -> list[Step]:
    conditions = []
    for cond in ["Asleep", "Burned", "Confused", "Paralyzed", "Poisoned"]:
        if re.search(cond, text, re.I):
            conditions.append(cond)
    if not conditions and re.search(r"special condition", text, re.I):
        conditions = ["chosen_special_condition"]
    return [
        {
            "op": "apply_special_condition",
            "target": "opponent.active",
            "conditions": conditions,
            "source_text": text,
        }
    ]


def build_continuous_rule(family: str) -> Builder:
    def _builder(m: re.Match[str], text: str) -> list[Step]:
        return [
            {
                "op": "register_continuous_modifier",
                "family": family,
                "scope": "text_defined",
                "source_text": text,
            }
        ]

    return _builder


def build_delayed_rule(family: str) -> Builder:
    def _builder(m: re.Match[str], text: str) -> list[Step]:
        return [
            {
                "op": "register_delayed_effect",
                "family": family,
                "trigger": "text_defined",
                "source_text": text,
            }
        ]

    return _builder


def build_templates() -> list[TextTemplate]:
    flags = re.I
    templates = [
        TextTemplate("heal-damage", "healing_or_damage_counter_removal", re.compile(r"heal (?P<amount>\d+) damage(?: from)? (?P<target>.+?)(?:\.|$)", flags), build_heal_damage, priority=10),
        TextTemplate("remove-damage-counters", "healing_or_damage_counter_removal", re.compile(r"remove (?P<amount>\d+) damage counters? from (?P<target>.+?)(?:\.|$)", flags), build_remove_damage_counters, priority=10),
        TextTemplate("put-damage-counters", "damage_counters", re.compile(r"put (?P<amount>\d+) damage counters? on (?P<target>.+?)(?:\.|$)", flags), build_damage_counters, priority=20),
        TextTemplate("search-deck-general", "search_deck", re.compile(r"search your deck for (?:(?:up to|any number of) )?(?P<amount>\d+|any)?\s*(?P<filter>.+?)(?:,| and|\.)(?:.*shuffle your deck(?: afterward)?\.)?", flags), build_search_deck, priority=30),
        TextTemplate("topdeck-look", "topdeck_look_choose_or_reorder", re.compile(r"look at the top (?P<amount>\d+|card) cards? of your deck", flags), build_topdeck, priority=35),
        TextTemplate("draw-cards", "draw_shuffle_hand", re.compile(r"draw (?P<amount>\d+) cards?", flags), build_draw_cards, priority=40),
        TextTemplate("attach-energy", "energy_attachment_or_acceleration", re.compile(r"attach (?:(?:up to|a|an) )?(?P<amount>\d+)?\s*(?P<energy_type>[A-Za-z ]*Energy(?: cards?)?) from your (?:hand|discard pile) to", flags), build_attach_energy, priority=50),
        TextTemplate("move-energy", "energy_discard_move_or_bounce", re.compile(r"move (?P<amount>\d+|a|an|all)?\s*(?:basic )?Energy (?:card )?(?:attached )?from .* to", flags), build_move_energy, priority=55),
        TextTemplate("switch-gust", "gust_before_or_during_damage", re.compile(r"switch (?:in )?(?:1 of )?(?:your opponent's |your |this ).*?(?:active|benched|bench)", flags), build_switch, priority=60),
        TextTemplate("prevention", "damage_prevention_or_reduction", re.compile(r"prevent (?:all )?(?:damage|effects|damage from and effects)", flags), build_prevention, priority=70),
        TextTemplate("special-condition", "special_conditions", re.compile(r"(?:is now|are now|affected by|special conditions?|recovers from).*?(?:Asleep|Burned|Confused|Paralyzed|Poisoned|Special Condition)", flags), build_special_condition, priority=80),
        TextTemplate("weakness-resistance", "weakness_resistance_global_rule", re.compile(r"(?:don't|do not) apply weakness (?:and|or) resistance|has no weakness|has no resistance|weakness .*×2", flags), build_continuous_rule("weakness_resistance_global_rule"), priority=90),
        TextTemplate("ability-lock", "ability_or_pokemon_power_lock", re.compile(r"(?:can't use|have no|stop working).*?(?:abilities|poké-powers|poké-bodies|pokemon powers)", flags), build_continuous_rule("ability_or_pokemon_power_lock"), priority=95),
        TextTemplate("retreat-cost", "retreat_cost_modification", re.compile(r"(?:retreat cost|can't retreat|less to retreat|more to retreat|no retreat cost)", flags), build_continuous_rule("retreat_cost_modification"), priority=100),
        TextTemplate("future-damage", "future_damage_bonus_or_mark", re.compile(r"during your next turn.*?(?:does|do) \d+ more damage|until the end of your next turn", flags), build_delayed_rule("future_damage_bonus_or_mark"), priority=110),
        TextTemplate("tool-lifecycle", "tool_attachment_or_lifecycle_rule", re.compile(r"(?:attach this card|pokemon tool|pokémon tool|discard this card at the end|when this card is removed)", flags), build_continuous_rule("tool_attachment_or_lifecycle_rule"), priority=120),
        TextTemplate("prize-rule", "prize_rule_or_visibility", re.compile(r"(?:prize cards?|take 1 more prize|more prize cards remaining|face up)", flags), build_continuous_rule("prize_rule_or_visibility"), priority=130),
    ]
    return sorted(templates, key=lambda t: t.priority)
