from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Pattern

from .models import Step, TemplateMatch

Builder = Callable[[re.Match[str], str], list[Step]]


def clean_text(text: str) -> str:
    text = re.sub(r"^Rules:\s*", "", text.strip(), flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text


def amount_exact(value: int | str) -> dict[str, int | str]:
    return {"mode": "exact", "value": value}


def amount_up_to(value: int | str) -> dict[str, int | str]:
    return {"mode": "up_to", "value": value}


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
        cleaned = clean_text(text)
        m = self.pattern.fullmatch(cleaned)
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


def step(op: str, text: str, **kwargs) -> Step:
    payload = {"op": op, **kwargs}
    payload.setdefault("source_text", text)
    return payload


def int_group(m: re.Match[str], name: str, default: int = 1) -> int:
    raw = m.groupdict().get(name)
    if raw is None or raw == "":
        return default
    raw_text = str(raw).strip().lower()
    if raw_text in {"a", "an", "one"}:
        return 1
    found = re.search(r"\d+", raw_text)
    if found:
        return int(found.group(0))
    return default


def build_heal_damage(m: re.Match[str], text: str) -> list[Step]:
    target_text = m.groupdict().get("target") or "from_text"
    target = "self.active" if re.search(r"active", target_text, re.I) else "self.in_play"
    steps = [step("heal_damage", text, target=target, amount=int_group(m, "amount"), target_text=target_text)]
    if re.search(r"remove (?:a |all )?special condition", clean_text(text), re.I):
        steps.append(step("remove_special_condition", text, target=target, amount={"mode": "all"}))
    return steps


def build_remove_damage_counters(m: re.Match[str], text: str) -> list[Step]:
    target_text = m.groupdict().get("target") or "from_text"
    target = "self.in_play" if re.search(r"your|each of your", target_text, re.I) else "from_text"
    return [step("remove_damage_counters", text, target=target, amount=int_group(m, "amount"), target_text=target_text)]


def build_mass_damage_counters(m: re.Match[str], text: str) -> list[Step]:
    target_text = m.groupdict().get("target") or "from_text"
    target = "opponent.in_play" if re.search(r"opponent", target_text, re.I) else "all.in_play"
    return [step("place_damage_counters", text, target=target, amount=int_group(m, "amount"), target_text=target_text)]


def build_weakness_resistance_damage(m: re.Match[str], text: str) -> list[Step]:
    amount = int_group(m, "amount", 0)
    target_text = m.groupdict().get("target") or "from_text"
    target = "all.in_play" if re.search(r"each pokémon", target_text, re.I) else "from_text"
    return [
        step("deal_damage", text, target=target, amount=amount, target_text=target_text, apply_weakness_resistance=False),
        step("modify_damage_calculation", text, scope="current_attack", apply_weakness=False, apply_resistance=False),
    ]


def build_weakness_resistance_rule(m: re.Match[str], text: str) -> list[Step]:
    return [step("modify_damage_calculation", text, scope="current_attack_or_text_defined", apply_weakness=False, apply_resistance=False)]


def build_coin_damage_status(m: re.Match[str], text: str) -> list[Step]:
    coin_count = int_group(m, "coin_count", 1)
    amount_per_heads = int_group(m, "amount", 0)
    status = m.groupdict().get("status")
    threshold = int(m.groupdict().get("threshold") or 1)
    steps: list[Step] = [
        step("coin_flip", text, player="self", count=coin_count, target_id="coin_results"),
        step("set_attack_damage_from_coin_heads", text, coin_results_ref="coin_results", amount_per_heads=amount_per_heads),
    ]
    if status:
        steps.append(step("branch_on_coin_heads", text, result_ref="coin_results", heads_at_least=threshold, then=[{"op": "apply_special_condition", "target": "opponent.active", "condition": status, "source_text": text}]))
    return steps


def build_conditional_damage_status(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("coin_flip", text, player="self", count=1, target_id="coin_result"),
        step("branch_on_result", text, result_ref="coin_result", **{"if": "heads"}, then=[{"op": "modify_attack_damage", "amount_delta": int_group(m, "heads_bonus", 0), "source_text": text}]),
        step("branch_on_result", text, result_ref="coin_result", **{"if": "tails"}, then=[{"op": "apply_special_condition", "target": "opponent.active", "condition": m.group("tails_status"), "source_text": text}]),
    ]


def build_search_deck(m: re.Match[str], text: str) -> list[Step]:
    player = "each_player" if re.search(r"each player's turn|that player", clean_text(text), re.I) else "self"
    filter_text = m.groupdict().get("filter") or "from_text"
    destination = "player.hand" if player == "each_player" else "self.hand"
    if re.search(r"onto your bench|onto your Bench|put as many", text, re.I):
        destination = "self.bench"
    elif re.search(r"attach", text, re.I):
        destination = "self.in_play.attached"
    steps = [step("search_deck", text, player=player, target_id="searched_cards", filter={"from_text": filter_text}, amount={"mode": "from_text"}, reveal=bool(re.search(r"reveal|show", text, re.I)), destination=destination)]
    if re.search(r"shuffle", text, re.I):
        steps.append(step("shuffle_deck", text, player="self" if player == "self" else "that_player"))
    return steps


def build_castform_switch(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("search_deck", text, player="self", target_id="searched_castform_form", filter={"names_from_text": True}, amount=amount_up_to(1), reveal=True, destination="replacement_candidate"),
        step("replace_pokemon_in_play", text, old_pokemon="source_pokemon", new_pokemon_ref="searched_castform_form", preserve_attached_cards=True, preserve_damage_counters=True, preserve_special_conditions=True, preserve_effects=True),
        step("shuffle_deck", text, player="self"),
        step("register_usage_limit", text, limit=1, scope="named_poke_power_per_turn"),
    ]


def build_switch(m: re.Match[str], text: str) -> list[Step]:
    cleaned = clean_text(text)
    if re.search(r"each player's turn|that player may switch", cleaned, re.I):
        return [step("register_player_turn_action", text, action={"op": "switch_active", "player": "that_player", "filter": {"from_text": cleaned}}, usage_limit={"scope": "per_player_turn", "limit": 1})]
    steps: list[Step] = []
    if re.search(r"opponent", cleaned, re.I):
        steps.append(step("switch_active", text, player="opponent"))
    if re.search(r"your Active|this Pokémon|your Bench", cleaned, re.I):
        steps.append(step("switch_active", text, player="self"))
    if not steps:
        steps.append(step("switch_active", text, player="self"))
    return steps


def build_gust_damage(m: re.Match[str], text: str) -> list[Step]:
    amount = int_group(m, "amount", 0)
    return [
        step("choose_target", text, player="self", target_id="chosen_opponent_benched_pokemon", zone="opponent.bench", filter={"supertype": "Pokémon"}, amount=amount_up_to(1)),
        step("switch_active", text, player="opponent", target_ref="chosen_opponent_benched_pokemon"),
        step("deal_damage", text, target="opponent.active", amount=amount if amount else {"mode": "from_text"}),
    ]


def build_energy_move_trigger(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_triggered_effect", text, trigger="self_active_knocked_out_by_opponent_attack", then=[{"op": "move_attached_energy", "energy_filter": {"subtypes": ["Basic"]}, "source": "knocked_out_pokemon", "destination": "attached_pokemon", "amount": {"mode": "up_to", "value": 1}, "source_text": text}])]


def build_damage_reduction(m: re.Match[str], text: str) -> list[Step]:
    amount = int(m.groupdict().get("amount") or m.groupdict().get("amount2") or 0)
    duration = "while_active" if re.search(r"as long as", text, re.I) else "opponent_next_turn"
    return [step("register_continuous_modifier", text, family="opponent_attack_damage_reduction", target="self_or_attached_pokemon", duration=duration, modification={"opponent_attack_damage_delta": -amount})]


def build_global_rule(family: str) -> Builder:
    def _builder(m: re.Match[str], text: str) -> list[Step]:
        return [step("register_continuous_modifier", text, family=family, scope="text_defined")]
    return _builder


def build_devolution(m: re.Match[str], text: str) -> list[Step]:
    return [step("devolve_pokemon", text, player="opponent", target="opponent.evolved_pokemon", amount={"mode": "highest_stage"}, destination="opponent.hand")]


def build_copy_attack(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="copy_or_grant_attack_access", modification={"can_use_attacks_from_text": True})]



def count_group(m: re.Match[str], name: str, default: int = 1) -> int:
    raw = (m.groupdict().get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in {"a", "an", "one"}:
        return 1
    return int(raw)


def build_coin_damage_only(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("coin_flip", text, player="self", count=count_group(m, "coin_count"), target_id="coin_results"),
        step("set_attack_damage_from_coin_heads", text, coin_results_ref="coin_results", amount_per_heads=int_group(m, "amount")),
    ]


def build_future_damage_bonus(m: re.Match[str], text: str) -> list[Step]:
    amount = int_group(m, "amount")
    return [step("register_delayed_modifier", text, modifier_id="future_damage_bonus", target="defending_pokemon_or_named_attack", duration={"until": "end_of_self_next_turn"}, modification={"attack_damage_delta": amount}, condition={"attack_does_damage": True})]


def build_opponent_hand_shuffle(m: re.Match[str], text: str) -> list[Step]:
    amount = int_group(m, "amount", 0)
    steps: list[Step] = []
    if re.search(r"Flip a coin", text, re.I):
        steps.append(step("coin_flip", text, player="self", count=1, target_id="coin_result"))
    steps.append(step("choose_cards", text, player="self", target_id="chosen_opponent_hand_cards", zone="opponent.hand", amount=amount if amount else {"mode": "from_text"}, random=bool(re.search(r"random", text, re.I))))
    steps.append(step("reveal_cards", text, player="opponent", cards_ref="chosen_opponent_hand_cards"))
    steps.append(step("move_cards", text, source="opponent.hand", destination="opponent.deck", cards_ref="chosen_opponent_hand_cards"))
    steps.append(step("shuffle_deck", text, player="opponent"))
    return steps


def build_recycle_to_deck(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("choose_cards", text, player="self", target_id="discard_recycle_cards", zone="self.discard", filter={"from_text": True}, amount={"mode": "up_to", "value": int_group(m, "amount")}),
        step("move_cards", text, source="self.discard", destination="self.deck", cards_ref="discard_recycle_cards"),
        step("shuffle_deck", text, player="self"),
    ]


def build_energy_move_general(m: re.Match[str], text: str) -> list[Step]:
    return [step("move_attached_energy", text, player="self", energy_filter={"from_text": True}, source="self.in_play", destination="self.in_play_or_bench", amount={"mode": "from_text"})]


def build_attach_energy_general(m: re.Match[str], text: str) -> list[Step]:
    source = "self.discard" if re.search(r"discard pile", text, re.I) else "self.hand"
    return [step("attach_card", text, player="self", source=source, target="self.pokemon", filter={"category": "Energy", "from_text": True}, amount={"mode": "from_text"})]


def build_heal_each_player(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_player_turn_action", text, action={"op": "heal_damage", "target": "that_player.pokemon", "amount": int_group(m, "amount")}, condition={"from_text": True}, usage_limit={"scope": "per_player_turn", "limit": 1})]


def build_draw_turn_action(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_player_turn_action", text, action={"op": "draw_cards", "player": "that_player", "amount": int_group(m, "amount")}, condition={"from_text": True}, usage_limit={"scope": "per_player_turn", "limit": 1})]


def build_draw_until(m: re.Match[str], text: str) -> list[Step]:
    amount = int_group(m, "amount", 0)
    if amount:
        return [step("draw_until_hand_size", text, player="self_or_that_player", hand_size=amount, condition={"from_text": True})]
    return [step("draw_until_hand_size_matches", text, player="that_player", comparison={"from_text": True}, condition={"from_text": True})]


def build_prevention(m: re.Match[str], text: str) -> list[Step]:
    scope = "self.bench" if re.search(r"Benched Pokémon", text, re.I) else "self_or_this_pokemon"
    return [step("register_prevention_effect", text, target=scope, prevents={"damage": bool(re.search(r"damage", text, re.I)), "effects": bool(re.search(r"effects", text, re.I))}, source_filter={"from_text": True}, duration={"from_text": True})]


def build_status_condition(m: re.Match[str], text: str) -> list[Step]:
    condition = m.groupdict().get("status") or "from_text"
    return [step("apply_special_condition", text, target="opponent.active_or_defending", condition=condition, condition_clause={"from_text": True})]


def build_bench_only_rule(m: re.Match[str], text: str) -> list[Step]:
    return [step("play_condition", text, condition={"bench_only_with_named_effect": True, "from_text": text})]


def build_return_to_hand(m: re.Match[str], text: str) -> list[Step]:
    target = "self.this_pokemon" if re.search(r"this Pokémon", text, re.I) else "self.pokemon"
    return [step("move_zone_to_zone", text, source="self.in_play", destination="self.hand", target=target, include_attached_cards=True)]


def build_damage_counter_move(m: re.Match[str], text: str) -> list[Step]:
    return [step("move_damage_counters", text, source="self.pokemon", destination="opponent.pokemon", amount=int_group(m, "amount"))]


def build_damage_counter_trigger(m: re.Match[str], text: str) -> list[Step]:
    amount = int_group(m, "amount")
    trigger = "opponent_poke_power_used" if re.search(r"Poké-Power", text, re.I) else "attached_or_active_pokemon_damaged_by_opponent_attack"
    return [step("register_triggered_effect", text, trigger=trigger, then=[{"op": "place_damage_counters", "target": "attacking_or_triggering_pokemon", "amount": amount, "source_text": text}])]


def build_tool_retreat_replacement(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_replacement_effect", text, event="attached_pokemon_retreats", replace={"discard_energy": False, "discard_this_card": True})]


def build_attack_cost_modifier(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="attack_cost_modification", target="attached_or_text_defined_pokemon", condition={"from_text": True}, modification={"attack_cost_delta_colorless": -1})]


def build_attack_damage_modifier(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="tool_or_attachment_damage_modifier", target="attached_or_text_defined_pokemon", condition={"from_text": True}, modification={"attack_damage_delta": int_group(m, "amount")})]


def build_self_weakness_rule(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="weakness_resistance_global_rule", target="self.pokemon", modification={"has_no_weakness": True}, condition={"from_text": True})]


def build_discard_top_deck(m: re.Match[str], text: str) -> list[Step]:
    amount = int_group(m, "amount", 1)
    player = "opponent" if re.search(r"opponent", text, re.I) else "self"
    return [step("discard_cards", text, player=player, source=f"{player}.deck", destination=f"{player}.discard", amount=amount, position="top")]


def build_coin_search_deck(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("coin_flip", text, player="self", count=1, target_id="coin_result"),
        step("branch_on_result", text, result_ref="coin_result", **{"if": "heads"}, then=[
            {"op": "search_deck", "player": "self", "target_id": "searched_card", "filter": {"from_text": True}, "amount": {"mode": "up_to", "value": int_group(m, "amount", 1)}, "destination": "self.hand", "source_text": text},
            {"op": "shuffle_deck", "player": "self", "source_text": text},
        ]),
    ]


def build_generic_damage_formula(m: re.Match[str], text: str) -> list[Step]:
    return [step("set_or_modify_attack_damage", text, formula={"from_text": True})]


def build_damage_plus_condition(m: re.Match[str], text: str) -> list[Step]:
    return [step("modify_attack_damage", text, amount_delta=int_group(m, "amount"), condition={"from_text": True})]


def build_discard_energy(m: re.Match[str], text: str) -> list[Step]:
    target = "self.active_or_source" if re.search(r"Raichu|this Pokémon|Garchomp|attached to", text, re.I) else "from_text"
    amount_raw = m.groupdict().get("amount")
    amount = {"mode": "all"} if re.search(r"all Energy", text, re.I) else ({"mode": "exact", "value": int(amount_raw)} if amount_raw else {"mode": "from_text"})
    return [step("discard_attached_energy", text, player="self", target=target, energy_filter={"from_text": True}, amount=amount)]


def build_discard_energy_for_bonus(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("optional_cost", text, cost={"op": "discard_attached_energy", "target": "self.active_or_source", "energy_filter": {"from_text": True}, "amount": {"mode": "from_text"}}),
        step("modify_attack_damage", text, amount_delta=int_group(m, "amount"), condition={"cost_paid": True}),
    ]


def build_no_retreat_condition(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="retreat_cost_modification", target="text_defined_pokemon", condition={"from_text": True}, modification={"retreat_cost": 0})]


def build_retreat_lock(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_delayed_modifier", text, family="retreat_lock_or_attack_lock", target="defending_or_opponent_active", duration={"from_text": True}, modification={"can_retreat": False, "can_use_poke_powers": not bool(re.search(r"Poké-Powers", text, re.I))})]


def build_heal_all(m: re.Match[str], text: str) -> list[Step]:
    target = "self.bench" if re.search(r"Benched", text, re.I) else "self.pokemon"
    return [step("heal_damage", text, target=target, amount={"mode": "all"})]


def build_remove_all_damage_counters(m: re.Match[str], text: str) -> list[Step]:
    target = "self.pokemon" if re.search(r"each of your", text, re.I) else "from_text"
    return [step("remove_damage_counters", text, target=target, amount={"mode": "all"}, condition={"from_text": True})]


def build_play_evolve_trigger(m: re.Match[str], text: str) -> list[Step]:
    then: list[Step] = []
    if re.search(r"search your deck", text, re.I):
        then.extend([
            {"op": "search_deck", "player": "self", "target_id": "searched_card", "filter": {"from_text": True}, "amount": {"mode": "from_text"}, "destination": "self.hand_or_bench", "source_text": text},
            {"op": "shuffle_deck", "player": "self", "source_text": text},
        ])
    if re.search(r"attach .*Energy from your discard pile", text, re.I):
        then.append({"op": "attach_card", "player": "self", "source": "self.discard", "target": "self.pokemon", "filter": {"category": "Energy", "from_text": True}, "amount": {"mode": "from_text"}, "source_text": text})
    if re.search(r"switch .*opponent", text, re.I):
        then.append({"op": "switch_active", "player": "opponent", "source_text": text})
    if re.search(r"discard the top (?P<n>\d+) cards? of your opponent's deck", text, re.I):
        m2 = re.search(r"discard the top (?P<n>\d+) cards? of your opponent's deck", text, re.I)
        then.append({"op": "discard_cards", "player": "opponent", "source": "opponent.deck", "destination": "opponent.discard", "amount": int(m2.group('n')) if m2 else {"mode": "from_text"}, "position": "top", "source_text": text})
    if re.search(r"put \d+ damage counters", text, re.I):
        n = re.search(r"put (?P<n>\d+) damage counters", text, re.I)
        then.append({"op": "place_damage_counters", "target": "opponent.pokemon", "amount": int(n.group('n')) if n else {"mode": "from_text"}, "source_text": text})
    if re.search(r"remove all damage counters", text, re.I):
        then.append({"op": "remove_damage_counters", "target": "self.pokemon", "amount": {"mode": "all"}, "source_text": text})
    if re.search(r"prevent all effects", text, re.I):
        then.append({"op": "register_prevention_effect", "target": "this_pokemon", "prevents": {"damage": True, "effects": True}, "duration": {"from_text": True}, "source_text": text})
    if not then:
        then.append({"op": "register_triggered_effect_payload", "payload": {"from_text": True}, "source_text": text})
    return [step("register_triggered_effect", text, trigger="when_played_from_hand_to_evolve_or_bench", then=then)]


def build_old_power_energy_action(m: re.Match[str], text: str) -> list[Step]:
    action_op = "move_attached_energy" if re.search(r"move", text, re.I) else "attach_card"
    source = "self.in_play" if action_op == "move_attached_energy" else ("self.discard" if re.search(r"discard pile", text, re.I) else "self.hand")
    return [step("register_player_turn_action", text, action={"op": action_op, "source": source, "target": "self.pokemon", "filter": {"category": "Energy", "from_text": True}, "amount": {"mode": "one_or_repeatable_from_text"}}, condition={"not_affected_by_special_condition": True, "from_text": True}, usage_limit={"from_text": True})]


def build_status_immunity(m: re.Match[str], text: str) -> list[Step]:
    target = "self.pokemon" if re.search(r"Each of your", text, re.I) else "this_pokemon"
    return [step("register_continuous_modifier", text, family="special_condition_global_immunity_or_recovery", target=target, modification={"immune_to_special_conditions": True, "remove_existing_special_conditions": bool(re.search(r"Remove", text, re.I))}, condition={"from_text": True})]


def build_attack_lock(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_delayed_modifier", text, family="conditional_attack_availability", target="defending_or_self", duration={"from_text": True}, modification={"attack_constraint": {"from_text": True}})]


def build_hand_or_trainer_count_damage(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("reveal_cards", text, player="opponent", zone="opponent.hand"),
        step("set_or_modify_attack_damage", text, formula={"count_zone": "opponent.hand", "filter": {"from_text": True}, "multiplier": int_group(m, "amount")}),
    ]


def build_choose_attack_copy(m: re.Match[str], text: str) -> list[Step]:
    return [step("copy_attack", text, source="opponent_or_text_defined_pokemon", selection={"from_text": True})]


def build_put_damage_counter_dynamic(m: re.Match[str], text: str) -> list[Step]:
    return [step("place_damage_counters", text, target="opponent.active_or_defending", amount={"formula_from_text": True})]



# ---- v0.5 template builders: broad remaining long-tail mechanics ----

def build_coin_control(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_delayed_modifier", text, family="coin_flip_replacement", duration="opponent_next_turn", replacement={"coin_flip_result": "tails"})]


def build_coin_damage_status_flex(m: re.Match[str], text: str) -> list[Step]:
    coin_count = int_group(m, "coin_count", 1)
    amount = int_group(m, "amount", 0)
    status = m.groupdict().get("status")
    steps: list[Step] = [
        step("coin_flip", text, player="self", count=coin_count, target_id="coin_results"),
        step("set_attack_damage_from_coin_heads", text, coin_results_ref="coin_results", amount_per_heads=amount),
    ]
    if status:
        steps.append(step("branch_on_coin_heads", text, result_ref="coin_results", heads_at_least=int_group(m, "threshold", 1), then=[{"op": "apply_special_condition", "target": "opponent.active", "condition": status, "source_text": text}]))
    return steps


def build_choose_damage_ignore_wr(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("choose_target", text, player="self", target_id="chosen_targets", zone="opponent.in_play", amount=amount_exact(int_group(m, "target_count", 1)), filter={"supertype": "Pokémon"}),
        step("deal_damage", text, target_ref="chosen_targets", amount=int_group(m, "amount", 0), apply_weakness_resistance=False),
    ]


def build_status_condition_simple(m: re.Match[str], text: str) -> list[Step]:
    target_text = m.groupdict().get("target") or "opponent.active"
    target = "self" if re.search(r"this Pokémon|Magby|Attacking Pokémon", target_text, re.I) else "opponent.active"
    return [step("apply_special_condition", text, target=target, condition=m.group("status"), target_text=target_text)]


def build_power_lock_next_turn(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_delayed_modifier", text, family="ability_or_pokemon_power_lock", duration="opponent_next_turn", scope="opponent.pokemon", modification={"can_use_poke_powers": False, "can_use_abilities": False})]


def build_marker_power_lock(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("choose_target", text, player="self", target_id="marked_pokemon", zone="opponent.in_play", amount=amount_up_to(1), filter={"supertype": "Pokémon"}),
        step("place_marker", text, target_ref="marked_pokemon", marker="Imprison"),
        step("register_continuous_modifier", text, family="ability_or_pokemon_power_lock", condition={"has_marker": "Imprison"}, modification={"can_use_poke_powers": False, "can_use_poke_bodies": False}),
    ]


def build_previous_level_rule(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="level_x_previous_level_rule", modification={"copy_from_previous_level": True})]


def build_when_bench_damage_counter_trigger(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_delayed_modifier", text, family="bench_play_damage_counter_trigger", duration="opponent_next_turn", trigger="opponent_benches_basic_from_hand", then=[{"op": "place_damage_counters", "amount": int_group(m, "amount", 2), "target": "benched_pokemon", "source_text": text}])]


def build_attack_availability_coin_tax(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_delayed_modifier", text, family="attack_availability_coin_check", duration="opponent_next_turn", trigger="defending_pokemon_attempts_attack", then=[{"op": "coin_flip", "player": "opponent", "target_id": "attack_check"}, {"op": "branch_on_result", "result_ref": "attack_check", "if": "tails", "then": [{"op": "cancel_attack", "source_text": text}], "source_text": text}])]


def build_self_or_attached_return(m: re.Match[str], text: str) -> list[Step]:
    destination = "self.deck" if re.search(r"shuffle", text, re.I) else "self.hand"
    return [step("move_zone_to_zone", text, source="self.in_play", destination=destination, target="this_pokemon_and_attached_cards", include_attached=True)]


def build_no_retreat_active(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="retreat_cost_modification", condition={"from_text": True}, modification={"retreat_cost": 0})]


def build_search_once_turn(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("register_player_turn_action", text, action={"op": "search_deck", "filter": {"from_text": m.groupdict().get("filter") or True}, "destination": "self.hand", "reveal": bool(re.search(r"reveal", text, re.I))}, usage_limit={"scope": "once_during_turn", "limit": 1}),
        step("shuffle_deck", text, player="self"),
    ]


def build_attach_any_energy(m: re.Match[str], text: str) -> list[Step]:
    source = "self.discard" if re.search(r"discard pile", text, re.I) else "self.hand"
    amount = int_group(m, "amount", "any") if m.groupdict().get("amount") else {"mode": "any"}
    return [step("attach_card", text, source=source, destination="self.in_play", filter={"card_type": "Energy", "from_text": True}, amount=amount)]


def build_ability_draw_cost(m: re.Match[str], text: str) -> list[Step]:
    cost_discards = int_group(m, "discard_amount", 1)
    draw_amount = int_group(m, "draw_amount", 1)
    return [step("register_player_turn_action", text, action={"cost": {"op": "discard_cards", "amount": amount_exact(cost_discards), "source": "self.hand"}, "op": "draw_cards", "amount": amount_exact(draw_amount)}, usage_limit={"scope": "once_during_turn", "limit": 1})]


def build_damage_counter_fixed(m: re.Match[str], text: str) -> list[Step]:
    target = "self.in_play" if re.search(r"your Pokémon", text, re.I) else "opponent.in_play"
    return [step("place_damage_counters", text, target=target, amount=int_group(m, "amount", 1), target_text=m.groupdict().get("target") or "from_text")]


def build_special_condition_immunity_general(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="special_condition_global_immunity_or_recovery", scope="text_defined", modification={"special_condition_immunity": True, "remove_existing_special_conditions": bool(re.search(r"Remove", text, re.I))})]


def build_prevent_effects_except_damage(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_prevention_effect", text, duration="opponent_next_turn_or_text_defined", prevent_effects=True, prevent_damage=False, source_filter={"from_text": True})]


def build_damage_unaffected(m: re.Match[str], text: str) -> list[Step]:
    return [step("modify_damage_calculation", text, scope="self_attacks", ignore_weakness_resistance=bool(re.search(r"Weakness|Resistance", text, re.I)), ignore_effects_on_defender=bool(re.search(r"effects on", text, re.I)))]


def build_knockout_condition(m: re.Match[str], text: str) -> list[Step]:
    return [step("conditional_knock_out", text, target="opponent.active", condition={"from_text": True})]


def build_extra_damage_then_lock(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("optional_effect", text, then=[{"op": "modify_attack_damage", "amount_delta": int_group(m, "amount", 0), "source_text": text}, {"op": "register_delayed_modifier", "family": "attack_lock_or_attack_tax", "duration": "self_next_turn", "modification": {"cannot_attack": True}, "source_text": text}]),
    ]


def build_discard_energy_and_discard_opponent(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("optional_cost", text, cost={"op": "discard_attached_energy", "source": "self.active", "amount": amount_up_to(1)}),
        step("discard_attached_energy", text, target="opponent.active", amount=amount_up_to(1)),
    ]


def build_type_modifier(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="type_modifier", modification={"types_from_text": True})]




# v0.6 targeted long-tail builders. These are intentionally template-level IR emitters:
# they preserve source_text and avoid pretending the runtime has perfect battle semantics yet.
def build_topdeck_look_choose(m: re.Match[str], text: str) -> list[Step]:
    amount = int_group(m, "amount", 7)
    return [
        step("look_at_top_cards", text, player="self", target_id="topdeck_cards", source="self.deck", amount=amount),
        step("choose_cards", text, player="self", target_id="chosen_topdeck_card", source_ref="topdeck_cards", amount=amount_up_to(1), filter={"from_text": True}),
        step("reveal_cards", text, cards_ref="chosen_topdeck_card"),
        step("move_cards", text, source="self.deck.top", destination="self.hand", cards_ref="chosen_topdeck_card"),
        step("shuffle_deck", text, player="self", cards="other_topdeck_cards"),
    ]


def build_energy_provides_rule(m: re.Match[str], text: str) -> list[Step]:
    steps: list[Step] = []
    if re.search(r"remove \d+ damage counter", text, re.I):
        steps.append(step("remove_damage_counters", text, target="attached_pokemon", amount=int_group(m, "amount", 1), condition={"when_played_from_hand": True}))
    steps.append(step("register_continuous_modifier", text, family="energy_provides_or_type_modifier", target="this_energy_card", modification={"provides_energy_from_text": True, "does_not_count_as_basic_energy": bool(re.search(r"Doesn'?t count as a basic Energy", text, re.I))}))
    return steps


def build_tyrogue_evolution(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_player_turn_action", text, action={"op": "evolve_pokemon_from_hand", "target": "Tyrogue", "candidate_names_from_text": True, "then": [{"op": "remove_damage_counters", "target": "Tyrogue", "amount": {"mode": "all"}, "source_text": text}]}, usage_limit={"scope": "once_during_turn", "limit": 1})]


def build_damage_counter_until_near_ko(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("coin_flip", text, player="self", count=1, target_id="coin_result"),
        step("branch_on_result", text, result_ref="coin_result", **{"if": "heads"}, then=[{"op": "place_damage_counters_until_hp_remaining", "target": "opponent.active_or_defending", "hp_remaining": int_group(m, "hp", 10), "source_text": text}]),
    ]


def build_coin_bench_damage(m: re.Match[str], text: str) -> list[Step]:
    amount = int_group(m, "amount", 10)
    coin_count = int_group(m, "coin_count", 1)
    if re.search(r"If tails", text, re.I):
        then_heads = [{"op": "deal_damage", "target": "opponent.bench", "amount": amount, "apply_weakness_resistance": False, "source_text": text}]
        then_tails = [{"op": "deal_damage", "target": "self.bench", "amount": amount, "apply_weakness_resistance": False, "source_text": text}]
        return [
            step("coin_flip", text, player="self", count=1, target_id="coin_result"),
            step("branch_on_result", text, result_ref="coin_result", **{"if": "heads"}, then=then_heads),
            step("branch_on_result", text, result_ref="coin_result", **{"if": "tails"}, then=then_tails),
        ]
    return [
        step("coin_flip", text, player="self", count=coin_count, target_id="coin_results"),
        step("deal_damage_per_coin_heads", text, target="opponent.bench", amount_per_heads=amount, coin_results_ref="coin_results", apply_weakness_resistance=False),
    ]


def build_energy_hand_damage_deck(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("choose_cards", text, player="self", target_id="chosen_energy_cards", source="self.hand", amount={"mode": "any"}, filter={"category": "Energy"}),
        step("reveal_cards", text, cards_ref="chosen_energy_cards"),
        step("set_attack_damage_from_card_count", text, cards_ref="chosen_energy_cards", amount_per_card=int_group(m, "amount", 20)),
        step("move_cards", text, source="self.hand", destination="self.deck.top", cards_ref="chosen_energy_cards"),
        step("shuffle_deck", text, player="self"),
    ]


def build_lv_x_gust_trigger(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_triggered_effect", text, trigger="play_from_hand_to_level_up_active", then=[{"op": "switch_active", "player": "opponent", "source_text": text}])]


def build_item_trainer_lock_active(m: re.Match[str], text: str) -> list[Step]:
    locked = "Item" if re.search(r"Item", text, re.I) else "Trainer"
    return [step("register_continuous_modifier", text, family="opponent_item_supporter_trainer_lock", condition={"source_pokemon_is_active": True}, modification={"opponent_cannot_play_card_type": locked})]


def build_switch_both_active(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_player_turn_action", text, action={"op": "switch_active", "player": "self", "then": [{"op": "switch_active", "player": "opponent", "source_text": text}]}, usage_limit={"scope": "once_during_turn", "limit": 1})]


def build_garbotoxin_like_ability_lock(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="ability_or_pokemon_power_lock", condition={"source_has_pokemon_tool_attached": True}, scope="pokemon_in_play_hand_and_discard", modification={"has_no_abilities": True}, exceptions={"from_text": True})]


def build_tool_return_to_hand(m: re.Match[str], text: str) -> list[Step]:
    target = "this_pokemon_tool" if re.search(r"this Pokémon", text, re.I) else "self.pokemon_tool"
    return [step("register_player_turn_action", text, action={"op": "move_cards", "source": "self.in_play.attached_tools", "destination": "self.hand", "target": target}, usage_limit={"scope": "once_during_turn_or_as_often_as_text", "limit": "from_text"})]


def build_conditional_bonus_damage_general(m: re.Match[str], text: str) -> list[Step]:
    return [step("modify_attack_damage", text, amount_delta=int_group(m, "amount", 0), condition={"from_text": True})]


def build_damage_reduction_general(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="damage_prevention_or_reduction", condition={"from_text": True}, modification={"damage_taken_delta": -int_group(m, "amount", 0)})]


def build_attack_cost_tax_general(m: re.Match[str], text: str) -> list[Step]:
    direction = 1 if re.search(r"more", text, re.I) else -1
    return [step("register_continuous_modifier", text, family="attack_cost_modification", condition={"from_text": True}, modification={"attack_cost_delta_colorless": direction})]


def build_heal_and_remove_all_status(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("heal_damage", text, target="self.this_pokemon", amount=int_group(m, "amount", 0)),
        step("remove_special_condition", text, target="self.this_pokemon", amount={"mode": "all"}),
    ]


def build_setup_active_rule(m: re.Match[str], text: str) -> list[Step]:
    return [step("play_condition", text, phase="setup", condition={"may_start_face_down_as_active": True})]


def build_no_weakness_general(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="weakness_resistance_global_rule", target="self.pokemon_or_text_defined", condition={"from_text": True}, modification={"has_no_weakness": True})]


def build_copy_attack_extended(m: re.Match[str], text: str) -> list[Step]:
    return [step("copy_attack", text, source="text_defined_pokemon_or_zone", target="this_attack", requires_necessary_energy=True)]


def build_cannot_be_healed(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_delayed_modifier", text, family="global_stat_or_type_rule_modifier", target="defending_pokemon", duration="opponent_next_turn", modification={"can_be_healed": False})]


def build_attack_requires_condition(m: re.Match[str], text: str) -> list[Step]:
    return [step("attack_condition", text, condition={"from_text": True}, otherwise={"attack_does_nothing": True})]


def build_search_shuffle_topdeck(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("search_deck", text, player="self", target_id="searched_card", filter={"from_text": True}, amount=amount_up_to(1), destination="deck_top_after_shuffle"),
        step("shuffle_deck", text, player="self"),
        step("move_cards", text, source="searched_card", destination="self.deck.top", cards_ref="searched_card"),
    ]


def build_bench_capacity_rule(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_continuous_modifier", text, family="bench_capacity_rule", condition={"from_text": True}, modification={"bench_limit": int_group(m, "amount", 8), "restriction_from_text": True})]


def build_prize_replacement_rule(m: re.Match[str], text: str) -> list[Step]:
    return [step("register_replacement_effect", text, event="source_pokemon_knocked_out_by_opponent_attack", replace={"prize_cards_taken_delta": -int_group(m, "amount", 1)}, condition={"from_text": True})]


def build_discard_energy_scaled_damage(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("optional_cost", text, cost={"op": "discard_attached_energy", "target": "self.this_pokemon", "amount": amount_up_to(int_group(m, "count", 2)), "source_text": text}),
        step("set_attack_damage_from_cost_count", text, amount_per_discarded=int_group(m, "amount", 0)),
    ]


def build_discard_random_and_deck_top(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("discard_cards", text, player="opponent", source="opponent.hand", amount=1, random=True),
        step("discard_cards", text, player="opponent", source="opponent.deck", destination="opponent.discard", amount=1, position="top"),
    ]


def build_flip_coin_prevent_all(m: re.Match[str], text: str) -> list[Step]:
    return [
        step("coin_flip", text, player="self", count=1, target_id="coin_result"),
        step("branch_on_result", text, result_ref="coin_result", **{"if": "heads"}, then=[{"op": "register_prevention_effect", "target": "self.this_pokemon", "prevents": {"damage": True, "effects": True}, "duration": "opponent_next_turn", "source_text": text}]),
    ]


def build_shuffle_opponent_deck(m: re.Match[str], text: str) -> list[Step]:
    return [step("shuffle_deck", text, player="opponent")]

def build_templates() -> list[TextTemplate]:
    flags = re.I
    templates: list[TextTemplate] = [
        # v0.5: broad remaining long-tail templates, kept early so they win before generic rules.
        # v0.6: current top leftovers after 57% coverage.
        TextTemplate("topdeck-look-choose-hand", "topdeck_look_choose_or_reorder", re.compile(r"Look at the top (?P<amount>\d+) cards of your deck\. You may reveal a Pokémon you find there and put it into your hand\. Shuffle the other cards back into your deck\.?,?", flags), build_topdeck_look_choose, priority=16),
        TextTemplate("potion-energy-rule", "energy_provides_or_type_modifier", re.compile(r"(?:Rules: )?If you play this card from your hand, remove (?P<amount>\d+) damage counter from the Pokémon you attach it to, if it has any\. .+? provides .+? energy\. \(Doesn'?t count as a basic Energy card\.\)", flags), build_energy_provides_rule, priority=16),
        TextTemplate("tyrogue-evolve-remove-counters", "evolution_devolution_or_levelup", re.compile(r"Once during your turn \(before your attack\), you may put .+? from your hand onto Tyrogue \(this counts as evolving Tyrogue\) and remove all damage counters from Tyrogue\.?,?", flags), build_tyrogue_evolution, priority=16),
        TextTemplate("coin-counters-until-near-ko", "damage_counter_mass_placement", re.compile(r"Flip a coin\. If heads, put damage counters on the Defending Pokémon until it is (?P<hp>\d+) HP away from being Knocked Out\.?,?", flags), build_damage_counter_until_near_ko, priority=16),
        TextTemplate("coin-bench-damage", "weakness_resistance_global_rule", re.compile(r"Flip (?P<coin_count>\d+|a) coins?\. (?:For each heads, )?this attack does (?P<amount>\d+) damage (?:times the number of heads )?to each of your opponent's Benched Pokémon\. \(Don'?t apply Weakness and Resistance for Benched Pokémon\.\)", flags), build_coin_bench_damage, priority=16),
        TextTemplate("coin-bench-damage-heads-tails", "weakness_resistance_global_rule", re.compile(r"Flip a coin\. If heads, this attack does (?P<amount>\d+) damage to each of your opponent's Benched Pokémon\. If tails, this attack does (?P<amount2>\d+) damage to each of your (?:own )?Benched Pokémon\. \(Don'?t apply Weakness and Resistance for Benched Pokémon\.\)", flags), build_coin_bench_damage, priority=16),
        TextTemplate("chosen-energy-hand-damage-topdeck", "damage_scaling_or_conditional_damage", re.compile(r"Choose as many Energy cards from your hand as you like and show them to your opponent\. This attack does (?P<amount>\d+) damage times the number of Energy cards you chose\. Put those Energy cards on top of your deck\. Shuffle your deck afterward\.?,?", flags), build_energy_hand_damage_deck, priority=16),
        TextTemplate("lvx-play-gust-trigger", "gust_before_or_during_damage", re.compile(r"Once during your turn \(before your attack\), when you put .+? LV\.X from your hand onto your Active .+?, you may switch the Defending Pokémon with 1 of your opponent's Benched Pokémon\.?,?", flags), build_lv_x_gust_trigger, priority=16),
        TextTemplate("active-item-trainer-lock", "opponent_item_supporter_trainer_lock", re.compile(r"As long as this Pokémon is your Active Pokémon, your opponent can't play any (?:Item|Trainer) cards from (?:his or her|their) hand\.?,?", flags), build_item_trainer_lock_active, priority=16),
        TextTemplate("switch-self-then-opponent", "switch_or_gust", re.compile(r"Once during your turn \(before your attack\), you may switch your Active Pokémon with 1 of your Benched Pokémon\. If you do, your opponent switches (?:his or her|their) Active Pokémon with 1 of (?:his or her|their) Benched Pokémon\.?,?", flags), build_switch_both_active, priority=16),
        TextTemplate("garbotoxin-like-ability-lock", "ability_or_pokemon_power_lock", re.compile(r"If this Pokémon has a Pokémon Tool card attached to it, each Pokémon in play, in each player's hand, and in each player's discard pile has no Abilities \(except for .+?\)\.?,?", flags), build_garbotoxin_like_ability_lock, priority=16),
        TextTemplate("pokemon-tool-return", "trainer_tool_stadium_rules", re.compile(r"(?:As often as you like during your turn \(before your attack\), you may|Once during your turn \(before your attack\), you may) put a Pokémon Tool card attached to (?:1 of your Pokémon|this Pokémon) into your hand\.?,?", flags), build_tool_return_to_hand, priority=16),
        TextTemplate("conditional-bonus-damage-general", "damage_scaling_or_conditional_damage", re.compile(r"If .+?, this attack does (?P<amount>\d+) more damage\.?,?", flags), build_conditional_bonus_damage_general, priority=16),
        TextTemplate("active-damage-reduction-general", "damage_prevention_or_reduction", re.compile(r"(?:As long as .+?, )?(?:Any damage done to this Pokémon by attacks from .+?|all of your Pokémon take|your opponent's Active Pokémon's attacks do|it takes|attacks used by your opponent's Active Pokémon do) (?P<amount>\d+) less damage.*", flags), build_damage_reduction_general, priority=16),
        TextTemplate("attack-cost-tax-general", "attack_cost_modification", re.compile(r"(?:As long as .+?, )?.+? attacks (?:used by .+? )?cost Colorless (?:more|less)\.?,?", flags), build_attack_cost_tax_general, priority=16),
        TextTemplate("heal-and-remove-all-status", "healing_or_damage_counter_removal", re.compile(r"Heal (?P<amount>\d+) damage and remove all Special Conditions from this Pokémon\.?,?", flags), build_heal_and_remove_all_status, priority=16),
        TextTemplate("setup-active-from-hand", "setup_rule", re.compile(r"If this Pokémon is in your hand when you are setting up to play, you may put it face down as your Active Pokémon\.?,?", flags), build_setup_active_rule, priority=16),
        TextTemplate("no-weakness-general", "weakness_resistance_global_rule", re.compile(r"(?:Your Pokémon in play have no Weakness|If there is any Stadium card in play, this Pokémon has no Weakness)\.?,?", flags), build_no_weakness_general, priority=16),
        TextTemplate("copy-attacks-extended", "copy_or_grant_attack_access", re.compile(r"This Pokémon can use the attacks of .+?\. \(You still need the necessary Energy to use each attack\.\)", flags), build_copy_attack_extended, priority=16),
        TextTemplate("defending-cant-be-healed", "global_stat_or_type_rule_modifier", re.compile(r"The Defending Pokémon can't be healed during your opponent's next turn\.?,?", flags), build_cannot_be_healed, priority=16),
        TextTemplate("attack-requires-condition", "conditional_attack_availability", re.compile(r"This attack can be used if this Pokémon is Asleep\. If it is not Asleep, this attack does nothing\.?,?", flags), build_attack_requires_condition, priority=16),
        TextTemplate("search-shuffle-put-top", "search_deck", re.compile(r"Once during your turn \(before your attack\), you may search your deck for a card, shuffle your deck, then put that card on top of it\.?,?", flags), build_search_shuffle_topdeck, priority=16),
        TextTemplate("bench-capacity-rule", "bench_capacity_rule", re.compile(r"If all of your Pokémon in play are .+? type, you can have up to (?P<amount>\d+) Pokémon on your Bench, and you can't put non-.+? Pokémon into play\. \(If this Ability stops working, discard Pokémon from your Bench until you have \d+\.\)", flags), build_bench_capacity_rule, priority=16),
        TextTemplate("prize-replacement-fewer-prizes", "delayed_knockout_or_prize_bonus", re.compile(r"If .+? is Knocked Out by damage from an attack from your opponent's .+?, .+? takes (?P<amount>\d+) fewer Prize card\.?.*", flags), build_prize_replacement_rule, priority=16),
        TextTemplate("discard-energy-scaled-damage", "energy_discard_move_or_bounce", re.compile(r"Discard up to (?P<count>\d+) Energy cards from this Pokémon, and this attack does (?P<amount>\d+) damage for each card you discarded in this way\.?,?", flags), build_discard_energy_scaled_damage, priority=16),
        TextTemplate("discard-random-hand-and-topdeck", "opponent_hand_shuffle_disruption", re.compile(r"Discard a random card from your opponent's hand\. Discard the top card of your opponent's deck\.?,?", flags), build_discard_random_and_deck_top, priority=16),
        TextTemplate("coin-prevent-all-effects-damage", "damage_prevention_or_reduction", re.compile(r"Flip a coin\. If heads, during your opponent's next turn, prevent all effects of attacks, including damage, done to .+?\.?,?", flags), build_flip_coin_prevent_all, priority=16),
        TextTemplate("shuffle-opponent-deck", "draw_shuffle_hand", re.compile(r"Shuffle your opponent's deck\.?,?", flags), build_shuffle_opponent_deck, priority=16),
        TextTemplate("coin-flip-replacement", "coin_flip_replacement", re.compile(r"Whenever your opponent flips a coin during (?:his or her|their) next turn, treat it as tails\.?,?", flags), build_coin_control, priority=18),
        TextTemplate("coin-damage-status-flex", "coin_damage_plus_special_condition", re.compile(r"Flip (?P<coin_count>\d+|a) coins?\. This attack does (?P<amount>\d+) damage times? the number of heads\. If (?:you get (?P<threshold>\d+) or more heads|either of the coins is heads), .+? is now (?P<status>Asleep|Burned|Confused|Paralyzed|Poisoned).*", flags), build_coin_damage_status_flex, priority=18),
        TextTemplate("choose-n-damage-ignore-wr", "weakness_resistance_global_rule", re.compile(r"Choose (?P<target_count>\d+) of your opponent's Pokémon\. This attack does (?P<amount>\d+) damage to each of those Pokémon\. \(Don'?t apply Weakness and Resistance for Benched Pokémon\.\)\.?", flags), build_choose_damage_ignore_wr, priority=18),
        TextTemplate("self-damage-ignore-wr", "weakness_resistance_global_rule", re.compile(r"Does (?P<amount>\d+) damage to 1 of your Pokémon, and don'?t apply Weakness and Resistance to this damage\.?,?", flags), build_weakness_resistance_damage, priority=18),
        TextTemplate("status-basic-or-attacker", "special_conditions", re.compile(r"(?:If .+?, )?(?P<target>the Defending Pokémon|your opponent's Active Pokémon|this Attacking Pokémon|the Attacking Pokémon|Magby|this Pokémon) is now (?P<status>Asleep|Burned|Confused|Paralyzed|Poisoned)\.?,?", flags), build_status_condition_simple, priority=18),
        TextTemplate("opponent-pokepower-lock-next-turn", "ability_or_pokemon_power_lock", re.compile(r"During your opponent's next turn, your opponent can't use any Poké-Powers on (?:his or her|their) Pokémon\.?,?", flags), build_power_lock_next_turn, priority=18),
        TextTemplate("imprison-marker-power-lock", "pokemon_power_global_lock", re.compile(r"Once during your turn \(before your attack\), if .+? is your Active Pokémon, you may put an Imprison marker on 1 of your opponent's Pokémon\. Any Pokémon that has any Imprison markers on it can't use any Poké-Powers or Poké-Bodies\. This power can't be used if .+? is affected by a Special Condition\.?,?", flags), build_marker_power_lock, priority=18),
        TextTemplate("level-x-previous-type-or-power", "evolution_devolution_or_levelup", re.compile(r"(?:Arceus LV\. X's type is the same type as its previous Level|Put this card onto your Active .+?\. .+? LV\.X can use any attack, Poké-Power, or Poké-Body from its previous level)\.?,?", flags), build_previous_level_rule, priority=18),
        TextTemplate("bench-play-damage-counter-trigger", "damage_counters", re.compile(r"During your opponent's next turn, when your opponent puts a Basic Pokémon from (?:his or her|their) hand onto (?:his or her|their) Bench, put (?P<amount>\d+) damage counters? on that Pokémon\.?,?", flags), build_when_bench_damage_counter_trigger, priority=18),
        TextTemplate("attack-availability-coin-tax", "conditional_attack_availability", re.compile(r"If the Defending Pokémon tries to attack during your opponent's next turn, your opponent flips a coin\. If tails, this attack does nothing\.?,?", flags), build_attack_availability_coin_tax, priority=18),
        TextTemplate("return-self-and-attached", "self_shuffle_or_return_to_deck", re.compile(r"(?:Once during your turn(?: \(before your attack\))?, you may |You may |Shuffle )(?P<target>this Pokémon|this Pokémon and all attached cards|this Pokémon and all cards attached to it)(?: and all cards attached to it| and all attached cards)? (?:into your hand|to your hand|into your deck)\.?,?", flags), build_self_or_attached_return, priority=18),
        TextTemplate("active-no-retreat", "retreat_cost_modification", re.compile(r"As long as this Pokémon is your Active Pokémon, your opponent's Active Pokémon can't retreat\.?,?", flags), build_no_retreat_active, priority=18),
        TextTemplate("once-turn-search-general", "search_deck", re.compile(r"Once during your turn \(before your attack\), you may search your deck for (?P<filter>.+?), reveal it, and put it into your hand\. Then, shuffle your deck\.?,?", flags), build_search_once_turn, priority=18),
        TextTemplate("attach-any-energy", "energy_attachment_or_acceleration", re.compile(r"Attach (?:(?P<amount>\d+|up to \d+) )?(?:any number of |up to \d+ )?(?:Basic |basic |Water |Fire |Grass |Darkness |Lightning )?Energy cards? from your (?:hand|discard pile) to your Pokémon in any way you like\.?,?", flags), build_attach_any_energy, priority=18),
        TextTemplate("ability-draw-with-discard-cost", "draw_shuffle_hand", re.compile(r"You must discard (?P<discard_amount>\d+|a) cards? from your hand in order to use this Ability\. Once during your turn, you may draw (?P<draw_amount>\d+|a) cards?\.?,?", flags), build_ability_draw_cost, priority=18),
        TextTemplate("fixed-damage-counter-placement", "damage_counters", re.compile(r"Put (?P<amount>\d+) damage counters? on (?P<target>1 of your Pokémon|your opponent's Active Pokémon|1 of your opponent's Pokémon)\.?,?", flags), build_damage_counter_fixed, priority=18),
        TextTemplate("special-condition-immunity-general", "special_condition_global_immunity_or_recovery", re.compile(r"(?:This Pokémon|Each of your Pokémon .+?|Zangoose) can't be affected by any Special Conditions\.(?: \(Remove any Special Conditions affecting .+?\)\.)?", flags), build_special_condition_immunity_general, priority=18),
        TextTemplate("prevent-effects-except-damage", "damage_prevention_or_reduction", re.compile(r"(?:During your opponent's next turn, )?Prevent all effects of (?:your opponent's )?attacks, except damage, done to this Pokémon(?: during your opponent's next turn)?\.?,?", flags), build_prevent_effects_except_damage, priority=18),
        TextTemplate("attacks-unaffected", "weakness_resistance_global_rule", re.compile(r"Damage from this Pokémon's attacks isn't affected by (?:any effects on your opponent's Active Pokémon|Weakness or Resistance)\.?,?", flags), build_damage_unaffected, priority=18),
        TextTemplate("conditional-knockout", "delayed_knockout_or_prize_bonus", re.compile(r"If your opponent's Active Pokémon is an Ultra Beast, it is Knocked Out\.?,?", flags), build_knockout_condition, priority=18),
        TextTemplate("optional-extra-damage-then-lock", "attack_lock_or_attack_tax", re.compile(r"You may do (?P<amount>\d+) more damage\. If you do, during your next turn, this Pokémon can't attack\.?,?", flags), build_extra_damage_then_lock, priority=18),
        TextTemplate("discard-energy-and-opponent-energy", "energy_discard_move_or_bounce", re.compile(r"You may discard an Energy from this Pokémon\. If you do, discard an Energy from your opponent's Active Pokémon\.?,?", flags), build_discard_energy_and_discard_opponent, priority=18),
        TextTemplate("type-modifier-in-play", "global_stat_or_type_rule_modifier", re.compile(r"As long as this Pokémon is in play, it is .+? type\.?,?", flags), build_type_modifier, priority=18),

        TextTemplate("discard-top-deck", "mill_or_deck_discard", re.compile(r"(?:Your opponent discards|Discard) the top (?:(?P<amount>\d+) )?cards? of (?:his or her|their|your opponent's) deck\.?,?", flags), build_discard_top_deck, priority=22),
        TextTemplate("coin-search-any-card", "search_deck", re.compile(r"Flip a coin\. If heads, search your deck for (?:any )?(?P<amount>\d+|a|an)? ?card and put it into your hand\. Shuffle your deck afterward\.?,?", flags), build_coin_search_deck, priority=23),
        TextTemplate("generic-damage-per-formula", "damage_scaling_or_conditional_damage", re.compile(r"(?:Does|This attack does) \d+ damage (?:plus|times|for each|more damage).*", flags), build_generic_damage_formula, priority=24),
        TextTemplate("damage-plus-if-condition", "damage_scaling_or_conditional_damage", re.compile(r"If .+?, this attack does \d+ (?:damage )?plus (?P<amount>\d+) more damage\.?,?", flags), build_damage_plus_condition, priority=24),
        TextTemplate("discard-energy-basic", "energy_discard_move_or_bounce", re.compile(r"Discard (?:(?P<amount>\d+) )?(?:all )?.*Energy attached to .+?\.?,?", flags), build_discard_energy, priority=24),
        TextTemplate("discard-energy-for-bonus", "energy_discard_move_or_bounce", re.compile(r"You may discard .*Energy attached to this Pokémon\. If you do, this attack does (?P<amount>\d+) more damage\.?,?", flags), build_discard_energy_for_bonus, priority=24),
        TextTemplate("no-retreat-condition", "retreat_cost_modification", re.compile(r"(?:If .+?, )?(?:this Pokémon|Your Basic Pokémon in play|your opponent's Active Pokémon|Your opponent's Active Pokémon|.+?) (?:has|have) no Retreat Cost\.?,?", flags), build_no_retreat_condition, priority=24),
        TextTemplate("retreat-lock-and-power-lock", "retreat_cost_modification", re.compile(r"(?:The Defending Pokémon|As long as this Pokémon is your Active Pokémon, your opponent's Active Pokémon|Your opponent's Active Pokémon) can't retreat(?: or use any Poké-Powers)? during your opponent's next turn\.?,?", flags), build_retreat_lock, priority=24),
        TextTemplate("heal-all-damage", "healing_or_damage_counter_removal", re.compile(r"Heal all damage from (?:all of your Pokémon|\d+ of your Benched Pokémon)\.?,?", flags), build_heal_all, priority=24),
        TextTemplate("remove-all-damage-counters-trigger", "healing_or_damage_counter_removal", re.compile(r"Once during your turn \(before your attack\), when you put .+? from your hand onto .+?, you may remove all damage counters from .+?\.?,?", flags), build_remove_all_damage_counters, priority=24),
        TextTemplate("play-evolve-trigger", "evolution_devolution_or_levelup", re.compile(r"When you play this Pokémon from your hand (?:to evolve 1 of your Pokémon|onto your Bench)(?: during your turn)?, you may .+", flags), build_play_evolve_trigger, priority=24),
        TextTemplate("old-power-energy-action", "old_pokemon_power_status_gated", re.compile(r"As often as you like during your turn \(before your attack\), (?:you may )?(?:attach|move) .+?Energy .+?\..*?This power can't be used if .+?(?:is|is affected by) (?:Asleep, Confused, or Paralyzed|a Special Condition)\.?,?", flags), build_old_power_energy_action, priority=24),
        TextTemplate("status-immunity-self", "special_condition_global_immunity_or_recovery", re.compile(r"(?:Zangoose can't|This Pokémon can't|Each of your Pokémon .+? can't) be affected by any Special Conditions\.?(?: \(Remove any Special Conditions affecting .+?\)\.?)?", flags), build_status_immunity, priority=24),
        TextTemplate("attack-lock-choice", "conditional_attack_availability", re.compile(r"Choose 1 of the Defending Pokémon's attacks\. That Pokémon can use only that attack during your opponent's next turn\.?,?", flags), build_attack_lock, priority=24),
        TextTemplate("trainer-hand-count-damage", "trainer_tool_stadium_rules", re.compile(r"Look at your opponent's hand\. This attack does (?P<amount>\d+) damage times the number of Trainer, Supporter, and Stadium cards in your opponent's hand\.?,?", flags), build_hand_or_trainer_count_damage, priority=24),
        TextTemplate("choose-opponent-attack-copy", "copy_or_grant_attack_access", re.compile(r"Choose 1 of your opponent's Pokémon's attacks and use it as this attack\.?,?", flags), build_choose_attack_copy, priority=24),
        TextTemplate("dynamic-damage-counter-placement", "damage_counter_mass_placement", re.compile(r"(?:Count the number of .+?\. )?Put that many damage counters on .+?\.?,?|For each card in your opponent's hand, put 1 damage counter on their Active Pokémon\.?,?", flags), build_put_damage_counter_dynamic, priority=24),
        TextTemplate("coin-damage-only", "damage_scaling_or_conditional_damage", re.compile(r"Flip (?P<coin_count>\d+|a) coins?\. This attack does (?P<amount>\d+) (?:damage )?times? the number of heads\.?,?", flags), build_coin_damage_only, priority=29),
        TextTemplate("future-damage-bonus", "future_damage_bonus_or_mark", re.compile(r"During your next turn, if an attack does damage to the Defending Pokémon .* that attack does (?P<amount>\d+) more damage\.?,?", flags), build_future_damage_bonus, priority=32),
        TextTemplate("future-named-attack-bonus", "future_damage_bonus_or_mark", re.compile(r"During your next turn, .+? does \d+ damage plus (?P<amount>\d+) more damage\.?,?", flags), build_future_damage_bonus, priority=33),
        TextTemplate("opponent-hand-shuffle-disruption", "opponent_hand_shuffle_disruption", re.compile(r"(?:Flip a coin\. If heads, )?choose (?P<amount>\d+) random cards? from your opponent's hand\. Your opponent reveals those cards and shuffles them into (?:his or her|their) deck\.?,?", flags), build_opponent_hand_shuffle, priority=34),
        TextTemplate("opponent-hand-refresh", "opponent_hand_shuffle_disruption", re.compile(r"(?:Rules: )?Your opponent shuffles (?:his or her|their) hand into (?:his or her|their) deck, then draws (?P<amount>\d+) cards\.?,?", flags), build_opponent_hand_shuffle, priority=35),
        TextTemplate("recycle-discard-to-deck", "self_recovery_or_recycle_from_discard", re.compile(r"Shuffle up to (?P<amount>\d+) .+? from your discard pile into your deck\.?,?", flags), build_recycle_to_deck, priority=36),
        TextTemplate("move-energy-general", "energy_discard_move_or_bounce", re.compile(r"(?:You may )?(?:move|Move) (?:all |\d+ |a |an )?.*Energy(?: card)? attached to .+? to .+?\.?", flags), build_energy_move_general, priority=37),
        TextTemplate("attach-energy-general", "energy_attachment_or_acceleration", re.compile(r"(?:As often as you like during your turn \(before your attack\), you may |Once during your turn \(before your attack\), you may |Flip \d+ coins\. For each heads, )?attach (?:a |\d+ )?.*Energy(?: card)? from your (?:hand|discard pile) to .+?\.?.*", flags), build_attach_energy_general, priority=38),
        TextTemplate("heal-each-player-turn", "healing_or_damage_counter_removal", re.compile(r"Once during each player's turn, if that player has \d+ Pokémon in play, (?:he or she|they) may heal (?P<amount>\d+) damage from each of (?:his or her|their) Pokémon\.?,?", flags), build_heal_each_player, priority=39),
        TextTemplate("turn-action-draw", "draw_shuffle_hand", re.compile(r"Once during each player's turn, if they played .+? from their hand this turn, they may draw (?P<amount>\d+) cards\.?,?", flags), build_draw_turn_action, priority=39),
        TextTemplate("draw-until-fixed", "draw_shuffle_hand", re.compile(r"Once during your turn \(before your attack\), you may draw cards until you have (?P<amount>\d+) cards in your hand\.?,?", flags), build_draw_until, priority=39),
        TextTemplate("draw-until-matching-board", "draw_shuffle_hand", re.compile(r"Once during each player's turn, that player may discard an Energy card from their hand in order to draw cards until they have as many cards in their hand as they have .+? in play\.?,?", flags), build_draw_until, priority=39),
        TextTemplate("prevention-effects", "damage_prevention_or_reduction", re.compile(r"Prevent all (?:damage|effects|effects of attacks, including damage).* done to .+? by .+?\.?,?", flags), build_prevention, priority=39),
        TextTemplate("special-condition-conditional", "special_conditions", re.compile(r"(?:If .+?, )?(?:the Defending Pokémon|your opponent's Active Pokémon|this Pokémon|the Attacking Pokémon) is now (?P<status>Asleep|Burned|Confused|Paralyzed|Poisoned)\.?.*", flags), build_status_condition, priority=39),
        TextTemplate("bench-only-fossil", "bench_from_discard_or_deck", re.compile(r"Put this card onto your Bench only with the effect of .+", flags), build_bench_only_rule, priority=39),
        TextTemplate("return-pokemon-to-hand", "self_shuffle_or_return_to_deck", re.compile(r"(?:Once during your turn \(before your attack\), you may |Put )(?P<target>this Pokémon|1 of your Pokémon)(?: and all cards attached to it| and all attached cards)? (?:to|into|return .* to) your hand\.?,?", flags), build_return_to_hand, priority=39),
        TextTemplate("move-damage-counter", "damage_counters", re.compile(r"Move (?P<amount>\d+) damage counters? from .+? to .+?\.?,?", flags), build_damage_counter_move, priority=39),
        TextTemplate("damage-counter-trigger", "damage_counter_reflect_or_retaliation", re.compile(r"(?:If .+? is damaged by an opponent's attack.*|After your opponent's Pokémon uses a Poké-Power, )put (?P<amount>\d+) damage counters? on .+?\.?,?", flags), build_damage_counter_trigger, priority=39),
        TextTemplate("tool-retreat-replacement", "tool_or_attachment_return_replacement", re.compile(r"When the Pokémon .+? is attached to retreats, discard .+? instead of discarding Energy cards\.?,?", flags), build_tool_retreat_replacement, priority=39),
        TextTemplate("attack-cost-modifier", "attack_cost_modification", re.compile(r"If you have more Prize cards remaining than your opponent, attacks used by the Pokémon this card is attached to cost Colorless less\.?,?", flags), build_attack_cost_modifier, priority=39),
        TextTemplate("attack-damage-modifier", "tool_or_attachment_damage_modifier", re.compile(r"(?:Attacks used by|The attacks of|Each of) .+? do(?:es)? (?P<amount>\d+) more damage .+", flags), build_attack_damage_modifier, priority=39),
        TextTemplate("self-no-weakness", "weakness_resistance_global_rule", re.compile(r"(?:Each of your Pokémon|Each of your Pokémon that has .+?|If you have .+?, each of your Pokémon) has no Weakness\.?,?", flags), build_self_weakness_rule, priority=39),
        TextTemplate("heal-damage", "healing_or_damage_counter_removal", re.compile(r"Heal (?P<amount>\d+) damage from (?P<target>.+?)\.?,?", flags), build_heal_damage, priority=10),
        TextTemplate("heal-and-remove-status", "healing_or_damage_counter_removal", re.compile(r"Heal (?P<amount>\d+) damage and remove a Special Condition from (?P<target>.+?)\.?,?", flags), build_heal_damage, priority=11),
        TextTemplate("remove-damage-counters", "healing_or_damage_counter_removal", re.compile(r"Remove (?P<amount>\d+) damage counters? from (?P<target>.+?)\.?,?", flags), build_remove_damage_counters, priority=12),
        TextTemplate("mass-damage-counters", "damage_counter_mass_placement", re.compile(r"(?:Flip a coin\. If heads, )?put (?P<amount>\d+) damage counters? on (?P<target>each of your opponent's (?:Benched )?Pokémon|each Pokémon|the Defending Pokémon)\.?,?", flags), build_mass_damage_counters, priority=20),
        TextTemplate("damage-ignore-wr", "weakness_resistance_global_rule", re.compile(r"(?:Does|This attack does) (?P<amount>\d+) damage to (?P<target>each Pokémon.+?|1 of your opponent's Pokémon.+?|that Pokémon.+?)\. Don'?t apply Weakness (?:and|or) Resistance(?: for this attack)?\.?.*", flags), build_weakness_resistance_damage, priority=25),
        TextTemplate("weakness-resistance", "weakness_resistance_global_rule", re.compile(r"Don'?t apply Weakness (?:and|or) Resistance\.?,?", flags), build_weakness_resistance_rule, priority=26),
        TextTemplate("coin-damage-status", "coin_damage_plus_special_condition", re.compile(r"Flip (?P<coin_count>\d+) coins?\. This attack does (?P<amount>\d+) damage times? the number of heads\. If (?:either of the coins is heads|you get (?P<threshold>\d+) or more heads), (?:the Defending Pokémon|[A-Za-z' -]+) is now (?P<status>Asleep|Burned|Confused|Paralyzed|Poisoned).*", flags), build_coin_damage_status, priority=30),
        TextTemplate("coin-conditional-damage-or-status", "damage_scaling_or_conditional_damage", re.compile(r"Flip a coin\. If heads, this attack does \d+ damage plus (?P<heads_bonus>\d+) more damage\. If tails, this attack does \d+ damage and the Defending Pokémon is now (?P<tails_status>Asleep|Burned|Confused|Paralyzed|Poisoned)\.?,?", flags), build_conditional_damage_status, priority=31),
        TextTemplate("search-deck-each-player", "search_deck", re.compile(r"Once during each player's turn, that player may search their deck for (?P<filter>.+?), reveal it, and put it into their hand\. Then, that player shuffles their deck\.?,?", flags), build_search_deck, priority=40),
        TextTemplate("search-deck-general", "search_deck", re.compile(r"Search your deck for (?P<filter>.+?)(?:, reveal (?:it|them), and put (?:it|them) into your hand| and put .* onto your Bench| and attach .*|\.)(?:.*Shuffle your deck(?: afterward)?\.?)?", flags), build_search_deck, priority=41),
        TextTemplate("castform-form-switch", "castform_form_switch_search", re.compile(r"Once during your turn \(before your attack\), you may search your deck for .+? and switch it with .+?\. \(Any cards attached to .+?, damage counters, Special Conditions, and effects on it are now on the new Pokémon\.\) Shuffle .+? back into your deck\. You can't use more than 1 Temperamental Weather Poké-Power each turn\.?,?", flags), build_castform_switch, priority=45),
        TextTemplate("player-turn-switch", "switch_or_gust", re.compile(r"Once during each player's turn, that player may switch their Active .+? Pokémon with 1 of their Benched .+? Pokémon\.?,?", flags), build_switch, priority=50),
        TextTemplate("bench-self-switch", "switch_or_gust", re.compile(r"Once during your turn \(before your attack\), if this Pokémon is on your Bench, you may switch this Pokémon with your Active Pokémon\.?,?", flags), build_switch, priority=51),
        TextTemplate("gust-before-damage", "gust_before_or_during_damage", re.compile(r"(?:Before doing damage, you may|Switch) .+?opponent's Benched Pokémon.+?(?:If you do, this attack does (?P<amount>\d+) damage to the new Defending Pokémon\.)?.*", flags), build_gust_damage, priority=52),
        TextTemplate("energy-move-trigger", "energy_discard_move_or_bounce", re.compile(r"When your Active Pokémon is Knocked Out by damage from an attack from your opponent's Pokémon, you may move a Basic Energy from that Pokémon to the Pokémon this card is attached to\.?,?", flags), build_energy_move_trigger, priority=60),
        TextTemplate("opponent-attack-damage-reduction", "opponent_attack_damage_reduction", re.compile(r"(?:As long as .+? is in the Active Spot, attacks used by your opponent's Active Pokémon do (?P<amount>\d+) less damage|During your opponent's next turn, any damage done to .+? by attacks is reduced by (?P<amount2>\d+)).*", flags), build_damage_reduction, priority=70),
        TextTemplate("special-condition-global", "special_condition_global_immunity_or_recovery", re.compile(r"Each Pokémon that has any Energy attached \(both yours and your opponent's\) recovers from all Special Conditions and can't be affected by any Special Conditions\.?,?", flags), build_global_rule("special_condition_global_immunity_or_recovery"), priority=80),
        TextTemplate("ability-lock", "ability_or_pokemon_power_lock", re.compile(r"(?:Each player's|Each Basic|All) .+?(?:can't use any Poké-Powers or Poké-Bodies|has no Abilities|have no Abilities|Pokémon Powers stop working).*", flags), build_global_rule("ability_or_pokemon_power_lock"), priority=85),
        TextTemplate("global-retreat-cost", "global_retreat_cost_rule", re.compile(r"(?:Each player pays .+? to retreat .+|The Retreat Cost of each Pokémon in play .+|As long as .+? is in play, each player must pay .+? to retreat .+)", flags), build_global_rule("global_retreat_cost_rule"), priority=90),
        TextTemplate("attached-retreat-cost", "retreat_cost_modification", re.compile(r"The Retreat Cost of the Pokémon this card is attached to is .+?(?:less|more|no Retreat Cost).+", flags), build_global_rule("retreat_cost_modification"), priority=91),
        TextTemplate("devolution", "evolution_devolution_or_levelup", re.compile(r"If your opponent has any Evolved Pokémon in play, remove the highest Stage Evolution card from each of them and put those cards back into (?:his or her|their) hand\.?,?", flags), build_devolution, priority=100),
        TextTemplate("copy-or-grant-attack", "copy_or_grant_attack_access", re.compile(r"(?:Attach this card to 1 of your Pokémon in play\. That Pokémon may use this card's attack instead of its own|.+ can use the attacks? of .+ as its own).*", flags), build_copy_attack, priority=110),
    ]
    return sorted(templates, key=lambda t: t.priority)
