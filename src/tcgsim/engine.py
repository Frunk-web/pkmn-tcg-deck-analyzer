from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .state import GameState


class RuntimeEngine:
    """Minimal execution engine for compiled Pokémon TCG effect steps.

    This is intentionally conservative. It executes high-frequency simple ops and logs
    unsupported or ambiguous ops instead of pretending to know the full TCG rules.
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = strict
        self.handlers: Dict[str, Callable[[GameState, Dict[str, Any], Dict[str, Any]], None]] = {
            "reference_global_rule": self.handle_reference_global_rule,
            "draw_cards": self.handle_draw_cards,
            "search_deck": self.handle_search_deck,
            "shuffle_deck": self.handle_shuffle_deck,
            "move_card": self.handle_move_card,
            "attach_card": self.handle_attach_card,
            "attach_cards": self.handle_attach_cards,
            "deal_attack_damage": self.handle_deal_attack_damage,
            "deal_damage": self.handle_deal_damage,
            "modify_attack_damage": self.handle_modify_attack_damage,
            "place_damage_counters": self.handle_place_damage_counters,
            "heal_damage": self.handle_heal_damage,
            "apply_special_condition": self.handle_apply_special_condition,
            "switch_active": self.handle_switch_active,
            "switch_active_with_bench": self.handle_switch_active_with_bench,
            "coin_flip": self.handle_coin_flip,
            "coin_flip_until": self.handle_coin_flip_until,
            "branch_on_result": self.handle_branch_on_result,
            "conditional": self.handle_conditional,
            "choose_target": self.handle_choose_target,
            "choose_cards": self.handle_choose_cards,
            "choose_amount": self.handle_choose_amount,
            "register_continuous_modifier": self.handle_register_continuous_modifier,
            "register_trigger": self.handle_register_trigger,
            "register_replacement_effect": self.handle_register_replacement_effect,
            "register_knockout_prize_rule": self.handle_register_knockout_prize_rule,
            "set_prize_cards_taken_for_knockout": self.handle_register_knockout_prize_rule,
            "declare_attack": self.handle_declare_attack,
            "attack_does_nothing": self.handle_attack_does_nothing,
            "remember_card_instance": self.handle_remember_card_instance,
            "reveal_zone": self.handle_reveal_zone,
            "count_cards": self.handle_count_cards,
            "register_retreat_lock": self.handle_register_generic_continuous,
            "register_attack_prevention": self.handle_register_generic_continuous,
            "register_damage_reduction": self.handle_register_generic_continuous,
            "for_each_player": self.handle_for_each_player,
            "optional_discard_cards": self.handle_optional_discard_cards,
            "draw_until_hand_size": self.handle_draw_until_hand_size,
            "draw_until_hand_size_matches": self.handle_draw_until_hand_size_matches,
            "move_cards": self.handle_move_cards,
            "move_zone_to_zone": self.handle_move_zone_to_zone,
            "discard_cards": self.handle_discard_cards,
            "discard_attached_energy": self.handle_discard_attached_energy,
            "play_condition": self.handle_play_condition,
            "register_play_condition": self.handle_register_play_condition,
            "look_at_top_cards": self.handle_look_at_top_cards,
            "remove_special_condition": self.handle_remove_special_conditions,
            "remove_special_conditions": self.handle_remove_special_conditions,
            "discard_attached_cards": self.handle_discard_attached_cards,
            "move_energy": self.handle_move_energy,
            "put_card_into_hand": self.handle_put_card_into_hand,
            "put_card_on_bench": self.handle_put_card_on_bench,
            "reveal_cards": self.handle_reveal_cards,
            "shuffle_cards": self.handle_shuffle_cards,
            "reveal_hand": self.handle_reveal_hand,
            "reorder_cards": self.handle_reorder_cards,
            "grant_attack_from_attached_card": self.handle_grant_attack_from_attached_card,
            "play_trainer_as_pokemon": self.handle_play_trainer_as_pokemon,
            "ignore_weakness_resistance": self.handle_ignore_weakness_resistance,
            "ignore_effects_on_defending_pokemon": self.handle_ignore_effects_on_defending_pokemon,
            "draw_cards_per_coin_heads": self.handle_draw_cards_per_coin_heads,
            "provide_energy": self.handle_provide_energy,
            "evolve_pokemon_from_hand": self.handle_evolve_pokemon_from_hand,
            "swap_pokemon_card_with_deck_card": self.handle_swap_pokemon_card_with_deck_card,
            "register_usage_limit": self.handle_register_usage_limit,
            "ignore_resistance": self.handle_ignore_resistance,
            "discard_stadium": self.handle_discard_stadium,
            "choose_attack": self.handle_choose_attack,
            "set_attack_damage_from_coin_heads": self.handle_set_attack_damage_from_coin_heads,
            "modify_attack_damage_per_coin_heads": self.handle_modify_attack_damage_per_coin_heads,
            "modify_special_condition": self.handle_modify_special_condition,
            "modify_attack_damage_per_damage_counter": self.handle_modify_attack_damage_per_damage_counter,
            "set_attack_damage_from_count": self.handle_set_attack_damage_from_count,
            "register_evolution_permission": self.handle_register_evolution_permission,
            "register_evolution_exception": self.handle_register_evolution_exception,
            "look_at_cards": self.handle_look_at_cards,
            "modify_attack_damage_from_count": self.handle_modify_attack_damage_from_count,
            "evolve_pokemon": self.handle_evolve_pokemon,
            "discard_attached_energy_per_coin_heads": self.handle_discard_attached_energy_per_coin_heads,
            "set_attack_damage_per_pokemon": self.handle_set_attack_damage_per_pokemon,
            "distribute_damage_counters": self.handle_distribute_damage_counters,
            "discard_card": self.handle_discard_card,
            "reveal_hand_to_player": self.handle_reveal_hand_to_player,
            "ignore_defending_pokemon_damage_modifiers": self.handle_ignore_defending_pokemon_damage_modifiers,
            "set_attack_damage_per_attached_energy": self.handle_set_attack_damage_per_attached_energy,
            "conditional_effect": self.handle_conditional_effect,
            "choose_player": self.handle_choose_player,
            "modify_attack_damage_by_heads_table": self.handle_modify_attack_damage_by_heads_table,
            "conditional_attack_does_nothing": self.handle_conditional_attack_does_nothing,
            "register_deck_construction_rule": self.handle_register_deck_construction_rule,
            "choose_special_condition": self.handle_choose_special_condition,
            "choose_yes_no": self.handle_choose_yes_no,
            "branch_on_choice": self.handle_branch_on_choice,
            "discard_cards_per_coin_heads": self.handle_discard_cards_per_coin_heads,
            "move_damage_counters": self.handle_move_damage_counters,
            "copy_and_use_attack": self.handle_copy_and_use_attack,
            "set_attack_damage_per_damage_counter": self.handle_set_attack_damage_per_damage_counter,
            "attack_does_nothing_if": self.handle_attack_does_nothing_if,
            "set_attack_damage_from_value": self.handle_set_attack_damage_from_value,
            "devolve_pokemon": self.handle_devolve_pokemon,
            "knock_out_pokemon": self.handle_knock_out_pokemon,
            "register_legality_note": self.handle_register_legality_note,
            "deal_damage_per_damage_counter": self.handle_deal_damage_per_damage_counter,
            "move_pokemon_and_attached_cards": self.handle_move_pokemon_and_attached_cards,
            "place_damage_counters_until_remaining_hp": self.handle_place_damage_counters_until_remaining_hp,
        }

    def execute_effect(self, state: GameState, effect: Dict[str, Any], source_instance_id: Optional[str] = None) -> None:
        context: Dict[str, Any] = {
            "effect": effect,
            "source_instance_id": source_instance_id,
            "self": state.turn_player,
            "opponent": state.opponent_of(state.turn_player),
            "choices": {},
        }
        self.autofill_choices(state, effect, context)
        state.log_event(
            "start_effect",
            effect_id=effect.get("effect_id"),
            kind=effect.get("kind"),
            text=effect.get("text"),
            source_instance_id=source_instance_id,
        )
        for step in effect.get("steps", []) or []:
            self.execute_step(state, step, context)
        state.log_event("end_effect", effect_id=effect.get("effect_id"))

    def execute_step(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        op = step.get("op")
        if not op:
            state.log_event("skip_malformed_step", step=step)
            return
        handler = self.handlers.get(op)
        if handler is None:
            self.handle_unknown(state, step, context)
            return
        handler(state, step, context)

    def amount_value(self, amount: Any, default: int = 1, context: Optional[Dict[str, Any]] = None) -> int:
        if amount is None:
            return default
        if isinstance(amount, int):
            return amount
        if isinstance(amount, float):
            return int(amount)
        if isinstance(amount, str):
            try:
                return int(amount)
            except ValueError:
                # Damage strings such as "30+" or "10x" should still contribute
                # their printed numeric base for deterministic smoke/game-flow tests.
                m = __import__("re").search(r"-?\d+", amount)
                return int(m.group(0)) if m else default
        if isinstance(amount, dict):
            mode = amount.get("mode")
            if mode in {"exact", "up_to", "at_least"}:
                try:
                    return int(amount.get("value", default))
                except Exception:
                    return default
            if mode == "count_ref" and context is not None:
                return int(context.get("counts", {}).get(amount.get("count_id"), default))
            # Compiler damage payloads commonly look like:
            # {"printed": "140", "base": 140, "modifier_symbol": null}.
            for key in ("value", "base", "damage", "amount", "count", "heads", "counters"):
                if key in amount:
                    try:
                        return int(amount[key])
                    except Exception:
                        pass
            if "printed" in amount:
                try:
                    return int(str(amount["printed"]).replace("+", "").replace("x", ""))
                except Exception:
                    m = __import__("re").search(r"-?\d+", str(amount["printed"]))
                    return int(m.group(0)) if m else default
        return default

    def player_for_ref(self, ref: Any, context: Dict[str, Any], default: str = "self") -> str:
        if ref in (None, "self", "owner", "controller"):
            return context.get("self", default)
        if ref == "opponent":
            return context.get("opponent", default)
        if isinstance(ref, str) and ref in context:
            return context[ref]
        return context.get(default, context.get("self", "p1"))

    def resolve_instance_ref(self, state: GameState, ref: Any, context: Dict[str, Any]) -> Optional[str]:
        if ref is None:
            return None
        if isinstance(ref, str):
            if ref in state.cards:
                return ref
            if ref.startswith("choice."):
                key = ref.split(".", 1)[1]
                val = context.get("choices", {}).get(key)
                if isinstance(val, list):
                    return val[0] if val else None
                return val
            if ref == "self.active":
                return state.players[context["self"]].active
            if ref == "opponent.active":
                return state.players[context["opponent"]].active
            if ref in {"defending_pokemon", "opponent_active", "target.active"}:
                return state.players[context["opponent"]].active
            if ref in {"attacking_pokemon", "this_pokemon", "source", "self"}:
                return context.get("source_instance_id") or state.players[context["self"]].active
            if ref.startswith("memory."):
                return state.memory.get(ref.split(".", 1)[1])
        return None

    def resolve_instances_from_choice_or_zone(self, state: GameState, ref: Any, context: Dict[str, Any]) -> List[str]:
        """Resolve common card-reference shapes used by the compiler.

        The compiler has evolved over time, so compiled steps may reference cards by
        direct instance id, choice id, memory id, list of ids, or short aliases like
        ``cards``. The runtime is permissive here because this is a smoke-test engine.
        """
        if ref is None:
            return []
        if isinstance(ref, list):
            out: List[str] = []
            for item in ref:
                out.extend(self.resolve_instances_from_choice_or_zone(state, item, context))
            return [iid for iid in out if iid]
        if isinstance(ref, str):
            if ref in state.cards:
                return [ref]
            if ref.startswith("choice."):
                key = ref.split(".", 1)[1]
                val = context.get("choices", {}).get(key)
                if isinstance(val, list):
                    return [iid for iid in val if iid]
                return [val] if val else []
            if ref in context.get("choices", {}):
                val = context.get("choices", {}).get(ref)
                if isinstance(val, list):
                    return [iid for iid in val if iid]
                return [val] if val else []
            if ref.startswith("memory."):
                key = ref.split(".", 1)[1]
                val = state.memory.get(key)
                if isinstance(val, list):
                    return [iid for iid in val if iid]
                return [val] if val else []
            if ref in state.memory:
                val = state.memory.get(ref)
                if isinstance(val, list):
                    return [iid for iid in val if iid]
                return [val] if val else []
        one = self.resolve_instance_ref(state, ref, context)
        return [one] if one else []

    def autofill_choices(self, state: GameState, effect: Dict[str, Any], context: Dict[str, Any]) -> None:
        for choice in effect.get("choices", []) or []:
            cid = choice.get("choice_id")
            if not cid:
                continue
            kind = choice.get("kind", "select_cards")
            amount = self.amount_value(choice.get("amount"), default=1, context=context)
            zones = choice.get("from") or []
            if isinstance(zones, str):
                zones = [zones]
            if kind in {"select_pokemon", "select_card", "select_cards"}:
                candidates = self.find_candidates(state, zones, context)
                selected = candidates[:amount]
                context["choices"][cid] = selected if amount != 1 or kind == "select_cards" else (selected[0] if selected else None)
                state.log_event("auto_choice", choice_id=cid, selected=context["choices"][cid], prompt=choice.get("prompt"))
            elif kind in {"choose_amount", "select_amount"}:
                context["choices"][cid] = amount
                state.log_event("auto_choice", choice_id=cid, selected=amount)

    def find_candidates(self, state: GameState, zones: Iterable[str], context: Dict[str, Any]) -> List[str]:
        out: List[str] = []
        for zone_ref in zones:
            if not isinstance(zone_ref, str):
                continue
            if zone_ref == "self.active":
                active = state.players[context["self"]].active
                if active:
                    out.append(active)
            elif zone_ref == "opponent.active":
                active = state.players[context["opponent"]].active
                if active:
                    out.append(active)
            elif zone_ref == "self.bench":
                out.extend(state.players[context["self"]].bench)
            elif zone_ref == "opponent.bench":
                out.extend(state.players[context["opponent"]].bench)
            elif zone_ref == "self.deck":
                out.extend(state.players[context["self"]].deck)
            elif zone_ref == "opponent.deck":
                out.extend(state.players[context["opponent"]].deck)
            elif zone_ref == "self.hand":
                out.extend(state.players[context["self"]].hand)
            elif zone_ref == "opponent.hand":
                out.extend(state.players[context["opponent"]].hand)
            elif zone_ref == "self.discard":
                out.extend(state.players[context["self"]].discard)
            elif zone_ref == "opponent.discard":
                out.extend(state.players[context["opponent"]].discard)
        return out

    # Handlers
    def handle_reference_global_rule(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        state.log_event("reference_global_rule", rule=step.get("rule") or step.get("rule_id") or step.get("global_rule_id"), step=step)

    def handle_declare_attack(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        state.log_event("declare_attack", step=step)

    def handle_draw_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        amount = self.amount_value(step.get("amount"), default=1, context=context)
        moved = []
        for _ in range(amount):
            if not state.players[player].deck:
                state.log_event("deck_empty_on_draw", player=player)
                break
            iid = state.players[player].deck.pop(0)
            state.players[player].hand.append(iid)
            state.cards[iid].zone = "hand"
            moved.append(iid)
        state.log_event("draw_cards", player=player, amount=amount, drawn=moved)

    def handle_search_deck(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        amount = self.amount_value(step.get("amount"), default=1, context=context)
        selected = state.players[player].deck[:amount]
        key = step.get("result_key") or step.get("search_id") or "last_search"
        state.memory[key] = selected
        state.log_event("search_deck", player=player, amount=amount, selected=selected, filter=step.get("filter"))

    def handle_shuffle_deck(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        state.rng.shuffle(state.players[player].deck)
        state.log_event("shuffle_deck", player=player)

    def handle_move_card(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        ids = self.resolve_instances_from_choice_or_zone(state, step.get("card") or step.get("cards") or step.get("subject"), context)
        to = step.get("to") or step.get("to_zone") or "hand"
        player = self.player_for_ref(step.get("player") or step.get("to_player"), context)
        if isinstance(to, str) and "." in to:
            player_ref, zone = to.split(".", 1)
            player = self.player_for_ref(player_ref, context)
            to = zone
        for iid in ids:
            state.move_instance(iid, player, str(to))
        state.log_event("move_card", moved=ids, to=to, player=player)

    def handle_attach_card(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        self._attach_many(state, self.resolve_instances_from_choice_or_zone(state, step.get("card") or step.get("cards"), context), step, context)

    def handle_attach_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        self._attach_many(state, self.resolve_instances_from_choice_or_zone(state, step.get("cards"), context), step, context)

    def _attach_many(self, state: GameState, card_ids: List[str], step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("to") or step.get("target"), context)
        if target is None:
            target = state.players[context["self"]].active
        attached = []
        for iid in card_ids:
            if iid is None or target is None:
                continue
            # Remove from current public zone then attach.
            c = state.cards[iid]
            if c.zone != "attached":
                try:
                    if c.zone == "active" and state.players[c.controller].active == iid:
                        state.players[c.controller].active = None
                    else:
                        state.players[c.controller].zone_ids(c.zone).remove(iid)
                except Exception:
                    pass
            c.zone = "attached"
            c.controller = state.cards[target].controller
            state.cards[target].attached_cards.append(iid)
            attached.append(iid)
        state.log_event("attach_cards", attached=attached, target=target)

    def handle_deal_attack_damage(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target"), context) or state.players[context["opponent"]].active
        amount = self.amount_value(step.get("damage") or step.get("amount"), default=0, context=context)
        if target:
            state.cards[target].damage_counters += amount // 10
        state.log_event("deal_attack_damage", target=target, damage=amount)

    def handle_deal_damage(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target"), context) or state.players[context["opponent"]].active
        amount = self.amount_value(step.get("damage") or step.get("amount"), default=0, context=context)
        if target:
            state.cards[target].damage_counters += amount // 10
        state.log_event("deal_damage", target=target, damage=amount)

    def handle_modify_attack_damage(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        delta = self.amount_value(step.get("amount") or step.get("delta"), default=0, context=context)
        context["attack_damage_modifier"] = context.get("attack_damage_modifier", 0) + delta
        state.log_event("modify_attack_damage", delta=delta, step=step)

    def handle_place_damage_counters(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target"), context) or state.players[context["opponent"]].active
        amount = self.amount_value(step.get("amount"), default=1, context=context)
        if target:
            state.cards[target].damage_counters += amount
        state.log_event("place_damage_counters", target=target, counters=amount)

    def handle_heal_damage(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target"), context) or state.players[context["self"]].active
        amount = self.amount_value(step.get("amount"), default=0, context=context)
        if target:
            remove = amount // 10
            state.cards[target].damage_counters = max(0, state.cards[target].damage_counters - remove)
        state.log_event("heal_damage", target=target, amount=amount)

    def handle_apply_special_condition(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target"), context) or state.players[context["opponent"]].active
        condition = step.get("condition") or step.get("special_condition") or step.get("status")
        if isinstance(condition, list):
            conditions = condition
        else:
            conditions = [condition] if condition else []
        if target:
            for cond in conditions:
                if cond and cond not in state.cards[target].special_conditions:
                    state.cards[target].special_conditions.append(str(cond))
        state.log_event("apply_special_condition", target=target, conditions=conditions)

    def handle_switch_active(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        bench = state.players[player].bench
        if not bench:
            state.log_event("switch_active_no_bench", player=player)
            return
        new_active = bench[0]
        old_active = state.players[player].active
        bench.remove(new_active)
        if old_active:
            bench.append(old_active)
            state.cards[old_active].zone = "bench"
        state.players[player].active = new_active
        state.cards[new_active].zone = "active"
        state.log_event("switch_active", player=player, old_active=old_active, new_active=new_active)

    def handle_switch_active_with_bench(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        desired = self.resolve_instance_ref(state, step.get("bench_pokemon"), context)
        if desired and desired in state.players[player].bench:
            old_active = state.players[player].active
            state.players[player].bench.remove(desired)
            if old_active:
                state.players[player].bench.append(old_active)
                state.cards[old_active].zone = "bench"
            state.players[player].active = desired
            state.cards[desired].zone = "active"
            state.log_event("switch_active_with_bench", player=player, old_active=old_active, new_active=desired)
        else:
            self.handle_switch_active(state, step, context)

    def handle_coin_flip(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        result = "heads" if state.rng.random() < 0.5 else "tails"
        key = step.get("result_key") or step.get("flip_id") or "last_coin_flip"
        context[key] = result
        state.memory[key] = result
        state.log_event("coin_flip", result=result, key=key)

    def handle_coin_flip_until(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        stop = step.get("until", "tails")
        max_flips = int(step.get("max_flips", 100))
        results = []
        for _ in range(max_flips):
            result = "heads" if state.rng.random() < 0.5 else "tails"
            results.append(result)
            if result == stop:
                break
        key = step.get("result_key") or "last_coin_flip_until"
        context[key] = results
        state.memory[key] = results
        state.log_event("coin_flip_until", results=results, until=stop)

    def handle_branch_on_result(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        key = step.get("result_key") or step.get("flip_id") or "last_coin_flip"
        result = context.get(key, state.memory.get(key, "heads"))
        branch = step.get(str(result)) or step.get("then" if result == "heads" else "else") or []
        state.log_event("branch_on_result", key=key, result=result, branch_len=len(branch))
        for substep in branch:
            self.execute_step(state, substep, context)

    def handle_conditional(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        # Minimal: run 'then' only for conditions marked always/assumed, otherwise log and skip.
        condition = step.get("if") or step.get("condition")
        if condition in (True, {"always": True}) or step.get("assume_true") is True:
            branch = step.get("then", [])
        else:
            branch = []
        state.log_event("conditional", condition=condition, executed_then=bool(branch))
        for substep in branch:
            self.execute_step(state, substep, context)

    def handle_choose_target(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        choice_id = step.get("choice_id") or step.get("target_key") or "target"
        target = self.resolve_instance_ref(state, step.get("default") or step.get("target"), context) or state.players[context["opponent"]].active
        context["choices"][choice_id] = target
        state.log_event("choose_target", choice_id=choice_id, target=target)

    def handle_choose_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        choice_id = step.get("choice_id") or "cards"
        player = self.player_for_ref(step.get("player"), context)
        amount = self.amount_value(step.get("amount"), default=1, context=context)
        zone = step.get("from_zone") or step.get("from") or "hand"
        if isinstance(zone, list):
            zone = zone[0]
        if isinstance(zone, str) and "." in zone:
            player_ref, zone_name = zone.split(".", 1)
            player = self.player_for_ref(player_ref, context)
            zone = zone_name
        selected = state.players[player].zone_ids(str(zone))[:amount]
        context["choices"][choice_id] = selected
        state.log_event("choose_cards", choice_id=choice_id, selected=selected)

    def handle_choose_amount(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        choice_id = step.get("choice_id") or "amount"
        amount = self.amount_value(step.get("amount"), default=1, context=context)
        context["choices"][choice_id] = amount
        state.log_event("choose_amount", choice_id=choice_id, amount=amount)

    def handle_register_continuous_modifier(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        state.log_event("register_continuous_modifier", step=step)

    def handle_register_trigger(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        state.log_event("register_trigger", step=step)

    def handle_register_replacement_effect(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        state.log_event("register_replacement_effect", step=step)

    def handle_register_knockout_prize_rule(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        state.log_event("register_knockout_prize_rule", step=step)

    def handle_attack_does_nothing(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        context["attack_does_nothing"] = True
        state.log_event("attack_does_nothing", step=step)

    def handle_remember_card_instance(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        key = step.get("memory_key") or step.get("key") or "remembered_card"
        subject = self.resolve_instance_ref(state, step.get("subject"), context)
        if subject:
            state.memory[key] = subject
        state.log_event("remember_card_instance", key=key, value=subject)

    def handle_reveal_zone(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        zone = step.get("zone", "hand")
        try:
            cards = list(state.players[player].zone_ids(zone))
        except Exception:
            cards = []
        state.log_event("reveal_zone", player=player, zone=zone, cards=cards, reveal_to=step.get("reveal_to"))

    def handle_count_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        count_id = step.get("count_id") or step.get("target_id") or "last_count"
        source = step.get("from") or step.get("zone") or step.get("source_zone") or "self.hand"
        player = context["self"]
        zone = "hand"
        if isinstance(source, str) and "." in source:
            player_ref, zone = source.split(".", 1)
            player = self.player_for_ref(player_ref, context)
        elif isinstance(source, str):
            zone = source
            player = self.player_for_ref(step.get("player"), context)
        try:
            cards = list(state.players[player].zone_ids(zone))
        except Exception:
            cards = []
        # Minimal engine: count all candidates. A later validator can enforce filters.
        context.setdefault("counts", {})[count_id] = len(cards)
        state.memory[count_id] = len(cards)
        state.log_event("count_cards", count_id=count_id, count=len(cards), source=source, player=player, zone=zone, filter=step.get("filter"))

    def handle_register_generic_continuous(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        state.log_event(step.get("op", "register_generic_continuous"), step=step)

    def handle_for_each_player(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        body = step.get("do") or step.get("steps") or []
        original_self = context.get("self")
        original_opponent = context.get("opponent")
        for player in list(state.players.keys()):
            context["self"] = player
            context["opponent"] = state.opponent_of(player)
            state.log_event("for_each_player_iteration", player=player, body_len=len(body))
            for substep in body:
                self.execute_step(state, substep, context)
        context["self"] = original_self
        context["opponent"] = original_opponent

    def handle_optional_discard_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        # Conservative smoke-test behavior: choose not to pay optional discard costs.
        state.log_event("optional_discard_cards_skipped", step=step)

    def handle_draw_until_hand_size(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        target_size = self.amount_value(step.get("hand_size") or step.get("target_hand_size") or step.get("amount"), default=0, context=context)
        need = max(0, target_size - len(state.players[player].hand))
        self.handle_draw_cards(state, {"op": "draw_cards", "player": player, "amount": {"mode": "exact", "value": need}}, context)
        state.log_event("draw_until_hand_size", player=player, target_size=target_size, drew=need)

    def _zone_cards(self, state: GameState, player: str, zone: str) -> List[str]:
        try:
            return state.players[player].zone_ids(zone)
        except Exception:
            return []

    def _select_from_zone(self, state: GameState, player: str, zone: str, selection: Any, context: Dict[str, Any]) -> List[str]:
        cards = list(self._zone_cards(state, player, zone))
        if not cards:
            return []
        if isinstance(selection, str):
            if selection.startswith("choice.") or selection in context.get("choices", {}) or selection in state.memory:
                selected = self.resolve_instances_from_choice_or_zone(state, selection, context)
                return [iid for iid in selected if iid in cards]
        if not isinstance(selection, dict):
            selection = {"mode": "exact", "value": 1}
        mode = selection.get("mode", "exact")
        if mode == "all":
            return cards
        amount = self.amount_value(selection, default=1, context=context)
        if mode in {"exact", "up_to", "at_least"}:
            return cards[:amount]
        if mode == "random":
            shuffled = cards[:]
            state.rng.shuffle(shuffled)
            return shuffled[:amount]
        return cards[:amount]

    def handle_move_zone_to_zone(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        source_zone = str(step.get("source_zone") or step.get("from_zone") or "hand")
        destination_zone = str(step.get("destination_zone") or step.get("to_zone") or step.get("to") or "discard")
        if "." in source_zone:
            player_ref, source_zone = source_zone.split(".", 1)
            player = self.player_for_ref(player_ref, context)
        to_player = self.player_for_ref(step.get("to_player"), context, default=player)
        if "." in destination_zone:
            to_player_ref, destination_zone = destination_zone.split(".", 1)
            to_player = self.player_for_ref(to_player_ref, context, default=player)
        selected = self._select_from_zone(state, player, source_zone, step.get("selection"), context)
        moved = []
        for iid in list(selected):
            if iid in state.cards:
                state.move_instance(iid, to_player, destination_zone)
                moved.append(iid)
        state.log_event("move_zone_to_zone", player=player, from_zone=source_zone, to_player=to_player, to_zone=destination_zone, moved=moved)

    def handle_discard_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        source_zone = str(step.get("source_zone") or step.get("from_zone") or "hand")
        if "." in source_zone:
            player_ref, source_zone = source_zone.split(".", 1)
            player = self.player_for_ref(player_ref, context)
        selected = self._select_from_zone(state, player, source_zone, step.get("selection"), context)
        discarded = []
        for iid in list(selected):
            if iid in state.cards:
                state.move_instance(iid, player, "discard")
                discarded.append(iid)
        required = bool(step.get("required_to_play"))
        if required and not discarded:
            context["play_condition_failed"] = True
        state.log_event("discard_cards", player=player, from_zone=source_zone, discarded=discarded, required_to_play=required)

    def handle_draw_until_hand_size_matches(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        other = self.player_for_ref(step.get("other_player") or "opponent", context)
        target_size = len(state.players[other].hand)
        before = len(state.players[player].hand)
        self.handle_draw_cards(state, {"op": "draw_cards", "player": player, "amount": {"mode": "exact", "value": max(0, target_size - before)}}, context)
        state.log_event("draw_until_hand_size_matches", player=player, other_player=other, target_size=target_size, before=before, after=len(state.players[player].hand))

    def handle_discard_attached_energy(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target"), context) or state.players[context["opponent"]].active
        amount = self.amount_value(step.get("amount"), default=1, context=context)
        discarded = []
        if target and target in state.cards:
            attached = list(state.cards[target].attached_cards)
            energy_like = []
            for iid in attached:
                cdef = state.cards[iid].definition
                supertype = cdef.get("identity", {}).get("supertype")
                name = cdef.get("identity", {}).get("name", "")
                if supertype == "Energy" or "Energy" in name:
                    energy_like.append(iid)
            if not energy_like:
                energy_like = attached
            for iid in energy_like[:amount]:
                if iid in state.cards[target].attached_cards:
                    state.cards[target].attached_cards.remove(iid)
                owner = state.cards[iid].owner
                state.move_instance(iid, owner, "discard")
                discarded.append(iid)
        state.log_event("discard_attached_energy", target=target, amount=amount, discarded=discarded)

    def handle_play_condition(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        condition = step.get("condition") or step.get("if") or {}
        # Smoke-test behavior: evaluate simple flags if present, but do not halt execution.
        passed = True
        if isinstance(condition, dict):
            for key, expected in condition.items():
                actual = state.flags.get(key, state.players[context["self"]].flags.get(key))
                if actual is not None and actual != expected:
                    passed = False
        context["last_play_condition"] = passed
        state.log_event("play_condition", condition=condition, passed=passed, enforced=False)


    def handle_register_play_condition(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Register a play/use condition without enforcing it in the smoke-test engine.

        Full enforcement belongs in the legal action generator. For simulator smoke tests,
        recording the condition is enough to avoid treating valid compiled cards as runtime
        failures.
        """
        conditions = state.memory.setdefault("registered_play_conditions", [])
        conditions.append({
            "source_instance_id": context.get("source_instance_id"),
            "condition": step.get("condition") or step.get("if") or step,
            "source_text": step.get("source_text"),
        })
        state.log_event("register_play_condition", condition=step.get("condition") or step.get("if"), step=step)

    def handle_look_at_top_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        amount = self.amount_value(step.get("amount") or step.get("count") or step.get("top"), default=1, context=context)
        zone = str(step.get("zone") or "deck")
        if "." in zone:
            player_ref, zone = zone.split(".", 1)
            player = self.player_for_ref(player_ref, context)
        cards = list(self._zone_cards(state, player, zone))[:amount]
        key = step.get("result_key") or step.get("look_id") or "last_looked_cards"
        state.memory[key] = cards
        context["choices"][key] = cards
        # Conservative default: keep order unchanged unless a later reorder op is implemented.
        state.log_event("look_at_top_cards", player=player, zone=zone, amount=amount, cards=cards, stored_as=key)

    def handle_remove_special_conditions(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target"), context) or state.players[context["self"]].active
        condition = step.get("condition") or step.get("special_condition") or step.get("status")
        removed: List[str] = []
        if target and target in state.cards:
            if condition in (None, "all", ["all"]):
                removed = list(state.cards[target].special_conditions)
                state.cards[target].special_conditions.clear()
            else:
                conditions = condition if isinstance(condition, list) else [condition]
                for cond in list(conditions):
                    if cond in state.cards[target].special_conditions:
                        state.cards[target].special_conditions.remove(cond)
                        removed.append(cond)
        state.log_event("remove_special_conditions", target=target, removed=removed, requested=condition)


    def handle_discard_attached_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target"), context) or state.players[context["opponent"]].active
        amount = self.amount_value(step.get("amount"), default=9999, context=context)
        discarded: List[str] = []
        if target and target in state.cards:
            attached = list(state.cards[target].attached_cards)[:amount]
            for iid in attached:
                if iid in state.cards[target].attached_cards:
                    state.cards[target].attached_cards.remove(iid)
                owner = state.cards[iid].owner
                state.move_instance(iid, owner, "discard")
                discarded.append(iid)
        state.log_event("discard_attached_cards", target=target, amount=amount, discarded=discarded, filter=step.get("filter"))

    def handle_move_energy(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        source = self.resolve_instance_ref(state, step.get("from") or step.get("source") or step.get("source_pokemon"), context)
        target = self.resolve_instance_ref(state, step.get("to") or step.get("target") or step.get("target_pokemon"), context)
        amount = self.amount_value(step.get("amount"), default=1, context=context)
        moved: List[str] = []
        if source and source in state.cards and target and target in state.cards:
            energy_like = []
            for iid in list(state.cards[source].attached_cards):
                cdef = state.cards[iid].definition
                supertype = cdef.get("identity", {}).get("supertype")
                name = cdef.get("identity", {}).get("name", "")
                if supertype == "Energy" or "Energy" in name:
                    energy_like.append(iid)
            for iid in energy_like[:amount]:
                state.cards[source].attached_cards.remove(iid)
                state.cards[target].attached_cards.append(iid)
                state.cards[iid].zone = "attached"
                state.cards[iid].controller = state.cards[target].controller
                moved.append(iid)
        state.log_event("move_energy", source=source, target=target, amount=amount, moved=moved)

    def handle_put_card_into_hand(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        patched = dict(step)
        patched["to"] = "hand"
        self.handle_move_card(state, patched, context)

    def handle_put_card_on_bench(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        patched = dict(step)
        patched["to"] = "bench"
        self.handle_move_card(state, patched, context)

    def handle_reveal_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        cards = self.resolve_instances_from_choice_or_zone(state, step.get("cards") or step.get("card") or step.get("subject"), context)
        state.log_event("reveal_cards", cards=cards, reveal_to=step.get("reveal_to") or "opponent")

    def handle_move_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        self.handle_move_card(state, step, context)

    def _resolve_card_expression(self, state: GameState, expr: Any, context: Dict[str, Any]) -> List[str]:
        """Resolve permissive card-reference expressions used by compiler long-tail ops.

        Supports direct refs plus simple set subtraction strings such as
        ``looked_cards - chosen_pokemon_to_bench``. This is intentionally
        smoke-test-oriented: unresolved pieces produce an empty list rather than
        a hard error.
        """
        if expr is None:
            return []
        if isinstance(expr, list):
            out: List[str] = []
            for item in expr:
                out.extend(self._resolve_card_expression(state, item, context))
            return [iid for iid in out if iid]
        if isinstance(expr, str) and " - " in expr:
            left, right = [part.strip() for part in expr.split(" - ", 1)]
            base = self._resolve_card_expression(state, left, context)
            remove = set(self._resolve_card_expression(state, right, context))
            return [iid for iid in base if iid not in remove]
        return self.resolve_instances_from_choice_or_zone(state, expr, context)

    def handle_shuffle_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        cards = self._resolve_card_expression(state, step.get("cards") or step.get("cards_ref") or step.get("subject"), context)
        shuffled = list(cards)
        state.rng.shuffle(shuffled)
        key = step.get("result_key") or "last_shuffled_cards"
        state.memory[key] = shuffled
        context["choices"][key] = shuffled
        state.log_event("shuffle_cards", cards=cards, shuffled=shuffled, stored_as=key, step=step)

    def handle_reveal_hand(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        cards = list(state.players[player].hand)
        key = step.get("result_key") or f"{player}_revealed_hand"
        state.memory[key] = cards
        context["choices"][key] = cards
        state.log_event("reveal_hand", player=player, cards=cards, reveal_to=step.get("reveal_to") or context.get("self"), stored_as=key)

    def handle_reorder_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        cards = self._resolve_card_expression(state, step.get("cards") or step.get("cards_ref") or step.get("subject"), context)
        order = step.get("order") or step.get("mode") or "keep"
        ordered = list(cards)
        if order == "random" or step.get("shuffle"):
            state.rng.shuffle(ordered)
        # If a target zone is supplied, put the ordered subset at top or bottom while preserving others.
        zone_ref = step.get("zone") or step.get("target_zone") or step.get("destination_zone")
        player = self.player_for_ref(step.get("player"), context)
        zone = None
        if isinstance(zone_ref, str):
            zone = zone_ref
            if "." in zone:
                player_ref, zone = zone.split(".", 1)
                player = self.player_for_ref(player_ref, context)
        if zone:
            zone_list = self._zone_cards(state, player, zone)
            existing = [iid for iid in zone_list if iid not in set(ordered)]
            position = step.get("position") or step.get("to") or "top"
            if position == "bottom":
                zone_list[:] = existing + ordered
            else:
                zone_list[:] = ordered + existing
        key = step.get("result_key") or "last_reordered_cards"
        state.memory[key] = ordered
        context["choices"][key] = ordered
        state.log_event("reorder_cards", cards=cards, ordered=ordered, zone=zone_ref, stored_as=key, step=step)

    def handle_grant_attack_from_attached_card(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        grants = state.memory.setdefault("granted_attacks", [])
        grants.append({
            "source_instance_id": context.get("source_instance_id"),
            "target": step.get("target"),
            "duration": step.get("duration"),
            "source_text": step.get("source_text"),
        })
        state.log_event("grant_attack_from_attached_card", step=step)

    def handle_play_trainer_as_pokemon(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        source = context.get("source_instance_id")
        if source and source in state.cards:
            card = state.cards[source]
            card.turn_memory["played_as_pokemon"] = {
                "hp": step.get("hp"),
                "stage": step.get("stage"),
                "types": step.get("types"),
                "retreat_allowed": step.get("retreat_allowed"),
                "cannot_be_affected_by_special_conditions": step.get("cannot_be_affected_by_special_conditions"),
            }
        state.log_event("play_trainer_as_pokemon", source_instance_id=source, step=step)

    def handle_ignore_weakness_resistance(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        context["ignore_weakness_resistance"] = True
        state.log_event("ignore_weakness_resistance", step=step)

    def handle_ignore_effects_on_defending_pokemon(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        context["ignore_effects_on_defending_pokemon"] = True
        state.log_event("ignore_effects_on_defending_pokemon", step=step)

    def _coin_heads_count(self, step: Dict[str, Any], context: Dict[str, Any]) -> int:
        """Best-effort count of heads for coin-dependent smoke-test effects."""
        ref = step.get("result_key") or step.get("flip_id") or step.get("coin_result_key") or "last_coin_flip"
        val = context.get(ref)
        if val is None:
            val = context.get("last_coin_flip")
        if val is None:
            val = context.get("last_coin_results") or context.get("coin_results")
        if isinstance(val, str):
            return 1 if val == "heads" else 0
        if isinstance(val, list):
            return sum(1 for x in val if x == "heads")
        if isinstance(val, dict):
            if "heads" in val:
                try:
                    return int(val["heads"])
                except Exception:
                    return 0
            if "results" in val and isinstance(val["results"], list):
                return sum(1 for x in val["results"] if x == "heads")
        return int(step.get("assume_heads", 0) or 0)

    def handle_draw_cards_per_coin_heads(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        heads = self._coin_heads_count(step, context)
        per_heads = self.amount_value(step.get("cards_per_heads") or step.get("per_heads") or step.get("amount_per_heads"), default=1, context=context)
        amount = heads * per_heads
        self.handle_draw_cards(state, {"op": "draw_cards", "player": step.get("player", "self"), "amount": {"mode": "exact", "value": amount}}, context)
        state.log_event("draw_cards_per_coin_heads", heads=heads, per_heads=per_heads, amount=amount, step=step)

    def handle_provide_energy(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        source = context.get("source_instance_id")
        if source and source in state.cards:
            provided = state.cards[source].turn_memory.setdefault("provided_energy", [])
            provided.append({"types": step.get("types") or step.get("energy_types") or step.get("provides"), "amount": step.get("amount"), "step": step})
        state.log_event("provide_energy", source_instance_id=source, types=step.get("types") or step.get("energy_types") or step.get("provides"), amount=step.get("amount"), step=step)

    def handle_evolve_pokemon_from_hand(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player"), context)
        target = self.resolve_instance_ref(state, step.get("target") or step.get("onto") or step.get("pokemon"), context) or state.players[player].active
        evolution = self.resolve_instance_ref(state, step.get("card") or step.get("evolution_card"), context)
        if evolution is None:
            hand = list(state.players[player].hand)
            evolution = hand[0] if hand else None
        evolved = False
        if target and evolution and evolution in state.cards and target in state.cards:
            try:
                if evolution in state.players[player].hand:
                    state.players[player].hand.remove(evolution)
            except Exception:
                pass
            state.cards[target].evolution_stack.append(evolution)
            state.cards[evolution].zone = "evolution_stack"
            state.cards[evolution].controller = player
            evolved = True
        state.log_event("evolve_pokemon_from_hand", player=player, target=target, evolution_card=evolution, evolved=evolved, step=step)

    def handle_swap_pokemon_card_with_deck_card(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        # Smoke-test approximation for old form-change powers. Do not attempt full rules;
        # record intent and optionally swap source with the top valid deck card.
        player = self.player_for_ref(step.get("player"), context)
        source = self.resolve_instance_ref(state, step.get("source") or step.get("pokemon") or "source", context) or context.get("source_instance_id")
        deck = state.players[player].deck
        replacement = deck[0] if deck else None
        swapped = False
        if source and replacement and source in state.cards and replacement in state.cards:
            # Put the source on top of deck and make replacement occupy source zone if possible.
            source_zone = state.cards[source].zone
            try:
                deck.remove(replacement)
            except ValueError:
                pass
            deck.insert(0, source)
            state.cards[source].zone = "deck"
            if source_zone in {"active", "bench"}:
                if source_zone == "active" and state.players[player].active == source:
                    state.players[player].active = replacement
                elif source_zone == "bench" and source in state.players[player].bench:
                    idx = state.players[player].bench.index(source)
                    state.players[player].bench[idx] = replacement
                state.cards[replacement].zone = source_zone
                state.cards[replacement].controller = player
                swapped = True
        state.log_event("swap_pokemon_card_with_deck_card", player=player, source=source, replacement=replacement, swapped=swapped, step=step)

    def handle_register_usage_limit(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        limits = state.memory.setdefault("registered_usage_limits", [])
        limits.append({"source_instance_id": context.get("source_instance_id"), "step": step})
        state.log_event("register_usage_limit", step=step)

    def handle_ignore_resistance(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        context["ignore_resistance"] = True
        state.log_event("ignore_resistance", step=step)

    def handle_discard_stadium(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        # No stadium zone exists yet in the seed state; record as a marker for the future board engine.
        state.memory["stadium_discard_requested"] = True
        state.log_event("discard_stadium", step=step)

    def handle_choose_attack(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        choice_id = step.get("choice_id") or "chosen_attack"
        attack = step.get("attack") or step.get("default") or step.get("attack_name") or "first_available_attack"
        context["choices"][choice_id] = attack
        state.log_event("choose_attack", choice_id=choice_id, attack=attack, step=step)

    def handle_set_attack_damage_from_coin_heads(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        heads = self._coin_heads_count(step, context)
        per_heads = self.amount_value(step.get("damage_per_heads") or step.get("per_heads") or step.get("amount_per_heads"), default=10, context=context)
        base = self.amount_value(step.get("base_damage") or step.get("base"), default=0, context=context)
        damage = base + heads * per_heads
        context["attack_damage"] = damage
        state.log_event("set_attack_damage_from_coin_heads", heads=heads, per_heads=per_heads, base=base, damage=damage, step=step)

    def handle_modify_attack_damage_per_coin_heads(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        heads = self._coin_heads_count(step, context)
        per_heads = self.amount_value(step.get("damage_per_heads") or step.get("per_heads") or step.get("amount_per_heads"), default=10, context=context)
        delta = heads * per_heads
        context["attack_damage_modifier"] = context.get("attack_damage_modifier", 0) + delta
        state.log_event("modify_attack_damage_per_coin_heads", heads=heads, per_heads=per_heads, delta=delta, step=step)

    def handle_modify_special_condition(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target"), context) or state.players[context["opponent"]].active
        modifier = step.get("modifier") or step.get("condition_modifier") or step
        if target and target in state.cards:
            state.cards[target].turn_memory.setdefault("special_condition_modifiers", []).append(modifier)
        state.log_event("modify_special_condition", target=target, modifier=modifier, step=step)

    def handle_modify_attack_damage_per_damage_counter(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target") or step.get("count_on"), context) or state.players[context["opponent"]].active
        counters = state.cards[target].damage_counters if target and target in state.cards else 0
        per_counter = self.amount_value(step.get("damage_per_counter") or step.get("per_counter") or step.get("amount_per_counter"), default=10, context=context)
        delta = counters * per_counter
        context["attack_damage_modifier"] = context.get("attack_damage_modifier", 0) + delta
        state.log_event("modify_attack_damage_per_damage_counter", target=target, counters=counters, per_counter=per_counter, delta=delta, step=step)

    def handle_set_attack_damage_from_count(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        count_id = step.get("count_id") or step.get("from_count") or step.get("count_ref") or "last_count"
        count = context.get("counts", {}).get(count_id, state.memory.get(count_id, 0))
        try:
            count = int(count)
        except Exception:
            count = 0
        per = self.amount_value(step.get("damage_per") or step.get("per_card") or step.get("amount_per"), default=10, context=context)
        base = self.amount_value(step.get("base_damage") or step.get("base"), default=0, context=context)
        damage = base + count * per
        context["attack_damage"] = damage
        state.log_event("set_attack_damage_from_count", count_id=count_id, count=count, per=per, base=base, damage=damage, step=step)

    def handle_register_evolution_permission(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        permissions = state.memory.setdefault("registered_evolution_permissions", [])
        permissions.append({"source_instance_id": context.get("source_instance_id"), "step": step})
        state.log_event("register_evolution_permission", step=step)

    def handle_register_evolution_exception(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        exceptions = state.memory.setdefault("registered_evolution_exceptions", [])
        exceptions.append({"source_instance_id": context.get("source_instance_id"), "step": step})
        state.log_event("register_evolution_exception", step=step)



    def handle_look_at_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Smoke-test handler for looking at arbitrary cards/zones.

        This records visible information and stores the looked-at ids in context. It does
        not enforce hidden information, because this seed engine is single-process and
        deterministic for testing.
        """
        player = self.player_for_ref(step.get("player") or step.get("viewer"), context)
        amount = self.amount_value(step.get("amount") or step.get("count"), default=1, context=context)
        zone_ref = step.get("zone") or step.get("from") or step.get("source_zone") or "self.deck"
        if isinstance(zone_ref, str) and "." not in zone_ref:
            zone_ref = f"self.{zone_ref}"
        candidates = self.find_candidates(state, [zone_ref] if isinstance(zone_ref, str) else zone_ref, context)
        looked = candidates[:amount]
        choice_id = step.get("choice_id") or step.get("remember_as") or "looked_at_cards"
        context[choice_id] = looked
        state.memory[choice_id] = looked
        state.log_event("look_at_cards", player=player, zone=zone_ref, amount=amount, cards=looked, step=step)

    def handle_modify_attack_damage_from_count(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Apply a damage modifier based on a previously stored or inferred count."""
        count_id = step.get("count_id") or step.get("from_count") or step.get("count_ref") or "last_count"
        count = context.get("counts", {}).get(count_id, state.memory.get(count_id))
        if count is None:
            # Some compiler steps encode the counted zone/filter directly.
            zone_ref = step.get("zone") or step.get("count_zone") or step.get("from")
            if zone_ref:
                if isinstance(zone_ref, str) and "." not in zone_ref:
                    zone_ref = f"self.{zone_ref}"
                count = len(self.find_candidates(state, [zone_ref] if isinstance(zone_ref, str) else zone_ref, context))
            else:
                count = 0
        try:
            count = int(count)
        except Exception:
            count = 0
        per = self.amount_value(step.get("damage_per") or step.get("per_card") or step.get("per") or step.get("amount_per"), default=10, context=context)
        base = self.amount_value(step.get("base_damage") or step.get("base"), default=0, context=context)
        delta = base + count * per
        context["attack_damage_modifier"] = context.get("attack_damage_modifier", 0) + delta
        state.log_event("modify_attack_damage_from_count", count_id=count_id, count=count, per=per, base=base, delta=delta, step=step)

    def handle_evolve_pokemon(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Generic evolution marker/approximation.

        If a concrete evolution card is available, reuse the from-hand handler. Otherwise
        register the intent so future rule enforcement can resolve it.
        """
        if step.get("card") or step.get("evolution_card"):
            self.handle_evolve_pokemon_from_hand(state, step, context)
            return
        player = self.player_for_ref(step.get("player"), context)
        target = self.resolve_instance_ref(state, step.get("target") or step.get("onto") or step.get("pokemon"), context) or state.players[player].active
        state.memory.setdefault("evolution_events", []).append({"target": target, "source_instance_id": context.get("source_instance_id"), "step": step})
        state.log_event("evolve_pokemon", player=player, target=target, applied=False, step=step)

    def handle_discard_attached_energy_per_coin_heads(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        heads = self._coin_heads_count(step, context)
        per_heads = self.amount_value(step.get("energy_per_heads") or step.get("per_heads") or step.get("amount_per_heads"), default=1, context=context)
        amount = heads * per_heads
        substep = dict(step)
        substep["amount"] = {"mode": "exact", "value": amount}
        self.handle_discard_attached_energy(state, substep, context)
        state.log_event("discard_attached_energy_per_coin_heads", heads=heads, per_heads=per_heads, amount=amount, step=step)

    def handle_set_attack_damage_per_pokemon(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Set attack damage from a Pokémon count in play/bench/discard etc."""
        scope = step.get("scope") or step.get("zone") or step.get("count_zone") or step.get("from") or "self.in_play"
        if scope in {"self.in_play", "your_pokemon", "self.pokemon_in_play"}:
            player = context["self"]
            count = (1 if state.players[player].active else 0) + len(state.players[player].bench)
        elif scope in {"opponent.in_play", "opponent_pokemon", "opponent.pokemon_in_play"}:
            player = context["opponent"]
            count = (1 if state.players[player].active else 0) + len(state.players[player].bench)
        elif scope in {"all.in_play", "all_pokemon", "both.in_play"}:
            count = 0
            for pstate in state.players.values():
                count += (1 if pstate.active else 0) + len(pstate.bench)
        else:
            zone_ref = scope if isinstance(scope, str) and "." in scope else f"self.{scope}"
            count = len(self.find_candidates(state, [zone_ref], context))
        per = self.amount_value(step.get("damage_per_pokemon") or step.get("per_pokemon") or step.get("damage_per") or step.get("per"), default=10, context=context)
        base = self.amount_value(step.get("base_damage") or step.get("base"), default=0, context=context)
        damage = base + count * per
        context["attack_damage"] = damage
        state.log_event("set_attack_damage_per_pokemon", scope=scope, count=count, per=per, base=base, damage=damage, step=step)

    def handle_distribute_damage_counters(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Best-effort distribution: put counters on the first legal/default target.

        A real engine needs player choice here. The smoke-test engine chooses a stable
        target and logs the distribution.
        """
        amount = self.amount_value(step.get("amount") or step.get("counters") or step.get("damage_counters"), default=1, context=context)
        targets = self.resolve_instances_from_choice_or_zone(state, step.get("targets") or step.get("target"), context)
        if not targets:
            opp = context["opponent"]
            targets = [iid for iid in [state.players[opp].active] if iid]
        placed = []
        if targets and amount > 0:
            target = targets[0]
            if target in state.cards:
                state.cards[target].damage_counters += amount
                placed.append({"target": target, "counters": amount})
        state.log_event("distribute_damage_counters", amount=amount, placed=placed, step=step)

    def handle_discard_card(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Singular discard wrapper around discard/move behavior."""
        player = self.player_for_ref(step.get("player"), context)
        card_id = self.resolve_instance_ref(state, step.get("card") or step.get("target"), context)
        if card_id is None:
            source_zone = step.get("source_zone") or step.get("from") or "hand"
            zone = state.players[player].zone_ids(source_zone) if source_zone != "active" else []
            card_id = zone[0] if zone else None
        discarded = []
        if card_id and card_id in state.cards:
            try:
                state.move_instance(card_id, player, "discard")
                discarded.append(card_id)
            except Exception:
                pass
        state.log_event("discard_card", player=player, discarded=discarded, step=step)


    def handle_reveal_hand_to_player(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Reveal one player's hand to another player.

        This is a visibility/logging action in the seed runtime. Hidden-information
        enforcement can be added later when the engine has UI/player-choice layers.
        """
        player = self.player_for_ref(step.get("player") or step.get("revealed_player") or step.get("target_player") or "opponent", context)
        reveal_to = self.player_for_ref(step.get("reveal_to") or step.get("to_player") or step.get("viewer") or "self", context)
        cards = list(state.players[player].hand)
        key = step.get("result_key") or f"{player}_hand_revealed_to_{reveal_to}"
        state.memory[key] = cards
        context["choices"][key] = cards
        state.log_event("reveal_hand_to_player", player=player, reveal_to=reveal_to, cards=cards, stored_as=key, step=step)

    def handle_ignore_defending_pokemon_damage_modifiers(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Marker for attacks that bypass defensive modifiers on the defending Pokémon."""
        context["ignore_defending_pokemon_damage_modifiers"] = True
        state.log_event("ignore_defending_pokemon_damage_modifiers", step=step)

    def _attached_energy_count(self, state: GameState, target: Optional[str], step: Dict[str, Any]) -> int:
        if not target or target not in state.cards:
            return 0
        requested_type = step.get("energy_type") or step.get("type") or step.get("attached_energy_type")
        count = 0
        for iid in list(state.cards[target].attached_cards):
            cdef = state.cards[iid].definition
            identity = cdef.get("identity", {}) if isinstance(cdef, dict) else {}
            supertype = identity.get("supertype")
            name = identity.get("name", "")
            types = identity.get("types") or cdef.get("gameplay", {}).get("types", []) if isinstance(cdef, dict) else []
            is_energy = supertype == "Energy" or "Energy" in str(name)
            if not is_energy:
                continue
            if requested_type and requested_type not in ("any", "Energy"):
                if requested_type not in str(name) and requested_type not in list(types or []):
                    continue
            count += 1
        return count

    def handle_set_attack_damage_per_attached_energy(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target") or step.get("attached_to") or step.get("count_on"), context)
        if target is None:
            # Most legacy wording counts Energy on the Defending/opponent Active Pokémon.
            target = state.players[context["opponent"]].active
        count = self._attached_energy_count(state, target, step)
        per = self.amount_value(step.get("damage_per_energy") or step.get("per_energy") or step.get("damage_per") or step.get("per"), default=10, context=context)
        base = self.amount_value(step.get("base_damage") or step.get("base"), default=0, context=context)
        damage = base + count * per
        context["attack_damage"] = damage
        state.log_event("set_attack_damage_per_attached_energy", target=target, energy_count=count, per=per, base=base, damage=damage, step=step)

    def handle_conditional_effect(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Best-effort conditional branch for compiled effect conditions.

        The smoke-test engine cannot fully evaluate every Pokémon TCG condition yet.
        It evaluates simple booleans and otherwise follows `assume_true` or defaults to
        not executing the branch while still logging the condition.
        """
        condition = step.get("condition") or step.get("if") or {}
        if condition in (True, {"always": True}) or step.get("assume_true") is True:
            passed = True
        elif step.get("assume_false") is True:
            passed = False
        else:
            # Keep conservative behavior for unknown conditions.
            passed = False
        branch = step.get("then") or step.get("steps") or step.get("do") or []
        state.log_event("conditional_effect", condition=condition, passed=passed, branch_len=len(branch), step=step)
        if passed:
            for substep in branch:
                self.execute_step(state, substep, context)

    def handle_choose_player(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        choice_id = step.get("choice_id") or step.get("target_id") or "chosen_player"
        choices = step.get("choices") or ["self", "opponent"]
        if isinstance(choices, str):
            choices = [choices]
        default = step.get("default") or step.get("player") or (choices[0] if choices else "self")
        chosen = self.player_for_ref(default, context)
        context["choices"][choice_id] = chosen
        state.memory[choice_id] = chosen
        state.log_event("choose_player", choice_id=choice_id, choices=choices, chosen=chosen, step=step)

    def handle_modify_attack_damage_by_heads_table(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Apply a coin-heads lookup table such as 1 heads => +10, 2 heads => +20."""
        heads = self._coin_heads_count(step, context)
        table = step.get("heads_table") or step.get("damage_table") or step.get("table") or {}
        value = None
        if isinstance(table, dict):
            value = table.get(str(heads), table.get(heads))
        elif isinstance(table, list) and 0 <= heads < len(table):
            value = table[heads]
        if value is None:
            per = self.amount_value(step.get("damage_per_heads") or step.get("per_heads") or step.get("per"), default=10, context=context)
            value = heads * per
        delta = self.amount_value(value, default=0, context=context)
        context["attack_damage_modifier"] = context.get("attack_damage_modifier", 0) + delta
        state.log_event("modify_attack_damage_by_heads_table", heads=heads, delta=delta, table=table, step=step)

    def handle_conditional_attack_does_nothing(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Marker for conditional fizzles, e.g. 'if tails, this attack does nothing'."""
        condition = step.get("condition") or step.get("if") or {}
        if condition in (True, {"always": True}) or step.get("assume_true") is True:
            fizzles = True
        elif step.get("result") in {"tails", "heads"}:
            key = step.get("result_key") or step.get("flip_id") or "last_coin_flip"
            fizzles = context.get(key, state.memory.get(key)) == step.get("result")
        else:
            fizzles = False
        if fizzles:
            context["attack_does_nothing"] = True
        state.log_event("conditional_attack_does_nothing", condition=condition, fizzles=fizzles, step=step)


    def handle_register_deck_construction_rule(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Record deck-construction constraints such as Prism Star / ACE SPEC.

        The smoke-test runtime does not validate deck legality yet; it just preserves the
        rule so a future deck validator can enforce it.
        """
        rules = state.flags.setdefault("deck_construction_rules", [])
        rules.append(step)
        state.log_event("register_deck_construction_rule", rule=step)

    def handle_choose_special_condition(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Auto-pick a Special Condition for smoke tests.

        Real play should ask the player. Here we choose the first listed option, or
        Poisoned as a deterministic default.
        """
        options = step.get("options") or step.get("choices") or step.get("special_conditions") or ["Poisoned"]
        if isinstance(options, str):
            options = [options]
        selected = options[0] if options else "Poisoned"
        key = step.get("choice_id") or step.get("result_key") or "chosen_special_condition"
        context.setdefault("choices", {})[key] = selected
        state.memory[key] = selected
        state.log_event("choose_special_condition", choice_id=key, selected=selected, options=options)

    def handle_choose_yes_no(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Auto-answer yes/no prompts for smoke tests.

        Defaults to yes when an effect is optional so deeper steps can be walked.
        """
        key = step.get("choice_id") or step.get("result_key") or "last_yes_no_choice"
        default = step.get("default", True)
        if isinstance(default, str):
            selected = default.strip().lower() not in {"no", "false", "0"}
        else:
            selected = bool(default)
        context.setdefault("choices", {})[key] = selected
        state.memory[key] = selected
        state.log_event("choose_yes_no", choice_id=key, selected=selected, prompt=step.get("prompt"), step=step)

    def _choice_value(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        key = step.get("choice_id") or step.get("choice") or step.get("result_key") or "last_yes_no_choice"
        if isinstance(key, str) and key.startswith("choice."):
            key = key.split(".", 1)[1]
        return context.get("choices", {}).get(key, context.get(key))

    def handle_branch_on_choice(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Branch on a previous smoke-test choice.

        Supports several compiler shapes: on_yes/on_no, if_true/if_false,
        branches['yes'/'no'], or then/else.
        """
        value = self._choice_value(step, context)
        truthy = bool(value)
        branch_steps = None
        branches = step.get("branches")
        if isinstance(branches, dict):
            branch_steps = branches.get("yes" if truthy else "no") or branches.get(str(value))
        if branch_steps is None:
            branch_steps = step.get("on_yes" if truthy else "on_no")
        if branch_steps is None:
            branch_steps = step.get("if_true" if truthy else "if_false")
        if branch_steps is None:
            branch_steps = step.get("then" if truthy else "else")
        state.log_event("branch_on_choice", choice=value, branch_taken=bool(branch_steps), step=step)
        if isinstance(branch_steps, dict):
            branch_steps = [branch_steps]
        for substep in branch_steps or []:
            if isinstance(substep, dict):
                self.execute_step(state, substep, context)

    def handle_discard_cards_per_coin_heads(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        heads = int(context.get("last_coin_heads", state.memory.get("last_coin_heads", 0)) or 0)
        per = self.amount_value(step.get("per_heads") or step.get("cards_per_heads") or step.get("amount_per_heads"), default=1, context=context)
        amount = heads * per
        player = self.player_for_ref(step.get("player") or step.get("target_player") or "opponent", context)
        source_zone = step.get("source_zone") or step.get("zone") or "deck"
        discarded = []
        zone = None
        try:
            zone = state.players[player].zone_ids(source_zone)
        except Exception:
            zone = None
        if zone is not None:
            for _ in range(min(amount, len(zone))):
                iid = zone.pop(0)
                state.players[player].discard.append(iid)
                state.cards[iid].zone = "discard"
                discarded.append(iid)
        state.log_event("discard_cards_per_coin_heads", player=player, heads=heads, amount=amount, source_zone=source_zone, discarded=discarded, step=step)

    def handle_move_damage_counters(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        amount = self.amount_value(step.get("amount") or step.get("counters"), default=1, context=context)
        source = self.resolve_instance_ref(state, step.get("from") or step.get("source"), context)
        target = self.resolve_instance_ref(state, step.get("to") or step.get("target"), context)
        if source is None:
            source = state.players[context["self"]].active
        if target is None:
            target = state.players[context["opponent"]].active
        moved = 0
        if source and target:
            moved = min(amount, state.cards[source].damage_counters)
            state.cards[source].damage_counters -= moved
            state.cards[target].damage_counters += moved
        state.log_event("move_damage_counters", source=source, target=target, requested=amount, moved=moved, step=step)


    def _pokemon_hp(self, state: GameState, instance_id: Optional[str], default: int = 60) -> int:
        """Return printed HP for a Pokémon instance when available."""
        if not instance_id or instance_id not in state.cards:
            return default
        cdef = state.cards[instance_id].definition or {}
        gameplay = cdef.get("gameplay", {}) if isinstance(cdef, dict) else {}
        raw = gameplay.get("hp") or cdef.get("hp") or cdef.get("printed", {}).get("hp") if isinstance(cdef, dict) else None
        try:
            return int(raw)
        except Exception:
            return default

    def _damage_counter_count_for_ref(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> Tuple[Optional[str], int]:
        target = self.resolve_instance_ref(
            state,
            step.get("target") or step.get("count_on") or step.get("pokemon") or step.get("source"),
            context,
        )
        if target is None:
            target = state.players[context["opponent"]].active
        counters = state.cards[target].damage_counters if target and target in state.cards else 0
        return target, counters

    def handle_copy_and_use_attack(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Record attack-copy intent.

        A complete implementation needs an attack-selection UI plus recursive attack execution.
        The smoke-test engine preserves the chosen attack metadata without recursively executing
        arbitrary attacks, which avoids infinite copy loops from legacy card text.
        """
        attack = step.get("attack") or step.get("attack_name") or step.get("chosen_attack") or context.get("choices", {}).get("chosen_attack")
        source = self.resolve_instance_ref(state, step.get("source") or step.get("from_pokemon"), context)
        if source is None:
            source = state.players[context["opponent"]].active
        context["copied_attack"] = {"source": source, "attack": attack, "step": step}
        state.log_event("copy_and_use_attack", source=source, attack=attack, step=step)

    def handle_set_attack_damage_per_damage_counter(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target, counters = self._damage_counter_count_for_ref(state, step, context)
        per_counter = self.amount_value(
            step.get("damage_per_counter") or step.get("per_counter") or step.get("amount_per_counter") or step.get("per"),
            default=10,
            context=context,
        )
        base = self.amount_value(step.get("base_damage") or step.get("base"), default=0, context=context)
        damage = base + counters * per_counter
        context["attack_damage"] = damage
        state.log_event("set_attack_damage_per_damage_counter", target=target, counters=counters, per_counter=per_counter, base=base, damage=damage, step=step)

    def handle_attack_does_nothing_if(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Conditional attack-fizzle marker.

        This supports simple boolean conditions and coin-result conditions. Unknown conditions are
        logged but treated as not currently fizzled so smoke tests continue walking effects.
        """
        condition = step.get("condition") or step.get("if") or {}
        fizzles = False
        if condition in (True, {"always": True}) or step.get("assume_true") is True:
            fizzles = True
        elif step.get("assume_false") is True:
            fizzles = False
        elif isinstance(condition, dict):
            result = condition.get("coin_result") or condition.get("last_coin_result") or condition.get("result")
            if result in {"heads", "tails"}:
                fizzles = context.get("last_coin_flip", state.memory.get("last_coin_flip")) == result
        if fizzles:
            context["attack_does_nothing"] = True
        state.log_event("attack_does_nothing_if", condition=condition, fizzles=fizzles, step=step)

    def handle_set_attack_damage_from_value(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        damage = self.amount_value(
            step.get("damage") or step.get("value") or step.get("amount") or step.get("base_damage"),
            default=0,
            context=context,
        )
        context["attack_damage"] = damage
        state.log_event("set_attack_damage_from_value", damage=damage, step=step)

    def handle_devolve_pokemon(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Best-effort devolve action.

        If the runtime instance has an evolution stack, remove the top stored evolution card and
        move it to the requested destination. If not, log the devolve intent only.
        """
        player = self.player_for_ref(step.get("player") or "opponent", context)
        target = self.resolve_instance_ref(state, step.get("target") or step.get("pokemon"), context) or state.players[player].active
        destination = step.get("destination") or step.get("to_zone") or "hand"
        removed = None
        if target and target in state.cards and state.cards[target].evolution_stack:
            removed = state.cards[target].evolution_stack.pop()
            if removed in state.cards:
                try:
                    state.move_instance(removed, state.cards[removed].owner, destination)
                except Exception:
                    state.cards[removed].zone = destination
        state.log_event("devolve_pokemon", player=player, target=target, removed=removed, destination=destination, step=step)

    def handle_knock_out_pokemon(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target") or step.get("pokemon"), context) or state.players[context["opponent"]].active
        moved = []
        if target and target in state.cards:
            card = state.cards[target]
            controller = card.controller
            attachments = list(card.attached_cards)
            for attached in attachments:
                if attached in state.cards:
                    try:
                        state.move_instance(attached, state.cards[attached].owner, "discard")
                    except Exception:
                        state.cards[attached].zone = "discard"
                    moved.append(attached)
            card.attached_cards.clear()
            try:
                state.move_instance(target, controller, "discard")
            except Exception:
                card.zone = "discard"
            moved.append(target)
            state.flags.setdefault("knockouts", []).append({"target": target, "by": context.get("source_instance_id")})
        state.log_event("knock_out_pokemon", target=target, moved_to_discard=moved, step=step)

    def handle_register_legality_note(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        notes = state.flags.setdefault("legality_notes", [])
        notes.append({"source_instance_id": context.get("source_instance_id"), "step": step})
        state.log_event("register_legality_note", step=step)

    def handle_deal_damage_per_damage_counter(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        count_target, counters = self._damage_counter_count_for_ref(state, step, context)
        damage_target = self.resolve_instance_ref(state, step.get("damage_target") or step.get("target_to_damage") or step.get("target"), context)
        if damage_target is None:
            damage_target = state.players[context["opponent"]].active
        per_counter = self.amount_value(
            step.get("damage_per_counter") or step.get("per_counter") or step.get("damage_per") or step.get("per"),
            default=10,
            context=context,
        )
        damage = counters * per_counter
        if damage_target and damage_target in state.cards:
            state.cards[damage_target].damage_counters += damage // 10
        state.log_event("deal_damage_per_damage_counter", count_target=count_target, damage_target=damage_target, counters=counters, per_counter=per_counter, damage=damage, step=step)

    def handle_move_pokemon_and_attached_cards(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        player = self.player_for_ref(step.get("player") or step.get("target_player") or "opponent", context)
        target = self.resolve_instance_ref(state, step.get("target") or step.get("pokemon"), context) or state.players[player].active
        destination = step.get("destination") or step.get("to_zone") or step.get("to") or "hand"
        moved = []
        if target and target in state.cards:
            card = state.cards[target]
            # Move attachments first so they remain associated with the same player/zone action.
            for attached in list(card.attached_cards):
                if attached in state.cards:
                    try:
                        state.move_instance(attached, state.cards[attached].owner, destination)
                    except Exception:
                        state.cards[attached].zone = destination
                    moved.append(attached)
            card.attached_cards.clear()
            try:
                state.move_instance(target, card.controller, destination)
            except Exception:
                card.zone = destination
            moved.append(target)
        state.log_event("move_pokemon_and_attached_cards", player=player, target=target, destination=destination, moved=moved, step=step)

    def handle_place_damage_counters_until_remaining_hp(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        target = self.resolve_instance_ref(state, step.get("target") or step.get("pokemon"), context) or state.players[context["opponent"]].active
        remaining_hp = self.amount_value(
            step.get("remaining_hp") or step.get("until_remaining_hp") or step.get("leave_hp"),
            default=10,
            context=context,
        )
        placed = 0
        if target and target in state.cards:
            hp = self._pokemon_hp(state, target, default=60)
            current_damage = state.cards[target].damage_counters * 10
            desired_damage = max(0, hp - remaining_hp)
            additional_damage = max(0, desired_damage - current_damage)
            placed = additional_damage // 10
            state.cards[target].damage_counters += placed
        else:
            hp = None
        state.log_event("place_damage_counters_until_remaining_hp", target=target, hp=hp, remaining_hp=remaining_hp, placed=placed, step=step)

    def handle_unknown(self, state: GameState, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        state.log_event("unsupported_op", op=step.get("op"), step=step)
        if self.strict:
            raise NotImplementedError(f"Unsupported op: {step.get('op')}")
