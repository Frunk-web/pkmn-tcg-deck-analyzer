# Pokémon TCG effect operation vocabulary v1

This is the recommended operation layer for the simulator. Card text should compile into these primitives rather than custom Python for every card.

## Core zone/card movement
- `draw_cards`
- `search_deck`
- `reveal_cards`
- `look_at_cards`
- `move_card`
- `put_card_into_hand`
- `put_card_on_bench`
- `attach_card`
- `discard_card`
- `shuffle_deck`
- `put_on_top_of_deck`
- `put_on_bottom_of_deck`
- `put_in_lost_zone`
- `take_prize_cards`

## Pokémon board actions
- `play_basic_to_bench`
- `evolve_pokemon`
- `devolve_pokemon`
- `switch_active`
- `force_switch`
- `retreat`
- `set_no_retreat_cost`
- `modify_retreat_cost`

## Combat and HP
- `deal_attack_damage`
- `place_damage_counters`
- `heal_damage`
- `modify_attack_damage`
- `prevent_damage`
- `prevent_effects`
- `apply_weakness`
- `apply_resistance`
- `knock_out`
- `set_prize_cards_taken_for_knockout`

## Special conditions
- `apply_special_condition`
- `remove_special_condition`
- `prevent_special_condition`

## Energy
- `provide_energy`
- `modify_provided_energy`
- `attach_energy`
- `move_energy`
- `discard_energy`
- `return_energy_to_hand`
- `modify_energy_cost`

## Rule locks/modifiers
- `forbid_action`
- `allow_action`
- `ignore_rule`
- `set_flag`
- `clear_flag`
- `register_continuous_modifier`
- `register_trigger`
- `register_replacement_effect`

## Randomness and choices
- `coin_flip`
- `branch_on_result`
- `choose_target`
- `choose_amount`
- `order_cards`
- `declare_card_name`

## Turn/game flow
- `begin_turn`
- `draw_for_turn`
- `end_turn`
- `pass_turn`
- `pokemon_checkup`
- `check_knockouts`
- `check_win_loss`
