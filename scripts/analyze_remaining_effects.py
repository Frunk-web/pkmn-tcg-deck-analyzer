from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_COVERAGE = Path("data/reports/compiler_coverage.json")
DEFAULT_REVIEW_QUEUE = Path("data/reports/review_queue.csv")
DEFAULT_OUT_DIR = Path("data/reports/long_tail")

TURN1_RELEVANCE_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}


def normalize_space(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


FAMILIES: list[tuple[str, str, str, list[re.Pattern[str]]]] = [

    (
        "global_retreat_cost_rule",
        "medium",
        "Global or conditional retreat-cost rules should compile as board-wide retreat-cost modifiers.",
        [
            re.compile(r"each player pays .* less to retreat", re.I),
            re.compile(r"each player pays .* more to retreat", re.I),
            re.compile(r"must pay an additional .* to retreat", re.I),
            re.compile(r"additional .* to retreat", re.I),
            re.compile(r"Retreat Cost of each Pokémon", re.I),
            re.compile(r"Retreat Cost .* except for", re.I),
        ],
    ),
    (
        "tool_or_attachment_return_replacement",
        "medium",
        "Tool and attachment replacement/lifecycle effects should compile as attachment replacement rules.",
        [
            re.compile(r"when .* is attached .* retreats, discard .* instead", re.I),
            re.compile(r"if this card is discarded from play, put it into your hand instead", re.I),
            re.compile(r"when this card is removed from a Pokémon", re.I),
            re.compile(r"discard .* instead of discarding Energy", re.I),
        ],
    ),
    (
        "opponent_hand_shuffle_disruption",
        "low",
        "Opponent hand reveal/shuffle/bottom-deck disruption should compile as hand-disruption operations.",
        [
            re.compile(r"choose .* random cards? from your opponent's hand", re.I),
            re.compile(r"opponent reveals .* shuffles .* into .* deck", re.I),
            re.compile(r"choose a card .* put it on the bottom of .* deck", re.I),
            re.compile(r"opponent shuffles .* hand into .* deck", re.I),
        ],
    ),
    (
        "special_condition_global_immunity_or_recovery",
        "none",
        "Global Special Condition recovery or immunity effects should compile as status-condition rule modifiers.",
        [
            re.compile(r"each Pokémon .* recovers from all Special Conditions", re.I),
            re.compile(r"can't be affected by any Special Conditions", re.I),
            re.compile(r"recovers from all Special Conditions", re.I),
            re.compile(r"remove a Special Condition", re.I),
        ],
    ),
    (
        "pokemon_power_global_lock",
        "high",
        "Pokémon Power / Poké-Power / Poké-Body global locks should compile as ability-lock rule modifiers.",
        [
            re.compile(r"all Pokémon Powers stop working", re.I),
            re.compile(r"can't use any Poké-Powers or Poké-Bodies", re.I),
            re.compile(r"have no Abilities", re.I),
            re.compile(r"no Trainer cards can be played.*power stops working", re.I),
        ],
    ),
    (
        "basic_fossil_pokemon_rule",
        "medium",
        "Fossil cards played as Basic Pokémon should compile as special in-play Trainer/Pokémon rules.",
        [
            re.compile(r"play .* Fossil as if it were a Basic Pokémon", re.I),
            re.compile(r"counts as a Colorless Pokémon.*Trainer card", re.I),
            re.compile(r"doesn't count as a Knocked Out Pokémon", re.I),
        ],
    ),
    (
        "castform_form_switch_search",
        "high",
        "Castform-style form-search and switch abilities should compile as search-deck plus replace/switch-in-play effects.",
        [
            re.compile(r"search your deck for .*Castform.*switch it with", re.I),
            re.compile(r"Temperamental Weather", re.I),
            re.compile(r"shuffle .* back into your deck.*can't use more than", re.I),
        ],
    ),
    (
        "conditional_status_damage_bonus",
        "none",
        "Damage bonuses based on the Defending/Active Pokémon status or evolution state should compile as conditional damage modifiers.",
        [
            re.compile(r"if the Defending Pokémon is affected by a Special Condition", re.I),
            re.compile(r"if .* Active Pokémon is Confused", re.I),
            re.compile(r"if the Defending Pokémon is an Evolved Pokémon", re.I),
            re.compile(r"if the Defending Pokémon is a Stage 2", re.I),
            re.compile(r"if the Defending Pokémon already has at least", re.I),
        ],
    ),
    (
        "damage_counter_reflect_or_retaliation",
        "none",
        "Effects that place counters on the attacker after being damaged should compile as retaliation damage-counter triggers.",
        [
            re.compile(r"if .* is damaged by .* attack.*put .* damage counters? on the Attacking Pokémon", re.I),
            re.compile(r"during your opponent's next turn.*damaged by an attack.*put .* damage counters", re.I),
            re.compile(r"damage counters on the Attacking Pokémon equal to the damage", re.I),
        ],
    ),
    (
        "damage_counter_mass_placement",
        "none",
        "Mass damage-counter placement should compile as spread damage-counter operations.",
        [
            re.compile(r"put \d+ damage counters? on each of your opponent", re.I),
            re.compile(r"put .* damage counters? on each of your opponent", re.I),
            re.compile(r"count the number .* put that many damage counters", re.I),
            re.compile(r"until .* remaining HP is", re.I),
            re.compile(r"until it is .* HP away from being Knocked Out", re.I),
        ],
    ),
    (
        "weakness_resistance_global_rule",
        "none",
        "Weakness and Resistance global modifiers should compile as board-wide battle modifiers.",
        [
            re.compile(r"apply Weakness .* ×2", re.I),
            re.compile(r"has no Resistance", re.I),
            re.compile(r"has no Weakness", re.I),
            re.compile(r"don't apply Weakness and Resistance", re.I),
            re.compile(r"don't apply Weakness or Resistance", re.I),
        ],
    ),
    (
        "stadium_or_tool_discard",
        "medium",
        "Effects that discard Stadiums or Pokémon Tools should compile as tool/stadium discard operations.",
        [
            re.compile(r"if your opponent has a Stadium in play, discard it", re.I),
            re.compile(r"discard all Pokémon Tool cards", re.I),
            re.compile(r"choose up to .* Pokémon Tools .* discard", re.I),
            re.compile(r"discard that Stadium", re.I),
        ],
    ),
    (
        "prize_visibility_or_play_condition",
        "medium",
        "Prize-card visibility and prize-count play restrictions should compile as prize rules or play conditions.",
        [
            re.compile(r"Prize cards face up", re.I),
            re.compile(r"play with .* Prize cards face up", re.I),
            re.compile(r"only if you have more Prize cards remaining", re.I),
            re.compile(r"if you have more Prize cards remaining", re.I),
        ],
    ),
    (
        "old_pokemon_power_status_gated",
        "high",
        "Old Pokémon Powers gated by Special Conditions should compile as activated/static abilities with status-based play conditions.",
        [
            re.compile(r"This power can't be used if .* (?:Asleep|Confused|Paralyzed|affected by a Special Condition)", re.I),
            re.compile(r"as often as you like during your turn.*This power can't be used", re.I),
            re.compile(r"once during your turn \(before your attack\).*This power can't be used", re.I),
        ],
    ),

    (
        "own_pokemon_bounce_to_hand",
        "medium",
        "Effects returning your own Pokémon and attached cards to hand should compile as own-board bounce / move-zone operations.",
        [
            re.compile(r"put 1 of your Pokémon and all attached cards into your hand", re.I),
            re.compile(r"return 1 of your Pokémon and all cards attached to it to your hand", re.I),
            re.compile(r"shuffle this Pokémon and all attached cards into your deck", re.I),
            re.compile(r"put .* and all attached cards .* into your hand", re.I),
        ],
    ),
    (
        "deck_mill_or_topdeck_discard",
        "low",
        "Effects discarding cards from the top of a deck should compile as mill / deck-discard operations.",
        [
            re.compile(r"discard \d+ cards? from the top of .* deck", re.I),
            re.compile(r"discard .* from the top of .* deck", re.I),
            re.compile(r"for each heads, discard \d+ cards? from the top", re.I),
            re.compile(r"discard the top card from .* deck", re.I),
        ],
    ),
    (
        "opponent_item_supporter_trainer_lock",
        "medium",
        "Effects preventing Item/Supporter/Trainer play should compile as temporary trainer-lock rule modifiers.",
        [
            re.compile(r"opponent can't play any Item cards", re.I),
            re.compile(r"opponent can't play any Supporter cards", re.I),
            re.compile(r"opponent can't play any Trainer cards", re.I),
            re.compile(r"no Trainer cards can be played", re.I),
        ],
    ),
    (
        "opponent_attack_damage_reduction",
        "none",
        "Effects reducing damage from attacks should compile as delayed or continuous damage-reduction modifiers.",
        [
            re.compile(r"attacks used by your opponent.*do \d+ less damage", re.I),
            re.compile(r"damage done to .* by attacks is reduced by \d+", re.I),
            re.compile(r"during your opponent's next turn.*damage .* reduced by", re.I),
            re.compile(r"does \d+ less damage", re.I),
        ],
    ),
    (
        "copy_or_grant_attack_access",
        "medium",
        "Effects that let a Pokémon use another attack should compile as attack-copy / granted-attack access rules.",
        [
            re.compile(r"can use the attacks? of", re.I),
            re.compile(r"can also use the attack on this card", re.I),
            re.compile(r"may use this card's attack instead of its own", re.I),
            re.compile(r"can use any attack from its previous Evolutions", re.I),
        ],
    ),
    (
        "delayed_knockout_or_prize_bonus",
        "none",
        "Delayed KO and bonus-prize effects should compile as delayed knockout / prize rule modifiers.",
        [
            re.compile(r"will be Knocked Out", re.I),
            re.compile(r"take 1 more Prize", re.I),
            re.compile(r"take .* more Prize", re.I),
            re.compile(r"Knocked Out by damage.*take", re.I),
        ],
    ),
    (
        "special_energy_attachment_restriction",
        "high",
        "Special Energy attachment restrictions and replacement effects should compile as energy attachment constraints/modifiers.",
        [
            re.compile(r"can only be attached to", re.I),
            re.compile(r"if this card is attached to anything other than", re.I),
            re.compile(r"attached to a Pokémon.*provides Colorless Energy.*switch", re.I),
            re.compile(r"return an Energy card attached to that Pokémon to your hand", re.I),
        ],
    ),
    # ------------------------------------------------------------------
    # Specific high-frequency / high-clarity battle families first.
    # Ordering matters: the first matching family becomes primary.
    # These split older "other_long_tail" / overly broad buckets into
    # compiler-actionable families.
    # ------------------------------------------------------------------
    (
        "damage_retaliation_or_reflect",
        "none",
        "Retaliation / reflect damage formulas are common battle-text patterns and should compile into conditional damage modifiers.",
        [
            re.compile(r"was damaged by an attack during your opponent's last turn.*does that much more damage", re.I),
            re.compile(r"this attack does that much more damage", re.I),
            re.compile(r"does damage equal to the damage done to", re.I),
            re.compile(r"does the same amount of damage", re.I),
            re.compile(r"if .* was damaged .* last turn", re.I),
        ],
    ),
    (
        "attack_cost_modification",
        "none",
        "Attack-cost modifiers should compile as conditional cost modifiers, separate from Energy movement/discard effects.",
        [
            re.compile(r"ignore all Energy in this Pokémon's attack costs", re.I),
            re.compile(r"ignore .* Energy .* attack costs", re.I),
            re.compile(r"attack(?:s)? cost .* less", re.I),
            re.compile(r"attack(?:s)? cost .* more", re.I),
            re.compile(r"(?:no|without) Energy .* attack", re.I),
            re.compile(r"you still need the necessary Energy", re.I),
        ],
    ),
    (
        "coin_damage_plus_special_condition",
        "none",
        "Coin-flip attacks that both scale damage and apply a Special Condition are a reusable battle pattern.",
        [
            re.compile(r"flip \d+ coins?.*damage times the number of heads.*(?:Paralyzed|Poisoned|Asleep|Confused|Burned|Special Condition)", re.I),
            re.compile(r"flip \d+ coins?.*number of heads.*(?:Defending|Active) Pokémon is now", re.I),
            re.compile(r"if .* coins? .* heads.*(?:Paralyzed|Poisoned|Asleep|Confused|Burned)", re.I),
        ],
    ),
    (
        "gust_before_or_during_damage",
        "none",
        "Gust/switch-before-damage effects combine switching with attack resolution and should be separated from spread damage.",
        [
            re.compile(r"before doing damage.*switch .* Benched Pokémon", re.I),
            re.compile(r"before doing damage.*choose .* Benched Pokémon.*switch", re.I),
            re.compile(r"switch .* opponent's Benched Pokémon .* Active", re.I),
            re.compile(r"switch .* Defending Pokémon", re.I),
        ],
    ),
    (
        "damage_scaling_by_revealed_hand_or_trainers",
        "none",
        "Damage based on revealed hand contents is a conditional damage formula, not a Trainer rule.",
        [
            re.compile(r"opponent reveals their hand.*damage .* each Trainer", re.I),
            re.compile(r"look at your opponent's hand.*damage .* each", re.I),
            re.compile(r"for each Trainer card .* opponent", re.I),
            re.compile(r"for each .* card .* opponent's hand", re.I),
        ],
    ),
    (
        "energy_provides_or_type_modifier",
        "high",
        "Energy-providing/modifying effects are common and should compile as attachment-provided energy or modifier rules.",
        [
            re.compile(r"provides? (?:\[?[A-Za-z]+\]?|Colorless|all types of) Energy", re.I),
            re.compile(r"provides .* Energy", re.I),
            re.compile(r"counts as .* Energy", re.I),
            re.compile(r"as long as this card is attached.*provides", re.I),
        ],
    ),
    (
        "self_recovery_or_recycle_from_discard",
        "medium",
        "Self-recovery/recycle effects from discard should compile into discard-to-deck/hand movement effects.",
        [
            re.compile(r"if this Pokémon is in your discard pile.*(?:bottom|top) of your deck", re.I),
            re.compile(r"put this Pokémon on the bottom of your deck", re.I),
            re.compile(r"put this card from your discard pile", re.I),
            re.compile(r"shuffle .* from your discard pile into your deck", re.I),
        ],
    ),
    (
        "between_turns_damage_counter_healing",
        "none",
        "Between-turn damage-counter removal is a battle/status pattern separate from damage-counter placement.",
        [
            re.compile(r"at any time between turns.*remove \d+ damage counters?", re.I),
            re.compile(r"between turns.*remove .* damage counters?", re.I),
        ],
    ),
    (
        "healing_or_damage_counter_removal",
        "none",
        "Healing and damage-counter removal should compile into heal_damage / remove_damage_counters operations.",
        [
            re.compile(r"heal \d+ damage from", re.I),
            re.compile(r"remove \d+ damage counters? from", re.I),
            re.compile(r"remove .* damage counters?", re.I),
            re.compile(r"heal .* damage", re.I),
        ],
    ),
    (
        "retreat_cost_modification",
        "low",
        "Retreat-cost changes should compile as conditional retreat-cost modifiers, separate from attack costs.",
        [
            re.compile(r"has no Retreat Cost", re.I),
            re.compile(r"Retreat Cost .* is 0", re.I),
            re.compile(r"Retreat Cost .* less", re.I),
            re.compile(r"Retreat Cost .* more", re.I),
            re.compile(r"can't retreat", re.I),
            re.compile(r"cannot retreat", re.I),
        ],
    ),
    (
        "self_shuffle_or_return_to_deck",
        "medium",
        "Self shuffle / return-to-deck effects should compile into zone movement of this Pokémon and attached cards.",
        [
            re.compile(r"shuffle this Pokémon and all attached cards into your deck", re.I),
            re.compile(r"shuffle this Pokémon .* into your deck", re.I),
            re.compile(r"return this Pokémon and all cards attached to it to your hand", re.I),
            re.compile(r"put this Pokémon and all attached cards .* into your deck", re.I),
        ],
    ),
    (
        "status_condition_immunity",
        "none",
        "Special-condition immunity/prevention is a battle/status rule family.",
        [
            re.compile(r"can't be (?:Paralyzed|Poisoned|Asleep|Confused|Burned|affected by a Special Condition)", re.I),
            re.compile(r"cannot be (?:Paralyzed|Poisoned|Asleep|Confused|Burned)", re.I),
            re.compile(r"prevent .* Special Condition", re.I),
        ],
    ),
    (
        "opponent_draw_or_may_draw",
        "low",
        "Opponent draw effects should compile as opponent draw decisions but usually matter less for solo consistency.",
        [
            re.compile(r"your opponent draws? (?:a|\d+) cards?", re.I),
            re.compile(r"opponent may draw", re.I),
        ],
    ),
    (
        "future_damage_bonus_or_mark",
        "none",
        "Future-turn damage bonuses/marks are delayed battle modifiers.",
        [
            re.compile(r"until the end of your next turn.*attack does \d+ more damage", re.I),
            re.compile(r"during your next turn.*attack does \d+ more damage", re.I),
            re.compile(r"if an attack damages the Defending Pokémon.*does \d+ more damage", re.I),
        ],
    ),
    (
        "direct_damage_ignore_weakness_resistance",
        "none",
        "Direct damage that ignores Weakness/Resistance should compile as targeted damage with weakness/resistance flags.",
        [
            re.compile(r"does \d+ damage to 1 of your opponent's Pokémon.*isn't affected by Weakness or Resistance", re.I),
            re.compile(r"damage isn't affected by Weakness or Resistance", re.I),
            re.compile(r"don't apply Weakness and Resistance", re.I),
        ],
    ),
    (
        "attachment_duration_or_delayed_discard",
        "medium",
        "Delayed discard of an attached card is a Tool/attachment duration rule.",
        [
            re.compile(r"if this card is attached .* discard it at the end of", re.I),
            re.compile(r"discard it at the end of your opponent's turn", re.I),
            re.compile(r"discard this card at the end of", re.I),
        ],
    ),
    (
        "conditional_damage_vs_pokemon_ex_or_damaged",
        "none",
        "Conditional damage based on defending Pokémon subtype or existing damage counters should compile as damage modifiers.",
        [
            re.compile(r"if the Defending Pokémon is Pokémon-ex.*does .* more damage", re.I),
            re.compile(r"if the Defending Pokémon .* has any damage counters", re.I),
            re.compile(r"already has any damage counters.*more damage", re.I),
        ],
    ),
    (
        "bounce_defending_pokemon_to_hand",
        "none",
        "Bounce effects return the defending/active Pokémon and attachments to hand.",
        [
            re.compile(r"opponent returns the Defending Pokémon and all cards attached to it to .* hand", re.I),
            re.compile(r"return the Defending Pokémon and all cards attached", re.I),
            re.compile(r"put the Defending Pokémon and all attached cards into .* hand", re.I),
        ],
    ),
    (
        "copy_supporter_or_trainer_effect",
        "medium",
        "Copying Supporter/Trainer effects should compile as a copy-effect / deferred execution pattern.",
        [
            re.compile(r"search your opponent's discard pile for a Supporter card and use the effect", re.I),
            re.compile(r"use the effect of that card as the effect of this attack", re.I),
            re.compile(r"copy .* Supporter", re.I),
        ],
    ),
    (
        "mill_or_deck_discard",
        "low",
        "Deck discard/mill effects should compile as opponent/self deck-to-discard movement.",
        [
            re.compile(r"discard the top card from your opponent's deck", re.I),
            re.compile(r"discard the top \d+ cards? of .* deck", re.I),
            re.compile(r"put the top card of .* deck .* discard", re.I),
        ],
    ),

    (
        "item_or_trainer_lock",
        "none",
        "Item/Trainer/Supporter lock effects are persistent battle/control rules and should be split from generic Trainer card text.",
        [
            re.compile(r"can't play any Item cards?", re.I),
            re.compile(r"can't play any Trainer cards?", re.I),
            re.compile(r"can't play any Supporter cards?", re.I),
            re.compile(r"cannot play any Item cards?", re.I),
            re.compile(r"cannot play any Trainer cards?", re.I),
            re.compile(r"No Trainer cards can be played", re.I),
            re.compile(r"opponent can't play any .* cards? from (?:their|his or her) hand", re.I),
        ],
    ),
    (
        "ability_or_pokemon_power_lock",
        "medium",
        "Ability / Poké-Power / Poké-Body lock effects are global or conditional rule modifiers.",
        [
            re.compile(r"have no Abilities", re.I),
            re.compile(r"has no Abilities", re.I),
            re.compile(r"can't use any Pok[ée]-Powers?", re.I),
            re.compile(r"can't use any Pok[ée]-Bodies?", re.I),
            re.compile(r"Pok[ée]mon Powers? stop working", re.I),
            re.compile(r"no Pok[ée]mon Powers?", re.I),
            re.compile(r"prevent all effects of .* Abilities", re.I),
            re.compile(r"Colorless Pokémon .* have no Abilities", re.I),
        ],
    ),
    (
        "prize_rule_or_visibility",
        "medium",
        "Prize-card visibility, play restrictions, and bonus-prize effects should compile as prize/rule modifiers.",
        [
            re.compile(r"Prize cards? face up", re.I),
            re.compile(r"turn all of your Prize cards? face up", re.I),
            re.compile(r"plays? with .* Prize cards? face up", re.I),
            re.compile(r"more Prize cards? remaining than your opponent", re.I),
            re.compile(r"take 1 more Prize card", re.I),
            re.compile(r"doesn't count as a Knocked Out Pokémon", re.I),
        ],
    ),
    (
        "fossil_as_basic_pokemon_rule",
        "medium",
        "Old Fossil Trainer cards that play as Basic Pokémon have multi-zone identity rules and should be split out explicitly.",
        [
            re.compile(r"play .* Fossil as if it were a Basic Pokémon", re.I),
            re.compile(r"counts as a .* Pokémon \(as well as a Trainer card\)", re.I),
            re.compile(r"has no attacks of its own, can't retreat", re.I),
            re.compile(r"At any time during your turn before your attack, you may discard .* Fossil", re.I),
        ],
    ),
    (
        "tool_attachment_or_lifecycle_rule",
        "medium",
        "Tool attachment, ownership, removal, and lifecycle rules should compile as attachment constraints / delayed discard rules.",
        [
            re.compile(r"Attach .* to 1 of your Pokémon", re.I),
            re.compile(r"Attach this .* Tool", re.I),
            re.compile(r"doesn't already have a Pok[ée]mon Tool", re.I),
            re.compile(r"When this card is removed from a Pokémon", re.I),
            re.compile(r"If this card is discarded from play, put it into your hand", re.I),
            re.compile(r"can only be attached to", re.I),
            re.compile(r"if this card is attached to anything other than", re.I),
            re.compile(r"discard .* at the end of your turn", re.I),
            re.compile(r"discard this card at the end of your turn", re.I),
        ],
    ),
    (
        "tool_attack_or_vstar_grant",
        "medium",
        "Tools that grant attacks or VSTAR Powers are copy/granted-action effects distinct from attack-cost modifiers.",
        [
            re.compile(r"can use the VSTAR Power on this card", re.I),
            re.compile(r"may use this card's attack instead of its own", re.I),
            re.compile(r"can also use the attack on this card", re.I),
            re.compile(r"can use any attack from its previous Evolutions", re.I),
        ],
    ),
    (
        "tool_or_attachment_damage_modifier",
        "none",
        "Damage or stat modifiers granted by attached Tools should compile as continuous modifiers.",
        [
            re.compile(r"attacks? of the Pokémon this card is attached to do \d+ more damage", re.I),
            re.compile(r"Attacks used by the Pokémon this card is attached to do \d+ more damage", re.I),
            re.compile(r"Pokémon this card is attached to gets \+\d+ HP", re.I),
            re.compile(r"Pokémon this card is attached to has no Weakness", re.I),
            re.compile(r"Retreat Cost of the Pokémon this card is attached to", re.I),
            re.compile(r"If this Pokémon has a Pokémon Tool card attached", re.I),
        ],
    ),
    (
        "global_stat_or_type_rule_modifier",
        "medium",
        "Global Stadium/Tool rules that change HP, Weakness, Resistance, healing, or types should compile as continuous board modifiers.",
        [
            re.compile(r"gets? \+\d+ HP", re.I),
            re.compile(r"Apply Weakness .* as ×?x?2", re.I),
            re.compile(r"has no Resistance", re.I),
            re.compile(r"Pokémon .* can't be healed", re.I),
            re.compile(r"can't be healed", re.I),
            re.compile(r"all Special Energy attached .* provide Colorless Energy", re.I),
            re.compile(r"provide Colorless Energy and have no other effect", re.I),
        ],
    ),
    (
        "topdeck_look_choose_or_reorder",
        "high",
        "Top-deck look / choose / reorder effects should compile as deck peek, selection, reorder, or conditional draw operations.",
        [
            re.compile(r"Look at the top card of your deck", re.I),
            re.compile(r"Look at the top \d+ cards? of your deck", re.I),
            re.compile(r"put them back in any order", re.I),
            re.compile(r"choose 1 of them, and put it into your hand", re.I),
            re.compile(r"reveal .* you find there and put it into your hand", re.I),
            re.compile(r"put the other cards? .* back", re.I),
        ],
    ),
    (
        "discard_pile_to_hand_or_deck_recovery",
        "medium",
        "Discard-pile recovery to hand/deck should be separated from generic Energy movement.",
        [
            re.compile(r"put .* from your discard pile into your hand", re.I),
            re.compile(r"put .* cards? .* from your discard pile into your hand", re.I),
            re.compile(r"put .* Energy .* from your discard pile into your hand", re.I),
            re.compile(r"shuffle .* from your discard pile into your deck", re.I),
            re.compile(r"from your discard pile into your deck", re.I),
            re.compile(r"for each heads, put .* from your discard pile into your hand", re.I),
        ],
    ),
    (
        "bench_from_discard_or_deck",
        "high",
        "Effects that put Pokémon directly onto the Bench from deck/discard should compile as zone-to-bench movement.",
        [
            re.compile(r"put .* Pokémon from your discard pile onto your Bench", re.I),
            re.compile(r"put .* onto your Bench", re.I),
            re.compile(r"put them onto your Bench", re.I),
            re.compile(r"put up to \d+ .* onto your Bench", re.I),
            re.compile(r"put .* Basic Pokémon .* onto .* Bench", re.I),
            re.compile(r"Treat the new Benched Pokémon as Basic Pokémon", re.I),
        ],
    ),
    (
        "delayed_knockout_or_prize_bonus",
        "none",
        "Delayed KO and prize-bonus effects are battle resolution rules.",
        [
            re.compile(r"At the end of your opponent's next turn, .* Knocked Out", re.I),
            re.compile(r"will be Knocked Out", re.I),
            re.compile(r"if .* Knocked Out by damage .* take 1 more Prize", re.I),
            re.compile(r"Knocked Out by damage .* search your deck", re.I),
        ],
    ),
    (
        "self_damage_or_recoil",
        "none",
        "Self-damage / recoil clauses should compile as self-targeted damage or damage counters.",
        [
            re.compile(r"does \d+ damage to itself", re.I),
            re.compile(r"does \d+ damage to 1 of your Benched Pokémon", re.I),
            re.compile(r"does .* damage to .* your Benched", re.I),
            re.compile(r"put .* damage counters? .* this Pokémon", re.I),
        ],
    ),
    (
        "conditional_attack_availability",
        "none",
        "Effects that make an attack unusable under a condition should compile as attack availability restrictions.",
        [
            re.compile(r"you can't use this attack", re.I),
            re.compile(r"can't use this attack during your next turn", re.I),
            re.compile(r"this attack does nothing", re.I),
            re.compile(r"if .* evolved during this turn, this attack does nothing", re.I),
            re.compile(r"if .* isn't Burned, this attack does nothing", re.I),
            re.compile(r"if you go second, you can't use this attack during your first turn", re.I),
        ],
    ),
    # ------------------------------------------------------------------
    # General setup/card-access families.
    # ------------------------------------------------------------------
    (
        "search_deck",
        "high",
        "Search/look-at-deck effects can directly change Turn-1 access lines.",
        [
            re.compile(r"search your deck", re.I),
            re.compile(r"look at the top \d+ cards", re.I),
            re.compile(r"look at the bottom \d+ cards", re.I),
            re.compile(r"reveal .* put .* into your hand", re.I),
        ],
    ),
    (
        "draw_shuffle_hand",
        "high",
        "Draw/refresh effects directly affect opening-hand and Turn-1 consistency.",
        [
            re.compile(r"draw \d+ cards?", re.I),
            re.compile(r"draw a card", re.I),
            re.compile(r"draw cards until", re.I),
            re.compile(r"shuffle your hand into your deck", re.I),
        ],
    ),
    (
        "energy_attachment_or_acceleration",
        "high",
        "Energy attach/acceleration matters for setup goals and can be a cost/enabler.",
        [
            re.compile(r"attach .* Energy .* from (?:your )?(?:discard pile|deck|hand)", re.I),
            re.compile(r"search your deck for .* Energy .* attach", re.I),
        ],
    ),
    (
        "energy_discard_move_or_bounce",
        "high",
        "Energy discard/move/bounce is often a cost or setup transformation.",
        [
            re.compile(r"discard .* Energy", re.I),
            re.compile(r"move .* Energy", re.I),
            re.compile(r"put .* Energy .* into (?:their|your) hand", re.I),
            re.compile(r"attached to .* Energy", re.I),
            re.compile(r"Basic [A-Za-z]+ Energy", re.I),
        ],
    ),
    (
        "evolution_devolution_or_levelup",
        "high",
        "Evolution effects matter for setup-package goals like Basic + evolution + Wally.",
        [
            re.compile(r"evolves? from", re.I),
            re.compile(r"evolve", re.I),
            re.compile(r"devolve", re.I),
            re.compile(r"LV\.X", re.I),
            re.compile(r"previous level", re.I),
        ],
    ),
    (
        "discard_recovery",
        "medium",
        "Discard recovery can matter after costs, but is usually less central than draw/search.",
        [
            re.compile(r"from your discard pile into your hand", re.I),
            re.compile(r"put .* from your discard pile", re.I),
            re.compile(r"search your discard pile", re.I),
        ],
    ),
    (
        "trainer_tool_stadium_rules",
        "medium",
        "Trainer/Stadium/Tool restrictions can affect whether a line is legal.",
        [
            re.compile(r"Stadium", re.I),
            re.compile(r"Pokémon Tool", re.I),
            re.compile(r"Trainer card", re.I),
            re.compile(r"Retreat Cost .* attached", re.I),
            re.compile(r"last card in your hand", re.I),
            re.compile(r"only if .* Knocked Out", re.I),
        ],
    ),
    (
        "switch_or_gust",
        "medium",
        "Switching can matter for active/bench setup goals, but not basic starting-hand statements.",
        [
            re.compile(r"switch .* Active Pokémon", re.I),
            re.compile(r"switch in .* Benched Pokémon", re.I),
            re.compile(r"opponent switches", re.I),
            re.compile(r"new Active Pokémon", re.I),
        ],
    ),
    (
        "hand_disruption",
        "low",
        "Opponent-hand effects are usually not relevant to solo Turn-1 consistency diagnostics.",
        [
            re.compile(r"opponent's hand", re.I),
            re.compile(r"opponent reveals", re.I),
            re.compile(r"without looking", re.I),
            re.compile(r"random card from your opponent", re.I),
            re.compile(r"opponent discards? \d+ cards? from their hand", re.I),
            re.compile(r"look at your opponent's hand", re.I),
        ],
    ),
    (
        "gx_vstar_rule_or_special_once",
        "medium",
        "Once-per-game/once-per-turn rules can matter if they gate draw/search/setup effects.",
        [
            re.compile(r"GX attack", re.I),
            re.compile(r"VSTAR Power", re.I),
            re.compile(r"can't use more than 1", re.I),
            re.compile(r"once during your turn", re.I),
        ],
    ),
    (
        "return_to_hand_or_deck",
        "low",
        "Bounce effects are usually full-game or attack-line effects, rarely core starting setup.",
        [
            re.compile(r"return this Pokémon", re.I),
            re.compile(r"put this Pokémon and all attached cards into your hand", re.I),
            re.compile(r"shuffle this Pokémon and all cards attached", re.I),
            re.compile(r"put a card from your discard pile on top of your deck", re.I),
        ],
    ),
    (
        "attack_lock_or_attack_tax",
        "none",
        "Attack locks/taxes are battle simulation concerns, not starting-hand consistency.",
        [
            re.compile(r"can't (?:use )?attacks?", re.I),
            re.compile(r"can't attack", re.I),
            re.compile(r"tries to (?:use an )?attack", re.I),
            re.compile(r"attack doesn't happen", re.I),
            re.compile(r"opponent flips a coin.*(?:attack|turn ends)", re.I),
        ],
    ),
    (
        "damage_prevention_or_reduction",
        "none",
        "Damage prevention is important for full battle simulation, not Turn-1 access.",
        [
            re.compile(r"prevent all (?:damage|effects)", re.I),
            re.compile(r"prevent .* damage", re.I),
            re.compile(r"reduced by \d+", re.I),
            re.compile(r"takes? \d+ less damage", re.I),
            re.compile(r"not knocked out", re.I),
            re.compile(r"remaining HP becomes", re.I),
            re.compile(r"has no Weakness", re.I),
        ],
    ),
    (
        "damage_scaling_or_conditional_damage",
        "none",
        "Damage formulas are battle simulation concerns, not starting-hand consistency.",
        [
            re.compile(r"does \d+ (?:more )?damage for each", re.I),
            re.compile(r"does \d+ damage times", re.I),
            re.compile(r"plus \d+ more damage for each", re.I),
            re.compile(r"already has any damage counters.*does \d+ more damage", re.I),
            re.compile(r"if .* this attack does \d+ more damage", re.I),
            re.compile(r"same number of cards in your hand", re.I),
            re.compile(r"more cards in your hand", re.I),
            re.compile(r"for each .* attached", re.I),
            re.compile(r"for each .* in (?:your|opponent's) discard", re.I),
            re.compile(r"for each .* in play", re.I),
            re.compile(r"for each .* Retreat Cost", re.I),
        ],
    ),
    (
        "spread_or_bench_damage",
        "none",
        "Bench/spread damage is battle simulation, not card-access consistency.",
        [
            re.compile(r"does .* damage to .* Benched Pokémon", re.I),
            re.compile(r"damage to each of your opponent's Pokémon", re.I),
            re.compile(r"each of your opponent's Pokémon", re.I),
            re.compile(r"each of your own Benched", re.I),
            re.compile(r"Choose \d+ of your opponent's Pokémon", re.I),
        ],
    ),
    (
        "damage_counters",
        "none",
        "Damage-counter placement is battle simulation, not starting-hand consistency.",
        [
            re.compile(r"put \d+ damage counters?", re.I),
            re.compile(r"move all damage counters", re.I),
            re.compile(r"until its remaining HP is", re.I),
            re.compile(r"remove \d+ damage counters?", re.I),
            re.compile(r"damage counters .* any way you like", re.I),
        ],
    ),
    (
        "special_conditions",
        "none",
        "Special Conditions are battle simulation concerns for this milestone.",
        [
            re.compile(r"is now (?:Asleep|Confused|Paralyzed|Poisoned|Burned)", re.I),
            re.compile(r"Special Conditions?", re.I),
            re.compile(r"choose (?:1|a) Special Condition", re.I),
        ],
    ),
    (
        "copy_or_use_other_attack",
        "none",
        "Copying attacks belongs to full battle simulation.",
        [
            re.compile(r"use .* as this attack", re.I),
            re.compile(r"can use the attack", re.I),
            re.compile(r"copy", re.I),
        ],
    ),
]

# Families that should generally be prioritized for Turn-1 consistency work.
CORE_FAMILIES = {family for family, relevance, _, _ in FAMILIES if relevance in {"high", "medium"}}


def classify_text(text: str) -> tuple[str, list[str], str, str]:
    text = normalize_space(text)
    matches: list[str] = []
    relevance_labels: list[str] = []
    notes: list[str] = []
    for family, turn1_relevance, turn1_note, patterns in FAMILIES:
        if any(p.search(text) for p in patterns):
            matches.append(family)
            relevance_labels.append(turn1_relevance)
            notes.append(turn1_note)
    if not matches:
        return "other_long_tail", [], "unknown", "Needs review; Turn-1 impact is not known yet."
    primary = matches[0]
    best_relevance = max(relevance_labels, key=lambda x: TURN1_RELEVANCE_RANK.get(x, 0))
    return primary, matches, best_relevance, notes[0]


def priority_for(count: int, family: str, all_families: list[str], turn1_relevance: str) -> str:
    """Global compiler-priority label for a single unparsed text row.

    This is intentionally based on frequency, not Turn-1 relevance. Turn-1
    relevance is still reported as a separate column, but compiler roadmap work
    should primarily follow repeated effect families across the whole corpus.
    """
    if count >= 25:
        return "high"
    if count >= 10:
        return "medium"
    if count >= 4:
        return "low"
    return "watch"


def read_coverage(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_unparsed_blob(blob: str) -> list[str]:
    blob = normalize_space(blob)
    if not blob:
        return []
    parts = [normalize_space(x) for x in blob.split(" || ")]
    return [x for x in parts if x]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify remaining unparsed Pokémon TCG effects into mechanic families.")
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--review-queue", type=Path, default=DEFAULT_REVIEW_QUEUE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-n", type=int, default=200)
    args = parser.parse_args()

    coverage = read_coverage(args.coverage)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    classified_rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    family_unique_texts: Counter[str] = Counter()
    family_turn1_counts: Counter[str] = Counter()
    family_samples: dict[str, str] = {}
    family_notes: dict[str, str] = {}

    top_unparsed = coverage.get("top_unparsed_text", []) or []
    for item in top_unparsed[: args.top_n]:
        text = normalize_space(item.get("text", ""))
        count = int(item.get("count", 0) or 0)
        primary, all_families, turn1_relevance, turn1_note = classify_text(text)
        priority = priority_for(count, primary, all_families, turn1_relevance)
        recognition_status = "unknown_unclassified" if primary == "other_long_tail" else "recognized_family"
        compiler_readiness = "recognized_not_executable" if primary != "other_long_tail" else "unrecognized_review_needed"
        classified_rows.append({
            "count": count,
            "priority": priority,
            "recognition_status": recognition_status,
            "compiler_readiness": compiler_readiness,
            "turn1_relevance": turn1_relevance,
            "primary_family": primary,
            "all_families": "|".join(all_families),
            "turn1_note": turn1_note,
            "text": text,
        })
        family_counts[primary] += count
        family_unique_texts[primary] += 1
        family_turn1_counts[turn1_relevance] += count
        family_samples.setdefault(primary, text)
        family_notes.setdefault(primary, turn1_note)

    write_csv(
        out_dir / "remaining_unparsed_text_classified.csv",
        classified_rows,
        ["count", "priority", "recognition_status", "compiler_readiness", "turn1_relevance", "primary_family", "all_families", "turn1_note", "text"],
    )

    family_rows = []
    for family, total_count in family_counts.most_common():
        # Choose highest relevance observed in this family from classified rows.
        observed = [r["turn1_relevance"] for r in classified_rows if r["primary_family"] == family]
        best_relevance = max(observed, key=lambda x: TURN1_RELEVANCE_RANK.get(x, 0)) if observed else "unknown"
        family_rows.append({
            "primary_family": family,
            "turn1_relevance": best_relevance,
            "weighted_occurrences_in_top_unparsed": total_count,
            "unique_texts_in_top_unparsed": family_unique_texts[family],
            "sample_text": family_samples.get(family, ""),
            "turn1_note": family_notes.get(family, ""),
        })

    write_csv(
        out_dir / "remaining_effect_family_summary.csv",
        family_rows,
        ["primary_family", "turn1_relevance", "weighted_occurrences_in_top_unparsed", "unique_texts_in_top_unparsed", "sample_text", "turn1_note"],
    )

    review_rows: list[dict[str, Any]] = []
    review_family_counts: Counter[str] = Counter()
    review_turn1_counts: Counter[str] = Counter()

    if args.review_queue.exists():
        with args.review_queue.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts = split_unparsed_blob(row.get("unparsed_text", ""))
                families = []
                relevances = []
                for text in texts:
                    primary, all_families, turn1_relevance, _ = classify_text(text)
                    families.append(primary)
                    relevances.append(turn1_relevance)
                    review_family_counts[primary] += 1
                    review_turn1_counts[turn1_relevance] += 1
                primary_family = families[0] if families else "none"
                max_relevance = max(relevances, key=lambda x: TURN1_RELEVANCE_RANK.get(x, 0)) if relevances else "none"
                review_rows.append({
                    "effect_group_id": row.get("effect_group_id", ""),
                    "representative_card_id": row.get("representative_card_id", ""),
                    "name": row.get("name", ""),
                    "supertype": row.get("supertype", ""),
                    "subtypes": row.get("subtypes", ""),
                    "same_effect_printing_count": row.get("same_effect_printing_count", ""),
                    "unparsed_text_count": row.get("unparsed_text_count", ""),
                    "primary_family": primary_family,
                    "turn1_relevance": max_relevance,
                    "all_families": "|".join(sorted(set(families))),
                    "unparsed_text": row.get("unparsed_text", ""),
                })

        write_csv(
            out_dir / "review_queue_classified.csv",
            review_rows,
            [
                "effect_group_id",
                "representative_card_id",
                "name",
                "supertype",
                "subtypes",
                "same_effect_printing_count",
                "unparsed_text_count",
                "primary_family",
                "turn1_relevance",
                "all_families",
                "unparsed_text",
            ],
        )

    recognized_weighted = sum(int(r.get("count", 0) or 0) for r in classified_rows if r.get("primary_family") != "other_long_tail")
    unknown_weighted = sum(int(r.get("count", 0) or 0) for r in classified_rows if r.get("primary_family") == "other_long_tail")
    recognized_unique = sum(1 for r in classified_rows if r.get("primary_family") != "other_long_tail")
    unknown_unique = sum(1 for r in classified_rows if r.get("primary_family") == "other_long_tail")

    recognition_summary = {
        "note": "This is reporting-only recognition. It does not change compiler complete/partial status and should not be treated as simulator-executable coverage.",
        "complete_executable_cards": (coverage.get("status_counts") or {}).get("complete"),
        "partial_cards": (coverage.get("status_counts") or {}).get("partial"),
        "recognized_but_not_executable_weighted_top_unparsed": recognized_weighted,
        "unknown_unclassified_weighted_top_unparsed": unknown_weighted,
        "recognized_but_not_executable_unique_top_unparsed": recognized_unique,
        "unknown_unclassified_unique_top_unparsed": unknown_unique,
    }

    with (out_dir / "effect_recognition_summary.json").open("w", encoding="utf-8") as f:
        json.dump(recognition_summary, f, ensure_ascii=False, indent=2)

    summary = {
        "source_coverage": str(args.coverage),
        "source_review_queue": str(args.review_queue) if args.review_queue.exists() else None,
        "coverage": coverage.get("coverage"),
        "status_counts": coverage.get("status_counts"),
        "top_unparsed_items_classified": len(classified_rows),
        "turn1_relevance_weighted_counts": dict(family_turn1_counts),
        "family_summary": family_rows,
        "review_queue_family_counts": review_family_counts.most_common(),
        "review_queue_turn1_relevance_counts": review_turn1_counts.most_common(),
        "recognition_summary": recognition_summary,
        "outputs": {
            "remaining_unparsed_text_classified": str(out_dir / "remaining_unparsed_text_classified.csv"),
            "remaining_effect_family_summary": str(out_dir / "remaining_effect_family_summary.csv"),
            "review_queue_classified": str(out_dir / "review_queue_classified.csv") if review_rows else None,
            "effect_recognition_summary": str(out_dir / "effect_recognition_summary.json"),
        },
    }

    with (out_dir / "long_tail_review_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "status_counts": summary["status_counts"],
        "coverage": summary["coverage"],
        "turn1_relevance_weighted_counts": summary["turn1_relevance_weighted_counts"],
        "recognition_summary": recognition_summary,
        "top_family_summary": family_rows[:15],
        "outputs": summary["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
