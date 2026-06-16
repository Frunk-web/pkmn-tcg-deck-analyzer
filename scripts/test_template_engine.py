from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tcgcompiler import default_template_engine  # noqa: E402


EXAMPLES = [
    "Heal 80 damage from your Active Pokémon that has 3 or more Energy attached.",
    "Remove 3 damage counters from each of your Benched Pokémon.",
    "Search your deck for up to 2 basic Energy cards, reveal them, and put them into your hand. Shuffle your deck afterward.",
    "Once during each player's turn, that player may search their deck for an Evolution Grass Pokémon, reveal it, and put it into their hand. Then, that player shuffles their deck.",
    "Once during each player's turn, that player may switch their Active Water Pokémon with 1 of their Benched Water Pokémon.",
    "During your opponent's next turn, any damage done to Wigglytuff by attacks is reduced by 10 (after applying Weakness and Resistance).",
    "Each Pokémon that has any Energy attached (both yours and your opponent's) recovers from all Special Conditions and can't be affected by any Special Conditions.",
    "Don't apply Weakness and Resistance.",
    "The Retreat Cost of each Pokémon in play (except for Team Aqua Pokémon) is Colorless more.",
    "If your opponent has any Evolved Pokémon in play, remove the highest Stage Evolution card from each of them and put those cards back into his or her hand.",
    "Mew ex can use the attacks of all Pokémon in play as its own. (You still need the necessary Energy to use each attack.)",
    "Flip 2 coins. This attack does 20 times the number of heads.",
    "During your next turn, if an attack does damage to the Defending Pokémon (after applying Weakness and Resistance), that attack does 40 more damage.",
    "Your opponent shuffles his or her hand into his or her deck, then draws 7 cards.",
    "Shuffle up to 5 basic Energy cards from your discard pile into your deck.",
    "You may move a Fire Energy card attached to Gardevoir ex to 1 of your Benched Pokémon.",
    "Once during your turn (before your attack), you may draw cards until you have 5 cards in your hand.",
    "Prevent all effects of attacks, including damage, done to this Pokémon by Pokémon-EX.",
    "Put this card onto your Bench only with the effect of Old Amber Aerodactyl",
    "When the Pokémon Balloon Berry is attached to retreats, discard Balloon Berry instead of discarding Energy cards.",
    "Your opponent discards the top card of his or her deck.",
    "Flip a coin. If heads, search your deck for any 1 card and put it into your hand. Shuffle your deck afterward.",
    "Does 20 damage plus 10 more damage for each Grass Energy attached to all of your Pokémon.",
    "Discard all Energy attached to Raichu.",
    "You may discard a Fire Energy attached to this Pokémon. If you do, this attack does 30 more damage.",
    "If this Pokémon has no Energy attached to it, this Pokémon has no Retreat Cost.",
    "Heal all damage from all of your Pokémon.",
    "When you play this Pokémon from your hand to evolve 1 of your Pokémon during your turn, you may search your deck for Shedinja and put it onto your Bench. Shuffle your deck afterward.",
    "As often as you like during your turn (before your attack), you may attach 1 Water Energy card to 1 of your Water Pokémon. (This doesn't use up your 1 Energy card attachment for the turn.) This power can't be used if Blastoise is Asleep, Confused, or Paralyzed.",
    "Zangoose can't be affected by any Special Conditions.",
    "Choose 1 of your opponent's Pokémon's attacks and use it as this attack.",

    "Whenever your opponent flips a coin during his or her next turn, treat it as tails.",
    "Flip a coins. This attack does 30 damage times the number of heads. If you get 2 or more heads, Dark Vileplume is now Confused (after doing damage).",
    "Choose 3 of your opponent's Pokémon. This attack does 10 damage to each of those Pokémon. (Don't apply Weakness and Resistance for Benched Pokémon.)",
    "During your opponent's next turn, your opponent can't use any Poké-Powers on his or her Pokémon.",
    "During your opponent's next turn, when your opponent puts a Basic Pokémon from his or her hand onto his or her Bench, put 2 damage counters on that Pokémon.",
    "If the Defending Pokémon tries to attack during your opponent's next turn, your opponent flips a coin. If tails, this attack does nothing.",
    "As long as this Pokémon is your Active Pokémon, your opponent's Active Pokémon can't retreat.",
    "Once during your turn (before your attack), you may search your deck for a Grass Pokémon, reveal it, and put it into your hand. Then, shuffle your deck.",
    "Attach any number of Water Energy cards from your hand to your Pokémon in any way you like.",
    "You must discard a card from your hand in order to use this Ability. Once during your turn, you may draw 3 cards.",
    "Damage from this Pokémon's attacks isn't affected by Weakness or Resistance.",
    "If your opponent's Active Pokémon is an Ultra Beast, it is Knocked Out.",
    "You may do 100 more damage. If you do, during your next turn, this Pokémon can't attack.",
    "As long as this Pokémon is in play, it is Psychic and Fighting type.",

    # v0.6 current report leftovers
    "Look at the top 7 cards of your deck. You may reveal a Pokémon you find there and put it into your hand. Shuffle the other cards back into your deck.",
    "Rules: If you play this card from your hand, remove 1 damage counter from the Pokémon you attach it to, if it has any. Potion Energy provides Colorless energy. (Doesn't count as a basic Energy card.)",
    "Once during your turn (before your attack), you may put Hitmonlee, Hitmonchan, or Hitmontop from your hand onto Tyrogue (this counts as evolving Tyrogue) and remove all damage counters from Tyrogue.",
    "Flip a coin. If heads, put damage counters on the Defending Pokémon until it is 10 HP away from being Knocked Out.",
    "Flip a coin. If heads, this attack does 10 damage to each of your opponent's Benched Pokémon. If tails, this attack does 10 damage to each of your Benched Pokémon. (Don't apply Weakness and Resistance for Benched Pokémon.)",
    "Choose as many Energy cards from your hand as you like and show them to your opponent. This attack does 20 damage times the number of Energy cards you chose. Put those Energy cards on top of your deck. Shuffle your deck afterward.",
    "Once during your turn (before your attack), when you put Luxray GL LV.X from your hand onto your Active Luxray GL, you may switch the Defending Pokémon with 1 of your opponent's Benched Pokémon.",
    "As long as this Pokémon is your Active Pokémon, your opponent can't play any Item cards from his or her hand.",
    "Once during your turn (before your attack), you may switch your Active Pokémon with 1 of your Benched Pokémon. If you do, your opponent switches his or her Active Pokémon with 1 of his or her Benched Pokémon.",
    "If this Pokémon has a Pokémon Tool card attached to it, each Pokémon in play, in each player's hand, and in each player's discard pile has no Abilities (except for Garbotoxin).",
    "Once during your turn (before your attack), you may put a Pokémon Tool card attached to this Pokémon into your hand.",
    "Heal 20 damage and remove all Special Conditions from this Pokémon.",
    "If this Pokémon is in your hand when you are setting up to play, you may put it face down as your Active Pokémon.",
    "Your Pokémon in play have no Weakness.",
    "The Defending Pokémon can't be healed during your opponent's next turn.",
]


def main() -> None:
    engine = default_template_engine()
    rows = []
    for text in EXAMPLES:
        match = engine.match_first(text)
        rows.append({
            "text": text,
            "matched": match is not None,
            "family": None if match is None else match.family,
            "template_id": None if match is None else match.template_id,
            "ops": [] if match is None else [s.get("op") for s in match.steps],
        })
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    if not all(r["matched"] for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
