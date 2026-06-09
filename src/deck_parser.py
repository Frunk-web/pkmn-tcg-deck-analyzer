"""
Explanation

This file handles decklist parsing.

It takes a pasted decklist from Pokémon TCG Live, Limitless, or a similar source
and converts it into structured DeckCard objects.

Main responsibilities:
- Read each line of the decklist.
- Track whether each card came from the Pokémon, Trainer, or Energy section.
- Extract the card count.
- Extract the card name.
- Extract the set code and collector number when available.
- Handle special energy formatting such as:
  "Basic {G} Energy Energy 1" -> "Basic Grass Energy"
  "Basic {L} Energy SVE 12" -> "Basic Lightning Energy"
- Combine duplicate identical printings.
- Store API metadata fields such as card images once they are attached later.
"""

import re
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple


@dataclass
class DeckCard:
    name: str
    count: int
    section: Optional[str] = None
    set_code: Optional[str] = None
    collector_number: Optional[str] = None
    api_id: Optional[str] = None
    supertype: Optional[str] = None
    subtypes: Optional[List[str]] = None
    image_url: Optional[str] = None
    image_large_url: Optional[str] = None

    @property
    def key(self) -> str:
        set_part = self.set_code or "NOSET"
        num_part = self.collector_number or "NONUM"
        return f"{self.name}__{set_part}__{num_part}"

    @property
    def label(self) -> str:
        if self.set_code and self.collector_number:
            return f"{self.name} [{self.set_code} {self.collector_number}]"
        return self.name

    @property
    def is_basic_pokemon(self) -> bool:
        return (
            self.supertype == "Pokémon"
            and self.subtypes is not None
            and "Basic" in self.subtypes
        )


def normalize_apostrophes(text: str) -> str:
    return text.replace("’", "'").replace("`", "'").strip()


def parse_limitless_energy(raw_name: str):
    """
    Handles Basic Energy lines from deck exports.

    Examples:
    - Basic {G} Energy Energy 1  -> Basic Grass Energy [Energy 1]
    - Basic {L} Energy SVE 12    -> Basic Lightning Energy [SVE 12]
    - Basic {W} Energy SVE 11    -> Basic Water Energy [SVE 11]
    """

    energy_map = {
        "{G}": "Grass",
        "{R}": "Fire",
        "{W}": "Water",
        "{L}": "Lightning",
        "{P}": "Psychic",
        "{F}": "Fighting",
        "{D}": "Darkness",
        "{M}": "Metal",
        "{Y}": "Fairy",
        "{C}": "Colorless",
    }

    # Format 1:
    # Basic {G} Energy Energy 1
    match_old = re.match(
        r"^Basic\s+(\{[A-Z]\})\s+Energy(?:\s+Energy)?(?:\s+([A-Za-z0-9]+))?$",
        raw_name,
    )

    if match_old:
        symbol = match_old.group(1)
        number = match_old.group(2)
        energy_type = energy_map.get(symbol)

        if energy_type is None:
            return None

        return f"Basic {energy_type} Energy", "Energy", number

    # Format 2:
    # Basic {L} Energy SVE 12
    # Basic {W} Energy SVE 11
    # Basic {F} Energy SVE 14
    match_new = re.match(
        r"^Basic\s+(\{[A-Z]\})\s+Energy\s+([A-Z]{2,6})\s+([0-9]+[a-zA-Z]?)$",
        raw_name,
    )

    if match_new:
        symbol = match_new.group(1)
        set_code = match_new.group(2)
        number = match_new.group(3)
        energy_type = energy_map.get(symbol)

        if energy_type is None:
            return None

        return f"Basic {energy_type} Energy", set_code, number

    return None


def parse_card_name_set_number(raw_name: str) -> Tuple[str, Optional[str], Optional[str]]:
    raw_name = normalize_apostrophes(raw_name)

    energy_result = parse_limitless_energy(raw_name)
    if energy_result is not None:
        return energy_result

    text = re.sub(r"\s+\[[^\]]+\]$", "", raw_name).strip()

    # Supports normal set codes like TWM, SSP, TEF
    # and promo-style set codes like PR-SV.
    match = re.match(
        r"^(.*?)\s+([A-Z]{2,6}(?:-[A-Z]{2,6})?)\s+([0-9]+[a-zA-Z]?(/[0-9]+[a-zA-Z]?)?)$",
        text,
    )

    if match:
        name = match.group(1).strip()
        set_code = match.group(2).strip()
        collector_number = match.group(3).split("/")[0].strip()
        return name, set_code, collector_number

    return text, None, None


def parse_decklist(decklist_text: str) -> List[DeckCard]:
    cards: List[DeckCard] = []
    current_section: Optional[str] = None

    for line in decklist_text.splitlines():
        line = line.strip()

        if not line:
            continue

        lower_line = line.lower()

        if lower_line.startswith(("pokémon:", "pokemon:")):
            current_section = "Pokémon"
            continue

        if lower_line.startswith("trainer:"):
            current_section = "Trainer"
            continue

        if lower_line.startswith("energy:"):
            current_section = "Energy"
            continue

        if lower_line.startswith("total cards"):
            continue

        match = re.match(r"^(\d+)\s+(.+)$", line)

        if not match:
            continue

        count = int(match.group(1))
        raw_name = match.group(2).strip()

        name, set_code, collector_number = parse_card_name_set_number(raw_name)

        cards.append(
            DeckCard(
                name=name,
                count=count,
                set_code=set_code,
                collector_number=collector_number,
                section=current_section,
            )
        )

    combined: Dict[Tuple[str, Optional[str], Optional[str]], DeckCard] = {}

    for card in cards:
        key = (card.name, card.set_code, card.collector_number)

        if key not in combined:
            combined[key] = card
        else:
            combined[key].count += card.count

            if combined[key].section is None:
                combined[key].section = card.section

    return list(combined.values())