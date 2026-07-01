from __future__ import annotations

import re
from typing import Iterable

from .models import CardRef, LogEvent


_CARD_TOKEN_RE = re.compile(r"\((?P<id>[^)]+)\)\s*(?P<name>.+?)\s*(?=$|,|\.|\s+-\s+|\s+to\s+|\s+on\s+|\s+in\s+|\s+from\s+|\s+was\s+|\s+were\s+|\s+used\s+|\s+for\s+|\s+and played\s+|\s+with\s+|\s+is now\s+)")
_TURN_RE = re.compile(r"^(?P<player>.+?)'s Turn$")
_DRAW_N_RE = re.compile(r"drew (?P<n>\d+) cards?")
_TAKE_PRIZE_RE = re.compile(r"took (?P<n>\d+|a) Prize cards?", re.IGNORECASE)
_DAMAGE_RE = re.compile(r"for (?P<n>\d+) damage", re.IGNORECASE)


def parse_card_refs(text: str) -> list[CardRef]:
    cards: list[CardRef] = []

    for m in _CARD_TOKEN_RE.finditer(text or ""):
        exported_id = m.group("id").strip()
        name = m.group("name").strip()
        raw = m.group(0).strip()
        if exported_id or name:
            cards.append(CardRef(exported_id=exported_id, name=name, raw=raw))

    return cards


def _actor_before(text: str, phrase: str) -> str:
    if phrase not in text:
        return ""
    return text.split(phrase, 1)[0].strip()


def _possessive_actor(text: str) -> str:
    m = re.match(r"^(?P<player>.+?)'s\s+", text)
    if m:
        return m.group("player").strip()
    return ""


def _target_player_from_possessive(text: str) -> str:
    # Works for lines like:
    # BananaHammer33's (sv8-5_4) Budew was Knocked Out!
    return _possessive_actor(text)


def _prize_amount(text: str) -> int:
    m = _TAKE_PRIZE_RE.search(text)
    if not m:
        return 0
    raw = m.group("n").lower()
    return 1 if raw == "a" else int(raw)


def _damage_amount(text: str) -> int | None:
    m = _DAMAGE_RE.search(text)
    if not m:
        return None
    return int(m.group("n"))


def _damage_counter_amount(text: str) -> int | None:
    """
    Convert PTCGL damage-counter wording to damage.

    Examples:
      "put a damage counter on ..." -> 10
      "put 5 damage counters on ..." -> 50
      "put 13 damage counters on ..." -> 130
    """
    m = re.search(r"put\s+(?P<n>\d+)\s+damage counters?", text, re.IGNORECASE)
    if m:
        return int(m.group("n")) * 10

    if re.search(r"put\s+a\s+damage counter", text, re.IGNORECASE):
        return 10

    return None



def _zone_hint(text: str) -> str:
    clean = str(text or "")
    if "Active Spot" in clean or "Active Pokémon" in clean:
        return "active"
    if "Bench" in clean:
        return "bench"
    return ""


def _target_owner_hint(text: str) -> str:
    """
    For lines like:
      FrunkUke's Raging Bolt used ... on BananaHammer33’s Budew
    the final possessive card owner is usually the target owner.
    """
    pattern = re.compile(r"(?P<player>[A-Za-z0-9_ .-]+)[\'’]s\s+\([^)]+\)")
    matches = [m.group("player").strip() for m in pattern.finditer(text or "")]
    return matches[-1] if matches else ""


def _damage_counter_amount(text: str) -> int | None:
    m = re.search(r"put\s+(?P<n>\d+)\s+damage counters?", text or "", re.IGNORECASE)
    if m:
        return int(m.group("n")) * 10

    if re.search(r"put\s+a\s+damage counter", text or "", re.IGNORECASE):
        return 10

    return None


def parse_battle_log(raw_log: str) -> list[LogEvent]:
    events: list[LogEvent] = []
    current_turn_player = ""
    previous_event: LogEvent | None = None

    lines = (raw_log or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        event_type = "note"
        actor = ""
        amount = None
        metadata: dict = {}
        cards = parse_card_refs(line)

        turn_match = _TURN_RE.match(line)
        if line == "Setup":
            event_type = "setup"
        elif turn_match:
            event_type = "turn_start"
            actor = turn_match.group("player").strip()
            current_turn_player = actor
        elif line.startswith("•"):
            event_type = "revealed_cards"
            actor = previous_event.actor if previous_event else current_turn_player
            metadata["parent_event_type"] = previous_event.event_type if previous_event else ""
            metadata["parent_index"] = previous_event.index if previous_event else None
        elif line.startswith("-"):
            actor = previous_event.actor if previous_event else current_turn_player
            detail = line[1:].strip()

            if " put " in detail and "damage counter" in detail and " on " in detail:
                event_type = "place_damage_counters"
                actor = detail.split(" put ", 1)[0].strip()
                amount = _damage_counter_amount(detail)
            elif "drew" in detail and ("played it to the Bench" in detail or "played them to the Bench" in detail):
                event_type = "draw_and_play_to_bench"
                actor = detail.split(" drew ", 1)[0].strip()
                amount = len(cards) if cards else None
            elif "drew" in detail and cards:
                event_type = "draw_revealed"
                actor = detail.split(" drew ", 1)[0].strip()
                amount = len(cards)
            elif re.search(r"\bdrew\s+\d+\s+cards?\b", detail):
                event_type = "draw_count"
                actor = detail.split(" drew ", 1)[0].strip() if " drew " in detail else actor
                m = re.search(r"\bdrew\s+(?P<n>\d+)\s+cards?\b", detail)
                amount = int(m.group("n")) if m else None
            elif "drawn cards" in line:
                event_type = "draw_count"
                amount = int(re.search(r"(\d+)", line).group(1)) if re.search(r"(\d+)", line) else None
            elif "was discarded from" in line or "were discarded from" in line:
                event_type = "discard_from_play"
            elif "discarded" in line:
                event_type = "discard_count"
                amount = int(re.search(r"(\d+)", line).group(1)) if re.search(r"(\d+)", line) else None
            elif "attached" in line:
                event_type = "attach_energy"
                actor = detail.split(" attached ", 1)[0].strip() if " attached " in detail else actor
            elif "moved" in line:
                event_type = "move_card"
                actor = detail.split(" moved ", 1)[0].strip() if " moved " in detail else actor
            elif "shuffled" in line:
                event_type = "shuffle_deck"
            elif "Damage breakdown" in line:
                event_type = "damage_breakdown_start"
            elif "damage" in line.lower():
                event_type = "damage_breakdown"
                amount = _damage_amount(line)
            else:
                event_type = "effect_detail"
        elif " chose " in line and "opening coin flip" in line:
            event_type = "coin_choice"
            actor = line.split(" chose ", 1)[0].strip()
        elif " won the coin toss" in line:
            event_type = "coin_winner"
            actor = line.split(" won the coin toss", 1)[0].strip()
        elif " decided to go " in line:
            event_type = "turn_order_choice"
            actor = line.split(" decided to go ", 1)[0].strip()
            metadata["choice"] = line.rsplit(" ", 1)[-1].rstrip(".")
        elif "drew 7 cards for the opening hand" in line:
            event_type = "opening_hand"
            actor = line.split(" drew 7 cards", 1)[0].strip()
            amount = 7
        elif " drew a card" in line:
            event_type = "draw_hidden"
            actor = line.split(" drew a card", 1)[0].strip()
            amount = 1
        elif " drew " in line and cards:
            event_type = "draw_revealed"
            actor = line.split(" drew ", 1)[0].strip()
            amount = len(cards)
        elif " drew " in line:
            event_type = "draw_count"
            actor = line.split(" drew ", 1)[0].strip()
            m = _DRAW_N_RE.search(line)
            amount = int(m.group("n")) if m else None
        elif " played " in line and " to the Active Spot" in line:
            event_type = "play_to_active"
            actor = line.split(" played ", 1)[0].strip()
        elif " played " in line and " to the Bench" in line:
            event_type = "play_to_bench"
            actor = line.split(" played ", 1)[0].strip()
        elif " played " in line and " to the Stadium spot" in line:
            event_type = "play_stadium"
            actor = line.split(" played ", 1)[0].strip()
        elif " played " in line:
            event_type = "play_card"
            actor = line.split(" played ", 1)[0].strip()
        elif " attached " in line:
            event_type = "attach_energy"
            actor = line.split(" attached ", 1)[0].strip()
        elif " evolved " in line and " to " in line:
            event_type = "evolve"
            actor = line.split(" evolved ", 1)[0].strip()
        elif " used " in line and " on " in line and " damage" in line:
            event_type = "attack"
            actor = _possessive_actor(line)
            amount = _damage_amount(line)
        elif " used " in line:
            event_type = "ability_or_attack"
            actor = _possessive_actor(line)
        elif " was Knocked Out" in line:
            event_type = "knockout"
            actor = _target_player_from_possessive(line)
        elif _TAKE_PRIZE_RE.search(line):
            event_type = "take_prize"
            amount = _prize_amount(line)
            actor = line.split(" took ", 1)[0].strip()
        elif " was added to " in line and "'s hand" in line:
            event_type = "add_to_hand_revealed"
            m = re.search(r"was added to (?P<player>.+?)'s hand", line)
            actor = m.group("player").strip() if m else current_turn_player
        elif line.startswith("A card was added to ") and "'s hand" in line:
            event_type = "add_to_hand_hidden"
            actor = line.replace("A card was added to ", "").split("'s hand", 1)[0].strip()
            amount = 1
        elif " is now in the Active Spot" in line:
            event_type = "promote_active"
            actor = _possessive_actor(line)
        elif " retreated " in line:
            event_type = "retreat"
            actor = line.split(" retreated ", 1)[0].strip()
        elif " was switched with " in line:
            event_type = "switch_active"
            actor = _possessive_actor(line)
        elif "was discarded from" in line or "were discarded from" in line:
            event_type = "discard_from_play"
            actor = _possessive_actor(line) or current_turn_player
        elif "Opponent conceded" in line or "wins" in line:
            event_type = "game_end"
            if " wins" in line:
                actor = line.rsplit(" ", 2)[-2].strip(". ")
                metadata["winner"] = actor

        zone_hint = _zone_hint(line)
        if zone_hint:
            metadata["target_zone"] = zone_hint

        target_owner = _target_owner_hint(line)
        if target_owner:
            metadata["target_owner"] = target_owner

        event = LogEvent(
            index=len(events),
            line_no=line_no,
            raw=line,
            event_type=event_type,
            actor=actor,
            turn_player=current_turn_player,
            cards=cards,
            amount=amount,
            metadata=metadata,
        )
        events.append(event)

        if event_type != "revealed_cards":
            previous_event = event

    return events
