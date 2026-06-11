from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import random


ZoneName = str
PlayerId = str


@dataclass
class CardInstance:
    instance_id: str
    card_id: str
    name: str
    owner: PlayerId
    controller: PlayerId
    zone: ZoneName
    definition: Dict[str, Any] = field(default_factory=dict)
    damage_counters: int = 0
    special_conditions: List[str] = field(default_factory=list)
    attached_cards: List[str] = field(default_factory=list)
    evolution_stack: List[str] = field(default_factory=list)
    turn_memory: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["definition"] = {
            "card_id": self.definition.get("card_id"),
            "name": self.definition.get("identity", {}).get("name"),
            "supertype": self.definition.get("identity", {}).get("supertype"),
            "subtypes": self.definition.get("identity", {}).get("subtypes", []),
        }
        return out


@dataclass
class PlayerState:
    player_id: PlayerId
    deck: List[str] = field(default_factory=list)
    hand: List[str] = field(default_factory=list)
    active: Optional[str] = None
    bench: List[str] = field(default_factory=list)
    discard: List[str] = field(default_factory=list)
    prizes: List[str] = field(default_factory=list)
    lost_zone: List[str] = field(default_factory=list)
    flags: Dict[str, Any] = field(default_factory=dict)

    def zone_ids(self, zone: ZoneName) -> List[str]:
        if zone == "deck":
            return self.deck
        if zone == "hand":
            return self.hand
        if zone == "bench":
            return self.bench
        if zone == "discard":
            return self.discard
        if zone == "prizes":
            return self.prizes
        if zone == "lost_zone":
            return self.lost_zone
        raise KeyError(f"Unknown zone: {zone}")


@dataclass
class GameState:
    players: Dict[PlayerId, PlayerState]
    cards: Dict[str, CardInstance]
    turn_player: PlayerId = "p1"
    turn_number: int = 1
    phase: str = "main_step"
    flags: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    log: List[Dict[str, Any]] = field(default_factory=list)
    rng_seed: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = random.Random(self.rng_seed)

    def opponent_of(self, player_id: PlayerId) -> PlayerId:
        for pid in self.players:
            if pid != player_id:
                return pid
        raise ValueError(f"No opponent found for {player_id}")

    def card(self, instance_id: str) -> CardInstance:
        return self.cards[instance_id]

    def log_event(self, event: str, **payload: Any) -> None:
        self.log.append({"event": event, **payload})

    def move_instance(self, instance_id: str, player_id: PlayerId, to_zone: ZoneName) -> None:
        card = self.cards[instance_id]
        old_controller = card.controller
        old_zone = card.zone
        # Remove from old zone.
        if old_zone == "active":
            if self.players[old_controller].active == instance_id:
                self.players[old_controller].active = None
        else:
            zone_list = self.players[old_controller].zone_ids(old_zone)
            if instance_id in zone_list:
                zone_list.remove(instance_id)
        # Add to new zone.
        if to_zone == "active":
            previous = self.players[player_id].active
            if previous is not None:
                self.move_instance(previous, player_id, "bench")
            self.players[player_id].active = instance_id
        else:
            self.players[player_id].zone_ids(to_zone).append(instance_id)
        card.controller = player_id
        card.zone = to_zone
        self.log_event("move_instance", instance_id=instance_id, card_id=card.card_id, name=card.name, from_zone=old_zone, to_zone=to_zone)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "turn_player": self.turn_player,
            "turn_number": self.turn_number,
            "phase": self.phase,
            "flags": self.flags,
            "memory": self.memory,
            "players": {pid: asdict(p) for pid, p in self.players.items()},
            "cards": {iid: c.to_dict() for iid, c in self.cards.items()},
            "log": self.log,
        }
