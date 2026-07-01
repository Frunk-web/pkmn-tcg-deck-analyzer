from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CardRef:
    exported_id: str = ""
    name: str = ""
    raw: str = ""
    unknown: bool = False

    @property
    def display_name(self) -> str:
        if self.unknown:
            return "Unknown card"
        if self.exported_id and self.name:
            return f"({self.exported_id}) {self.name}"
        return self.name or self.exported_id or "Unknown card"


@dataclass
class PokemonInPlay:
    card: CardRef
    damage: int = 0
    attached: list[CardRef] = field(default_factory=list)
    evolution_stack: list[CardRef] = field(default_factory=list)

    # Board-instance identity. This is different from card identity.
    # Example: two copies of (sv6_129) Drakloak become Drakloak #1 and Drakloak #2.
    instance_id: str = ""
    copy_number: int = 0
    created_event_index: int = -1
    inferred: bool = False

    @property
    def display_name(self) -> str:
        return self.card.display_name

    @property
    def copy_label(self) -> str:
        base = self.card.name or self.card.exported_id or "Pokémon"
        if self.copy_number > 0:
            return f"{base} #{self.copy_number}"
        return base


@dataclass
class PlayerState:
    name: str
    active: PokemonInPlay | None = None
    bench: list[PokemonInPlay] = field(default_factory=list)

    hand_known: list[CardRef] = field(default_factory=list)
    hand_unknown_count: int = 0

    discard: list[CardRef] = field(default_factory=list)

    prizes_taken: list[CardRef] = field(default_factory=list)
    user_known_prizes: list[CardRef] = field(default_factory=list)
    starting_prize_count: int = 6

    # Per-player board-instance counters.
    pokemon_instance_counters: dict[str, int] = field(default_factory=dict)

    @property
    def remaining_prize_count(self) -> int:
        return max(0, self.starting_prize_count - len(self.prizes_taken))

    @property
    def bench_count(self) -> int:
        return len(self.bench)


@dataclass
class LogEvent:
    index: int
    line_no: int
    raw: str
    event_type: str
    actor: str = ""
    turn_player: str = ""
    cards: list[CardRef] = field(default_factory=list)
    amount: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameState:
    players: dict[str, PlayerState] = field(default_factory=dict)
    player_order: list[str] = field(default_factory=list)
    turn_player: str = ""
    stadium: CardRef | None = None
    winner: str = ""
    last_event: LogEvent | None = None

    # Accumulated honesty log: exact/inferred/ambiguous target decisions.
    ambiguities: list[dict[str, Any]] = field(default_factory=list)

    def ensure_player(self, name: str) -> PlayerState:
        clean = str(name or "").strip()
        if not clean:
            clean = "Unknown Player"

        if clean not in self.players:
            self.players[clean] = PlayerState(name=clean)
            self.player_order.append(clean)

        return self.players[clean]


@dataclass
class ReplayFrame:
    step: int
    event: LogEvent | None
    state: GameState
