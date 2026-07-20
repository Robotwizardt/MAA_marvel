from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class PlayStrategy(str, Enum):
    RANDOM = "random"
    AGATHA = "agatha"
    OCR = "ocr"


class ConquestTier(str, Enum):
    PROVING_GROUNDS = "proving_grounds"
    SILVER = "silver"
    GOLD = "gold"
    INFINITE = "infinite"


class NoTicketBehavior(str, Enum):
    FALLBACK = "fallback"
    STOP = "stop"


class AfterRetreat(str, Enum):
    CONTINUE = "continue"
    CONCEDE = "concede"


class SnapMode(str, Enum):
    OFF = "off"
    PROBABILITY = "probability"
    ALWAYS = "always"


@dataclass(frozen=True, slots=True)
class SessionConfig:
    play_strategy: PlayStrategy = PlayStrategy.RANDOM
    max_tier: ConquestTier = ConquestTier.PROVING_GROUNDS
    no_ticket: NoTicketBehavior = NoTicketBehavior.FALLBACK
    retreat_after_turn: int = 0
    after_retreat: AfterRetreat = AfterRetreat.CONTINUE
    snap_mode: SnapMode = SnapMode.OFF
    snap_probability: int = 46
    max_matches: int = 0
    max_minutes: int = 0
    matchmaking_timeout_seconds: int = 600
    auto_restart: bool = True
    unknown_timeout_seconds: int = 120
    max_restarts: int = 3

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SessionConfig":
        auto_restart = values.get("auto_restart", True)
        if not isinstance(auto_restart, bool):
            raise ValueError("auto_restart must be a boolean")

        config = cls(
            play_strategy=PlayStrategy(values.get("play_strategy", "random")),
            max_tier=ConquestTier(values.get("max_tier", "proving_grounds")),
            no_ticket=NoTicketBehavior(values.get("no_ticket", "fallback")),
            retreat_after_turn=int(values.get("retreat_after_turn", 0)),
            after_retreat=AfterRetreat(values.get("after_retreat", "continue")),
            snap_mode=SnapMode(values.get("snap_mode", "off")),
            snap_probability=int(values.get("snap_probability", 46)),
            max_matches=int(values.get("max_matches", 0)),
            max_minutes=int(values.get("max_minutes", 0)),
            matchmaking_timeout_seconds=int(
                values.get("matchmaking_timeout_seconds", 600)
            ),
            auto_restart=auto_restart,
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not 0 <= self.retreat_after_turn <= 6:
            raise ValueError("retreat_after_turn must be between 0 and 6")
        if not 0 <= self.snap_probability <= 100:
            raise ValueError("snap_probability must be between 0 and 100")
        if self.max_matches < 0 or self.max_minutes < 0:
            raise ValueError("stop limits must be non-negative")
        if self.matchmaking_timeout_seconds <= 0:
            raise ValueError("matchmaking timeout must be positive")
