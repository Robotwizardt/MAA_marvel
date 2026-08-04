from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class PlayStrategy(str, Enum):
    """界面 play_strategy 字符串允许的值。"""
    AGATHA = "agatha"
    OCR = "ocr"


class LaneOrder(str, Enum):
    """把卡牌拖向三个场地时允许的尝试顺序。"""
    RANDOM = "random"
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"


class GameMode(str, Enum):
    CONQUEST = "conquest"
    LADDER = "ladder"


class ConquestTier(str, Enum):
    PROVING_GROUNDS = "proving_grounds"
    SILVER = "silver"
    GOLD = "gold"
    INFINITE = "infinite"


class AfterRetreat(str, Enum):
    CONTINUE = "continue"
    CONCEDE = "concede"


class SnapMode(str, Enum):
    OFF = "off"
    PROBABILITY = "probability"
    ALWAYS = "always"


@dataclass(frozen=True, slots=True)
class SessionConfig:
    """一轮任务中不会变化的用户配置；默认值必须与 interface 选项一致。"""
    play_strategy: PlayStrategy = PlayStrategy.OCR
    lane_order: LaneOrder = LaneOrder.LEFT_TO_RIGHT
    game_mode: GameMode = GameMode.CONQUEST
    daily_routine: bool = False
    max_tier: ConquestTier = ConquestTier.PROVING_GROUNDS
    reserve_silver_tickets: int = 1
    reserve_gold_tickets: int = 1
    reserve_infinite_tickets: int = 1
    stop_on_daily_pass_limit: bool = False
    retreat_after_turn: int = 0
    after_retreat: AfterRetreat = AfterRetreat.CONTINUE
    snap_mode: SnapMode = SnapMode.ALWAYS
    snap_probability: int = 46
    claim_task_rewards_hours: int = 0
    matchmaking_timeout_seconds: int = 600
    auto_restart: bool = True
    deck_name: str = "0"
    unknown_timeout_seconds: int = 120
    max_restarts: int = 3

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SessionConfig":
        """把 Pipeline 传来的字符串/数字转换成强类型配置并校验。"""
        auto_restart = values.get("auto_restart", True)
        if not isinstance(auto_restart, bool):
            raise ValueError("auto_restart must be a boolean")
        deck_name = str(values.get("deck_name", "0")).strip()
        stop_on_daily_pass_limit = values.get("stop_on_daily_pass_limit", False)
        if not isinstance(stop_on_daily_pass_limit, bool):
            raise ValueError("stop_on_daily_pass_limit must be a boolean")
        daily_routine = values.get("daily_routine", False)
        if not isinstance(daily_routine, bool):
            raise ValueError("daily_routine must be a boolean")

        config = cls(
            play_strategy=PlayStrategy(values.get("play_strategy", "ocr")),
            lane_order=LaneOrder(values.get("lane_order", "left_to_right")),
            game_mode=GameMode(values.get("game_mode", "conquest")),
            daily_routine=daily_routine,
            max_tier=ConquestTier(values.get("max_tier", "proving_grounds")),
            reserve_silver_tickets=int(values.get("reserve_silver_tickets", 1)),
            reserve_gold_tickets=int(values.get("reserve_gold_tickets", 1)),
            reserve_infinite_tickets=int(values.get("reserve_infinite_tickets", 1)),
            stop_on_daily_pass_limit=stop_on_daily_pass_limit,
            retreat_after_turn=int(values.get("retreat_after_turn", 0)),
            after_retreat=AfterRetreat(values.get("after_retreat", "continue")),
            snap_mode=SnapMode(values.get("snap_mode", "always")),
            snap_probability=int(values.get("snap_probability", 46)),
            claim_task_rewards_hours=int(
                values.get("claim_task_rewards_hours", 0)
            ),
            matchmaking_timeout_seconds=int(
                values.get("matchmaking_timeout_seconds", 600)
            ),
            auto_restart=auto_restart,
            deck_name=deck_name,
        )
        config.validate()
        return config

    def validate(self) -> None:
        """尽早拒绝危险或无意义参数，防止错误配置进入游戏流程。"""
        if not 0 <= self.retreat_after_turn <= 6:
            raise ValueError("retreat_after_turn must be between 0 and 6")
        if not 0 <= self.snap_probability <= 100:
            raise ValueError("snap_probability must be between 0 and 100")
        if self.claim_task_rewards_hours < 0:
            raise ValueError("claim_task_rewards_hours must be non-negative")
        if min(
            self.reserve_silver_tickets,
            self.reserve_gold_tickets,
            self.reserve_infinite_tickets,
        ) < 0:
            raise ValueError("ticket reserves must be non-negative")
        if self.matchmaking_timeout_seconds <= 0:
            raise ValueError("matchmaking timeout must be positive")
