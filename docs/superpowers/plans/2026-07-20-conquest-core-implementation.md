# MAA Marvel Common Battle Engine and Conquest Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在国服《漫威终极逆转》720×1280 安卓画面上，实现可由 MaaFramework 通用 UI 配置和启动的公共战斗引擎与征服模式自动对战，并把随机出牌、阿加莎托管、停止条件、付费保护和异常恢复做成后续天梯/活动可复用的模块。

**Architecture:** ProjectInterface V2 负责用户配置；JSON Pipeline 负责页面识别、已知页面跳转和受识别结果约束的点击；Python Agent 负责不可变配置、会话状态、随机策略、档位候选、SNAP/撤退判断和恢复计数。所有可单测的规则下沉为不依赖 `maa` 的纯 Python 模块，Maa 自定义动作/识别只做薄适配。

**Tech Stack:** MaaFramework/MaaFw 5.12.1、ProjectInterface V2、Maa JSON Pipeline、Python 3.12、标准库 `unittest`、Pillow（模板裁剪与离线样例检查）、Node.js 22、`@nekosu/maa-tools` 1.0.23。

## Global Constraints

- 设计基线是 `docs/superpowers/specs/2026-07-20-maa-marvel-conquest-core-design.md`；实现如需改变已批准行为，先更新设计并征得用户确认。
- 只支持 ADB、MuMu 12 优先、短边 720、竖屏 720×1280；包名固定为 `com.netease.ms`，设备序列号不写入仓库。
- 保持 MIT 许可证。只参考 booster-bot 的行为思路，不复制其 C#、图片或文本资源；README 只做来源致谢。
- 未知页面不得盲点；所有关键点击必须以当次模板/OCR 命中的 box 为目标。固定坐标只允许用于已验证的安全滑动、安卓返回键和无消费风险的局部操作。
- 任何金块图标、金块金额、付费确认或证据冲突都使入场失败；恢复路径不得经过入场点击节点。
- OCR 选牌是实验功能：失败时记录原因并结束回合，不回退随机策略；它不阻塞本阶段验收。
- 首次开始匹配、撤退、整场认输、领取奖励、停止游戏进程前，执行者必须在当时再次向用户取得明确授权。
- 每个任务遵循 RED → GREEN → REFACTOR；先运行并看到指定测试因预期原因失败，再写最小实现。
- 每个任务结束只提交该任务涉及的文件；不改动用户的无关工作树内容。

---

### Task 1: 建立可重复的 Python、schema 与 CI 验证基线

**Files:**

- Modify: `.gitignore`
- Modify: `tools/requirements.txt`
- Add: `agent/requirements.txt`
- Modify: `.github/workflows/check.yml`
- Add: `tests/__init__.py`
- Add: `tests/test_repository_contract.py`

- [ ] **Step 1: 写失败的仓库契约测试**

```python
# tests/test_repository_contract.py
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_tool_dependencies_match_ci(self) -> None:
        requirements = (ROOT / "tools" / "requirements.txt").read_text("utf-8")
        self.assertIn("jsonschema==4.26.0", requirements)
        self.assertIn("referencing==0.37.0", requirements)
        self.assertIn("Pillow>=10.4.0,<13", requirements)

    def test_agent_runtime_is_pinned(self) -> None:
        requirements = (ROOT / "agent" / "requirements.txt").read_text("utf-8")
        self.assertEqual(requirements.strip(), "maafw==5.12.1")

    def test_local_runtime_outputs_are_ignored(self) -> None:
        ignore = (ROOT / ".gitignore").read_text("utf-8")
        self.assertIn(".venv/", ignore)
        self.assertIn(".tmp/", ignore)
        self.assertIn("tests/artifacts/", ignore)
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```powershell
& 'C:\Users\19858\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m unittest tests.test_repository_contract -v
```

Expected: 3 个测试因缺少固定依赖/忽略项而失败，不是导入错误。

- [ ] **Step 3: 建立本地虚拟环境并写入最小配置**

`.gitignore` 追加：

```gitignore
# Local development and ADB captures
.venv/
.tmp/
tests/artifacts/
```

`tools/requirements.txt` 改为：

```text
json-with-comments
jsonschema==4.26.0
referencing==0.37.0
Pillow>=10.4.0,<13
```

`agent/requirements.txt`：

```text
maafw==5.12.1
```

CI 中用同一依赖源并把 task 文件纳入 schema 检查：

```yaml
- name: Install Python dependencies
  run: python -m pip install -r tools/requirements.txt -r agent/requirements.txt

- name: Validate JSON Schema
  run: >-
    python tools/validate_schema.py
    --schema-dir deps/tools
    --resource-dirs assets/resource
    --exclude-dirs assets/resource/announcement
    --interface-files assets/interface.json
    --task-dirs assets/tasks

- name: Run Python unit tests
  run: python -m unittest discover -s tests -v
```

创建并安装：

```powershell
& 'C:\Users\19858\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r tools\requirements.txt -r agent\requirements.txt
```

Expected: 依赖安装成功，`maa`、`jsonschema`、`PIL` 可导入。

- [ ] **Step 4: 运行测试并确认 GREEN**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_repository_contract -v`

Expected: 3 tests, `OK`。

- [ ] **Step 5: 提交基线**

```powershell
git add .gitignore tools/requirements.txt agent/requirements.txt .github/workflows/check.yml tests
git commit -m "build: establish python validation baseline"
```

---

### Task 2: 实现不可变会话配置与输入校验

**Files:**

- Add: `agent/__init__.py`
- Add: `agent/session/__init__.py`
- Add: `agent/session/config.py`
- Add: `tests/test_session_config.py`

- [ ] **Step 1: 写配置默认值、枚举转换和边界失败测试**

```python
# tests/test_session_config.py
import unittest

from agent.session.config import (
    AfterRetreat,
    ConquestTier,
    NoTicketBehavior,
    PlayStrategy,
    SessionConfig,
    SnapMode,
)


class SessionConfigTests(unittest.TestCase):
    def test_defaults_match_approved_design(self) -> None:
        config = SessionConfig.from_mapping({})
        self.assertEqual(config.play_strategy, PlayStrategy.RANDOM)
        self.assertEqual(config.max_tier, ConquestTier.PROVING_GROUNDS)
        self.assertEqual(config.no_ticket, NoTicketBehavior.FALLBACK)
        self.assertEqual(config.after_retreat, AfterRetreat.CONTINUE)
        self.assertEqual(config.snap_mode, SnapMode.OFF)
        self.assertEqual(config.snap_probability, 46)
        self.assertEqual(config.matchmaking_timeout_seconds, 600)
        self.assertTrue(config.auto_restart)

    def test_converts_interface_values(self) -> None:
        config = SessionConfig.from_mapping({
            "play_strategy": "agatha",
            "max_tier": "silver",
            "retreat_after_turn": 3,
            "snap_mode": "probability",
            "max_matches": 8,
        })
        self.assertEqual(config.play_strategy, PlayStrategy.AGATHA)
        self.assertEqual(config.max_tier, ConquestTier.SILVER)
        self.assertEqual(config.retreat_after_turn, 3)
        self.assertEqual(config.max_matches, 8)

    def test_rejects_out_of_range_values(self) -> None:
        for values in (
            {"retreat_after_turn": 7},
            {"snap_probability": 101},
            {"max_matches": -1},
            {"max_minutes": -1},
            {"matchmaking_timeout_seconds": 0},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    SessionConfig.from_mapping(values)
```

- [ ] **Step 2: 运行并确认因模块不存在而失败**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_session_config -v`

Expected: `ModuleNotFoundError: agent.session`。

- [ ] **Step 3: 实现枚举与冻结 dataclass**

`config.py` 使用以下公开接口：

```python
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
            matchmaking_timeout_seconds=int(values.get("matchmaking_timeout_seconds", 600)),
            auto_restart=bool(values.get("auto_restart", True)),
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
```

- [ ] **Step 4: 运行配置测试**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_session_config -v`

Expected: all tests `OK`。

- [ ] **Step 5: 提交**

```powershell
git add agent/__init__.py agent/session tests/test_session_config.py
git commit -m "feat: add validated session configuration"
```

---

### Task 3: 实现停止、SNAP、撤退和恢复状态机

**Files:**

- Add: `agent/session/state.py`
- Add: `tests/test_session_state.py`

- [ ] **Step 1: 写会话状态失败测试**

测试必须覆盖：`max_matches`、`max_minutes`、完成第 3 回合后在第 4 回合撤退、每局最多 SNAP 一次、概率边界、3 次截图重试、3 次返回、120 秒后重启、最多重启 3 次后停止，并验证重启不清空已完成场次。

核心测试形态：

```python
class FixedRng:
    def __init__(self, value: int) -> None:
        self.value = value

    def randrange(self, stop: int) -> int:
        self.stop = stop
        return self.value


def test_retreat_happens_on_following_turn(self) -> None:
    state = SessionState(SessionConfig(retreat_after_turn=3), started_at=100.0)
    state.begin_match()
    state.begin_turn(3)
    self.assertFalse(state.should_retreat())
    state.begin_turn(4)
    self.assertTrue(state.should_retreat())


def test_probability_snap_is_once_per_match(self) -> None:
    config = SessionConfig(snap_mode=SnapMode.PROBABILITY, snap_probability=46)
    state = SessionState(config, started_at=0.0)
    state.begin_match()
    self.assertTrue(state.decide_snap(FixedRng(45)))
    self.assertFalse(state.decide_snap(FixedRng(0)))
```

- [ ] **Step 2: 运行并确认 RED**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_session_state -v`

Expected: 因 `agent.session.state` 不存在失败。

- [ ] **Step 3: 实现状态机**

公开枚举和方法固定为：

```python
class StopReason(str, Enum):
    MAX_MATCHES = "max_matches"
    MAX_RUNTIME = "max_runtime"
    NO_TICKET = "no_ticket"
    RECOVERY_EXHAUSTED = "recovery_exhausted"
    USER_STOPPED = "user_stopped"


class RecoveryAction(str, Enum):
    RETRY = "retry"
    ANDROID_BACK = "android_back"
    WAIT = "wait"
    RESTART = "restart"
    STOP = "stop"


@dataclass(slots=True)
class SessionState:
    config: SessionConfig
    started_at: float
    completed_matches: int = 0
    current_turn: int = 0
    snapped_this_match: bool = False
    retry_count: int = 0
    back_count: int = 0
    restart_count: int = 0
    unknown_since: float | None = None
    last_known_state: str = "task_started"
    stop_reason: StopReason | None = None
```

行为约束：

- `should_stop(now)` 先尊重已有 `stop_reason`，再检查场次，再检查运行分钟；命中后保存原因。
- `begin_match()` 清空回合和本局 SNAP，但不清空总场次/重启数。
- `complete_match()` 只增加整场征服计数。
- `begin_turn(turn)` 要求 `turn >= 1`。
- `decide_snap(rng)` 在 OFF 返回 false；ALWAYS 第一次 true；PROBABILITY 使用 `rng.randrange(100) < probability`；决定 true 时立即标记本局已 SNAP。
- `mark_known(name)` 保存名称并清空未知计数。
- `next_recovery_action(now)`：先返回 3 次 RETRY，再返回 3 次 ANDROID_BACK；未满 120 秒返回 WAIT；允许且未达 3 次则增加重启数、清空当前恢复阶段并返回 RESTART；否则保存 `RECOVERY_EXHAUSTED` 并返回 STOP。

- [ ] **Step 4: 运行测试并重构重复断言**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_session_state -v`

Expected: all tests `OK`。

- [ ] **Step 5: 提交**

```powershell
git add agent/session/state.py tests/test_session_state.py
git commit -m "feat: add battle session state machine"
```

---

### Task 4: 实现征服档位选择和付费保护纯逻辑

**Files:**

- Add: `agent/conquest/__init__.py`
- Add: `agent/conquest/tier_policy.py`
- Add: `tests/test_tier_policy.py`

- [ ] **Step 1: 写档位候选与入口证据失败测试**

```python
class TierPolicyTests(unittest.TestCase):
    def test_candidates_descend_from_allowed_maximum(self) -> None:
        self.assertEqual(
            candidate_tiers(ConquestTier.GOLD),
            (ConquestTier.GOLD, ConquestTier.SILVER, ConquestTier.PROVING_GROUNDS),
        )

    def test_no_ticket_falls_back_or_stops(self) -> None:
        self.assertEqual(
            choose_tier(ConquestTier.INFINITE, set(), NoTicketBehavior.FALLBACK),
            ConquestTier.PROVING_GROUNDS,
        )
        self.assertIsNone(
            choose_tier(ConquestTier.INFINITE, set(), NoTicketBehavior.STOP)
        )

    def test_any_paid_evidence_rejects_entry(self) -> None:
        evidence = EntryEvidence(
            tier=ConquestTier.INFINITE,
            free_label=False,
            ticket_label=False,
            gold_icon=True,
            gold_amount=True,
            paid_confirmation=False,
        )
        self.assertFalse(is_safe_entry(evidence))
```

另加测试：试炼之地仅 `free_label=True` 可进入；白银/黄金/无限仅 `ticket_label=True` 且三种付费证据全 false 可进入；同时出现 ticket 与 gold 属于冲突并拒绝。

- [ ] **Step 2: 运行并确认 RED**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_tier_policy -v`

Expected: 模块不存在。

- [ ] **Step 3: 实现语义顺序，不依赖枚举偶然值**

```python
TIER_ORDER = (
    ConquestTier.PROVING_GROUNDS,
    ConquestTier.SILVER,
    ConquestTier.GOLD,
    ConquestTier.INFINITE,
)


def candidate_tiers(max_tier: ConquestTier) -> tuple[ConquestTier, ...]:
    last = TIER_ORDER.index(max_tier)
    return tuple(reversed(TIER_ORDER[: last + 1]))


@dataclass(frozen=True, slots=True)
class EntryEvidence:
    tier: ConquestTier
    free_label: bool
    ticket_label: bool
    gold_icon: bool
    gold_amount: bool
    paid_confirmation: bool


def is_safe_entry(evidence: EntryEvidence) -> bool:
    if evidence.gold_icon or evidence.gold_amount or evidence.paid_confirmation:
        return False
    if evidence.tier is ConquestTier.PROVING_GROUNDS:
        return evidence.free_label and not evidence.ticket_label
    return evidence.ticket_label and not evidence.free_label
```

`choose_tier` 从允许范围内的付费档位由高到低取第一个有票档位。最高档位本身是试炼之地时直接返回试炼；最高档位高于试炼但所有允许的付费档位都无票时，FALLBACK 返回试炼，STOP 返回 `None`。不要因为 `candidate_tiers` 的展示顺序中包含试炼就绕过无票行为。

- [ ] **Step 4: 运行测试**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_tier_policy -v`

Expected: all tests `OK`。

- [ ] **Step 5: 提交**

```powershell
git add agent/conquest tests/test_tier_policy.py
git commit -m "feat: add safe conquest tier policy"
```

---

### Task 5: 实现随机、阿加莎与实验 OCR 决策核心

**Files:**

- Add: `agent/strategies/__init__.py`
- Add: `agent/strategies/model.py`
- Add: `agent/strategies/random_play.py`
- Add: `agent/strategies/agatha.py`
- Add: `agent/strategies/ocr.py`
- Add: `tests/test_strategies.py`

- [ ] **Step 1: 写三种策略失败测试**

测试要求：

- 随机策略每轮覆盖四个不同手牌槽，最多两轮，共 8 次计划拖动；起点/终点加入的偏移不越过定义 ROI。
- 同一个 RNG 种子得到同一计划，不同种子允许不同计划。
- 阿加莎计划永远没有拖动，但仍要求结束回合。
- OCR 选择最高可支付费用；同费用取更靠左且置信度达标的牌；无费用、无候选、低置信度均返回明确失败原因。

```python
def test_ocr_chooses_highest_affordable_cost(self) -> None:
    cards = (
        CardCandidate(slot=0, cost=1, confidence=0.95),
        CardCandidate(slot=1, cost=4, confidence=0.91),
        CardCandidate(slot=2, cost=3, confidence=0.93),
    )
    decision = choose_card(energy=3, cards=cards, minimum_confidence=0.80)
    self.assertEqual(decision.card.slot, 2)
    self.assertEqual(decision.reason, "selected")


def test_agatha_only_ends_turn(self) -> None:
    plan = build_agatha_plan()
    self.assertEqual(plan.swipes, ())
    self.assertTrue(plan.end_turn)
```

- [ ] **Step 2: 运行并确认 RED**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_strategies -v`

Expected: 策略模块不存在。

- [ ] **Step 3: 实现统一动作模型和已批准几何边界**

```python
@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Swipe:
    start: Point
    end: Point
    duration_ms: int = 350


@dataclass(frozen=True, slots=True)
class TurnPlan:
    swipes: tuple[Swipe, ...]
    end_turn: bool = True


HAND_SLOTS = (
    Point(90, 1050),
    Point(270, 1050),
    Point(450, 1050),
    Point(630, 1050),
)
LANE_TARGETS = (Point(120, 650), Point(360, 650), Point(600, 650))
```

随机策略对两轮分别复制并 `rng.shuffle` 手牌槽，为每张牌 `rng.choice` 一条区域，x/y 使用 `rng.randint(-12, 12)` 偏移，并夹紧到手牌 `[55, 665]×[1020, 1090]`、区域 `[85, 635]×[600, 720]`。实际执行阶段每次拖动后检查零能量并可提前停止。

OCR 数据结构固定为：

```python
@dataclass(frozen=True, slots=True)
class CardCandidate:
    slot: int
    cost: int
    confidence: float


@dataclass(frozen=True, slots=True)
class CardDecision:
    card: CardCandidate | None
    reason: str
```

选择键为 `(cost, -slot)`；仅接受 `0 <= cost <= energy` 和置信度达标的候选。

- [ ] **Step 4: 运行策略测试**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_strategies -v`

Expected: all tests `OK`。

- [ ] **Step 5: 提交**

```powershell
git add agent/strategies tests/test_strategies.py
git commit -m "feat: add reusable play strategies"
```

---

### Task 6: 建立 Maa Agent 薄适配层和运行时存储

**Files:**

- Add: `agent/runtime/__init__.py`
- Add: `agent/runtime/store.py`
- Add: `agent/runtime/commands.py`
- Add: `agent/actions/__init__.py`
- Add: `agent/actions/configure_session.py`
- Add: `agent/actions/play_turn.py`
- Add: `agent/actions/record_event.py`
- Add: `agent/actions/route_conquest_tier.py`
- Add: `agent/actions/recovery.py`
- Add: `agent/recognitions/__init__.py`
- Add: `agent/recognitions/session_gate.py`
- Add: `agent/recognitions/card_selection.py`
- Add: `agent/recognitions/safe_entry.py`
- Modify: `agent/main.py`
- Delete: `agent/my_action.py`
- Delete: `agent/my_reco.py`
- Delete: `agent/chose_cards_reco.py`
- Add: `tests/test_runtime_commands.py`
- Add: `tests/test_agent_imports.py`

- [ ] **Step 1: 先测试不依赖 Maa 的命令解析和状态变化**

`commands.py` 公开纯函数：

```python
def parse_json_object(raw: str) -> dict[str, object]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("custom action parameter must be a JSON object")
    return value


def apply_event(state: SessionState, event: str, value: int | None = None) -> None:
    if event == "match_started":
        state.begin_match()
    elif event == "turn_started":
        state.begin_turn(value if value is not None else state.current_turn + 1)
    elif event == "match_completed":
        state.complete_match()
    elif event == "known_state":
        state.mark_known("pipeline")
    else:
        raise ValueError(f"unsupported session event: {event}")
```

测试合法 JSON、数组参数拒绝、未知 event 拒绝、配置后 store 含冻结配置、重启后计数保留。

- [ ] **Step 2: 运行并确认 RED**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_runtime_commands -v`

Expected: runtime 模块不存在。

- [ ] **Step 3: 实现线程安全 RuntimeStore 和 Maa 注册适配**

`RuntimeStore` 使用 `threading.RLock`，提供 `configure(mapping, now)`、`require_state()`、`reset_tier_candidates()`、`next_tier_candidate()` 和 `reject_tier()`；所有公开方法在锁内完成。

配置动作的核心必须只有解析与委托：

```python
@AgentServer.custom_action("MarvelConfigureSession")
class ConfigureSession(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        values = parse_json_object(argv.custom_action_param)
        STORE.configure(values, time.monotonic())
        return True
```

`MarvelPlayTurn`：

- RANDOM：取 `build_random_plan`；逐次调用 `post_swipe(swipe.start.x, swipe.start.y, swipe.end.x, swipe.end.y, swipe.duration_ms).wait()`；每次 `post_screencap().wait()` 后读取 `controller.cached_image` 并运行 `公共-零能量`，命中即停止拖动。
- AGATHA：不拖动，直接返回 true。
- OCR：在最新截图上运行 `MarvelCardSelection`；无安全候选时打印结构化失败原因并返回 true；有候选时仅拖动该识别 box 到一个经验证区域，然后重新截图，最多 4 张。
- 每次发布动作前再次读取 store 的停止状态；停止后不再发布任何控制器 job。

`MarvelSessionGate` 的 `custom_recognition_param` 支持精确命令：`should_stop`、`should_retreat`、`should_snap`、`after_retreat_concede`、`can_auto_restart`、`tier_available:proving_grounds`、`tier_available:silver`、`tier_available:gold`、`tier_available:infinite`。true 返回 `(0, 0, 720, 1280)`，false 返回 `(0, 0, 0, 0)`，detail 写入命令与结果。

`MarvelRouteConquestTier` 使用 `context.override_next(argv.node_name, [node])`，映射固定为：

```python
TIER_NODE = {
    ConquestTier.PROVING_GROUNDS: "征服-准备试炼之地",
    ConquestTier.SILVER: "征服-准备白银",
    ConquestTier.GOLD: "征服-准备黄金",
    ConquestTier.INFINITE: "征服-准备无限",
}
```

候选耗尽且无票行为 STOP 时设置 `StopReason.NO_TICKET` 并路由 `公共-安全停止`。

`MarvelRecoveryAction` 委托 `SessionState.next_recovery_action(time.monotonic())` 并覆盖当前节点的 next，映射固定为：

```python
RECOVERY_NODE = {
    RecoveryAction.RETRY: "公共-恢复重试",
    RecoveryAction.ANDROID_BACK: "公共-恢复返回",
    RecoveryAction.WAIT: "公共-恢复等待",
    RecoveryAction.RESTART: "公共-恢复重启",
    RecoveryAction.STOP: "公共-安全停止",
}
```

`MarvelSafeEntry` 在 Task 6 先注册并完成 JSON 参数校验；Task 10 再接入已采集模板的五项识别证据。任何异常都返回零 box，不得把异常当成安全入口。

- [ ] **Step 4: 重写 Agent 入口并做导入冒烟测试**

`agent/main.py`：

```python
import sys

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit

from agent.actions import (
    configure_session,
    play_turn,
    record_event,
    recovery,
    route_conquest_tier,
)
from agent.recognitions import card_selection, safe_entry, session_gate


def main() -> None:
    Toolkit.init_option("./")
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m agent.main <socket_id>")
    AgentServer.start_up(sys.argv[-1])
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
```

`tests/test_agent_imports.py` 只导入 `agent.main` 并断言注册列表包含 8 个名称：5 个动作 `MarvelConfigureSession`、`MarvelPlayTurn`、`MarvelRecordEvent`、`MarvelRouteConquestTier`、`MarvelRecoveryAction`，以及 3 个识别 `MarvelSessionGate`、`MarvelCardSelection`、`MarvelSafeEntry`；不启动 socket。

Run:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.test_runtime_commands tests.test_agent_imports -v
```

Expected: all tests `OK`，退出时无 Maa 原生库加载异常。

- [ ] **Step 5: 提交**

```powershell
git add agent tests/test_runtime_commands.py tests/test_agent_imports.py
git commit -m "feat: wire session logic into maa agent"
```

---

### Task 7: 重建通用 UI 的征服任务与全部配置映射

**Files:**

- Modify: `assets/interface.json`
- Add: `assets/tasks/征服模式.json`
- Delete: `assets/tasks/开始战斗.json`
- Delete: `assets/tasks/开始游戏.json`
- Add: `tests/test_interface_contract.py`

- [ ] **Step 1: 写失败的 Interface 契约测试**

测试 JSONC 解析后断言：项目名 `MAA_marvel`、GitHub 为 `https://github.com/Robotwizardt/MAA_marvel`、仅一个 `Adb` controller、短边 720、Agent 以 `python -m agent.main` 启动、只导入 `tasks/征服模式.json`、任务 entry 为 `征服-任务入口`，并且 11 个批准选项全部存在且默认值正确。

- [ ] **Step 2: 运行并确认 RED**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_interface_contract -v`

Expected: 现有 MaaXXX 模板元信息和任务列表导致断言失败。

- [ ] **Step 3: 重写 interface 元信息、控制器和 Agent**

关键内容：

```json
{
    "interface_version": 2,
    "name": "MAA_marvel",
    "description": "国服《漫威终极逆转》安卓自动化助手",
    "license": "MIT",
    "github": "https://github.com/Robotwizardt/MAA_marvel",
    "version": "0.1.0",
    "controller": [
        { "name": "安卓端", "type": "Adb", "display_short_side": 720 }
    ],
    "resource": [
        { "name": "官服", "path": ["./resource"] }
    ],
    "agent": {
        "child_exec": "python",
        "child_args": ["-m", "agent.main"]
    },
    "task": [],
    "option": {},
    "import": ["tasks/征服模式.json"]
}
```

不保留虚假 QQ 联系方式、Win32 控制器或 MaaPracticeBoilerplate 地址。

- [ ] **Step 4: 定义征服任务与 11 个选项**

任务：

```json
{
    "task": [
        {
            "name": "征服模式自动对战",
            "entry": "征服-任务入口",
            "resource": ["官服"],
            "controller": ["安卓端"],
            "description": "仅支持国服安卓端 720×1280；不会购买门票或消耗金块。",
            "option": [
                "征服-出牌策略",
                "征服-最高档位",
                "征服-无票行为",
                "征服-自动撤退",
                "征服-撤退后",
                "征服-SNAP",
                "征服-最大对局数",
                "征服-最大运行分钟",
                "征服-匹配超时",
                "征服-自动重启"
            ]
        }
    ]
}
```

`征服-SNAP` 的概率 case 嵌套第 11 个选项 `征服-SNAP概率`。每个 case 只覆盖 `征服-初始化会话.action.param.custom_action_param` 中对应字段，例如：

```json
"征服-出牌策略": {
    "type": "select",
    "label": "出牌策略",
    "default_case": "random",
    "cases": [
        {
            "name": "random",
            "label": "随机出牌",
            "pipeline_override": {
                "征服-初始化会话": {
                    "action": { "param": { "custom_action_param": { "play_strategy": "random" } } }
                }
            }
        },
        {
            "name": "agatha",
            "label": "阿加莎托管",
            "pipeline_override": {
                "征服-初始化会话": {
                    "action": { "param": { "custom_action_param": { "play_strategy": "agatha" } } }
                }
            }
        },
        {
            "name": "ocr",
            "label": "OCR 选牌（实验）",
            "pipeline_override": {
                "征服-初始化会话": {
                    "action": { "param": { "custom_action_param": { "play_strategy": "ocr" } } }
                }
            }
        }
    ]
}
```

输入正则固定：非负整数 `^(0|[1-9][0-9]*)$`，正整数 `^[1-9][0-9]*$`，概率 `^(100|[1-9]?[0-9])$`；`pipeline_type` 均为 `int`。默认值严格采用设计文档：random、proving_grounds、fallback、0、continue、off、46、0、0、600、Yes。

- [ ] **Step 5: 运行 schema 与契约测试**

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.test_interface_contract -v
& '.\.venv\Scripts\python.exe' tools\validate_schema.py --schema-dir deps\tools --resource-dirs assets\resource --interface-files assets\interface.json --task-dirs assets\tasks
```

Expected: unittest `OK`；validator 最后一行 `All validations passed!`。

- [ ] **Step 6: 提交**

```powershell
git add assets/interface.json assets/tasks tests/test_interface_contract.py
git commit -m "feat: expose conquest task configuration"
```

---

### Task 8: 建立 720×1280 截图、模板裁剪和回归清单

**Files:**

- Add: `tools/crop_templates.py`
- Add: `tests/test_template_manifest.py`
- Add: `tests/fixtures/screens/README.md`
- Add: `tests/fixtures/screens/manifest.json`
- Add after capture: `tests/fixtures/screens/**/*.png`
- Add after crop: `assets/resource/image/common/**/*.png`
- Add after crop: `assets/resource/image/conquest/**/*.png`
- Add after crop: `assets/resource/image/recovery/**/*.png`

- [ ] **Step 1: 写 manifest 校验与裁剪失败测试**

Manifest 每项结构固定为：

```json
{
    "source": "tests/fixtures/screens/main/home.png",
    "output": "assets/resource/image/common/main/game_modes.png",
    "box": [474, 1138, 82, 120],
    "positive_nodes": ["公共-主界面"],
    "negative_sources": ["tests/fixtures/screens/main/modes.png"]
}
```

测试断言所有源图为 720×1280、box 在画面内、输出路径唯一、输出不是整屏、正/负样本存在、裁剪结果尺寸等于 box。先以不存在的源图运行，确认测试因缺文件失败。

- [ ] **Step 2: 实现确定性裁剪工具**

```python
def crop_entry(root: Path, entry: dict[str, object]) -> None:
    source = root / str(entry["source"])
    output = root / str(entry["output"])
    x, y, width, height = (int(value) for value in entry["box"])
    with Image.open(source) as image:
        if image.size != (720, 1280):
            raise ValueError(f"{source} must be 720x1280, got {image.size}")
        output.parent.mkdir(parents=True, exist_ok=True)
        image.crop((x, y, x + width, y + height)).save(output, format="PNG")
```

CLI 固定为 `python tools/crop_templates.py tests/fixtures/screens/manifest.json`，不修改源截图。

- [ ] **Step 3: 采集安全页面并填写精确裁剪框**

先连接 `D:\MuMuPlayer\nx_device\12.0\shell\adb.exe` 的 `emulator-5556`，只采集不会开始匹配的页面：

- `main/home.png`
- `main/modes.png`
- `conquest/lobby_proving_grounds.png`
- `conquest/lobby_silver.png`
- `conquest/lobby_gold.png`
- `conquest/lobby_infinite_paid.png`
- `conquest/prematch.png`

每次截图先验证 `wm size` 为 720×1280。manifest 至少裁出主界面游戏模式图标、模式列表征服卡片、四个档位标题、试炼免费进入、门票标志、金块图标/金额、赛前开战、赛前删除和商店负样本。裁剪时避开昵称、货币数值、轮播图、动态背景。

- [ ] **Step 4: 在取得首次匹配授权后采集战斗/结算/恢复页面**

授权检查点后按顺序补齐：

- `battle/matchmaking.png`
- `battle/actionable.png`
- `battle/waiting.png`
- `battle/zero_energy.png`
- `battle/retreat_menu.png`
- `conquest/round_result.png`
- `conquest/match_result.png`
- `conquest/ticket_reward.png`
- `recovery/reconnect.png`（仅自然出现时保存，不人为断网）

每个关键状态至少一张相似负样本。执行者在提交前人工打开每个裁剪，确认文字/图标完整且背景占比低。

- [ ] **Step 5: 生成模板并运行回归测试**

```powershell
& '.\.venv\Scripts\python.exe' tools\crop_templates.py tests\fixtures\screens\manifest.json
& '.\.venv\Scripts\python.exe' -m unittest tests.test_template_manifest -v
```

Expected: 所有清单项生成，测试 `OK`；`git status` 不包含 `.tmp`。

- [ ] **Step 6: 提交安全页面资源；需授权页面随对应流水线提交**

```powershell
git add tools/crop_templates.py tests/test_template_manifest.py tests/fixtures/screens assets/resource/image
git commit -m "test: add android screenshot recognition fixtures"
```

---

### Task 9: 实现公共启动、已知页面归一化和恢复流水线

**Files:**

- Add: `assets/resource/pipeline/common/bootstrap.json`
- Add: `assets/resource/pipeline/common/home.json`
- Add: `assets/resource/pipeline/recovery/recovery.json`
- Add: `tests/test_pipeline_safety.py`
- Delete after migration: `assets/resource/pipeline/开始游戏.json`
- Delete after migration: `assets/resource/pipeline/Colormatcocr.json`

- [ ] **Step 1: 写静态流水线安全失败测试**

递归读取所有生产 pipeline，断言：

- 所有数值 ROI/target/end 都在 720×1280 内，任何 `1920`、`1080` 坐标直接失败。
- Click 动作不得配 `DirectHit`；Click 必须使用自身识别 box 或明确命名的已识别节点 target。
- `ClickKey` 只允许 key 4（Android Back）。
- `StartApp`/`StopApp` 的 package 都是 `com.netease.ms`。
- 名称含“恢复”的节点不得 next 到名称含“点击进入”的节点。
- 名称含“金块”或“付费”的识别节点不得配置 Click。

- [ ] **Step 2: 运行并确认旧流水线导致 RED**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_pipeline_safety -v`

Expected: 旧 1920×1080 与 DirectHit 节点被报告。

- [ ] **Step 3: 实现启动与主界面识别**

入口节点：

```json
"公共-启动游戏": {
    "action": {
        "type": "StartApp",
        "param": { "package": "com.netease.ms" }
    },
    "post_delay": 5000,
    "next": ["公共-主界面", "公共-已知弹窗", "公共-恢复决策"]
}
```

`公共-主界面` 用 `common/main/game_modes.png` 在 `[445, 1100, 140, 180]` 匹配，阈值从离线正负样本选取并记录在 manifest；只在命中后转征服导航。已知弹窗、重连、继续、下一步均各有局部模板及 Click，不能用全屏 OCR 查找 `X`。

- [ ] **Step 4: 实现分级恢复**

`公共-恢复决策` 调用 Task 6 在 `agent/actions/recovery.py` 注册的 `MarvelRecoveryAction`，由该动作覆盖 next 并路由以下确定节点：

- `公共-恢复重试`：延迟 1500 ms，再回当前识别入口。
- `公共-恢复返回`：`ClickKey` key 4，延迟 1500 ms。
- `公共-恢复等待`：延迟 3000 ms。
- `公共-恢复重启`：先进入授权过的 StopApp/StartApp 链；两者之间等待 3000 ms，启动后等待 8000 ms。
- `公共-安全停止`：CustomAction `MarvelRecordEvent` 写入最终原因，无 next。

恢复路由不能包含任何档位/入场/开战按钮。

- [ ] **Step 5: 运行测试和 schema**

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.test_pipeline_safety -v
& '.\.venv\Scripts\python.exe' tools\validate_schema.py --schema-dir deps\tools --resource-dirs assets\resource --interface-files assets\interface.json --task-dirs assets\tasks
```

Expected: all tests 和 schema 均通过。

- [ ] **Step 6: 提交**

```powershell
git add assets/resource/pipeline tests/test_pipeline_safety.py
git commit -m "feat: add safe boot and recovery pipelines"
```

---

### Task 10: 实现征服导航、档位轮播与无消费入场

**Files:**

- Add: `assets/resource/pipeline/conquest/navigation.json`
- Add: `assets/resource/pipeline/conquest/tier_selection.json`
- Add: `assets/resource/pipeline/conquest/prematch.json`
- Modify: `agent/recognitions/safe_entry.py`
- Add: `tests/test_conquest_pipeline.py`
- Delete after migration: `assets/resource/pipeline/征服模式刷币.json`

- [ ] **Step 1: 写征服图结构失败测试**

断言以下入口均可通过有向图到达 `征服-选择档位候选`：主界面、模式列表、征服大厅、赛前页。另断言：

- `征服-点击免费进入` 的唯一直接前驱是 `征服-安全入口确认`。
- `征服-点击门票进入` 的唯一直接前驱是 `征服-安全入口确认`。
- `征服-安全入口确认` 必须是 Custom recognition `MarvelSafeEntry`。
- `征服-金块入口`、`征服-付费确认` 只路由 `征服-拒绝当前档位`。
- `征服-点击开战` 使用赛前“开战”模板 box，ROI 不与“删除”或商店负样本相交。

- [ ] **Step 2: 运行并确认 RED**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_conquest_pipeline -v`

Expected: 新节点缺失。

- [ ] **Step 3: 实现主界面到征服大厅导航**

- 主界面只点击已识别的底部游戏模式图标。
- 模式列表先识别征服卡片；未出现时只在列表内容区域 `[80, 260, 560, 700]` 向上滑一次并重识别，最多 3 次。
- 点击征服卡片后必须识别四档标题之一才视为进入大厅。
- 从赛前页启动时用 Android Back 返回大厅，不点击“删除”或商店。

- [ ] **Step 4: 实现候选路由与轮播归一化**

`征服-初始化会话` 的默认参数完整写入 11 个配置值，调用 `MarvelConfigureSession` 后进入 `征服-选择档位候选`。该节点调用 `MarvelRouteConquestTier`。

每个候选分支先在轮播区域连续向右滑 3 次并识别试炼标题完成归一化，再按以下次数向左滑并验证标题：试炼 0、白银 1、黄金 2、无限 3。每次滑动后等待 700 ms。标题不匹配不点击入口，交给恢复。

- [ ] **Step 5: 实现入口证据收集和安全门**

`MarvelSafeEntry` 的参数包含 tier，并通过 `context.run_recognition` 收集 `free_label`、`ticket_label`、`gold_icon`、`gold_amount`、`paid_confirmation` 五个布尔证据，再委托 Task 4 的 `is_safe_entry`。

- true：转到对应按钮模板识别节点，点击当次识别 box。
- false 且存在付费证据：记录 `paid_entry_rejected`，转 `征服-拒绝当前档位`。
- false 且只是没票：同样拒绝当前候选并取下一档。
- 候选耗尽按 fallback 回试炼或以 NO_TICKET 安全停止。

赛前页仅 `TemplateMatch(conquest/prematch/battle.png)` 在中央 ROI 命中后点击；首次执行到此处必须暂停请求开始匹配授权。

- [ ] **Step 6: 运行单元、图结构和 schema 测试**

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.test_tier_policy tests.test_conquest_pipeline tests.test_pipeline_safety -v
& '.\.venv\Scripts\python.exe' tools\validate_schema.py --schema-dir deps\tools --resource-dirs assets\resource --interface-files assets\interface.json --task-dirs assets\tasks
```

Expected: all pass。

- [ ] **Step 7: 在无匹配授权范围内做实机导航验收并提交**

从主界面、模式页、征服大厅、赛前页分别启动，确认都停在正确赛前页；检查日志中 tier 和入口证据与画面一致，且无限无票页从未发布点击入场动作。

```powershell
git add assets/resource/pipeline/conquest tests/test_conquest_pipeline.py
git commit -m "feat: add guarded conquest navigation"
```

---

### Task 11: 实现公共战斗回合、SNAP、撤退与结算循环

**Files:**

- Add: `assets/resource/pipeline/common/battle.json`
- Add: `assets/resource/pipeline/common/retreat.json`
- Add: `assets/resource/pipeline/conquest/results.json`
- Add: `tests/test_battle_pipeline.py`
- Delete after migration: `assets/resource/pipeline/战斗.json`

- [ ] **Step 1: 写战斗图结构失败测试**

断言主路径严格包含：`match_started → turn_started → play_turn → snap_gate → end_turn → waiting → new_turn/round_result/match_result`。另断言：

- 每个 PlayTurn 前都存在 `should_stop` gate。
- SNAP Click 的前驱只能是 `MarvelSessionGate:should_snap`。
- 撤退 Click 的前驱只能是 `MarvelSessionGate:should_retreat`。
- 认输整场确认的前驱只能是 `after_retreat_concede`。
- end-turn、撤退、继续、认输、结算按钮都点击各自模板 box。
- 结果循环最终只到征服大厅或安全停止。

- [ ] **Step 2: 运行并确认 RED**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_battle_pipeline -v`

Expected: 战斗节点缺失。

- [ ] **Step 3: 实现组合状态识别和回合主循环**

节点职责：

- `公共-比赛开始` 记录 `match_started` 和 turn 1。
- `公共-可操作` 同时要求可用结束回合模板命中，且等待模板、轮间模板、整场结果模板均未命中。
- `公共-停止判断` 调用 `MarvelSessionGate:should_stop`；true 安全停止，false 执行 `MarvelPlayTurn`。
- `公共-SNAP判断` 只在本局未 SNAP 且按钮模板可见时执行配置决策；点击成功后记录事件。
- `公共-结束回合` 点击可用按钮并必须转入 `公共-等待对手`，避免同一画面重复计 turn。
- `公共-等待对手` 每 1500 ms 依次识别新回合、撤退结果、轮间、整场结果、重连；匹配/战斗超时交恢复状态机。
- 新回合先递增 turn，再评估撤退；这样 `retreat_after_turn=3` 在 turn 4 执行。

- [ ] **Step 4: 实现撤退与整场认输**

首次实机执行前再次请求授权。撤退链只在撤退入口模板命中后点击，再识别“撤退”确认按钮。`after_retreat=continue` 转轮间继续；`concede` 必须再次经过认输整场 gate 和确认模板。任一确认模板不明确时进入恢复，不使用固定坐标。

- [ ] **Step 5: 实现征服结果与停止条件**

轮间只点击明确的继续/下一轮；整场结果逐页识别“继续”“下一步”“领取”并回到大厅。首次领取奖励前再次请求授权。重新识别大厅后才记录 `match_completed`，然后检查最大场次/时间；达到条件转安全停止，否则重新选择档位。

- [ ] **Step 6: 运行离线测试、schema 和 Agent 单元测试**

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' tools\validate_schema.py --schema-dir deps\tools --resource-dirs assets\resource --interface-files assets\interface.json --task-dirs assets\tasks
npm ci
npx @nekosu/maa-tools check
```

Expected: Python tests `OK`；schema `All validations passed!`；maa-tools 无 error。

- [ ] **Step 7: 提交**

```powershell
git add assets/resource/pipeline/common assets/resource/pipeline/conquest tests/test_battle_pipeline.py
git commit -m "feat: add common battle and conquest result loop"
```

---

### Task 12: 完成实验 OCR 识别适配与可诊断日志

**Files:**

- Modify: `agent/recognitions/card_selection.py`
- Modify: `agent/actions/play_turn.py`
- Add: `tests/test_ocr_adapter.py`
- Add: `assets/resource/pipeline/common/ocr.json`

- [ ] **Step 1: 写 OCR 适配失败测试**

用 fake recognition results 覆盖：能量缺失、费用字符串无法解析、同一牌多 OCR 冲突、置信度过低、费用超出能量、合法最高费用。断言 detail 是紧凑 JSON，包含 `energy`、`candidates`、`selected_slot`、`reason`，且失败从不调用随机策略。

- [ ] **Step 2: 运行并确认 RED**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_ocr_adapter -v`

Expected: fake adapter 接口尚未实现。

- [ ] **Step 3: 实现分区 OCR，不再使用“隔一个数字就是费用”的假设**

- 当前能量 ROI 固定从实机 `battle/actionable.png` 校准，expected 为单个 `0–7`。
- 四个手牌费用分别拥有独立 ROI，结果按 x 排序并绑定 slot，不读取卡牌战力数字。
- `_parse_single_digit` 只接受完整单个数字；多个结果冲突即失败。
- `minimum_confidence=0.80`；失败返回零 box 和诊断 JSON。
- 成功返回卡牌费用标记的 box；PlayTurn 根据 slot 的已验证手牌中心拖动，不直接点击费用文字。

- [ ] **Step 4: 运行测试和一次只识别不操作的实机诊断**

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.test_ocr_adapter tests.test_strategies -v
```

将战斗截图作为 `argv.image` 离线运行识别，只查看日志，不发布拖动。Expected: 输出结构化结果；准确率不足可保留实验状态，但不能异常退出。

- [ ] **Step 5: 提交**

```powershell
git add agent/recognitions/card_selection.py agent/actions/play_turn.py assets/resource/pipeline/common/ocr.json tests/test_ocr_adapter.py
git commit -m "feat: add diagnostic ocr play mode"
```

---

### Task 13: 清理模板项目痕迹并补齐安装、来源和用户说明

**Files:**

- Modify: `README.md`
- Modify: `tools/install.py`
- Add: `docs/testing/adb-acceptance.md`
- Add: `tests/test_install_layout.py`
- Delete if still present: `assets/resource/image/不在弹出勾.png`
- Delete if still present: `assets/resource/image/empty.png`
- Delete if unused: `assets/resource/image/1费.png`
- Delete if unused: `assets/resource/image/2费.png`
- Delete if unused: `assets/resource/image/3费.png`
- Delete if unused: `assets/resource/image/5费.png`

- [ ] **Step 1: 写安装布局失败测试**

在临时目录模拟 `install_resource/install_agent` 的目标布局，断言打包产物含 `interface.json`、`resource`、`agent`、`agent/requirements.txt`、README、LICENSE，且 Interface 的 `python -m agent.main` 在安装根目录可导入。测试还扫描仓库，拒绝 MaaXXX、虚假 QQ、MaaPracticeBoilerplate 和旧自定义动作名。

- [ ] **Step 2: 运行并确认 RED**

Run: `& '.\.venv\Scripts\python.exe' -m unittest tests.test_install_layout -v`

Expected: README/安装布局或模板残留断言失败。

- [ ] **Step 3: 更新文档和安装脚本**

README 必须写清：

- 当前仅国服安卓 720×1280、MuMu 12 优先。
- 安装 `agent/requirements.txt`、通用 UI 选择 ADB 设备、三种策略与所有默认值。
- 付费保护和“不会自动购买门票/金块”。
- OCR 是实验功能。
- 恢复/停止原因从日志查看。
- booster-bot 仅作为状态机思路参考，链接 `https://github.com/little-fort/booster-bot`，未复制代码或资源。
- 天梯和限时活动属于后续阶段。

`tools/install.py` 保证 `agent/requirements.txt` 随 agent 复制，不把 `tests`、`.tmp`、`.venv` 带入 install。

- [ ] **Step 4: 删除未引用样例并验证无悬空资源**

用 `rg` 检查所有删除图片/节点都没有引用；若费用图片被 OCR 回归实际引用则保留并移动到 `image/common/ocr/`，不得留在资源根目录。

- [ ] **Step 5: 运行测试并提交**

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.test_install_layout tests.test_repository_contract -v
git add README.md tools/install.py docs/testing tests/test_install_layout.py assets/resource/image
git commit -m "docs: document conquest automation and safety"
```

---

### Task 14: 全量离线验证、分级 ADB 验收与 60 分钟稳定性测试

**Files:**

- Modify as evidence changes require: `tests/fixtures/screens/manifest.json`
- Modify as calibration requires: `assets/resource/pipeline/**/*.json`
- Add: `tests/artifacts/.gitkeep` only if the directory must exist; normally artifacts remain ignored
- Modify: `docs/testing/adb-acceptance.md`

- [ ] **Step 1: 在干净状态运行全量离线验证**

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
& '.\.venv\Scripts\python.exe' tools\validate_schema.py --schema-dir deps\tools --resource-dirs assets\resource --exclude-dirs assets\resource\announcement --interface-files assets\interface.json --task-dirs assets\tasks
npm ci
npx @nekosu/maa-tools check
git diff --check
git status --short
```

Expected: tests `OK`；schema 全通过；maa-tools 无 error；`git diff --check` 无输出；只出现预期修改。

- [ ] **Step 2: 安全导航验收（无需开始匹配）**

连接 MuMu 12，依次从主界面、模式页、四档大厅、赛前页启动；验证最高档位、无票回退和无限金块拒绝。保存日志与失败截图到 ignored 的 `tests/artifacts/`。

- [ ] **Step 3: 请求用户明确批准首次匹配和游戏内操作**

批准范围必须明确包含：开始免费试炼匹配、随机/阿加莎出牌、SNAP（如测试）、撤退、整场认输、奖励领取。若用户只批准其中一部分，只执行批准部分，其余验收保持未完成，不能宣称第一阶段完成。

- [ ] **Step 4: 按风险递增完成实机矩阵**

1. 随机策略、SNAP 关闭、撤退关闭，完成一次整场并回大厅。
2. 阿加莎策略完成一次整场并回大厅。
3. SNAP 概率 100%，验证每局最多一次。
4. 完成第 1 回合后撤退，验证在第 2 回合开始执行。
5. 分别验证撤退后继续和整场认输。
6. 最大对局数 1，验证回大厅后停止。
7. 最大运行分钟 1，验证不再发布新动作并输出 `max_runtime`。

每项记录：开始页面、配置、最终页面、停止原因、是否发生消费、失败截图路径。

- [ ] **Step 5: 请求用户明确批准停止游戏进程，再测试恢复**

批准后验证：切回桌面可返回；显式 StopApp 后 StartApp 恢复；未知状态 3 次重试、3 次返回、120 秒重启；第 4 次重启请求时安全停止。确认场次计数不因重启清零。

- [ ] **Step 6: 完成 60 分钟连续运行**

使用免费试炼、随机或阿加莎、SNAP 关闭、无付费档位运行至少 60 分钟。验收条件：无金块消费、无未知盲点、无重复点击风暴、停止按钮立即生效、所有恢复均有日志理由。

- [ ] **Step 7: 根据证据只调阈值/ROI，并重新跑全套**

任何阈值或 ROI 修改都必须同步 manifest 正负样本并重跑 Step 1。不得只为单张截图降低阈值；相似负样本必须仍不命中。

- [ ] **Step 8: 最终提交与验收记录**

```powershell
git add assets tests/fixtures docs/testing/adb-acceptance.md
git commit -m "test: verify conquest automation on mumu 12"
git status --short --branch
```

Expected: 工作树干净；分支只领先预期提交；`adb-acceptance.md` 中所有完成项有日期、配置和结果，没有未经执行的勾选。

---

## Final Review Checklist

- [ ] 对照设计文档第 6–12 节逐条勾选：11 个配置、两种正式策略、OCR 实验、四档选择、付费保护、撤退/SNAP、停止条件、恢复、实机矩阵均有实现或明确的验收证据。
- [ ] 运行 `rg -n "TODO|FIXME|pass$|NotImplemented|MaaXXX|my_action_111|my_reco_222|1920|1080" agent assets tests README.md`；Expected: 无未解释命中。
- [ ] 检查 Python 类型在层间一致：Interface 字符串 → `SessionConfig` 枚举/整数/布尔 → store → gate/action；`custom_action_param` 只在 Maa 边界解析一次。
- [ ] 检查所有 JSON Click 的 target 来源、所有进入/开战/撤退/认输/领取节点的直接前驱和负样本。
- [ ] 检查 `git diff --check`、完整 unittest、schema、maa-tools 和 60 分钟验收都是本次修改后的新结果，再报告完成。
