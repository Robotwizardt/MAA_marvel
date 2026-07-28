# MAA Marvel Agent 开发教程

本文面向第一次阅读本项目 Agent 代码的开发者，目标是让你能够自己增加配置、动作、识别和任务入口。

## 1. 先理解项目的两部分

项目不是只有 Python：

- `assets/resource/pipeline/**/*.json`：负责识别页面、点击、等待和页面跳转。
- `agent/**/*.py`：负责运行时数据、复杂决策、动态选牌和动态路由。

基本调用链如下：

```text
MFAAvalonia 读取 interface/tasks
    ↓
运行某个 task.entry
    ↓
Pipeline 节点识别当前画面
    ↓
遇到 action: Custom 或 recognition: Custom
    ↓
AgentServer 根据注册名调用 Python 类
    ↓
Python 返回成功/失败、识别 box 或动态 next
    ↓
Pipeline 继续处理下一个页面
```

简单、稳定、可枚举的页面流程优先写 Pipeline；需要计数、随机、动态列表、复杂 OCR 后处理或跨回合状态时再写 Python。

## 2. Agent 是怎么启动的

入口是 `agent/main.py`。

`interface.json` 中声明：

```json
"agent": {
    "child_exec": "./agent_runtime/MAA_marvel_agent.exe",
    "child_args": []
}
```

客户端启动 Agent 后追加一个 `socket_id`。`agent/main.py` 使用它连接 MaaFramework：

```python
AgentServer.start_up(sys.argv[-1])
AgentServer.join()
```

`main.py` 还必须导入所有包含装饰器的模块。漏掉导入时，即使类已经写好，也不会注册。

## 3. CustomAction 是什么

CustomAction 用于“执行决策或动作”，返回 `True` 表示成功，返回 `False` 表示失败。

Python：

```python
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction


@AgentServer.custom_action("MarvelMyAction")
class MyAction(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        # 在这里读取参数、操作 controller 或修改运行状态。
        return True
```

Pipeline：

```json
"示例-执行动作": {
    "action": {
        "type": "Custom",
        "param": {
            "custom_action": "MarvelMyAction",
            "custom_action_param": {
                "value": 123
            }
        }
    },
    "next": [
        "示例-动作后页面"
    ],
    "on_error": [
        "公共-恢复决策"
    ]
}
```

注意：

1. `MarvelMyAction` 必须完全同名。
2. 新模块必须在 `agent/main.py` 间接或直接导入。
3. `CustomAction` 成功不代表游戏操作成功。点击或拖动后仍应识别画面变化。

## 4. CustomRecognition 是什么

CustomRecognition 用于复杂识别或布尔条件。返回的 `box` 不为空代表“命中”，`box=None` 代表“未命中”。

```python
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition


@AgentServer.custom_recognition("MarvelMyRecognition")
class MyRecognition(CustomRecognition):
    def analyze(self, context: Context, argv):
        matched = True
        return CustomRecognition.AnalyzeResult(
            box=(0, 0, 720, 1280) if matched else None,
            detail={"matched": matched},
        )
```

Pipeline：

```json
"示例-条件命中": {
    "recognition": {
        "type": "Custom",
        "param": {
            "custom_recognition": "MarvelMyRecognition",
            "custom_recognition_param": {
                "command": "example"
            }
        }
    },
    "next": [
        "示例-成功分支"
    ]
}
```

如果只是固定文字或固定图片，直接使用 Pipeline OCR/TemplateMatch，不要为了统一而写 Python。

## 5. 参数是怎么进入 Agent 的

`征服-初始化会话` 调用 `MarvelConfigureSession`：

```text
custom_action_param
    ↓
parse_json_object()
    ↓
SessionConfig.from_mapping()
    ↓
SessionState(config=...)
    ↓
STORE.configure()
```

各文件职责：

- `agent/session/config.py`：不会在运行中变化的用户配置。
- `agent/session/state.py`：局数、回合、SNAP、恢复次数等动态状态。
- `agent/runtime/store.py`：保存当前唯一会话，供所有 Agent 类共享。
- `agent/runtime/commands.py`：解析参数和集中处理 Pipeline 事件。

不要在不同 Action 中各自保存一份全局变量，否则任务重启后很容易残留脏状态。

## 6. 如何新增一个普通配置项

假设增加“每回合最多出牌数”。

### 第一步：在 SessionConfig 增加字段

```python
@dataclass(frozen=True, slots=True)
class SessionConfig:
    max_cards_per_turn: int = 12
```

在 `from_mapping()` 读取：

```python
max_cards_per_turn=int(values.get("max_cards_per_turn", 12)),
```

在 `validate()` 校验：

```python
if not 1 <= self.max_cards_per_turn <= 20:
    raise ValueError("max_cards_per_turn must be between 1 and 20")
```

### 第二步：在任务选项文件增加 UI

编辑 `assets/tasks/征服模式.json`：

```json
"征服-每回合最多出牌数": {
    "type": "input",
    "label": "每回合最多出牌数",
    "inputs": [
        {
            "name": "数量",
            "default": "12",
            "pipeline_type": "int",
            "verify": "^([1-9]|1[0-9]|20)$"
        }
    ],
    "pipeline_override": {
        "征服-初始化会话": {
            "action": {
                "param": {
                    "custom_action_param": {
                        "max_cards_per_turn": "{数量}"
                    }
                }
            }
        }
    }
}
```

并把选项名加入任务的 `option` 数组。

### 第三步：读取配置

```python
state = STORE.require_state()
limit = state.config.max_cards_per_turn
```

### 第四步：测试默认值、输入转换和非法范围

至少补充：

- 默认值测试。
- 字符串转整数测试。
- 0、负数、过大值拒绝测试。
- UI 默认值与 Python 默认值一致测试。

## 7. 如何新增一个运行事件

如果 Pipeline 已经确认“第 3 回合开始”，不要让 Python 再猜一次画面。让 Pipeline 调用 `MarvelRecordEvent`：

```json
"custom_action_param": {
    "event": "turn_started",
    "value": 3
}
```

如需新事件，在 `agent/runtime/commands.py` 的 `apply_event()` 增加分支，并给 `SessionState` 增加明确方法：

```python
if event == "my_event":
    state.record_my_event()
    return
```

不要从外部直接到处修改 `state.some_field`，否则状态变化来源很难追踪。

## 8. 如何新增一种出牌策略

假设新增 `lowest_cost`。

### 第一步：增加枚举

```python
class PlayStrategy(str, Enum):
    AGATHA = "agatha"
    OCR = "ocr"
    LOWEST_COST = "lowest_cost"
```

### 第二步：写纯决策函数

纯决策函数不要截图和拖动，输入手牌，输出选择：

```python
def choose_lowest(hand: BattleHand) -> DetectedCard | None:
    cards = _affordable(hand)
    if not cards:
        return None
    return min(cards, key=lambda card: (card.cost, card.slot))
```

### 第三步：接入 PlayTurn

```python
if strategy is PlayStrategy.LOWEST_COST:
    choose = choose_lowest
```

拖牌、成功确认、场地满处理继续复用现有循环。

### 第四步：增加 UI case

在“征服-出牌策略”的 `cases` 中增加：

```json
{
    "name": "lowest_cost",
    "label": "最低费用优先",
    "pipeline_override": {
        "征服-初始化会话": {
            "action": {
                "param": {
                    "custom_action_param": {
                        "play_strategy": "lowest_cost"
                    }
                }
            }
        }
    }
}
```

## 9. 如何新增 OCR 识别

先明确：识别什么、ROI 在哪里、失败是否允许点击。

本项目使用 720×1280 坐标。推荐流程：

1. 保存一张真实截图。
2. 确定尽量小且稳定的 ROI。
3. 先写不点击的识别节点或测试探针。
4. 记录真实 OCR 文本和置信度。
5. 只有识别稳定后才接动作。

Python 直接 OCR 示例：

```python
detail = context.run_recognition_direct(
    JRecognitionType.OCR,
    JOCR(roi=(x, y, w, h), threshold=0.8),
    image,
)
```

不要把“调用 OCR 没抛异常”当成命中。必须检查 `detail` 和其中的识别结果。

## 10. 如何新增一个训练测试任务

训练任务应满足：

- 用户手动进入稳定页面。
- 入口先识别页面，不命中就不操作。
- 只测试一个小功能。
- 不自动进入付费或消耗资源页面。

任务定义：

```json
{
    "name": "训练模式-我的功能测试",
    "entry": "训练-我的功能测试入口",
    "default_check": false,
    "resource": ["官服"],
    "controller": ["安卓端"]
}
```

Pipeline：

```json
"训练-我的功能测试入口": {
    "recognition": {
        "type": "OCR",
        "param": {
            "roi": [500, 1120, 220, 120],
            "expected": ["结束回[合会]"]
        }
    },
    "action": {
        "type": "Custom",
        "param": {
            "custom_action": "MarvelMyAction",
            "custom_action_param": {}
        }
    }
}
```

入口识别失败时不会执行动作，这比 `DirectHit + CustomAction` 安全得多。

## 11. 必须执行的检查

修改后至少运行：

```powershell
python -m unittest discover -s tests -v
python tools/validate_schema.py --schema-dir deps/tools --resource-dirs assets/resource --exclude-dirs assets/resource/announcement --interface-files assets/interface.json --task-dirs assets/tasks
```

然后检查 Custom 名称是否双向一致：

```powershell
rg -n 'custom_action|custom_recognition' assets/resource
rg -n '@AgentServer\.custom_action|@AgentServer\.custom_recognition' agent
```

最后才进行真机测试。涉及购买、门票、开战、结算等动作时，先用免费或训练环境，并确认动作后状态。

## 12. 常见错误

### 写了类但 Pipeline 找不到

检查：

- 装饰器名字是否完全一致。
- 模块是否被 `agent/main.py` 导入。
- 发布包是否重新构建了独立 Agent。

### 修改了 UI 但 Python 没收到

检查 `pipeline_override` 是否覆盖到代码实际读取的 `custom_action_param` 路径。

### 拖牌执行了但其实没成功

控制器返回成功只表示 swipe 指令执行成功。必须再截图，通过能量、手牌数量或稳定页面变化确认业务成功。

### 任务一启动就全部完成

查看日志中的“准备执行任务队列：任务数量”。如果是 0，说明任务没有勾选，不是 Agent 执行完了。

### 本地源码改了，发布软件行为没变

发布软件运行的是 `agent_runtime/MAA_marvel_agent.exe`，不是工作区 Python 文件。需要重新构建并替换 Agent，或重新发布版本。

## 13. 推荐阅读顺序

按下面顺序阅读最容易理解：

1. `assets/tasks/征服模式.json`
2. `assets/resource/pipeline/conquest/navigation.json`
3. `agent/actions/configure_session.py`
4. `agent/session/config.py`
5. `agent/runtime/store.py`
6. `agent/session/state.py`
7. `agent/actions/play_turn.py`
8. `agent/recognitions/card_selection.py`
9. `agent/strategies/ocr.py`
10. `assets/resource/pipeline/common/battle.json`

沿着这个顺序，可以看到“界面配置 → 初始化 → 保存状态 → 读取状态 → 执行动作 → 回到 Pipeline”的完整链路。
