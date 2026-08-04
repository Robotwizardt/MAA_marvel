import sys

from maa.agent.agent_server import AgentServer
from maa.toolkit import Toolkit

# 这些模块虽然没有在 main.py 中直接调用，但导入时会执行
# @AgentServer.custom_action / custom_recognition 装饰器，从而完成注册。
from agent.actions import (
    configure_session,
    play_turn,
    record_event,
    recovery,
    route_conquest_tier,
    trace_runtime,
    warmup,
)
from agent.runtime import event_listener
from agent.runtime.task_cache import migrate_runtime_task_cache
from agent.recognitions import (
    card_selection,
    daily_task_reward,
    safe_entry,
    session_gate,
)


def main() -> None:
    """启动 AgentServer，并通过 socket_id 与 MaaFramework 客户端通信。"""
    Toolkit.init_option("./")
    migrate_runtime_task_cache()
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m agent.main <socket_id>")
    # MFAAvalonia 启动 Agent 时会把通信标识符放在最后一个命令行参数中。
    AgentServer.start_up(sys.argv[-1])
    # join() 会持续等待客户端下发 CustomAction / CustomRecognition 请求。
    AgentServer.join()
    AgentServer.shut_down()


if __name__ == "__main__":
    main()
