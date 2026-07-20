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
