import random
import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.store import STORE
from agent.session.config import PlayStrategy
from agent.strategies.random_play import build_random_plan


RNG = random.Random()


@AgentServer.custom_action("MarvelPlayTurn")
class PlayTurn(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        state = STORE.require_state()
        if state.should_stop(time.monotonic()):
            return True
        if state.config.play_strategy is PlayStrategy.AGATHA:
            return True
        if state.config.play_strategy is PlayStrategy.OCR:
            return True

        controller = context.tasker.controller
        for swipe in build_random_plan(RNG).swipes:
            if state.should_stop(time.monotonic()):
                break
            job = controller.post_swipe(
                swipe.start.x,
                swipe.start.y,
                swipe.end.x,
                swipe.end.y,
                swipe.duration_ms,
            ).wait()
            if not job.succeeded:
                return False
            image = controller.post_screencap().get(wait=True)
            if context.run_recognition("公共-零能量", image) is not None:
                break
        return True
