import random
import time

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction

from agent.runtime.store import STORE
from agent.session.config import PlayStrategy
from agent.strategies.random_play import LANE_TARGETS, build_random_plan


RNG = random.Random()


def _box_center(box: object) -> tuple[int, int]:
    if isinstance(box, (list, tuple)):
        x, y, width, height = box
    else:
        x = getattr(box, "x")
        y = getattr(box, "y")
        width = getattr(box, "w")
        height = getattr(box, "h")
    return int(x + width // 2), int(y + height // 2)


@AgentServer.custom_action("MarvelPlayTurn")
class PlayTurn(CustomAction):
    def run(self, context: Context, argv: CustomAction.RunArg) -> bool:
        state = STORE.require_state()
        if state.should_stop(time.monotonic()):
            return True
        if state.config.play_strategy is PlayStrategy.AGATHA:
            return True
        if state.config.play_strategy is PlayStrategy.OCR:
            return self._play_ocr(context, state)

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

    def _play_ocr(self, context: Context, state) -> bool:
        controller = context.tasker.controller
        for _ in range(4):
            if state.should_stop(time.monotonic()):
                break
            image = controller.post_screencap().get(wait=True)
            detail = context.run_recognition("公共-OCR选牌", image)
            if detail is None or detail.box is None:
                break
            start_x, start_y = _box_center(detail.box)
            lane = RNG.choice(LANE_TARGETS)
            end_x = max(85, min(lane.x + RNG.randint(-12, 12), 635))
            end_y = max(600, min(lane.y + RNG.randint(-12, 12), 720))
            job = controller.post_swipe(
                start_x,
                start_y,
                end_x,
                end_y,
                350,
            ).wait()
            if not job.succeeded:
                return False
        return True
