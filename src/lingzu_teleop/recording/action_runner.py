from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import Sequence

from lingzu_teleop.robot import ELA3Robot
from lingzu_teleop.types import ArmObservation

logger = logging.getLogger(__name__)

PolicyFn = Callable[[ArmObservation], Sequence[float] | None]


class ActionRunner:
    """Execute recorded actions or model policy outputs on the robot."""

    def __init__(
        self,
        robot: ELA3Robot,
        *,
        rate_hz: float = 20.0,
        clip_actions: bool = True,
        source: str = "runner",
    ):
        if rate_hz <= 0:
            raise ValueError("rate_hz must be positive")
        self.robot = robot
        self.rate_hz = float(rate_hz)
        self.clip_actions = clip_actions
        self.source = source

    def run_actions(self, actions: Iterable[Sequence[float]]) -> int:
        period = 1.0 / self.rate_hz
        next_tick = time.monotonic()
        count = 0
        for action in actions:
            next_tick += period
            ok = self.robot.send_joint_action(
                action,
                source=self.source,
                clip=self.clip_actions,
            )
            if not ok:
                logger.warning("Action rejected at index %d", count)
            count += 1
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif sleep_s < -period:
                next_tick = time.monotonic()
        return count

    def run_policy(self, policy: PolicyFn, *, max_steps: int | None = None) -> int:
        period = 1.0 / self.rate_hz
        next_tick = time.monotonic()
        count = 0
        while max_steps is None or count < max_steps:
            next_tick += period
            observation = self.robot.get_observation()
            action = policy(observation)
            if action is None:
                break
            self.robot.send_joint_action(
                action,
                source=self.source,
                clip=self.clip_actions,
            )
            count += 1
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif sleep_s < -period:
                next_tick = time.monotonic()
        return count
