from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class SafetyLimits:
    joint_lower: list[float] = field(
        default_factory=lambda: [-2.79, 0.0, -4.01, -1.57, -1.57, -1.57, 0.0]
    )
    joint_upper: list[float] = field(
        default_factory=lambda: [2.79, 3.67, 0.0, 1.57, 1.57, 1.57, 1.5708]
    )
    max_joint_step: float = 0.08
    max_gripper_step: float = 0.12


class ActionLimiter:
    """Clip model/teleop actions before they reach the SDK."""

    def __init__(self, limits: SafetyLimits | None = None):
        self.limits = limits or SafetyLimits()

    def clip_joint_action(
        self,
        action: Sequence[float],
        *,
        previous: Sequence[float] | None = None,
    ) -> list[float]:
        arr = np.asarray(action, dtype=float)
        if arr.ndim != 1 or arr.size not in (6, 7):
            raise ValueError("Joint action must be a 6D or 7D vector")

        lower = np.asarray(self.limits.joint_lower[: arr.size], dtype=float)
        upper = np.asarray(self.limits.joint_upper[: arr.size], dtype=float)
        arr = np.clip(arr, lower, upper)

        if previous is not None:
            prev = np.asarray(previous[: arr.size], dtype=float)
            step_limits = np.full(arr.size, self.limits.max_joint_step, dtype=float)
            if arr.size == 7:
                step_limits[6] = self.limits.max_gripper_step
            arr = np.clip(arr, prev - step_limits, prev + step_limits)

        return arr.tolist()
