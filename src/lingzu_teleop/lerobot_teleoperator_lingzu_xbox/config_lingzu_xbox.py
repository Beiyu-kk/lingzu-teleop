from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LingzuXboxTeleoperatorConfig:
    type: str = "lingzu_xbox"
    device: str = "/dev/input/js0"
    profile: str = "auto"
    update_rate_hz: float = 100.0
    max_linear_velocity: float = 0.15
    max_angular_velocity: float = 1.5
