from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LingzuELA3RobotConfig:
    type: str = "lingzu_ela3"
    id: str = "ela3"
    can_name: str = "can0"
    backend: str = "socketcan"
    serial_port: str | None = None
    control_rate_hz: float = 200.0
    default_kp: float = 80.0
    default_kd: float = 4.0
    cameras: dict = field(default_factory=dict)
