from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


@dataclass
class RobotConnectionConfig:
    can_name: str = "can0"
    backend: str = "socketcan"
    serial_port: Optional[str] = None
    serial_baudrate: int = 2_000_000
    host_can_id: int = 0xFD
    default_kp: float = 80.0
    default_kd: float = 4.0
    control_rate_hz: float = 200.0
    gravity_feedforward_ratio: float = 1.0
    auto_enable: bool = True
    start_control_loop: bool = True


@dataclass
class XboxTeleopConfig:
    device: str = "/dev/input/js0"
    profile: str = "auto"
    update_rate_hz: float = 100.0
    max_linear_velocity: float = 0.15
    max_angular_velocity: float = 1.5
    deadzone: Optional[float] = None
    input_smoothing: float = 0.35
    filter_omega: float = 14.0
    max_joint_velocity: float = 1.5
    max_ik_jump: float = 0.5


@dataclass
class RecordingConfig:
    sample_rate_hz: float = 20.0
    output_dir: str = "recordings"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def robot_config_from_mapping(data: Mapping[str, Any] | None) -> RobotConnectionConfig:
    return RobotConnectionConfig(**dict(data or {}))


def xbox_config_from_mapping(data: Mapping[str, Any] | None) -> XboxTeleopConfig:
    return XboxTeleopConfig(**dict(data or {}))


def recording_config_from_mapping(data: Mapping[str, Any] | None) -> RecordingConfig:
    return RecordingConfig(**dict(data or {}))
