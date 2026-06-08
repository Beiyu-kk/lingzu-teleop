from __future__ import annotations

from typing import Any, Mapping

from lingzu_teleop.config import RobotConnectionConfig
from lingzu_teleop.lerobot_adapter import build_lerobot_frame, infer_lerobot_features
from lingzu_teleop.robot import ELA3Robot

from lingzu_teleop.lerobot_robot_lingzu_ela3.config_lingzu_ela3 import LingzuELA3RobotConfig


class LingzuELA3Robot:
    """Plugin-shaped robot adapter for later LeRobot integration."""

    name = "lingzu_ela3"

    def __init__(
        self,
        config: LingzuELA3RobotConfig | None = None,
        *,
        camera_provider: Any | None = None,
    ):
        self.config = config or LingzuELA3RobotConfig()
        self.camera_provider = camera_provider
        self.robot = ELA3Robot(
            RobotConnectionConfig(
                can_name=self.config.can_name,
                backend=self.config.backend,
                serial_port=self.config.serial_port,
                control_rate_hz=self.config.control_rate_hz,
                default_kp=self.config.default_kp,
                default_kd=self.config.default_kd,
            )
        )
        self._frame_index = 0

    @property
    def observation_features(self) -> dict[str, dict[str, Any]]:
        camera_shapes = {}
        for name, cfg in self.config.cameras.items():
            height = int(cfg.get("height", 480))
            width = int(cfg.get("width", 640))
            camera_shapes[name] = (height, width, 3)
        return infer_lerobot_features(camera_shapes=camera_shapes)

    @property
    def action_features(self) -> dict[str, dict[str, Any]]:
        return {"action": infer_lerobot_features()["action"]}

    @property
    def is_connected(self) -> bool:
        return self.robot.connected

    def connect(self) -> None:
        self.robot.connect()
        if self.camera_provider is not None:
            self.camera_provider.connect()

    def disconnect(self) -> None:
        try:
            if self.camera_provider is not None:
                self.camera_provider.disconnect()
        finally:
            self.robot.disconnect()

    def get_observation(self) -> dict[str, Any]:
        obs = self.robot.get_observation()
        images = self.camera_provider.get_frames() if self.camera_provider is not None else {}
        frame = build_lerobot_frame(
            obs,
            action=self.robot.last_action.action if self.robot.last_action else None,
            images=images,
            frame_index=self._frame_index,
        )
        self._frame_index += 1
        return frame

    def send_action(self, action: Mapping[str, Any] | list[float]) -> list[float]:
        if isinstance(action, Mapping):
            values = action.get("action")
            if values is None:
                raise ValueError("Action mapping must contain an 'action' key")
        else:
            values = action
        values = list(values)
        self.robot.send_joint_action(values, source="lerobot", clip=True)
        return values

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None
