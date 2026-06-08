from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence

from lingzu_teleop.config import RobotConnectionConfig
from lingzu_teleop.safety import ActionLimiter, SafetyLimits
from lingzu_teleop.types import ArmObservation, RecordedAction

logger = logging.getLogger(__name__)


class ELA3Robot:
    """Application-layer wrapper around lingzu_teleop.sdk.ELA3Interface."""

    def __init__(
        self,
        config: RobotConnectionConfig | None = None,
        *,
        arm: Any | None = None,
        safety_limits: SafetyLimits | None = None,
    ):
        self.config = config or RobotConnectionConfig()
        self._arm = arm
        self._connected = False
        self._control_loop_started = False
        self._last_action: RecordedAction | None = None
        self._limiter = ActionLimiter(safety_limits)

    @property
    def arm(self) -> Any:
        if self._arm is None:
            raise RuntimeError("Robot is not initialized. Call connect() first.")
        return self._arm

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_action(self) -> RecordedAction | None:
        return self._last_action

    def connect(self) -> None:
        if self._arm is None:
            from lingzu_teleop.sdk import ELA3Interface, LogLevel

            self._arm = ELA3Interface(
                can_name=self.config.can_name,
                host_can_id=self.config.host_can_id,
                default_kp=self.config.default_kp,
                default_kd=self.config.default_kd,
                logger_level=LogLevel.INFO,
                gravity_feedforward_ratio=self.config.gravity_feedforward_ratio,
                backend=self.config.backend,
                serial_port=self.config.serial_port,
                serial_baudrate=self.config.serial_baudrate,
            )

        if not self.arm.ConnectPort():
            raise RuntimeError(f"Failed to connect CAN interface: {self.config.can_name}")

        self._connected = True
        if self.config.auto_enable:
            self.arm.EnableArm()
            time.sleep(0.3)
        if self.config.start_control_loop:
            self.arm.start_control_loop(rate_hz=self.config.control_rate_hz)
            self._control_loop_started = True

    def disconnect(self) -> None:
        if self._arm is None:
            return
        try:
            if self._control_loop_started:
                self.arm.stop_control_loop()
                self._control_loop_started = False
        finally:
            try:
                self.arm.DisableArm()
            finally:
                self.arm.DisconnectPort()
                self._connected = False

    def __enter__(self) -> "ELA3Robot":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.disconnect()

    def get_observation(self) -> ArmObservation:
        q = self.arm.GetArmJointMsgs()
        dq = self.arm.GetArmJointVelocities()
        effort = self.arm.GetArmJointEfforts()
        pose = self.arm.GetArmEndPoseMsgs()
        status = self.arm.GetArmStatus()

        return ArmObservation(
            timestamp=time.monotonic(),
            joint_positions=q.to_list(include_gripper=True),
            joint_velocities=dq.to_list(include_gripper=True),
            joint_efforts=effort.to_list(include_gripper=True),
            end_pose=[pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz],
            status={
                "arm_status": status.arm_status,
                "ctrl_mode": status.ctrl_mode,
                "move_mode": status.move_mode,
                "motion_status": status.motion_status,
                "joint_enabled": list(status.joint_enabled),
                "joint_faults": list(status.joint_faults),
            },
        )

    def send_joint_action(
        self,
        action: Sequence[float],
        *,
        velocities: Optional[Sequence[float]] = None,
        source: str = "external",
        clip: bool = True,
    ) -> bool:
        previous = None
        if clip:
            try:
                previous = self.get_observation().joint_positions
            except Exception:
                previous = None
            action = self._limiter.clip_joint_action(action, previous=previous)
        else:
            action = list(action)

        if len(action) == 7:
            ok = self.arm.JointCtrlList(
                list(action),
                velocities=list(velocities[:6]) if velocities else None,
            )
        elif len(action) == 6:
            ok = self.arm.JointCtrl(
                *list(action),
                velocities=list(velocities[:6]) if velocities else None,
            )
        else:
            raise ValueError("Action must contain 6 arm joints or 7 joints including gripper")

        if ok:
            self._last_action = RecordedAction(
                timestamp=time.monotonic(),
                action=list(action),
                source=source,
                velocities=list(velocities) if velocities else None,
            )
        return bool(ok)

    def move_joints(self, positions: Sequence[float], *, duration: float = 2.0, block: bool = True) -> bool:
        positions = list(positions)
        if len(positions) not in (6, 7):
            raise ValueError("positions must contain 6 or 7 values")
        ok = self.arm.MoveJ(positions[:6], duration=duration, block=block)
        if ok and len(positions) == 7:
            ok = self.set_gripper(positions[6]) and ok
        return bool(ok)

    def set_gripper(self, angle: float) -> bool:
        return bool(self.arm.GripperCtrl(gripper_angle=float(angle)))

    def zero_torque(self, enabled: bool) -> bool:
        return bool(self.arm.ZeroTorqueMode(bool(enabled)))

    def emergency_stop(self) -> bool:
        return bool(self.arm.EmergencyStop())

    def reset_after_estop(self) -> None:
        self.arm.EnableArm()
        time.sleep(0.3)
        if self.config.start_control_loop and not self._control_loop_started:
            self.arm.start_control_loop(rate_hz=self.config.control_rate_hz)
            self._control_loop_started = True

    def kinematics(self) -> Any | None:
        return self.arm._get_kinematics()
