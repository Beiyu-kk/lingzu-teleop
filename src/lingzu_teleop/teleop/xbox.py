from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional

from lingzu_teleop.sdk import ArmEndPose
from lingzu_teleop.sdk.controller_profiles import ControllerProfile
from lingzu_teleop.sdk.joystick import LinuxJoystick

from lingzu_teleop.config import XboxTeleopConfig
from lingzu_teleop.robot import ELA3Robot

logger = logging.getLogger(__name__)

SPEED_LEVELS = [
    ("极慢", 0.10),
    ("慢", 0.25),
    ("中", 0.50),
    ("快", 0.75),
    ("最大", 1.00),
]

HOME_POSITIONS = [0.0, 0.785, -0.785, 0.0, 0.0, 0.0]
ZERO_POSITIONS = [0.0] * 6

GRIPPER_OPEN_ANGLE = 0.0
GRIPPER_CLOSE_ANGLE = 1.5708
GRIPPER_TAP_STEP = math.radians(2.0)
GRIPPER_HOLD_DELAY_S = 0.12
GRIPPER_HOLD_RATE_RAD_S = 0.45
GRIPPER_COMMAND_EPS = 1e-4
GRIPPER_LOG_INTERVAL_S = 0.45


class XboxCartesianTeleop:
    """基于 SDK 的 Xbox 手柄机械臂控制器（末端坐标模式）。"""

    def __init__(
        self,
        robot: ELA3Robot,
        joystick: LinuxJoystick,
        profile: ControllerProfile,
        config: XboxTeleopConfig | None = None,
    ):
        self._robot = robot
        self._arm = robot.arm
        self._joy = joystick
        self._profile = profile
        self._config = config or XboxTeleopConfig()
        self._rate = self._config.update_rate_hz
        self._dt = 1.0 / self._rate

        self._max_lin_vel = self._config.max_linear_velocity
        self._max_ang_vel = self._config.max_angular_velocity
        self._dz_threshold = (
            self._config.deadzone
            if self._config.deadzone is not None
            else self._profile.default_deadzone
        )
        self._input_alpha = self._config.input_smoothing
        self._filter_omega = self._config.filter_omega
        self._max_ik_jump = self._config.max_ik_jump

        self._kin = robot.kinematics()
        if self._kin is None:
            logger.warning("Pinocchio 不可用，仅支持按钮功能（无笛卡尔控制）")

        self._speed_idx = 2
        self._speed_factor = SPEED_LEVELS[self._speed_idx][1]
        self._running = False
        self._zero_torque = False
        self._is_moving = False
        self._exit_requested = False
        self._estop = False
        self._initialized = threading.Event()

        self._target_pose: Optional[ArmEndPose] = None
        self._prev_pose: Optional[ArmEndPose] = None

        self._ik_seed: Optional[list[float]] = None
        self._ik_filter_pos: Optional[list[float]] = None
        self._ik_filter_vel: list[float] = [0.0] * 6
        self._ik_raw: Optional[list[float]] = None
        self._consecutive_rejects = 0
        self._consecutive_ik_fails = 0
        self._seed_just_init = False
        self._resync_cooldown = 0

        self._sv = [0.0] * 6
        self._gripper_angle = GRIPPER_OPEN_ANGLE
        self._gripper_dpad_dir = 0
        self._gripper_hold_started_at = 0.0
        self._gripper_hold_last_at = 0.0
        self._last_gripper_log_at = 0.0

        self._prev_btn = [0] * LinuxJoystick.MAX_BUTTONS
        self._diag_tick = 0

    @property
    def exit_requested(self) -> bool:
        return self._exit_requested

    @property
    def current_action(self) -> list[float] | None:
        if self._ik_filter_pos is None:
            return None
        return list(self._ik_filter_pos) + [self._gripper_angle]

    @property
    def initialized(self) -> bool:
        return self._initialized.is_set()

    def wait_until_initialized(self, timeout: float | None = None) -> bool:
        return self._initialized.wait(timeout=timeout)

    def start(self, *, initialize_to_zero: bool = True) -> None:
        self._running = True
        self._initialize(initialize_to_zero=initialize_to_zero)
        self._initialized.set()
        self._print_banner()

        period = 1.0 / self._rate
        next_tick = time.monotonic()
        while self._running and not self._exit_requested:
            next_tick += period
            try:
                self._tick()
            except Exception as exc:
                logger.error("控制循环异常: %s", exc)
            now = time.monotonic()
            sleep_s = next_tick - now
            if sleep_s < -period:
                next_tick = now + period
            elif sleep_s > 0:
                time.sleep(sleep_s)

    def stop(self) -> None:
        self._running = False

    def _initialize(self, *, initialize_to_zero: bool) -> None:
        if initialize_to_zero:
            logger.info("移动到零位...")
            self._robot.move_joints(ZERO_POSITIONS, duration=3.0, block=True)
            time.sleep(0.3)

        q = self._read_averaged_feedback(n_samples=3)
        self._ik_seed = list(q)
        self._ik_filter_pos = list(q)
        self._ik_filter_vel = [0.0] * 6
        self._ik_raw = list(q)
        self._seed_just_init = True
        self._consecutive_rejects = 0
        self._consecutive_ik_fails = 0

        if self._kin is not None:
            self._target_pose = self._kin.forward_kinematics(q)
            self._prev_pose = None
            p = self._target_pose
            logger.info(
                "初始化完成, 末端位姿: (%.3f, %.3f, %.3f) m  (%.2f, %.2f, %.2f) rad",
                p.x,
                p.y,
                p.z,
                p.rx,
                p.ry,
                p.rz,
            )
        else:
            logger.info("初始化完成 (无运动学)")
        self._sync_gripper_feedback()

    def _print_banner(self) -> None:
        print("\n" + "=" * 52)
        print("     EL-A3 手柄控制（纯 SDK 模式）")
        print("=" * 52)
        print(f"  控制器映射:  {self._profile.display_name} [{self._profile.profile_id}]")
        print("  左摇杆       →  XY 平移")
        print("  LT / RT      →  Z 下/上")
        print("  右摇杆       →  Yaw / Roll")
        print("  LB / RB      →  Pitch")
        print("  A            →  切换速度档")
        print("  B            →  回 Home")
        print("  X            →  回零位")
        print("  Y            →  零力矩模式（可拖动）")
        print("  D-pad ↑      →  夹爪小步打开")
        print("  D-pad ↓      →  夹爪小步关闭")
        print("  Back         →  急停")
        print("  Start        →  退出")
        print("=" * 52)
        ik_status = "末端坐标 IK" if self._kin else "不可用（缺少 pinocchio）"
        print(f"  笛卡尔控制:  {ik_status}")
        if self._target_pose is not None:
            p = self._target_pose
            print(f"  初始末端:    ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m")
            print(f"               ({p.rx:.2f}, {p.ry:.2f}, {p.rz:.2f}) rad")
        self._log_speed()

    def _log_speed(self) -> None:
        name, factor = SPEED_LEVELS[self._speed_idx]
        lin_mm = self._max_lin_vel * factor * 1000
        ang = self._max_ang_vel * factor
        print(f"  速度档位:    {self._speed_idx + 1}/5 [{name}] ({lin_mm:.0f}mm/s, {ang:.2f}rad/s)")

    def _apply_dz(self, val: float) -> float:
        if abs(val) < self._dz_threshold:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        return sign * (abs(val) - self._dz_threshold) / (1.0 - self._dz_threshold)

    def _apply_trigger(self, raw: float) -> float:
        norm = max(0.0, min(raw, 1.0))
        dz = self._dz_threshold * 1.5
        if norm < dz:
            return 0.0
        return (norm - dz) / (1.0 - dz)

    def _axis_value(self, binding) -> float:
        return binding.read(self._joy.axes)

    def _trigger_value(self, binding) -> float:
        return binding.read(self._joy.axes, self._joy.buttons)

    def _button_state(self, idx: Optional[int]) -> int:
        if idx is None or idx >= len(self._joy.buttons):
            return 0
        return self._joy.buttons[idx]

    def _btn_edge(self, idx: Optional[int]) -> bool:
        if idx is None or idx >= len(self._joy.buttons):
            return False
        return self._joy.buttons[idx] == 1 and self._prev_btn[idx] == 0

    def _tick(self) -> None:
        if not self._joy.connected:
            return

        buttons = self._profile.buttons
        sticks = self._profile.sticks

        if self._btn_edge(buttons.south):
            self._speed_idx = (self._speed_idx + 1) % len(SPEED_LEVELS)
            self._speed_factor = SPEED_LEVELS[self._speed_idx][1]
            self._log_speed()

        if self._btn_edge(buttons.east):
            self._async_move(HOME_POSITIONS, "Home")
        if self._btn_edge(buttons.west):
            self._async_move(ZERO_POSITIONS, "零位")
        if self._btn_edge(buttons.north):
            self._toggle_zero_torque()
        if self._btn_edge(buttons.back):
            self._emergency_stop()
        if self._btn_edge(buttons.start):
            logger.info("收到退出请求")
            self._exit_requested = True

        self._handle_gripper_dpad(self._axis_value(sticks.dpad_y))
        self._prev_btn = list(self._joy.buttons)

        if self._zero_torque or self._is_moving or self._estop:
            self._periodic_status()
            return
        if self._kin is None or self._target_pose is None:
            self._periodic_status()
            return

        max_lin = self._max_lin_vel * self._speed_factor
        max_ang = self._max_ang_vel * self._speed_factor
        raw = [
            -self._apply_dz(self._axis_value(sticks.ly)) * max_lin,
            -self._apply_dz(self._axis_value(sticks.lx)) * max_lin,
            (
                self._apply_trigger(self._trigger_value(sticks.rt))
                - self._apply_trigger(self._trigger_value(sticks.lt))
            )
            * max_lin,
            self._apply_dz(self._axis_value(sticks.ry)) * max_ang,
            (self._button_state(buttons.rb) - self._button_state(buttons.lb)) * max_ang,
            self._apply_dz(self._axis_value(sticks.rx)) * max_ang,
        ]

        if sum(abs(r) for r in raw) < 1e-6:
            decay = min(self._input_alpha * 3.0, 1.0)
            self._sv = [(1 - decay) * s for s in self._sv]
        else:
            a = self._input_alpha
            self._sv = [a * r + (1 - a) * s for r, s in zip(raw, self._sv)]

        if self._resync_cooldown > 0:
            self._resync_cooldown -= 1
            self._send_filtered()
            self._periodic_status()
            return

        if sum(abs(v) for v in self._sv) > 1e-7:
            p = self._target_pose
            self._prev_pose = ArmEndPose(x=p.x, y=p.y, z=p.z, rx=p.rx, ry=p.ry, rz=p.rz)
            self._target_pose.x += self._sv[0] * self._dt
            self._target_pose.y += self._sv[1] * self._dt
            self._target_pose.z += self._sv[2] * self._dt
            self._target_pose.rx += self._sv[3] * self._dt
            self._target_pose.ry += self._sv[4] * self._dt
            self._target_pose.rz += self._sv[5] * self._dt

            try:
                q_sol, ik_err = self._kin.ik_step(
                    self._target_pose,
                    self._ik_seed,
                    damping=5e-3,
                    max_step=self._max_ik_jump,
                )
                if q_sol is not None and self._accept_ik(q_sol):
                    self._ik_raw = q_sol
                    self._ik_seed = list(q_sol)
                    self._consecutive_ik_fails = 0
                    if ik_err > 0.01:
                        achieved = self._kin.forward_kinematics(q_sol)
                        blend = min((ik_err - 0.01) * 20.0, 0.7)
                        self._target_pose.x += blend * (achieved.x - self._target_pose.x)
                        self._target_pose.y += blend * (achieved.y - self._target_pose.y)
                        self._target_pose.z += blend * (achieved.z - self._target_pose.z)
                        self._target_pose.rx += blend * (achieved.rx - self._target_pose.rx)
                        self._target_pose.ry += blend * (achieved.ry - self._target_pose.ry)
                        self._target_pose.rz += blend * (achieved.rz - self._target_pose.rz)
                else:
                    self._target_pose = self._prev_pose
                    self._consecutive_ik_fails += 1
                    if self._consecutive_ik_fails >= 10:
                        logger.warning(
                            "IK 连续失败 %d 次 (err=%.4f)，目标可能超出工作空间",
                            self._consecutive_ik_fails,
                            ik_err,
                        )
                    if self._consecutive_ik_fails >= 50:
                        logger.warning("IK 连续失败 50+ 次，自动重新同步...")
                        self._resync_ik()
            except Exception as exc:
                logger.error("IK 异常: %s", exc)
                self._target_pose = self._prev_pose
        else:
            self._consecutive_ik_fails = 0

        self._send_filtered()
        self._periodic_status()

    def _sync_gripper_feedback(self) -> None:
        try:
            positions = self._arm.GetArmJointMsgs().to_list(include_gripper=True)
            if len(positions) >= 7:
                self._gripper_angle = self._clamp_gripper(positions[6])
        except Exception:
            pass

    @staticmethod
    def _clamp_gripper(angle: float) -> float:
        return max(GRIPPER_OPEN_ANGLE, min(GRIPPER_CLOSE_ANGLE, float(angle)))

    def _handle_gripper_dpad(self, dpad_y: float) -> None:
        if dpad_y < -0.5:
            direction = -1
        elif dpad_y > 0.5:
            direction = 1
        else:
            direction = 0

        now = time.monotonic()
        if direction != self._gripper_dpad_dir:
            self._gripper_dpad_dir = direction
            self._gripper_hold_started_at = now
            self._gripper_hold_last_at = now
            if direction:
                self._nudge_gripper(direction * GRIPPER_TAP_STEP, force=True, log_now=True)
            return

        if direction == 0:
            return
        if now - self._gripper_hold_started_at < GRIPPER_HOLD_DELAY_S:
            self._gripper_hold_last_at = now
            return

        dt = max(0.0, now - self._gripper_hold_last_at)
        self._gripper_hold_last_at = now
        self._nudge_gripper(direction * GRIPPER_HOLD_RATE_RAD_S * dt)

    def _nudge_gripper(self, delta: float, *, force: bool = False, log_now: bool = False) -> None:
        target = self._clamp_gripper(self._gripper_angle + delta)
        if not force and abs(target - self._gripper_angle) < GRIPPER_COMMAND_EPS:
            return
        self._gripper_angle = target
        self._robot.set_gripper(self._gripper_angle)
        if self._ik_filter_pos is not None:
            self._robot.send_joint_action(
                list(self._ik_filter_pos) + [self._gripper_angle],
                velocities=list(self._ik_filter_vel),
                source="xbox",
                clip=False,
            )

        now = time.monotonic()
        if log_now or now - self._last_gripper_log_at >= GRIPPER_LOG_INTERVAL_S:
            action = "打开" if delta < 0 else "关闭"
            logger.info("夹爪%s: %.1f°", action, math.degrees(self._gripper_angle))
            self._last_gripper_log_at = now

    def _accept_ik(self, q_new: list[float]) -> bool:
        if self._ik_seed is None:
            return True
        max_diff = max(abs(q_new[i] - self._ik_seed[i]) for i in range(6))
        if max_diff <= self._max_ik_jump:
            self._consecutive_rejects = 0
            self._seed_just_init = False
            return True
        if self._seed_just_init:
            self._seed_just_init = False
            return True
        self._consecutive_rejects += 1
        if self._consecutive_rejects >= 5:
            logger.warning(
                "疑似奇异区: IK 跳变=%.3frad, 已保护 %d 帧",
                max_diff,
                self._consecutive_rejects,
            )
        if self._consecutive_rejects >= 50:
            logger.warning("IK 连续拒绝 50+ 帧，自动重新同步...")
            self._resync_ik()
        return False

    def _read_averaged_feedback(self, n_samples: int = 5, interval: float = 0.004) -> list[float]:
        samples = []
        for _ in range(n_samples):
            samples.append(self._arm.GetArmJointMsgs().to_list()[:6])
            time.sleep(interval)
        return [sum(s[i] for s in samples) / len(samples) for i in range(6)]

    def _resync_ik(self) -> None:
        q_avg = self._read_averaged_feedback()
        self._ik_seed = list(q_avg)
        self._ik_filter_pos = list(q_avg)
        self._ik_raw = list(q_avg)
        self._ik_filter_vel = [v * 0.2 for v in self._ik_filter_vel]
        self._seed_just_init = True
        self._consecutive_rejects = 0
        self._consecutive_ik_fails = 0
        self._resync_cooldown = 5
        if self._kin is not None:
            self._target_pose = self._kin.forward_kinematics(q_avg)
            self._prev_pose = None

    def _send_filtered(self) -> None:
        if self._ik_raw is None and self._ik_filter_pos is None:
            return
        if self._ik_filter_pos is None and self._ik_raw is not None:
            self._ik_filter_pos = list(self._ik_raw)
            self._ik_filter_vel = [0.0] * 6
        if self._ik_raw is not None and self._ik_filter_pos is not None:
            omega = self._filter_omega
            dt = self._dt
            a = omega * dt
            ea = math.exp(-a)
            for i in range(6):
                err = self._ik_raw[i] - self._ik_filter_pos[i]
                vel = self._ik_filter_vel[i]
                err_new = ea * ((1.0 + a) * err - dt * vel)
                vel_new = ea * (omega * omega * dt * err + (1.0 - a) * vel)
                self._ik_filter_pos[i] = self._ik_raw[i] - err_new
                self._ik_filter_vel[i] = vel_new
        if self._ik_filter_pos is not None:
            self._robot.send_joint_action(
                list(self._ik_filter_pos) + [self._gripper_angle],
                velocities=list(self._ik_filter_vel),
                source="xbox",
                clip=False,
            )

    def _async_move(self, positions: list[float], name: str) -> None:
        if self._is_moving:
            logger.warning("正在执行其他动作，请稍后再试")
            return
        logger.info("正在移动到 %s...", name)
        self._is_moving = True
        threading.Thread(target=self._do_move, args=(positions, name), daemon=True).start()

    def _do_move(self, positions: list[float], name: str) -> None:
        try:
            if self._zero_torque:
                self._robot.zero_torque(False)
                self._zero_torque = False
                time.sleep(0.2)
            if self._estop:
                self._robot.reset_after_estop()
                self._estop = False
            self._robot.move_joints(positions, duration=2.0, block=True)
            self._ik_seed = list(positions)
            self._ik_filter_pos = list(positions)
            self._ik_filter_vel = [0.0] * 6
            self._ik_raw = list(positions)
            self._seed_just_init = True
            self._consecutive_rejects = 0
            self._consecutive_ik_fails = 0
            if self._kin is not None:
                self._target_pose = self._kin.forward_kinematics(positions)
                self._prev_pose = None
            logger.info("已到达 %s", name)
        except Exception as exc:
            logger.error("运动异常: %s", exc)
        finally:
            self._is_moving = False

    def _toggle_zero_torque(self) -> None:
        if self._is_moving:
            return
        new_state = not self._zero_torque
        logger.info("%s 零力矩模式...", "开启" if new_state else "关闭")
        if not new_state:
            q = self._read_averaged_feedback()
            self._robot.send_joint_action(q, source="xbox_resync", clip=False)
            time.sleep(0.05)
        ok = self._robot.zero_torque(new_state)
        if ok:
            self._zero_torque = new_state
            if new_state:
                print(">>> 零力矩模式已开启: 可手动拖动机械臂 <<<")
            else:
                self._resync_ik()
                print(">>> 零力矩模式已关闭: 恢复手柄控制 <<<")
        else:
            logger.error("零力矩模式切换失败")

    def _emergency_stop(self) -> None:
        self._robot.emergency_stop()
        self._estop = True
        print("\n!!! 急停已执行 — 按 B(Home) 或 X(零位) 恢复 !!!")

    def _periodic_status(self) -> None:
        self._diag_tick += 1
        if self._diag_tick < int(self._rate * 5):
            return
        self._diag_tick = 0
        q = self._arm.GetArmJointMsgs().to_list()[:6]
        degs = [f"{v * 180 / math.pi:.1f}" for v in q]
        mode = "零力矩" if self._zero_torque else ("急停" if self._estop else "正常")
        try:
            fps = self._arm.GetCanFps()
            _ok, _fail, fail_rate = self._arm.GetCanTxStats()
            tx_tag = "OK" if fail_rate < 0.01 else f"WARN({fail_rate:.1%})"
            print(f"  [{mode}] 关节(°): [{', '.join(degs)}]  CAN: {fps:.0f}fps  TX: {tx_tag}")
        except Exception:
            print(f"  [{mode}] 关节(°): [{', '.join(degs)}]")

        if self._target_pose is not None:
            p = self._target_pose
            print(
                f"  末端目标: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m  "
                f"({p.rx:.2f}, {p.ry:.2f}, {p.rz:.2f}) rad"
            )


def dump_input(joy: LinuxJoystick, profile: ControllerProfile) -> None:
    print("\n" + "=" * 60)
    print("  手柄输入调试模式")
    print("=" * 60)
    print(f"设备: {joy.device}")
    print(f"Profile: {profile.display_name} [{profile.profile_id}]")
    print("按 Ctrl+C 退出，移动摇杆或按键查看原始索引变化。")
    last_axes = [None] * len(joy.axes)
    last_buttons = [None] * len(joy.buttons)
    try:
        while joy.connected:
            changed = False
            for idx, value in enumerate(joy.axes):
                prev = last_axes[idx]
                if prev is None or abs(value - prev) >= 0.05:
                    print(f"axis[{idx}] = {value:+.3f}")
                    last_axes[idx] = value
                    changed = True
            for idx, value in enumerate(joy.buttons):
                prev = last_buttons[idx]
                if prev is None or value != prev:
                    print(f"button[{idx}] = {value}")
                    last_buttons[idx] = value
                    changed = True
            if not changed:
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
