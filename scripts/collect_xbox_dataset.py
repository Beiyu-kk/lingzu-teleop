"""
默认启动方式（连接腕部 D405 和主视角 D455）:
  python scripts/collect_xbox_dataset.py \
    --can can0 \
    --js /dev/input/js0 \
    --wrist-serial <D405_SERIAL> \
    --front-serial <D455_SERIAL> \
    --task "把物体放入盒子" \
    --output-dir data

无相机测试启动方式（只采集机械臂状态和动作）:
  python scripts/collect_xbox_dataset.py \
    --can can0 \
    --js /dev/input/js0 \
    --no-cameras \
    --task "无相机遥操作测试" \
    --output-dir data

采集流程参考 DROID:
  1. 连接 EL-A3、Xbox 手柄、腕部 RealSense D405、主视角 RealSense D455
  2. 启动 Xbox 遥操作，等待机械臂完成初始化
  3. 进入待机状态，按键盘 s + Enter 开始一条 episode
  4. 创建 failure/YYYY-MM-DD/<episode>/ 目录
  5. 按固定频率采集 observation、action，以及可选的 wrist/front 图像
  6. 按键盘 e + Enter 结束当前 episode
  7. 输入 y/n 将 episode 标记为 success/failure
  8. 回到待机状态，可继续采集下一条；输入 q + Enter 退出整个会话

键盘命令:
  s / start / 开始    开始采集一条 episode
  e / end / 结束      结束当前 episode
  y / yes / 成功      将刚结束的 episode 标记为成功
  n / no / 失败       将刚结束的 episode 标记为失败
  q / quit / 退出     退出整个采集会话

如果启用相机，默认会显示实时预览窗口。预览窗口也支持直接按 s/e/y/n/q。

启动前请确认:
  1. 已在当前环境执行 pip install -e .
  2. 默认会连接 RealSense；如果没有相机，请加 --no-cameras
  3. RealSense 采集需要安装 pyrealsense2
  4. 实时相机预览需要安装 opencv-python
  5. 图像保存为 jpg/png 时需要安装 Pillow
  6. CAN 接口已激活，例如:
     sudo ip link set can0 up type can bitrate 1000000
     sudo ip link set can0 txqueuelen 128
  7. 机械臂已上电，D405/D455 已连接
"""
from __future__ import annotations

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from typing import Any

import numpy as np

from lingzu_teleop.camera.base import NullCameraProvider
from lingzu_teleop.config import RobotConnectionConfig, XboxTeleopConfig
from lingzu_teleop.recording.droid_style import DroidStyleEpisodeWriter
from lingzu_teleop.robot import ELA3Robot
from lingzu_teleop.types import RecordedAction


COMMAND_ALIASES = {
    "s": "start",
    "start": "start",
    "开始": "start",
    "e": "end",
    "end": "end",
    "stop": "end",
    "结束": "end",
    "y": "success",
    "yes": "success",
    "1": "success",
    "true": "success",
    "success": "success",
    "成功": "success",
    "n": "failure",
    "no": "failure",
    "0": "failure",
    "false": "failure",
    "failure": "failure",
    "fail": "failure",
    "失败": "failure",
    "q": "quit",
    "quit": "quit",
    "exit": "quit",
    "退出": "quit",
}


def normalize_command(raw: str) -> str | None:
    command = raw.strip().lower()
    if not command:
        return None
    return COMMAND_ALIASES.get(command, command)


class KeyboardCommandReader:
    def __init__(self, shutdown: threading.Event):
        self._shutdown = shutdown
        self._commands: queue.Queue[str] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="keyboard_command_reader")
        self._thread.start()

    def drain(self) -> list[str]:
        commands: list[str] = []
        while True:
            try:
                commands.append(self._commands.get_nowait())
            except queue.Empty:
                return commands

    def _read_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                self._commands.put("eof")
                return
            if line == "":
                self._commands.put("eof")
                return
            command = normalize_command(line)
            if command is not None:
                self._commands.put(command)


class CameraPreview:
    def __init__(self, *, enabled: bool, window_name: str, scale: float):
        self.enabled = enabled
        self.window_name = window_name
        self.scale = max(0.1, float(scale))
        self._cv2 = None
        if not self.enabled:
            return
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "实时相机预览需要 opencv-python。请执行 `pip install opencv-python`，"
                "或启动时加 `--no-preview`。"
            ) from exc
        self._cv2 = cv2

    def show(self, frames: dict[str, Any], status_text: str) -> str | None:
        if not self.enabled or not frames or self._cv2 is None:
            return None
        try:
            canvas = self._compose(frames, status_text)
            if canvas is None:
                return None
            self._cv2.imshow(self.window_name, canvas)
            key = self._cv2.waitKey(1)
        except Exception as exc:
            print(f"\n相机预览已关闭: {exc}")
            self.enabled = False
            return None
        if key < 0:
            return None
        key &= 0xFF
        if key == 255:
            return None
        return normalize_command(chr(key))

    def close(self) -> None:
        if self._cv2 is not None:
            try:
                self._cv2.destroyWindow(self.window_name)
            except Exception:
                pass

    def _compose(self, frames: dict[str, Any], status_text: str):
        tiles = []
        for camera_name, frame in sorted(frames.items()):
            image = self._to_bgr(frame.image)
            if image is None:
                continue
            if self.scale != 1.0:
                width = max(1, int(image.shape[1] * self.scale))
                height = max(1, int(image.shape[0] * self.scale))
                image = self._cv2.resize(image, (width, height))
            self._cv2.putText(
                image,
                camera_name,
                (10, 28),
                self._cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                self._cv2.LINE_AA,
            )
            tiles.append(image)
        if not tiles:
            return None

        max_height = max(tile.shape[0] for tile in tiles)
        padded = []
        for tile in tiles:
            if tile.shape[0] < max_height:
                pad = np.zeros((max_height - tile.shape[0], tile.shape[1], 3), dtype=tile.dtype)
                tile = np.vstack([tile, pad])
            padded.append(tile)
        canvas = np.hstack(padded)
        self._cv2.putText(
            canvas,
            status_text,
            (10, max(24, canvas.shape[0] - 16)),
            self._cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            self._cv2.LINE_AA,
        )
        return canvas

    def _to_bgr(self, image: Any):
        array = np.asarray(image)
        if array.size == 0:
            return None
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            return self._cv2.cvtColor(array, self._cv2.COLOR_GRAY2BGR)
        if array.ndim != 3:
            return None
        if array.shape[2] == 3:
            return self._cv2.cvtColor(array, self._cv2.COLOR_RGB2BGR)
        if array.shape[2] == 4:
            return self._cv2.cvtColor(array, self._cv2.COLOR_RGBA2BGR)
        return np.ascontiguousarray(array[:, :, :3])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 Xbox 遥操作采集 DROID 风格 episode 数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--can", default="can0", help="CAN 接口名 (默认: can0)")
    parser.add_argument("--backend", default="socketcan", choices=["socketcan", "slcan"], help="CAN 后端 (默认: socketcan)")
    parser.add_argument("--serial-port", default=None, help="SLCAN 串口路径")
    parser.add_argument("--js", default="/dev/input/js0", help="手柄设备路径 (默认: /dev/input/js0)")
    parser.add_argument("--profile", default="auto", help="控制器映射 profile (默认: auto)")
    parser.add_argument("--list-profiles", action="store_true", help="列出内置 profile 并退出")
    parser.add_argument("--output-dir", default="data", help="数据输出根目录 (默认: data)")
    parser.add_argument("--episode-name", default=None, help="手动指定 episode 名称；连续采集时会自动追加序号")
    parser.add_argument("--task", default="", help="任务描述，例如：把物体放入盒子")
    parser.add_argument("--operator", default="", help="采集人员或设备操作者")
    parser.add_argument("--scene-id", default="", help="场景编号或备注")
    parser.add_argument("--rate", type=float, default=15.0, help="数据采样频率 Hz (默认: 15，参考 DROID)")
    parser.add_argument("--duration", type=float, default=None, help="单条 episode 最大采集时长，单位秒；不填则按键盘 e 结束")
    parser.add_argument("--teleop-rate", type=float, default=100.0, help="手柄控制频率 Hz (默认: 100)")
    parser.add_argument("--max-lin-vel", type=float, default=0.15, help="最大线速度 m/s (默认: 0.15)")
    parser.add_argument("--max-ang-vel", type=float, default=1.5, help="最大角速度 rad/s (默认: 1.5)")
    parser.add_argument("--kp", type=float, default=80.0, help="位置增益 Kp (默认: 80)")
    parser.add_argument("--kd", type=float, default=4.0, help="速度增益 Kd (默认: 4)")
    parser.add_argument("--deadzone", type=float, default=None, help="摇杆死区 (默认: 使用 profile 推荐值)")
    parser.add_argument("--no-zero-init", action="store_true", help="启动后不自动移动到零位")
    parser.add_argument("--init-timeout", type=float, default=15.0, help="等待遥操作初始化完成的最长时间，单位秒")
    parser.add_argument("--wrist-serial", default=None, help="腕部 RealSense D405 序列号")
    parser.add_argument("--front-serial", default=None, help="主视角 RealSense D455 序列号")
    parser.add_argument("--camera-width", type=int, default=640, help="相机 RGB 宽度 (默认: 640)")
    parser.add_argument("--camera-height", type=int, default=480, help="相机 RGB 高度 (默认: 480)")
    parser.add_argument("--camera-fps", type=int, default=30, help="相机帧率 (默认: 30)")
    parser.add_argument("--image-format", default="jpg", choices=["jpg", "png", "npy"], help="图像保存格式")
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument("--with-cameras", dest="use_cameras", action="store_true", default=True, help="连接腕部和主视角相机 (默认)")
    camera_group.add_argument("--no-cameras", dest="use_cameras", action="store_false", help="不连接相机，仅采集低维状态和动作")
    parser.add_argument("--no-preview", dest="preview", action="store_false", default=True, help="启用相机时不显示实时预览窗口")
    parser.add_argument("--preview-rate", type=float, default=15.0, help="实时预览刷新频率 Hz (默认: 15)")
    parser.add_argument("--preview-scale", type=float, default=0.75, help="实时预览缩放比例 (默认: 0.75)")
    parser.add_argument("--preview-window", default="Lingzu Teleop Cameras", help="实时预览窗口名称")
    result_group = parser.add_mutually_exclusive_group()
    result_group.add_argument("--success", action="store_true", help="结束后直接标记为 success，不再询问")
    result_group.add_argument("--failure", action="store_true", help="结束后直接标记为 failure，不再询问")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    return parser


def build_camera_provider(args: argparse.Namespace):
    if not args.use_cameras:
        return NullCameraProvider(), {"cameras_enabled": False, "cameras": {}}

    from lingzu_teleop.camera import MultiCameraProvider, RealSenseProvider, RealSenseProviderConfig

    if not args.wrist_serial or not args.front_serial:
        logging.getLogger(__name__).warning(
            "建议同时提供 --wrist-serial 和 --front-serial，避免 RealSense 枚举顺序不稳定"
        )

    providers = {
        "wrist": RealSenseProvider(
            RealSenseProviderConfig(
                name="wrist",
                serial_number=args.wrist_serial,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
            )
        ),
        "front": RealSenseProvider(
            RealSenseProviderConfig(
                name="front",
                serial_number=args.front_serial,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
            )
        ),
    }
    metadata = {
        "cameras_enabled": True,
        "cameras": {
            "wrist": {
                "model": "Intel RealSense D405",
                "serial": args.wrist_serial or "",
                "role": "wrist",
            },
            "front": {
                "model": "Intel RealSense D455",
                "serial": args.front_serial or "",
                "role": "front",
            },
        }
    }
    return MultiCameraProvider(providers), metadata


def resolve_success(args: argparse.Namespace) -> bool | None:
    if args.success:
        return True
    if args.failure:
        return False
    return None


def episode_name_for(args: argparse.Namespace, index: int) -> str | None:
    if not args.episode_name:
        return None
    if index <= 1:
        return args.episode_name
    return f"{args.episode_name}_{index:03d}"


def print_interactive_help(*, use_cameras: bool, preview_enabled: bool) -> None:
    print("\n" + "=" * 56)
    print("     数据采集会话已就绪")
    print("=" * 56)
    print("  s + Enter    开始采集一条 episode")
    print("  e + Enter    结束当前 episode")
    print("  y + Enter    将刚结束的 episode 标记为 success")
    print("  n + Enter    将刚结束的 episode 标记为 failure")
    print("  q + Enter    退出整个采集会话")
    if use_cameras and preview_enabled:
        print("  预览窗口中也可以直接按 s / e / y / n / q")
    elif use_cameras:
        print("  相机已启用，实时预览已关闭")
    else:
        print("  当前为无相机测试模式")
    print("=" * 56)


def close_episode(
    writer: DroidStyleEpisodeWriter,
    *,
    success: bool,
    reason: str,
) -> None:
    output_dir = writer.close(
        success=success,
        metadata={"closed_reason": reason, "marked_success": success},
    )
    print(f"\n采集完成: {output_dir}")
    print(f"样本数: {writer.num_samples}")
    print(f"结果: {'success' if success else 'failure'}")


def action_for_sample(robot: ELA3Robot, teleop: Any) -> RecordedAction | None:
    if robot.last_action is not None:
        return robot.last_action
    current = teleop.current_action
    if current is None:
        return None
    return RecordedAction(
        timestamp=time.monotonic(),
        action=list(current),
        source="xbox_snapshot",
    )


def main() -> int:
    args = build_parser().parse_args()

    from lingzu_teleop.sdk.controller_profiles import PROFILES, detect_controller
    from lingzu_teleop.sdk.joystick import LinuxJoystick
    from lingzu_teleop.teleop.xbox import XboxCartesianTeleop

    if args.list_profiles:
        for profile_id, profile in PROFILES.items():
            print(f"{profile_id}: {profile.display_name} - {profile.description}")
        return 0
    if args.profile != "auto" and args.profile not in PROFILES:
        choices = ", ".join(["auto", *PROFILES.keys()])
        raise SystemExit(f"未知 profile '{args.profile}'，可选项: {choices}")

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(name)s][%(levelname)s] %(message)s",
    )
    logging.getLogger("lingzu_teleop.sdk").propagate = False
    logger = logging.getLogger(__name__)

    detection = detect_controller(args.js, requested_profile=args.profile)
    deadzone = args.deadzone if args.deadzone is not None else detection.profile.default_deadzone
    logger.info(
        "手柄检测: device=%s name=%s vid=%s pid=%s profile=%s source=%s",
        detection.resolved_device,
        detection.name or "unknown",
        detection.vendor or "unknown",
        detection.product or "unknown",
        detection.profile.profile_id,
        detection.source,
    )

    joy = LinuxJoystick(device=args.js)
    if not joy.connect():
        print(f"\n无法打开手柄 {args.js}")
        print("请确认:")
        print("  1. 手柄已连接 (蓝牙或 USB)")
        print("  2. 设备存在: ls /dev/input/js*")
        print("  3. 权限足够: sudo chmod 666 /dev/input/js0")
        print("  4. 驱动已加载: sudo modprobe joydev")
        return 1

    robot = ELA3Robot(
        RobotConnectionConfig(
            can_name=args.can,
            backend=args.backend,
            serial_port=args.serial_port,
            default_kp=args.kp,
            default_kd=args.kd,
        )
    )
    cameras, camera_metadata = build_camera_provider(args)
    teleop_thread: threading.Thread | None = None
    teleop = None
    shutdown = threading.Event()
    keyboard = KeyboardCommandReader(shutdown)
    preview: CameraPreview | None = None
    active_writer: DroidStyleEpisodeWriter | None = None
    pending_writer: DroidStyleEpisodeWriter | None = None

    def on_signal(_sig, _frame):
        shutdown.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        cameras.connect()
        preview = CameraPreview(
            enabled=args.use_cameras and args.preview,
            window_name=args.preview_window,
            scale=args.preview_scale,
        )
        robot.connect()
        teleop = XboxCartesianTeleop(
            robot=robot,
            joystick=joy,
            profile=detection.profile,
            config=XboxTeleopConfig(
                device=args.js,
                profile=args.profile,
                update_rate_hz=args.teleop_rate,
                max_linear_velocity=args.max_lin_vel,
                max_angular_velocity=args.max_ang_vel,
                deadzone=deadzone,
            ),
        )
        teleop_thread = threading.Thread(
            target=teleop.start,
            kwargs={"initialize_to_zero": not args.no_zero_init},
            daemon=True,
        )
        teleop_thread.start()
        if not teleop.wait_until_initialized(timeout=args.init_timeout):
            raise RuntimeError("遥操作初始化超时，请检查机械臂和 CAN 状态")

        base_metadata = {
            **camera_metadata,
            "collection_flow": "droid_style_xbox",
            "task": args.task,
            "current_task": args.task,
            "operator": args.operator,
            "scene_id": args.scene_id,
            "robot": {
                "name": "Lingzu EL-A3",
                "dof": 6,
                "gripper": "angle_controlled",
                "can_name": args.can,
                "backend": args.backend,
            },
            "controller": {
                "type": "xbox",
                "profile": detection.profile.profile_id,
                "device": detection.resolved_device,
            },
            "sample_rate_hz": args.rate,
            "teleop_rate_hz": args.teleop_rate,
            "control_mode": "interactive_keyboard_episode",
        }
        keyboard.start()
        print_interactive_help(use_cameras=args.use_cameras, preview_enabled=args.preview)

        period = 1.0 / args.rate
        preview_period = 1.0 / max(0.1, args.preview_rate)
        next_sample = 0.0
        next_preview = 0.0
        episode_started_at: float | None = None
        episode_index = 0
        last_frames = {}
        fixed_success = resolve_success(args)
        last_status_log = 0.0

        def start_episode() -> None:
            nonlocal active_writer, episode_started_at, next_sample, episode_index
            if active_writer is not None:
                print("\n当前已经在采集中，请先输入 e 结束当前 episode。")
                return
            if pending_writer is not None:
                print("\n上一条 episode 还没有标记结果，请先输入 y 或 n。")
                return
            episode_index += 1
            metadata = {
                **base_metadata,
                "episode_index_in_session": episode_index,
                "session_started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            active_writer = DroidStyleEpisodeWriter(
                args.output_dir,
                metadata=metadata,
                episode_name=episode_name_for(args, episode_index),
                image_format=args.image_format,
            )
            episode_started_at = time.monotonic()
            next_sample = episode_started_at
            print(f"\n开始采集 episode #{episode_index}: {active_writer.episode_dir}")
            print("输入 e + Enter 结束当前 episode。")

        def finish_active_episode(*, reason: str) -> None:
            nonlocal active_writer, pending_writer, episode_started_at
            if active_writer is None:
                print("\n当前没有正在采集的 episode。输入 s 开始采集。")
                return
            writer = active_writer
            active_writer = None
            episode_started_at = None
            if fixed_success is not None:
                close_episode(writer, success=fixed_success, reason=reason)
                return
            pending_writer = writer
            print("\n当前 episode 已停止写入。")
            print("请输入 y + Enter 标记为 success，或 n + Enter 标记为 failure。")

        def mark_pending(success: bool) -> None:
            nonlocal pending_writer
            if pending_writer is None:
                print("\n当前没有等待标记的 episode。")
                return
            writer = pending_writer
            pending_writer = None
            close_episode(writer, success=success, reason="user_mark")

        def handle_command(command: str | None) -> None:
            if command is None:
                return
            if command == "start":
                start_episode()
            elif command == "end":
                finish_active_episode(reason="user_stop")
            elif command == "success":
                mark_pending(True)
            elif command == "failure":
                mark_pending(False)
            elif command == "quit":
                shutdown.set()
            elif command == "eof":
                return
            else:
                print(f"\n未知命令: {command}。可用命令: s/e/y/n/q")

        while not shutdown.is_set() and not teleop.exit_requested:
            now = time.monotonic()

            for command in keyboard.drain():
                handle_command(command)

            if (
                active_writer is not None
                and args.duration is not None
                and episode_started_at is not None
                and (now - episode_started_at) >= args.duration
            ):
                finish_active_episode(reason="duration_reached")

            should_refresh_preview = (
                args.use_cameras
                and preview is not None
                and preview.enabled
                and now >= next_preview
            )
            should_sample = active_writer is not None and now >= next_sample
            frames = last_frames
            if should_refresh_preview or should_sample:
                frames = cameras.get_frames()
                last_frames = frames

            if should_sample and active_writer is not None:
                observation = robot.get_observation()
                action = action_for_sample(robot, teleop)
                active_writer.write_sample(
                    observation=observation,
                    action=action,
                    frames=frames,
                    controller_info={
                        "movement_enabled": action is not None,
                        "exit_requested": teleop.exit_requested,
                        "controller_profile": detection.profile.profile_id,
                    },
                )
                if active_writer.num_samples % max(1, int(args.rate)) == 0:
                    logger.info("episode #%d 已采集 %d 帧", episode_index, active_writer.num_samples)
                next_sample += period
                if next_sample < now - period:
                    next_sample = now + period

            if should_refresh_preview:
                if active_writer is not None:
                    status = f"REC #{episode_index} samples={active_writer.num_samples}  s/e/y/n/q"
                elif pending_writer is not None:
                    status = f"WAIT MARK samples={pending_writer.num_samples}  y/n/q"
                else:
                    status = "IDLE  s=start q=quit"
                command = preview.show(frames, status) if preview is not None else None
                handle_command(command)
                next_preview += preview_period
                if next_preview < now - preview_period:
                    next_preview = now + preview_period

            if active_writer is None and pending_writer is None and now - last_status_log >= 10.0:
                print("\n等待采集命令：输入 s + Enter 开始，q + Enter 退出。")
                last_status_log = now

            time.sleep(0.002)

    except KeyboardInterrupt:
        shutdown.set()
    except Exception:
        if active_writer is not None:
            output_dir = active_writer.close(success=False, metadata={"closed_reason": "exception"})
            print(f"\n采集异常，episode 已按 failure 保存: {output_dir}")
            active_writer = None
        if pending_writer is not None:
            output_dir = pending_writer.close(success=False, metadata={"closed_reason": "exception_before_mark"})
            print(f"\n采集异常，未标记 episode 已按 failure 保存: {output_dir}")
            pending_writer = None
        raise
    finally:
        shutdown.set()
        if active_writer is not None:
            output_dir = active_writer.close(success=False, metadata={"closed_reason": "session_shutdown"})
            print(f"\n会话退出，正在采集的 episode 已按 failure 保存: {output_dir}")
        if pending_writer is not None:
            output_dir = pending_writer.close(success=False, metadata={"closed_reason": "session_shutdown_before_mark"})
            print(f"\n会话退出，未标记的 episode 已按 failure 保存: {output_dir}")
        if preview is not None:
            preview.close()
        if teleop is not None:
            teleop.stop()
        if teleop_thread is not None:
            teleop_thread.join(timeout=2.0)
        try:
            robot.zero_torque(False)
        except Exception:
            pass
        robot.disconnect()
        cameras.disconnect()
        joy.disconnect()

    print("\n采集会话已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
