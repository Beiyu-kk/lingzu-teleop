"""
启动方式:
  python scripts/xbox_control.py --can can0 --js /dev/input/js0
  python scripts/xbox_control.py --can can1
  python scripts/xbox_control.py --js /dev/input/js1
  python scripts/xbox_control.py --profile auto
  python scripts/xbox_control.py --dump-input
  python scripts/xbox_control.py --no-zero-init

EL-A3 手柄遥操作控制映射:
  左摇杆 Y/X     -> 末端 X/Y 平移
  LT / RT        -> 末端 Z 下/上
  右摇杆 X/Y     -> Yaw / Roll 旋转
  LB / RB        -> Pitch 旋转
  A              -> 切换速度档位（5 档）
  B              -> 回 Home 位置
  X              -> 回零位
  Y              -> 切换零力矩模式（可手动拖动）
  D-pad 上       -> 夹爪小步打开（按住连续平滑打开）
  D-pad 下       -> 夹爪小步关闭（按住连续平滑关闭）
  Back           -> 急停
  Start          -> 退出程序

启动前请确认:
  1. CAN 接口已激活，例如:
     sudo ip link set can0 up type can bitrate 1000000
     sudo ip link set can0 txqueuelen 128
  2. 机械臂已上电
  3. 手柄已连接（USB 或蓝牙）
  4. 已在当前环境执行 pip install -e .
"""
from __future__ import annotations

import argparse
import logging
import signal
import threading

from lingzu_teleop.config import RobotConnectionConfig, XboxTeleopConfig
from lingzu_teleop.robot import ELA3Robot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EL-A3 手柄控制（纯 SDK 模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                          # 默认参数
  %(prog)s --can can1 --js /dev/input/js1
  %(prog)s --profile zikway_3537_1041
  %(prog)s --dump-input
  %(prog)s --max-lin-vel 0.10 --kp 60 --kd 3.5
  %(prog)s --list-profiles
""",
    )
    parser.add_argument("--can", default="can0", help="CAN 接口名 (默认: can0)")
    parser.add_argument("--backend", default="socketcan", choices=["socketcan", "slcan"], help="CAN 后端 (默认: socketcan)")
    parser.add_argument("--serial-port", default=None, help="SLCAN 串口路径")
    parser.add_argument("--js", default="/dev/input/js0", help="手柄设备路径 (默认: /dev/input/js0)")
    parser.add_argument("--profile", default="auto", help="控制器映射 profile (默认: auto)")
    parser.add_argument("--list-profiles", action="store_true", help="列出内置 profile 并退出")
    parser.add_argument("--dump-input", action="store_true", help="仅打印手柄原始输入并退出")
    parser.add_argument("--rate", type=float, default=100.0, help="输入处理频率 Hz (默认: 100)")
    parser.add_argument("--max-lin-vel", type=float, default=0.15, help="最大线速度 m/s (默认: 0.15)")
    parser.add_argument("--max-ang-vel", type=float, default=1.5, help="最大角速度 rad/s (默认: 1.5)")
    parser.add_argument("--kp", type=float, default=80.0, help="位置增益 Kp (默认: 80)")
    parser.add_argument("--kd", type=float, default=4.0, help="速度增益 Kd (默认: 4)")
    parser.add_argument("--deadzone", type=float, default=None, help="摇杆死区 (默认: 使用 profile 推荐值)")
    parser.add_argument("--no-zero-init", action="store_true", help="启动后不自动移动到零位")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from lingzu_teleop.sdk.controller_profiles import PROFILES, detect_controller
    from lingzu_teleop.sdk.joystick import LinuxJoystick

    from lingzu_teleop.teleop.xbox import XboxCartesianTeleop, dump_input

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

    detection = detect_controller(args.js, requested_profile=args.profile)
    deadzone = args.deadzone if args.deadzone is not None else detection.profile.default_deadzone
    logging.getLogger(__name__).info(
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
    logging.getLogger(__name__).info("手柄已连接: %s", args.js)

    if args.dump_input:
        try:
            dump_input(joy, detection.profile)
        finally:
            joy.disconnect()
        return 0

    robot = ELA3Robot(
        RobotConnectionConfig(
            can_name=args.can,
            backend=args.backend,
            serial_port=args.serial_port,
            default_kp=args.kp,
            default_kd=args.kd,
        )
    )

    shutdown = threading.Event()

    def on_signal(_sig, _frame):
        shutdown.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    try:
        robot.connect()
    except Exception:
        joy.disconnect()
        print(f"\nCAN 接口 {args.can} 连接失败")
        print("请确认:")
        if args.backend == "slcan":
            print("  1. SLCAN 串口路径正确，例如: --serial-port /dev/ttyACM0")
            print("  2. 当前用户有串口权限，例如加入 dialout 组")
            print("  3. 转接器已连接并且波特率配置正确")
        else:
            print(f"  sudo ip link set {args.can} up type can bitrate 1000000")
            print(f"  sudo ip link set {args.can} txqueuelen 128")
        return 1

    try:
        teleop = XboxCartesianTeleop(
            robot=robot,
            joystick=joy,
            profile=detection.profile,
            config=XboxTeleopConfig(
                device=args.js,
                profile=args.profile,
                update_rate_hz=args.rate,
                max_linear_velocity=args.max_lin_vel,
                max_angular_velocity=args.max_ang_vel,
                deadzone=deadzone,
            ),
        )
        thread = threading.Thread(
            target=teleop.start,
            kwargs={"initialize_to_zero": not args.no_zero_init},
            daemon=True,
        )
        thread.start()
        while not shutdown.is_set() and not teleop.exit_requested:
            shutdown.wait(timeout=0.5)
        teleop.stop()
        thread.join(timeout=2.0)
    finally:
        print("\n正在清理...")
        try:
            robot.zero_torque(False)
        except Exception:
            pass
        robot.disconnect()
        joy.disconnect()
        print("已退出")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
