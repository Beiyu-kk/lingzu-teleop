"""
启动方式:
  python scripts/run_actions.py \
    --can can0 \
    --input recordings/test.jsonl \
    --rate 20

回放普通 JSON 动作数组:
  python scripts/run_actions.py \
    --can can0 \
    --input actions.json \
    --rate 20

如果确认动作已经安全，可以关闭限幅:
  python scripts/run_actions.py \
    --can can0 \
    --input actions.json \
    --rate 20 \
    --no-clip

启动前请确认:
  1. 已在当前环境执行 pip install -e .
  2. CAN 接口已激活，例如:
     sudo ip link set can0 up type can bitrate 1000000
     sudo ip link set can0 txqueuelen 128
  3. 机械臂已上电
"""
from __future__ import annotations

import argparse
import logging

from lingzu_teleop.config import RobotConnectionConfig
from lingzu_teleop.recording.action_runner import ActionRunner
from lingzu_teleop.recording.trajectory import TrajectoryRecorder
from lingzu_teleop.robot import ELA3Robot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="回放 JSON/JSONL 关节动作到 EL-A3")
    parser.add_argument("--can", default="can0", help="CAN 接口名 (默认: can0)")
    parser.add_argument("--backend", default="socketcan", choices=["socketcan", "slcan"], help="CAN 后端 (默认: socketcan)")
    parser.add_argument("--serial-port", default=None, help="SLCAN 串口路径")
    parser.add_argument("--input", required=True, help="动作 JSON/JSONL 文件")
    parser.add_argument("--rate", type=float, default=20.0, help="回放频率 Hz (默认: 20)")
    parser.add_argument("--no-clip", action="store_true", help="关闭动作限幅")
    parser.add_argument("--debug", action="store_true", help="调试模式")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(name)s][%(levelname)s] %(message)s",
    )
    robot = ELA3Robot(
        RobotConnectionConfig(
            can_name=args.can,
            backend=args.backend,
            serial_port=args.serial_port,
        )
    )
    try:
        robot.connect()
        actions = TrajectoryRecorder.actions_from_json_or_jsonl(args.input)
        count = ActionRunner(
            robot,
            rate_hz=args.rate,
            clip_actions=not args.no_clip,
            source="replay",
        ).run_actions(actions)
        print(f"已执行 {count} 条动作")
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
