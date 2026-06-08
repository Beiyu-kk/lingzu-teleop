"""
启动方式:
  python scripts/record_trajectory.py \
    --can can0 \
    --duration 10 \
    --rate 20 \
    --output recordings/test.jsonl

如果只想先记录当前关节状态，并把当前状态也作为动作保存:
  python scripts/record_trajectory.py \
    --can can0 \
    --duration 10 \
    --rate 20 \
    --action-from-state \
    --output recordings/test.jsonl

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
import time
from pathlib import Path

from lingzu_teleop.config import RobotConnectionConfig
from lingzu_teleop.recording.trajectory import TrajectoryRecorder
from lingzu_teleop.robot import ELA3Robot
from lingzu_teleop.types import RecordedAction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="记录 EL-A3 状态和动作到 JSONL")
    parser.add_argument("--can", default="can0", help="CAN 接口名 (默认: can0)")
    parser.add_argument("--backend", default="socketcan", choices=["socketcan", "slcan"], help="CAN 后端 (默认: socketcan)")
    parser.add_argument("--serial-port", default=None, help="SLCAN 串口路径")
    parser.add_argument("--duration", type=float, required=True, help="记录时长，单位秒")
    parser.add_argument("--rate", type=float, default=20.0, help="采样频率 Hz (默认: 20)")
    parser.add_argument("--output", required=True, help="输出 JSONL 路径")
    parser.add_argument(
        "--action-from-state",
        action="store_true",
        help="没有命令动作时，使用当前关节状态作为 action",
    )
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
    recorder = TrajectoryRecorder(sample_rate_hz=args.rate)
    try:
        robot.connect()
        recorder.start(metadata={"can_name": args.can, "mode": "low_dim"})
        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            observation = robot.get_observation()
            action = robot.last_action
            if action is None and args.action_from_state:
                action = RecordedAction(
                    timestamp=time.monotonic(),
                    action=observation.joint_positions,
                    source="state_snapshot",
                )
            recorder.add_sample(observation, action=action)
            time.sleep(max(0.001, 1.0 / args.rate / 2.0))
        trajectory = recorder.stop()
        if trajectory is None:
            raise RuntimeError("没有记录到轨迹")
        output = recorder.save_jsonl(Path(args.output), trajectory)
        print(f"已保存 {trajectory.num_samples} 条样本到 {output}")
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
