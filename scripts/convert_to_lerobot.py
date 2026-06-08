"""
启动方式:
  python scripts/convert_to_lerobot.py \
    --input data/success/2026-06-06/20260606_143012 \
    --output lerobot_datasets/pick_place

批量转换 success 目录下的所有 episode:
  python scripts/convert_to_lerobot.py \
    --input data \
    --output lerobot_datasets/pick_place \
    --format both \
    --fps 15 \
    --overwrite

说明:
  输入是 collect_xbox_dataset.py 生成的 droid_style_jsonl_v1 episode。
  --format 2.1 输出 LeRobot v2.1 风格目录:
    meta/info.json
    meta/tasks.jsonl
    meta/episodes.jsonl
    meta/stats.json
    data/chunk-000/episode_000000.parquet
    videos/chunk-000/observation.images.<camera>/episode_000000.mp4

  --format 3.0 输出 LeRobot v3.0 风格目录:
    meta/info.json
    meta/tasks.jsonl
    meta/episodes/chunk-000/file-000.parquet
    meta/episodes_stats/chunk-000/file-000.parquet
    data/chunk-000/file-000.parquet
    videos/observation.images.<camera>/chunk-000/file-000.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lingzu_teleop.lerobot_convert import convert_droid_style_to_lerobot_v21, convert_droid_style_to_lerobot_v30


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将本项目采集的 droid_style_jsonl_v1 数据转换为 LeRobot 2.1/3.0 风格数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="输入 episode 目录或 data 根目录")
    parser.add_argument("--output", required=True, help="输出 LeRobot 数据集目录")
    parser.add_argument("--format", choices=["2.1", "3.0", "both"], default="both", help="导出格式 (默认: both)")
    parser.add_argument("--fps", type=float, default=15.0, help="导出视频和时间轴 FPS (默认: 15)")
    parser.add_argument("--include-failure", action="store_true", help="同时转换 failure episode")
    parser.add_argument("--overwrite", action="store_true", help="输出目录存在时先删除再生成")
    parser.add_argument(
        "--missing-action",
        choices=["state", "zeros", "skip"],
        default="state",
        help="样本 action 缺失时的处理方式 (默认: state)",
    )
    parser.add_argument("--robot-type", default="lingzu_ela3", help="写入 meta/info.json 的 robot_type")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summaries = []
    output_root = Path(args.output)
    if args.format in {"2.1", "both"}:
        output = output_root / "v2.1" if args.format == "both" else output_root
        summaries.append(
            convert_droid_style_to_lerobot_v21(
                args.input,
                output,
                fps=args.fps,
                include_failure=args.include_failure,
                overwrite=args.overwrite,
                missing_action=args.missing_action,
                robot_type=args.robot_type,
            )
        )
    if args.format in {"3.0", "both"}:
        output = output_root / "v3.0" if args.format == "both" else output_root
        summaries.append(
            convert_droid_style_to_lerobot_v30(
                args.input,
                output,
                fps=args.fps,
                include_failure=args.include_failure,
                overwrite=args.overwrite,
                missing_action=args.missing_action,
                robot_type=args.robot_type,
            )
        )

    summary = {"exports": summaries} if args.format == "both" else summaries[0]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
