from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from lingzu_teleop.lerobot_adapter import infer_lerobot_features

MissingActionPolicy = Literal["state", "zeros", "skip"]


@dataclass
class DroidStyleEpisode:
    path: Path
    metadata: dict[str, Any]
    samples: list[dict[str, Any]]

    @property
    def task(self) -> str:
        task = self.metadata.get("current_task") or self.metadata.get("task")
        return str(task or "unknown")

    @property
    def success(self) -> bool:
        return bool(self.metadata.get("success", False))


def find_droid_style_episodes(input_dir: str | Path, *, include_failure: bool = False) -> list[Path]:
    root = Path(input_dir)
    if (root / "trajectory.jsonl").is_file():
        return [root]

    episodes: list[Path] = []
    for trajectory_path in root.rglob("trajectory.jsonl"):
        episode_dir = trajectory_path.parent
        if not include_failure and "failure" in episode_dir.parts:
            continue
        episodes.append(episode_dir)
    return sorted(set(episodes))


def load_droid_style_episode(episode_dir: str | Path) -> DroidStyleEpisode:
    episode_dir = Path(episode_dir)
    trajectory_path = episode_dir / "trajectory.jsonl"
    metadata_path = episode_dir / "metadata.json"
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory.jsonl not found: {episode_dir}")

    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata.update(json.loads(metadata_path.read_text(encoding="utf-8")))

    samples: list[dict[str, Any]] = []
    with trajectory_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            item_type = payload.get("type")
            if item_type == "trajectory_header":
                metadata.update(payload.get("metadata") or {})
                metadata.setdefault("episode_name", payload.get("episode_name"))
            elif item_type == "trajectory_footer":
                metadata.update(
                    {
                        "success": bool(payload.get("success", False)),
                        "failure": bool(payload.get("failure", False)),
                        "num_samples": int(payload.get("num_samples", 0)),
                        "duration_s": float(payload.get("duration_s", 0.0)),
                    }
                )
            elif item_type == "sample":
                samples.append(payload)

    metadata.setdefault("episode_name", episode_dir.name)
    return DroidStyleEpisode(path=episode_dir, metadata=metadata, samples=samples)


def convert_droid_style_to_lerobot_v21(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 15.0,
    include_failure: bool = False,
    overwrite: bool = False,
    missing_action: MissingActionPolicy = "state",
    robot_type: str = "lingzu_ela3",
) -> dict[str, Any]:
    """Convert local droid_style_jsonl_v1 episodes to LeRobot v2.1-style files."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    episode_dirs = find_droid_style_episodes(input_dir, include_failure=include_failure)
    if not episode_dirs:
        raise RuntimeError(f"No droid_style_jsonl_v1 episodes found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = output_dir / "meta"
    data_chunk_dir = output_dir / "data" / "chunk-000"
    videos_chunk_dir = output_dir / "videos" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    videos_chunk_dir.mkdir(parents=True, exist_ok=True)

    tasks: dict[str, int] = {}
    episodes_meta: list[dict[str, Any]] = []
    episodes_stats: list[dict[str, Any]] = []
    global_state_rows: list[list[float]] = []
    global_action_rows: list[list[float]] = []
    camera_shapes: dict[str, tuple[int, int, int]] = {}
    total_frames = 0

    for episode_dir in episode_dirs:
        episode_index = len(episodes_meta)
        episode = load_droid_style_episode(episode_dir)
        task_index = tasks.setdefault(episode.task, len(tasks))
        rows: list[dict[str, Any]] = []
        image_paths_by_camera: dict[str, list[Path]] = {}
        episode_state_rows: list[list[float]] = []
        episode_action_rows: list[list[float]] = []

        for sample in episode.samples:
            observation = dict(sample.get("observation") or {})
            state = observation.get("joint_positions")
            if state is None:
                state = observation.get("state")
            if state is None:
                continue
            state = [float(value) for value in state]

            action_obj = sample.get("action")
            action_values = None
            if isinstance(action_obj, dict):
                action_values = action_obj.get("action")
            elif action_obj is not None:
                action_values = action_obj
            if action_values is None:
                if missing_action == "skip":
                    continue
                if missing_action == "zeros":
                    action_values = [0.0] * len(state)
                else:
                    action_values = list(state)
            action = [float(value) for value in action_values]

            frame_index = len(rows)
            timestamp = float(sample.get("elapsed_s", frame_index / fps))
            rows.append(
                {
                    "observation.state": state,
                    "action": action,
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "task_index": task_index,
                    "index": total_frames + frame_index,
                    "next.done": False,
                }
            )
            episode_state_rows.append(state)
            episode_action_rows.append(action)

            camera_refs = dict(sample.get("camera_refs") or {})
            for camera_name, rel_path in camera_refs.items():
                image_path = episode.path / rel_path
                if not image_path.is_file():
                    continue
                image_paths_by_camera.setdefault(camera_name, []).append(image_path)
                if camera_name not in camera_shapes:
                    camera_shapes[camera_name] = _read_image_shape(image_path)

        if not rows:
            continue
        rows[-1]["next.done"] = True

        parquet_path = data_chunk_dir / f"episode_{episode_index:06d}.parquet"
        _write_parquet(rows, parquet_path)
        for camera_name, image_paths in image_paths_by_camera.items():
            video_key = f"observation.images.{camera_name}"
            video_dir = videos_chunk_dir / video_key
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"episode_{episode_index:06d}.mp4"
            _write_video(image_paths, video_path, fps=fps)

        episode_length = len(rows)
        episodes_meta.append(
            {
                "episode_index": episode_index,
                "tasks": [task_index],
                "length": episode_length,
                "dataset_from_index": total_frames,
                "dataset_to_index": total_frames + episode_length,
                "source_episode_path": episode.path.as_posix(),
                "success": episode.success,
            }
        )
        episodes_stats.append(
            {
                "episode_index": episode_index,
                "stats": _stats_payload(
                    {
                        "observation.state": episode_state_rows,
                        "action": episode_action_rows,
                    }
                ),
            }
        )
        global_state_rows.extend(episode_state_rows)
        global_action_rows.extend(episode_action_rows)
        total_frames += episode_length

    if total_frames == 0:
        raise RuntimeError("No valid samples were converted")

    _write_jsonl(meta_dir / "tasks.jsonl", ({"task_index": idx, "task": task} for task, idx in tasks.items()))
    _write_jsonl(meta_dir / "episodes.jsonl", episodes_meta)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", episodes_stats)

    state_dim = len(global_state_rows[0])
    action_dim = len(global_action_rows[0])
    features = infer_lerobot_features(
        state_dim=state_dim,
        action_dim=action_dim,
        camera_shapes={f"{name}": shape for name, shape in camera_shapes.items()},
    )
    features["index"] = {"dtype": "int64", "shape": (1,)}
    features["next.done"] = {"dtype": "bool", "shape": (1,)}
    info = {
        "codebase_version": "v2.1",
        "format_note": "LeRobot v2.1-style episode parquet/video export",
        "robot_type": robot_type,
        "fps": float(fps),
        "features": _jsonable_features(features),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "total_episodes": len(episodes_meta),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
    }
    (meta_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = _stats_payload(
        {
            "observation.state": global_state_rows,
            "action": global_action_rows,
        }
    )
    (meta_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": output_dir.as_posix(),
        "format": "2.1",
        "episodes": len(episodes_meta),
        "frames": total_frames,
        "tasks": len(tasks),
        "cameras": sorted(camera_shapes.keys()),
    }


def convert_droid_style_to_lerobot_v30(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    fps: float = 15.0,
    include_failure: bool = False,
    overwrite: bool = False,
    missing_action: MissingActionPolicy = "state",
    robot_type: str = "lingzu_ela3",
) -> dict[str, Any]:
    """Convert local droid_style_jsonl_v1 episodes to LeRobot v3.0-style files."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)

    episode_dirs = find_droid_style_episodes(input_dir, include_failure=include_failure)
    if not episode_dirs:
        raise RuntimeError(f"No droid_style_jsonl_v1 episodes found in {input_dir}")

    meta_dir = output_dir / "meta"
    data_chunk_dir = output_dir / "data" / "chunk-000"
    episodes_chunk_dir = meta_dir / "episodes" / "chunk-000"
    episodes_stats_chunk_dir = meta_dir / "episodes_stats" / "chunk-000"
    for directory in (meta_dir, data_chunk_dir, episodes_chunk_dir, episodes_stats_chunk_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tasks: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    episodes_meta: list[dict[str, Any]] = []
    episodes_stats: list[dict[str, Any]] = []
    global_state_rows: list[list[float]] = []
    global_action_rows: list[list[float]] = []
    camera_shapes: dict[str, tuple[int, int, int]] = {}
    image_paths_by_camera: dict[str, list[Path]] = {}

    for episode_dir in episode_dirs:
        episode_index = len(episodes_meta)
        episode = load_droid_style_episode(episode_dir)
        task_index = tasks.setdefault(episode.task, len(tasks))
        episode_start = len(rows)
        episode_state_rows: list[list[float]] = []
        episode_action_rows: list[list[float]] = []

        for sample in episode.samples:
            observation = dict(sample.get("observation") or {})
            state = observation.get("joint_positions")
            if state is None:
                state = observation.get("state")
            if state is None:
                continue
            state = [float(value) for value in state]

            action_obj = sample.get("action")
            action_values = None
            if isinstance(action_obj, dict):
                action_values = action_obj.get("action")
            elif action_obj is not None:
                action_values = action_obj
            if action_values is None:
                if missing_action == "skip":
                    continue
                if missing_action == "zeros":
                    action_values = [0.0] * len(state)
                else:
                    action_values = list(state)
            action = [float(value) for value in action_values]

            frame_index = len(rows) - episode_start
            timestamp = float(sample.get("elapsed_s", frame_index / fps))
            rows.append(
                {
                    "observation.state": state,
                    "action": action,
                    "timestamp": timestamp,
                    "frame_index": frame_index,
                    "episode_index": episode_index,
                    "task_index": task_index,
                    "index": len(rows),
                    "next.done": False,
                }
            )
            episode_state_rows.append(state)
            episode_action_rows.append(action)

            camera_refs = dict(sample.get("camera_refs") or {})
            for camera_name, rel_path in camera_refs.items():
                image_path = episode.path / rel_path
                if not image_path.is_file():
                    continue
                image_paths_by_camera.setdefault(camera_name, []).append(image_path)
                if camera_name not in camera_shapes:
                    camera_shapes[camera_name] = _read_image_shape(image_path)

        episode_length = len(rows) - episode_start
        if episode_length == 0:
            continue
        rows[-1]["next.done"] = True
        episodes_meta.append(
            {
                "episode_index": episode_index,
                "tasks": [task_index],
                "length": episode_length,
                "dataset_from_index": episode_start,
                "dataset_to_index": episode_start + episode_length,
                "data_file": "data/chunk-000/file-000.parquet",
                "video_from_index": episode_start,
                "video_to_index": episode_start + episode_length,
                "source_episode_path": episode.path.as_posix(),
                "success": episode.success,
            }
        )
        episodes_stats.append(
            {
                "episode_index": episode_index,
                **_flatten_stats(
                    _stats_payload(
                        {
                            "observation.state": episode_state_rows,
                            "action": episode_action_rows,
                        }
                    )
                ),
            }
        )
        global_state_rows.extend(episode_state_rows)
        global_action_rows.extend(episode_action_rows)

    if not rows:
        raise RuntimeError("No valid samples were converted")

    _write_parquet(rows, data_chunk_dir / "file-000.parquet")
    for camera_name, image_paths in image_paths_by_camera.items():
        video_key = f"observation.images.{camera_name}"
        video_dir = output_dir / "videos" / video_key / "chunk-000"
        video_dir.mkdir(parents=True, exist_ok=True)
        _write_video(image_paths, video_dir / "file-000.mp4", fps=fps)

    _write_parquet(episodes_meta, episodes_chunk_dir / "file-000.parquet")
    _write_parquet(episodes_stats, episodes_stats_chunk_dir / "file-000.parquet")
    _write_jsonl(meta_dir / "tasks.jsonl", ({"task_index": idx, "task": task} for task, idx in tasks.items()))

    state_dim = len(global_state_rows[0])
    action_dim = len(global_action_rows[0])
    features = infer_lerobot_features(
        state_dim=state_dim,
        action_dim=action_dim,
        camera_shapes={f"{name}": shape for name, shape in camera_shapes.items()},
    )
    features["index"] = {"dtype": "int64", "shape": (1,)}
    features["next.done"] = {"dtype": "bool", "shape": (1,)}
    info = {
        "codebase_version": "v3.0",
        "format_note": "LeRobot v3.0-style file-based parquet/video export",
        "robot_type": robot_type,
        "fps": float(fps),
        "features": _jsonable_features(features),
        "data_path": "data/chunk-{file_chunk:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{file_chunk:03d}/file-{file_index:03d}.mp4",
        "episodes_path": "meta/episodes/chunk-{file_chunk:03d}/file-{file_index:03d}.parquet",
        "total_episodes": len(episodes_meta),
        "total_frames": len(rows),
        "total_tasks": len(tasks),
    }
    (meta_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = _stats_payload(
        {
            "observation.state": global_state_rows,
            "action": global_action_rows,
        }
    )
    (meta_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": output_dir.as_posix(),
        "format": "3.0",
        "episodes": len(episodes_meta),
        "frames": len(rows),
        "tasks": len(tasks),
        "cameras": sorted(camera_shapes.keys()),
    }


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("转换 LeRobot parquet 需要 pandas，请先安装 pandas。") from exc
    try:
        pd.DataFrame(rows).to_parquet(path, engine="pyarrow", index=False)
    except ImportError as exc:
        raise RuntimeError("转换 LeRobot parquet 需要 pyarrow，请先安装 pyarrow。") from exc


def _write_video(image_paths: list[Path], video_path: Path, *, fps: float) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("转换 LeRobot 视频需要 imageio 和 imageio-ffmpeg。") from exc
    writer = imageio.get_writer(video_path, fps=fps, macro_block_size=1)
    try:
        for image_path in image_paths:
            writer.append_data(_read_image(image_path))
    finally:
        writer.close()


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        image = np.load(path)
    else:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise RuntimeError("读取 jpg/png 图像需要 imageio。") from exc
        image = iio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _read_image_shape(path: Path) -> tuple[int, int, int]:
    image = _read_image(path)
    if image.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape {image.shape}: {path}")
    return int(image.shape[0]), int(image.shape[1]), int(image.shape[2])


def _stats_payload(values_by_key: dict[str, list[list[float]]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, values in values_by_key.items():
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        payload[key] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "count": int(arr.shape[0]),
        }
    return payload


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _jsonable_features(features: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        key: {field: list(value) if isinstance(value, tuple) else value for field, value in spec.items()}
        for key, spec in features.items()
    }


def _flatten_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {f"{key}.{stat_name}": stat_value for key, stat_values in stats.items() for stat_name, stat_value in stat_values.items()}
