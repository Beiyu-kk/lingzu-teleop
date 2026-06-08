from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from lingzu_teleop.camera.base import CameraFrame
from lingzu_teleop.types import ArmObservation, RecordedAction


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class DroidStyleEpisodeWriter:
    """Stream a DROID-like episode as JSONL plus per-camera image files."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        episode_name: str | None = None,
        image_format: str = "jpg",
        initial_status: str = "failure",
    ):
        if initial_status not in {"success", "failure"}:
            raise ValueError("initial_status must be 'success' or 'failure'")
        image_format = image_format.lower().lstrip(".")
        if image_format == "jpeg":
            image_format = "jpg"
        if image_format not in {"jpg", "png", "npy"}:
            raise ValueError("image_format must be one of: jpg, png, npy")

        self.root_dir = Path(root_dir)
        self.status = initial_status
        self.date = time.strftime("%Y-%m-%d")
        self.episode_name = episode_name or time.strftime("%Y%m%d_%H%M%S")
        self.image_format = image_format
        self.metadata = dict(metadata or {})
        self._start_monotonic = time.monotonic()
        self._sample_index = 0
        self._closed = False

        self.episode_dir = self.root_dir / self.status / self.date / self.episode_name
        self.images_dir = self.episode_dir / "images"
        self.trajectory_path = self.episode_dir / "trajectory.jsonl"
        self.metadata_path = self.episode_dir / "metadata.json"
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self._handle = self.trajectory_path.open("w", encoding="utf-8")
        self._write_jsonl(
            {
                "type": "trajectory_header",
                "episode_name": self.episode_name,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "format": "droid_style_jsonl_v1",
                "image_format": self.image_format,
                "metadata": self.metadata,
            }
        )

    @property
    def num_samples(self) -> int:
        return self._sample_index

    @property
    def duration_s(self) -> float:
        return time.monotonic() - self._start_monotonic

    def write_sample(
        self,
        *,
        observation: ArmObservation | dict[str, Any],
        action: RecordedAction | list[float] | None,
        frames: dict[str, CameraFrame],
        controller_info: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Episode writer is already closed")

        camera_refs: dict[str, str] = {}
        camera_metadata: dict[str, Any] = {}
        for camera_name, frame in sorted(frames.items()):
            image_ref = self._save_frame(camera_name, self._sample_index, frame)
            camera_refs[camera_name] = image_ref
            camera_metadata[camera_name] = {
                "timestamp": frame.timestamp,
                **dict(frame.metadata or {}),
            }

        if isinstance(observation, ArmObservation):
            observation_payload = observation.to_dict()
        else:
            observation_payload = dict(observation)

        action_payload: dict[str, Any] | None
        if isinstance(action, RecordedAction):
            action_payload = action.to_dict()
        elif action is None:
            action_payload = None
        else:
            action_payload = {
                "timestamp": time.monotonic(),
                "action": list(action),
                "source": "external",
                "velocities": None,
            }

        self._write_jsonl(
            {
                "type": "sample",
                "sample_index": self._sample_index,
                "timestamp": time.monotonic(),
                "elapsed_s": self.duration_s,
                "observation": observation_payload,
                "action": action_payload,
                "camera_refs": camera_refs,
                "camera_metadata": camera_metadata,
                "controller_info": dict(controller_info or {}),
                "metadata": dict(metadata or {}),
            }
        )
        self._sample_index += 1

    def close(self, *, success: bool, metadata: dict[str, Any] | None = None) -> Path:
        if self._closed:
            return self.episode_dir

        final_metadata = {
            **self.metadata,
            **dict(metadata or {}),
            "success": bool(success),
            "failure": not bool(success),
            "num_samples": self.num_samples,
            "duration_s": self.duration_s,
            "episode_name": self.episode_name,
            "trajectory_path": "trajectory.jsonl",
        }
        self._write_jsonl(
            {
                "type": "trajectory_footer",
                "success": bool(success),
                "failure": not bool(success),
                "num_samples": self.num_samples,
                "duration_s": self.duration_s,
            }
        )
        self._handle.close()
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(final_metadata), handle, ensure_ascii=False, indent=2)
        self._closed = True

        target_status = "success" if success else "failure"
        if target_status != self.status:
            target_dir = self._unique_episode_dir(target_status)
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(self.episode_dir), str(target_dir))
            self.status = target_status
            self.episode_dir = target_dir
            self.images_dir = target_dir / "images"
            self.trajectory_path = target_dir / "trajectory.jsonl"
            self.metadata_path = target_dir / "metadata.json"
        return self.episode_dir

    def _write_jsonl(self, payload: dict[str, Any]) -> None:
        self._handle.write(json.dumps(_jsonable(payload), ensure_ascii=False) + "\n")
        self._handle.flush()

    def _save_frame(self, camera_name: str, sample_index: int, frame: CameraFrame) -> str:
        camera_dir = self.images_dir / camera_name
        camera_dir.mkdir(parents=True, exist_ok=True)
        suffix = "npy" if self.image_format == "npy" else self.image_format
        image_path = camera_dir / f"{sample_index:06d}.{suffix}"
        image = np.asarray(frame.image)
        if self.image_format == "npy":
            np.save(image_path, image)
        else:
            self._save_image(image_path, image)
        return image_path.relative_to(self.episode_dir).as_posix()

    def _save_image(self, path: Path, image: np.ndarray) -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("保存 jpg/png 图像需要 Pillow，请先执行 `pip install pillow`。") from exc

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        pil_image = Image.fromarray(image)
        if self.image_format == "jpg":
            pil_image.save(path, quality=90)
        else:
            pil_image.save(path)

    def _unique_episode_dir(self, status: str) -> Path:
        base = self.root_dir / status / self.date / self.episode_name
        if not base.exists():
            return base
        index = 1
        while True:
            candidate = self.root_dir / status / self.date / f"{self.episode_name}_{index:02d}"
            if not candidate.exists():
                return candidate
            index += 1
