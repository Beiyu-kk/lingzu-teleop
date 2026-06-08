from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from lingzu_teleop.camera.base import CameraFrame
from lingzu_teleop.types import ArmObservation


def build_lerobot_frame(
    observation: ArmObservation,
    *,
    action: list[float] | np.ndarray | None = None,
    images: Mapping[str, CameraFrame | Any] | None = None,
    frame_index: int = 0,
    episode_index: int = 0,
    task_index: int = 0,
) -> dict[str, Any]:
    """Map local observations/actions to LeRobot-style frame keys."""

    frame: dict[str, Any] = {
        "timestamp": float(observation.timestamp),
        "frame_index": int(frame_index),
        "episode_index": int(episode_index),
        "task_index": int(task_index),
        "observation.state": np.asarray(observation.joint_positions, dtype=np.float32),
    }
    if action is not None:
        frame["action"] = np.asarray(action, dtype=np.float32)

    for name, image_or_frame in (images or {}).items():
        if isinstance(image_or_frame, CameraFrame):
            image = image_or_frame.image
            frame[f"observation.images.{name}"] = image
            frame[f"observation.images.{name}.timestamp"] = float(image_or_frame.timestamp)
        else:
            frame[f"observation.images.{name}"] = image_or_frame
    return frame


def infer_lerobot_features(
    *,
    state_dim: int = 7,
    action_dim: int = 7,
    camera_shapes: Mapping[str, tuple[int, int, int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Return a lightweight feature map matching the default data contract."""

    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"joint_{idx + 1}.pos" for idx in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": [f"joint_{idx + 1}.target" for idx in range(action_dim)],
        },
        "timestamp": {"dtype": "float64", "shape": (1,)},
        "frame_index": {"dtype": "int64", "shape": (1,)},
        "episode_index": {"dtype": "int64", "shape": (1,)},
        "task_index": {"dtype": "int64", "shape": (1,)},
    }
    for name, shape in (camera_shapes or {}).items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": tuple(shape),
            "names": ["height", "width", "channel"],
        }
    return features


def require_lerobot() -> Any:
    try:
        import lerobot
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed. Install it only when using official LeRobot tooling."
        ) from exc
    return lerobot
