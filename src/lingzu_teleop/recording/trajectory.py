from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from lingzu_teleop.types import ArmObservation, RecordedAction


@dataclass
class TrajectorySample:
    timestamp: float
    observation: dict[str, Any]
    action: Optional[list[float]] = None
    action_source: str = "unknown"
    camera_refs: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecordedTrajectory:
    name: str
    created_at: str
    sample_rate_hz: float
    samples: list[TrajectorySample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].timestamp - self.samples[0].timestamp

    @property
    def num_samples(self) -> int:
        return len(self.samples)


class TrajectoryRecorder:
    """GUI-independent recorder for robot observations and actions."""

    def __init__(self, sample_rate_hz: float = 20.0):
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        self.sample_rate_hz = float(sample_rate_hz)
        self._recording = False
        self._start_monotonic = 0.0
        self._last_sample_elapsed = -1e9
        self._trajectory: RecordedTrajectory | None = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def trajectory(self) -> RecordedTrajectory | None:
        return self._trajectory

    def start(self, name: str = "", *, metadata: dict[str, Any] | None = None) -> None:
        self._trajectory = RecordedTrajectory(
            name=name or time.strftime("trajectory_%Y%m%d_%H%M%S"),
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            sample_rate_hz=self.sample_rate_hz,
            metadata=dict(metadata or {}),
        )
        self._start_monotonic = time.monotonic()
        self._last_sample_elapsed = -1e9
        self._recording = True

    def stop(self) -> RecordedTrajectory | None:
        self._recording = False
        return self._trajectory

    def should_sample(self) -> bool:
        if not self._recording:
            return False
        elapsed = time.monotonic() - self._start_monotonic
        return elapsed - self._last_sample_elapsed >= (1.0 / self.sample_rate_hz) * 0.95

    def add_sample(
        self,
        observation: ArmObservation | dict[str, Any],
        *,
        action: RecordedAction | list[float] | None = None,
        camera_refs: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        if not self._recording or self._trajectory is None:
            return False
        if not force and not self.should_sample():
            return False

        elapsed = time.monotonic() - self._start_monotonic
        self._last_sample_elapsed = elapsed

        if isinstance(observation, ArmObservation):
            observation_payload = observation.to_dict()
        else:
            observation_payload = dict(observation)

        action_values: list[float] | None = None
        action_source = "unknown"
        if isinstance(action, RecordedAction):
            action_values = list(action.action)
            action_source = action.source
        elif action is not None:
            action_values = list(action)

        self._trajectory.samples.append(
            TrajectorySample(
                timestamp=elapsed,
                observation=observation_payload,
                action=action_values,
                action_source=action_source,
                camera_refs=dict(camera_refs or {}),
                metadata=dict(metadata or {}),
            )
        )
        return True

    def save_json(self, path: str | Path, trajectory: RecordedTrajectory | None = None) -> Path:
        trajectory = trajectory or self._trajectory
        if trajectory is None:
            raise RuntimeError("No trajectory to save")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(asdict(trajectory), handle, indent=2, ensure_ascii=True)
        return output

    def save_jsonl(self, path: str | Path, trajectory: RecordedTrajectory | None = None) -> Path:
        trajectory = trajectory or self._trajectory
        if trajectory is None:
            raise RuntimeError("No trajectory to save")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            header = {
                "type": "trajectory_header",
                "name": trajectory.name,
                "created_at": trajectory.created_at,
                "sample_rate_hz": trajectory.sample_rate_hz,
                "metadata": trajectory.metadata,
            }
            handle.write(json.dumps(header, ensure_ascii=True) + "\n")
            for idx, sample in enumerate(trajectory.samples):
                payload = sample.to_dict()
                payload["type"] = "sample"
                payload["sample_index"] = idx
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return output

    @staticmethod
    def load_jsonl(path: str | Path) -> RecordedTrajectory:
        output_name = Path(path).stem
        trajectory = RecordedTrajectory(
            name=output_name,
            created_at="",
            sample_rate_hz=0.0,
        )
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("type") == "trajectory_header":
                    trajectory.name = payload.get("name") or output_name
                    trajectory.created_at = payload.get("created_at", "")
                    trajectory.sample_rate_hz = float(payload.get("sample_rate_hz", 0.0))
                    trajectory.metadata = dict(payload.get("metadata") or {})
                elif payload.get("type") == "sample":
                    trajectory.samples.append(
                        TrajectorySample(
                            timestamp=float(payload.get("timestamp", 0.0)),
                            observation=dict(payload.get("observation") or {}),
                            action=payload.get("action"),
                            action_source=payload.get("action_source", "unknown"),
                            camera_refs=dict(payload.get("camera_refs") or {}),
                            metadata=dict(payload.get("metadata") or {}),
                        )
                    )
        return trajectory

    @staticmethod
    def actions_from_json_or_jsonl(path: str | Path) -> Iterable[list[float]]:
        p = Path(path)
        if p.suffix == ".jsonl":
            trajectory = TrajectoryRecorder.load_jsonl(p)
            for sample in trajectory.samples:
                if sample.action is not None:
                    yield list(sample.action)
            return

        with p.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    action = item.get("action")
                else:
                    action = item
                if action is not None:
                    yield list(action)
            return
        if isinstance(payload, dict) and "samples" in payload:
            for sample in payload["samples"]:
                action = sample.get("action")
                if action is not None:
                    yield list(action)
            return
        raise ValueError(f"Unsupported action file: {path}")
