from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class ArmObservation:
    timestamp: float
    joint_positions: list[float]
    joint_velocities: list[float] = field(default_factory=list)
    joint_efforts: list[float] = field(default_factory=list)
    end_pose: list[float] = field(default_factory=list)
    status: dict[str, Any] = field(default_factory=dict)

    def state_vector(
        self,
        *,
        include_velocities: bool = False,
        include_efforts: bool = False,
        include_end_pose: bool = False,
    ) -> list[float]:
        values = list(self.joint_positions)
        if include_velocities:
            values.extend(self.joint_velocities)
        if include_efforts:
            values.extend(self.joint_efforts)
        if include_end_pose:
            values.extend(self.end_pose)
        return values

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecordedAction:
    timestamp: float
    action: list[float]
    source: str = "unknown"
    velocities: Optional[list[float]] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
