from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class CameraFrame:
    name: str
    timestamp: float
    image: Any
    metadata: dict[str, Any] = field(default_factory=dict)


class CameraProvider(Protocol):
    def connect(self) -> None:
        ...

    def disconnect(self) -> None:
        ...

    def get_frames(self) -> dict[str, CameraFrame]:
        ...


class NullCameraProvider:
    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def get_frames(self) -> dict[str, CameraFrame]:
        return {}
