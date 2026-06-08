from __future__ import annotations

import time
from dataclasses import dataclass

from lingzu_teleop.camera.base import CameraFrame


@dataclass
class RealSenseProviderConfig:
    name: str = "front"
    serial_number: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30


class RealSenseProvider:
    """CameraProvider backed by a RealSense RGB-D camera."""

    def __init__(self, config: RealSenseProviderConfig | None = None):
        self.config = config or RealSenseProviderConfig()
        self._camera = None

    def connect(self) -> None:
        from lingzu_teleop.sdk import RealSenseD435

        self._camera = RealSenseD435(
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            serial=self.config.serial_number,
        )
        self._camera.start()

    def disconnect(self) -> None:
        if self._camera is not None:
            self._camera.stop()
            self._camera = None

    def get_frames(self) -> dict[str, CameraFrame]:
        if self._camera is None:
            raise RuntimeError("RealSenseProvider is not connected")
        frame = self._camera.get_frame()
        return {
            self.config.name: CameraFrame(
                name=self.config.name,
                timestamp=time.monotonic(),
                image=frame.color_rgb,
                metadata={
                    "realsense_timestamp_ms": frame.timestamp_ms,
                    "frame_number": frame.frame_number,
                    "depth_scale": frame.depth_scale,
                    "intrinsics": {
                        "width": frame.intrinsics.width,
                        "height": frame.intrinsics.height,
                        "fx": frame.intrinsics.fx,
                        "fy": frame.intrinsics.fy,
                        "ppx": frame.intrinsics.ppx,
                        "ppy": frame.intrinsics.ppy,
                    },
                },
            )
        }
