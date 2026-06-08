from __future__ import annotations

from collections.abc import Mapping

from lingzu_teleop.camera.base import CameraFrame, CameraProvider


class MultiCameraProvider:
    """Small fan-in wrapper for named camera providers."""

    def __init__(self, providers: Mapping[str, CameraProvider]):
        self.providers = dict(providers)

    def connect(self) -> None:
        connected: list[CameraProvider] = []
        try:
            for provider in self.providers.values():
                provider.connect()
                connected.append(provider)
        except Exception:
            for provider in reversed(connected):
                try:
                    provider.disconnect()
                except Exception:
                    pass
            raise

    def disconnect(self) -> None:
        for provider in reversed(list(self.providers.values())):
            provider.disconnect()

    def get_frames(self) -> dict[str, CameraFrame]:
        frames: dict[str, CameraFrame] = {}
        for provider in self.providers.values():
            frames.update(provider.get_frames())
        return frames
