"""CAN driver package entry."""

from lingzu_teleop.sdk.drivers.base import CanDriverProtocol, create_can_driver

__all__ = ["CanDriverProtocol", "create_can_driver"]
