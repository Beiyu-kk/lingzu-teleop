from lingzu_teleop.sdk.data_types import (
    MotorFeedback,
    ArmStatus,
    ArmJointStates,
    ArmEndPose,
    MotorHighSpdInfo,
    MotorLowSpdInfo,
    MotorAngleLimitMaxVel,
    DynamicsInfo,
    TrajectoryResult,
)
from lingzu_teleop.sdk.protocol import (
    MotorType,
    RunMode,
    ControlMode,
    MoveMode,
    ArmState,
    LogLevel,
)

__version__ = "1.0.0"


_REALSENSE_EXPORTS = {
    "CameraIntrinsics",
    "DepthUnit",
    "PointCloud",
    "RGBDFrame",
    "RealSenseD435",
    "RigidTransform",
    "ColorOrder",
    "import_pyrealsense2",
    "rpy_to_matrix",
}


def __getattr__(name):
    if name == "ELA3Interface":
        from lingzu_teleop.sdk.interface import ELA3Interface

        return ELA3Interface
    if name == "ArmManager":
        from lingzu_teleop.sdk.arm_manager import ArmManager

        return ArmManager
    if name in _REALSENSE_EXPORTS:
        from lingzu_teleop.sdk import realsense

        return getattr(realsense, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_kinematics():
    """延迟导入 ELA3Kinematics（避免无 pinocchio 环境下 import 失败）"""
    from lingzu_teleop.sdk.kinematics import ELA3Kinematics

    return ELA3Kinematics


__all__ = [
    "ELA3Interface",
    "ArmManager",
    "get_kinematics",
    "MotorFeedback",
    "ArmStatus",
    "ArmJointStates",
    "ArmEndPose",
    "MotorHighSpdInfo",
    "MotorLowSpdInfo",
    "MotorAngleLimitMaxVel",
    "DynamicsInfo",
    "TrajectoryResult",
    "MotorType",
    "RunMode",
    "ControlMode",
    "MoveMode",
    "ArmState",
    "LogLevel",
    *_REALSENSE_EXPORTS,
]
