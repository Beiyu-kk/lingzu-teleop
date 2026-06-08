import numpy as np

from lingzu_teleop.camera.base import CameraFrame
from lingzu_teleop.lerobot_adapter import build_lerobot_frame, infer_lerobot_features
from lingzu_teleop.types import ArmObservation


def test_build_lerobot_frame_with_image():
    obs = ArmObservation(timestamp=12.5, joint_positions=[0.0] * 7)
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    frame = build_lerobot_frame(
        obs,
        action=[0.1] * 7,
        images={"front": CameraFrame(name="front", timestamp=12.6, image=image)},
        frame_index=3,
        episode_index=2,
        task_index=1,
    )

    assert frame["frame_index"] == 3
    assert frame["episode_index"] == 2
    assert frame["task_index"] == 1
    assert frame["observation.state"].dtype == np.float32
    assert frame["action"].shape == (7,)
    assert frame["observation.images.front"].shape == (4, 5, 3)


def test_infer_lerobot_features():
    features = infer_lerobot_features(camera_shapes={"front": (480, 640, 3)})
    assert features["observation.state"]["shape"] == (7,)
    assert features["action"]["shape"] == (7,)
    assert features["observation.images.front"]["dtype"] == "video"
