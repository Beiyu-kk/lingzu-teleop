import json
import time

import numpy as np

from lingzu_teleop.camera.base import CameraFrame
from lingzu_teleop.recording.droid_style import DroidStyleEpisodeWriter
from lingzu_teleop.types import ArmObservation, RecordedAction


def test_droid_style_writer_moves_success_episode(tmp_path):
    writer = DroidStyleEpisodeWriter(
        tmp_path,
        metadata={"task": "test"},
        episode_name="episode_001",
        image_format="npy",
    )
    initial_dir = writer.episode_dir
    assert "failure" in initial_dir.parts

    frame = CameraFrame(
        name="wrist",
        timestamp=time.monotonic(),
        image=np.zeros((4, 5, 3), dtype=np.uint8),
        metadata={"serial": "camera"},
    )
    observation = ArmObservation(
        timestamp=time.monotonic(),
        joint_positions=[0.0] * 7,
    )
    action = RecordedAction(
        timestamp=time.monotonic(),
        action=[0.0] * 7,
        source="test",
    )
    writer.write_sample(
        observation=observation,
        action=action,
        frames={"wrist": frame},
        controller_info={"movement_enabled": True},
    )
    final_dir = writer.close(success=True)

    assert "success" in final_dir.parts
    assert not initial_dir.exists()
    assert (final_dir / "images" / "wrist" / "000000.npy").is_file()
    assert (final_dir / "trajectory.jsonl").is_file()
    metadata = json.loads((final_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["success"] is True
    assert metadata["num_samples"] == 1
