from lingzu_teleop.recording.trajectory import TrajectoryRecorder
from lingzu_teleop.types import ArmObservation, RecordedAction


def test_trajectory_recorder_jsonl_roundtrip(tmp_path):
    recorder = TrajectoryRecorder(sample_rate_hz=20)
    recorder.start(name="unit")
    obs = ArmObservation(
        timestamp=1.0,
        joint_positions=[0.0] * 7,
        joint_velocities=[0.0] * 7,
        joint_efforts=[0.0] * 7,
        end_pose=[0.0] * 6,
    )
    action = RecordedAction(timestamp=1.0, action=[0.1] * 7, source="test")
    assert recorder.add_sample(obs, action=action, force=True)
    trajectory = recorder.stop()
    path = recorder.save_jsonl(tmp_path / "trajectory.jsonl", trajectory)

    loaded = TrajectoryRecorder.load_jsonl(path)
    assert loaded.name == "unit"
    assert loaded.num_samples == 1
    assert loaded.samples[0].action == [0.1] * 7
    assert loaded.samples[0].action_source == "test"


def test_actions_from_jsonl(tmp_path):
    recorder = TrajectoryRecorder(sample_rate_hz=10)
    recorder.start(name="actions")
    obs = ArmObservation(timestamp=1.0, joint_positions=[0.0] * 7)
    recorder.add_sample(obs, action=[0.2] * 7, force=True)
    path = recorder.save_jsonl(tmp_path / "actions.jsonl", recorder.stop())

    assert list(TrajectoryRecorder.actions_from_json_or_jsonl(path)) == [[0.2] * 7]
