from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_reward_v3.py"
SPEC = importlib.util.spec_from_file_location("evaluate_reward_v3", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(
    *,
    x: float,
    z: float,
    progress: float,
    steer: float,
    speed: int = 100,
    race_time_ms: int = 0,
    vertical_offset: float = 0.0,
    upright_cosine: float = 1.0,
) -> dict[str, object]:
    return {
        "position": [x, 0.0, z],
        "progress": progress,
        "display_speed": speed,
        "raw_action": [steer, 1.0, 0.0],
        "race_time_ms": race_time_ms,
        "lateral_offset": z,
        "upright_cosine": upright_cosine,
        "vertical_offset": vertical_offset,
    }


class RewardV3EvaluationTests(unittest.TestCase):
    def test_race_record_filter_discloses_a_restarted_countdown_prefix(self) -> None:
        records = [
            {"race_time_ms": 200},
            {"race_time_ms": 300},
            {"race_time_ms": -200},
            {"race_time_ms": -100},
            {"race_time_ms": 0},
            {"race_time_ms": 100},
        ]

        evaluated, discarded = MODULE.evaluated_race_records(records)

        self.assertEqual(discarded, 4)
        self.assertEqual([row["race_time_ms"] for row in evaluated], [0, 100])

    def test_trajectory_preserves_terminal_vertical_and_upright_evidence(self) -> None:
        records = [
            record(
                x=float(index),
                z=0.0,
                progress=float(index),
                steer=0.0,
                race_time_ms=index * 100,
                vertical_offset=-11.0 if index == 4 else 0.0,
                upright_cosine=-0.5 if index == 4 else 1.0,
            )
            for index in range(5)
        ]

        metrics = MODULE.trajectory_metrics(records)

        self.assertEqual(metrics["minimum_vertical_offset"], -11.0)
        self.assertEqual(metrics["terminal_vertical_offset"], -11.0)
        self.assertEqual(metrics["minimum_upright_cosine"], -0.5)
        self.assertEqual(metrics["terminal_upright_cosine"], -0.5)

    def test_efficient_forward_trajectory_reports_progress_efficiency(self) -> None:
        records = [
            record(
                x=float(index * 20),
                z=0.0,
                progress=float(index * 20),
                steer=0.0,
                race_time_ms=index * 100,
            )
            for index in range(12)
        ]

        metrics = MODULE.trajectory_metrics(records)

        self.assertGreater(metrics["progress_efficiency"], 0.9)

    def test_replay_wait_accepts_a_nonempty_file(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            replay = Path(directory) / "episode.txt"
            replay.write_text("0 steer 0\n", encoding="utf-8")

            MODULE.wait_for_replay(replay, timeout_seconds=0.01)


if __name__ == "__main__":
    unittest.main()
