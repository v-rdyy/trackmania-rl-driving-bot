from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_reward_v2.py"
SPEC = importlib.util.spec_from_file_location("evaluate_reward_v2", SCRIPT_PATH)
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
) -> dict[str, object]:
    return {
        "position": [x, 0.0, z],
        "progress": progress,
        "display_speed": speed,
        "raw_action": [steer, 1.0, 0.0],
        "race_time_ms": race_time_ms,
        "lateral_offset": z,
        "upright_cosine": 1.0,
    }


class RewardV2EvaluationTests(unittest.TestCase):
    def test_looping_trajectory_is_flagged(self) -> None:
        points = [
            (0.0, 0.0),
            (40.0, 0.0),
            (40.0, 40.0),
            (0.0, 40.0),
            (0.0, 0.0),
        ] * 2
        records = [
            record(
                x=x,
                z=z,
                progress=5.0,
                steer=0.8,
                race_time_ms=index * 100,
            )
            for index, (x, z) in enumerate(points)
        ]

        metrics = MODULE.trajectory_metrics(records)

        self.assertTrue(metrics["loop_candidate"])
        self.assertTrue(metrics["speed_farming_candidate"])
        self.assertLess(metrics["progress_efficiency"], 0.25)

    def test_oscillating_trajectory_is_flagged(self) -> None:
        records = [
            record(
                x=float(index * 15),
                z=20.0 if index % 2 else -20.0,
                progress=2.0,
                steer=1.0 if index % 2 else -1.0,
                race_time_ms=index * 100,
            )
            for index in range(12)
        ]

        metrics = MODULE.trajectory_metrics(records)

        self.assertTrue(metrics["oscillation_candidate"])
        self.assertTrue(metrics["speed_farming_candidate"])
        self.assertGreaterEqual(metrics["steering_sign_changes"], 6)

    def test_efficient_forward_trajectory_is_not_flagged(self) -> None:
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

        self.assertFalse(metrics["speed_farming_candidate"])
        self.assertGreater(metrics["progress_efficiency"], 0.9)


if __name__ == "__main__":
    unittest.main()
