from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.evaluation_metrics import (
    steering_precision_metrics,
    trajectory_precision_metrics,
)


def trajectory_record(
    index: int,
    *,
    x: float,
    progress: float,
    lateral: float = 0.0,
    upright: float = 1.0,
) -> dict[str, object]:
    return {
        "race_time_ms": index * 100,
        "position": [x, 0.0, 0.0],
        "progress": progress,
        "lateral_offset": lateral,
        "upright_cosine": upright,
    }


class EvaluationMetricsTests(unittest.TestCase):
    def test_steering_oscillation_uses_hysteresis_and_fixed_window(self) -> None:
        samples = [
            (index * 0.25, 0.6 if index % 2 else -0.6)
            for index in range(13)
        ]

        metrics = steering_precision_metrics(samples)

        self.assertTrue(metrics["oscillation_detected"])
        self.assertGreaterEqual(metrics["peak_sign_crossings_in_2_seconds"], 4)
        self.assertEqual(metrics["hysteresis_sign_crossings"], 12)

    def test_lateral_and_upside_down_durations_use_telemetry_time(self) -> None:
        records = [
            trajectory_record(
                index,
                x=float(index),
                progress=float(index),
                lateral=12.0 if 10 <= index <= 30 else 2.0,
                upright=-1.0 if 15 <= index <= 25 else 1.0,
            )
            for index in range(41)
        ]

        metrics = trajectory_precision_metrics(records)

        self.assertAlmostEqual(
            metrics["lateral_deviation"]["seconds_above_10_units"],
            2.1,
        )
        self.assertTrue(metrics["upside_down"]["upside_down_detected"])
        self.assertAlmostEqual(
            metrics["upside_down"]["maximum_duration_seconds"],
            1.1,
        )

    def test_stuck_duration_requires_low_progress_and_low_world_motion(self) -> None:
        records = []
        for index in range(61):
            moving_index = min(index, 10)
            records.append(
                trajectory_record(
                    index,
                    x=float(moving_index),
                    progress=float(moving_index),
                )
            )

        metrics = trajectory_precision_metrics(records)

        self.assertTrue(metrics["stuck"]["stuck_detected"])
        self.assertGreaterEqual(metrics["stuck"]["maximum_duration_seconds"], 4.9)

    def test_world_motion_prevents_false_stuck_classification(self) -> None:
        records = [
            trajectory_record(
                index,
                x=float(index),
                progress=0.0,
            )
            for index in range(41)
        ]

        metrics = trajectory_precision_metrics(records)

        self.assertFalse(metrics["stuck"]["stuck_detected"])


if __name__ == "__main__":
    unittest.main()
