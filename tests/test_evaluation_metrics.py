from __future__ import annotations

import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.evaluation_metrics import (
    aggregate_precision_metrics,
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

    def test_duplicate_terminal_timestamp_contributes_zero_duration(self) -> None:
        records = [
            trajectory_record(0, x=0.0, progress=0.0),
            trajectory_record(1, x=1.0, progress=1.0),
            trajectory_record(1, x=2.0, progress=2.0, lateral=12.0, upright=-1.0),
        ]

        metrics = trajectory_precision_metrics(records)

        self.assertAlmostEqual(metrics["duration_seconds"], 0.1)
        self.assertAlmostEqual(
            metrics["lateral_deviation"]["seconds_above_10_units"],
            0.0,
        )
        self.assertAlmostEqual(
            metrics["upside_down"]["total_duration_seconds"],
            0.0,
        )

    def test_trajectory_rejects_backward_race_time(self) -> None:
        records = [
            trajectory_record(1, x=0.0, progress=0.0),
            trajectory_record(0, x=1.0, progress=1.0),
        ]

        with self.assertRaisesRegex(ValueError, "nondecreasing"):
            trajectory_precision_metrics(records)

    def test_precision_aggregation_preserves_severity_totals(self) -> None:
        first = {
            "steering": steering_precision_metrics([(0.0, -0.5), (0.1, 0.5)]),
            **trajectory_precision_metrics(
                [
                    trajectory_record(0, x=0.0, progress=0.0),
                    trajectory_record(1, x=0.0, progress=0.0, lateral=12.0, upright=-1.0),
                ]
            ),
        }
        second = {
            "steering": steering_precision_metrics([(0.0, 0.0), (0.1, 0.0)]),
            **trajectory_precision_metrics(
                [
                    trajectory_record(0, x=0.0, progress=0.0),
                    trajectory_record(1, x=1.0, progress=1.0, lateral=4.0),
                ]
            ),
        }

        summary = aggregate_precision_metrics([first, second])

        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["upside_down_detected_episodes"], 1)
        self.assertAlmostEqual(summary["total_upside_down_seconds"], 0.1)
        self.assertAlmostEqual(summary["maximum_absolute_lateral_offset"], 12.0)
        self.assertAlmostEqual(summary["mean_steering_total_variation"], 0.5)
        self.assertAlmostEqual(summary["maximum_steering_total_variation"], 1.0)
        self.assertAlmostEqual(summary["mean_absolute_steering"], 0.25)


if __name__ == "__main__":
    unittest.main()
