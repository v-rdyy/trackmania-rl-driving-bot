from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from scripts.capture_telemetry import sample_summary, state_record
from trackmania_rl.tmi_bridge import ProtocolError


class CaptureTelemetryTests(unittest.TestCase):
    def test_state_record_serializes_finite_simulation_state(self) -> None:
        state = SimpleNamespace(
            position=np.array([1.0, 2.0, 3.0]),
            velocity=np.array([4.0, 5.0, 6.0]),
            rotation_matrix=np.eye(3),
            yaw_pitch_roll=np.array([0.1, 0.2, 0.3]),
            race_time=1234,
            display_speed=42,
        )

        record = state_record(state, 0.25)

        self.assertEqual(record["race_time_ms"], 1234)
        self.assertEqual(record["position"], [1.0, 2.0, 3.0])
        self.assertEqual(record["display_speed"], 42)

    def test_state_record_rejects_nonfinite_telemetry(self) -> None:
        state = SimpleNamespace(
            position=np.array([np.nan, 0.0, 0.0]),
            velocity=np.zeros(3),
            rotation_matrix=np.eye(3),
            yaw_pitch_roll=np.zeros(3),
            race_time=0,
            display_speed=0,
        )

        with self.assertRaisesRegex(ProtocolError, "NaN or infinite"):
            state_record(state, 0.0)

    def test_sample_summary_rejects_stale_race_times(self) -> None:
        samples = [
            {
                "race_time_ms": 100,
                "position": [float(index), 0.0, 0.0],
                "velocity": [1.0, 0.0, 0.0],
                "rotation_matrix": np.eye(3).tolist(),
            }
            for index in range(10)
        ]

        with self.assertRaisesRegex(ProtocolError, "stale race times"):
            sample_summary(samples)


if __name__ == "__main__":
    unittest.main()
