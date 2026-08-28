from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "analyze_wr_stage2_policy_mode.py"
SPEC = importlib.util.spec_from_file_location(
    "analyze_wr_stage2_policy_mode", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def row(step: int, progress: float, speed: float, lateral: float = 0.0) -> dict:
    return {
        "episode": 0,
        "step": step,
        "race_time_ms": step * 100,
        "progress": progress,
        "lateral_offset": lateral,
        "display_speed": speed,
        "raw_action": [0.0, 1.0, 0.0],
        "upright_cosine": 1.0,
        "position": [progress, 0.0, lateral],
    }


class WrStage2PolicyModeAnalysisTests(unittest.TestCase):
    def test_largest_speed_loss_is_selected_after_progress_gate(self) -> None:
        records = [
            row(0, 1900.0, 400.0),
            row(1, 1950.0, 100.0),
            row(2, 2000.0, 350.0),
            row(3, 2050.0, 80.0, -20.0),
            row(4, 2100.0, 60.0),
        ]
        event = MODULE.largest_speed_loss(records, 2000.0)
        self.assertEqual(event["race_time_ms"], 300)
        self.assertEqual(event["speed_loss"], 270.0)
        self.assertEqual(event["lateral_offset"], -20.0)

    def test_landmark_uses_first_sample_at_or_after_threshold(self) -> None:
        records = [
            row(0, 1999.0, 300.0, -1.0),
            row(1, 2001.0, 300.0, -7.0),
            row(2, 2010.0, 300.0, -9.0),
        ]
        result = MODULE.landmark_lateral_offsets(records, (2000.0,))
        self.assertEqual(result["2000"]["values"], [-7.0])

    def test_rolling_finish_rate_uses_all_episodes_in_window(self) -> None:
        rows = [
            {"race_finished": "True" if index % 2 == 0 else "False"}
            for index in range(20)
        ]
        result = MODULE.rolling_finish_rates(rows)
        self.assertEqual(result["last_10"]["finishes"], 5)
        self.assertEqual(result["last_20"]["finish_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
