from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_reward_v1.py"
SPEC = importlib.util.spec_from_file_location("evaluate_reward_v1", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RewardV1EvaluationTests(unittest.TestCase):
    def test_behavior_description_calls_out_stationary_policy(self) -> None:
        episodes = [
            {
                "finished": False,
                "max_progress": 0.0,
                "max_display_speed": 0,
            }
            for _ in range(20)
        ]

        description = MODULE.describe_behavior(episodes)

        self.assertTrue(any("stationary" in line for line in description))
        self.assertIn("Finished 0 of 20 deterministic episodes.", description)

    def test_behavior_description_does_not_hide_unexpected_finish(self) -> None:
        episodes = [
            {
                "finished": index == 0,
                "max_progress": 2_000.0 if index == 0 else 20.0,
                "max_display_speed": 200,
            }
            for index in range(20)
        ]

        description = MODULE.describe_behavior(episodes)

        self.assertIn("Finished 1 of 20 deterministic episodes.", description)
        self.assertFalse(any("stationary" in line for line in description))

    def test_behavior_description_calls_out_repeated_fall_without_progress(self) -> None:
        episodes = [
            {
                "finished": False,
                "fallen": True,
                "max_progress": 0.0,
                "max_display_speed": 63,
            }
            for _ in range(20)
        ]

        description = MODULE.describe_behavior(episodes)

        self.assertTrue(any("fell in every episode" in line for line in description))
        self.assertFalse(any("stationary" in line for line in description))


if __name__ == "__main__":
    unittest.main()
