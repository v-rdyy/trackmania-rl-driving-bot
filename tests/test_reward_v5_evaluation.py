from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_reward_v5.py"
SPEC = importlib.util.spec_from_file_location("evaluate_reward_v5", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def summary(*, oscillation: int, finishes: int, mean_lap_ms: float) -> dict:
    return {
        "precision_summary": {"oscillation_detected_episodes": oscillation},
        "finishes": finishes,
        "average_finish_time_ms": mean_lap_ms,
    }


class RewardV5EvaluationTests(unittest.TestCase):
    def test_wrapper_selects_v5_reward_checkpoint_and_artifacts(self) -> None:
        MODULE.configure_evaluator()

        self.assertEqual(MODULE.evaluator.EXPERIMENT_LABEL, "reward-v5")
        self.assertEqual(MODULE.evaluator.EXPECTED_EPISODES, 20)
        self.assertEqual(
            MODULE.evaluator.REWARD_FUNCTION.__name__,
            "steering_rate_smoothness_reward",
        )
        self.assertEqual(
            MODULE.evaluator.DEFAULT_CHECKPOINT,
            WORKSPACE_ROOT / "checkpoints" / "reward_v5" / "final_model.zip",
        )
        self.assertEqual(
            MODULE.evaluator.DEFAULT_REPLAY_DIR.name,
            "reward_v5_evaluation",
        )

    def test_scale_gate_requires_every_preregistered_condition(self) -> None:
        passing = MODULE.scale_gate(
            summary(oscillation=10, finishes=19, mean_lap_ms=25_500.0)
        )
        self.assertTrue(passing["passed"])

        for key, candidate in (
            (
                "oscillation",
                summary(oscillation=11, finishes=20, mean_lap_ms=25_000),
            ),
            ("finish", summary(oscillation=0, finishes=18, mean_lap_ms=25_000)),
            ("lap", summary(oscillation=0, finishes=20, mean_lap_ms=25_501)),
        ):
            with self.subTest(key=key):
                self.assertFalse(MODULE.scale_gate(candidate)["passed"])


if __name__ == "__main__":
    unittest.main()
