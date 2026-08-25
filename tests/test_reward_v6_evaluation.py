from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_reward_v6.py"
SPEC = importlib.util.spec_from_file_location("evaluate_reward_v6", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def summary(
    *,
    oscillation: int = 10,
    finishes: int = 19,
    mean_lap_ms: float = 25_500.0,
    mean_reversals: float = 28.0,
    mean_hotspot_reversals: float = 6.4,
) -> dict:
    return {
        "precision_summary": {
            "oscillation_detected_episodes": oscillation,
            "mean_significant_direction_reversals": mean_reversals,
        },
        "finishes": finishes,
        "average_finish_time_ms": mean_lap_ms,
        "v6_reversal_frequency": {
            "hotspots": {"mean_events_per_episode": mean_hotspot_reversals}
        },
    }


class RewardV6EvaluationTests(unittest.TestCase):
    def test_wrapper_selects_v6_reward_checkpoint_and_artifacts(self) -> None:
        MODULE.configure_evaluator()
        self.assertEqual(MODULE.evaluator.EXPERIMENT_LABEL, "reward-v6")
        self.assertEqual(MODULE.evaluator.EXPECTED_EPISODES, 20)
        self.assertEqual(
            MODULE.evaluator.REWARD_FUNCTION.__name__,
            "clustered_reversal_frequency_reward",
        )
        self.assertEqual(
            MODULE.evaluator.DEFAULT_CHECKPOINT,
            WORKSPACE_ROOT / "checkpoints" / "reward_v6" / "final_model.zip",
        )
        self.assertEqual(MODULE.evaluator.DEFAULT_REPLAY_DIR.name, "reward_v6_evaluation")

    def test_location_metrics_preserve_frozen_hotspot_bins(self) -> None:
        events = [
            {"progress_bin": 0},
            {"progress_bin": 1},
            {"progress_bin": 2},
            {"progress_bin": 9},
        ]
        metrics = MODULE.location_metrics_from_events(events, episode_count=2)
        self.assertEqual(metrics["hotspots"]["event_count"], 3)
        self.assertEqual(metrics["hotspots"]["mean_events_per_episode"], 1.5)
        self.assertEqual(metrics["hotspots"]["outside_hotspot_event_count"], 1)

    def test_gate_requires_all_five_preregistered_conditions(self) -> None:
        self.assertTrue(MODULE.success_gate(summary())["passed"])
        failing = (
            summary(oscillation=11),
            summary(finishes=18),
            summary(mean_lap_ms=25_501.0),
            summary(mean_reversals=28.1),
            summary(mean_hotspot_reversals=6.5),
        )
        for candidate in failing:
            with self.subTest(candidate=candidate):
                self.assertFalse(MODULE.success_gate(candidate)["passed"])


if __name__ == "__main__":
    unittest.main()
