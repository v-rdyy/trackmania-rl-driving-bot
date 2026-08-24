from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_reward_v4_100.py"
SPEC = importlib.util.spec_from_file_location("evaluate_reward_v4_100", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RewardV4ScaleEvaluationTests(unittest.TestCase):
    def test_wrapper_preserves_policy_and_uses_distinct_100_episode_artifacts(self) -> None:
        MODULE.configure_evaluator()
        evaluator = MODULE.reward_v4_evaluation.evaluator

        self.assertEqual(evaluator.EXPECTED_EPISODES, 100)
        self.assertEqual(evaluator.DEFAULT_RUN_TAG, "scale100")
        self.assertEqual(evaluator.EXPERIMENT_LABEL, "reward-v4")
        self.assertEqual(
            evaluator.REWARD_FUNCTION.__name__,
            "signed_progress_efficiency_reward",
        )
        self.assertEqual(
            evaluator.DEFAULT_CHECKPOINT,
            WORKSPACE_ROOT / "checkpoints" / "reward_v4" / "final_model.zip",
        )
        self.assertEqual(
            evaluator.DEFAULT_ACTION_LOG,
            WORKSPACE_ROOT / "runs" / "reward_v4" / "evaluation_100_actions.jsonl",
        )
        self.assertEqual(
            evaluator.DEFAULT_SUMMARY,
            WORKSPACE_ROOT / "runs" / "reward_v4" / "evaluation_100_summary.json",
        )
        self.assertEqual(
            evaluator.DEFAULT_REPLAY_DIR.name,
            "reward_v4_evaluation_100",
        )


if __name__ == "__main__":
    unittest.main()
