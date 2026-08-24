from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_reward_v5_100.py"
SPEC = importlib.util.spec_from_file_location("evaluate_reward_v5_100", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RewardV5ScaleEvaluationTests(unittest.TestCase):
    def test_wrapper_uses_distinct_100_episode_artifacts(self) -> None:
        MODULE.configure_evaluator()
        evaluator = MODULE.reward_v5_evaluation.evaluator

        self.assertEqual(evaluator.EXPECTED_EPISODES, 100)
        self.assertEqual(evaluator.DEFAULT_RUN_TAG, "scale100")
        self.assertEqual(evaluator.EXPERIMENT_LABEL, "reward-v5")
        self.assertEqual(
            evaluator.DEFAULT_ACTION_LOG,
            WORKSPACE_ROOT / "runs" / "reward_v5" / "evaluation_100_actions.jsonl",
        )
        self.assertEqual(
            evaluator.DEFAULT_REPLAY_DIR.name,
            "reward_v5_evaluation_100",
        )


if __name__ == "__main__":
    unittest.main()
