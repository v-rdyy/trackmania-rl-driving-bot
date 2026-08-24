from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_reward_v4.py"
SPEC = importlib.util.spec_from_file_location("evaluate_reward_v4", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RewardV4EvaluationTests(unittest.TestCase):
    def test_wrapper_selects_v4_reward_checkpoint_and_artifacts(self) -> None:
        MODULE.configure_evaluator()

        self.assertEqual(MODULE.evaluator.EXPERIMENT_LABEL, "reward-v4")
        self.assertEqual(MODULE.evaluator.EXPERIMENT_SLUG, "reward_v4")
        self.assertEqual(
            MODULE.evaluator.REWARD_FUNCTION.__name__,
            "signed_progress_efficiency_reward",
        )
        self.assertEqual(
            MODULE.evaluator.DEFAULT_CHECKPOINT,
            WORKSPACE_ROOT / "checkpoints" / "reward_v4" / "final_model.zip",
        )
        self.assertEqual(
            MODULE.evaluator.DEFAULT_REPLAY_DIR.name,
            "reward_v4_evaluation",
        )


if __name__ == "__main__":
    unittest.main()
