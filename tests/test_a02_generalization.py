from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_reward_v4_a02.py"
SPEC = importlib.util.spec_from_file_location("evaluate_reward_v4_a02", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class A02GeneralizationTests(unittest.TestCase):
    def test_wrapper_freezes_v4_checkpoint_reward_and_a02_artifacts(self) -> None:
        MODULE.configure_evaluator()

        self.assertEqual(MODULE.evaluator.TRACK_LABEL, "A02-Race")
        self.assertEqual(
            MODULE.evaluator.MAP_TO_LOAD,
            "A02-Race.Challenge.Gbx",
        )
        self.assertEqual(
            MODULE.evaluator.REWARD_FUNCTION.__name__,
            "signed_progress_efficiency_reward",
        )
        self.assertEqual(MODULE.evaluator.EXPECTED_EPISODES, 20)
        self.assertEqual(
            MODULE.evaluator.DEFAULT_CHECKPOINT,
            WORKSPACE_ROOT / "checkpoints" / "reward_v4" / "final_model.zip",
        )
        self.assertEqual(
            MODULE.evaluator.EXPECTED_CHECKPOINT_SHA256,
            MODULE.V4_CHECKPOINT_SHA256,
        )
        self.assertIsNone(MODULE.evaluator.HUMAN_PB_MS)


if __name__ == "__main__":
    unittest.main()
