from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    WORKSPACE_ROOT / "scripts" / "diagnose_wr_stage2_intermediate_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location(
    "diagnose_wr_stage2_intermediate_checkpoint", SCRIPT_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WrStage2IntermediateCheckpointTests(unittest.TestCase):
    def test_configuration_is_frozen_to_deterministic_gate750k(self) -> None:
        MODULE.configure_evaluator()
        self.assertTrue(MODULE.evaluator.POLICY_DETERMINISTIC)
        self.assertIsNone(MODULE.evaluator.POLICY_RANDOM_SEED)
        self.assertIsNone(MODULE.evaluator.POLICY_SDE_SAMPLE_FREQ)
        self.assertEqual(MODULE.evaluator.EXPECTED_EPISODES, 10)
        self.assertEqual(MODULE.MODEL_TIMESTEPS, 3_756_176)
        self.assertEqual(MODULE.STAGE2_INTERACTIONS, 751_760)
        self.assertEqual(
            MODULE.evaluator.EXPECTED_CHECKPOINT_SHA256,
            MODULE.CHECKPOINT_SHA256,
        )

    def test_binding_records_zero_training_and_source_checkpoint(self) -> None:
        summary = {
            "deterministic": True,
            "policy_sampling": {"mode": "deterministic"},
        }
        bound = MODULE.bind_diagnostic_summary(
            summary,
            source_evidence={"checkpoint_sha256": "A" * 64},
            direct_live_audit={"complete": True},
        )
        diagnostic = bound["intermediate_checkpoint_diagnostic"]
        self.assertEqual(diagnostic["training_interactions"], 0)
        self.assertEqual(
            diagnostic["source_evidence"]["checkpoint_sha256"],
            "A" * 64,
        )
        self.assertTrue(diagnostic["direct_live_simstate_audit"]["complete"])

    def test_binding_rejects_stochastic_result(self) -> None:
        with self.assertRaisesRegex(Exception, "was not deterministic"):
            MODULE.bind_diagnostic_summary(
                {"deterministic": False, "policy_sampling": {"mode": "stochastic"}},
                source_evidence={},
                direct_live_audit={},
            )


if __name__ == "__main__":
    unittest.main()
