from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "train_wr_stage1.py"
SPEC = importlib.util.spec_from_file_location("train_wr_stage1", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WrStage1TrainingTests(unittest.TestCase):
    def test_protocol_freezes_gate_timebox_and_unchanged_reward(self) -> None:
        self.assertEqual(MODULE.GATE_SIZE, 500_000)
        self.assertEqual(MODULE.CHECKPOINT_INTERVAL, 250_000)
        self.assertEqual(MODULE.INITIAL_TIMEBOX, 5_000_000)
        self.assertEqual(MODULE.SAFETY_CEILING, 10_000_000)
        self.assertIs(
            MODULE.signed_progress_efficiency_reward,
            MODULE.signed_progress_efficiency_reward,
        )
        self.assertNotIn("slip", MODULE.REWARD_CONTRACT)
        self.assertNotIn("steer", MODULE.REWARD_CONTRACT)

    def test_target_must_be_a_gate_not_past_safety_ceiling(self) -> None:
        MODULE.validate_target(500_000)
        MODULE.validate_target(10_000_000)
        for invalid in (0, 250_000, 10_500_000):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MODULE.validate_target(invalid)

    def test_first_gate_loads_pinned_v4_without_manifest(self) -> None:
        selected = MODULE.select_load_path(
            target_additional_steps=500_000,
            resume=None,
            manifest={},
        )
        self.assertEqual(selected, MODULE.V4_CHECKPOINT)

    def test_later_gate_requires_immediately_previous_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = MODULE.CHECKPOINT_DIR
            try:
                MODULE.CHECKPOINT_DIR = Path(directory)
                previous = MODULE.gate_checkpoint(500_000)
                previous.write_bytes(b"checkpoint")
                manifest = {
                    "gates": [
                        {
                            "status": "complete",
                            "target_additional_steps": 500_000,
                            "telemetry_status": "complete",
                            "stage1_decision": "continue",
                        }
                    ]
                }
                selected = MODULE.select_load_path(
                    target_additional_steps=1_000_000,
                    resume=None,
                    manifest=manifest,
                )
                self.assertEqual(selected, previous)
                with self.assertRaises(MODULE.ProtocolError):
                    MODULE.select_load_path(
                        target_additional_steps=1_500_000,
                        resume=None,
                        manifest=manifest,
                    )
            finally:
                MODULE.CHECKPOINT_DIR = original

    def test_gate_names_sort_in_training_order(self) -> None:
        self.assertEqual(MODULE.gate_slug(500_000), "gate_00500000")
        self.assertEqual(MODULE.gate_slug(10_000_000), "gate_10000000")


if __name__ == "__main__":
    unittest.main()
