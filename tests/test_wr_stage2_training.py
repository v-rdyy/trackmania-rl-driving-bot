from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "train_wr_stage2.py"
SPEC = importlib.util.spec_from_file_location("train_wr_stage2", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def reviewed_gate(target: int, *, status: str = "complete") -> dict[str, object]:
    return {
        "status": status,
        "target_additional_steps": target,
        "evaluation_status": "complete",
        "telemetry_status": "complete",
        "stage2_decision": "continue",
    }


class WrStage2TrainingTests(unittest.TestCase):
    def test_protocol_freezes_base_gate_cadence_and_timebox(self) -> None:
        self.assertEqual(MODULE.INITIAL_MODEL_TIMESTEPS, 3_004_416)
        self.assertEqual(MODULE.GATE_SIZE, 500_000)
        self.assertEqual(MODULE.CHECKPOINT_INTERVAL, 250_000)
        self.assertEqual(MODULE.TIMEBOX, 2_000_000)
        self.assertEqual(MODULE.SIMULATION_SPEED, 100.0)
        self.assertEqual(
            MODULE.TENSORBOARD_RUN_NAME,
            "wr_chase_stage2_localized_drift",
        )
        self.assertEqual(
            MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
            "BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044",
        )

    def test_stage2_uses_isolated_reward_and_artifact_roots(self) -> None:
        self.assertEqual(
            MODULE.localized_drift_assistance_reward.__name__,
            "localized_drift_assistance_reward",
        )
        self.assertEqual(MODULE.RUN_DIR.name, "wr_chase_stage2")
        self.assertEqual(MODULE.CHECKPOINT_DIR.name, "wr_chase_stage2")
        self.assertNotIn("wr_chase_stage1", str(MODULE.RUN_DIR))
        self.assertIn("0.50", MODULE.REWARD_CONTRACT)
        self.assertIn("680..930", MODULE.REWARD_CONTRACT)
        self.assertIn("1100..1410", MODULE.REWARD_CONTRACT)

    def test_training_rejects_target_zero_and_non_gate_targets(self) -> None:
        MODULE.validate_training_target(500_000)
        MODULE.validate_training_target(2_000_000)
        for invalid in (0, -500_000, 250_000, 2_500_000):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                MODULE.validate_training_target(invalid)

    def test_pinned_gate2_hash_is_a_hard_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "gate_01000000_model.zip"
            checkpoint.write_bytes(b"pinned Stage 1 Gate 2")
            expected_hash = MODULE.sha256(checkpoint)
            original = (
                MODULE.WORKSPACE_ROOT,
                MODULE.INITIAL_CHECKPOINT,
                MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
            )
            try:
                MODULE.WORKSPACE_ROOT = root
                MODULE.INITIAL_CHECKPOINT = checkpoint
                MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256 = expected_hash
                result = MODULE.verify_initial_checkpoint()
                MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256 = "0" * 64
                with self.assertRaisesRegex(Exception, "hash changed"):
                    MODULE.verify_initial_checkpoint()
            finally:
                (
                    MODULE.WORKSPACE_ROOT,
                    MODULE.INITIAL_CHECKPOINT,
                    MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
                ) = original
            self.assertEqual(result["sha256"], expected_hash)
            self.assertEqual(result["num_timesteps"], 3_004_416)

    def test_first_training_gate_requires_reviewed_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "gate2.zip"
            checkpoint.write_bytes(b"gate2")
            expected_hash = MODULE.sha256(checkpoint)
            original = (
                MODULE.INITIAL_CHECKPOINT,
                MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
            )
            try:
                MODULE.INITIAL_CHECKPOINT = checkpoint
                MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256 = expected_hash
                manifest = {
                    "stage2_initialization": {"sha256": expected_hash},
                    "gates": [reviewed_gate(0, status="baseline_complete")],
                }
                self.assertEqual(
                    MODULE.select_load_path(
                        target_additional_steps=500_000,
                        resume=None,
                        manifest=manifest,
                    ),
                    checkpoint,
                )
                manifest["gates"][0]["telemetry_status"] = "pending"
                with self.assertRaisesRegex(Exception, "direct-live analysis"):
                    MODULE.select_load_path(
                        target_additional_steps=500_000,
                        resume=None,
                        manifest=manifest,
                    )
            finally:
                (
                    MODULE.INITIAL_CHECKPOINT,
                    MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
                ) = original

    def test_later_gate_requires_exact_prior_prefix_and_continue_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = root / "gate_00500000_model.zip"
            previous.write_bytes(b"stage2 gate one")
            previous_hash = MODULE.sha256(previous)
            original_checkpoint_dir = MODULE.CHECKPOINT_DIR
            try:
                MODULE.CHECKPOINT_DIR = root
                manifest = {
                    "stage2_initialization": {
                        "sha256": MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256
                    },
                    "gates": [
                        reviewed_gate(0, status="baseline_complete"),
                        {
                            **reviewed_gate(500_000),
                            "checkpoint_sha256": previous_hash,
                        },
                    ],
                }
                selected = MODULE.select_load_path(
                    target_additional_steps=1_000_000,
                    resume=None,
                    manifest=manifest,
                )
                self.assertEqual(selected, previous)
                manifest["gates"][1]["stage2_decision"] = "induction_review"
                with self.assertRaisesRegex(Exception, "training must stop"):
                    MODULE.select_load_path(
                        target_additional_steps=1_000_000,
                        resume=None,
                        manifest=manifest,
                    )
            finally:
                MODULE.CHECKPOINT_DIR = original_checkpoint_dir

    def test_gate_names_sort_in_training_order_including_baseline(self) -> None:
        self.assertEqual(MODULE.gate_slug(0), "gate_00000000")
        self.assertEqual(MODULE.gate_slug(500_000), "gate_00500000")
        self.assertEqual(MODULE.gate_slug(2_000_000), "gate_02000000")


if __name__ == "__main__":
    unittest.main()
