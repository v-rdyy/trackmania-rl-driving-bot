from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "evaluate_wr_stage2b.py"
SPEC = importlib.util.spec_from_file_location("evaluate_wr_stage2b", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WrStage2bEvaluationTests(unittest.TestCase):
    def test_wrapper_requires_all_graduated_reward_fields(self) -> None:
        for field in (
            "previous_display_speed",
            "upright_cosine",
            "heading_error",
            "lateral_offset",
            "slip_angle_degrees",
            "sliding_wheel_count",
        ):
            self.assertIn(field, MODULE.REQUIRED_LIVE_FIELDS)

    def test_configuration_uses_stage2b_gate_reward_and_paths(self) -> None:
        evaluator_names = (
            "EXPECTED_INITIAL_CHECKPOINT_SHA256",
            "INITIAL_CHECKPOINT",
            "INITIAL_MODEL_TIMESTEPS",
            "MANIFEST_PATH",
            "RUN_DIR",
            "TIMEBOX",
            "GATE_SIZE",
            "RUN_LABEL",
            "EXPERIMENT_SLUG",
            "PROTOCOL_LABEL",
            "REWARD_FUNCTION",
            "REWARD_FUNCTION_NAME",
            "REQUIRED_LIVE_FIELDS",
        )
        trainer_names = (
            "RUN_DIR",
            "GATE_SUMMARY_DIR",
            "CHECKPOINT_DIR",
            "MANIFEST_PATH",
            "MONITOR_PREFIX",
            "INITIAL_CHECKPOINT",
            "EXPECTED_INITIAL_CHECKPOINT_SHA256",
            "INITIAL_MODEL_TIMESTEPS",
            "GATE_SIZE",
            "CHECKPOINT_INTERVAL",
            "TIMEBOX",
            "TENSORBOARD_RUN_NAME",
            "REWARD_FUNCTION",
            "REWARD_FUNCTION_NAME",
            "RUN_LABEL",
            "CHECKPOINT_NAME_PREFIX",
            "PPO_CLASS",
            "TARGET_KL",
            "INITIAL_SOURCE",
            "REWARD_CONTRACT",
        )
        evaluator_original = {
            name: getattr(MODULE.evaluator, name) for name in evaluator_names
        }
        trainer_original = {
            name: getattr(MODULE.training.trainer, name) for name in trainer_names
        }
        generic_names = (
            "EXPERIMENT_LABEL",
            "EXPERIMENT_SLUG",
            "PROTOCOL_LABEL",
            "REWARD_FUNCTION",
            "RUN_DIR",
            "DEFAULT_CHECKPOINT",
            "DEFAULT_ACTION_LOG",
            "DEFAULT_SUMMARY",
            "DEFAULT_REPLAY_DIR",
            "EXPECTED_EPISODES",
            "DEFAULT_RUN_TAG",
            "EXPECTED_CHECKPOINT_SHA256",
        )
        generic_original = {
            name: getattr(MODULE.evaluator.evaluator, name)
            for name in generic_names
        }
        try:
            MODULE.configure_evaluator()
            MODULE.evaluator.validate_evaluation_target(50_000)
            gate = {
                "checkpoint_sha256": MODULE.training.EXPECTED_INITIAL_CHECKPOINT_SHA256
            }
            paths = MODULE.evaluator.configure_evaluator(0, gate)
            self.assertIs(
                MODULE.evaluator.evaluator.REWARD_FUNCTION,
                MODULE.training.REWARD_FUNCTION,
            )
            self.assertIn("wr_chase_stage2b", str(paths["summary"]))
            self.assertIn("gate_00000000", str(paths["replay_dir"]))
            self.assertIn("wr_chase_stage2b", str(paths["replay_dir"]))
        finally:
            for name, value in evaluator_original.items():
                setattr(MODULE.evaluator, name, value)
            for name, value in trainer_original.items():
                setattr(MODULE.training.trainer, name, value)
            for name, value in generic_original.items():
                setattr(MODULE.evaluator.evaluator, name, value)


if __name__ == "__main__":
    unittest.main()
