from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "train_reward_v2.py"
SPEC = importlib.util.spec_from_file_location("train_reward_v2", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RewardV2TrainingTests(unittest.TestCase):
    def test_power_setting_parser_reads_disabled_ac_and_dc(self) -> None:
        output = """
        Current AC Power Setting Index: 0x00000000
        Current DC Power Setting Index: 0x00000000
        """

        self.assertEqual(
            MODULE.parse_power_setting_indices(output),
            {"AC": 0, "DC": 0},
        )

    def test_power_setting_parser_rejects_incomplete_output(self) -> None:
        with self.assertRaisesRegex(Exception, "both AC and DC"):
            MODULE.parse_power_setting_indices(
                "Current AC Power Setting Index: 0x00000000"
            )

    def test_legacy_failed_manifest_wall_time_is_recoverable(self) -> None:
        manifest = {
            "started_at_utc": "2026-08-22T18:00:00+00:00",
            "failed_at_utc": "2026-08-22T18:02:03.5+00:00",
        }

        self.assertEqual(MODULE.prior_wall_seconds(manifest), 123.5)

    def test_runner_uses_distinct_v2_artifact_names(self) -> None:
        self.assertEqual(MODULE.MINIMUM_TIMESTEPS, 1_000_000)
        self.assertEqual(MODULE.TENSORBOARD_RUN_NAME, "reward_v2_dense_speed")
        self.assertEqual(MODULE.RUN_DIR.name, "reward_v2")


if __name__ == "__main__":
    unittest.main()
