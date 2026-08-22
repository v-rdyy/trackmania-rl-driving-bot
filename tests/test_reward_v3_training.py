from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "train_reward_v3.py"
SPEC = importlib.util.spec_from_file_location("train_reward_v3", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class RewardV3TrainingTests(unittest.TestCase):
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

    def test_runner_uses_distinct_v3_artifact_names(self) -> None:
        self.assertEqual(MODULE.MINIMUM_TIMESTEPS, 1_000_000)
        self.assertEqual(
            MODULE.TENSORBOARD_RUN_NAME,
            "reward_v3_centerline_progress",
        )
        self.assertEqual(MODULE.RUN_DIR.name, "reward_v3")

    def test_runner_freezes_v3_stuck_thresholds(self) -> None:
        self.assertEqual(MODULE.STUCK_WINDOW_MS, 2_000)
        self.assertEqual(MODULE.STUCK_PROGRESS_GAIN_UNITS, 1.0)
        self.assertEqual(MODULE.STUCK_WORLD_DISTANCE_UNITS, 2.0)

    def test_backup_gate_verifies_the_archive_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "backup.zip"
            archive.write_bytes(b"pinned evidence")
            checksum = root / "backup.zip.sha256"
            expected_hash = MODULE.sha256(archive)
            checksum.write_text(f"{expected_hash}  backup.zip\n", encoding="utf-8")
            original = (
                MODULE.WORKSPACE_ROOT,
                MODULE.PRE_V3_BACKUP,
                MODULE.PRE_V3_BACKUP_CHECKSUM,
                MODULE.EXPECTED_PRE_V3_BACKUP_SHA256,
            )
            try:
                MODULE.WORKSPACE_ROOT = root
                MODULE.PRE_V3_BACKUP = archive
                MODULE.PRE_V3_BACKUP_CHECKSUM = checksum
                MODULE.EXPECTED_PRE_V3_BACKUP_SHA256 = expected_hash

                result = MODULE.verify_pre_v3_backup()
            finally:
                (
                    MODULE.WORKSPACE_ROOT,
                    MODULE.PRE_V3_BACKUP,
                    MODULE.PRE_V3_BACKUP_CHECKSUM,
                    MODULE.EXPECTED_PRE_V3_BACKUP_SHA256,
                ) = original

            self.assertEqual(result["sha256"], expected_hash)
            self.assertEqual(result["size_bytes"], len(b"pinned evidence"))

    def test_logger_cleanup_tolerates_failure_before_sb3_setup(self) -> None:
        class ModelWithoutLogger:
            pass

        MODULE.close_model_logger(ModelWithoutLogger())

    def test_logger_cleanup_closes_initialized_logger(self) -> None:
        class Logger:
            closed = False

            def close(self) -> None:
                self.closed = True

        class Model:
            _logger = Logger()

        model = Model()
        MODULE.close_model_logger(model)

        self.assertTrue(model._logger.closed)


if __name__ == "__main__":
    unittest.main()
