from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "validate_progress_replays.py"
SPEC = importlib.util.spec_from_file_location("validate_progress_replays", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProgressReplayValidationTests(unittest.TestCase):
    def test_catalog_cases_preserve_manifest_outcomes_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory(dir=WORKSPACE_ROOT) as directory:
            root = Path(directory)
            replay = root / "run.txt"
            replay.write_text("0.10 gas -65536\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "stage": {"id": "v2_test"},
                        "episodes": [
                            {
                                "episode": 0,
                                "finished": True,
                                "fallen": False,
                                "off_track": False,
                                "timeout": False,
                                "elapsed_ms": 1234,
                                "maximum_progress": 99.5,
                                "input_replay": str(replay.relative_to(WORKSPACE_ROOT)),
                                "input_replay_sha256": "ABC",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            catalog = root / "catalog.json"
            catalog.write_text(
                json.dumps(
                    {
                        "successful_takes": [
                            {"manifest": str(manifest.relative_to(WORKSPACE_ROOT))}
                        ]
                    }
                ),
                encoding="utf-8",
            )

            cases = MODULE.load_cases(catalog)

            case = cases["v2_test:ep1"]
            self.assertEqual(case.expected_outcome, "finish")
            self.assertEqual(case.expected_elapsed_ms, 1234)
            self.assertEqual(case.expected_maximum_progress, 99.5)
            self.assertEqual(case.input_replay_sha256, "ABC")

    def test_all_v2_selection_is_sorted_and_excludes_other_versions(self) -> None:
        make_case = lambda case_id: MODULE.ReplayCase(
            case_id=case_id,
            stage_id=case_id.split(":")[0],
            episode=1,
            input_replay=Path("run.txt"),
            input_replay_sha256="ABC",
            expected_outcome="timeout",
            expected_elapsed_ms=45_000,
            expected_maximum_progress=0.0,
        )
        available = {
            case_id: make_case(case_id)
            for case_id in ("v2_late:ep1", "v1_final:ep1", "v2_early:ep1")
        }

        selected = MODULE.select_cases(available, None, all_v2=True)

        self.assertEqual(
            [case.case_id for case in selected],
            ["v2_early:ep1", "v2_late:ep1"],
        )

    def test_playback_comparison_enforces_outcome_time_and_progress(self) -> None:
        case = MODULE.ReplayCase(
            case_id="v2_final:ep1",
            stage_id="v2_final",
            episode=1,
            input_replay=Path("run.txt"),
            input_replay_sha256="ABC",
            expected_outcome="finish",
            expected_elapsed_ms=30_000,
            expected_maximum_progress=2205.0,
        )

        passing = MODULE.compare_playback(
            case,
            observed_outcome="finish",
            observed_elapsed_ms=30_100,
            observed_maximum_progress=2201.0,
        )
        failing = MODULE.compare_playback(
            case,
            observed_outcome="timeout",
            observed_elapsed_ms=45_000,
            observed_maximum_progress=2100.0,
        )

        self.assertTrue(passing["passed"])
        self.assertFalse(failing["passed"])

    def test_input_stats_reject_invalid_timestamp_and_count_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.txt"
            valid.write_text(
                "0.10 steer 100\n0.10 gas -65536\n0.20 steer 0\n",
                encoding="utf-8",
            )
            invalid = root / "invalid.txt"
            invalid.write_text("soon gas -65536\n", encoding="utf-8")

            stats = MODULE.input_file_stats(valid)

            self.assertEqual(stats["command_lines"], 3)
            self.assertEqual(stats["command_counts"], {"steer": 2, "gas": 1})
            self.assertEqual(stats["maximum_command_time_seconds"], 0.2)
            with self.assertRaisesRegex(ValueError, "invalid timestamp"):
                MODULE.input_file_stats(invalid)


if __name__ == "__main__":
    unittest.main()
