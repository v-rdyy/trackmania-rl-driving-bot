from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "analyze_wr_stage1_gate.py"
SPEC = importlib.util.spec_from_file_location("analyze_wr_stage1_gate", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def record(
    step: int,
    *,
    progress: float,
    sliding: int = 2,
    slip: float = 4.0,
    yaw: float = 0.5,
) -> dict[str, object]:
    return {
        "race_time_ms": step * 100,
        "progress": progress,
        "ground_contact_count": 4,
        "display_speed": 400,
        "sliding_wheel_count": sliding,
        "slip_angle_degrees": slip,
        "body_up_yaw_rate": yaw,
    }


class WrStage1AnalysisTests(unittest.TestCase):
    def test_success_and_failure_replay_stop_conditions_are_distinct(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn("if case.finished and current_finish:", source)
        self.assertIn(
            "if not case.finished and current_time >= case.terminal_race_time_ms:",
            source,
        )

    def test_confirmed_slide_requires_three_consecutive_two_wheel_samples(self) -> None:
        records = [
            record(0, progress=700),
            record(1, progress=710),
            record(2, progress=720),
        ]
        metrics = MODULE.drift_metrics(records)
        self.assertEqual(
            len(metrics["known_zones"]["first_turn"]["confirmed_windows"]),
            1,
        )
        records[1]["sliding_wheel_count"] = 1
        metrics = MODULE.drift_metrics(records)
        self.assertEqual(
            len(metrics["known_zones"]["first_turn"]["confirmed_windows"]),
            0,
        )
        self.assertEqual(
            len(metrics["known_zones"]["first_turn"]["candidate_windows"]),
            1,
        )

    def test_known_zone_discovery_requires_three_episodes(self) -> None:
        cases = []
        for episode in range(1, 4):
            cases.append(
                {
                    "episode": episode,
                    "drift": {
                        "known_zones": {
                            "first_turn": {"confirmed_windows": [{}]},
                            "final_corner": {"confirmed_windows": []},
                        },
                        "track_wide_confirmed_windows": [],
                    },
                }
            )
        discovery = MODULE.aggregate_discovery(cases[:2])
        self.assertFalse(discovery["telemetry_discovery_triggered"])
        discovery = MODULE.aggregate_discovery(cases)
        self.assertTrue(discovery["telemetry_discovery_triggered"])
        self.assertEqual(discovery["trigger_locations"], ["first_turn"])

    def test_plateau_gate_requires_all_frozen_checks(self) -> None:
        self.assertEqual(MODULE.assess_stage1({"gates": []})["decision"], "continue")
        self.assertEqual(MODULE.PLATEAU_MIN_ADDITIONAL_STEPS, 2_000_000)
        self.assertEqual(MODULE.SIGNIFICANT_LAP_IMPROVEMENT_MS, 50)
        self.assertEqual(MODULE.STOCHASTIC_WINDOW_IMPROVEMENT_MS, 100)
        self.assertEqual(MODULE.SUCCESS_REPLAY_FINISH_TOLERANCE_MS, 1_000)

    def test_plateau_requires_three_flat_gates_and_stochastic_window(self) -> None:
        original_root = MODULE.WORKSPACE_ROOT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gates = []
            try:
                MODULE.WORKSPACE_ROOT = root
                for target, mean in (
                    (1_000_000, 24_940.0),
                    (1_500_000, 24_930.0),
                    (2_000_000, 24_920.0),
                ):
                    training = root / f"gate_{target}.json"
                    training.write_text(
                        json.dumps(
                            {
                                "stochastic_finish_time_mean_previous_500_ms": 25_000,
                                "stochastic_finish_time_mean_last_500_ms": 24_950,
                            }
                        ),
                        encoding="utf-8",
                    )
                    gates.append(
                        {
                            "target_additional_steps": target,
                            "telemetry_status": "complete",
                            "telemetry_discovery_triggered": False,
                            "deterministic_best_finish_time_ms": 24_900,
                            "deterministic_mean_finish_time_ms": mean,
                            "training_summary": training.name,
                        }
                    )
                assessment = MODULE.assess_stage1({"gates": gates})
            finally:
                MODULE.WORKSPACE_ROOT = original_root
        self.assertEqual(assessment["decision"], "plateau")
        self.assertTrue(all(assessment["plateau_checks"].values()))


if __name__ == "__main__":
    unittest.main()
