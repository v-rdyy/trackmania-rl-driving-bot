from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "analyze_wr_stage2_gate.py"
SPEC = importlib.util.spec_from_file_location("analyze_wr_stage2_gate", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def expected_reward(
    *,
    progress: float,
    previous_progress: float,
    high_water_delta: float,
    sliding: int,
    terminated: bool,
    truncated: bool = False,
) -> float:
    progress_delta = progress - previous_progress
    v4 = min(max(progress_delta, -20.0), 20.0) / 10.0 - 0.10
    if terminated:
        v4 += 50.0
    elif truncated:
        v4 -= 250.0
    in_zone = 680.0 <= progress <= 930.0 or 1100.0 <= progress <= 1410.0
    bonus = 0.5 * min(max(high_water_delta, 0.0), 20.0) / 20.0 if in_zone and sliding else 0.0
    return v4 + bonus


def record(
    episode: int,
    step: int,
    *,
    progress: float,
    previous_progress: float,
    high_water_delta: float,
    sliding: int = 0,
    terminated: bool = False,
    truncated: bool = False,
) -> dict[str, object]:
    wheel_sliding = [index < sliding for index in range(4)]
    reward = expected_reward(
        progress=progress,
        previous_progress=previous_progress,
        high_water_delta=high_water_delta,
        sliding=sliding,
        terminated=terminated,
        truncated=truncated,
    )
    return {
        "episode": episode,
        "step": step,
        "race_time_ms": step * 100,
        "raw_action": [0.0, 1.0, 0.0],
        "full_simstate_available": True,
        "position": [progress, 0.0, 0.0],
        "velocity": [
            math.tan(math.radians(4.0)) * 100.0 if sliding else 0.0,
            0.0,
            100.0,
        ],
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "angular_velocity": [0.0, 0.5 if sliding else 0.0, 0.0],
        "wheel_ground_contacts": [True, True, True, True],
        "wheel_sliding": wheel_sliding,
        "ground_contact_count": 4,
        "sliding_wheel_count": sliding,
        "display_speed": 400,
        "slip_angle_degrees": 4.0 if sliding else 0.0,
        "body_up_yaw_rate": 0.5 if sliding else 0.0,
        "previous_progress": previous_progress,
        "progress": progress,
        "new_high_water_progress_delta": high_water_delta,
        "lateral_offset": 2.0,
        "vertical_offset": 0.0,
        "heading_error": 0.0,
        "upright_cosine": 1.0,
        "reward": reward,
        "reward_function": "localized_drift_assistance_reward",
        "terminated": terminated,
        "truncated": truncated,
        "race_finished": terminated,
    }


def evaluation_and_records(*, discovered_episodes: int = 0):
    details = []
    records = []
    for episode in range(10):
        slide = 2 if episode < discovered_episodes else 0
        progresses = (700.0, 710.0, 720.0) if slide else (10.0, 20.0, 30.0)
        previous = progresses[0] - 10.0
        for step, progress in enumerate(progresses):
            records.append(
                record(
                    episode,
                    step,
                    progress=progress,
                    previous_progress=previous,
                    high_water_delta=10.0,
                    sliding=slide,
                    terminated=step == 2,
                )
            )
            previous = progress
        details.append(
            {
                "episode": episode,
                "finished": True,
                "terminal_race_time_ms": 200,
                "timeout": False,
                "off_track": False,
                "fallen": False,
                "stuck": False,
            }
        )
    return {"episodes": 10, "episodes_detail": details, "finishes": 10}, records


def gate(
    target: int,
    finishes: int,
    *,
    valid: bool = True,
    induction: bool = False,
) -> dict[str, object]:
    return {
        "target_additional_steps": target,
        "live_analysis_status": "complete" if valid else "invalid",
        "telemetry_status": "complete" if valid else "invalid",
        "telemetry_conclusion_valid": valid,
        "telemetry_induction_triggered": induction,
        "deterministic_finishes": finishes,
    }


def bound_evidence(root: Path, *, malformed_action_log: bool = False):
    reference = root / "data" / "tracks" / "a01_reference_path.csv"
    reference.parent.mkdir(parents=True)
    reference.write_text("x,y,z\n0,0,0\n1,0,0\n", encoding="utf-8")
    checkpoint = root / "checkpoints" / "base.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"approved optimizer-bearing checkpoint")
    replay_dir = root / "artifacts" / "replays" / "wr_chase_stage2" / "gate_00000000"
    replay_dir.mkdir(parents=True)
    details = []
    replay_hashes = []
    for episode in range(10):
        replay = replay_dir / f"episode_{episode:02d}.txt"
        replay.write_text(f"episode {episode}\n", encoding="utf-8")
        replay_hash = MODULE.sha256(replay)
        replay_hashes.append(replay_hash)
        details.append(
            {
                "episode": episode,
                "finished": False,
                "terminal_race_time_ms": 45_000,
                "timeout": True,
                "off_track": False,
                "fallen": False,
                "stuck": False,
                "input_replay": str(replay.relative_to(root)),
                "input_replay_bytes": replay.stat().st_size,
                "input_replay_sha256": replay_hash,
            }
        )
    action_path = root / "runs" / "wr_chase_stage2" / "gates" / "gate_00000000_evaluation_actions.jsonl"
    action_path.parent.mkdir(parents=True)
    action_path.write_text(
        "{not-json}\n" if malformed_action_log else "{}\n",
        encoding="utf-8",
    )
    evaluation = {
        "gate_target_additional_steps": 0,
        "measurement_source": MODULE.EXPECTED_TELEMETRY_SOURCE,
        "replay_telemetry_fallback_used": False,
        "run_tag": "gate_00000000_baseline",
        "deterministic": True,
        "simulation_speed": 6.0,
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": MODULE.sha256(checkpoint),
        "reference_path_sha256": MODULE.sha256(reference),
        "action_log_sha256": MODULE.sha256(action_path),
        "episodes": 10,
        "finishes": 0,
        "best_finish_time_ms": None,
        "average_finish_time_ms": None,
        "input_replay_count": 10,
        "input_replay_sha256": replay_hashes,
        "episodes_detail": details,
    }
    evaluation_path = action_path.with_name("gate_00000000_evaluation.json")
    evaluation_path.write_text(
        json.dumps(evaluation, allow_nan=False) + "\n", encoding="utf-8"
    )
    gate_record = {
        "status": "baseline_complete",
        "target_additional_steps": 0,
        "evaluation_status": "complete",
        "checkpoint": str(checkpoint.relative_to(root)),
        "checkpoint_sha256": MODULE.sha256(checkpoint),
        "evaluation_summary": str(evaluation_path.relative_to(root)),
        "evaluation_summary_sha256": MODULE.sha256(evaluation_path),
        "evaluation_action_log": str(action_path.relative_to(root)),
        "evaluation_action_log_sha256": MODULE.sha256(action_path),
        "evaluation_replay_dir": str(replay_dir.relative_to(root)),
        "deterministic_finishes": 0,
        "deterministic_best_finish_time_ms": None,
        "deterministic_mean_finish_time_ms": None,
        "direct_live_simstate_audit": {
            "complete": True,
            "source": MODULE.EXPECTED_TELEMETRY_SOURCE,
            "replay_fallback_used": False,
            "episodes": 10,
            "terminal_outcomes_matched": True,
        },
        "telemetry_status": "pending",
        "stage2_decision": "pending",
    }
    manifest = {
        "status": "awaiting_gate_live_analysis",
        "stage2_initialization": {
            "path": str(checkpoint.relative_to(root)),
            "sha256": MODULE.sha256(checkpoint),
        },
        "reward_function": MODULE.EXPECTED_REWARD_FUNCTION,
        "telemetry_source_required": MODULE.EXPECTED_TELEMETRY_SOURCE,
        "replay_telemetry_fallback_allowed": False,
        "gate_size_nominal": MODULE.GATE_SIZE,
        "timebox": MODULE.TIMEBOX,
        "evaluation_episodes": 10,
        "step_period_ms": MODULE.STEP_PERIOD_MS,
        "latest_evaluated_gate": 0,
        "gates": [gate_record],
    }
    manifest_path = root / "runs" / "wr_chase_stage2" / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest, evaluation, evaluation_path, action_path, manifest_path, reference


class FixtureReference:
    def project(self, position):
        return SimpleNamespace(
            progress=float(position[0]),
            lateral_offset=2.0,
            vertical_offset=0.0,
            tangent_xz=MODULE.np.asarray([0.0, 1.0]),
        )


class WrStage2AnalysisTests(unittest.TestCase):
    def test_source_has_no_game_or_input_playback_dependency(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("trackmania_rl.env", source)
        self.assertNotIn("analyze_wr_stage1_gate", source)
        self.assertNotIn("analyze_v4_wr_prerequisites", source)
        self.assertNotIn("advance_playback", source)
        self.assertNotIn("execute_command", source)

    def test_exact_reward_and_high_water_are_recomputed(self) -> None:
        row = MODULE.normalize_live_record(
            record(
                0,
                0,
                progress=700.0,
                previous_progress=690.0,
                high_water_delta=10.0,
                sliding=2,
            )
        )
        audit = MODULE.recompute_reward_audit([row])
        self.assertTrue(audit["valid"])
        self.assertAlmostEqual(audit["recomputed_localized_bonus_total"], 0.25)
        self.assertAlmostEqual(audit["recomputed_reward_total"], 1.15)
        self.assertEqual(audit["known_zones"]["first_turn"]["rewarded_steps"], 1)

        row["new_high_water_progress_delta"] = 9.0
        audit = MODULE.recompute_reward_audit([row])
        self.assertFalse(audit["valid"])
        self.assertGreater(audit["maximum_absolute_high_water_delta_error"], 0.9)

    def test_logged_reward_mismatch_is_invalid(self) -> None:
        row = MODULE.normalize_live_record(
            record(
                0,
                0,
                progress=700.0,
                previous_progress=690.0,
                high_water_delta=10.0,
                sliding=2,
            )
        )
        row["reward"] = float(row["reward"]) + 0.01
        audit = MODULE.recompute_reward_audit([row])
        self.assertFalse(audit["valid"])
        self.assertEqual(len(audit["mismatch_rows"]), 1)

    def test_live_aliases_are_normalized(self) -> None:
        source = record(
            0,
            0,
            progress=10.0,
            previous_progress=0.0,
            high_water_delta=10.0,
        )
        source["lateral_offset_from_reference"] = source.pop("lateral_offset")
        source["vertical_offset_from_reference"] = source.pop("vertical_offset")
        source["heading_error_radians"] = source.pop("heading_error")
        normalized = MODULE.normalize_live_record(source)
        self.assertEqual(normalized["lateral_offset"], 2.0)
        self.assertEqual(normalized["vertical_offset"], 0.0)
        self.assertEqual(normalized["heading_error"], 0.0)

    def test_every_row_requires_complete_simstate_and_four_boolean_wheels(self) -> None:
        source = record(
            0,
            0,
            progress=10.0,
            previous_progress=0.0,
            high_water_delta=10.0,
        )
        source["full_simstate_available"] = False
        with self.assertRaises(MODULE.AnalysisError):
            MODULE.normalize_live_record(source)

        source["full_simstate_available"] = True
        source["wheel_sliding"] = [False, False, False]
        with self.assertRaises(MODULE.AnalysisError):
            MODULE.normalize_live_record(source)

        source["wheel_sliding"] = [0, 0, 0, 0]
        with self.assertRaises(MODULE.AnalysisError):
            MODULE.normalize_live_record(source)

    def test_logged_slide_dynamics_must_match_raw_live_simstate(self) -> None:
        source = record(
            0,
            0,
            progress=700.0,
            previous_progress=690.0,
            high_water_delta=10.0,
            sliding=2,
        )
        source["slip_angle_degrees"] = 12.0
        with self.assertRaisesRegex(MODULE.AnalysisError, "slip angle"):
            MODULE.normalize_live_record(source)

        source = record(
            0,
            0,
            progress=700.0,
            previous_progress=690.0,
            high_water_delta=10.0,
            sliding=2,
        )
        source["body_up_yaw_rate"] = 1.5
        with self.assertRaisesRegex(MODULE.AnalysisError, "yaw rate"):
            MODULE.normalize_live_record(source)

    def test_action_aliases_and_reference_diagnostics_are_recomputed(self) -> None:
        source = record(
            0,
            0,
            progress=700.0,
            previous_progress=690.0,
            high_water_delta=10.0,
        )
        source["input_steer"] = 0.5
        with self.assertRaisesRegex(MODULE.AnalysisError, "input_steer"):
            MODULE.normalize_live_record(source)

        normalized = MODULE.normalize_live_record(
            record(
                0,
                0,
                progress=700.0,
                previous_progress=690.0,
                high_water_delta=10.0,
            )
        )
        normalized["progress"] = 701.0
        with self.assertRaisesRegex(MODULE.AnalysisError, "progress"):
            MODULE.validate_reference_diagnostics(
                [normalized], FixtureReference()
            )

    def test_exactly_ten_episode_logs_and_matching_terminals_are_required(self) -> None:
        evaluation, records = evaluation_and_records()
        grouped, active = MODULE.group_live_records(records, evaluation)
        self.assertEqual(len(grouped), 10)
        self.assertEqual(len(active), 10)

        with self.assertRaises(MODULE.AnalysisError):
            MODULE.group_live_records(
                [row for row in records if int(row["episode"]) != 9],
                evaluation,
            )

        bad_evaluation, bad_records = evaluation_and_records()
        bad_evaluation["episodes_detail"][0]["terminal_race_time_ms"] = 400
        with self.assertRaises(MODULE.AnalysisError):
            MODULE.group_live_records(bad_records, bad_evaluation)

        bad_evaluation, bad_records = evaluation_and_records()
        bad_evaluation["episodes_detail"][0]["terminal_race_time_ms"] = 300
        with self.assertRaises(MODULE.AnalysisError):
            MODULE.group_live_records(bad_records, bad_evaluation)

        bad_evaluation, bad_records = evaluation_and_records()
        bad_evaluation["episodes"] = 9
        with self.assertRaises(MODULE.AnalysisError):
            MODULE.group_live_records(bad_records, bad_evaluation)

        bad_evaluation, bad_records = evaluation_and_records()
        bad_evaluation["episodes_detail"][0]["steps"] = 4
        with self.assertRaisesRegex(MODULE.AnalysisError, "step count"):
            MODULE.group_live_records(bad_records, bad_evaluation)

        bad_evaluation, bad_records = evaluation_and_records()
        bad_evaluation["episodes_detail"][0]["total_reward"] = 999.0
        with self.assertRaisesRegex(MODULE.AnalysisError, "reward total"):
            MODULE.group_live_records(bad_records, bad_evaluation)

    def test_episode_must_end_once_and_cannot_have_post_terminal_rows(self) -> None:
        evaluation, records = evaluation_and_records()
        records[1]["terminated"] = True
        records[1]["race_finished"] = True
        with self.assertRaisesRegex(MODULE.AnalysisError, "terminal row"):
            MODULE.group_live_records(records, evaluation)

        evaluation, records = evaluation_and_records()
        terminal = records[2]
        terminal["terminated"] = False
        terminal["race_finished"] = False
        terminal["reward"] = expected_reward(
            progress=float(terminal["progress"]),
            previous_progress=float(terminal["previous_progress"]),
            high_water_delta=float(terminal["new_high_water_progress_delta"]),
            sliding=0,
            terminated=False,
        )
        detail = evaluation["episodes_detail"][0]
        detail["finished"] = False
        with self.assertRaisesRegex(MODULE.AnalysisError, "terminal row"):
            MODULE.group_live_records(records, evaluation)

    def test_confirmed_window_requires_exact_100ms_sample_cadence(self) -> None:
        rows = [
            MODULE.normalize_live_record(
                {
                    **record(
                        0,
                        step,
                        progress=700.0 + 10.0 * step,
                        previous_progress=690.0 + 10.0 * step,
                        high_water_delta=10.0,
                        sliding=2,
                        terminated=step == 2,
                    ),
                    "race_time_ms": step,
                }
            )
            for step in range(3)
        ]
        metrics = MODULE.drift_metrics(rows)
        self.assertEqual(
            metrics["known_zones"]["first_turn"]["confirmed_windows"], []
        )

        bad_evaluation, bad_records = evaluation_and_records()
        bad_evaluation["episodes_detail"][0]["finished"] = False
        with self.assertRaises(MODULE.AnalysisError):
            MODULE.group_live_records(bad_records, bad_evaluation)

        bad_evaluation, bad_records = evaluation_and_records()
        detail = bad_evaluation["episodes_detail"][0]
        detail["finished"] = False
        detail["timeout"] = True
        terminal = bad_records[2]
        terminal["terminated"] = False
        terminal["race_finished"] = False
        terminal["reward"] = expected_reward(
            progress=float(terminal["progress"]),
            previous_progress=float(terminal["previous_progress"]),
            high_water_delta=float(terminal["new_high_water_progress_delta"]),
            sliding=0,
            terminated=False,
        )
        with self.assertRaises(MODULE.AnalysisError):
            MODULE.group_live_records(bad_records, bad_evaluation)

    def test_previous_progress_must_join_consecutive_live_rows(self) -> None:
        evaluation, records = evaluation_and_records()
        records[1]["previous_progress"] = 999.0
        records[1]["reward"] = expected_reward(
            progress=float(records[1]["progress"]),
            previous_progress=999.0,
            high_water_delta=float(records[1]["new_high_water_progress_delta"]),
            sliding=0,
            terminated=False,
        )
        with self.assertRaises(MODULE.AnalysisError):
            MODULE.group_live_records(records, evaluation)

        evaluation, records = evaluation_and_records()
        records[1]["step"] = 7
        with self.assertRaisesRegex(MODULE.AnalysisError, "step"):
            MODULE.group_live_records(records, evaluation)

    def test_progress_continuity_is_required_across_clock_restart(self) -> None:
        before = MODULE.normalize_live_record(
            record(
                0,
                5,
                progress=20.0,
                previous_progress=10.0,
                high_water_delta=10.0,
            )
        )
        after = MODULE.normalize_live_record(
            record(
                0,
                0,
                progress=30.0,
                previous_progress=20.0,
                high_water_delta=10.0,
            )
        )
        MODULE.validate_progress_continuity([before, after])
        after["previous_progress"] = 90.0
        with self.assertRaises(MODULE.AnalysisError):
            MODULE.validate_progress_continuity([before, after])

    def test_three_of_ten_confirmed_live_windows_trigger_induction(self) -> None:
        evaluation, records = evaluation_and_records(discovered_episodes=3)
        result = MODULE.analyze_evaluation(
            evaluation,
            records,
            reference=FixtureReference(),
        )
        self.assertTrue(result["instrumentation_valid"])
        self.assertEqual(
            result["discovery"]["known_zone_confirmed_episode_counts"]["first_turn"],
            3,
        )
        self.assertTrue(result["discovery"]["telemetry_induction_triggered"])
        self.assertEqual(result["discovery"]["trigger_locations"], ["first_turn"])
        self.assertEqual(
            result["reward_audit"]["known_zones"]["first_turn"]["rewarded_steps"],
            9,
        )
        self.assertEqual(result["zones"]["first_turn"]["candidate_window_count"], 3)
        self.assertEqual(result["zones"]["first_turn"]["confirmed_window_count"], 3)
        self.assertEqual(
            [window["episode"] for window in result["zones"]["first_turn"]["confirmed_windows"]],
            [0, 1, 2],
        )
        self.assertTrue(
            math.isclose(
                result["reward_audit"]["known_zones"]["first_turn"]["localized_bonus"],
                2.25,
            )
        )

    def test_target_zero_baseline_continues_and_is_not_a_safety_gate(self) -> None:
        assessment = MODULE.assess_stage2({"gates": [gate(0, 0)]})
        self.assertEqual(assessment["decision"], "continue")
        self.assertTrue(assessment["baseline_excluded_from_safety_sequence"])

    def test_target_zero_preexisting_slide_requires_comparability_review(self) -> None:
        assessment = MODULE.assess_stage2(
            {"gates": [gate(0, 10, induction=True)]}
        )
        self.assertEqual(
            assessment["decision"], "baseline_comparability_review"
        )
        self.assertTrue(assessment["baseline_excluded_from_safety_sequence"])

    def test_invalid_latest_gate_requires_instrumentation(self) -> None:
        assessment = MODULE.assess_stage2(
            {"gates": [gate(0, 10), gate(500_000, 10, valid=False)]}
        )
        self.assertEqual(assessment["decision"], "instrumentation_required")

    def test_any_invalid_or_malformed_prior_gate_stops_the_sequence(self) -> None:
        assessment = MODULE.assess_stage2(
            {
                "gates": [
                    gate(0, 10, valid=False),
                    gate(500_000, 10),
                ]
            }
        )
        self.assertEqual(assessment["decision"], "instrumentation_required")

        assessment = MODULE.assess_stage2(
            {"gates": [gate(0, 10), gate(1_000_000, 10)]}
        )
        self.assertEqual(assessment["decision"], "instrumentation_required")

    def test_discovery_precedes_safety_and_timebox_review(self) -> None:
        assessment = MODULE.assess_stage2(
            {
                "gates": [
                    gate(0, 10),
                    gate(500_000, 10),
                    gate(1_000_000, 0, induction=True),
                ]
            }
        )
        self.assertEqual(assessment["decision"], "review_induction")

    def test_safety_review_rules_exclude_baseline(self) -> None:
        one_weak = MODULE.assess_stage2(
            {"gates": [gate(0, 1), gate(500_000, 5)]}
        )
        self.assertEqual(one_weak["decision"], "continue")

        zero = MODULE.assess_stage2(
            {"gates": [gate(0, 10), gate(500_000, 0)]}
        )
        self.assertEqual(zero["decision"], "safety_review")

        two_weak = MODULE.assess_stage2(
            {
                "gates": [
                    gate(0, 10),
                    gate(500_000, 5),
                    gate(1_000_000, 4),
                ]
            }
        )
        self.assertEqual(two_weak["decision"], "safety_review")

    def test_timebox_and_continue_decisions(self) -> None:
        continuing = MODULE.assess_stage2(
            {"gates": [gate(0, 10), gate(500_000, 8)]}
        )
        self.assertEqual(continuing["decision"], "continue")

        timebox = MODULE.assess_stage2(
            {
                "gates": [
                    gate(0, 10),
                    gate(500_000, 8),
                    gate(1_000_000, 8),
                    gate(1_500_000, 8),
                    gate(2_000_000, 8),
                ]
            }
        )
        self.assertEqual(timebox["decision"], "ineffective_within_timebox")

    def test_evidence_binding_covers_action_checkpoint_and_all_replays(self) -> None:
        original = (
            MODULE.WORKSPACE_ROOT,
            MODULE.REFERENCE_PATH,
            MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                manifest,
                evaluation,
                evaluation_path,
                action_path,
                _,
                reference,
            ) = bound_evidence(root)
            try:
                MODULE.WORKSPACE_ROOT = root
                MODULE.REFERENCE_PATH = reference
                MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256 = manifest[
                    "stage2_initialization"
                ]["sha256"]
                bound = MODULE.validate_evidence_binding(
                    manifest,
                    0,
                    evaluation_path,
                    action_path,
                    evaluation,
                )
                self.assertEqual(bound["target_additional_steps"], 0)

                evaluation["action_log_sha256"] = "0" * 64
                with self.assertRaisesRegex(MODULE.AnalysisError, "action log"):
                    MODULE.validate_evidence_binding(
                        manifest,
                        0,
                        evaluation_path,
                        action_path,
                        evaluation,
                    )
                evaluation["action_log_sha256"] = MODULE.sha256(action_path)

                replay = root / manifest["gates"][0]["evaluation_replay_dir"] / "episode_03.txt"
                replay.write_text("changed evidence\n", encoding="utf-8")
                with self.assertRaisesRegex(MODULE.AnalysisError, "replay"):
                    MODULE.validate_evidence_binding(
                        manifest,
                        0,
                        evaluation_path,
                        action_path,
                        evaluation,
                    )
            finally:
                (
                    MODULE.WORKSPACE_ROOT,
                    MODULE.REFERENCE_PATH,
                    MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
                ) = original

    def test_malformed_action_log_is_recorded_as_instrumentation_failure(self) -> None:
        original = (
            MODULE.WORKSPACE_ROOT,
            MODULE.REFERENCE_PATH,
            MODULE.ANALYSIS_ROOT,
            MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
            MODULE.parse_args,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                manifest,
                _,
                evaluation_path,
                action_path,
                manifest_path,
                reference,
            ) = bound_evidence(root, malformed_action_log=True)
            analysis_root = root / "artifacts" / "analysis" / "wr_chase_stage2"
            try:
                MODULE.WORKSPACE_ROOT = root
                MODULE.REFERENCE_PATH = reference
                MODULE.ANALYSIS_ROOT = analysis_root
                MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256 = manifest[
                    "stage2_initialization"
                ]["sha256"]
                MODULE.parse_args = lambda: SimpleNamespace(
                    gate_target=0,
                    evaluation_summary=evaluation_path,
                    action_log=action_path,
                    manifest=manifest_path,
                    output=None,
                )
                self.assertEqual(MODULE.main(), 2)
            finally:
                (
                    MODULE.WORKSPACE_ROOT,
                    MODULE.REFERENCE_PATH,
                    MODULE.ANALYSIS_ROOT,
                    MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
                    MODULE.parse_args,
                ) = original
            updated = json.loads(manifest_path.read_text(encoding="utf-8"))
            analysis = json.loads(
                (analysis_root / "gate_00000000" / "analysis_summary.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(updated["status"], "instrumentation_required")
        self.assertEqual(updated["gates"][0]["telemetry_status"], "invalid")
        self.assertFalse(analysis["instrumentation_valid"])

    def test_missing_action_log_is_recorded_as_instrumentation_failure(self) -> None:
        original = (
            MODULE.WORKSPACE_ROOT,
            MODULE.REFERENCE_PATH,
            MODULE.ANALYSIS_ROOT,
            MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
            MODULE.parse_args,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                manifest,
                _,
                evaluation_path,
                action_path,
                manifest_path,
                reference,
            ) = bound_evidence(root)
            action_path.unlink()
            analysis_root = root / "artifacts" / "analysis" / "wr_chase_stage2"
            try:
                MODULE.WORKSPACE_ROOT = root
                MODULE.REFERENCE_PATH = reference
                MODULE.ANALYSIS_ROOT = analysis_root
                MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256 = manifest[
                    "stage2_initialization"
                ]["sha256"]
                MODULE.parse_args = lambda: SimpleNamespace(
                    gate_target=0,
                    evaluation_summary=evaluation_path,
                    action_log=action_path,
                    manifest=manifest_path,
                    output=None,
                )
                self.assertEqual(MODULE.main(), 2)
            finally:
                (
                    MODULE.WORKSPACE_ROOT,
                    MODULE.REFERENCE_PATH,
                    MODULE.ANALYSIS_ROOT,
                    MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
                    MODULE.parse_args,
                ) = original
            updated = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(updated["status"], "instrumentation_required")
        self.assertEqual(updated["gates"][0]["telemetry_status"], "invalid")

    def test_stale_repeat_and_noncanonical_requests_do_not_mutate_manifest(self) -> None:
        original = (
            MODULE.WORKSPACE_ROOT,
            MODULE.REFERENCE_PATH,
            MODULE.ANALYSIS_ROOT,
            MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
            MODULE.parse_args,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (
                manifest,
                _,
                evaluation_path,
                action_path,
                manifest_path,
                reference,
            ) = bound_evidence(root)
            analysis_root = root / "artifacts" / "analysis" / "wr_chase_stage2"
            try:
                MODULE.WORKSPACE_ROOT = root
                MODULE.REFERENCE_PATH = reference
                MODULE.ANALYSIS_ROOT = analysis_root
                MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256 = manifest[
                    "stage2_initialization"
                ]["sha256"]

                custom_output = root / "custom-analysis.json"
                MODULE.parse_args = lambda: SimpleNamespace(
                    gate_target=0,
                    evaluation_summary=evaluation_path,
                    action_log=action_path,
                    manifest=manifest_path,
                    output=custom_output,
                )
                before = manifest_path.read_bytes()
                with self.assertRaises(SystemExit):
                    MODULE.main()
                self.assertEqual(manifest_path.read_bytes(), before)
                self.assertFalse(custom_output.exists())

                manifest["gates"][0]["telemetry_status"] = "complete"
                manifest["gates"][0]["stage2_decision"] = "continue"
                manifest_path.write_text(
                    json.dumps(manifest, allow_nan=False) + "\n", encoding="utf-8"
                )
                MODULE.parse_args = lambda: SimpleNamespace(
                    gate_target=0,
                    evaluation_summary=evaluation_path,
                    action_log=action_path,
                    manifest=manifest_path,
                    output=None,
                )
                before = manifest_path.read_bytes()
                with self.assertRaises(SystemExit):
                    MODULE.main()
                self.assertEqual(manifest_path.read_bytes(), before)

                manifest["gates"][0]["telemetry_status"] = "pending"
                manifest["gates"][0]["stage2_decision"] = "pending"
                manifest["gates"].append(
                    {
                        "target_additional_steps": 500_000,
                        "evaluation_status": "complete",
                        "telemetry_status": "pending",
                        "stage2_decision": "pending",
                    }
                )
                manifest_path.write_text(
                    json.dumps(manifest, allow_nan=False) + "\n", encoding="utf-8"
                )
                before = manifest_path.read_bytes()
                with self.assertRaises(SystemExit):
                    MODULE.main()
                self.assertEqual(manifest_path.read_bytes(), before)
            finally:
                (
                    MODULE.WORKSPACE_ROOT,
                    MODULE.REFERENCE_PATH,
                    MODULE.ANALYSIS_ROOT,
                    MODULE.EXPECTED_INITIAL_CHECKPOINT_SHA256,
                    MODULE.parse_args,
                ) = original

    def test_manifest_records_completed_live_analysis_and_gate_decision(self) -> None:
        original_root = MODULE.WORKSPACE_ROOT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "run_manifest.json"
            summary_path = root / "analysis_summary.json"
            pending = gate(0, 10)
            pending.update(
                {
                    "telemetry_status": "pending",
                    "stage2_decision": "pending",
                }
            )
            manifest_path.write_text(
                json.dumps({"gates": [pending]}, allow_nan=False),
                encoding="utf-8",
            )
            summary_path.write_text("{}\n", encoding="utf-8")
            try:
                MODULE.WORKSPACE_ROOT = root
                MODULE.update_manifest(
                    manifest_path,
                    0,
                    summary_path,
                    {
                        "instrumentation_valid": True,
                        "lap_metrics": {
                            "finishes": 10,
                            "best_finish_time_ms": 24_850,
                            "average_finish_time_ms": 24_865.0,
                        },
                        "discovery": {"telemetry_induction_triggered": False},
                    },
                )
            finally:
                MODULE.WORKSPACE_ROOT = original_root
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["gates"][0]["live_analysis_status"], "complete")
        self.assertEqual(manifest["gates"][0]["telemetry_status"], "complete")
        self.assertEqual(manifest["gates"][0]["stage2_decision"], "continue")
        self.assertEqual(manifest["latest_stage2_assessment"]["decision"], "continue")

    def test_invalid_analysis_preserves_evaluator_lap_fields(self) -> None:
        original_root = MODULE.WORKSPACE_ROOT
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "run_manifest.json"
            summary_path = root / "analysis_summary.json"
            existing = gate(500_000, 7)
            existing.update(
                {
                    "telemetry_status": "pending",
                    "stage2_decision": "pending",
                    "deterministic_best_finish_time_ms": 24_900,
                    "deterministic_mean_finish_time_ms": 25_000.0,
                }
            )
            manifest_path.write_text(
                json.dumps({"gates": [existing]}, allow_nan=False),
                encoding="utf-8",
            )
            summary_path.write_text("{}\n", encoding="utf-8")
            try:
                MODULE.WORKSPACE_ROOT = root
                MODULE.update_manifest(
                    manifest_path,
                    500_000,
                    summary_path,
                    {"instrumentation_valid": False},
                )
            finally:
                MODULE.WORKSPACE_ROOT = original_root
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        updated = manifest["gates"][0]
        self.assertEqual(updated["telemetry_status"], "invalid")
        self.assertEqual(updated["deterministic_finishes"], 7)
        self.assertEqual(updated["deterministic_best_finish_time_ms"], 24_900)
        self.assertEqual(updated["deterministic_mean_finish_time_ms"], 25_000.0)
        self.assertEqual(updated["stage2_decision"], "instrumentation_required")


if __name__ == "__main__":
    unittest.main()
