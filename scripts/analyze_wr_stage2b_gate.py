"""Audit one Stage 2b gate and apply the frozen onset/safety stop rules."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import analyze_wr_stage2_gate as base  # noqa: E402
import train_wr_stage2b as training  # noqa: E402
from trackmania_rl.evaluation_metrics import aggregate_precision_metrics  # noqa: E402
from trackmania_rl.observations import ObservationDiagnostics, ReferencePath  # noqa: E402
from trackmania_rl.rewards import (  # noqa: E402
    RewardTransition,
    graduated_final_corner_assistance_reward,
    graduated_final_corner_precursor_bonus,
)
from trackmania_rl.slide_scoring import (  # noqa: E402
    FINAL_CORNER_ZONE,
    FIRST_TURN_ZONE,
    valid_slide_onset_sample,
    score_episode,
)

RUN_DIR = training.RUN_DIR
GATE_DIR = RUN_DIR / "gates"
MANIFEST_PATH = training.MANIFEST_PATH
ANALYSIS_ROOT = WORKSPACE_ROOT / "artifacts" / "analysis" / "wr_chase_stage2b"
REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"
EXPECTED_EPISODES = 10
REWARD_TOLERANCE = 1e-6


def configure_base() -> None:
    replacements = {
        "RUN_DIR": RUN_DIR,
        "GATE_DIR": GATE_DIR,
        "MANIFEST_PATH": MANIFEST_PATH,
        "ANALYSIS_ROOT": ANALYSIS_ROOT,
        "REFERENCE_PATH": REFERENCE_PATH,
        "EXPECTED_EPISODES": EXPECTED_EPISODES,
        "GATE_SIZE": training.GATE_SIZE,
        "TIMEBOX": training.TIMEBOX,
        "EXPECTED_INITIAL_CHECKPOINT_SHA256": training.EXPECTED_INITIAL_CHECKPOINT_SHA256,
        "EXPECTED_REWARD_FUNCTION": training.REWARD_FUNCTION_NAME,
    }
    for name, value in replacements.items():
        setattr(base, name, value)


def gate_slug(target: int) -> str:
    return f"gate_{target:08d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-target", type=int, required=True)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def transition_from_record(record: dict[str, Any]) -> RewardTransition:
    current = ObservationDiagnostics(
        progress=float(record["progress"]),
        lateral_offset=float(record["lateral_offset"]),
        heading_error=float(record["heading_error"]),
        segment_index=0,
        vertical_offset=float(record["vertical_offset"]),
    )
    previous = ObservationDiagnostics(
        progress=float(record["previous_progress"]),
        lateral_offset=current.lateral_offset,
        heading_error=current.heading_error,
        segment_index=0,
        vertical_offset=current.vertical_offset,
    )
    previous_speed = record.get("previous_display_speed")
    if previous_speed is not None:
        previous_speed = int(previous_speed)
    return RewardTransition(
        previous_diagnostics=previous,
        diagnostics=current,
        display_speed=int(record["display_speed"]),
        elapsed_ms=100,
        terminated=bool(record["terminated"]),
        truncated=bool(record["truncated"]),
        timed_out=bool(record.get("timeout", False)),
        off_track=bool(record.get("off_track", False)),
        fallen=bool(record.get("fallen", False)),
        stuck=bool(record.get("stuck", False)),
        previous_display_speed=previous_speed,
        upright_cosine=float(record["upright_cosine"]),
        new_high_water_progress_delta=float(
            record["new_high_water_progress_delta"]
        ),
        full_simstate_available=bool(record["full_simstate_available"]),
        ground_contact_count=int(record["ground_contact_count"]),
        sliding_wheel_count=int(record["sliding_wheel_count"]),
        slip_angle_degrees=float(record["slip_angle_degrees"]),
        body_up_yaw_rate=float(record["body_up_yaw_rate"]),
    )


def precursor_score(transition: RewardTransition) -> float:
    slip = min(abs(transition.slip_angle_degrees), 1.0)
    wheels = min(transition.sliding_wheel_count / 2.0, 1.0)
    return (slip + wheels) / 2.0


def reward_and_precursor_audit(
    grouped: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    maximum_error = 0.0
    bonus_total = 0.0
    rewarded_steps = 0
    eligible_precursors: list[float] = []
    final_zone_slips: list[float] = []
    final_zone_yaws: list[float] = []
    final_zone_sliding_samples = 0
    first_turn_sliding_samples = 0
    for records in grouped.values():
        for record in records:
            transition = transition_from_record(record)
            recomputed = graduated_final_corner_assistance_reward(transition)
            error = abs(recomputed - float(record["reward"]))
            maximum_error = max(maximum_error, error)
            bonus = graduated_final_corner_precursor_bonus(transition)
            bonus_total += bonus
            rewarded_steps += int(bonus > 0.0)
            progress = transition.diagnostics.progress
            if FINAL_CORNER_ZONE[0] <= progress <= FINAL_CORNER_ZONE[1]:
                final_zone_slips.append(abs(transition.slip_angle_degrees))
                final_zone_yaws.append(abs(transition.body_up_yaw_rate))
                final_zone_sliding_samples += int(
                    transition.sliding_wheel_count >= 1
                )
                if valid_slide_onset_sample(record):
                    eligible_precursors.append(precursor_score(transition))
            if FIRST_TURN_ZONE[0] <= progress <= FIRST_TURN_ZONE[1]:
                first_turn_sliding_samples += int(
                    transition.sliding_wheel_count >= 1
                )
    return {
        "valid": maximum_error <= REWARD_TOLERANCE,
        "reward_tolerance": REWARD_TOLERANCE,
        "maximum_absolute_reward_error": maximum_error,
        "graduated_bonus_total": bonus_total,
        "rewarded_step_count": rewarded_steps,
        "eligible_precursor_sample_count": len(eligible_precursors),
        "precursor_score_p50": percentile(eligible_precursors, 0.50),
        "precursor_score_p95": percentile(eligible_precursors, 0.95),
        "precursor_score_maximum": max(eligible_precursors, default=None),
        "final_zone_abs_slip_degrees_p50": percentile(final_zone_slips, 0.50),
        "final_zone_abs_slip_degrees_p95": percentile(final_zone_slips, 0.95),
        "final_zone_abs_slip_degrees_maximum": max(final_zone_slips, default=None),
        "final_zone_abs_yaw_rate_p50": percentile(final_zone_yaws, 0.50),
        "final_zone_abs_yaw_rate_p95": percentile(final_zone_yaws, 0.95),
        "final_zone_abs_yaw_rate_maximum": max(final_zone_yaws, default=None),
        "final_zone_sliding_wheel_samples": final_zone_sliding_samples,
        "first_turn_negative_control_sliding_wheel_samples": first_turn_sliding_samples,
    }


def failure_signature(
    details: dict[int, dict[str, Any]],
    active: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    signatures: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for episode, detail in details.items():
        if bool(detail["finished"]):
            continue
        final = active[episode][-1]
        causes = [
            name
            for name in ("timeout", "off_track", "fallen", "stuck")
            if bool(detail.get(name, False))
        ]
        cause = "+".join(causes) or "other"
        progress_bin = int(math.floor(float(final["progress"]) / 25.0) * 25)
        signature = f"{cause}@progress_{progress_bin}_{progress_bin + 25}"
        signatures[signature] += 1
        rows.append(
            {
                "episode": episode,
                "cause": cause,
                "terminal_progress": float(final["progress"]),
                "terminal_lateral_offset": float(final["lateral_offset"]),
                "signature": signature,
            }
        )
    repeated = [
        {"signature": signature, "episodes": count}
        for signature, count in signatures.items()
        if count >= 3
    ]
    return {"failures": rows, "repeated_fixed_location_failures": repeated}


def onset_summary(
    active: dict[int, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes = [score_episode(active[index]) for index in range(EXPECTED_EPISODES)]
    accepted = [
        score
        for score in episodes
        if int(score["final_corner"]["accepted_slide_onset_count"]) > 0
    ]
    raw = [
        score
        for score in episodes
        if int(score["final_corner"]["raw_slide_onset_count"]) > 0
    ]
    return episodes, {
        "raw_onset_episode_count": len(raw),
        "accepted_onset_episode_count": len(accepted),
        "raw_onset_window_count": sum(
            int(score["final_corner"]["raw_slide_onset_count"])
            for score in episodes
        ),
        "accepted_onset_window_count": sum(
            int(score["final_corner"]["accepted_slide_onset_count"])
            for score in episodes
        ),
        "above_envelope_near_miss_count": sum(
            int(score["final_corner"]["above_envelope_near_miss_count"])
            for score in episodes
        ),
        "legacy_candidate_count": sum(
            int(score["final_corner"]["legacy_candidate_count"])
            for score in episodes
        ),
        "legacy_confirmed_count": sum(
            int(score["final_corner"]["legacy_confirmed_count"])
            for score in episodes
        ),
    }


def trend_against_prior(
    manifest: dict[str, Any], target: int, onset_count: int, precursor_p95: float | None
) -> dict[str, Any]:
    if target == 0:
        return {"onset": "baseline", "precursor": "baseline"}
    prior = manifest["gates"][-2]
    prior_onset = int(prior.get("accepted_onset_episode_count", 0))
    onset = "up" if onset_count > prior_onset else "down" if onset_count < prior_onset else "flat"
    prior_precursor = prior.get("precursor_score_p95")
    if precursor_p95 is None or prior_precursor is None:
        precursor = "unavailable"
    else:
        delta = precursor_p95 - float(prior_precursor)
        precursor = "up" if delta > 0.01 else "down" if delta < -0.01 else "flat"
    return {
        "onset": onset,
        "precursor": precursor,
        "prior_target": int(prior["target_additional_steps"]),
        "prior_accepted_onset_episodes": prior_onset,
        "prior_precursor_score_p95": prior_precursor,
    }


def decision_for(
    target: int,
    finishes: int,
    onset_count: int,
    failures: dict[str, Any],
    instrumentation_valid: bool,
) -> dict[str, str]:
    if not instrumentation_valid:
        return {"decision": "instrumentation_required", "reason": "live reward audit failed"}
    if target == 0:
        if onset_count >= 3:
            return {"decision": "baseline_comparability_review", "reason": "untouched base already meets the induction gate"}
        return {"decision": "continue", "reason": "valid Stage 2b target-zero baseline completed"}
    if finishes <= 7:
        return {"decision": "safety_review", "reason": f"finish rate hit the frozen safety pause ({finishes}/10)"}
    if failures["repeated_fixed_location_failures"]:
        return {"decision": "safety_review", "reason": "at least 3/10 episodes share a fixed-location failure signature"}
    if onset_count >= 3:
        return {"decision": "review_induction", "reason": f"accepted onset occurred in {onset_count}/10 episodes"}
    if target >= training.TIMEBOX:
        return {"decision": "ineffective_within_timebox", "reason": "no repeatable induction within 500,000 interactions"}
    return {"decision": "continue", "reason": "onset and safety pause thresholds remain clear"}


def main() -> int:
    args = parse_args()
    configure_base()
    base.validate_target(args.gate_target)
    evaluation_path = base.evaluation_summary_path(args.gate_target).resolve()
    action_path = base.evaluation_action_log_path(args.gate_target).resolve()
    output_path = base.output_path(args.gate_target).resolve()
    if output_path.exists():
        raise SystemExit(f"Stage 2b analysis already exists: {output_path}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    base.validate_pending_analysis_request(manifest, args.gate_target, output_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    base.validate_evidence_binding(
        manifest,
        args.gate_target,
        evaluation_path,
        action_path,
        evaluation,
    )
    raw_records = [
        json.loads(line)
        for line in action_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    details = base._episode_details(evaluation)
    grouped, active = base.group_live_records(raw_records, evaluation)
    reference = ReferencePath.from_csv(REFERENCE_PATH)
    for episode in range(EXPECTED_EPISODES):
        base.validate_reference_diagnostics(grouped[episode], reference)
        for index, record in enumerate(grouped[episode]):
            if "previous_display_speed" not in record:
                raise base.AnalysisError("Stage 2b row is missing previous_display_speed")
            previous_speed = record["previous_display_speed"]
            if previous_speed is not None and (
                not math.isfinite(float(previous_speed)) or int(previous_speed) < 0
            ):
                raise base.AnalysisError("previous_display_speed is invalid")
            if index > 0:
                expected_speed = (
                    None
                    if bool(record.get("race_clock_boundary", False))
                    else int(grouped[episode][index - 1]["display_speed"])
                )
                if previous_speed != expected_speed:
                    raise base.AnalysisError(
                        "previous_display_speed breaks live continuity"
                    )

    reward_audit = reward_and_precursor_audit(grouped)
    episode_scores, onset = onset_summary(active)
    lap = base.lap_metrics(evaluation, details)
    precision_cases = [
        base.precision_metrics(active[episode]) for episode in range(EXPECTED_EPISODES)
    ]
    precision = aggregate_precision_metrics(precision_cases)
    failures = failure_signature(details, active)
    trend = trend_against_prior(
        manifest,
        args.gate_target,
        onset["accepted_onset_episode_count"],
        reward_audit["precursor_score_p95"],
    )
    decision = decision_for(
        args.gate_target,
        int(lap["finishes"]),
        int(onset["accepted_onset_episode_count"]),
        failures,
        bool(reward_audit["valid"]),
    )
    result = {
        "status": "complete" if reward_audit["valid"] else "invalid",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_target_additional_steps": args.gate_target,
        "instrumentation_valid": bool(reward_audit["valid"]),
        "measurement_source": "direct_live_evaluation_simstate",
        "evaluation_summary": base.display_path(evaluation_path),
        "evaluation_summary_sha256": base.sha256(evaluation_path),
        "evaluation_action_log": base.display_path(action_path),
        "evaluation_action_log_sha256": base.sha256(action_path),
        "reward_and_precursor_audit": reward_audit,
        "lap_metrics": lap,
        "onset": onset,
        "onset_trend": trend,
        "episode_scores": episode_scores,
        "failures": failures,
        "precision": precision,
        "decision": decision,
        "comparison_ms": {
            "owner_pb": 24_500,
            "world_record_benchmark": 23_770,
            "best_gap_to_owner_pb": None if lap["best_finish_time_ms"] is None else int(lap["best_finish_time_ms"]) - 24_500,
            "best_gap_to_world_record_benchmark": None if lap["best_finish_time_ms"] is None else int(lap["best_finish_time_ms"]) - 23_770,
        },
    }
    base.write_json(output_path, result)

    gate = manifest["gates"][-1]
    gate.update(
        {
            "telemetry_status": "complete" if reward_audit["valid"] else "invalid",
            "live_analysis_status": "complete" if reward_audit["valid"] else "invalid",
            "telemetry_conclusion_valid": bool(reward_audit["valid"]),
            "telemetry_summary": str(output_path.relative_to(WORKSPACE_ROOT)),
            "telemetry_summary_sha256": base.sha256(output_path),
            "accepted_onset_episode_count": onset["accepted_onset_episode_count"],
            "raw_onset_episode_count": onset["raw_onset_episode_count"],
            "precursor_score_p95": reward_audit["precursor_score_p95"],
            "graduated_bonus_total": reward_audit["graduated_bonus_total"],
            "onset_trend": trend,
            "stage2_decision": decision["decision"],
        }
    )
    manifest["latest_stage2b_assessment"] = decision
    manifest["latest_stage2_assessment"] = decision
    manifest["status"] = {
        "continue": "ready_for_next_gate",
        "review_induction": "induction_requires_visual_review",
        "baseline_comparability_review": "baseline_comparability_review_required",
        "instrumentation_required": "instrumentation_required",
        "safety_review": "safety_review_required",
        "ineffective_within_timebox": "bonus_ineffective_within_timebox",
    }[decision["decision"]]
    base.write_json(MANIFEST_PATH, manifest)
    print(f"Stage 2b analysis SHA-256={base.sha256(output_path)}", flush=True)
    print(
        f"Stage 2b gate={args.gate_target} finishes={lap['finishes']}/10 "
        f"accepted_onset={onset['accepted_onset_episode_count']}/10 "
        f"onset_trend={trend['onset']} precursor_trend={trend['precursor']} "
        f"decision={decision['decision']}",
        flush=True,
    )
    return 0 if reward_audit["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
