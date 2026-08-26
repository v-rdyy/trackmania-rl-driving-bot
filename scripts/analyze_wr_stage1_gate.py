"""Replay one WR-chase Stage 1 gate and apply the frozen discovery/plateau gates."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import analyze_v4_wr_prerequisites as replay_analysis
from evaluate_wr_stage1 import EXPECTED_EPISODES
from train_reward_v3 import sha256, write_json
from train_wr_stage1 import (
    GATE_SIZE,
    INITIAL_TIMEBOX,
    MANIFEST_PATH,
    RUN_DIR,
    gate_slug,
    validate_target,
)
from trackmania_rl.env import EnvironmentConfig, LiveTmiSession
from trackmania_rl.observations import ReferencePath
from trackmania_rl.tmi_bridge import ProtocolError
from trackmania_rl.video_capture import restart_trackmania_race

REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"
ANALYSIS_ROOT = WORKSPACE_ROOT / "artifacts" / "analysis" / "wr_chase_stage1"
KNOWN_ZONES = {
    "first_turn": (680.0, 930.0),
    "final_corner": (1100.0, 1410.0),
}
CANDIDATE_MIN_SAMPLES = 2
CONFIRMED_MIN_SAMPLES = 3
MIN_DISPLAY_SPEED = 350
MIN_ABS_SLIP_ANGLE_DEGREES = 1.0
MIN_ABS_YAW_RATE = 0.25
MIN_GROUND_CONTACTS = 3
CANDIDATE_MIN_SLIDING_WHEELS = 1
CONFIRMED_MIN_SLIDING_WHEELS = 2
DISCOVERY_EPISODES = 3
PLATEAU_MIN_ADDITIONAL_STEPS = 2_000_000
PLATEAU_GATE_COUNT = 3
SIGNIFICANT_LAP_IMPROVEMENT_MS = 50
STOCHASTIC_WINDOW_IMPROVEMENT_MS = 100
BASELINE_BEST_MS = 24_900
BASELINE_MEAN_MS = 24_930.51020408163


@dataclass(frozen=True)
class GateReplay:
    episode: int
    finished: bool
    terminal_race_time_ms: int
    replay: Path
    replay_sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-target", type=int, required=True)
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--simulation-speed", type=float, default=100.0)
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument("--reuse-game", action="store_true")
    parser.add_argument("--postprocess-existing", action="store_true")
    return parser.parse_args()


def evaluation_summary_path(target: int) -> Path:
    return RUN_DIR / "gates" / f"{gate_slug(target)}_evaluation.json"


def output_dir(target: int) -> Path:
    return ANALYSIS_ROOT / gate_slug(target)


def select_replays(summary: dict[str, Any]) -> list[GateReplay]:
    episodes = summary.get("episodes_detail", [])
    if len(episodes) != EXPECTED_EPISODES:
        raise ProtocolError(
            f"Stage 1 gate has {len(episodes)} episodes, expected {EXPECTED_EPISODES}"
        )
    selected: list[GateReplay] = []
    for episode in episodes:
        replay = WORKSPACE_ROOT / str(episode["input_replay"])
        selected.append(
            GateReplay(
                episode=int(episode["episode"]) + 1,
                finished=bool(episode["finished"]),
                terminal_race_time_ms=int(episode["terminal_race_time_ms"]),
                replay=replay,
                replay_sha256=str(episode["input_replay_sha256"]),
            )
        )
    return selected


def sample_meets_drift_gate(record: dict[str, Any], *, sliding_wheels: int) -> bool:
    return bool(
        int(record["ground_contact_count"]) >= MIN_GROUND_CONTACTS
        and int(record["display_speed"]) >= MIN_DISPLAY_SPEED
        and int(record["sliding_wheel_count"]) >= sliding_wheels
        and abs(float(record["slip_angle_degrees"]))
        >= MIN_ABS_SLIP_ANGLE_DEGREES
        and abs(float(record["body_up_yaw_rate"])) >= MIN_ABS_YAW_RATE
    )


def contiguous_sample_runs(
    records: list[dict[str, Any]],
    *,
    sliding_wheels: int,
    progress_range: tuple[float, float] | None = None,
) -> list[list[dict[str, Any]]]:
    runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for record in records:
        in_range = bool(
            progress_range is None
            or progress_range[0] <= float(record["progress"]) <= progress_range[1]
        )
        meets = in_range and sample_meets_drift_gate(
            record,
            sliding_wheels=sliding_wheels,
        )
        consecutive = bool(
            current
            and 0
            < int(record["race_time_ms"]) - int(current[-1]["race_time_ms"])
            <= 100
        )
        if meets:
            if current and not consecutive:
                runs.append(current)
                current = []
            current.append(record)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def summarize_run(run: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(run),
        "duration_ms": len(run) * 100,
        "start_race_time_ms": int(run[0]["race_time_ms"]),
        "end_race_time_ms": int(run[-1]["race_time_ms"]),
        "start_progress": float(run[0]["progress"]),
        "end_progress": float(run[-1]["progress"]),
        "maximum_display_speed": max(int(record["display_speed"]) for record in run),
        "maximum_abs_slip_angle_degrees": max(
            abs(float(record["slip_angle_degrees"])) for record in run
        ),
        "maximum_abs_body_up_yaw_rate": max(
            abs(float(record["body_up_yaw_rate"])) for record in run
        ),
        "minimum_sliding_wheels": min(
            int(record["sliding_wheel_count"]) for record in run
        ),
    }


def drift_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_zone: dict[str, Any] = {}
    for name, progress_range in KNOWN_ZONES.items():
        candidate_runs = [
            run
            for run in contiguous_sample_runs(
                records,
                sliding_wheels=CANDIDATE_MIN_SLIDING_WHEELS,
                progress_range=progress_range,
            )
            if len(run) >= CANDIDATE_MIN_SAMPLES
        ]
        confirmed_runs = [
            run
            for run in contiguous_sample_runs(
                records,
                sliding_wheels=CONFIRMED_MIN_SLIDING_WHEELS,
                progress_range=progress_range,
            )
            if len(run) >= CONFIRMED_MIN_SAMPLES
        ]
        zone_records = [
            record
            for record in records
            if progress_range[0]
            <= float(record["progress"])
            <= progress_range[1]
        ]
        by_zone[name] = {
            "progress_range": list(progress_range),
            "samples": len(zone_records),
            "maximum_sliding_wheels": max(
                (int(record["sliding_wheel_count"]) for record in zone_records),
                default=0,
            ),
            "maximum_abs_slip_angle_degrees": max(
                (
                    abs(float(record["slip_angle_degrees"]))
                    for record in zone_records
                ),
                default=0.0,
            ),
            "candidate_windows": [summarize_run(run) for run in candidate_runs],
            "confirmed_windows": [summarize_run(run) for run in confirmed_runs],
        }

    track_candidates = [
        run
        for run in contiguous_sample_runs(
            records,
            sliding_wheels=CANDIDATE_MIN_SLIDING_WHEELS,
        )
        if len(run) >= CANDIDATE_MIN_SAMPLES
    ]
    track_confirmed = [
        run
        for run in contiguous_sample_runs(
            records,
            sliding_wheels=CONFIRMED_MIN_SLIDING_WHEELS,
        )
        if len(run) >= CONFIRMED_MIN_SAMPLES
    ]
    return {
        "known_zones": by_zone,
        "track_wide_candidate_windows": [
            summarize_run(run) for run in track_candidates
        ],
        "track_wide_confirmed_windows": [
            summarize_run(run) for run in track_confirmed
        ],
    }


def aggregate_discovery(cases: list[dict[str, Any]]) -> dict[str, Any]:
    known_counts = {
        zone: sum(
            bool(case["drift"]["known_zones"][zone]["confirmed_windows"])
            for case in cases
        )
        for zone in KNOWN_ZONES
    }
    outside_bins: dict[int, set[int]] = defaultdict(set)
    for case in cases:
        for event in case["drift"]["track_wide_confirmed_windows"]:
            midpoint = 0.5 * (
                float(event["start_progress"]) + float(event["end_progress"])
            )
            if any(low <= midpoint <= high for low, high in KNOWN_ZONES.values()):
                continue
            outside_bins[int(midpoint // 100) * 100].add(int(case["episode"]))
    outside_counts = {
        f"{start}-{start + 100}": len(episodes)
        for start, episodes in sorted(outside_bins.items())
    }
    triggers = [
        zone for zone, count in known_counts.items() if count >= DISCOVERY_EPISODES
    ]
    triggers.extend(
        f"outside_{progress_bin}"
        for progress_bin, count in outside_counts.items()
        if count >= DISCOVERY_EPISODES
    )
    return {
        "required_episodes": DISCOVERY_EPISODES,
        "known_zone_confirmed_episode_counts": known_counts,
        "outside_known_zones_confirmed_episode_counts_by_progress_bin": outside_counts,
        "telemetry_discovery_triggered": bool(triggers),
        "trigger_locations": triggers,
        "visual_confirmation_required": bool(triggers),
    }


def lap_metrics(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        "finishes": int(evaluation["finishes"]),
        "finish_rate": float(evaluation["finish_rate"]),
        "best_finish_time_ms": evaluation["best_finish_time_ms"],
        "average_finish_time_ms": evaluation["average_finish_time_ms"],
        "worst_finish_time_ms": evaluation["worst_finish_time_ms"],
    }


def assess_stage1(manifest: dict[str, Any]) -> dict[str, Any]:
    complete = [
        gate
        for gate in manifest.get("gates", [])
        if gate.get("telemetry_status") == "complete"
    ]
    complete.sort(key=lambda gate: int(gate["target_additional_steps"]))
    if not complete:
        return {"decision": "continue", "reason": "no analyzed gate"}
    latest = complete[-1]
    if any(bool(gate.get("telemetry_discovery_triggered")) for gate in complete):
        return {
            "decision": "review_discovery",
            "reason": "telemetry slide repeated in at least 3/10 deterministic episodes",
        }

    prior_best = BASELINE_BEST_MS
    significant_best_by_target: dict[int, bool] = {}
    for gate in complete:
        target = int(gate["target_additional_steps"])
        best = gate.get("deterministic_best_finish_time_ms")
        significant = bool(
            best is not None and prior_best - int(best) >= SIGNIFICANT_LAP_IMPROVEMENT_MS
        )
        significant_best_by_target[target] = significant
        if best is not None:
            prior_best = min(prior_best, int(best))

    target = int(latest["target_additional_steps"])
    recent = complete[-PLATEAU_GATE_COUNT:]
    plateau_checks: dict[str, bool] = {
        "minimum_training_reached": target >= PLATEAU_MIN_ADDITIONAL_STEPS,
        "three_analyzed_gates_available": len(recent) == PLATEAU_GATE_COUNT,
        "no_significant_stage_best_across_three_gates": False,
        "deterministic_mean_improvement_under_50ms": False,
        "stochastic_500_finish_window_improvement_under_100ms": False,
    }
    if len(recent) == PLATEAU_GATE_COUNT:
        plateau_checks["no_significant_stage_best_across_three_gates"] = not any(
            significant_best_by_target[int(gate["target_additional_steps"])]
            for gate in recent
        )
        first_mean = recent[0].get("deterministic_mean_finish_time_ms")
        last_mean = recent[-1].get("deterministic_mean_finish_time_ms")
        plateau_checks["deterministic_mean_improvement_under_50ms"] = bool(
            first_mean is not None
            and last_mean is not None
            and float(first_mean) - float(last_mean) < SIGNIFICANT_LAP_IMPROVEMENT_MS
        )
    training_summary = json.loads(
        (WORKSPACE_ROOT / str(latest["training_summary"])).read_text(
            encoding="utf-8"
        )
    )
    previous_stochastic = training_summary.get(
        "stochastic_finish_time_mean_previous_500_ms"
    )
    current_stochastic = training_summary.get(
        "stochastic_finish_time_mean_last_500_ms"
    )
    plateau_checks["stochastic_500_finish_window_improvement_under_100ms"] = bool(
        previous_stochastic is not None
        and current_stochastic is not None
        and float(previous_stochastic) - float(current_stochastic)
        < STOCHASTIC_WINDOW_IMPROVEMENT_MS
    )
    if all(plateau_checks.values()):
        return {
            "decision": "plateau",
            "reason": "all frozen plateau checks passed",
            "plateau_checks": plateau_checks,
            "significant_stage_best_by_target": significant_best_by_target,
        }

    if target >= INITIAL_TIMEBOX:
        latest_one_million = complete[-2:]
        recent_significant = any(
            significant_best_by_target[int(gate["target_additional_steps"])]
            for gate in latest_one_million
        )
        if not recent_significant:
            return {
                "decision": "timebox_inconclusive",
                "reason": (
                    "initial five-million-step time box ended without the frozen "
                    "recent-improvement condition for extension"
                ),
                "plateau_checks": plateau_checks,
                "significant_stage_best_by_target": significant_best_by_target,
            }
    return {
        "decision": "continue",
        "reason": "no discovery or frozen plateau yet",
        "plateau_checks": plateau_checks,
        "significant_stage_best_by_target": significant_best_by_target,
    }


def finalize_case(
    case: GateReplay,
    records: list[dict[str, Any]],
    *,
    reference: ReferencePath,
    destination: Path,
) -> dict[str, Any]:
    if sha256(case.replay) != case.replay_sha256:
        raise ProtocolError(f"Stage 1 replay hash changed: {case.replay}")
    observed_time = int(records[-1]["race_time_ms"])
    if abs(observed_time - case.terminal_race_time_ms) > 100:
        raise ProtocolError(
            f"episode {case.episode} telemetry time differs by "
            f"{observed_time - case.terminal_race_time_ms}ms"
        )
    observed_finish = bool(records[-1]["race_finished"])
    if case.finished and not observed_finish:
        raise ProtocolError(f"episode {case.episode} lost its recorded finish")
    events = replay_analysis.parse_replay_input_events(
        case.replay.read_text(encoding="utf-8")
    )
    replay_analysis.annotate_replay_inputs(records, events)
    telemetry = destination / f"episode_{case.episode:02d}.jsonl"
    replay_analysis.write_jsonl(telemetry, records)
    prerequisite: dict[str, Any] | None
    try:
        prerequisite = replay_analysis.analyze_records(records, reference)
    except ValueError:
        prerequisite = None
    return {
        "episode": case.episode,
        "expected_finished": case.finished,
        "observed_finished": observed_finish,
        "expected_terminal_race_time_ms": case.terminal_race_time_ms,
        "observed_terminal_race_time_ms": observed_time,
        "input_replay": str(case.replay.relative_to(WORKSPACE_ROOT)),
        "input_replay_sha256": case.replay_sha256,
        "telemetry": str(telemetry.relative_to(WORKSPACE_ROOT)),
        "telemetry_sha256": sha256(telemetry),
        "records": len(records),
        "prerequisites": prerequisite,
        "drift": drift_metrics(records),
    }


def update_manifest(target: int, result_path: Path, result: dict[str, Any]) -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    matching = [
        gate
        for gate in manifest.get("gates", [])
        if int(gate["target_additional_steps"]) == target
    ]
    if len(matching) != 1:
        raise ProtocolError("manifest does not contain exactly one analyzed gate")
    gate = matching[0]
    gate.update(
        {
            "telemetry_status": "complete",
            "telemetry_summary": str(result_path.relative_to(WORKSPACE_ROOT)),
            "telemetry_summary_sha256": sha256(result_path),
            "telemetry_discovery_triggered": bool(
                result["discovery"]["telemetry_discovery_triggered"]
            ),
            **result["lap_metrics"],
        }
    )
    decision = assess_stage1(manifest)
    gate["stage1_decision"] = decision["decision"]
    manifest["latest_stage1_assessment"] = decision
    manifest["status"] = {
        "continue": "ready_for_next_gate",
        "review_discovery": "discovery_requires_visual_review",
        "plateau": "plateau_detected",
        "timebox_inconclusive": "timebox_complete_inconclusive",
    }[decision["decision"]]
    write_json(MANIFEST_PATH, manifest)


def main() -> int:
    args = parse_args()
    validate_target(args.gate_target)
    if args.simulation_speed <= 0 or args.simulation_speed > 1000:
        raise SystemExit("--simulation-speed must be in (0, 1000]")
    evaluation_path = evaluation_summary_path(args.gate_target)
    if not evaluation_path.is_file():
        raise SystemExit(f"Stage 1 evaluation is missing: {evaluation_path}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    cases = select_replays(evaluation)
    destination = output_dir(args.gate_target)
    summary_path = destination / "analysis_summary.json"
    if summary_path.exists() and not args.postprocess_existing:
        raise SystemExit(f"Stage 1 telemetry analysis already exists: {summary_path}")
    destination.mkdir(parents=True, exist_ok=True)
    reference = ReferencePath.from_csv(REFERENCE_PATH)

    results: list[dict[str, Any]] = []
    if args.postprocess_existing:
        for case in cases:
            telemetry = destination / f"episode_{case.episode:02d}.jsonl"
            if not telemetry.is_file():
                raise FileNotFoundError(telemetry)
            records = [
                json.loads(line)
                for line in telemetry.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            results.append(
                finalize_case(
                    case,
                    records,
                    reference=reference,
                    destination=destination,
                )
            )
    else:
        if not args.reuse_game:
            replay_analysis.close_all_trackmania_windows()
            replay_analysis.terminate_verified_tmforever_processes()
        replay_analysis.ensure_trackmania_running(port=args.port, confirm_existing=True)
        restart_trackmania_race()
        session = LiveTmiSession(
            EnvironmentConfig(
                port=args.port,
                simulation_speed=args.simulation_speed,
                max_episode_ms=45_000,
                map_to_load="A01-Race.Challenge.Gbx",
                auto_respawn_on_connect=False,
            )
        )
        try:
            prepared = session.prepare()
            reset_projection = reference.project(prepared.position)
            session.reset(reset_projection)
            for index, case in enumerate(cases, start=1):
                if not case.replay.is_file():
                    raise FileNotFoundError(case.replay)
                if sha256(case.replay) != case.replay_sha256:
                    raise ProtocolError(f"Stage 1 replay hash changed: {case.replay}")
                external_replay = args.tmi_scripts_dir.resolve() / case.replay.name
                if not external_replay.is_file():
                    raise FileNotFoundError(external_replay)
                if sha256(external_replay) != case.replay_sha256:
                    raise ProtocolError(f"external replay hash changed: {external_replay}")
                session.client.execute_command("unload")
                session.client.execute_command(f"load {external_replay.name}")
                state = session.reset(reset_projection if index == 1 else None)
                records = [
                    replay_analysis.state_record(
                        state,
                        reference,
                        race_finished=False,
                    )
                ]
                while int(records[-1]["race_time_ms"]) < case.terminal_race_time_ms:
                    step = session.advance_playback()
                    records.append(
                        replay_analysis.state_record(
                            step.state,
                            reference,
                            race_finished=step.race_finished,
                        )
                    )
                    if int(records[-1]["race_time_ms"]) > 45_100:
                        raise ProtocolError("Stage 1 replay exceeded safety limit")
                results.append(
                    finalize_case(
                        case,
                        records,
                        reference=reference,
                        destination=destination,
                    )
                )
                print(
                    f"analyzed Stage 1 {gate_slug(args.gate_target)} "
                    f"episode {index}/{len(cases)}",
                    flush=True,
                )
        finally:
            try:
                session.client.execute_command("unload")
            except (OSError, RuntimeError):
                pass
            session.close()
            if not args.reuse_game:
                replay_analysis.terminate_verified_tmforever_processes()

    result = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gate_target_additional_steps": args.gate_target,
        "evaluation_summary": str(evaluation_path.relative_to(WORKSPACE_ROOT)),
        "evaluation_summary_sha256": sha256(evaluation_path),
        "reference_path": str(REFERENCE_PATH.relative_to(WORKSPACE_ROOT)),
        "reference_path_sha256": sha256(REFERENCE_PATH),
        "measurement_protocol": {
            "step_period_ms": 100,
            "known_zones": {name: list(bounds) for name, bounds in KNOWN_ZONES.items()},
            "candidate_min_samples": CANDIDATE_MIN_SAMPLES,
            "confirmed_min_samples": CONFIRMED_MIN_SAMPLES,
            "minimum_display_speed": MIN_DISPLAY_SPEED,
            "minimum_abs_slip_angle_degrees": MIN_ABS_SLIP_ANGLE_DEGREES,
            "minimum_abs_body_up_yaw_rate": MIN_ABS_YAW_RATE,
            "minimum_ground_contacts": MIN_GROUND_CONTACTS,
            "candidate_minimum_sliding_wheels": CANDIDATE_MIN_SLIDING_WHEELS,
            "confirmed_minimum_sliding_wheels": CONFIRMED_MIN_SLIDING_WHEELS,
        },
        "lap_metrics": lap_metrics(evaluation),
        "cases": results,
        "discovery": aggregate_discovery(results),
    }
    write_json(summary_path, result)
    update_manifest(args.gate_target, summary_path, result)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    print(f"analysis summary SHA-256={sha256(summary_path)}", flush=True)
    print(
        f"Stage 1 decision={manifest['latest_stage1_assessment']['decision']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
