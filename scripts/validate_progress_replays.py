"""Round-trip saved TMInterface input replays and record playback telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, LiveTmiSession
from trackmania_rl.observations import ObservationDiagnostics, ReferencePath, build_observation
from trackmania_rl.tmi_bridge import ProtocolError
from trackmania_rl.video_capture import restart_trackmania_race

DEFAULT_CATALOG = (
    WORKSPACE_ROOT / "artifacts" / "replays" / "progression" / "capture_catalog.json"
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "artifacts" / "replays" / "progression" / "round_trip"
)
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"
REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"
DEFAULT_CASE_IDS = (
    "v0_phase1_smoke_2048:ep1",
    "v1_sparse_final:ep1",
    "v2_speed_final:ep4",
    "v2_speed_final:ep1",
)


@dataclass(frozen=True)
class ReplayCase:
    case_id: str
    stage_id: str
    episode: int
    input_replay: Path
    input_replay_sha256: str
    expected_outcome: str
    expected_elapsed_ms: int
    expected_maximum_progress: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load preserved inputs back into TMInterface and verify outcomes."
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--all-v2", action="store_true")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--simulation-speed", type=float, default=6.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def outcome_name(episode: dict[str, Any]) -> str:
    if bool(episode["finished"]):
        return "finish"
    if bool(episode["fallen"]):
        return "fall"
    if bool(episode["off_track"]):
        return "off_track"
    if bool(episode["timeout"]):
        return "timeout"
    raise ValueError(f"episode has no terminal outcome: {episode}")


def load_cases(catalog_path: Path) -> dict[str, ReplayCase]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    cases: dict[str, ReplayCase] = {}
    for take in catalog["successful_takes"]:
        manifest_path = WORKSPACE_ROOT / str(take["manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stage_id = str(manifest["stage"]["id"])
        for episode in manifest["episodes"]:
            episode_number = int(episode["episode"]) + 1
            case_id = f"{stage_id}:ep{episode_number}"
            if case_id in cases:
                raise ValueError(f"duplicate replay case in catalog: {case_id}")
            cases[case_id] = ReplayCase(
                case_id=case_id,
                stage_id=stage_id,
                episode=episode_number,
                input_replay=WORKSPACE_ROOT / str(episode["input_replay"]),
                input_replay_sha256=str(episode["input_replay_sha256"]),
                expected_outcome=outcome_name(episode),
                expected_elapsed_ms=int(episode["elapsed_ms"]),
                expected_maximum_progress=float(episode["maximum_progress"]),
            )
    return cases


def select_cases(
    available: dict[str, ReplayCase],
    requested: list[str] | None,
    *,
    all_v2: bool,
) -> list[ReplayCase]:
    if requested and all_v2:
        raise ValueError("--case and --all-v2 cannot be combined")
    if all_v2:
        selected_ids = sorted(
            case_id for case_id in available if case_id.startswith("v2_")
        )
    else:
        selected_ids = list(requested or DEFAULT_CASE_IDS)
    missing = [case_id for case_id in selected_ids if case_id not in available]
    if missing:
        raise ValueError(f"replay catalog does not contain cases: {missing}")
    return [available[case_id] for case_id in selected_ids]


def input_file_stats(path: Path) -> dict[str, Any]:
    command_counts: dict[str, int] = {}
    maximum_time = 0.0
    lines = 0
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"invalid replay line {line_number} in {path}: {line!r}")
        try:
            timestamp = float(parts[0])
        except ValueError as error:
            raise ValueError(
                f"invalid timestamp on replay line {line_number}: {parts[0]!r}"
            ) from error
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise ValueError(f"invalid replay timestamp on line {line_number}")
        command = parts[1].lower()
        command_counts[command] = command_counts.get(command, 0) + 1
        maximum_time = max(maximum_time, timestamp)
        lines += 1
    if lines == 0:
        raise ValueError(f"replay contains no commands: {path}")
    return {
        "command_lines": lines,
        "command_counts": command_counts,
        "maximum_command_time_seconds": maximum_time,
    }


def telemetry_record(
    state: object,
    reference_path: ReferencePath,
    *,
    race_finished: bool,
) -> dict[str, Any]:
    position = np.asarray(state.position, dtype=np.float64)
    velocity = np.asarray(state.velocity, dtype=np.float64)
    rotation = np.asarray(state.rotation_matrix, dtype=np.float64)
    values = np.concatenate((position, velocity, rotation.ravel()))
    if not np.isfinite(values).all():
        raise ProtocolError("replay playback produced nonfinite telemetry")
    _, diagnostics = build_observation(state, reference_path)
    return {
        "race_time_ms": int(state.race_time),
        "display_speed": int(state.display_speed),
        "position": position.tolist(),
        "velocity": velocity.tolist(),
        "rotation_matrix": rotation.tolist(),
        "upright_cosine": float(rotation[1, 1]),
        "progress": diagnostics.progress,
        "lateral_offset": diagnostics.lateral_offset,
        "vertical_offset": diagnostics.vertical_offset,
        "heading_error": diagnostics.heading_error,
        "race_finished": race_finished,
    }


def terminal_outcome(record: dict[str, Any]) -> str | None:
    if bool(record["race_finished"]):
        return "finish"
    if float(record["vertical_offset"]) < -10.0:
        return "fall"
    if abs(float(record["lateral_offset"])) > 50.0:
        return "off_track"
    if int(record["race_time_ms"]) >= 45_000:
        return "timeout"
    return None


def compare_playback(
    case: ReplayCase,
    *,
    observed_outcome: str,
    observed_elapsed_ms: int,
    observed_maximum_progress: float,
) -> dict[str, Any]:
    elapsed_delta = observed_elapsed_ms - case.expected_elapsed_ms
    progress_delta = observed_maximum_progress - case.expected_maximum_progress
    checks = {
        "outcome_matches": observed_outcome == case.expected_outcome,
        "elapsed_within_100_ms": abs(elapsed_delta) <= 100,
        "maximum_progress_within_5_units": abs(progress_delta) <= 5.0,
    }
    return {
        "expected_outcome": case.expected_outcome,
        "observed_outcome": observed_outcome,
        "expected_elapsed_ms": case.expected_elapsed_ms,
        "observed_elapsed_ms": observed_elapsed_ms,
        "elapsed_delta_ms": elapsed_delta,
        "expected_maximum_progress": case.expected_maximum_progress,
        "observed_maximum_progress": observed_maximum_progress,
        "maximum_progress_delta": progress_delta,
        "checks": checks,
        "passed": all(checks.values()),
    }


def play_case(
    case: ReplayCase,
    *,
    session: LiveTmiSession,
    reset_diagnostics: ObservationDiagnostics | None,
    tmi_scripts_dir: Path,
    output_dir: Path,
    reference_path: ReferencePath,
) -> dict[str, Any]:
    if not case.input_replay.is_file():
        raise FileNotFoundError(f"preserved replay does not exist: {case.input_replay}")
    local_hash = sha256(case.input_replay)
    if local_hash != case.input_replay_sha256:
        raise ProtocolError(
            f"preserved replay hash mismatch for {case.case_id}: {local_hash}"
        )
    external_replay = tmi_scripts_dir / case.input_replay.name
    if not external_replay.is_file():
        raise FileNotFoundError(
            f"TMInterface Scripts copy is missing for {case.case_id}: {external_replay}"
        )
    external_hash = sha256(external_replay)
    if external_hash != local_hash:
        raise ProtocolError(
            f"TMInterface Scripts copy differs for {case.case_id}: {external_hash}"
        )

    session.client.execute_command("unload")
    session.client.execute_command(f"load {external_replay.name}")
    state = session.reset(reset_diagnostics)
    records: list[dict[str, Any]] = []
    record = telemetry_record(
        state,
        reference_path,
        race_finished=False,
    )
    records.append(record)
    outcome = terminal_outcome(record)
    while outcome is None:
        result = session.advance_playback()
        record = telemetry_record(
            result.state,
            reference_path,
            race_finished=result.race_finished,
        )
        records.append(record)
        outcome = terminal_outcome(record)
        if result.race_time_ms > 45_100:
            raise ProtocolError(
                f"replay {case.case_id} exceeded the 45-second safety limit"
            )

    if not records:
        raise ProtocolError(f"replay {case.case_id} produced no run telemetry")
    observed_outcome = terminal_outcome(records[-1])
    if observed_outcome is None:
        raise ProtocolError(f"replay {case.case_id} ended without a terminal outcome")
    maximum_progress = max(float(record["progress"]) for record in records)
    comparison = compare_playback(
        case,
        observed_outcome=observed_outcome,
        observed_elapsed_ms=int(records[-1]["race_time_ms"]),
        observed_maximum_progress=maximum_progress,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = output_dir / f"{case.stage_id}_ep_{case.episode:02d}.jsonl"
    with telemetry_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            output.write("\n")

    return {
        "case_id": case.case_id,
        "stage_id": case.stage_id,
        "episode": case.episode,
        "input_replay": str(case.input_replay.relative_to(WORKSPACE_ROOT)),
        "input_replay_sha256": local_hash,
        "tmi_scripts_copy": str(external_replay),
        "tmi_scripts_copy_sha256": external_hash,
        "input_stats": input_file_stats(case.input_replay),
        "telemetry": str(telemetry_path.relative_to(WORKSPACE_ROOT)),
        "telemetry_sha256": sha256(telemetry_path),
        "telemetry_records": len(records),
        "shared_snapshot_rewind": True,
        "comparison": comparison,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if not 0.0 < args.simulation_speed <= 1000.0:
        raise SystemExit("--simulation-speed must be in (0, 1000]")
    cases = select_cases(
        load_cases(args.catalog),
        args.case_ids,
        all_v2=args.all_v2,
    )
    reference_path = ReferencePath.from_csv(REFERENCE_PATH)
    results: list[dict[str, Any]] = []
    restart_trackmania_race()
    session = LiveTmiSession(
        EnvironmentConfig(
            port=args.port,
            simulation_speed=args.simulation_speed,
        )
    )
    try:
        start_state = session.prepare()
        _, start_diagnostics = build_observation(start_state, reference_path)
        for index, case in enumerate(cases, start=1):
            result = play_case(
                case,
                session=session,
                reset_diagnostics=start_diagnostics if index == 1 else None,
                tmi_scripts_dir=args.tmi_scripts_dir,
                output_dir=args.output_dir,
                reference_path=reference_path,
            )
            results.append(result)
            comparison = result["comparison"]
            print(
                f"round-trip {index}/{len(cases)} {case.case_id}: "
                f"expected={comparison['expected_outcome']} "
                f"observed={comparison['observed_outcome']} "
                f"elapsed_delta={comparison['elapsed_delta_ms']}ms "
                f"progress_delta={comparison['maximum_progress_delta']:.3f} "
                f"passed={comparison['passed']}",
                flush=True,
            )
    finally:
        try:
            session.client.execute_command("unload")
        except (OSError, RuntimeError):
            pass
        session.close()

    summary = {
        "status": "passed" if all(item["comparison"]["passed"] for item in results) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulation_speed": args.simulation_speed,
        "shared_snapshot_rewind": True,
        "case_count": len(results),
        "cases": results,
    }
    summary_path = args.output_dir / "round_trip_summary.json"
    write_json(summary_path, summary)
    summary_hash = sha256(summary_path)
    print(f"round-trip summary SHA-256={summary_hash}", flush=True)
    if summary["status"] != "passed":
        raise ProtocolError("one or more replay round-trip comparisons failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
