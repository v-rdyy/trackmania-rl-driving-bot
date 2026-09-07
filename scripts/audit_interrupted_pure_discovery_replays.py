"""Replay preserved interrupted-run inputs and score live SimState slide telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, LiveTmiSession, TrackmaniaEnv
from trackmania_rl.game_launch import close_trackmania, ensure_trackmania_running
from trackmania_rl.observations import ReferencePath, signed_heading_error
from trackmania_rl.slide_scoring import score_episode
from trackmania_rl.tmi_bridge import ProtocolError


DEFAULT_RUN = (
    WORKSPACE_ROOT / "runs" / "wr_pure_continuous_verified2x_20260905_045450"
)
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"
DEFAULT_REFERENCE = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--map-to-load", default="A01-Race.Challenge.Gbx")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--simulation-speed", type=float, default=2.0)
    parser.add_argument("--reuse-game", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_replays(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "replays.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ProtocolError(f"interrupted run has no preserved replays: {path}")
    episodes = [int(row["episode"]) for row in rows]
    if len(episodes) != len(set(episodes)):
        raise ProtocolError("interrupted-run replay catalog has duplicate episodes")
    return rows


def verify_replay(
    row: dict[str, Any], tmi_scripts_dir: Path
) -> tuple[Path, Path, str]:
    local = Path(str(row["replay"])).resolve()
    if not local.is_file():
        raise FileNotFoundError(local)
    expected_hash = str(row["sha256"]).upper()
    local_hash = sha256(local)
    if local_hash != expected_hash:
        raise ProtocolError(f"preserved replay hash mismatch: {local.name}")
    external = tmi_scripts_dir / local.name
    if not external.is_file() or sha256(external) != local_hash:
        raise ProtocolError(f"TMInterface replay copy mismatch: {external}")
    return local, external, local_hash


def state_record(
    state: object,
    reference: ReferencePath,
    *,
    race_finished: bool,
    episode: int,
) -> dict[str, Any]:
    position = np.asarray(state.position, dtype=np.float64)
    rotation = np.asarray(state.rotation_matrix, dtype=np.float64)
    if position.shape != (3,) or rotation.shape != (3, 3):
        raise ProtocolError("live replay state has invalid position or rotation shape")
    dynamics = TrackmaniaEnv._full_simstate_log(state)
    if dynamics.get("full_simstate_available") is not True:
        raise ProtocolError("live replay state is missing complete SimState dynamics")
    projection = reference.project(position)
    car_forward_xz = rotation[[0, 2], 2]
    heading_error = signed_heading_error(projection.tangent_xz, car_forward_xz)
    scalars = (
        float(projection.progress),
        float(projection.lateral_offset),
        float(projection.vertical_offset),
        float(heading_error),
        float(rotation[1, 1]),
    )
    if not all(math.isfinite(value) for value in scalars):
        raise ProtocolError("live replay produced nonfinite projected telemetry")
    return {
        "episode": episode,
        "race_time_ms": int(state.race_time),
        "race_finished": bool(race_finished),
        "display_speed": int(state.display_speed),
        "position": position.tolist(),
        "progress": float(projection.progress),
        "lateral_offset": float(projection.lateral_offset),
        "vertical_offset": float(projection.vertical_offset),
        "heading_error": float(heading_error),
        "upright_cosine": float(rotation[1, 1]),
        "timeout": False,
        "off_track": False,
        "fallen": False,
        "stuck": False,
        "truncated": False,
        **dynamics,
    }


def annotate_progress(records: list[dict[str, Any]]) -> None:
    high_water = float(records[0]["progress"])
    previous = high_water
    for record in records:
        progress = float(record["progress"])
        record["previous_progress"] = previous
        record["new_high_water_progress_delta"] = max(0.0, progress - high_water)
        high_water = max(high_water, progress)
        previous = progress


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            output.write("\n")


def replay_one(
    row: dict[str, Any],
    *,
    session: LiveTmiSession,
    reset_diagnostics: object,
    reference: ReferencePath,
    tmi_scripts_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    local, external, replay_hash = verify_replay(row, tmi_scripts_dir)
    episode = int(row["episode"])
    expected_finished = bool(row["finished"])
    expected_terminal_ms = int(row["race_time_ms"])
    session.client.execute_command("unload")
    session.client.execute_command(f"load {external.name}")
    state = session.reset(reset_diagnostics)
    records = [
        state_record(
            state,
            reference,
            race_finished=False,
            episode=episode,
        )
    ]
    while True:
        step = session.advance_playback()
        records.append(
            state_record(
                step.state,
                reference,
                race_finished=step.race_finished,
                episode=episode,
            )
        )
        if step.race_finished or step.race_time_ms >= expected_terminal_ms:
            break
        if step.race_time_ms > expected_terminal_ms + 200:
            raise ProtocolError(f"replay episode {episode} exceeded its terminal bound")
    if not records:
        raise ProtocolError(f"replay episode {episode} produced no records")
    if not expected_finished:
        records[-1]["truncated"] = True
    annotate_progress(records)
    score = score_episode(records)
    telemetry_path = output_dir / f"episode_{episode:08d}.live_simstate.jsonl"
    write_jsonl(telemetry_path, records)
    observed_finished = bool(records[-1]["race_finished"])
    observed_terminal_ms = int(records[-1]["race_time_ms"])
    fidelity = {
        "finish_matches": observed_finished == expected_finished,
        "terminal_time_delta_ms": observed_terminal_ms - expected_terminal_ms,
        "terminal_time_within_100ms": (
            abs(observed_terminal_ms - expected_terminal_ms) <= 100
        ),
    }
    fidelity["passed"] = bool(
        fidelity["finish_matches"] and fidelity["terminal_time_within_100ms"]
    )
    return {
        "episode": episode,
        "additional_interactions": int(row["additional_interactions"]),
        "input_replay": str(local.relative_to(WORKSPACE_ROOT)),
        "input_replay_sha256": replay_hash,
        "expected": {
            "finished": expected_finished,
            "terminal_race_time_ms": expected_terminal_ms,
        },
        "observed": {
            "finished": observed_finished,
            "terminal_race_time_ms": observed_terminal_ms,
            "maximum_progress": max(float(record["progress"]) for record in records),
        },
        "fidelity": fidelity,
        "telemetry": str(telemetry_path.relative_to(WORKSPACE_ROOT)),
        "telemetry_sha256": sha256(telemetry_path),
        "telemetry_records": len(records),
        "slide_score": score,
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (run_dir / "surviving_replay_live_audit")
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite live audit: {output_dir}")
    if not math.isfinite(args.simulation_speed) or args.simulation_speed <= 0.0:
        raise ValueError("simulation speed must be positive and finite")
    rows = load_replays(run_dir)
    tmi_scripts_dir = args.tmi_scripts_dir.resolve()
    reference = ReferencePath.from_csv(args.reference_path.resolve())
    for row in rows:
        verify_replay(row, tmi_scripts_dir)

    if not args.reuse_game:
        close_trackmania()
    _, launched = ensure_trackmania_running(
        port=args.port,
        confirm_existing=not args.reuse_game,
    )
    print(f"TrackMania ready (launched={launched})", flush=True)
    session = LiveTmiSession(
        EnvironmentConfig(
            port=args.port,
            simulation_speed=args.simulation_speed,
            step_period_ms=100,
            max_episode_ms=45_000,
            map_to_load=None if args.reuse_game else args.map_to_load,
            auto_respawn_on_connect=False,
            wait_for_race_start_on_connect=True,
            bridge_response_timeout_ms=90_000,
        )
    )
    results: list[dict[str, Any]] = []
    try:
        prepared = session.prepare()
        reset_diagnostics = reference.project(prepared.position)
        session.reset(reset_diagnostics)
        output_dir.mkdir(parents=True)
        for row in rows:
            result = replay_one(
                row,
                session=session,
                reset_diagnostics=reset_diagnostics,
                reference=reference,
                tmi_scripts_dir=tmi_scripts_dir,
                output_dir=output_dir,
            )
            results.append(result)
            final_score = result["slide_score"]["final_corner"]
            print(
                f"episode {result['episode']}: "
                f"fidelity={result['fidelity']['passed']} "
                f"accepted_onsets={final_score['accepted_slide_onset_count']}",
                flush=True,
            )
    finally:
        session.close()

    summary = {
        "status": "complete",
        "scope": (
            "Five selected input trajectories preserved before the host shutdown; "
            "not a census of all interrupted-run episodes."
        ),
        "run": str(run_dir.relative_to(WORKSPACE_ROOT)),
        "simulation_speed": args.simulation_speed,
        "source": "TMInterface input playback sampled from live SimState",
        "all_replays_passed_fidelity": all(
            bool(result["fidelity"]["passed"]) for result in results
        ),
        "episodes_with_accepted_final_corner_slide_onset": sum(
            int(result["slide_score"]["final_corner"]["accepted_slide_onset_count"] > 0)
            for result in results
        ),
        "episodes_with_accepted_first_turn_slide_onset": sum(
            int(
                result["slide_score"]["first_turn_negative_control"]
                ["accepted_slide_onset_count"]
                > 0
            )
            for result in results
        ),
        "episodes": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
