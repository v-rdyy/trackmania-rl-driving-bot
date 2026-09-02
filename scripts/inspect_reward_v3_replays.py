"""Render preserved input replays against an explicit map reference."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, LiveTmiSession
from trackmania_rl.game_launch import close_trackmania, ensure_trackmania_running
from trackmania_rl.observations import ReferencePath
from trackmania_rl.tmi_bridge import ProtocolError
from trackmania_rl.video_capture import ProgressVideoRecorder, restart_trackmania_race, sha256


DEFAULT_REFERENCE_PATH = WORKSPACE_ROOT / "data" / "tracks" / "a01_reference_path.csv"
DEFAULT_REPLAY_DIR = WORKSPACE_ROOT / "artifacts" / "replays" / "reward_v3_evaluation"
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "artifacts" / "videos" / "reward_v3_final_jump"
DEFAULT_MAX_RACE_MS = 45_000


def replay_experiment_label(stem: str) -> str:
    match = next(
        (
            version.upper()
            for version in ("v6", "v5", "v4", "v3", "v2", "v1", "v0")
            if stem.startswith(f"reward_{version}_")
        ),
        None,
    )
    return match or "TrackMania"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path)
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--simulation-speed", type=float, default=1.0)
    parser.add_argument("--max-race-ms", type=int, default=DEFAULT_MAX_RACE_MS)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--map-to-load", default="A01-Race.Challenge.Gbx")
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reuse-game", action="store_true")
    parser.add_argument(
        "--video-only-after-disconnect",
        action="store_true",
        help=(
            "continue clean capture through the known replay horizon if "
            "loading inputs closes the telemetry bridge"
        ),
    )
    return parser.parse_args()


def telemetry_record(state: object, reference: ReferencePath, race_time_ms: int) -> dict[str, Any]:
    projection = reference.project(state.position)
    return {
        "race_time_ms": int(race_time_ms),
        "display_speed": int(state.display_speed),
        "position": [float(value) for value in state.position],
        "progress": float(projection.progress),
        "lateral_offset": float(projection.lateral_offset),
        "vertical_offset": float(projection.vertical_offset),
        "upright_cosine": float(state.rotation_matrix[1, 1]),
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            output.write("\n")


def resolve_replay_path(replay: Path) -> Path:
    if replay.is_absolute():
        return replay
    workspace_replay = WORKSPACE_ROOT / replay
    if workspace_replay.is_file():
        return workspace_replay
    return DEFAULT_REPLAY_DIR / replay


def inspect_replay(
    replay: Path,
    *,
    session: LiveTmiSession,
    reset_diagnostics: object,
    reference: ReferencePath,
    tmi_scripts_dir: Path,
    output_dir: Path,
    max_race_ms: int,
    fps: int,
    max_width: int,
    simulation_speed: float,
    video_only_after_disconnect: bool,
) -> dict[str, Any]:
    local_replay = resolve_replay_path(replay)
    if not local_replay.is_file():
        raise FileNotFoundError(local_replay)
    external_replay = tmi_scripts_dir / local_replay.name
    if not external_replay.is_file():
        raise FileNotFoundError(external_replay)
    local_hash = sha256(local_replay)
    if sha256(external_replay) != local_hash:
        raise ProtocolError(f"TMInterface replay differs from preserved copy: {local_replay.name}")

    if video_only_after_disconnect:
        session.client.set_response_timeout(5_000)
    session.client.execute_command("unload")
    session.client.execute_command(f"load {external_replay.name}")
    state = session.reset(reset_diagnostics)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{local_replay.stem}.mp4"
    telemetry_path = output_dir / f"{local_replay.stem}.jsonl"
    if video_path.exists() or telemetry_path.exists():
        raise FileExistsError(f"refusing to overwrite replay inspection: {local_replay.stem}")

    label = (
        f"{replay_experiment_label(local_replay.stem)} preserved replay | "
        f"{local_replay.stem} | {simulation_speed:g}x"
    )
    recorder = ProgressVideoRecorder(video_path, label=label, fps=fps, max_width=max_width)
    initial = telemetry_record(state, reference, 0)
    records = [initial]
    recorder.update(
        state="PLAYBACK",
        elapsed_ms=0,
        display_speed=initial["display_speed"],
        progress=initial["progress"],
    )
    recorder.start()
    playback_started_at = time.perf_counter()
    recorder_stopped = False
    race_finished = False
    playback_disconnect: str | None = None
    try:
        try:
            while True:
                step = session.advance_playback()
                record = telemetry_record(step.state, reference, step.race_time_ms)
                if not all(
                    math.isfinite(float(record[key]))
                    for key in ("progress", "lateral_offset", "vertical_offset", "upright_cosine")
                ):
                    raise ProtocolError("replay inspection produced nonfinite telemetry")
                records.append(record)
                race_finished = bool(step.race_finished)
                recorder.update(
                    state="FINISH" if race_finished else "PLAYBACK",
                    elapsed_ms=record["race_time_ms"],
                    display_speed=record["display_speed"],
                    progress=record["progress"],
                )
                if race_finished or step.race_time_ms >= max_race_ms:
                    break
        except OSError as error:
            if not video_only_after_disconnect:
                raise
            playback_disconnect = repr(error)
            target_wall_seconds = max_race_ms / 1000.0 / simulation_speed
            remaining = target_wall_seconds - (
                time.perf_counter() - playback_started_at
            )
            if remaining > 0.0:
                time.sleep(remaining)
        video = recorder.stop()
        recorder_stopped = True
    finally:
        if not recorder_stopped:
            try:
                recorder.update(state="CAPTURE ERROR")
                recorder.stop()
            except Exception:
                pass

    write_jsonl(telemetry_path, records)
    terminal = records[-1]
    return {
        "replay": str(local_replay.relative_to(WORKSPACE_ROOT)),
        "replay_sha256": local_hash,
        "race_finished": race_finished,
        "terminal_race_time_ms": (
            None if playback_disconnect is not None else terminal["race_time_ms"]
        ),
        "terminal_progress": (
            None if playback_disconnect is not None else terminal["progress"]
        ),
        "terminal_lateral_offset": (
            None if playback_disconnect is not None else terminal["lateral_offset"]
        ),
        "terminal_vertical_offset": (
            None if playback_disconnect is not None else terminal["vertical_offset"]
        ),
        "minimum_vertical_offset": min(float(row["vertical_offset"]) for row in records),
        "minimum_upright_cosine": min(float(row["upright_cosine"]) for row in records),
        "telemetry_complete": playback_disconnect is None,
        "video_only_after_bridge_disconnect": playback_disconnect is not None,
        "playback_disconnect": playback_disconnect,
        "video": {**video, "path": str(video_path.relative_to(WORKSPACE_ROOT))},
        "telemetry": str(telemetry_path.relative_to(WORKSPACE_ROOT)),
        "telemetry_sha256": sha256(telemetry_path),
    }


def main() -> int:
    args = parse_args()
    args.tmi_scripts_dir = args.tmi_scripts_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not math.isfinite(args.simulation_speed) or args.simulation_speed <= 0:
        raise SystemExit("--simulation-speed must be positive and finite")
    if args.max_race_ms <= 0:
        raise SystemExit("--max-race-ms must be positive")
    reference = ReferencePath.from_csv(args.reference_path.resolve())
    if not args.reuse_game:
        close_trackmania()
    _, launched = ensure_trackmania_running(port=args.port, confirm_existing=True)
    print(f"TrackMania ready (launched={launched})", flush=True)
    if not launched:
        restart_trackmania_race()
    session = LiveTmiSession(
        EnvironmentConfig(
            port=args.port,
            simulation_speed=args.simulation_speed,
            max_episode_ms=args.max_race_ms,
            map_to_load=args.map_to_load,
            auto_respawn_on_connect=False,
            bridge_response_timeout_ms=90_000,
        )
    )
    results: list[dict[str, Any]] = []
    try:
        prepared = session.prepare()
        reset_projection = reference.project(prepared.position)
        session.reset(reset_projection)
        for replay in args.replays:
            result = inspect_replay(
                replay,
                session=session,
                reset_diagnostics=reset_projection,
                reference=reference,
                tmi_scripts_dir=args.tmi_scripts_dir,
                output_dir=args.output_dir,
                max_race_ms=args.max_race_ms,
                fps=args.fps,
                max_width=args.max_width,
                simulation_speed=args.simulation_speed,
                video_only_after_disconnect=args.video_only_after_disconnect,
            )
            results.append(result)
            print(
                f"inspected {Path(result['replay']).name}: "
                f"finished={result['race_finished']} "
                f"progress={result['terminal_progress']}",
                flush=True,
            )
    finally:
        session.close()

    manifest_path = args.output_dir / "inspection_manifest.json"
    manifest_path.write_text(
        json.dumps({"status": "complete", "replays": results}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
