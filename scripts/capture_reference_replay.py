"""Replay extracted inputs and capture finite, monotonic reference telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, LiveTmiSession
from trackmania_rl.game_launch import close_trackmania, ensure_trackmania_running
from trackmania_rl.tmi_bridge import ProtocolError

DEFAULT_INPUT = (
    WORKSPACE_ROOT / "artifacts" / "replays" / "a02_reference" / "nadeo_author.txt"
)
DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "artifacts" / "telemetry" / "a02_nadeo_author.jsonl"
)
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument("--map", default="A02-Race.Challenge.Gbx")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--simulation-speed", type=float, default=6.0)
    parser.add_argument("--max-race-ms", type=int, default=45_000)
    parser.add_argument(
        "--reuse-game",
        action="store_true",
        help="reuse a verified stable game instead of launching a fresh process",
    )
    return parser.parse_args()


def telemetry_record(state: object, race_time_ms: int) -> dict[str, object]:
    position = np.asarray(state.position, dtype=np.float64)
    velocity = np.asarray(state.velocity, dtype=np.float64)
    rotation = np.asarray(state.rotation_matrix, dtype=np.float64)
    yaw_pitch_roll = np.asarray(state.yaw_pitch_roll, dtype=np.float64)
    values = np.concatenate((position, velocity, rotation.ravel(), yaw_pitch_roll))
    if not np.isfinite(values).all():
        raise ProtocolError("reference replay produced nonfinite telemetry")
    return {
        "race_time_ms": int(race_time_ms),
        "display_speed": int(state.display_speed),
        "position": position.tolist(),
        "velocity": velocity.tolist(),
        "rotation_matrix": rotation.tolist(),
        "yaw_pitch_roll": yaw_pitch_roll.tolist(),
    }


def validate_records(records: list[dict[str, object]]) -> dict[str, object]:
    if len(records) < 2:
        raise ProtocolError("reference replay needs at least two active-race samples")
    race_times = np.asarray([row["race_time_ms"] for row in records], dtype=np.int64)
    if race_times[0] != 0 or np.any(np.diff(race_times) <= 0):
        raise ProtocolError("reference replay race clock must start at zero and increase")
    positions = np.asarray([row["position"] for row in records], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if not np.isfinite(segment_lengths).all():
        raise ProtocolError("reference replay path lengths are nonfinite")
    return {
        "samples": len(records),
        "terminal_race_time_ms": int(race_times[-1]),
        "path_length": float(segment_lengths.sum()),
        "maximum_segment_length": float(segment_lengths.max()),
        "position_span": np.ptp(positions, axis=0).tolist(),
    }


def main() -> int:
    args = parse_args()
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.tmi_scripts_dir = args.tmi_scripts_dir.resolve()
    if not math.isfinite(args.simulation_speed) or args.simulation_speed <= 0:
        raise SystemExit("--simulation-speed must be positive and finite")
    if args.max_race_ms <= 0:
        raise SystemExit("--max-race-ms must be positive")
    if not args.input.is_file():
        raise SystemExit(f"input replay does not exist: {args.input}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite telemetry: {args.output}")
    external_inputs = args.tmi_scripts_dir / args.input.name
    if not external_inputs.is_file() or sha256(external_inputs) != sha256(args.input):
        raise SystemExit(
            "TMInterface Scripts copy is missing or differs from the preserved inputs"
        )

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
            max_episode_ms=args.max_race_ms,
            map_to_load=args.map,
            auto_respawn_on_connect=False,
            wait_for_race_start_on_connect=True,
        )
    )
    records: list[dict[str, object]] = []
    finished = False
    countdown_seen = False
    try:
        session.prepare()
        session.client.execute_command("unload")
        session.client.execute_command(f"load {external_inputs.name}")
        session.client.give_up()
        while True:
            step = session.advance_playback()
            if step.race_time_ms < 0:
                countdown_seen = True
                continue
            if not countdown_seen:
                continue
            records.append(telemetry_record(step.state, step.race_time_ms))
            finished = bool(step.race_finished)
            if finished or step.race_time_ms >= args.max_race_ms:
                break
    finally:
        session.close()
    if not countdown_seen:
        raise ProtocolError("reference playback never entered a fresh countdown")
    if not finished:
        raise ProtocolError("reference playback did not finish before the time limit")

    summary = validate_records(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, separators=(",", ":"), allow_nan=False))
            output.write("\n")
    manifest = {
        "track": "A02-Race",
        "map": args.map,
        "kind": "official_nadeo_author_replay_telemetry",
        "input": str(args.input.relative_to(WORKSPACE_ROOT)),
        "input_sha256": sha256(args.input),
        "telemetry": str(args.output.relative_to(WORKSPACE_ROOT)),
        "telemetry_sha256": sha256(args.output),
        "simulation_speed": args.simulation_speed,
        "step_period_ms": 100,
        **summary,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"captured {summary['samples']} A02 author samples; "
        f"finish={summary['terminal_race_time_ms']} ms",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
