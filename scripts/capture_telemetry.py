"""Capture a short Phase 0 TMInterface telemetry sample as JSON Lines."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.tmi_bridge import MessageType, ProtocolError, TmiBridgeClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a local track and capture decoded TMInterface telemetry."
    )
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument(
        "--bridge-timeout",
        type=float,
        default=120.0,
        help="seconds to wait for the next bridge event while navigating menus",
    )
    parser.add_argument("--duration", type=float, default=8.0, help="capture seconds")
    parser.add_argument(
        "--sample-period-ms",
        type=int,
        default=100,
        help="minimum in-game time between saved samples",
    )
    parser.add_argument(
        "--map",
        default="A01-Race.Challenge.Gbx",
        help="TMInterface map command argument",
    )
    parser.add_argument(
        "--autologin",
        default="1",
        help="local TrackMania profile index/name; pass an empty value to skip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "artifacts" / "telemetry" / "phase0_sample.jsonl",
    )
    return parser.parse_args()


def state_record(state: object, wall_seconds: float) -> dict[str, object]:
    position = np.asarray(state.position, dtype=np.float64)
    velocity = np.asarray(state.velocity, dtype=np.float64)
    rotation = np.asarray(state.rotation_matrix, dtype=np.float64)
    yaw_pitch_roll = np.asarray(state.yaw_pitch_roll, dtype=np.float64)
    numeric_values = np.concatenate(
        (position, velocity, rotation.reshape(-1), yaw_pitch_roll)
    )
    if not np.isfinite(numeric_values).all():
        raise ProtocolError("simulation state contains NaN or infinite telemetry")

    return {
        "wall_seconds": round(wall_seconds, 6),
        "race_time_ms": int(state.race_time),
        "display_speed": int(state.display_speed),
        "position": position.tolist(),
        "velocity": velocity.tolist(),
        "rotation_matrix": rotation.tolist(),
        "yaw_pitch_roll": yaw_pitch_roll.tolist(),
    }


def sample_summary(samples: list[dict[str, object]]) -> dict[str, object]:
    """Validate a captured sequence and return useful motion ranges."""

    if len(samples) < 2:
        raise ProtocolError("telemetry capture needs at least two samples")

    race_times = np.asarray(
        [sample["race_time_ms"] for sample in samples], dtype=np.int64
    )
    if np.any(np.diff(race_times) < 0):
        raise ProtocolError("race time moved backwards during telemetry capture")
    unique_race_times = int(np.unique(race_times).size)
    if unique_race_times / len(samples) < 0.9:
        raise ProtocolError("more than 10% of telemetry samples have stale race times")

    positions = np.asarray([sample["position"] for sample in samples], dtype=np.float64)
    velocities = np.asarray([sample["velocity"] for sample in samples], dtype=np.float64)
    rotations = np.asarray(
        [sample["rotation_matrix"] for sample in samples], dtype=np.float64
    )
    return {
        "race_time_min_ms": int(race_times.min()),
        "race_time_max_ms": int(race_times.max()),
        "unique_race_times": unique_race_times,
        "position_span": np.ptp(positions, axis=0).tolist(),
        "path_length": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        "max_velocity": float(np.linalg.norm(velocities, axis=1).max()),
        "max_rotation_change": float(
            np.linalg.norm(rotations - rotations[0], axis=(1, 2)).max()
        ),
    }


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        raise SystemExit("--duration must be a positive finite number")
    if args.sample_period_ms <= 0:
        raise SystemExit("--sample-period-ms must be positive")
    if not math.isfinite(args.bridge_timeout) or args.bridge_timeout <= 0:
        raise SystemExit("--bridge-timeout must be a positive finite number")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    capture_started: float | None = None
    last_sample_race_time: int | None = None
    completion_reason = "duration reached"
    checkpoint_progress: tuple[int, int] | None = None
    lap_progress: tuple[int, int] | None = None

    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        with TmiBridgeClient(
            port=args.port, timeout_seconds=args.bridge_timeout
        ) as client:
            while True:
                message_type = client.read_message_type()

                if message_type is MessageType.SC_ON_CONNECT_SYNC:
                    if args.autologin:
                        client.execute_command(f"set autologin {args.autologin}")
                    client.execute_command("set unfocused_fps_limit false")
                    client.execute_command("set disable_forced_camera true")
                    client.execute_command("set autorewind false")
                    client.execute_command(f"map {args.map}")
                    client.respond(message_type)
                    continue

                if message_type is MessageType.SC_RUN_STEP_SYNC:
                    race_time = client.read_int32()
                    now = time.monotonic()
                    if capture_started is None:
                        capture_started = now
                    should_sample = (
                        last_sample_race_time is None
                        or race_time < last_sample_race_time
                        or race_time - last_sample_race_time >= args.sample_period_ms
                    )
                    if should_sample:
                        record = state_record(
                            client.get_simulation_state(), now - capture_started
                        )
                        samples.append(record)
                        output.write(json.dumps(record, separators=(",", ":")) + "\n")
                        output.flush()
                        last_sample_race_time = race_time

                    race_finished = race_time > 0 and client.race_finished()
                    duration_reached = now - capture_started >= args.duration
                    client.respond(message_type)
                    if race_finished:
                        completion_reason = "race finished"
                        break
                    if duration_reached:
                        break
                    continue

                if message_type is MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
                    checkpoint_progress = (client.read_int32(), client.read_int32())

                if message_type is MessageType.SC_LAP_COUNT_CHANGED_SYNC:
                    lap_progress = (client.read_int32(), client.read_int32())

                client.respond(message_type)

    if not samples:
        raise ProtocolError("connected, but received no race telemetry samples")

    summary = sample_summary(samples)
    print(f"captured {len(samples)} finite telemetry samples to {args.output}")
    print(
        "ranges: "
        f"race_time={summary['race_time_min_ms']}..{summary['race_time_max_ms']} ms, "
        f"position_span={summary['position_span']}, "
        f"path_length={summary['path_length']:.3f}, "
        f"max_velocity={summary['max_velocity']:.3f}, "
        f"max_rotation_change={summary['max_rotation_change']:.6f}"
    )
    print(
        f"completion={completion_reason}, "
        f"checkpoints={checkpoint_progress}, laps={lap_progress}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
