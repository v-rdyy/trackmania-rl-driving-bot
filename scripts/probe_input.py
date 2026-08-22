"""Verify Phase 0 acceleration and steering through the TMInterface bridge."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.tmi_bridge import MessageType, ProtocolError, TmiBridgeClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reset the current race, accelerate, steer left, and verify motion."
    )
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--bridge-timeout", type=float, default=30.0)
    parser.add_argument("--accelerate-ms", type=int, default=2500)
    parser.add_argument("--steer-ms", type=int, default=1500)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "artifacts" / "telemetry" / "phase0_input.jsonl",
    )
    return parser.parse_args()


def angle_change(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(second - first), math.cos(second - first)))


def main() -> int:
    args = parse_args()
    if args.accelerate_ms <= 0 or args.steer_ms <= 0:
        raise SystemExit("input phase durations must be positive")
    if not math.isfinite(args.bridge_timeout) or args.bridge_timeout <= 0:
        raise SystemExit("--bridge-timeout must be a positive finite number")

    phase_end_ms = args.accelerate_ms + args.steer_ms
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    last_saved_time: int | None = None
    start_race_time: int | None = None

    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        with TmiBridgeClient(
            port=args.port, timeout_seconds=args.bridge_timeout
        ) as client:
            while True:
                message_type = client.read_message_type()

                if message_type is MessageType.SC_ON_CONNECT_SYNC:
                    client.execute_command("set unfocused_fps_limit false")
                    client.execute_command("set disable_forced_camera true")
                    client.respond(message_type)
                    continue

                if message_type is MessageType.SC_RUN_STEP_SYNC:
                    race_time = client.read_int32()
                    if start_race_time is None:
                        start_race_time = race_time
                    elapsed_ms = race_time - start_race_time
                    state = client.get_simulation_state()
                    if last_saved_time is None or race_time - last_saved_time >= 100:
                        record = {
                            "race_time_ms": int(state.race_time),
                            "elapsed_ms": elapsed_ms,
                            "display_speed": int(state.display_speed),
                            "position": np.asarray(
                                state.position, dtype=np.float64
                            ).tolist(),
                            "yaw_pitch_roll": np.asarray(
                                state.yaw_pitch_roll, dtype=np.float64
                            ).tolist(),
                        }
                        numeric = np.concatenate(
                            (
                                np.asarray(record["position"]),
                                np.asarray(record["yaw_pitch_roll"]),
                            )
                        )
                        if not np.isfinite(numeric).all():
                            raise ProtocolError("input probe received nonfinite telemetry")
                        samples.append(record)
                        output.write(json.dumps(record, separators=(",", ":")) + "\n")
                        output.flush()
                        last_saved_time = race_time

                    if elapsed_ms < args.accelerate_ms:
                        client.set_input_state(accelerate=True)
                    elif elapsed_ms < phase_end_ms:
                        client.set_input_state(left=True, accelerate=True)
                    else:
                        client.set_input_state()

                    client.respond(message_type)
                    if elapsed_ms >= phase_end_ms:
                        break
                    continue

                if message_type is MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
                    client.read_int32()
                    client.read_int32()
                elif message_type is MessageType.SC_LAP_COUNT_CHANGED_SYNC:
                    client.read_int32()
                    client.read_int32()
                client.respond(message_type)

    driving = samples
    if len(driving) < 3:
        raise ProtocolError("input probe received too few driving samples")

    positions = np.asarray([sample["position"] for sample in driving], dtype=np.float64)
    speeds = np.asarray([sample["display_speed"] for sample in driving], dtype=np.float64)
    steer_start = min(
        driving,
        key=lambda sample: abs(sample["elapsed_ms"] - args.accelerate_ms),
    )
    final = driving[-1]
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    yaw_delta = angle_change(
        float(steer_start["yaw_pitch_roll"][0]),
        float(final["yaw_pitch_roll"][0]),
    )
    max_speed = float(speeds.max())

    if displacement < 5.0:
        raise ProtocolError(f"accelerate input moved only {displacement:.3f} units")
    if max_speed < 20.0:
        raise ProtocolError(f"accelerate input reached only {max_speed:.1f} speed")
    if yaw_delta < 0.05:
        raise ProtocolError(f"steer input changed yaw by only {yaw_delta:.6f} rad")

    print(f"scripted input verified with {len(driving)} driving samples")
    print(
        f"displacement={displacement:.3f}, max_speed={max_speed:.1f}, "
        f"steer_yaw_change={yaw_delta:.6f} rad, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
