"""Verify the Decision 0002 analog control path against live telemetry."""

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
        description="Apply half throttle, then half steering, through analog inputs."
    )
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--bridge-timeout", type=float, default=30.0)
    parser.add_argument("--straight-ms", type=int, default=2000)
    parser.add_argument("--steer-ms", type=int, default=1500)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT / "artifacts" / "telemetry" / "phase1_analog.jsonl",
    )
    return parser.parse_args()


def angle_change(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(second - first), math.cos(second - first)))


def main() -> int:
    args = parse_args()
    phase_end_ms = args.straight_ms + args.steer_ms
    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    start_race_time: int | None = None
    last_saved_elapsed: int | None = None

    client = TmiBridgeClient(port=args.port, timeout_seconds=args.bridge_timeout)
    client.connect()
    try:
        with args.output.open("w", encoding="utf-8", newline="\n") as output:
            while True:
                message_type = client.read_message_type()

                if message_type is MessageType.SC_ON_CONNECT_SYNC:
                    client.execute_command("set unfocused_fps_limit false")
                    client.respond(message_type)
                    continue

                if message_type is MessageType.SC_RUN_STEP_SYNC:
                    race_time = client.read_int32()
                    if start_race_time is None:
                        start_race_time = race_time
                    elapsed_ms = race_time - start_race_time

                    if elapsed_ms < args.straight_ms:
                        raw_action = (0.0, 0.5, 0.0)
                    elif elapsed_ms < phase_end_ms:
                        raw_action = (0.5, 0.5, 0.0)
                    else:
                        raw_action = (0.0, 0.0, 0.0)

                    applied = client.set_continuous_input(
                        steer=raw_action[0],
                        throttle=raw_action[1],
                        brake=raw_action[2],
                    )
                    if (
                        last_saved_elapsed is None
                        or elapsed_ms - last_saved_elapsed >= 100
                    ):
                        state = client.get_simulation_state()
                        position = np.asarray(state.position, dtype=np.float64)
                        yaw_pitch_roll = np.asarray(
                            state.yaw_pitch_roll, dtype=np.float64
                        )
                        if not np.isfinite(
                            np.concatenate((position, yaw_pitch_roll))
                        ).all():
                            raise ProtocolError("analog probe received nonfinite telemetry")
                        record = {
                            "race_time_ms": int(state.race_time),
                            "elapsed_ms": elapsed_ms,
                            "raw_action": list(raw_action),
                            "applied_steer": applied[0],
                            "applied_gas": applied[1],
                            "display_speed": int(state.display_speed),
                            "position": position.tolist(),
                            "yaw_pitch_roll": yaw_pitch_roll.tolist(),
                        }
                        samples.append(record)
                        output.write(json.dumps(record, separators=(",", ":")) + "\n")
                        output.flush()
                        last_saved_elapsed = elapsed_ms

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
    finally:
        try:
            client.set_continuous_input(steer=0.0, throttle=0.0, brake=0.0)
        finally:
            client.close()

    if len(samples) < 3:
        raise ProtocolError("analog probe received too few telemetry samples")
    positions = np.asarray([sample["position"] for sample in samples], dtype=np.float64)
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    max_speed = max(int(sample["display_speed"]) for sample in samples)
    steer_start = min(
        samples, key=lambda sample: abs(sample["elapsed_ms"] - args.straight_ms)
    )
    yaw_delta = angle_change(
        float(steer_start["yaw_pitch_roll"][0]),
        float(samples[-1]["yaw_pitch_roll"][0]),
    )
    if displacement < 5.0 or max_speed < 15 or yaw_delta < 0.03:
        raise ProtocolError(
            "analog controls did not produce enough speed, movement, and steering"
        )

    print(f"analog input verified with {len(samples)} finite telemetry samples")
    print(
        f"displacement={displacement:.3f}, max_speed={max_speed}, "
        f"steer_yaw_change={yaw_delta:.6f} rad, output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
