"""Verify accelerated TMInterface time while telemetry and input remain active."""

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
        description="Run input and telemetry at accelerated simulation speed."
    )
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--speed", type=float, default=6.0)
    parser.add_argument("--game-duration-ms", type=int, default=4000)
    parser.add_argument("--bridge-timeout", type=float, default=30.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT
        / "artifacts"
        / "telemetry"
        / "phase0_accelerated.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.speed) or not 1.0 < args.speed <= 1000.0:
        raise SystemExit("--speed must be finite and greater than 1, up to 1000")
    if args.game_duration_ms <= 0:
        raise SystemExit("--game-duration-ms must be positive")
    if not math.isfinite(args.bridge_timeout) or args.bridge_timeout <= 0:
        raise SystemExit("--bridge-timeout must be a positive finite number")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    start_race_time: int | None = None
    start_wall_time: float | None = None
    last_saved_elapsed: int | None = None

    client = TmiBridgeClient(port=args.port, timeout_seconds=args.bridge_timeout)
    client.connect()
    try:
        with args.output.open("w", encoding="utf-8", newline="\n") as output:
            while True:
                message_type = client.read_message_type()

                if message_type is MessageType.SC_ON_CONNECT_SYNC:
                    client.execute_command("set unfocused_fps_limit false")
                    client.set_speed(args.speed)
                    client.respond(message_type)
                    continue

                if message_type is MessageType.SC_RUN_STEP_SYNC:
                    race_time = client.read_int32()
                    now = time.monotonic()
                    if start_race_time is None:
                        start_race_time = race_time
                        start_wall_time = now
                    elapsed_ms = race_time - start_race_time

                    if (
                        last_saved_elapsed is None
                        or elapsed_ms - last_saved_elapsed >= 100
                    ):
                        state = client.get_simulation_state()
                        position = np.asarray(state.position, dtype=np.float64)
                        velocity = np.asarray(state.velocity, dtype=np.float64)
                        if not np.isfinite(np.concatenate((position, velocity))).all():
                            raise ProtocolError(
                                "accelerated probe received nonfinite telemetry"
                            )
                        record = {
                            "wall_seconds": now - start_wall_time,
                            "race_time_ms": int(state.race_time),
                            "elapsed_ms": elapsed_ms,
                            "display_speed": int(state.display_speed),
                            "position": position.tolist(),
                            "velocity": velocity.tolist(),
                        }
                        samples.append(record)
                        output.write(json.dumps(record, separators=(",", ":")) + "\n")
                        output.flush()
                        last_saved_elapsed = elapsed_ms

                    client.set_input_state(accelerate=True)
                    client.respond(message_type)
                    if elapsed_ms >= args.game_duration_ms:
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
            client.set_input_state()
            client.set_speed(1.0)
        finally:
            client.close()

    if len(samples) < 3:
        raise ProtocolError("accelerated probe received too few telemetry samples")
    game_seconds = (samples[-1]["elapsed_ms"] - samples[0]["elapsed_ms"]) / 1000.0
    wall_seconds = samples[-1]["wall_seconds"] - samples[0]["wall_seconds"]
    effective_speed = game_seconds / wall_seconds
    positions = np.asarray([sample["position"] for sample in samples], dtype=np.float64)
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    max_display_speed = max(int(sample["display_speed"]) for sample in samples)

    minimum_effective_speed = max(2.0, args.speed * 0.5)
    if effective_speed < minimum_effective_speed:
        raise ProtocolError(
            f"requested {args.speed:.1f}x but measured only {effective_speed:.3f}x"
        )
    if displacement < 5.0 or max_display_speed < 20:
        raise ProtocolError(
            "accelerated input did not produce enough verified vehicle movement"
        )

    print(f"accelerated time verified with {len(samples)} finite telemetry samples")
    print(
        f"requested_speed={args.speed:.1f}x, effective_speed={effective_speed:.3f}x, "
        f"game_seconds={game_seconds:.3f}, wall_seconds={wall_seconds:.3f}, "
        f"displacement={displacement:.3f}, max_display_speed={max_display_speed}, "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
