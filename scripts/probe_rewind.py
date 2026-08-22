"""Live-test snapshot rewind for reliable Gymnasium episode resets."""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.tmi_bridge import MessageType, ProtocolError, TmiBridgeClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a state, move under analog throttle, rewind, and compare."
    )
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--move-ms", type=int, default=1500)
    parser.add_argument("--step-period-ms", type=int, default=100)
    parser.add_argument("--bridge-timeout", type=float, default=30.0)
    return parser.parse_args()


def handle_non_step(client: TmiBridgeClient, message_type: MessageType) -> None:
    if message_type is MessageType.SC_CHECKPOINT_COUNT_CHANGED_SYNC:
        client.read_int32()
        client.read_int32()
    elif message_type is MessageType.SC_LAP_COUNT_CHANGED_SYNC:
        client.read_int32()
        client.read_int32()
    client.respond(message_type)


def main() -> int:
    args = parse_args()
    if args.move_ms <= 0:
        raise SystemExit("--move-ms must be positive")
    if not math.isfinite(args.bridge_timeout) or args.bridge_timeout <= 0:
        raise SystemExit("--bridge-timeout must be a positive finite number")

    client = TmiBridgeClient(port=args.port, timeout_seconds=args.bridge_timeout)
    client.connect()
    snapshot: bytes | None = None
    start_position: np.ndarray | None = None
    moved_position: np.ndarray | None = None
    restored_position: np.ndarray | None = None
    start_race_time: int | None = None
    restored_race_time: int | None = None
    rewound = False
    try:
        while True:
            message_type = client.read_message_type()
            if message_type is MessageType.SC_ON_CONNECT_SYNC:
                client.set_speed(1.0)
                client.set_on_step_period(args.step_period_ms)
                client.respond(message_type)
                continue
            if message_type is not MessageType.SC_RUN_STEP_SYNC:
                handle_non_step(client, message_type)
                continue

            race_time = client.read_int32()
            state = client.get_simulation_state()
            position = np.asarray(state.position, dtype=np.float64)
            if not np.isfinite(position).all():
                raise ProtocolError("rewind probe received nonfinite position")

            if snapshot is None:
                snapshot = bytes(state.data)
                start_position = position.copy()
                start_race_time = race_time
                client.set_continuous_input(steer=0.0, throttle=1.0, brake=0.0)
                client.respond(message_type)
                continue

            if not rewound and race_time - start_race_time < args.move_ms:
                client.set_continuous_input(steer=0.0, throttle=1.0, brake=0.0)
                client.respond(message_type)
                continue

            if not rewound:
                moved_position = position.copy()
                client.rewind_to_state(snapshot)
                client.set_continuous_input(steer=0.0, throttle=0.0, brake=0.0)
                rewound = True
                client.respond(message_type)
                continue

            restored_position = position.copy()
            restored_race_time = race_time
            client.set_continuous_input(steer=0.0, throttle=0.0, brake=0.0)
            client.respond(message_type)
            break
    finally:
        try:
            client.set_continuous_input(steer=0.0, throttle=0.0, brake=0.0)
            client.set_speed(1.0)
        finally:
            client.close()

    moved_distance = float(np.linalg.norm(moved_position - start_position))
    restore_error = float(np.linalg.norm(restored_position - start_position))
    expected_restored_time = start_race_time + args.step_period_ms
    if moved_distance < 5.0:
        raise ProtocolError(f"rewind probe moved only {moved_distance:.3f} units")
    if restore_error > 1.0:
        raise ProtocolError(f"rewind restored with {restore_error:.3f}-unit error")
    if abs(restored_race_time - expected_restored_time) > args.step_period_ms:
        raise ProtocolError(
            f"rewind restored race time {restored_race_time}, expected near "
            f"{expected_restored_time}"
        )

    print(
        f"snapshot rewind verified: moved={moved_distance:.3f}, "
        f"restore_error={restore_error:.6f}, race_time={start_race_time}->"
        f"{restored_race_time}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
