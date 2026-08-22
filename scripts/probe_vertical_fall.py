"""Reverse off the A01 start and verify vertical fall truncation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.rewards import sparse_finish_reward
from trackmania_rl.tmi_bridge import ProtocolError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deliberately reverse off A01 and measure vertical truncation."
    )
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--timeout-ms", type=int, default=10_000)
    parser.add_argument("--vertical-drop", type=float, default=10.0)
    parser.add_argument(
        "--action-log",
        type=Path,
        default=WORKSPACE_ROOT
        / "artifacts"
        / "logs"
        / "vertical_fall_probe_actions.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=WORKSPACE_ROOT
        / "artifacts"
        / "smoke"
        / "vertical_fall_probe.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    args = parse_args()
    config = EnvironmentConfig(
        port=args.port,
        max_episode_ms=args.timeout_ms,
        max_vertical_drop=args.vertical_drop,
    )
    env = TrackmaniaEnv(
        config=config,
        reward_function=sparse_finish_reward,
        action_log_path=args.action_log,
    )
    action = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    observation, reset_info = env.reset()
    if not np.isfinite(observation).all():
        raise ProtocolError("vertical fall probe reset observation is nonfinite")

    minimum_vertical_offset = float(reset_info["vertical_offset"])
    terminated = False
    truncated = False
    steps = 0
    info = reset_info
    try:
        while not (terminated or truncated):
            observation, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(observation).all() or not np.isfinite(reward):
                raise ProtocolError("vertical fall probe produced nonfinite data")
            minimum_vertical_offset = min(
                minimum_vertical_offset,
                float(info["vertical_offset"]),
            )
            steps += 1
            if steps > args.timeout_ms // config.step_period_ms + 1:
                raise ProtocolError("vertical fall probe exceeded its safety limit")
    finally:
        env.close()

    if terminated or not truncated:
        raise ProtocolError("vertical fall probe did not truncate")
    if not bool(info["fallen"]):
        raise ProtocolError(
            "reverse run ended without vertical fall detection: "
            f"minimum_vertical_offset={minimum_vertical_offset:.3f}, "
            f"timeout={info['timeout']}, off_track={info['off_track']}"
        )
    if bool(info["timeout"]):
        raise ProtocolError("vertical fall was not detected before timeout")

    summary = {
        "steps": steps,
        "elapsed_ms": int(info["elapsed_ms"]),
        "configured_vertical_drop": args.vertical_drop,
        "minimum_vertical_offset": minimum_vertical_offset,
        "terminal_vertical_offset": float(info["vertical_offset"]),
        "terminal_lateral_offset": float(info["lateral_offset"]),
        "fallen": bool(info["fallen"]),
        "timeout": bool(info["timeout"]),
        "off_track": bool(info["off_track"]),
        "raw_action": action.tolist(),
        "action_log_sha256": sha256(args.action_log),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"vertical fall detected after {steps} steps at "
        f"offset={minimum_vertical_offset:.3f}",
        flush=True,
    )
    print(f"summary SHA-256={sha256(args.summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
