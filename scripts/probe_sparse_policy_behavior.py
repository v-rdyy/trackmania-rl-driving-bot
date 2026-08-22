"""Reproduce early sparse PPO behavior with detailed live telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.ppo_audit import BoundedPpoActionStatsCallback
from trackmania_rl.rewards import sparse_finish_reward

DEFAULT_ACTION_LOG = (
    WORKSPACE_ROOT / "artifacts" / "logs" / "sparse_policy_probe_actions.jsonl"
)
DEFAULT_MONITOR = (
    WORKSPACE_ROOT / "artifacts" / "logs" / "sparse_policy_probe"
)
DEFAULT_SUMMARY = (
    WORKSPACE_ROOT / "artifacts" / "smoke" / "sparse_policy_probe.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 8192 detailed steps through the reward-v1 PPO policy path."
    )
    parser.add_argument("--timesteps", type=int, default=8_192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--action-log", type=Path, default=DEFAULT_ACTION_LOG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    args = parse_args()
    if args.timesteps < 2_048 or args.timesteps % 2_048 != 0:
        raise SystemExit("--timesteps must be a positive multiple of 2048")

    base_env = TrackmaniaEnv(
        config=EnvironmentConfig(
            port=args.port,
            legacy_reversed_pedal_mapping=True,
        ),
        reward_function=sparse_finish_reward,
        action_log_path=args.action_log,
    )
    env = Monitor(
        base_env,
        filename=str(DEFAULT_MONITOR),
        info_keywords=(
            "timeout",
            "off_track",
            "fallen",
            "race_finished",
            "progress",
        ),
    )
    callback = BoundedPpoActionStatsCallback()
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2_048,
        batch_size=64,
        n_epochs=10,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs={"squash_output": True, "log_std_init": -1.0},
        seed=args.seed,
        device="cpu",
        verbose=0,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=False,
        )
    finally:
        env.close()

    records = [
        json.loads(line)
        for line in args.action_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terminal_records = [
        record
        for record in records
        if record["terminated"] or record["truncated"]
    ]
    summary = {
        "timesteps": int(model.num_timesteps),
        "seed": args.seed,
        "episodes": len(terminal_records),
        "finishes": sum(record["terminated"] for record in terminal_records),
        "timeouts": sum(record["timeout"] for record in terminal_records),
        "off_tracks": sum(record["off_track"] for record in terminal_records),
        "falls": sum(record["fallen"] for record in terminal_records),
        "minimum_vertical_offset": min(
            float(record["vertical_offset"]) for record in records
        ),
        "minimum_world_y": min(float(record["position"][1]) for record in records),
        "maximum_progress": max(float(record["progress"]) for record in records),
        "maximum_display_speed": max(
            int(record["display_speed"]) for record in records
        ),
        "action_validation": callback.summary(),
        "action_log_sha256": sha256(args.action_log),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"summary SHA-256={sha256(args.summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
