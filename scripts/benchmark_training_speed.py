"""Benchmark live reward-v1 PPO throughput at candidate simulation speeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.ppo_audit import BoundedPpoActionStatsCallback
from trackmania_rl.rewards import sparse_finish_reward

DEFAULT_OUTPUT = (
    WORKSPACE_ROOT / "artifacts" / "smoke" / "training_speed_benchmark.json"
)
MONITOR_DIR = WORKSPACE_ROOT / "artifacts" / "logs" / "training_speed_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare end-to-end PPO throughput at multiple TMI speeds."
    )
    parser.add_argument("--speeds", type=float, nargs="+", default=[6, 20, 50, 100])
    parser.add_argument("--timesteps", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_monitor(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        data_lines = [line for line in source if not line.startswith("#")]
    return list(csv.DictReader(data_lines))


def benchmark_speed(
    *, speed: float, timesteps: int, seed: int, port: int
) -> dict[str, object]:
    monitor_prefix = MONITOR_DIR / f"speed_{speed:g}x"
    monitor_path = monitor_prefix.with_suffix(".monitor.csv")
    base_env = TrackmaniaEnv(
        config=EnvironmentConfig(
            port=port,
            simulation_speed=speed,
            legacy_reversed_pedal_mapping=True,
        ),
        reward_function=sparse_finish_reward,
    )
    env = Monitor(
        base_env,
        filename=str(monitor_prefix),
        info_keywords=(
            "timeout",
            "off_track",
            "fallen",
            "race_finished",
            "progress",
        ),
        override_existing=True,
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
        seed=seed,
        device="cpu",
        verbose=0,
    )
    started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            progress_bar=False,
        )
    finally:
        env.close()
    wall_seconds = time.perf_counter() - started
    rows = read_monitor(monitor_path)
    return {
        "requested_speed": speed,
        "timesteps": int(model.num_timesteps),
        "wall_seconds": wall_seconds,
        "environment_steps_per_second": int(model.num_timesteps) / wall_seconds,
        "episodes": len(rows),
        "finishes": sum(row["race_finished"] == "True" for row in rows),
        "timeouts": sum(row["timeout"] == "True" for row in rows),
        "off_tracks": sum(row["off_track"] == "True" for row in rows),
        "falls": sum(row["fallen"] == "True" for row in rows),
        "action_validation": callback.summary(),
    }


def main() -> int:
    args = parse_args()
    if args.timesteps < 2_048 or args.timesteps % 2_048 != 0:
        raise SystemExit("--timesteps must be a positive multiple of 2048")
    if any(not math.isfinite(speed) or not 1.0 < speed <= 1_000 for speed in args.speeds):
        raise SystemExit("every --speeds value must be finite and in (1, 1000]")

    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for speed in args.speeds:
        print(f"benchmarking {speed:g}x simulation speed", flush=True)
        result = benchmark_speed(
            speed=speed,
            timesteps=args.timesteps,
            seed=args.seed,
            port=args.port,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    summary = {
        "seed": args.seed,
        "timesteps_per_speed": args.timesteps,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
