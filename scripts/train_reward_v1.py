"""Train the pre-registered sparse finish-only PPO reward experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.ppo_audit import BoundedPpoActionStatsCallback
from trackmania_rl.rewards import sparse_finish_reward
from trackmania_rl.tmi_bridge import ProtocolError

RUN_DIR = WORKSPACE_ROOT / "runs" / "reward_v1"
TENSORBOARD_ROOT = WORKSPACE_ROOT / "tensorboard"
CHECKPOINT_DIR = WORKSPACE_ROOT / "checkpoints" / "reward_v1"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "final_model"
MANIFEST_PATH = RUN_DIR / "run_manifest.json"
SUMMARY_PATH = RUN_DIR / "training_summary.json"
MONITOR_PREFIX = RUN_DIR / "training"
REWARD_DOC = WORKSPACE_ROOT / "reward_v1.md"
PROTOCOL_DOC = (
    WORKSPACE_ROOT / "docs" / "decisions" / "0006-reward-v1-training-protocol.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed 500k-step reward-v1 sparse PPO experiment."
    )
    parser.add_argument("--minimum-timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class PeriodicProgressCallback(BaseCallback):
    def __init__(self, interval: int = 25_000) -> None:
        super().__init__(verbose=0)
        self.interval = interval
        self._next_report = interval
        self._started = 0.0

    def _on_training_start(self) -> None:
        self._started = time.perf_counter()
        self._next_report = (
            (self.num_timesteps // self.interval) + 1
        ) * self.interval

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next_report:
            elapsed = time.perf_counter() - self._started
            print(
                f"progress timesteps={self.num_timesteps} "
                f"segment_elapsed_seconds={elapsed:.1f}",
                flush=True,
            )
            self._next_report += self.interval
        return True


def read_monitor_rows() -> list[dict[str, str]]:
    monitor_files = sorted(RUN_DIR.glob("*.monitor.csv"))
    if not monitor_files:
        raise ProtocolError("reward-v1 run did not create a Monitor CSV")
    rows: list[dict[str, str]] = []
    for path in monitor_files:
        with path.open("r", encoding="utf-8", newline="") as source:
            data_lines = [line for line in source if not line.startswith("#")]
        rows.extend(csv.DictReader(data_lines))
    if not rows:
        raise ProtocolError("reward-v1 Monitor CSV contains no completed episodes")
    return rows


def read_tensorboard_metrics(event_files: list[Path]) -> dict[str, dict[str, Any]]:
    required_tags = ("rollout/ep_rew_mean", "rollout/ep_len_mean")
    values_by_tag: dict[str, list[float]] = {tag: [] for tag in required_tags}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file.parent))
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        for tag in required_tags:
            if tag in available:
                values_by_tag[tag].extend(
                    event.value for event in accumulator.Scalars(tag)
                )
    missing = [tag for tag, values in values_by_tag.items() if not values]
    if missing:
        raise ProtocolError(f"TensorBoard is missing reward-v1 metrics: {missing}")
    return {
        tag: {
            "points": len(values),
            "first": values[0],
            "last": values[-1],
            "minimum": min(values),
            "maximum": max(values),
        }
        for tag, values in values_by_tag.items()
    }


def main() -> int:
    args = parse_args()
    if args.minimum_timesteps < 500_000:
        raise SystemExit(
            "Decision 0006 requires --minimum-timesteps to be at least 500000"
        )
    if args.seed != 42:
        raise SystemExit("Decision 0006 fixes the reward-v1 seed at 42")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    event_files_before = set(TENSORBOARD_ROOT.rglob("events.out.tfevents.*"))
    started_at = datetime.now(timezone.utc)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "minimum_timesteps": args.minimum_timesteps,
        "seed": args.seed,
        "reward_function": "sparse_finish_reward",
        "tensorboard_run_name": "reward_v1_sparse",
        "simulation_speed": 6.0,
        "step_period_ms": 100,
        "max_episode_ms": 45_000,
        "max_lateral_offset": 50.0,
        "n_steps": 2_048,
        "batch_size": 64,
        "n_epochs": 10,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "checkpoint_interval": 50_000,
        "reward_doc_sha256": sha256(REWARD_DOC),
        "protocol_doc_sha256": sha256(PROTOCOL_DOC),
        "resume_checkpoint": str(args.resume) if args.resume else None,
    }
    write_json(MANIFEST_PATH, manifest)

    config = EnvironmentConfig(port=args.port)
    base_env = TrackmaniaEnv(
        config=config,
        reward_function=sparse_finish_reward,
    )
    monitored_env = Monitor(
        base_env,
        filename=str(MONITOR_PREFIX),
        info_keywords=("timeout", "off_track", "race_finished", "progress"),
        override_existing=args.resume is None,
    )
    if args.resume:
        model = PPO.load(
            args.resume,
            env=monitored_env,
            device="cpu",
            tensorboard_log=str(TENSORBOARD_ROOT),
        )
        reset_num_timesteps = False
    else:
        model = PPO(
            "MlpPolicy",
            monitored_env,
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
            tensorboard_log=str(TENSORBOARD_ROOT),
            seed=args.seed,
            device="cpu",
            verbose=0,
        )
        reset_num_timesteps = True

    starting_timesteps = int(model.num_timesteps)
    remaining_timesteps = max(0, args.minimum_timesteps - starting_timesteps)
    if remaining_timesteps == 0:
        raise SystemExit(
            f"checkpoint already has {starting_timesteps} timesteps; "
            "no reward-v1 training remains"
        )

    action_stats = BoundedPpoActionStatsCallback()
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=50_000,
                save_path=str(CHECKPOINT_DIR),
                name_prefix="ppo_reward_v1",
            ),
            action_stats,
            PeriodicProgressCallback(),
        ]
    )
    print(
        f"reward-v1 training started: initial={starting_timesteps}, "
        f"minimum_target={args.minimum_timesteps}",
        flush=True,
    )
    wall_started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            tb_log_name="reward_v1_sparse",
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=False,
        )
        model.save(FINAL_CHECKPOINT)
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(error),
                "timesteps_at_failure": int(model.num_timesteps),
            }
        )
        write_json(MANIFEST_PATH, manifest)
        raise
    finally:
        monitored_env.close()
        model.logger.close()
    wall_seconds = time.perf_counter() - wall_started

    event_files_after = set(TENSORBOARD_ROOT.rglob("events.out.tfevents.*"))
    event_files = sorted(event_files_after - event_files_before)
    if not event_files:
        raise ProtocolError("reward-v1 training created no TensorBoard event file")
    tensorboard_metrics = read_tensorboard_metrics(event_files)
    monitor_rows = read_monitor_rows()
    episode_rewards = [float(row["r"]) for row in monitor_rows]
    episode_lengths = [int(row["l"]) for row in monitor_rows]
    finishes = sum(row["race_finished"] == "True" for row in monitor_rows)
    timeouts = sum(row["timeout"] == "True" for row in monitor_rows)
    off_tracks = sum(row["off_track"] == "True" for row in monitor_rows)
    final_checkpoint_path = FINAL_CHECKPOINT.with_suffix(".zip")
    checkpoint_paths = sorted(CHECKPOINT_DIR.glob("*.zip"))
    completed_at = datetime.now(timezone.utc)

    summary = {
        "status": "complete",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "wall_seconds": wall_seconds,
        "minimum_timesteps": args.minimum_timesteps,
        "starting_timesteps": starting_timesteps,
        "actual_timesteps": int(model.num_timesteps),
        "seed": args.seed,
        "reward_function": "sparse_finish_reward",
        "episodes": len(monitor_rows),
        "finishes": finishes,
        "timeouts": timeouts,
        "off_tracks": off_tracks,
        "episode_reward_minimum": min(episode_rewards),
        "episode_reward_maximum": max(episode_rewards),
        "episode_length_minimum": min(episode_lengths),
        "episode_length_maximum": max(episode_lengths),
        "action_validation": action_stats.summary(),
        "tensorboard_metrics": tensorboard_metrics,
        "final_checkpoint": str(final_checkpoint_path.relative_to(WORKSPACE_ROOT)),
        "final_checkpoint_sha256": sha256(final_checkpoint_path),
        "checkpoints": [
            {
                "path": str(path.relative_to(WORKSPACE_ROOT)),
                "sha256": sha256(path),
            }
            for path in checkpoint_paths
        ],
        "tensorboard_event_files": [
            {
                "path": str(path.relative_to(WORKSPACE_ROOT)),
                "sha256": sha256(path),
            }
            for path in event_files
        ],
    }
    write_json(SUMMARY_PATH, summary)
    manifest.update(
        {
            "status": "complete",
            "completed_at_utc": completed_at.isoformat(),
            "actual_timesteps": int(model.num_timesteps),
            "summary_sha256": sha256(SUMMARY_PATH),
        }
    )
    write_json(MANIFEST_PATH, manifest)

    print(
        f"reward-v1 training complete: timesteps={model.num_timesteps}, "
        f"episodes={len(monitor_rows)}, finishes={finishes}",
        flush=True,
    )
    print(f"summary SHA-256={sha256(SUMMARY_PATH)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
