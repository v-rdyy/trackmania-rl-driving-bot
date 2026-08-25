"""Train the pre-registered V5 policy from the pinned V4 checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from train_reward_v3 import (
    PeriodicProgressCallback,
    close_model_logger,
    prior_wall_seconds,
    read_tensorboard_metrics,
    sha256,
    verify_sleep_disabled,
    write_json,
)
from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.game_launch import close_trackmania, ensure_trackmania_running
from trackmania_rl.ppo_audit import BoundedPpoActionStatsCallback
from trackmania_rl.rewards import (
    V5_STEERING_RATE_COEFFICIENT,
    steering_rate_smoothness_reward,
)
from trackmania_rl.tmi_bridge import ProtocolError

RUN_DIR = WORKSPACE_ROOT / "runs" / "reward_v5"
TENSORBOARD_ROOT = WORKSPACE_ROOT / "tensorboard"
CHECKPOINT_DIR = WORKSPACE_ROOT / "checkpoints" / "reward_v5"
FINAL_CHECKPOINT = CHECKPOINT_DIR / "final_model"
MANIFEST_PATH = RUN_DIR / "run_manifest.json"
SUMMARY_PATH = RUN_DIR / "training_summary.json"
MONITOR_PREFIX = RUN_DIR / "training"
REWARD_DOC = WORKSPACE_ROOT / "reward_v5.md"
EXPERIMENT_LABEL = "reward-v5"
REWARD_FUNCTION = steering_rate_smoothness_reward
REWARD_FUNCTION_NAME = "steering_rate_smoothness_reward"
REWARD_METADATA: dict[str, Any] = {
    "steering_rate_coefficient": V5_STEERING_RATE_COEFFICIENT,
}
CHECKPOINT_NAME_PREFIX = "ppo_reward_v5"
V4_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "reward_v4" / "final_model.zip"
EXPECTED_V4_CHECKPOINT_SHA256 = (
    "6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2"
)
V4_STARTING_TIMESTEPS = 2_002_944
ADDITIONAL_TIMESTEPS = 1_000_000
TARGET_TOTAL_TIMESTEPS = V4_STARTING_TIMESTEPS + ADDITIONAL_TIMESTEPS
SIMULATION_SPEED = 100.0
TENSORBOARD_RUN_NAME = "reward_v5_steering_rate_smoothness"
STUCK_WINDOW_MS = 2_000
STUCK_PROGRESS_GAIN_UNITS = 1.0
STUCK_WORLD_DISTANCE_UNITS = 2.0
REWARD_CONTRACT = (
    "clip(progress_delta, -20, 20) / 10 - 0.10 "
    "- 0.05 * abs(steer_t - steer_t_minus_1) "
    "+ 50 on finish - 250 on verified failure; first step rate cost is zero"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Run one million additional {EXPERIMENT_LABEL} PPO steps from V4."
    )
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def verify_v4_checkpoint() -> dict[str, Any]:
    if not V4_CHECKPOINT.is_file():
        raise ProtocolError(f"V4 initialization checkpoint is missing: {V4_CHECKPOINT}")
    actual_hash = sha256(V4_CHECKPOINT)
    if actual_hash != EXPECTED_V4_CHECKPOINT_SHA256:
        raise ProtocolError(
            "V4 initialization checkpoint hash changed: expected "
            f"{EXPECTED_V4_CHECKPOINT_SHA256}, got {actual_hash}"
        )
    return {
        "path": str(V4_CHECKPOINT.relative_to(WORKSPACE_ROOT)),
        "size_bytes": V4_CHECKPOINT.stat().st_size,
        "sha256": actual_hash,
        "num_timesteps": V4_STARTING_TIMESTEPS,
    }


def formal_tensorboard_event_files() -> list[Path]:
    return sorted(
        path
        for path in TENSORBOARD_ROOT.rglob("events.out.tfevents.*")
        if re.fullmatch(
            rf"{re.escape(TENSORBOARD_RUN_NAME)}_\d+",
            path.parent.name,
        )
    )


def read_monitor_rows() -> list[dict[str, str]]:
    monitor_files = sorted(RUN_DIR.glob("*.monitor.csv"))
    if not monitor_files:
        raise ProtocolError(f"{EXPERIMENT_LABEL} run did not create a Monitor CSV")
    rows: list[dict[str, str]] = []
    for path in monitor_files:
        with path.open("r", encoding="utf-8", newline="") as source:
            data_lines = [line for line in source if not line.startswith("#")]
        rows.extend(csv.DictReader(data_lines))
    if not rows:
        raise ProtocolError(f"{EXPERIMENT_LABEL} Monitor CSV contains no completed episodes")
    return rows


def main() -> int:
    args = parse_args()
    if args.resume is not None:
        args.resume = args.resume.resolve()
        if not args.resume.is_file():
            raise SystemExit(f"resume checkpoint does not exist: {args.resume}")
    if not args.resume and (
        MANIFEST_PATH.exists()
        or formal_tensorboard_event_files()
        or any(CHECKPOINT_DIR.glob("*.zip"))
    ):
        raise SystemExit(
            f"{EXPERIMENT_LABEL} artifacts already exist; preserve them and use --resume"
        )

    v4_checkpoint = verify_v4_checkpoint()
    power_preflight = verify_sleep_disabled()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    attempt_started_at = datetime.now(timezone.utc)
    previous_manifest: dict[str, Any] = {}
    if args.resume:
        if not MANIFEST_PATH.exists():
            raise SystemExit("--resume requires the existing reward-v5 manifest")
        previous_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    original_started_at = str(
        previous_manifest.get(
            "original_started_at_utc",
            previous_manifest.get("started_at_utc", attempt_started_at.isoformat()),
        )
    )
    attempt_number = int(
        previous_manifest.get("attempt_number", 1 if previous_manifest else 0)
    ) + 1
    cumulative_wall_before = prior_wall_seconds(previous_manifest)
    previous_failure_timesteps = int(
        previous_manifest.get("timesteps_at_failure", V4_STARTING_TIMESTEPS)
    )
    discarded_steps_before = int(
        previous_manifest.get("discarded_steps_total", 0)
    )
    manifest: dict[str, Any] = {
        "status": "starting",
        "started_at_utc": attempt_started_at.isoformat(),
        "original_started_at_utc": original_started_at,
        "attempt_number": attempt_number,
        "cumulative_wall_seconds_before_attempt": cumulative_wall_before,
        "v4_initialization": v4_checkpoint,
        "additional_timesteps_requested": ADDITIONAL_TIMESTEPS,
        "target_total_timesteps": TARGET_TOTAL_TIMESTEPS,
        "reward_function": REWARD_FUNCTION_NAME,
        "reward_contract": REWARD_CONTRACT,
        **REWARD_METADATA,
        "tensorboard_run_name": TENSORBOARD_RUN_NAME,
        "simulation_speed": SIMULATION_SPEED,
        "step_period_ms": 100,
        "bridge_response_timeout_ms": 30_000,
        "max_episode_ms": 45_000,
        "max_lateral_offset": 50.0,
        "max_vertical_drop": 10.0,
        "fall_confirmation": "same 2-second no-progress/no-motion window as V4",
        "n_steps": 2_048,
        "batch_size": 64,
        "n_epochs": 10,
        "learning_rate": 0.0003,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "checkpoint_interval_additional_steps": 50_000,
        "stuck_window_ms": STUCK_WINDOW_MS,
        "stuck_progress_gain_units": STUCK_PROGRESS_GAIN_UNITS,
        "stuck_world_distance_units": STUCK_WORLD_DISTANCE_UNITS,
        "legacy_reversed_pedal_mapping": False,
        "power_preflight": power_preflight,
        "reward_doc_sha256": sha256(REWARD_DOC),
        "resume_checkpoint": str(args.resume) if args.resume else None,
    }
    write_json(MANIFEST_PATH, manifest)

    try:
        closed = close_trackmania()
        if closed:
            print("closed stale TrackMania session", flush=True)
        _, launched = ensure_trackmania_running(
            port=args.port,
            confirm_existing=True,
        )
        print(f"TrackMania ready (launched={launched})", flush=True)
    except Exception as error:
        manifest.update(
            {
                "status": "failed_before_training",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(error),
                "timesteps_at_failure": V4_STARTING_TIMESTEPS,
                "cumulative_wall_seconds": cumulative_wall_before,
            }
        )
        write_json(MANIFEST_PATH, manifest)
        raise

    config = EnvironmentConfig(
        port=args.port,
        simulation_speed=SIMULATION_SPEED,
        stuck_window_ms=STUCK_WINDOW_MS,
        stuck_progress_gain_units=STUCK_PROGRESS_GAIN_UNITS,
        stuck_world_distance_units=STUCK_WORLD_DISTANCE_UNITS,
        legacy_reversed_pedal_mapping=False,
        map_to_load="A01-Race.Challenge.Gbx",
        auto_respawn_on_connect=False,
        wait_for_race_start_on_connect=True,
    )
    base_env = TrackmaniaEnv(
        config=config,
        reward_function=REWARD_FUNCTION,
    )
    monitored_env = Monitor(
        base_env,
        filename=str(MONITOR_PREFIX),
        info_keywords=(
            "timeout",
            "off_track",
            "fallen",
            "stuck",
            "race_finished",
            "progress",
            "display_speed",
            "steering_rate_change",
        ),
        override_existing=args.resume is None,
    )
    load_path = args.resume if args.resume else V4_CHECKPOINT
    model = PPO.load(
        load_path,
        env=monitored_env,
        device="cpu",
        tensorboard_log=str(TENSORBOARD_ROOT),
    )
    starting_timesteps = int(model.num_timesteps)
    if not args.resume and starting_timesteps != V4_STARTING_TIMESTEPS:
        raise ProtocolError(
            f"pinned V4 checkpoint reports {starting_timesteps} timesteps, "
            f"expected {V4_STARTING_TIMESTEPS}"
        )
    if starting_timesteps < V4_STARTING_TIMESTEPS:
        raise ProtocolError(f"{EXPERIMENT_LABEL} resume checkpoint predates the pinned V4 model")
    if starting_timesteps >= TARGET_TOTAL_TIMESTEPS:
        raise SystemExit(
            f"checkpoint already has {starting_timesteps} timesteps; "
            f"no formal {EXPERIMENT_LABEL} training remains"
        )

    discarded_steps_total = discarded_steps_before + max(
        0, previous_failure_timesteps - starting_timesteps
    )
    manifest.update(
        {
            "status": "running",
            "attempt_starting_timesteps": starting_timesteps,
            "discarded_steps_total": discarded_steps_total,
        }
    )
    write_json(MANIFEST_PATH, manifest)
    remaining_timesteps = TARGET_TOTAL_TIMESTEPS - starting_timesteps

    action_stats = BoundedPpoActionStatsCallback()
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=50_000,
                save_path=str(CHECKPOINT_DIR),
                name_prefix=CHECKPOINT_NAME_PREFIX,
            ),
            action_stats,
            PeriodicProgressCallback(),
        ]
    )
    print(
        f"{EXPERIMENT_LABEL} training started: initial={starting_timesteps}, "
        f"target={TARGET_TOTAL_TIMESTEPS}, remaining={remaining_timesteps}",
        flush=True,
    )
    wall_started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=callbacks,
            tb_log_name=TENSORBOARD_RUN_NAME,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        model.save(FINAL_CHECKPOINT)
    except Exception as error:
        attempt_wall_seconds = time.perf_counter() - wall_started
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(error),
                "timesteps_at_failure": int(model.num_timesteps),
                "attempt_wall_seconds": attempt_wall_seconds,
                "cumulative_wall_seconds": (
                    cumulative_wall_before + attempt_wall_seconds
                ),
            }
        )
        write_json(MANIFEST_PATH, manifest)
        raise
    finally:
        monitored_env.close()
        close_model_logger(model)
    wall_seconds = time.perf_counter() - wall_started

    event_files = formal_tensorboard_event_files()
    if not event_files:
        raise ProtocolError(f"{EXPERIMENT_LABEL} training created no TensorBoard event file")
    tensorboard_metrics = read_tensorboard_metrics(event_files)
    monitor_rows = read_monitor_rows()
    episode_rewards = [float(row["r"]) for row in monitor_rows]
    episode_lengths = [int(row["l"]) for row in monitor_rows]
    terminal_progress = [float(row["progress"]) for row in monitor_rows]
    terminal_speeds = [int(row["display_speed"]) for row in monitor_rows]
    terminal_steering_rate = [
        float(row["steering_rate_change"]) for row in monitor_rows
    ]
    finishes = sum(row["race_finished"] == "True" for row in monitor_rows)
    timeouts = sum(row["timeout"] == "True" for row in monitor_rows)
    off_tracks = sum(row["off_track"] == "True" for row in monitor_rows)
    falls = sum(row["fallen"] == "True" for row in monitor_rows)
    stuck_truncations = sum(row["stuck"] == "True" for row in monitor_rows)
    final_checkpoint_path = FINAL_CHECKPOINT.with_suffix(".zip")
    checkpoint_paths = sorted(CHECKPOINT_DIR.glob("*.zip"))
    completed_at = datetime.now(timezone.utc)

    summary = {
        "status": "complete",
        "started_at_utc": original_started_at,
        "final_attempt_started_at_utc": attempt_started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "wall_seconds": cumulative_wall_before + wall_seconds,
        "final_attempt_wall_seconds": wall_seconds,
        "attempts": attempt_number,
        "v4_initialization": v4_checkpoint,
        "additional_timesteps_requested": ADDITIONAL_TIMESTEPS,
        "starting_timesteps": V4_STARTING_TIMESTEPS,
        "final_model_timesteps": int(model.num_timesteps),
        "actual_additional_timesteps": (
            int(model.num_timesteps) - V4_STARTING_TIMESTEPS
        ),
        "discarded_steps_after_checkpoint": discarded_steps_total,
        "environment_interactions_observed": (
            int(model.num_timesteps)
            - V4_STARTING_TIMESTEPS
            + discarded_steps_total
        ),
        "reward_function": REWARD_FUNCTION_NAME,
        "reward_contract": REWARD_CONTRACT,
        **REWARD_METADATA,
        "episodes": len(monitor_rows),
        "finishes": finishes,
        "timeouts": timeouts,
        "off_tracks": off_tracks,
        "falls": falls,
        "stuck_truncations": stuck_truncations,
        "episode_reward_minimum": min(episode_rewards),
        "episode_reward_maximum": max(episode_rewards),
        "episode_length_minimum": min(episode_lengths),
        "episode_length_maximum": max(episode_lengths),
        "terminal_progress_minimum": min(terminal_progress),
        "terminal_progress_maximum": max(terminal_progress),
        "terminal_speed_minimum": min(terminal_speeds),
        "terminal_speed_maximum": max(terminal_speeds),
        "terminal_steering_rate_minimum": min(terminal_steering_rate),
        "terminal_steering_rate_maximum": max(terminal_steering_rate),
        "action_validation": action_stats.summary(),
        "action_validation_segment_started_at_timestep": starting_timesteps,
        "tensorboard_metrics": tensorboard_metrics,
        "power_preflight": power_preflight,
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
            "final_model_timesteps": int(model.num_timesteps),
            "actual_additional_timesteps": (
                int(model.num_timesteps) - V4_STARTING_TIMESTEPS
            ),
            "cumulative_wall_seconds": cumulative_wall_before + wall_seconds,
            "discarded_steps_total": discarded_steps_total,
            "summary_sha256": sha256(SUMMARY_PATH),
        }
    )
    write_json(MANIFEST_PATH, manifest)

    print(
        f"{EXPERIMENT_LABEL} training complete: additional="
        f"{int(model.num_timesteps) - V4_STARTING_TIMESTEPS}, "
        f"episodes={len(monitor_rows)}, finishes={finishes}",
        flush=True,
    )
    print(f"summary SHA-256={sha256(SUMMARY_PATH)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
