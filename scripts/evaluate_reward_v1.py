"""Run the fixed deterministic reward-v1 evaluation and record behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.rewards import sparse_finish_reward
from trackmania_rl.tmi_bridge import ProtocolError

RUN_DIR = WORKSPACE_ROOT / "runs" / "reward_v1"
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "reward_v1" / "final_model.zip"
DEFAULT_ACTION_LOG = RUN_DIR / "evaluation_actions.jsonl"
DEFAULT_SUMMARY = RUN_DIR / "evaluation_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reward v1 for at least 20 deterministic episodes."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=20)
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


def describe_behavior(episodes: list[dict[str, Any]]) -> list[str]:
    finishes = sum(bool(episode["finished"]) for episode in episodes)
    falls = sum(bool(episode.get("fallen", False)) for episode in episodes)
    best_progress = max(float(episode["max_progress"]) for episode in episodes)
    moving_episodes = sum(
        float(episode["max_progress"]) >= 10.0
        or int(episode["max_display_speed"]) >= 20
        for episode in episodes
    )
    descriptions = [
        f"Finished {finishes} of {len(episodes)} deterministic episodes.",
        f"Meaningful motion threshold was reached in {moving_episodes} episodes.",
        f"Best projected path progress was {best_progress:.3f} units.",
    ]
    if finishes == 0 and falls == len(episodes) and best_progress < 1.0:
        descriptions.append(
            "Telemetry indicates the deterministic policy made no forward path "
            "progress and fell in every episode."
        )
    elif finishes == 0 and moving_episodes == 0:
        descriptions.append(
            "Telemetry indicates the deterministic policy remained effectively "
            "stationary in every episode."
        )
    elif finishes == 0:
        descriptions.append(
            "Telemetry indicates movement without a completed lap; inspect the "
            "visible evaluation for steering, looping, or oscillation patterns."
        )
    return descriptions


def main() -> int:
    args = parse_args()
    if args.episodes < 20:
        raise SystemExit("reward-v1 evaluation requires at least 20 episodes")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")

    env = TrackmaniaEnv(
        config=EnvironmentConfig(
            port=args.port,
            legacy_reversed_pedal_mapping=True,
        ),
        reward_function=sparse_finish_reward,
        action_log_path=args.action_log,
    )
    model = PPO.load(args.checkpoint, device="cpu")
    episode_records: list[dict[str, Any]] = []
    try:
        for episode in range(args.episodes):
            observation, reset_info = env.reset()
            finished = False
            truncated = False
            steps = 0
            total_reward = 0.0
            max_progress = float(reset_info["progress"])
            max_display_speed = int(reset_info["display_speed"])
            final_info = reset_info
            while not (finished or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, finished, truncated, final_info = env.step(action)
                if not np.isfinite(observation).all() or not np.isfinite(reward):
                    raise ProtocolError(
                        f"evaluation episode {episode} produced nonfinite data"
                    )
                steps += 1
                total_reward += float(reward)
                max_progress = max(max_progress, float(final_info["progress"]))
                max_display_speed = max(
                    max_display_speed,
                    int(final_info["display_speed"]),
                )
                if steps > 451:
                    raise ProtocolError(
                        f"evaluation episode {episode} exceeded its safety limit"
                    )

            episode_record = {
                "episode": episode,
                "steps": steps,
                "elapsed_ms": int(final_info["elapsed_ms"]),
                "total_reward": total_reward,
                "finished": bool(finished),
                "timeout": bool(final_info["timeout"]),
                "off_track": bool(final_info["off_track"]),
                "fallen": bool(final_info["fallen"]),
                "max_progress": max_progress,
                "final_progress": float(final_info["progress"]),
                "max_display_speed": max_display_speed,
                "final_lateral_offset": float(final_info["lateral_offset"]),
                "final_heading_error": float(final_info["heading_error"]),
            }
            episode_records.append(episode_record)
            print(
                f"evaluation episode={episode + 1}/{args.episodes} "
                f"finished={finished} steps={steps} "
                f"max_progress={max_progress:.3f}",
                flush=True,
            )
    finally:
        env.close()

    action_records = [
        json.loads(line)
        for line in args.action_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_actions = sum(int(episode["steps"]) for episode in episode_records)
    if len(action_records) != expected_actions:
        raise ProtocolError(
            f"evaluation action log has {len(action_records)} records for "
            f"{expected_actions} steps"
        )
    if not all(
        record.get("finite")
        and record.get("within_range")
        and record.get("valid")
        for record in action_records
    ):
        raise ProtocolError("evaluation action log contains an invalid action")

    raw_actions = np.asarray(
        [record["raw_action"] for record in action_records],
        dtype=np.float64,
    )
    finishes = sum(bool(episode["finished"]) for episode in episode_records)
    timeouts = sum(bool(episode["timeout"]) for episode in episode_records)
    off_tracks = sum(bool(episode["off_track"]) for episode in episode_records)
    falls = sum(bool(episode["fallen"]) for episode in episode_records)
    finish_times_ms = [
        int(episode["elapsed_ms"])
        for episode in episode_records
        if episode["finished"]
    ]
    summary = {
        "episodes": args.episodes,
        "deterministic": True,
        "checkpoint": str(args.checkpoint.relative_to(WORKSPACE_ROOT)),
        "checkpoint_sha256": sha256(args.checkpoint),
        "finishes": finishes,
        "finish_rate": finishes / args.episodes,
        "timeouts": timeouts,
        "off_tracks": off_tracks,
        "falls": falls,
        "average_episode_steps": statistics.fmean(
            int(episode["steps"]) for episode in episode_records
        ),
        "minimum_episode_steps": min(
            int(episode["steps"]) for episode in episode_records
        ),
        "maximum_episode_steps": max(
            int(episode["steps"]) for episode in episode_records
        ),
        "average_max_progress": statistics.fmean(
            float(episode["max_progress"]) for episode in episode_records
        ),
        "best_max_progress": max(
            float(episode["max_progress"]) for episode in episode_records
        ),
        "best_finish_time_ms": min(finish_times_ms) if finish_times_ms else None,
        "action_records": len(action_records),
        "action_minimum": raw_actions.min(axis=0).tolist(),
        "action_maximum": raw_actions.max(axis=0).tolist(),
        "action_mean": raw_actions.mean(axis=0).tolist(),
        "behavior_description": describe_behavior(episode_records),
        "episodes_detail": episode_records,
        "action_log_sha256": sha256(args.action_log),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"reward-v1 evaluation complete: finishes={finishes}/{args.episodes}, "
        f"best_progress={summary['best_max_progress']:.3f}",
        flush=True,
    )
    print(f"summary SHA-256={sha256(args.summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
