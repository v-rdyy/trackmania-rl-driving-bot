"""Evaluate reward v2 and detect speed-farming loop/oscillation behavior."""

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
from trackmania_rl.rewards import dense_speed_reward
from trackmania_rl.tmi_bridge import ProtocolError

RUN_DIR = WORKSPACE_ROOT / "runs" / "reward_v2"
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "reward_v2" / "final_model.zip"
DEFAULT_ACTION_LOG = RUN_DIR / "evaluation_actions.jsonl"
DEFAULT_SUMMARY = RUN_DIR / "evaluation_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reward v2 for 20 deterministic episodes at 6x."
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


def steering_sign_changes(steering: np.ndarray, deadzone: float = 0.05) -> int:
    signs = np.sign(np.where(np.abs(steering) >= deadzone, steering, 0.0))
    nonzero = signs[signs != 0.0]
    if len(nonzero) < 2:
        return 0
    return int(np.count_nonzero(nonzero[1:] != nonzero[:-1]))


def trajectory_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ProtocolError("cannot analyze an empty evaluation trajectory")
    positions = np.asarray([record["position"] for record in records], dtype=np.float64)
    progress = np.asarray([record["progress"] for record in records], dtype=np.float64)
    speeds = np.asarray([record["display_speed"] for record in records], dtype=np.float64)
    actions = np.asarray([record["raw_action"] for record in records], dtype=np.float64)
    if not np.isfinite(np.concatenate((positions.ravel(), progress, speeds, actions.ravel()))).all():
        raise ProtocolError("evaluation trajectory contains nonfinite telemetry")

    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    world_distance = float(segment_lengths.sum())
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    forward_progress = max(0.0, float(progress.max() - progress[0]))
    progress_efficiency = forward_progress / max(world_distance, 1e-9)
    progress_deltas = np.diff(progress)
    backward_progress = float(np.maximum(-progress_deltas, 0.0).sum())
    lookback = max(1, min(10, len(positions) // 4))
    prior_positions = positions[:-lookback]
    minimum_return_distance = (
        float(np.linalg.norm(prior_positions - positions[-1], axis=1).min())
        if len(prior_positions)
        else displacement
    )
    sign_changes = steering_sign_changes(actions[:, 0])
    average_speed = float(speeds.mean())

    sustained_motion = world_distance >= 100.0 and average_speed >= 40.0
    poor_progress = progress_efficiency < 0.25
    loop_candidate = bool(
        sustained_motion and poor_progress and minimum_return_distance <= 20.0
    )
    oscillation_candidate = bool(
        sustained_motion and poor_progress and sign_changes >= 6
    )
    speed_farming_candidate = bool(
        sustained_motion and poor_progress and (loop_candidate or oscillation_candidate)
    )
    return {
        "logged_world_distance": world_distance,
        "logged_start_to_end_displacement": displacement,
        "minimum_return_distance": minimum_return_distance,
        "first_progress": float(progress[0]),
        "maximum_progress": float(progress.max()),
        "final_progress": float(progress[-1]),
        "forward_progress": forward_progress,
        "backward_progress": backward_progress,
        "progress_efficiency": progress_efficiency,
        "average_display_speed": average_speed,
        "maximum_display_speed": int(speeds.max()),
        "steering_sign_changes": sign_changes,
        "mean_absolute_steering": float(np.abs(actions[:, 0]).mean()),
        "mean_throttle": float(actions[:, 1].mean()),
        "mean_brake": float(actions[:, 2].mean()),
        "loop_candidate": loop_candidate,
        "oscillation_candidate": oscillation_candidate,
        "speed_farming_candidate": speed_farming_candidate,
    }


def describe_behavior(episodes: list[dict[str, Any]]) -> list[str]:
    finishes = sum(bool(episode["finished"]) for episode in episodes)
    exploit_candidates = sum(
        bool(episode["trajectory"]["speed_farming_candidate"])
        for episode in episodes
    )
    falls = sum(bool(episode["fallen"]) for episode in episodes)
    timeouts = sum(bool(episode["timeout"]) for episode in episodes)
    best_progress = max(
        float(episode["trajectory"]["maximum_progress"]) for episode in episodes
    )
    descriptions = [
        f"Finished {finishes} of {len(episodes)} deterministic episodes.",
        f"Telemetry flagged {exploit_candidates} speed-farming loop/oscillation candidates.",
        f"Terminal causes included {falls} falls and {timeouts} timeouts.",
        f"Best projected path progress was {best_progress:.3f} units.",
    ]
    if exploit_candidates:
        descriptions.append(
            "Telemetry supports sustained speed with poor progress efficiency; "
            "compare the visible evaluation against looping/oscillation."
        )
    elif finishes == 0 and falls == len(episodes):
        descriptions.append(
            "Every evaluation ended in a fall before the sustained-motion "
            "threshold, so no looping/oscillation exploit was established."
        )
    elif finishes == 0:
        descriptions.append(
            "No deterministic finish or telemetry-qualified loop/oscillation "
            "exploit was observed; inspect the visible trajectory."
        )
    return descriptions


def main() -> int:
    args = parse_args()
    if args.episodes != 20:
        raise SystemExit("Decision 0007 fixes reward-v2 evaluation at 20 episodes")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")

    env = TrackmaniaEnv(
        config=EnvironmentConfig(
            port=args.port,
            simulation_speed=6.0,
            legacy_reversed_pedal_mapping=True,
        ),
        reward_function=dense_speed_reward,
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
                if steps > 451:
                    raise ProtocolError(
                        f"evaluation episode {episode} exceeded its safety limit"
                    )
            episode_records.append(
                {
                    "episode": episode,
                    "steps": steps,
                    "elapsed_ms": int(final_info["elapsed_ms"]),
                    "total_reward": total_reward,
                    "finished": bool(finished),
                    "timeout": bool(final_info["timeout"]),
                    "off_track": bool(final_info["off_track"]),
                    "fallen": bool(final_info["fallen"]),
                }
            )
            print(
                f"evaluation episode={episode + 1}/{args.episodes} "
                f"finished={finished} steps={steps} reward={total_reward:.3f}",
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

    records_by_episode: dict[int, list[dict[str, Any]]] = {
        episode: [] for episode in range(args.episodes)
    }
    for record in action_records:
        records_by_episode[int(record["episode"])].append(record)
    for episode in episode_records:
        episode["trajectory"] = trajectory_metrics(
            records_by_episode[int(episode["episode"])]
        )

    raw_actions = np.asarray(
        [record["raw_action"] for record in action_records],
        dtype=np.float64,
    )
    finishes = sum(bool(episode["finished"]) for episode in episode_records)
    timeouts = sum(bool(episode["timeout"]) for episode in episode_records)
    off_tracks = sum(bool(episode["off_track"]) for episode in episode_records)
    falls = sum(bool(episode["fallen"]) for episode in episode_records)
    exploit_candidates = sum(
        bool(episode["trajectory"]["speed_farming_candidate"])
        for episode in episode_records
    )
    finish_times_ms = [
        int(episode["elapsed_ms"])
        for episode in episode_records
        if episode["finished"]
    ]
    summary = {
        "episodes": args.episodes,
        "deterministic": True,
        "simulation_speed": 6.0,
        "checkpoint": str(args.checkpoint.relative_to(WORKSPACE_ROOT)),
        "checkpoint_sha256": sha256(args.checkpoint),
        "finishes": finishes,
        "finish_rate": finishes / args.episodes,
        "timeouts": timeouts,
        "off_tracks": off_tracks,
        "falls": falls,
        "speed_farming_candidate_episodes": exploit_candidates,
        "average_episode_steps": statistics.fmean(
            int(episode["steps"]) for episode in episode_records
        ),
        "minimum_episode_steps": min(
            int(episode["steps"]) for episode in episode_records
        ),
        "maximum_episode_steps": max(
            int(episode["steps"]) for episode in episode_records
        ),
        "average_episode_reward": statistics.fmean(
            float(episode["total_reward"]) for episode in episode_records
        ),
        "minimum_episode_reward": min(
            float(episode["total_reward"]) for episode in episode_records
        ),
        "maximum_episode_reward": max(
            float(episode["total_reward"]) for episode in episode_records
        ),
        "average_logged_world_distance": statistics.fmean(
            float(episode["trajectory"]["logged_world_distance"])
            for episode in episode_records
        ),
        "average_forward_progress": statistics.fmean(
            float(episode["trajectory"]["forward_progress"])
            for episode in episode_records
        ),
        "best_max_progress": max(
            float(episode["trajectory"]["maximum_progress"])
            for episode in episode_records
        ),
        "best_finish_time_ms": min(finish_times_ms) if finish_times_ms else None,
        "action_records": len(action_records),
        "action_minimum": raw_actions.min(axis=0).tolist(),
        "action_maximum": raw_actions.max(axis=0).tolist(),
        "action_mean": raw_actions.mean(axis=0).tolist(),
        "behavior_description": describe_behavior(episode_records),
        "visible_observation": "Pending project-owner observation during 6x evaluation.",
        "episodes_detail": episode_records,
        "action_log_sha256": sha256(args.action_log),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"reward-v2 evaluation complete: finishes={finishes}/{args.episodes}, "
        f"exploit_candidates={exploit_candidates}/{args.episodes}",
        flush=True,
    )
    print(f"summary SHA-256={sha256(args.summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
