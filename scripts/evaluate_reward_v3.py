"""Evaluate reward v3 reliability and final-jump precision."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.game_launch import close_trackmania, ensure_trackmania_running
from trackmania_rl.evaluation_metrics import (
    aggregate_precision_metrics,
    evaluation_metric_protocol,
    steering_precision_metrics,
    trajectory_precision_metrics,
)
from trackmania_rl.rewards import clamped_forward_progress_reward
from trackmania_rl.tmi_bridge import ProtocolError

EXPERIMENT_LABEL = "reward-v3"
EXPERIMENT_SLUG = "reward_v3"
PROTOCOL_LABEL = "Decision 0009"
REWARD_FUNCTION = clamped_forward_progress_reward
RUN_DIR = WORKSPACE_ROOT / "runs" / "reward_v3"
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "reward_v3" / "final_model.zip"
DEFAULT_ACTION_LOG = RUN_DIR / "evaluation_actions.jsonl"
DEFAULT_SUMMARY = RUN_DIR / "evaluation_summary.json"
DEFAULT_REPLAY_DIR = WORKSPACE_ROOT / "artifacts" / "replays" / "reward_v3_evaluation"
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"
HUMAN_PB_MS = 24_500
FINAL_JUMP_PROGRESS = 1_700.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Evaluate {EXPERIMENT_LABEL} for 20 deterministic episodes at 6x."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--action-log", type=Path, default=DEFAULT_ACTION_LOG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR)
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument(
        "--run-tag",
        default=None,
        help="optional identifier added to replay filenames and the summary",
    )
    parser.add_argument(
        "--reuse-game",
        action="store_true",
        help="reuse an existing game instead of forcing a clean bridge lifecycle",
    )
    parser.add_argument(
        "--postprocess-existing",
        action="store_true",
        help="build the summary from an already completed action log and replay set",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def wait_for_replay(path: Path, timeout_seconds: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout_seconds
    while time.perf_counter() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        time.sleep(0.05)
    raise ProtocolError(f"TMInterface did not create input replay: {path}")


def evaluated_race_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Drop a disclosed pre-race prefix when the raw clock restarts countdown."""
    resets = [
        index
        for index in range(1, len(records))
        if int(records[index]["race_time_ms"])
        < int(records[index - 1]["race_time_ms"])
    ]
    if not resets:
        return records, 0
    start = resets[-1]
    while start < len(records) and int(records[start]["race_time_ms"]) < 0:
        start += 1
    if start >= len(records):
        raise ProtocolError("race clock restarted without reaching active race time")
    return records[start:], start


def trajectory_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ProtocolError("cannot analyze an empty evaluation trajectory")
    positions = np.asarray([record["position"] for record in records], dtype=np.float64)
    progress = np.asarray([record["progress"] for record in records], dtype=np.float64)
    speeds = np.asarray([record["display_speed"] for record in records], dtype=np.float64)
    actions = np.asarray([record["raw_action"] for record in records], dtype=np.float64)
    vertical = np.asarray(
        [record["vertical_offset"] for record in records],
        dtype=np.float64,
    )
    upright = np.asarray(
        [record["upright_cosine"] for record in records],
        dtype=np.float64,
    )
    if not np.isfinite(
        np.concatenate(
            (positions.ravel(), progress, speeds, actions.ravel(), vertical, upright)
        )
    ).all():
        raise ProtocolError("evaluation trajectory contains nonfinite telemetry")

    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    world_distance = float(segment_lengths.sum())
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    forward_progress = max(0.0, float(progress.max() - progress[0]))
    progress_efficiency = forward_progress / max(world_distance, 1e-9)
    progress_deltas = np.diff(progress)
    backward_progress = float(np.maximum(-progress_deltas, 0.0).sum())
    average_speed = float(speeds.mean())
    steering_samples = [
        (
            float(record.get("step", index)) / 10.0,
            float(record["raw_action"][0]),
        )
        for index, record in enumerate(records)
    ]
    metric_records = [
        {**record, "race_time_ms": int(record.get("step", index)) * 100}
        for index, record in enumerate(records)
    ]
    precision = {
        "steering": steering_precision_metrics(steering_samples),
        **trajectory_precision_metrics(metric_records),
    }

    return {
        "logged_world_distance": world_distance,
        "logged_start_to_end_displacement": displacement,
        "first_progress": float(progress[0]),
        "maximum_progress": float(progress.max()),
        "final_progress": float(progress[-1]),
        "forward_progress": forward_progress,
        "backward_progress": backward_progress,
        "progress_efficiency": progress_efficiency,
        "average_display_speed": average_speed,
        "maximum_display_speed": int(speeds.max()),
        "mean_absolute_steering": float(np.abs(actions[:, 0]).mean()),
        "mean_throttle": float(actions[:, 1].mean()),
        "mean_brake": float(actions[:, 2].mean()),
        "minimum_vertical_offset": float(vertical.min()),
        "terminal_vertical_offset": float(vertical[-1]),
        "minimum_upright_cosine": float(upright.min()),
        "terminal_upright_cosine": float(upright[-1]),
        "terminal_position": positions[-1].tolist(),
        "precision": precision,
    }


def describe_behavior(episodes: list[dict[str, Any]]) -> list[str]:
    finishes = sum(bool(episode["finished"]) for episode in episodes)
    oscillation_episodes = sum(
        bool(episode["trajectory"]["precision"]["steering"]["oscillation_detected"])
        for episode in episodes
    )
    upside_down_episodes = sum(
        bool(episode["trajectory"]["precision"]["upside_down"]["upside_down_detected"])
        for episode in episodes
    )
    stuck_episodes = sum(
        bool(episode["trajectory"]["precision"]["stuck"]["stuck_detected"])
        for episode in episodes
    )
    falls = sum(bool(episode["fallen"]) for episode in episodes)
    late_falls = sum(
        bool(episode["fallen"])
        and float(episode["trajectory"]["final_progress"])
        >= FINAL_JUMP_PROGRESS
        for episode in episodes
    )
    stuck_truncations = sum(bool(episode["stuck"]) for episode in episodes)
    timeouts = sum(bool(episode["timeout"]) for episode in episodes)
    best_progress = max(
        float(episode["trajectory"]["maximum_progress"]) for episode in episodes
    )
    descriptions = [
        f"Finished {finishes} of {len(episodes)} deterministic episodes.",
        f"Fixed precision metrics flagged steering oscillation in {oscillation_episodes}, "
        f"inversion in {upside_down_episodes}, and stuck periods in {stuck_episodes} episodes.",
        f"Terminal causes included {falls} falls, {stuck_truncations} stuck "
        f"cutoffs, and {timeouts} timeouts.",
        f"{late_falls} falls terminated at or after final-jump progress "
        f"{FINAL_JUMP_PROGRESS:.0f}.",
        f"Best projected path progress was {best_progress:.3f} units.",
    ]
    if finishes == 0 and falls == len(episodes):
        descriptions.append(
            "Every deterministic evaluation ended in a fall; inspect preserved "
            "replays for checkpoint-frame contact versus a lower valid jump arc."
        )
    elif finishes == 0:
        descriptions.append(
            "No deterministic finish was observed; inspect the live view and "
            "preserved input replays for the terminal mechanism."
        )
    return descriptions


def main() -> int:
    args = parse_args()
    for path_argument in (
        "checkpoint",
        "action_log",
        "summary",
        "replay_dir",
        "tmi_scripts_dir",
    ):
        setattr(args, path_argument, getattr(args, path_argument).resolve())
    if args.episodes != 20:
        raise SystemExit(
            f"{PROTOCOL_LABEL} fixes {EXPERIMENT_LABEL} evaluation at 20 episodes"
        )
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    if not args.tmi_scripts_dir.is_dir():
        raise SystemExit(
            f"TMInterface Scripts directory does not exist: {args.tmi_scripts_dir}"
        )
    if (
        args.run_tag is not None
        and re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag) is None
    ):
        raise SystemExit(
            "--run-tag must contain only letters, numbers, underscores, or hyphens"
        )

    if not args.postprocess_existing:
        if not args.reuse_game:
            closed = close_trackmania()
            if closed:
                print("closed stale TrackMania session", flush=True)
        _, launched = ensure_trackmania_running(
            port=args.port,
            confirm_existing=True,
        )
        print(f"TrackMania ready (launched={launched})", flush=True)

    checkpoint_hash = sha256(args.checkpoint)
    replay_prefix = f"{EXPERIMENT_SLUG}_final_{checkpoint_hash[:8].lower()}"
    if args.run_tag is not None:
        replay_prefix = f"{replay_prefix}_{args.run_tag}"
    args.replay_dir.mkdir(parents=True, exist_ok=True)
    replay_targets = []
    for episode in range(args.episodes):
        filename = f"{replay_prefix}_ep_{episode + 1:02d}.txt"
        local_path = args.replay_dir / filename
        external_path = args.tmi_scripts_dir / filename
        if args.postprocess_existing:
            if not local_path.is_file():
                raise SystemExit(
                    f"missing completed {EXPERIMENT_LABEL} replay: {local_path}"
                )
        elif local_path.exists() or external_path.exists():
            raise SystemExit(
                f"refusing to overwrite existing {EXPERIMENT_LABEL} "
                f"evaluation replay: {filename}"
            )
        replay_targets.append((filename, local_path, external_path))

    episode_records: list[dict[str, Any]] = []
    if not args.postprocess_existing:
        env = TrackmaniaEnv(
            config=EnvironmentConfig(
                port=args.port,
                simulation_speed=6.0,
                stuck_window_ms=2_000,
                stuck_progress_gain_units=1.0,
                stuck_world_distance_units=2.0,
                legacy_reversed_pedal_mapping=False,
                map_to_load="A01-Race.Challenge.Gbx",
                auto_respawn_on_connect=False,
                wait_for_race_start_on_connect=True,
            ),
            reward_function=REWARD_FUNCTION,
            action_log_path=args.action_log,
        )
        model = PPO.load(args.checkpoint, device="cpu")
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
                    observation, reward, finished, truncated, final_info = env.step(
                        action
                    )
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
                replay_filename, local_replay, external_replay = replay_targets[episode]
                env.session.recover_inputs(replay_filename)
                wait_for_replay(external_replay)
                shutil.copy2(external_replay, local_replay)
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
                        "stuck": bool(final_info["stuck"]),
                        "final_progress": float(final_info["progress"]),
                        "final_vertical_offset": float(final_info["vertical_offset"]),
                        "input_replay": str(local_replay.relative_to(WORKSPACE_ROOT)),
                        "input_replay_bytes": local_replay.stat().st_size,
                        "input_replay_sha256": sha256(local_replay),
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
    evaluated_by_episode: dict[int, list[dict[str, Any]]] = {}
    discarded_startup_records = 0
    for episode, records in records_by_episode.items():
        evaluated, discarded = evaluated_race_records(records)
        evaluated_by_episode[episode] = evaluated
        discarded_startup_records += discarded
    if args.postprocess_existing:
        for episode, (_, local_replay, _) in enumerate(replay_targets):
            records = evaluated_by_episode[episode]
            if not records:
                raise ProtocolError(f"completed action log is missing episode {episode}")
            final = records[-1]
            start_race_time = int(records[0]["race_time_ms"]) - 100
            episode_records.append(
                {
                    "episode": episode,
                    "steps": len(records),
                    "elapsed_ms": max(
                        0,
                        int(final["race_time_ms"]) - start_race_time,
                    ),
                    "total_reward": sum(float(record["reward"]) for record in records),
                    "finished": bool(final["race_finished"]),
                    "timeout": bool(final["timeout"]),
                    "off_track": bool(final["off_track"]),
                    "fallen": bool(final["fallen"]),
                    "stuck": bool(final["stuck"]),
                    "final_progress": float(final["progress"]),
                    "final_vertical_offset": float(final["vertical_offset"]),
                    "input_replay": str(local_replay.relative_to(WORKSPACE_ROOT)),
                    "input_replay_bytes": local_replay.stat().st_size,
                    "input_replay_sha256": sha256(local_replay),
                }
            )
    expected_actions = sum(int(episode["steps"]) for episode in episode_records)
    if len(action_records) != expected_actions + discarded_startup_records:
        raise ProtocolError(
            f"evaluation action log has {len(action_records)} records for "
            f"{expected_actions} evaluated steps and {discarded_startup_records} "
            "startup records"
        )
    for episode in episode_records:
        episode["trajectory"] = trajectory_metrics(
            evaluated_by_episode[int(episode["episode"])]
        )

    evaluated_action_records = [
        record
        for episode in range(args.episodes)
        for record in evaluated_by_episode[episode]
    ]
    raw_actions = np.asarray(
        [record["raw_action"] for record in evaluated_action_records],
        dtype=np.float64,
    )
    finishes = sum(bool(episode["finished"]) for episode in episode_records)
    timeouts = sum(bool(episode["timeout"]) for episode in episode_records)
    off_tracks = sum(bool(episode["off_track"]) for episode in episode_records)
    falls = sum(bool(episode["fallen"]) for episode in episode_records)
    stuck_truncations = sum(
        bool(episode["stuck"]) for episode in episode_records
    )
    late_falls = sum(
        bool(episode["fallen"])
        and float(episode["trajectory"]["final_progress"])
        >= FINAL_JUMP_PROGRESS
        for episode in episode_records
    )
    precision_summary = aggregate_precision_metrics(
        [episode["trajectory"]["precision"] for episode in episode_records]
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
        "checkpoint_sha256": checkpoint_hash,
        "run_tag": args.run_tag,
        "postprocessed_existing_completed_run": args.postprocess_existing,
        "fall_detector": {
            "vertical_drop_units": 10.0,
            "confirmation": (
                "frozen stuck window must also report no progress and no motion"
            ),
            "stuck_window_ms": 2_000,
            "stuck_progress_gain_units": 1.0,
            "stuck_world_distance_units": 2.0,
        },
        "finishes": finishes,
        "finish_rate": finishes / args.episodes,
        "timeouts": timeouts,
        "off_tracks": off_tracks,
        "falls": falls,
        "stuck_truncations": stuck_truncations,
        "late_final_jump_falls": late_falls,
        "precision_metric_protocol": evaluation_metric_protocol(),
        "precision_summary": precision_summary,
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
        "average_finish_time_ms": (
            statistics.fmean(finish_times_ms) if finish_times_ms else None
        ),
        "worst_finish_time_ms": max(finish_times_ms) if finish_times_ms else None,
        "human_pb_ms": HUMAN_PB_MS,
        "best_finish_gap_to_human_pb_ms": (
            min(finish_times_ms) - HUMAN_PB_MS if finish_times_ms else None
        ),
        "action_records": len(evaluated_action_records),
        "raw_action_records": len(action_records),
        "discarded_startup_action_records": discarded_startup_records,
        "action_minimum": raw_actions.min(axis=0).tolist(),
        "action_maximum": raw_actions.max(axis=0).tolist(),
        "action_mean": raw_actions.mean(axis=0).tolist(),
        "behavior_description": describe_behavior(episode_records),
        "visible_observation": (
            "Pending review of the live 6x run and preserved TMInterface replays."
        ),
        "input_replay_count": len(episode_records),
        "input_replay_sha256": [
            episode["input_replay_sha256"] for episode in episode_records
        ],
        "episodes_detail": episode_records,
        "action_log_sha256": sha256(args.action_log),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"{EXPERIMENT_LABEL} evaluation complete: "
        f"finishes={finishes}/{args.episodes}, "
        f"falls={falls}, stuck={stuck_truncations}",
        flush=True,
    )
    print(f"summary SHA-256={sha256(args.summary)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
