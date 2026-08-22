"""Exercise 20 consecutive short episodes through the live Gymnasium wrapper."""

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
from trackmania_rl.rewards import (
    clamped_forward_progress_reward,
    dense_speed_reward,
    phase1_smoke_reward,
    sparse_finish_reward,
)
from trackmania_rl.tmi_bridge import ProtocolError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify finite reset/step behavior across consecutive episodes."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--episode-ms", type=int, default=500)
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument(
        "--reward",
        choices=(
            "phase1",
            "sparse-finish",
            "dense-speed",
            "clamped-progress",
        ),
        default="phase1",
    )
    parser.add_argument(
        "--action-log",
        type=Path,
        default=WORKSPACE_ROOT
        / "artifacts"
        / "logs"
        / "phase1_reset_actions.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=WORKSPACE_ROOT
        / "artifacts"
        / "smoke"
        / "phase1_env_resets.json",
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
    if args.episodes < 20:
        raise SystemExit("--episodes must be at least 20 for the Phase 1 check")
    if args.episode_ms <= 0 or args.episode_ms % 10 != 0:
        raise SystemExit("--episode-ms must be a positive 10 ms multiple")
    if args.reward == "clamped-progress" and args.episode_ms <= 2_000:
        raise SystemExit("V3 live smoke requires --episode-ms above 2000")

    config = EnvironmentConfig(
        port=args.port,
        max_episode_ms=args.episode_ms,
        stuck_window_ms=2_000 if args.reward == "clamped-progress" else None,
    )
    reward_functions = {
        "phase1": phase1_smoke_reward,
        "sparse-finish": sparse_finish_reward,
        "dense-speed": dense_speed_reward,
        "clamped-progress": clamped_forward_progress_reward,
    }
    reward_function = reward_functions[args.reward]
    env = TrackmaniaEnv(
        config=config,
        action_log_path=args.action_log,
        reward_function=reward_function,
    )
    action = np.asarray(
        [0.0, 0.0, 0.0]
        if args.reward == "clamped-progress"
        else [0.0, 0.35, 0.0],
        dtype=np.float32,
    )
    reset_observations: list[np.ndarray] = []
    episode_records: list[dict[str, object]] = []
    total_steps = 0
    rewards: list[float] = []

    try:
        for episode in range(args.episodes):
            observation, reset_info = env.reset()
            if not np.isfinite(observation).all():
                raise ProtocolError(f"episode {episode} reset observation is nonfinite")
            if abs(float(reset_info["lateral_offset"])) > config.max_start_lateral_offset:
                raise ProtocolError(f"episode {episode} reset is outside the start corridor")
            if float(reset_info["progress"]) > config.max_start_progress:
                raise ProtocolError(f"episode {episode} reset is too far along A01")
            if int(reset_info["display_speed"]) > config.max_start_speed:
                raise ProtocolError(f"episode {episode} reset speed is too high")
            reset_observations.append(observation.copy())

            terminated = False
            truncated = False
            steps = 0
            reward = 0.0
            info = reset_info
            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(observation).all() or not np.isfinite(reward):
                    raise ProtocolError(
                        f"episode {episode} step {steps} produced nonfinite data"
                    )
                steps += 1
                total_steps += 1
                rewards.append(float(reward))
                if steps > args.episode_ms // config.step_period_ms + 1:
                    raise ProtocolError(f"episode {episode} did not truncate on time")

            if args.reward == "clamped-progress":
                if terminated or not truncated or not bool(info["stuck"]):
                    raise ProtocolError(
                        f"episode {episode} did not reach the expected V3 stuck cutoff"
                    )
            elif terminated or not truncated or not bool(info["timeout"]):
                raise ProtocolError(
                    f"episode {episode} ended without the expected timeout truncation"
                )
            episode_records.append(
                {
                    "episode": episode,
                    "steps": steps,
                    "reset_race_time_ms": int(reset_info["race_time_ms"]),
                    "reset_speed": int(reset_info["display_speed"]),
                    "reset_progress": float(reset_info["progress"]),
                    "reset_lateral_offset": float(reset_info["lateral_offset"]),
                    "final_reward": float(reward),
                    "stuck": bool(info.get("stuck", False)),
                }
            )
    finally:
        env.close()

    audit_records = [
        json.loads(line)
        for line in args.action_log.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(audit_records) != total_steps:
        raise ProtocolError(
            f"action audit has {len(audit_records)} records for {total_steps} steps"
        )
    if not all(
        record.get("finite")
        and record.get("within_range")
        and record.get("valid")
        for record in audit_records
    ):
        raise ProtocolError("action audit contains an invalid raw action")
    if args.reward == "sparse-finish" and any(reward != 0.0 for reward in rewards):
        raise ProtocolError("sparse finish reward fired without a finished race")
    if args.reward == "dense-speed" and any(
        not np.isclose(reward, int(record["display_speed"]) / 1000.0)
        for reward, record in zip(rewards, audit_records, strict=True)
    ):
        raise ProtocolError("dense speed reward diverged from displayed_speed / 1000")
    if args.reward == "clamped-progress" and any(
        reward < 0.0 or reward > 1.0 for reward in rewards
    ):
        raise ProtocolError("V3 progress reward left its registered [0, 1] range")

    repeatable_resets = np.stack(reset_observations)
    max_repeat_reset_delta = float(
        np.max(np.abs(repeatable_resets - repeatable_resets[0]))
    )
    if max_repeat_reset_delta > 1e-5:
        raise ProtocolError(
            "rewound reset observations diverged by "
            f"{max_repeat_reset_delta:.9f}, expected at most 0.00001"
        )

    summary = {
        "episodes": args.episodes,
        "total_steps": total_steps,
        "simulation_speed": config.simulation_speed,
        "step_period_ms": config.step_period_ms,
        "episode_timeout_ms": config.max_episode_ms,
        "stuck_window_ms": config.stuck_window_ms,
        "raw_action": action.tolist(),
        "reward_function": reward_function.__name__,
        "reward_minimum": min(rewards),
        "reward_maximum": max(rewards),
        "all_actions_finite_and_in_range": True,
        "max_repeat_reset_observation_delta": max_repeat_reset_delta,
        "episodes_detail": episode_records,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"verified {args.episodes} consecutive episodes and {total_steps} finite steps"
    )
    print(f"max repeat reset observation delta={max_repeat_reset_delta:.9f}")
    print(f"action log SHA-256={sha256(args.action_log)}")
    print(f"summary SHA-256={sha256(args.summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
