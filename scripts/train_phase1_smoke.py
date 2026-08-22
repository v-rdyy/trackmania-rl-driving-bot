"""Run the bounded Phase 1 PPO/TensorBoard integration smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.ppo_audit import RawPpoActionAuditCallback
from trackmania_rl.tmi_bridge import ProtocolError

DEFAULT_ENV_ACTION_LOG = (
    WORKSPACE_ROOT / "artifacts" / "logs" / "phase1_ppo_env_actions.jsonl"
)
DEFAULT_POLICY_ACTION_LOG = (
    WORKSPACE_ROOT / "artifacts" / "logs" / "phase1_ppo_policy_actions.jsonl"
)
DEFAULT_MONITOR_PATH = (
    WORKSPACE_ROOT / "artifacts" / "logs" / "phase1_ppo_monitor"
)
DEFAULT_TENSORBOARD_ROOT = WORKSPACE_ROOT / "tensorboard"
DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "phase1_ppo_smoke"
DEFAULT_SUMMARY = WORKSPACE_ROOT / "artifacts" / "smoke" / "phase1_ppo.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train bounded PPO briefly and verify its integration artifacts."
    )
    parser.add_argument("--timesteps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_tensorboard_metrics(event_files: list[Path]) -> dict[str, dict[str, Any]]:
    required_tags = ("rollout/ep_rew_mean", "rollout/ep_len_mean")
    collected: dict[str, list[float]] = {tag: [] for tag in required_tags}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file.parent))
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        for tag in required_tags:
            if tag in available:
                collected[tag].extend(event.value for event in accumulator.Scalars(tag))

    missing = [tag for tag, values in collected.items() if not values]
    if missing:
        raise ProtocolError(f"TensorBoard is missing scalar metrics: {missing}")
    return {
        tag: {
            "points": len(values),
            "first": values[0],
            "last": values[-1],
            "minimum": min(values),
            "maximum": max(values),
        }
        for tag, values in collected.items()
    }


def main() -> int:
    args = parse_args()
    if args.timesteps < 1000:
        raise SystemExit("--timesteps must be at least 1000 for the Phase 1 smoke test")

    DEFAULT_TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)
    DEFAULT_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    event_files_before = set(DEFAULT_TENSORBOARD_ROOT.rglob("events.out.tfevents.*"))

    base_env = TrackmaniaEnv(
        config=EnvironmentConfig(port=args.port),
        action_log_path=DEFAULT_ENV_ACTION_LOG,
    )
    monitored_env = Monitor(base_env, filename=str(DEFAULT_MONITOR_PATH))
    action_audit = RawPpoActionAuditCallback(DEFAULT_POLICY_ACTION_LOG)
    model = PPO(
        "MlpPolicy",
        monitored_env,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs={"squash_output": True, "log_std_init": -1.0},
        tensorboard_log=str(DEFAULT_TENSORBOARD_ROOT),
        seed=args.seed,
        device="cpu",
        verbose=1,
    )
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=action_audit,
            tb_log_name="phase1_smoke",
            progress_bar=False,
        )
        model.save(DEFAULT_CHECKPOINT)
    finally:
        action_audit.close()
        monitored_env.close()
        model.logger.close()

    checkpoint_path = DEFAULT_CHECKPOINT.with_suffix(".zip")
    event_files_after = set(DEFAULT_TENSORBOARD_ROOT.rglob("events.out.tfevents.*"))
    event_files = sorted(event_files_after - event_files_before)
    if not event_files:
        raise ProtocolError("PPO smoke test did not create a TensorBoard event file")

    env_actions = load_jsonl(DEFAULT_ENV_ACTION_LOG)
    policy_actions = load_jsonl(DEFAULT_POLICY_ACTION_LOG)
    actual_timesteps = int(model.num_timesteps)
    if len(env_actions) != actual_timesteps:
        raise ProtocolError(
            f"environment audit has {len(env_actions)} records for "
            f"{actual_timesteps} timesteps"
        )
    if len(policy_actions) != actual_timesteps:
        raise ProtocolError(
            f"policy audit has {len(policy_actions)} records for "
            f"{actual_timesteps} timesteps"
        )
    if not all(
        record.get("finite")
        and record.get("within_range")
        and record.get("valid")
        for record in env_actions
    ):
        raise ProtocolError("environment audit contains an invalid raw action")
    if not all(
        record.get("finite")
        and record.get("normalized_in_range")
        and record.get("environment_in_range")
        and record.get("affine_unscale_match")
        and record.get("hidden_clipping") is False
        and record.get("valid")
        for record in policy_actions
    ):
        raise ProtocolError("policy audit contains an invalid or clipped raw action")

    episode_end_records = [
        record
        for record in env_actions
        if record.get("terminated") or record.get("truncated")
    ]
    if not episode_end_records:
        raise ProtocolError("PPO smoke test did not complete an episode")

    tensorboard_metrics = read_tensorboard_metrics(event_files)
    reward_metric = tensorboard_metrics["rollout/ep_rew_mean"]
    reward_changed = bool(
        abs(float(reward_metric["maximum"]) - float(reward_metric["minimum"]))
        > 1e-6
    )
    if not reward_changed:
        raise ProtocolError("TensorBoard episode reward did not change during training")

    summary = {
        "requested_timesteps": args.timesteps,
        "actual_timesteps": actual_timesteps,
        "seed": args.seed,
        "algorithm": "PPO",
        "policy": "MlpPolicy with squashed gSDE",
        "n_steps": 256,
        "batch_size": 64,
        "n_epochs": 4,
        "episode_boundaries": len(episode_end_records),
        "environment_action_records": len(env_actions),
        "policy_action_records": len(policy_actions),
        "all_actions_finite_and_in_range": True,
        "hidden_action_clipping": False,
        "tensorboard_metrics": tensorboard_metrics,
        "tensorboard_reward_changed": reward_changed,
        "checkpoint_sha256": sha256(checkpoint_path),
        "environment_action_log_sha256": sha256(DEFAULT_ENV_ACTION_LOG),
        "policy_action_log_sha256": sha256(DEFAULT_POLICY_ACTION_LOG),
        "tensorboard_event_files": [
            {
                "path": str(path.relative_to(WORKSPACE_ROOT)),
                "sha256": sha256(path),
            }
            for path in event_files
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(
        f"PPO smoke verified: timesteps={actual_timesteps}, "
        f"episodes={len(episode_end_records)}, policy_actions={len(policy_actions)}"
    )
    print(
        "TensorBoard reward points="
        f"{reward_metric['points']}, range={reward_metric['minimum']:.6f}.."
        f"{reward_metric['maximum']:.6f}"
    )
    print(f"checkpoint SHA-256={summary['checkpoint_sha256']}")
    print(f"summary SHA-256={sha256(args.summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
