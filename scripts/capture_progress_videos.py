"""Record deterministic checkpoint progression clips from the live A01 race."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.rewards import (
    dense_speed_reward,
    phase1_smoke_reward,
    sparse_finish_reward,
)
from trackmania_rl.tmi_bridge import ProtocolError
from trackmania_rl.video_capture import ProgressVideoRecorder, sha256

DEFAULT_PLAN = WORKSPACE_ROOT / "config" / "video_progression.json"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "artifacts" / "videos" / "progression"
REWARDS = {
    "phase1_smoke": phase1_smoke_reward,
    "reward_v1": sparse_finish_reward,
    "reward_v2": dense_speed_reward,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture website-ready checkpoint progression videos."
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", action="append", dest="stages")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if int(plan.get("protocol_version", 0)) != 1:
        raise ValueError("video plan protocol_version must be 1")
    stages = plan.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("video plan must contain at least one stage")
    seen: set[str] = set()
    for stage in stages:
        stage_id = str(stage.get("id", ""))
        if not stage_id or stage_id in seen:
            raise ValueError(f"video stage id is empty or duplicated: {stage_id!r}")
        seen.add(stage_id)
        if stage.get("reward") not in REWARDS:
            raise ValueError(f"unknown reward for {stage_id}: {stage.get('reward')}")
        if int(stage.get("episodes", 0)) <= 0:
            raise ValueError(f"episodes must be positive for {stage_id}")
        if int(stage.get("model_timesteps", -1)) < 0:
            raise ValueError(f"model_timesteps must be nonnegative for {stage_id}")
    return plan


def selected_stages(
    plan: dict[str, Any],
    requested: list[str] | None,
) -> list[dict[str, Any]]:
    stages = list(plan["stages"])
    if not requested:
        return stages
    by_id = {str(stage["id"]): stage for stage in stages}
    missing = [stage_id for stage_id in requested if stage_id not in by_id]
    if missing:
        raise ValueError(f"video plan does not contain stages: {missing}")
    return [by_id[stage_id] for stage_id in requested]


def next_take_directory(root: Path, stage_id: str) -> Path:
    stage_root = root / stage_id
    stage_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.removeprefix("take_"))
        for path in stage_root.glob("take_[0-9][0-9][0-9]")
        if path.is_dir() and path.name.removeprefix("take_").isdigit()
    ]
    take = max(existing, default=0) + 1
    destination = stage_root / f"take_{take:03d}"
    destination.mkdir()
    return destination


def build_untrained_model(env: TrackmaniaEnv, seed: int) -> PPO:
    return PPO(
        "MlpPolicy",
        env,
        n_steps=256,
        batch_size=64,
        n_epochs=4,
        use_sde=True,
        sde_sample_freq=4,
        policy_kwargs={"squash_output": True, "log_std_init": -1.0},
        seed=seed,
        device="cpu",
        verbose=0,
    )


def load_model(stage: dict[str, Any], env: TrackmaniaEnv, seed: int) -> tuple[PPO, str | None]:
    checkpoint_value = stage.get("checkpoint")
    if checkpoint_value is None:
        if int(stage["model_timesteps"]) != 0:
            raise ValueError("only the zero-step reference may omit a checkpoint")
        return build_untrained_model(env, seed), None
    checkpoint = WORKSPACE_ROOT / str(checkpoint_value)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    return PPO.load(checkpoint, device="cpu"), sha256(checkpoint)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def capture_stage(
    stage: dict[str, Any],
    *,
    output_root: Path,
    port: int,
    simulation_speed: float,
    fps: int,
    max_width: int,
    seed: int,
) -> dict[str, Any]:
    stage_id = str(stage["id"])
    take_dir = next_take_directory(output_root, stage_id)
    action_log = take_dir / "actions.jsonl"
    reward_function = REWARDS[str(stage["reward"])]
    env = TrackmaniaEnv(
        config=EnvironmentConfig(port=port, simulation_speed=simulation_speed),
        reward_function=reward_function,
        action_log_path=action_log,
    )
    clips: list[dict[str, Any]] = []
    model: PPO | None = None
    checkpoint_hash: str | None = None
    failure: str | None = None
    try:
        model, checkpoint_hash = load_model(stage, env, seed)
        for episode in range(int(stage["episodes"])):
            observation, reset_info = env.reset()
            video_path = take_dir / f"episode_{episode + 1:02d}.mp4"
            label = (
                f"{stage['display_name']}  |  "
                f"{int(stage['model_timesteps']):,} model steps  |  "
                f"deterministic {simulation_speed:g}x"
            )
            recorder = ProgressVideoRecorder(
                video_path,
                label=label,
                fps=fps,
                max_width=max_width,
            )
            recorder.update(
                state="RUNNING",
                elapsed_ms=0,
                display_speed=int(reset_info["display_speed"]),
                progress=float(reset_info["progress"]),
            )
            recorder.start()
            recorder_stopped = False
            try:
                terminated = False
                truncated = False
                steps = 0
                total_reward = 0.0
                final_info = reset_info
                maximum_progress = float(reset_info["progress"])
                while not (terminated or truncated):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, final_info = env.step(
                        action
                    )
                    if not np.isfinite(observation).all() or not np.isfinite(reward):
                        raise ProtocolError(
                            f"video stage {stage_id} episode {episode} produced "
                            "nonfinite data"
                        )
                    steps += 1
                    total_reward += float(reward)
                    maximum_progress = max(
                        maximum_progress,
                        float(final_info["progress"]),
                    )
                    recorder.update(
                        state="RUNNING",
                        elapsed_ms=int(final_info["elapsed_ms"]),
                        display_speed=int(final_info["display_speed"]),
                        progress=float(final_info["progress"]),
                    )
                    if steps > 451:
                        raise ProtocolError(
                            f"video stage {stage_id} episode {episode} exceeded "
                            "safety limit"
                        )
                outcome = (
                    "FINISH"
                    if terminated
                    else "FALL"
                    if final_info["fallen"]
                    else "OFF TRACK"
                    if final_info["off_track"]
                    else "TIMEOUT"
                )
                recorder.update(
                    state=outcome,
                    elapsed_ms=int(final_info["elapsed_ms"]),
                    display_speed=int(final_info["display_speed"]),
                    progress=float(final_info["progress"]),
                )
                time.sleep(0.75)
                video = recorder.stop()
                recorder_stopped = True
            finally:
                if not recorder_stopped:
                    try:
                        recorder.update(state="CAPTURE ERROR")
                        recorder.stop()
                    except Exception:
                        pass
            video["path"] = str(video_path.relative_to(WORKSPACE_ROOT))
            clips.append(
                {
                    "episode": episode,
                    "steps": steps,
                    "elapsed_ms": int(final_info["elapsed_ms"]),
                    "total_reward": total_reward,
                    "finished": bool(terminated),
                    "timeout": bool(final_info["timeout"]),
                    "off_track": bool(final_info["off_track"]),
                    "fallen": bool(final_info["fallen"]),
                    "maximum_progress": maximum_progress,
                    "final_progress": float(final_info["progress"]),
                    "video": video,
                }
            )
            print(
                f"captured stage={stage_id} episode={episode + 1}/{stage['episodes']} "
                f"outcome={outcome} progress={maximum_progress:.1f}",
                flush=True,
            )
    except Exception as error:
        failure = repr(error)
        raise
    finally:
        env.close()
        if failure is not None:
            partial_files = [
                {
                    "path": str(path.relative_to(WORKSPACE_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in sorted(take_dir.iterdir())
                if path.is_file()
            ]
            write_json(
                take_dir / "failure_manifest.json",
                {
                    "status": "failed",
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "stage": stage,
                    "error": failure,
                    "checkpoint_sha256": checkpoint_hash,
                    "completed_clips": clips,
                    "partial_files": partial_files,
                },
            )

    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "seed": seed,
        "simulation_speed": simulation_speed,
        "deterministic": True,
        "fps": fps,
        "maximum_output_width": max_width,
        "checkpoint_sha256": checkpoint_hash,
        "untrained_reference_reconstructed": stage.get("checkpoint") is None,
        "clips": clips,
        "action_log": str(action_log.relative_to(WORKSPACE_ROOT)),
        "action_log_sha256": sha256(action_log),
    }
    manifest_path = take_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {
        "stage_id": stage_id,
        "take": take_dir.name,
        "manifest": str(manifest_path.relative_to(WORKSPACE_ROOT)),
        "manifest_sha256": sha256(manifest_path),
    }


def main() -> int:
    args = parse_args()
    plan = load_plan(args.plan)
    stages = selected_stages(plan, args.stages)
    if args.dry_run:
        for stage in stages:
            print(
                f"{stage['id']}: {stage['display_name']} "
                f"steps={stage['model_timesteps']} episodes={stage['episodes']}"
            )
        return 0

    results = []
    for stage in stages:
        results.append(
            capture_stage(
                stage,
                output_root=args.output_dir,
                port=args.port,
                simulation_speed=float(plan["simulation_speed"]),
                fps=int(plan["fps"]),
                max_width=int(plan["maximum_output_width"]),
                seed=int(plan["seed"]),
            )
        )
    catalog = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(args.plan.relative_to(WORKSPACE_ROOT)),
        "captures": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "latest_capture_catalog.json", catalog)
    print(f"video capture complete: stages={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
