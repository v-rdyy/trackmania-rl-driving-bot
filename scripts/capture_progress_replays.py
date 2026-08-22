"""Preserve checkpoint progression as compact TMInterface input replays."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

from capture_progress_videos import (
    REWARDS,
    load_model,
    load_plan,
    next_take_directory,
    selected_stages,
    write_json,
)
from trackmania_rl.env import EnvironmentConfig, LiveTmiSession, TrackmaniaEnv
from trackmania_rl.tmi_bridge import ProtocolError
from trackmania_rl.video_capture import restart_trackmania_race, sha256

DEFAULT_PLAN = WORKSPACE_ROOT / "config" / "video_progression.json"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "artifacts" / "replays" / "progression"
DEFAULT_TMI_SCRIPTS = Path.home() / "Documents" / "TMInterface" / "Scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture compact input replays for staged checkpoint evidence."
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tmi-scripts-dir", type=Path, default=DEFAULT_TMI_SCRIPTS)
    parser.add_argument("--stage", action="append", dest="stages")
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def wait_for_replay(path: Path, timeout_seconds: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout_seconds
    while time.perf_counter() < deadline:
        if path.is_file() and path.stat().st_size > 0:
            return
        time.sleep(0.05)
    raise ProtocolError(f"TMInterface did not create input replay: {path}")


def outcome_name(terminated: bool, info: dict[str, Any]) -> str:
    if terminated:
        return "finish"
    if info["fallen"]:
        return "fall"
    if info["off_track"]:
        return "off_track"
    return "timeout"


def capture_stage(
    stage: dict[str, Any],
    *,
    output_root: Path,
    tmi_scripts_dir: Path,
    port: int,
    simulation_speed: float,
    seed: int,
) -> dict[str, Any]:
    stage_id = str(stage["id"])
    take_dir = next_take_directory(output_root, stage_id)
    restart_trackmania_race()
    action_log = take_dir / "actions.jsonl"
    reward_function = REWARDS[str(stage["reward"])]
    env = TrackmaniaEnv(
        config=EnvironmentConfig(port=port, simulation_speed=simulation_speed),
        reward_function=reward_function,
        action_log_path=action_log,
    )
    if not isinstance(env.session, LiveTmiSession):
        raise RuntimeError("replay capture requires the live TMInterface session")
    model = None
    checkpoint_hash: str | None = None
    episodes: list[dict[str, Any]] = []
    failure: str | None = None
    try:
        model, checkpoint_hash = load_model(stage, env, seed)
        for episode in range(int(stage["episodes"])):
            observation, reset_info = env.reset()
            terminated = False
            truncated = False
            steps = 0
            total_reward = 0.0
            final_info = reset_info
            maximum_progress = float(reset_info["progress"])
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, final_info = env.step(action)
                if not np.isfinite(observation).all() or not np.isfinite(reward):
                    raise ProtocolError(
                        f"replay stage {stage_id} episode {episode} produced "
                        "nonfinite data"
                    )
                steps += 1
                total_reward += float(reward)
                maximum_progress = max(
                    maximum_progress,
                    float(final_info["progress"]),
                )
                if steps > 451:
                    raise ProtocolError(
                        f"replay stage {stage_id} episode {episode} exceeded "
                        "its safety limit"
                    )

            replay_filename = (
                f"{stage_id}_{take_dir.name}_ep_{episode + 1:02d}.txt"
            )
            external_replay = tmi_scripts_dir / replay_filename
            if external_replay.exists():
                raise FileExistsError(
                    f"refusing to replace existing TMInterface replay: {external_replay}"
                )
            env.session.recover_inputs(replay_filename)
            wait_for_replay(external_replay)
            local_replay = take_dir / replay_filename
            shutil.copy2(external_replay, local_replay)
            outcome = outcome_name(terminated, final_info)
            episodes.append(
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
                    "input_replay": str(local_replay.relative_to(WORKSPACE_ROOT)),
                    "input_replay_bytes": local_replay.stat().st_size,
                    "input_replay_sha256": sha256(local_replay),
                    "tm_interface_source": str(external_replay),
                }
            )
            print(
                f"saved replay stage={stage_id} "
                f"episode={episode + 1}/{stage['episodes']} "
                f"outcome={outcome} progress={maximum_progress:.1f}",
                flush=True,
            )
    except Exception as error:
        failure = repr(error)
        raise
    finally:
        env.close()
        if failure is not None:
            write_json(
                take_dir / "failure_manifest.json",
                {
                    "status": "failed",
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "stage": stage,
                    "error": failure,
                    "checkpoint_sha256": checkpoint_hash,
                    "completed_episodes": episodes,
                },
            )

    manifest = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "seed": seed,
        "simulation_speed": simulation_speed,
        "deterministic": True,
        "checkpoint_sha256": checkpoint_hash,
        "untrained_reference_reconstructed": stage.get("checkpoint") is None,
        "episodes": episodes,
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
    if not args.tmi_scripts_dir.is_dir():
        raise SystemExit(
            f"TMInterface Scripts directory does not exist: {args.tmi_scripts_dir}"
        )

    captures = [
        capture_stage(
            stage,
            output_root=args.output_dir,
            tmi_scripts_dir=args.tmi_scripts_dir,
            port=args.port,
            simulation_speed=float(plan["simulation_speed"]),
            seed=int(plan["seed"]),
        )
        for stage in stages
    ]
    catalog = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan": str(args.plan.relative_to(WORKSPACE_ROOT)),
        "captures": captures,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "latest_capture_catalog.json", catalog)
    print(f"replay capture complete: stages={len(captures)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
