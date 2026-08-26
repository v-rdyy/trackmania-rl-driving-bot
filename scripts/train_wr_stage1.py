"""Continue the unchanged V4 policy through one WR-chase Stage 1 gate."""

from __future__ import annotations

import argparse
import csv
import json
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
    sha256,
    verify_sleep_disabled,
    write_json,
)
from trackmania_rl.env import EnvironmentConfig, TrackmaniaEnv
from trackmania_rl.game_launch import close_trackmania, ensure_trackmania_running
from trackmania_rl.ppo_audit import BoundedPpoActionStatsCallback
from trackmania_rl.rewards import signed_progress_efficiency_reward
from trackmania_rl.tmi_bridge import ProtocolError

RUN_DIR = WORKSPACE_ROOT / "runs" / "wr_chase_stage1"
GATE_SUMMARY_DIR = RUN_DIR / "gates"
TENSORBOARD_ROOT = WORKSPACE_ROOT / "tensorboard"
CHECKPOINT_DIR = WORKSPACE_ROOT / "checkpoints" / "wr_chase_stage1"
MANIFEST_PATH = RUN_DIR / "run_manifest.json"
MONITOR_PREFIX = RUN_DIR / "training"
PROTOCOL_DOC = WORKSPACE_ROOT / "wr-chase-plan.md"
V4_CHECKPOINT = WORKSPACE_ROOT / "checkpoints" / "reward_v4" / "final_model.zip"
EXPECTED_V4_CHECKPOINT_SHA256 = (
    "6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2"
)
V4_STARTING_TIMESTEPS = 2_002_944
GATE_SIZE = 500_000
CHECKPOINT_INTERVAL = 250_000
INITIAL_TIMEBOX = 5_000_000
SAFETY_CEILING = 10_000_000
SIMULATION_SPEED = 100.0
TENSORBOARD_RUN_NAME = "wr_chase_stage1_v4_pure_discovery"
REWARD_CONTRACT = (
    "clip(progress_delta, -20, 20) / 10 - 0.10 "
    "+ 50 on finish - 250 on verified failure"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-additional-steps",
        type=int,
        required=True,
        help="nominal cumulative Stage 1 gate (500000, 1000000, ...)",
    )
    parser.add_argument("--port", type=int, default=8478)
    parser.add_argument(
        "--resume",
        type=Path,
        help="optimizer-bearing checkpoint for a resumed interrupted gate",
    )
    return parser.parse_args()


def gate_slug(target_additional_steps: int) -> str:
    return f"gate_{target_additional_steps:08d}"


def gate_checkpoint(target_additional_steps: int) -> Path:
    return CHECKPOINT_DIR / f"{gate_slug(target_additional_steps)}_model.zip"


def gate_summary(target_additional_steps: int) -> Path:
    return GATE_SUMMARY_DIR / f"{gate_slug(target_additional_steps)}_training.json"


def validate_target(target_additional_steps: int) -> None:
    if target_additional_steps <= 0:
        raise ValueError("Stage 1 gate must be positive")
    if target_additional_steps % GATE_SIZE:
        raise ValueError(f"Stage 1 gate must be a multiple of {GATE_SIZE}")
    if target_additional_steps > SAFETY_CEILING:
        raise ValueError(
            f"Stage 1 gate exceeds the frozen {SAFETY_CEILING}-step ceiling"
        )


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
        "sha256": actual_hash,
        "size_bytes": V4_CHECKPOINT.stat().st_size,
        "num_timesteps": V4_STARTING_TIMESTEPS,
    }


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def completed_targets(manifest: dict[str, Any]) -> list[int]:
    return [
        int(gate["target_additional_steps"])
        for gate in manifest.get("gates", [])
        if gate.get("status") == "complete"
    ]


def select_load_path(
    *,
    target_additional_steps: int,
    resume: Path | None,
    manifest: dict[str, Any],
) -> Path:
    targets = completed_targets(manifest)
    expected_previous = target_additional_steps - GATE_SIZE
    if targets and targets != sorted(set(targets)):
        raise ProtocolError("Stage 1 manifest has duplicate or unordered gates")
    if targets and targets[-1] > expected_previous:
        raise ProtocolError("requested Stage 1 gate predates completed evidence")
    if expected_previous == 0:
        if targets:
            raise ProtocolError("first Stage 1 gate is already complete")
        default_load = V4_CHECKPOINT
    else:
        if not targets or targets[-1] != expected_previous:
            raise ProtocolError(
                f"Stage 1 gate {target_additional_steps} requires completed gate "
                f"{expected_previous}"
            )
        previous_gate = next(
            gate
            for gate in manifest["gates"]
            if int(gate["target_additional_steps"]) == expected_previous
        )
        if previous_gate.get("telemetry_status") != "complete":
            raise ProtocolError(
                f"Stage 1 gate {expected_previous} must complete telemetry review "
                "before more training"
            )
        if previous_gate.get("stage1_decision") != "continue":
            raise ProtocolError(
                f"Stage 1 gate {expected_previous} decision is "
                f"{previous_gate.get('stage1_decision')!r}; training must stop"
            )
        default_load = gate_checkpoint(expected_previous)
    if resume is None:
        return default_load
    resolved = resume.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def read_monitor_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(RUN_DIR.glob("*.monitor.csv")):
        with path.open("r", encoding="utf-8", newline="") as source:
            data_lines = [line for line in source if not line.startswith("#")]
        rows.extend(csv.DictReader(data_lines))
    if not rows:
        raise ProtocolError("Stage 1 training produced no completed episodes")
    return rows


def successful_race_times(rows: list[dict[str, str]]) -> list[int]:
    return [
        int(float(row["race_time_ms"]))
        for row in rows
        if row["race_finished"] == "True"
    ]


def main() -> int:
    args = parse_args()
    validate_target(args.target_additional_steps)
    if args.resume is not None:
        args.resume = args.resume.resolve()

    v4_initialization = verify_v4_checkpoint()
    manifest = load_manifest()
    load_path = select_load_path(
        target_additional_steps=args.target_additional_steps,
        resume=args.resume,
        manifest=manifest,
    )
    if not load_path.is_file():
        raise SystemExit(f"Stage 1 load checkpoint is missing: {load_path}")
    output_checkpoint = gate_checkpoint(args.target_additional_steps)
    output_summary = gate_summary(args.target_additional_steps)
    if output_checkpoint.exists() or output_summary.exists():
        raise SystemExit(
            f"Stage 1 gate {args.target_additional_steps} already has frozen output"
        )

    power_preflight = verify_sleep_disabled()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    GATE_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
    if not manifest:
        manifest = {
            "status": "running",
            "started_at_utc": started_at.isoformat(),
            "v4_initialization": v4_initialization,
            "reward_function": "signed_progress_efficiency_reward",
            "reward_contract": REWARD_CONTRACT,
            "reward_unchanged_from_v4": True,
            "gate_size_nominal": GATE_SIZE,
            "checkpoint_interval_nominal": CHECKPOINT_INTERVAL,
            "initial_timebox": INITIAL_TIMEBOX,
            "safety_ceiling": SAFETY_CEILING,
            "simulation_speed": SIMULATION_SPEED,
            "step_period_ms": 100,
            "tensorboard_run_name": TENSORBOARD_RUN_NAME,
            "protocol_doc": str(PROTOCOL_DOC.relative_to(WORKSPACE_ROOT)),
            "protocol_doc_sha256_before_training": sha256(PROTOCOL_DOC),
            "gates": [],
        }
    manifest.update(
        {
            "status": "training_gate",
            "active_gate_target": args.target_additional_steps,
            "active_gate_started_at_utc": started_at.isoformat(),
            "active_gate_load_checkpoint": str(load_path.relative_to(WORKSPACE_ROOT)),
            "active_gate_load_checkpoint_sha256": sha256(load_path),
            "active_gate_power_preflight": power_preflight,
        }
    )
    write_json(MANIFEST_PATH, manifest)

    try:
        closed = close_trackmania()
        if closed:
            print("closed stale TrackMania session", flush=True)
        _, launched = ensure_trackmania_running(port=args.port, confirm_existing=True)
        print(f"TrackMania ready (launched={launched})", flush=True)
    except Exception as error:
        manifest.update(
            {
                "status": "failed_before_training",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(error),
            }
        )
        write_json(MANIFEST_PATH, manifest)
        raise

    base_env = TrackmaniaEnv(
        config=EnvironmentConfig(
            port=args.port,
            simulation_speed=SIMULATION_SPEED,
            stuck_window_ms=2_000,
            stuck_progress_gain_units=1.0,
            stuck_world_distance_units=2.0,
            legacy_reversed_pedal_mapping=False,
            map_to_load="A01-Race.Challenge.Gbx",
            auto_respawn_on_connect=False,
            wait_for_race_start_on_connect=True,
        ),
        reward_function=signed_progress_efficiency_reward,
    )
    monitored_env = Monitor(
        base_env,
        filename=str(MONITOR_PREFIX),
        info_keywords=(
            "race_time_ms",
            "timeout",
            "off_track",
            "fallen",
            "stuck",
            "race_finished",
            "progress",
            "display_speed",
        ),
        override_existing=not any(RUN_DIR.glob("*.monitor.csv")),
    )
    model = PPO.load(
        load_path,
        env=monitored_env,
        device="cpu",
        tensorboard_log=str(TENSORBOARD_ROOT),
    )
    starting_timesteps = int(model.num_timesteps)
    target_model_timesteps = V4_STARTING_TIMESTEPS + args.target_additional_steps
    previous_nominal_target = V4_STARTING_TIMESTEPS + (
        args.target_additional_steps - GATE_SIZE
    )
    if starting_timesteps < previous_nominal_target:
        raise ProtocolError(
            f"load checkpoint has {starting_timesteps} timesteps, before the prior gate"
        )
    if starting_timesteps >= target_model_timesteps:
        raise ProtocolError(
            f"load checkpoint already reached target {target_model_timesteps}"
        )
    remaining = target_model_timesteps - starting_timesteps
    action_stats = BoundedPpoActionStatsCallback()
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=CHECKPOINT_INTERVAL,
                save_path=str(CHECKPOINT_DIR),
                name_prefix="ppo_wr_stage1",
            ),
            action_stats,
            PeriodicProgressCallback(),
        ]
    )
    print(
        f"WR Stage 1 gate started: model={starting_timesteps}, "
        f"nominal_target={target_model_timesteps}, remaining={remaining}",
        flush=True,
    )
    wall_started = time.perf_counter()
    try:
        model.learn(
            total_timesteps=remaining,
            callback=callbacks,
            tb_log_name=TENSORBOARD_RUN_NAME,
            reset_num_timesteps=False,
            progress_bar=False,
        )
        model.save(output_checkpoint.with_suffix(""))
    except Exception as error:
        wall_seconds = time.perf_counter() - wall_started
        manifest.update(
            {
                "status": "failed",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error": repr(error),
                "active_gate_failure_timesteps": int(model.num_timesteps),
                "active_gate_wall_seconds": wall_seconds,
            }
        )
        write_json(MANIFEST_PATH, manifest)
        raise
    finally:
        monitored_env.close()
        close_model_logger(model)
    wall_seconds = time.perf_counter() - wall_started

    rows = read_monitor_rows()
    race_times = successful_race_times(rows)
    if not race_times:
        raise ProtocolError("Stage 1 has no successful stochastic training episodes")
    final_checkpoint_hash = sha256(output_checkpoint)
    completed_at = datetime.now(timezone.utc)
    summary = {
        "status": "complete",
        "gate_target_additional_steps": args.target_additional_steps,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "wall_seconds": wall_seconds,
        "load_checkpoint": str(load_path.relative_to(WORKSPACE_ROOT)),
        "load_checkpoint_sha256": sha256(load_path),
        "starting_model_timesteps": starting_timesteps,
        "nominal_target_model_timesteps": target_model_timesteps,
        "final_model_timesteps": int(model.num_timesteps),
        "actual_stage1_additional_timesteps": int(model.num_timesteps)
        - V4_STARTING_TIMESTEPS,
        "actual_gate_interactions": int(model.num_timesteps) - starting_timesteps,
        "reward_function": "signed_progress_efficiency_reward",
        "reward_contract": REWARD_CONTRACT,
        "reward_unchanged_from_v4": True,
        "episodes_cumulative": len(rows),
        "finishes_cumulative": len(race_times),
        "finish_rate_cumulative": len(race_times) / len(rows),
        "stochastic_finish_time_best_ms": min(race_times),
        "stochastic_finish_time_mean_last_500_ms": sum(race_times[-500:])
        / len(race_times[-500:]),
        "stochastic_finish_time_mean_previous_500_ms": (
            sum(race_times[-1000:-500]) / len(race_times[-1000:-500])
            if len(race_times) >= 1000
            else None
        ),
        "action_validation_this_attempt": action_stats.summary(),
        "power_preflight": power_preflight,
        "protocol_doc_sha256_before_training": manifest[
            "protocol_doc_sha256_before_training"
        ],
        "checkpoint": str(output_checkpoint.relative_to(WORKSPACE_ROOT)),
        "checkpoint_sha256": final_checkpoint_hash,
    }
    write_json(output_summary, summary)
    manifest["gates"].append(
        {
            "status": "complete",
            "target_additional_steps": args.target_additional_steps,
            "actual_additional_steps": summary["actual_stage1_additional_timesteps"],
            "model_timesteps": summary["final_model_timesteps"],
            "checkpoint": summary["checkpoint"],
            "checkpoint_sha256": final_checkpoint_hash,
            "training_summary": str(output_summary.relative_to(WORKSPACE_ROOT)),
            "training_summary_sha256": sha256(output_summary),
            "completed_at_utc": completed_at.isoformat(),
        }
    )
    manifest.update(
        {
            "status": "awaiting_gate_evaluation",
            "active_gate_target": None,
            "latest_completed_gate": args.target_additional_steps,
        }
    )
    for key in list(manifest):
        if key.startswith("active_gate_") and key != "active_gate_target":
            del manifest[key]
    write_json(MANIFEST_PATH, manifest)
    print(
        f"WR Stage 1 gate complete: nominal={args.target_additional_steps}, "
        f"actual={summary['actual_stage1_additional_timesteps']}, "
        f"checkpoint={output_checkpoint}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
