"""Train one gated A01 WR-chase Stage 2 localized-assistance segment."""

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
from trackmania_rl.rewards import localized_drift_assistance_reward
from trackmania_rl.tmi_bridge import ProtocolError

RUN_DIR = WORKSPACE_ROOT / "runs" / "wr_chase_stage2"
GATE_SUMMARY_DIR = RUN_DIR / "gates"
TENSORBOARD_ROOT = WORKSPACE_ROOT / "tensorboard"
CHECKPOINT_DIR = WORKSPACE_ROOT / "checkpoints" / "wr_chase_stage2"
MANIFEST_PATH = RUN_DIR / "run_manifest.json"
MONITOR_PREFIX = RUN_DIR / "training"
PROTOCOL_DOC = WORKSPACE_ROOT / "wr-chase-plan.md"
INITIAL_CHECKPOINT = (
    WORKSPACE_ROOT
    / "checkpoints"
    / "wr_chase_stage1"
    / "gate_01000000_model.zip"
)
EXPECTED_INITIAL_CHECKPOINT_SHA256 = (
    "BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044"
)
INITIAL_MODEL_TIMESTEPS = 3_004_416
GATE_SIZE = 500_000
CHECKPOINT_INTERVAL = 250_000
TIMEBOX = 2_000_000
SIMULATION_SPEED = 100.0
TENSORBOARD_RUN_NAME = "wr_chase_stage2_localized_drift"
REWARD_CONTRACT = (
    "V4 + 0.50 * clip(new high-water progress, 0, 20) / 20 only at "
    "progress 680..930 or 1100..1410 when live SimState reports >=3 "
    "grounded wheels, speed >=350, >=1 sliding wheel, abs slip >=1 degree, "
    "and abs body-up yaw >=0.25 rad/s"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-additional-steps",
        type=int,
        required=True,
        help="nominal cumulative Stage 2 gate (500000, 1000000, ...)",
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
    if target_additional_steps == 0:
        return INITIAL_CHECKPOINT
    return CHECKPOINT_DIR / f"{gate_slug(target_additional_steps)}_model.zip"


def gate_summary(target_additional_steps: int) -> Path:
    return GATE_SUMMARY_DIR / f"{gate_slug(target_additional_steps)}_training.json"


def validate_training_target(target_additional_steps: int) -> None:
    if target_additional_steps <= 0:
        raise ValueError("Stage 2 training gate must be positive; target 0 is evaluation-only")
    if target_additional_steps % GATE_SIZE:
        raise ValueError(f"Stage 2 gate must be a multiple of {GATE_SIZE}")
    if target_additional_steps > TIMEBOX:
        raise ValueError(
            f"Stage 2 gate exceeds the frozen {TIMEBOX}-interaction time box"
        )


def verify_initial_checkpoint() -> dict[str, Any]:
    if not INITIAL_CHECKPOINT.is_file():
        raise ProtocolError(
            f"Stage 2 initialization checkpoint is missing: {INITIAL_CHECKPOINT}"
        )
    actual_hash = sha256(INITIAL_CHECKPOINT)
    if actual_hash != EXPECTED_INITIAL_CHECKPOINT_SHA256:
        raise ProtocolError(
            "Stage 2 initialization checkpoint hash changed: expected "
            f"{EXPECTED_INITIAL_CHECKPOINT_SHA256}, got {actual_hash}"
        )
    return {
        "path": str(INITIAL_CHECKPOINT.relative_to(WORKSPACE_ROOT)),
        "sha256": actual_hash,
        "size_bytes": INITIAL_CHECKPOINT.stat().st_size,
        "num_timesteps": INITIAL_MODEL_TIMESTEPS,
        "source": "Stage 1 Gate 2",
    }


def new_manifest(initialization: dict[str, Any]) -> dict[str, Any]:
    """Build the frozen manifest shared by baseline evaluation and training."""
    created_at = datetime.now(timezone.utc).isoformat()
    return {
        "status": "awaiting_baseline_evaluation",
        "started_at_utc": created_at,
        "stage2_initialization": initialization,
        "reward_function": "localized_drift_assistance_reward",
        "reward_contract": REWARD_CONTRACT,
        "reward_base": "signed_progress_efficiency_reward (V4 unchanged)",
        "gate_size_nominal": GATE_SIZE,
        "checkpoint_interval_nominal": CHECKPOINT_INTERVAL,
        "timebox": TIMEBOX,
        "simulation_speed": SIMULATION_SPEED,
        "evaluation_simulation_speed": 6.0,
        "evaluation_episodes": 10,
        "step_period_ms": 100,
        "tensorboard_run_name": TENSORBOARD_RUN_NAME,
        "protocol_doc": str(PROTOCOL_DOC.relative_to(WORKSPACE_ROOT)),
        "protocol_doc_sha256_before_training": sha256(PROTOCOL_DOC),
        "telemetry_source_required": "direct_live_evaluation_simstate",
        "replay_telemetry_fallback_allowed": False,
        "gates": [],
    }


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def completed_training_targets(manifest: dict[str, Any]) -> list[int]:
    return [
        int(gate["target_additional_steps"])
        for gate in manifest.get("gates", [])
        if gate.get("status") == "complete"
        and int(gate.get("target_additional_steps", 0)) > 0
    ]


def require_prior_continue(
    manifest: dict[str, Any], prior_target: int
) -> dict[str, Any]:
    matches = [
        gate
        for gate in manifest.get("gates", [])
        if int(gate.get("target_additional_steps", -1)) == prior_target
    ]
    if len(matches) != 1:
        raise ProtocolError(
            f"Stage 2 requires exactly one prior gate {prior_target} record"
        )
    gate = matches[0]
    if gate.get("evaluation_status") != "complete":
        raise ProtocolError(
            f"Stage 2 gate {prior_target} must complete deterministic evaluation"
        )
    if gate.get("telemetry_status") != "complete":
        raise ProtocolError(
            f"Stage 2 gate {prior_target} must complete direct-live analysis"
        )
    if gate.get("stage2_decision") != "continue":
        raise ProtocolError(
            f"Stage 2 gate {prior_target} decision is "
            f"{gate.get('stage2_decision')!r}; training must stop"
        )
    return gate


def select_load_path(
    *,
    target_additional_steps: int,
    resume: Path | None,
    manifest: dict[str, Any],
) -> Path:
    """Enforce ordered gates and the prior direct-live `continue` decision."""
    validate_training_target(target_additional_steps)
    if not manifest:
        raise ProtocolError(
            "Stage 2 baseline target 0 must be evaluated and analyzed before training"
        )
    initialization = manifest.get("stage2_initialization", {})
    if initialization.get("sha256") != EXPECTED_INITIAL_CHECKPOINT_SHA256:
        raise ProtocolError("Stage 2 manifest does not pin the approved Gate 2 base")

    completed = completed_training_targets(manifest)
    expected_completed = list(range(GATE_SIZE, target_additional_steps, GATE_SIZE))
    if completed != expected_completed:
        raise ProtocolError(
            "Stage 2 completed training gates are not the exact ordered prefix "
            f"required for target {target_additional_steps}: {completed}"
        )

    prior_target = target_additional_steps - GATE_SIZE
    prior_gate = require_prior_continue(manifest, prior_target)
    if prior_target == 0:
        default_load = INITIAL_CHECKPOINT
        expected_hash = EXPECTED_INITIAL_CHECKPOINT_SHA256
    else:
        default_load = gate_checkpoint(prior_target)
        expected_hash = str(prior_gate.get("checkpoint_sha256", ""))
    if not default_load.is_file():
        raise ProtocolError(f"Stage 2 prior checkpoint is missing: {default_load}")
    if sha256(default_load) != expected_hash:
        raise ProtocolError(
            f"Stage 2 prior checkpoint hash changed before gate {target_additional_steps}"
        )

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
        raise ProtocolError("Stage 2 training produced no completed episodes")
    return rows


def successful_race_times(rows: list[dict[str, str]]) -> list[int]:
    return [
        int(float(row["race_time_ms"]))
        for row in rows
        if row["race_finished"] == "True"
    ]


def main() -> int:
    args = parse_args()
    validate_training_target(args.target_additional_steps)
    if args.resume is not None:
        args.resume = args.resume.resolve()

    initialization = verify_initial_checkpoint()
    manifest = load_manifest()
    load_path = select_load_path(
        target_additional_steps=args.target_additional_steps,
        resume=args.resume,
        manifest=manifest,
    )
    output_checkpoint = gate_checkpoint(args.target_additional_steps)
    output_summary = gate_summary(args.target_additional_steps)
    if output_checkpoint.exists() or output_summary.exists():
        raise SystemExit(
            f"Stage 2 gate {args.target_additional_steps} already has frozen output"
        )

    power_preflight = verify_sleep_disabled()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    GATE_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_ROOT.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc)
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
        reward_function=localized_drift_assistance_reward,
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
    target_model_timesteps = INITIAL_MODEL_TIMESTEPS + args.target_additional_steps
    previous_nominal_target = INITIAL_MODEL_TIMESTEPS + (
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
                name_prefix="ppo_wr_stage2",
            ),
            action_stats,
            PeriodicProgressCallback(),
        ]
    )
    print(
        f"WR Stage 2 gate started: model={starting_timesteps}, "
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
    final_checkpoint_hash = sha256(output_checkpoint)
    completed_at = datetime.now(timezone.utc)
    last_500 = race_times[-500:]
    previous_500 = race_times[-1000:-500]
    summary = {
        "status": "complete",
        "gate_target_additional_steps": args.target_additional_steps,
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "wall_seconds": wall_seconds,
        "stage2_initialization": initialization,
        "load_checkpoint": str(load_path.relative_to(WORKSPACE_ROOT)),
        "load_checkpoint_sha256": sha256(load_path),
        "starting_model_timesteps": starting_timesteps,
        "nominal_target_model_timesteps": target_model_timesteps,
        "final_model_timesteps": int(model.num_timesteps),
        "actual_stage2_additional_timesteps": int(model.num_timesteps)
        - INITIAL_MODEL_TIMESTEPS,
        "actual_gate_interactions": int(model.num_timesteps) - starting_timesteps,
        "reward_function": "localized_drift_assistance_reward",
        "reward_contract": REWARD_CONTRACT,
        "episodes_cumulative": len(rows),
        "finishes_cumulative": len(race_times),
        "finish_rate_cumulative": len(race_times) / len(rows),
        "stochastic_finish_time_best_ms": min(race_times) if race_times else None,
        "stochastic_finish_time_mean_last_500_ms": (
            sum(last_500) / len(last_500) if last_500 else None
        ),
        "stochastic_finish_time_mean_previous_500_ms": (
            sum(previous_500) / len(previous_500) if previous_500 else None
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
            "actual_additional_steps": summary["actual_stage2_additional_timesteps"],
            "model_timesteps": summary["final_model_timesteps"],
            "checkpoint": summary["checkpoint"],
            "checkpoint_sha256": final_checkpoint_hash,
            "training_summary": str(output_summary.relative_to(WORKSPACE_ROOT)),
            "training_summary_sha256": sha256(output_summary),
            "evaluation_status": "pending",
            "telemetry_status": "pending",
            "stage2_decision": "pending",
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
        f"WR Stage 2 gate complete: nominal={args.target_additional_steps}, "
        f"actual={summary['actual_stage2_additional_timesteps']}, "
        f"checkpoint={output_checkpoint}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
