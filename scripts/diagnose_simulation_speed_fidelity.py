"""Measure closed-loop A01 policy fidelity from playable 1x through 100x."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_reward_v3 as evaluator
from train_reward_v3 import sha256, write_json
from trackmania_rl.rewards import signed_progress_efficiency_reward

RUN_NAME = "simulation_speed_fidelity_20260904"
RUN_DIR = ROOT / "runs" / RUN_NAME
REPLAY_ROOT = ROOT / "artifacts" / "replays" / RUN_NAME
SPEEDS = (1.0, 2.0, 4.0, 6.0, 10.0, 20.0, 50.0, 100.0)
EPISODES = 5
CHECKPOINTS = (
    {
        "label": "reliable_base",
        "path": ROOT / "checkpoints/wr_chase_stage1/gate_01000000_model.zip",
        "sha256": "BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044",
        "reason": "the checksum-pinned reliable Stage 1 Gate 2 base",
    },
    {
        "label": "speed_sensitive_251904",
        "path": ROOT
        / "checkpoints/wr_pure_continuous_20260904_062051/step_000000251904.zip",
        "sha256": "C658E45A679474C5231E7AA90B08D472633E5B580C51EDE40219A9DB4D7629BA",
        "reason": "the 100x 10/10 versus 6x 0/10 canary checkpoint",
    },
)
LAP_TIME_TOLERANCE_MS = 50
MAXIMUM_POSITION_DELTA = 1.0
MAXIMUM_PROGRESS_DELTA = 1.0
MAXIMUM_DISPLAY_SPEED_DELTA = 5.0
MAXIMUM_ACTION_DELTA = 0.30
_map_request_needed = True


def _records_by_episode(path: Path) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        result.setdefault(int(record["episode"]), []).append(record)
    return result


def _at_time(records: list[dict[str, Any]], race_time_ms: int) -> dict[str, Any]:
    """Return the closest active-race sample with deterministic tie-breaking."""
    active = [record for record in records if int(record["race_time_ms"]) >= 0]
    if not active:
        raise ValueError("trajectory contains no active-race samples")
    return min(
        active,
        key=lambda record: (
            abs(int(record["race_time_ms"]) - race_time_ms),
            int(record["race_time_ms"]),
        ),
    )


def trajectory_signature(path: Path) -> dict[str, Any]:
    grouped = _records_by_episode(path)
    anchors = (5_000, 10_000, 15_000, 20_000)
    episodes = []
    for episode in sorted(grouped):
        records = grouped[episode]
        episodes.append(
            {
                "episode": episode,
                "anchors": {
                    str(anchor): {
                        "sample_race_time_ms": int(_at_time(records, anchor)["race_time_ms"]),
                        "position": _at_time(records, anchor)["position"],
                        "progress": float(_at_time(records, anchor)["progress"]),
                        "display_speed": int(_at_time(records, anchor)["display_speed"]),
                        "action": _at_time(records, anchor)["raw_action"],
                    }
                    for anchor in anchors
                },
            }
        )
    return {"anchors_ms": list(anchors), "episodes": episodes}


def median_anchor(signature: dict[str, Any], anchor: int, field: str) -> np.ndarray:
    values = np.asarray(
        [episode["anchors"][str(anchor)][field] for episode in signature["episodes"]],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("trajectory signature contains nonfinite values")
    return np.median(values, axis=0)


def compare_to_one_x(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    distances = []
    progress_deltas = []
    speed_deltas = []
    action_distances = []
    for anchor in reference["anchors_ms"]:
        ref_position = median_anchor(reference, anchor, "position")
        candidate_position = median_anchor(candidate, anchor, "position")
        distances.append(float(np.linalg.norm(candidate_position - ref_position)))
        progress_deltas.append(
            float(
                median_anchor(candidate, anchor, "progress")
                - median_anchor(reference, anchor, "progress")
            )
        )
        speed_deltas.append(
            float(
                median_anchor(candidate, anchor, "display_speed")
                - median_anchor(reference, anchor, "display_speed")
            )
        )
        action_distances.append(
            float(
                np.linalg.norm(
                    median_anchor(candidate, anchor, "action")
                    - median_anchor(reference, anchor, "action")
                )
            )
        )
    return {
        "median_position_distance_by_anchor": distances,
        "maximum_median_position_distance": max(distances),
        "median_progress_delta_by_anchor": progress_deltas,
        "maximum_absolute_median_progress_delta": max(map(abs, progress_deltas)),
        "median_display_speed_delta_by_anchor": speed_deltas,
        "maximum_absolute_median_display_speed_delta": max(map(abs, speed_deltas)),
        "median_action_l2_distance_by_anchor": action_distances,
        "maximum_median_action_l2_distance": max(action_distances),
    }


def _configure(
    checkpoint: dict[str, Any], speed: float, *, request_map: bool
) -> tuple[Path, Path, Path]:
    speed_label = f"{speed:g}x".replace(".", "p")
    label = str(checkpoint["label"])
    output = RUN_DIR / label / speed_label
    action_log = output / "actions.jsonl"
    summary = output / "evaluation.json"
    replay_dir = REPLAY_ROOT / label / speed_label
    evaluator.EXPERIMENT_LABEL = f"simulation-speed fidelity {label} {speed:g}x"
    evaluator.EXPERIMENT_SLUG = "simulation_speed_fidelity"
    evaluator.PROTOCOL_LABEL = "simulation-speed-fidelity protocol"
    evaluator.REWARD_FUNCTION = signed_progress_efficiency_reward
    evaluator.RUN_DIR = RUN_DIR
    evaluator.DEFAULT_CHECKPOINT = checkpoint["path"]
    evaluator.DEFAULT_ACTION_LOG = action_log
    evaluator.DEFAULT_SUMMARY = summary
    evaluator.DEFAULT_REPLAY_DIR = replay_dir
    evaluator.DEFAULT_RUN_TAG = f"{label}_{speed_label}"
    evaluator.EXPECTED_EPISODES = EPISODES
    evaluator.EXPECTED_CHECKPOINT_SHA256 = checkpoint["sha256"]
    evaluator.POLICY_DETERMINISTIC = True
    evaluator.POLICY_RANDOM_SEED = None
    evaluator.POLICY_SDE_SAMPLE_FREQ = None
    evaluator.SIMULATION_SPEED = speed
    evaluator.MAP_TO_LOAD = "A01-Race.Challenge.Gbx" if request_map else None
    evaluator.AUTO_RESPAWN_ON_CONNECT = not request_map
    evaluator.WAIT_FOR_RACE_START_ON_CONNECT = request_map
    return action_log, summary, replay_dir


def evaluate(checkpoint: dict[str, Any], speed: float) -> dict[str, Any]:
    global _map_request_needed
    action_log, summary_path, replay_dir = _configure(
        checkpoint,
        speed,
        request_map=_map_request_needed,
    )
    existing_replays = list(replay_dir.glob("*.txt"))
    complete = action_log.is_file() and summary_path.is_file()
    if complete != (len(existing_replays) == EPISODES):
        raise RuntimeError(
            f"partial fidelity evaluation needs review: {checkpoint['label']} {speed:g}x"
        )
    if not complete:
        sys.argv = [
            sys.argv[0],
            "--checkpoint",
            str(checkpoint["path"]),
            "--episodes",
            str(EPISODES),
            "--action-log",
            str(action_log),
            "--summary",
            str(summary_path),
            "--replay-dir",
            str(replay_dir),
            "--run-tag",
            f"{checkpoint['label']}_{speed:g}x".replace(".", "p"),
            "--reuse-game",
        ]
        if evaluator.main() != 0:
            raise RuntimeError(
                f"fidelity evaluation failed: {checkpoint['label']} {speed:g}x"
            )
        _map_request_needed = False
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not math.isclose(float(summary["simulation_speed"]), speed):
        raise RuntimeError("evaluator reported the wrong simulation speed")
    return {
        "speed": speed,
        "finishes": int(summary["finishes"]),
        "finish_rate": float(summary["finish_rate"]),
        "best_finish_time_ms": summary["best_finish_time_ms"],
        "mean_finish_time_ms": summary["average_finish_time_ms"],
        "worst_finish_time_ms": summary["worst_finish_time_ms"],
        "falls": int(summary["falls"]),
        "stuck": int(summary["stuck_truncations"]),
        "signature": trajectory_signature(action_log),
        "action_log": str(action_log.relative_to(ROOT)),
        "action_log_sha256": sha256(action_log),
        "summary": str(summary_path.relative_to(ROOT)),
        "summary_sha256": sha256(summary_path),
    }


def classify(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reference = next(result for result in results if result["speed"] == 1.0)
    reference_signature = reference["signature"]
    classified = []
    for result in results:
        comparison = compare_to_one_x(reference_signature, result["signature"])
        same_finish_count = result["finishes"] == reference["finishes"]
        same_failure_profile = (
            result["falls"] == reference["falls"]
            and result["stuck"] == reference["stuck"]
        )
        if reference["mean_finish_time_ms"] is None:
            lap_delta = None
            lap_within_tolerance = result["mean_finish_time_ms"] is None
        elif result["mean_finish_time_ms"] is None:
            lap_delta = None
            lap_within_tolerance = False
        else:
            lap_delta = float(result["mean_finish_time_ms"] - reference["mean_finish_time_ms"])
            lap_within_tolerance = abs(lap_delta) <= LAP_TIME_TOLERANCE_MS
        trajectory_within_tolerance = (
            comparison["maximum_median_position_distance"]
            <= MAXIMUM_POSITION_DELTA
            and comparison["maximum_absolute_median_progress_delta"]
            <= MAXIMUM_PROGRESS_DELTA
            and comparison["maximum_absolute_median_display_speed_delta"]
            <= MAXIMUM_DISPLAY_SPEED_DELTA
            and comparison["maximum_median_action_l2_distance"]
            <= MAXIMUM_ACTION_DELTA
        )
        classified.append(
            {
                **result,
                "comparison_to_1x": comparison,
                "mean_finish_time_delta_from_1x_ms": lap_delta,
                "same_finish_count_as_1x": same_finish_count,
                "same_failure_profile_as_1x": same_failure_profile,
                "lap_time_within_50ms_of_1x": lap_within_tolerance,
                "trajectory_within_tolerance": trajectory_within_tolerance,
                "fidelity_gate": (
                    same_finish_count
                    and same_failure_profile
                    and lap_within_tolerance
                    and trajectory_within_tolerance
                ),
            }
        )
    return classified


def main() -> int:
    if (RUN_DIR / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed study: {RUN_DIR}")
    for checkpoint in CHECKPOINTS:
        if not checkpoint["path"].is_file() or sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint provenance failed: {checkpoint['label']}")
    checkpoint_results = []
    for checkpoint in CHECKPOINTS:
        raw = [evaluate(checkpoint, speed) for speed in SPEEDS]
        checkpoint_results.append(
            {
                "label": checkpoint["label"],
                "checkpoint": str(checkpoint["path"].relative_to(ROOT)),
                "checkpoint_sha256": checkpoint["sha256"],
                "selection_reason": checkpoint["reason"],
                "speeds": classify(raw),
            }
        )
    accepted = [
        speed
        for speed in SPEEDS
        if all(
            next(row for row in checkpoint["speeds"] if row["speed"] == speed)[
                "fidelity_gate"
            ]
            for checkpoint in checkpoint_results
        )
    ]
    contiguous_accepted = []
    for speed in SPEEDS:
        if speed not in accepted:
            break
        contiguous_accepted.append(speed)
    aggregate = {
        "run_name": RUN_NAME,
        "protocol": {
            "ground_truth_speed": 1.0,
            "speeds": list(SPEEDS),
            "episodes_per_checkpoint_speed": EPISODES,
            "deterministic": True,
            "lap_time_tolerance_ms": LAP_TIME_TOLERANCE_MS,
            "fidelity_gate": (
                "same finish/fall/stuck counts as 1x, mean finish time within 50ms "
                "when 1x has finishes, and median trajectory at 5/10/15/20s within "
                "1 position unit, 1 progress unit, 5 displayed-speed units, and "
                "0.30 action L2 for both checkpoints"
            ),
            "selection_rule": (
                "choose the highest speed in the contiguous passing range from 1x; "
                "a speed after an earlier failure is not considered verified because "
                "accelerated behavior was observably non-monotonic"
            ),
            "trajectory_measurement": (
                "median closed-loop position, progress, speed, and action at 5/10/15/20s"
            ),
        },
        "checkpoints": checkpoint_results,
        "speeds_passing_fidelity_gate_for_both_checkpoints": accepted,
        "contiguous_verified_speeds": contiguous_accepted,
        "verified_simulation_speed": (
            max(contiguous_accepted) if contiguous_accepted else None
        ),
    }
    write_json(RUN_DIR / "summary.json", aggregate)
    print(json.dumps(aggregate, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
