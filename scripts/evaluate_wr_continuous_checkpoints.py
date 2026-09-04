"""Evaluate every usable checkpoint from the continuous pure-discovery run."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_wr_stage1_gate as discovery
import evaluate_reward_v3 as evaluator
from train_reward_v3 import sha256, write_json
from trackmania_rl.rewards import signed_progress_efficiency_reward

SOURCE_RUN = "wr_pure_continuous_20260904_062051"
RUN_DIR = ROOT / "runs" / SOURCE_RUN / "retrospective_6x"
CHECKPOINT_DIR = ROOT / "checkpoints" / SOURCE_RUN
BASE = ROOT / "checkpoints/wr_chase_stage1/gate_01000000_model.zip"
BASE_SHA256 = "BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044"
EXPECTED_EPISODES = 10
EVALUATION_SPEED = 6.0


def checkpoint_inventory() -> list[dict[str, Any]]:
    inventory = [{"label": "base", "additional_interactions": 0,
                  "path": BASE, "sha256": BASE_SHA256}]
    for sidecar in sorted(CHECKPOINT_DIR.glob("step_*.json")):
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT / path
        additional = int(path.stem.removeprefix("step_"))
        inventory.append({"label": f"step_{additional:012d}",
                          "additional_interactions": additional,
                          "path": path, "sha256": record["sha256"]})
    for item in inventory:
        if not item["path"].is_file() or sha256(item["path"]) != item["sha256"]:
            raise RuntimeError(f"checkpoint provenance failed: {item['label']}")
    return inventory


def training_windows(targets: list[int]) -> list[dict[str, Any]]:
    path = ROOT / "runs" / SOURCE_RUN / "training.monitor.csv"
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(line for line in source if not line.startswith("#")))
    result, lower, cumulative = [], 0, 0
    for target in targets:
        selected = []
        while rows and cumulative + int(rows[0]["l"]) <= target:
            row = rows.pop(0)
            cumulative += int(row["l"])
            if cumulative > lower:
                selected.append(row)
        finishes = [int(float(row["race_time_ms"])) for row in selected
                    if row["race_finished"] == "True"]
        result.append({
            "through_interactions": target, "episodes": len(selected),
            "finishes": len(finishes),
            "finish_rate": len(finishes) / len(selected) if selected else None,
            "best_finish_time_ms": min(finishes) if finishes else None,
            "mean_finish_time_ms": statistics.fmean(finishes) if finishes else None,
        })
        lower = target
    return result


def configure(item: dict[str, Any]) -> dict[str, Path]:
    label = item["label"]
    output = RUN_DIR / label
    paths = {"action_log": output / "actions.jsonl",
             "summary": output / "evaluation.json",
             "replay_dir": ROOT / "artifacts/replays" / SOURCE_RUN / "retrospective_6x" / label}
    evaluator.EXPERIMENT_LABEL = f"continuous pure-discovery {label}"
    evaluator.EXPERIMENT_SLUG = "wr_pure_continuous"
    evaluator.PROTOCOL_LABEL = "continuous-pure-discovery-launch.md"
    evaluator.REWARD_FUNCTION = signed_progress_efficiency_reward
    evaluator.RUN_DIR = RUN_DIR
    evaluator.DEFAULT_CHECKPOINT = item["path"]
    evaluator.DEFAULT_ACTION_LOG = paths["action_log"]
    evaluator.DEFAULT_SUMMARY = paths["summary"]
    evaluator.DEFAULT_REPLAY_DIR = paths["replay_dir"]
    evaluator.DEFAULT_RUN_TAG = label
    evaluator.EXPECTED_EPISODES = EXPECTED_EPISODES
    evaluator.EXPECTED_CHECKPOINT_SHA256 = item["sha256"]
    evaluator.POLICY_DETERMINISTIC = True
    evaluator.POLICY_RANDOM_SEED = None
    evaluator.POLICY_SDE_SAMPLE_FREQ = None
    evaluator.SIMULATION_SPEED = EVALUATION_SPEED
    evaluator.MAP_TO_LOAD = None
    evaluator.AUTO_RESPAWN_ON_CONNECT = True
    evaluator.WAIT_FOR_RACE_START_ON_CONNECT = False
    return paths


def slide_summary(action_log: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in action_log.read_text(encoding="utf-8").splitlines()]
    episodes = sorted({int(record["episode"]) for record in records})
    counts = {zone: 0 for zone in discovery.KNOWN_ZONES}
    track = 0
    for episode in episodes:
        metrics = discovery.drift_metrics([r for r in records if int(r["episode"]) == episode])
        for zone in counts:
            counts[zone] += bool(metrics["known_zones"][zone]["confirmed_windows"])
        track += bool(metrics["track_wide_confirmed_windows"])
    return {"contract": {"minimum_speed": discovery.MIN_DISPLAY_SPEED,
                         "minimum_abs_slip_angle_degrees": discovery.MIN_ABS_SLIP_ANGLE_DEGREES,
                         "minimum_abs_yaw_rate": discovery.MIN_ABS_YAW_RATE,
                         "minimum_ground_contacts": discovery.MIN_GROUND_CONTACTS,
                         "minimum_sliding_wheels": discovery.CONFIRMED_MIN_SLIDING_WHEELS,
                         "consecutive_samples": discovery.CONFIRMED_MIN_SAMPLES},
            "confirmed_episode_counts_by_zone": counts,
            "track_wide_confirmed_episode_count": track}


def evaluate_item(item: dict[str, Any], *, postprocess: bool = False) -> dict[str, Any]:
    paths = configure(item)
    replay_count = len(list(paths["replay_dir"].glob("*.txt")))
    completed = paths["summary"].is_file() and paths["action_log"].is_file()
    if completed != (replay_count == EXPECTED_EPISODES):
        raise RuntimeError(f"partial evaluation needs manual review: {item['label']}")
    if not completed:
        sys.argv = [sys.argv[0], "--checkpoint", str(item["path"]),
                    "--episodes", str(EXPECTED_EPISODES), "--action-log", str(paths["action_log"]),
                    "--summary", str(paths["summary"]), "--replay-dir", str(paths["replay_dir"]),
                    "--run-tag", item["label"], "--reuse-game"]
        if postprocess:
            sys.argv.append("--postprocess-existing")
        if evaluator.main() != 0:
            raise RuntimeError(f"evaluation failed: {item['label']}")
    else:
        print(f"reusing completed evaluation: {item['label']}", flush=True)
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    return {"label": item["label"],
            "additional_interactions": item["additional_interactions"],
            "checkpoint": str(item["path"].relative_to(ROOT)), "sha256": item["sha256"],
            "finishes": summary["finishes"], "finish_rate": summary["finish_rate"],
            "best_finish_time_ms": summary["best_finish_time_ms"],
            "average_finish_time_ms": summary["average_finish_time_ms"],
            "worst_finish_time_ms": summary["worst_finish_time_ms"],
            "precision": summary["precision_summary"],
            "episodes_detail": summary["episodes_detail"],
            "slide_detection": slide_summary(paths["action_log"]),
            "evaluation_summary": str(paths["summary"].relative_to(ROOT)),
            "evaluation_summary_sha256": sha256(paths["summary"]),
            "action_log": str(paths["action_log"].relative_to(ROOT)),
            "action_log_sha256": sha256(paths["action_log"])}


def main() -> int:
    inventory = checkpoint_inventory()
    if (RUN_DIR / "retrospective.json").exists():
        raise FileExistsError(f"refusing to overwrite completed retrospective: {RUN_DIR}")
    results = [evaluate_item(item) for item in inventory]
    aggregate = {"source_run": SOURCE_RUN,
                 "evaluation_protocol": f"10 deterministic episodes each at {EVALUATION_SPEED:g}x",
                 "checkpoints": results,
                 "stochastic_training_windows": training_windows([x["additional_interactions"] for x in inventory[1:]])}
    write_json(RUN_DIR / "retrospective.json", aggregate)
    print(json.dumps(aggregate, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
