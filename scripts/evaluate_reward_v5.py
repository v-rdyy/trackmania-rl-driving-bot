"""Evaluate V5 through the fixed V4-comparable 20-episode protocol."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import evaluate_reward_v3 as evaluator
from trackmania_rl.rewards import steering_rate_smoothness_reward

MAX_PROMISING_OSCILLATION_EPISODES = 10
MIN_PROMISING_FINISHES = 19
MAX_PROMISING_MEAN_LAP_MS = 25_500.0


def configure_evaluator() -> None:
    run_dir = WORKSPACE_ROOT / "runs" / "reward_v5"
    evaluator.EXPERIMENT_LABEL = "reward-v5"
    evaluator.EXPERIMENT_SLUG = "reward_v5"
    evaluator.PROTOCOL_LABEL = "reward_v5.md"
    evaluator.REWARD_FUNCTION = steering_rate_smoothness_reward
    evaluator.EXPECTED_EPISODES = 20
    evaluator.DEFAULT_RUN_TAG = None
    evaluator.RUN_DIR = run_dir
    evaluator.DEFAULT_CHECKPOINT = (
        WORKSPACE_ROOT / "checkpoints" / "reward_v5" / "final_model.zip"
    )
    evaluator.DEFAULT_ACTION_LOG = run_dir / "evaluation_actions.jsonl"
    evaluator.DEFAULT_SUMMARY = run_dir / "evaluation_summary.json"
    evaluator.DEFAULT_REPLAY_DIR = (
        WORKSPACE_ROOT / "artifacts" / "replays" / "reward_v5_evaluation"
    )


def scale_gate(summary: dict[str, Any]) -> dict[str, Any]:
    oscillation_episodes = int(
        summary["precision_summary"]["oscillation_detected_episodes"]
    )
    finishes = int(summary["finishes"])
    mean_lap_ms = summary["average_finish_time_ms"]
    checks = {
        "oscillation_at_most_10_of_20": (
            oscillation_episodes <= MAX_PROMISING_OSCILLATION_EPISODES
        ),
        "finishes_at_least_19_of_20": finishes >= MIN_PROMISING_FINISHES,
        "mean_lap_at_most_25_500_ms": (
            mean_lap_ms is not None
            and float(mean_lap_ms) <= MAX_PROMISING_MEAN_LAP_MS
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "oscillation_detected_episodes": oscillation_episodes,
            "finishes": finishes,
            "average_finish_time_ms": mean_lap_ms,
        },
    }


def main() -> int:
    configure_evaluator()
    result = evaluator.main()
    summary = json.loads(evaluator.DEFAULT_SUMMARY.read_text(encoding="utf-8"))
    gate = scale_gate(summary)
    print(
        "V5 100-episode scale gate: "
        f"passed={gate['passed']} checks={gate['checks']}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
