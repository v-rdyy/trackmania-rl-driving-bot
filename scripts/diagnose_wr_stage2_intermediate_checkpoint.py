"""Evaluate the frozen Stage 2 nominal Gate 750k checkpoint without training."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import evaluate_reward_v3 as evaluator
from evaluate_wr_stage2 import validate_direct_live_action_log
from train_reward_v3 import sha256, write_json
from trackmania_rl.rewards import localized_drift_assistance_reward
from trackmania_rl.tmi_bridge import ProtocolError

CHECKPOINT = (
    WORKSPACE_ROOT
    / "checkpoints"
    / "wr_chase_stage2"
    / "ppo_wr_stage2_3756176_steps.zip"
)
CHECKPOINT_SHA256 = (
    "0F0326F84B442D5F506B2FDD6A646D4B1168F9887BF74152B8CD695E3B4635B7"
)
MODEL_TIMESTEPS = 3_756_176
STAGE2_INTERACTIONS = 751_760
PROTOCOL_DOC = WORKSPACE_ROOT / "wr-chase-plan.md"
RUN_DIR = WORKSPACE_ROOT / "runs" / "wr_chase_stage2_intermediate"
ACTION_LOG = RUN_DIR / "gate_00750000_deterministic_actions.jsonl"
SUMMARY = RUN_DIR / "gate_00750000_deterministic_summary.json"
REPLAY_DIR = (
    WORKSPACE_ROOT
    / "artifacts"
    / "replays"
    / "wr_chase_stage2_intermediate"
    / "gate_00750000_deterministic"
)
EPISODES = 10


def verify_source_evidence() -> dict[str, Any]:
    if not CHECKPOINT.is_file():
        raise ProtocolError(f"intermediate checkpoint is missing: {CHECKPOINT}")
    actual_hash = sha256(CHECKPOINT)
    if actual_hash != CHECKPOINT_SHA256:
        raise ProtocolError(
            f"intermediate checkpoint changed: expected {CHECKPOINT_SHA256}, "
            f"got {actual_hash}"
        )
    if not PROTOCOL_DOC.is_file():
        raise ProtocolError(f"intermediate protocol is missing: {PROTOCOL_DOC}")
    return {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "model_timesteps": MODEL_TIMESTEPS,
        "stage2_interactions": STAGE2_INTERACTIONS,
        "protocol_doc_sha256_before_diagnostic": sha256(PROTOCOL_DOC),
    }


def configure_evaluator() -> None:
    evaluator.EXPERIMENT_LABEL = "WR-chase Stage 2 nominal Gate 750k diagnostic"
    evaluator.EXPERIMENT_SLUG = "wr_chase_stage2_intermediate"
    evaluator.PROTOCOL_LABEL = "wr-chase-plan.md intermediate-checkpoint diagnostic"
    evaluator.REWARD_FUNCTION = localized_drift_assistance_reward
    evaluator.RUN_DIR = RUN_DIR
    evaluator.DEFAULT_CHECKPOINT = CHECKPOINT
    evaluator.DEFAULT_ACTION_LOG = ACTION_LOG
    evaluator.DEFAULT_SUMMARY = SUMMARY
    evaluator.DEFAULT_REPLAY_DIR = REPLAY_DIR
    evaluator.EXPECTED_EPISODES = EPISODES
    evaluator.DEFAULT_RUN_TAG = "gate_00750000_deterministic"
    evaluator.EXPECTED_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.POLICY_DETERMINISTIC = True
    evaluator.POLICY_RANDOM_SEED = None
    evaluator.POLICY_SDE_SAMPLE_FREQ = None


def bind_diagnostic_summary(
    summary: dict[str, Any],
    *,
    source_evidence: dict[str, Any],
    direct_live_audit: dict[str, Any],
) -> dict[str, Any]:
    if summary.get("deterministic") is not True:
        raise ProtocolError("intermediate-checkpoint diagnostic was not deterministic")
    sampling = summary.get("policy_sampling", {})
    if sampling.get("mode") != "deterministic":
        raise ProtocolError("intermediate-checkpoint policy mode changed")
    return {
        **summary,
        "intermediate_checkpoint_diagnostic": {
            "status": "complete",
            "purpose": (
                "locate when the Stage 2 deterministic final-line regression "
                "appeared without selecting or training a checkpoint"
            ),
            "training_interactions": 0,
            "source_evidence": source_evidence,
            "direct_live_simstate_audit": direct_live_audit,
        },
    }


def main() -> int:
    source_evidence = verify_source_evidence()
    configure_evaluator()
    result = evaluator.main()
    if result != 0:
        return result
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    direct_live_audit = validate_direct_live_action_log(ACTION_LOG, summary)
    bound = bind_diagnostic_summary(
        summary,
        source_evidence=source_evidence,
        direct_live_audit=direct_live_audit,
    )
    write_json(SUMMARY, bound)
    print(
        f"intermediate-checkpoint summary SHA-256={sha256(SUMMARY)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
