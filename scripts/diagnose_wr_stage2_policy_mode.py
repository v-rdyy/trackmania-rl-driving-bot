"""Run the frozen Gate 1M stochastic policy-mode safety diagnostic."""

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
    WORKSPACE_ROOT / "checkpoints" / "wr_chase_stage2" / "gate_01000000_model.zip"
)
CHECKPOINT_SHA256 = (
    "F1FCCB19F80192C7A6528BC183C9CD8317A7F7EAA35472E189A56699A85C92FF"
)
DETERMINISTIC_SUMMARY = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2"
    / "gates"
    / "gate_01000000_evaluation.json"
)
DETERMINISTIC_SUMMARY_SHA256 = (
    "F96A762148D448FBDF0D6D4B74EE277791C037527E6FE3DE31C30207CDAF34F1"
)
DETERMINISTIC_ACTIONS = (
    WORKSPACE_ROOT
    / "runs"
    / "wr_chase_stage2"
    / "gates"
    / "gate_01000000_evaluation_actions.jsonl"
)
DETERMINISTIC_ACTIONS_SHA256 = (
    "3CB45B494A1695A120F2BB78B69B648E19A9C4BFAB64473A390DC30AAB77E015"
)
PROTOCOL_DOC = WORKSPACE_ROOT / "wr-chase-plan.md"
RUN_DIR = WORKSPACE_ROOT / "runs" / "wr_chase_stage2_policy_mode"
ACTION_LOG = RUN_DIR / "gate_01000000_stochastic_actions.jsonl"
SUMMARY = RUN_DIR / "gate_01000000_stochastic_summary.json"
REPLAY_DIR = (
    WORKSPACE_ROOT
    / "artifacts"
    / "replays"
    / "wr_chase_stage2_policy_mode"
    / "gate_01000000_stochastic"
)
EPISODES = 10
RANDOM_SEED = 20_260_828
SDE_SAMPLE_FREQ = 4


def verify_source_evidence() -> dict[str, str]:
    expected = {
        CHECKPOINT: CHECKPOINT_SHA256,
        DETERMINISTIC_SUMMARY: DETERMINISTIC_SUMMARY_SHA256,
        DETERMINISTIC_ACTIONS: DETERMINISTIC_ACTIONS_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.is_file():
            raise ProtocolError(f"policy-mode source evidence is missing: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ProtocolError(
                f"policy-mode source evidence changed: {path} "
                f"expected {expected_hash}, got {actual_hash}"
            )
    if not PROTOCOL_DOC.is_file():
        raise ProtocolError(f"policy-mode protocol is missing: {PROTOCOL_DOC}")
    return {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "deterministic_summary_sha256": DETERMINISTIC_SUMMARY_SHA256,
        "deterministic_action_log_sha256": DETERMINISTIC_ACTIONS_SHA256,
        "protocol_doc_sha256_before_diagnostic": sha256(PROTOCOL_DOC),
    }


def configure_evaluator() -> None:
    evaluator.EXPERIMENT_LABEL = "WR-chase Stage 2 Gate 1M policy-mode diagnostic"
    evaluator.EXPERIMENT_SLUG = "wr_chase_stage2_policy_mode"
    evaluator.PROTOCOL_LABEL = "wr-chase-plan.md frozen Gate 1M policy-mode diagnostic"
    evaluator.REWARD_FUNCTION = localized_drift_assistance_reward
    evaluator.RUN_DIR = RUN_DIR
    evaluator.DEFAULT_CHECKPOINT = CHECKPOINT
    evaluator.DEFAULT_ACTION_LOG = ACTION_LOG
    evaluator.DEFAULT_SUMMARY = SUMMARY
    evaluator.DEFAULT_REPLAY_DIR = REPLAY_DIR
    evaluator.EXPECTED_EPISODES = EPISODES
    evaluator.DEFAULT_RUN_TAG = "gate_01000000_stochastic_seed_20260828"
    evaluator.EXPECTED_CHECKPOINT_SHA256 = CHECKPOINT_SHA256
    evaluator.POLICY_DETERMINISTIC = False
    evaluator.POLICY_RANDOM_SEED = RANDOM_SEED
    evaluator.POLICY_SDE_SAMPLE_FREQ = SDE_SAMPLE_FREQ


def bind_diagnostic_summary(
    summary: dict[str, Any],
    *,
    source_evidence: dict[str, str],
    direct_live_audit: dict[str, Any],
) -> dict[str, Any]:
    if summary.get("deterministic") is not False:
        raise ProtocolError("policy-mode diagnostic was not stochastic")
    sampling = summary.get("policy_sampling", {})
    if (
        sampling.get("mode") != "stochastic"
        or int(sampling.get("random_seed", -1)) != RANDOM_SEED
        or sampling.get("use_sde") is not True
        or int(sampling.get("sde_sample_freq", -1)) != SDE_SAMPLE_FREQ
    ):
        raise ProtocolError("policy-mode diagnostic sampling contract changed")
    return {
        **summary,
        "policy_mode_diagnostic": {
            "status": "complete",
            "purpose": (
                "distinguish deterministic mean-action failure from final sampled "
                "policy collapse without changing model weights"
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
    print(f"policy-mode diagnostic summary SHA-256={sha256(SUMMARY)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
