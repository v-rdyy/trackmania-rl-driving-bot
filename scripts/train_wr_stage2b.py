"""Train one frozen 50k A01 WR-chase Stage 2b gate."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import train_wr_stage2 as trainer  # noqa: E402
from trackmania_rl.ppo_audit import AuditedKlPPO  # noqa: E402
from trackmania_rl.rewards import (  # noqa: E402
    graduated_final_corner_assistance_reward,
)

RUN_DIR = WORKSPACE_ROOT / "runs" / "wr_chase_stage2b"
GATE_SUMMARY_DIR = RUN_DIR / "gates"
CHECKPOINT_DIR = WORKSPACE_ROOT / "checkpoints" / "wr_chase_stage2b"
MANIFEST_PATH = RUN_DIR / "run_manifest.json"
MONITOR_PREFIX = RUN_DIR / "training"
INITIAL_CHECKPOINT = (
    WORKSPACE_ROOT
    / "checkpoints"
    / "wr_chase_stage2"
    / "gate_00500000_model.zip"
)
EXPECTED_INITIAL_CHECKPOINT_SHA256 = (
    "8A06E00055886E8E671E988D5D6948C6688B74780EDB32A87C6CC213870C8326"
)
INITIAL_MODEL_TIMESTEPS = 3_506_176
GATE_SIZE = 50_000
CHECKPOINT_INTERVAL = 50_000
TIMEBOX = 500_000
TENSORBOARD_RUN_NAME = "wr_chase_stage2b_graduated_final_corner"
REWARD_FUNCTION = graduated_final_corner_assistance_reward
REWARD_FUNCTION_NAME = "graduated_final_corner_assistance_reward"
RUN_LABEL = "WR Stage 2b"
CHECKPOINT_NAME_PREFIX = "ppo_wr_stage2b"
PPO_CLASS = AuditedKlPPO
TARGET_KL = 0.01
INITIAL_SOURCE = "Stage 2 Gate 500,000"
REWARD_CONTRACT = (
    "V4 + 0.50 * clip(new high-water progress, 0, 20) / 20 * "
    "mean(clip(abs(slip_degrees) / 1, 0, 1), "
    "clip(sliding_wheels / 2, 0, 1)) only at progress 1100..1410 "
    "when speed >=400, >=3 wheels are grounded, high-water progress is "
    "positive, upright >=0.8, abs heading <=pi/4, abs lateral <=20, "
    "speed loss <25, and the transition is nonterminal"
)


def configure_trainer() -> None:
    for name in (
        "RUN_DIR",
        "GATE_SUMMARY_DIR",
        "CHECKPOINT_DIR",
        "MANIFEST_PATH",
        "MONITOR_PREFIX",
        "INITIAL_CHECKPOINT",
        "EXPECTED_INITIAL_CHECKPOINT_SHA256",
        "INITIAL_MODEL_TIMESTEPS",
        "GATE_SIZE",
        "CHECKPOINT_INTERVAL",
        "TIMEBOX",
        "TENSORBOARD_RUN_NAME",
        "REWARD_FUNCTION",
        "REWARD_FUNCTION_NAME",
        "RUN_LABEL",
        "CHECKPOINT_NAME_PREFIX",
        "PPO_CLASS",
        "TARGET_KL",
        "INITIAL_SOURCE",
        "REWARD_CONTRACT",
    ):
        setattr(trainer, name, globals()[name])


def main() -> int:
    configure_trainer()
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
