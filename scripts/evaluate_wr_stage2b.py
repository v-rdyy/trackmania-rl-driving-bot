"""Run one frozen direct-live evaluation for a WR-chase Stage 2b gate."""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))
sys.path.insert(0, str(WORKSPACE_ROOT / "scripts"))

import evaluate_wr_stage2 as evaluator  # noqa: E402
import train_wr_stage2b as training  # noqa: E402

REQUIRED_LIVE_FIELDS = tuple(
    dict.fromkeys(
        (
            *evaluator.REQUIRED_LIVE_FIELDS,
            "previous_display_speed",
            "upright_cosine",
            "heading_error",
            "lateral_offset",
        )
    )
)


def configure_evaluator() -> None:
    training.configure_trainer()
    replacements = {
        "EXPECTED_INITIAL_CHECKPOINT_SHA256": training.EXPECTED_INITIAL_CHECKPOINT_SHA256,
        "INITIAL_CHECKPOINT": training.INITIAL_CHECKPOINT,
        "INITIAL_MODEL_TIMESTEPS": training.INITIAL_MODEL_TIMESTEPS,
        "MANIFEST_PATH": training.MANIFEST_PATH,
        "RUN_DIR": training.RUN_DIR,
        "TIMEBOX": training.TIMEBOX,
        "GATE_SIZE": training.GATE_SIZE,
        "RUN_LABEL": training.RUN_LABEL,
        "EXPERIMENT_SLUG": "wr_chase_stage2b",
        "PROTOCOL_LABEL": "wr-chase-plan.md Stage 2b",
        "REWARD_FUNCTION": training.REWARD_FUNCTION,
        "REWARD_FUNCTION_NAME": training.REWARD_FUNCTION_NAME,
        "REQUIRED_LIVE_FIELDS": REQUIRED_LIVE_FIELDS,
    }
    for name, value in replacements.items():
        setattr(evaluator, name, value)


def main() -> int:
    configure_evaluator()
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
