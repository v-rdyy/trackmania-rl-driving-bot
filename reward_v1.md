# Reward v1: Sparse finish-only

Status: Hypothesis pre-registered; implementation and training not started

Date pre-registered: 2026-08-22

Approved by the project owner on 2026-08-22.

## Pre-registered hypothesis

A reward that only fires on finishing the track (sparse) will be too sparse for PPO to find gradient signal in a reasonable number of training steps. Expected outcome: the agent never learns to finish, reward stays near zero/flat, with no meaningful behavior change across training.

This statement is recorded exactly as supplied by the project owner. It must not
be revised after observing training or evaluation results.

## Reward contract

- Normal step: `0.0`.
- Timeout or off-track truncation: `0.0`.
- Finished race: `1.0`.

The implementation will be an isolated reward function passed into the
environment, not a conditional hardcoded into `TrackmaniaEnv`.

## Evaluation plan

- Use a distinct reward-v1 TensorBoard run name and checkpoint namespace.
- Train substantially longer than the Phase 1 integration smoke test; fix the
  exact budget before training begins.
- Evaluate finish rate and episode-length trend quantitatively.
- Inspect the TensorBoard reward curve for sparse/flat behavior or unexpected
  learning signal.
- Watch deterministic evaluation episodes and describe actual behavior,
  including any finish or meaningful behavioral change.

## Actual outcome

Pending. No reward-v1 training has started as of this pre-registration commit.
