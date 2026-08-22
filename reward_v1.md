# Reward v1: Sparse finish-only

Status: Hypothesis pre-registered; reward implementation verified; training not started

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

## Implementation verification

`sparse_finish_reward` is implemented in `src/trackmania_rl/rewards.py` as a
callable over an immutable transition record. `TrackmaniaEnv` accepts a
`reward_function` argument; its default remains the separate Phase 1 smoke
reward. Nonfinite reward outputs fail before Gymnasium returns the step.

The reward-v1 wiring check completed 20 live timeout-truncated episodes and 100
steps. Every reward was exactly `0.0`, every action was finite and in range, and
all reset observations were identical. The `+1.0` finish branch is covered by an
isolated unit test. Ignored local evidence hashes:

- action log: `71C2A4A7685A590D23138AD50CAC69B1A70B054C89ACA752A6A52655F904D513`;
- summary: `08FDF010A1A996BBBD2BC95E0207D817FA19B287DB6374443111CFA0BAA86A88`.

`scripts/train_reward_v1.py` implements Decision 0006 with periodic checkpoints,
a distinct `reward_v1_sparse` TensorBoard run, Monitor episode outcomes, exact
artifact hashes, failure manifests, and resumable checkpoints. Every action is
still checked for finite/bounded/affine behavior, but the long run stores compact
aggregate extrema instead of flushing two JSON records per step.

## Evaluation plan

- Use a distinct reward-v1 TensorBoard run name and checkpoint namespace.
- Train for at least 500,000 environment steps under the fixed hyperparameters
  in Decision 0006. Stable-Baselines3 may complete its current 2,048-step rollout,
  so record the exact actual total.
- Evaluate finish rate and episode-length trend quantitatively.
- Inspect the TensorBoard reward curve for sparse/flat behavior or unexpected
  learning signal.
- Watch deterministic evaluation episodes and describe actual behavior,
  including any finish or meaningful behavioral change.

## Actual outcome

Pending. No reward-v1 training has started.
