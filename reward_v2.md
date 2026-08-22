# Reward v2: Dense speed

Status: Hypothesis pre-registered; reward implementation verified; training not started

Date pre-registered: 2026-08-22

Approved by the project owner on 2026-08-22.

## Pre-registered hypothesis

A dense speed reward will provide learning signal (unlike v1's flat zero), but because it rewards raw speed rather than progress toward finishing, the agent is expected to find a way to exploit it, e.g. looping or oscillating in place to farm speed, rather than learning to drive the track toward completion.

This statement is recorded exactly as supplied by the project owner. It must not
be revised after observing training or evaluation results.

## Reward contract

- Every step: `displayed_speed / 1000`.
- No finish bonus.
- No explicit crash, fall, off-track, or timeout penalty.
- Termination and truncation only stop future reward collection.

The implementation must be an isolated reward function passed into
`TrackmaniaEnv`, not a conditional hardcoded into the environment.

## Implementation verification

`dense_speed_reward` is implemented in `src/trackmania_rl/rewards.py` as an
isolated callable over `RewardTransition`. Unit coverage verifies that normal,
finished, and truncated transitions all return exactly `display_speed / 1000`
with no terminal adjustment.

The live wiring check completed 20 timeout-truncated episodes and 100 steps.
Every returned reward matched the same step's logged displayed speed divided by
1000. Rewards ranged from `0.005` to `0.036`; terminal timeout steps retained
their positive speed reward. All actions were finite/in range and reset
observations were identical. Ignored local evidence hashes:

- action log: `A3FF2542A2C9570824AEDA87F7F501F87D4475C1380EFD575441B9324F6D5FB6`;
- summary: `51DE7425B47298247C2A7E395458C10670E0D2BF1138E755740F6E27E7E5CE57`.

## Host power preflight

Before implementation or training, the active Windows plan was verified as
High performance (`8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`). Automatic sleep and
hibernate timers are both `0` (Never) on AC and battery. No battery or lid-action
setting was exposed on this host. Hybrid sleep is enabled, but cannot trigger
automatically while automatic sleep is disabled; an explicit manual sleep action
could still suspend the host and must be avoided during training.

The display-only timeout remains 900 seconds on AC and 600 seconds on battery.
Turning off the display does not suspend the host or the training process, so it
does not require a system-setting change.

## Training protocol

Pending. The step budget and fixed PPO/evaluation settings will be documented
before training begins.

## Actual outcome

Pending. No reward-v2 training has started.
