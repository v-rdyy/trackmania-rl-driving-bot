# Decision 0005: Bounded PPO smoke-test actions

Status: Implemented for Phase 1 smoke testing

Date: 2026-08-22

## Context

The project owner requires every raw Phase 1 action to be logged, checked for
NaN/infinity, and checked against the continuous action bounds. Stable-Baselines3
2.9.0's default PPO continuous policy samples an unbounded diagonal Gaussian.
Its rollout collector clips those samples to the Gymnasium `Box` immediately
before calling `env.step()`. Auditing only inside the environment would therefore
hide how often the policy originally requested an out-of-range action.

## Decision

For the Phase 1 integration smoke test, enable PPO's built-in generalized State
Dependent Exploration (gSDE) with `squash_output=True`. PPO then produces a
tanh-bounded normalized action in `[-1, 1]` and Stable-Baselines3 affinely
unscales it to the asymmetric environment bounds:

- steer: `[-1, 1]` to `[-1, 1]`;
- throttle: `[-1, 1]` to `[0, 1]`;
- brake: `[-1, 1]` to `[0, 1]`.

The smoke runner records both representations at every step and asserts that the
environment action exactly matches this affine transformation. The environment
separately records and validates the action it receives before analog transport.
Any nonfinite value, out-of-range value, or unexplained transformation fails the
run instead of silently clipping.

## Tradeoff and scope

This uses an SB3-supported bounded policy path and avoids maintaining a custom
squashed Gaussian distribution. It also changes the exploration process from
PPO's default independent Gaussian noise to gSDE, which may affect learning
quality. Phase 1 only proves the integration loop, so reliability and transparent
action semantics take priority here. Before long Phase 2 experiments, compare
gSDE against a custom bounded independent distribution or explicitly justify
retaining gSDE; do not silently revert to default clipping.
