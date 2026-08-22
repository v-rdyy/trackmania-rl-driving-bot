# Decision 0006: Reward-v1 training protocol

Status: Fixed before training

Date: 2026-08-22

## Context

Reward v1 tests the pre-registered sparse finish-only hypothesis in
`reward_v1.md`. The run must be long enough to expose real behavior, must not be
redefined after observing results, and should avoid changing unrelated policy or
environment choices at the same time as the reward.

The Phase 1 smoke run sustained approximately 59 environment steps per wall-clock
second. A 500,000-step run therefore plans for roughly 2.35 hours, subject to
training and machine variability. At the 450-step timeout, that budget could
contain about 1,111 complete episodes before shorter off-track truncations.

## Decision

Run reward v1 with this protocol:

- minimum environment-step budget: `500,000`;
- Stable-Baselines3 may finish the current 2,048-step rollout, so actual steps
  may exceed the minimum and must be reported exactly;
- seed: `42`;
- algorithm/policy: PPO `MlpPolicy` with bounded squashed gSDE per Decision 0005;
- rollout steps: `2,048`;
- batch size: `64`;
- optimization epochs per rollout: `10`;
- learning rate: `0.0003`;
- gamma: `0.99`;
- GAE lambda: `0.95`;
- PPO clip range: `0.2`;
- environment: 6x simulation, 100 ms control period, 45-second timeout, and
  50-unit lateral off-track boundary;
- checkpoint interval: every `50,000` callback steps, plus a final checkpoint;
- TensorBoard run name: `reward_v1_sparse`;
- post-training evaluation: at least 20 deterministic episodes.

The runner validates every policy/environment action and accumulates action-range
statistics in memory. It does not write two verbose JSON records per training
step: 500,000 steps would create large redundant logs and add synchronous disk
flush overhead. Per-step raw logging remains preserved for the Phase 1 smoke test
where it was explicitly required. Episode Monitor logs, TensorBoard, checkpoints,
aggregate action validation, and evaluation telemetry remain required for v1.

## Tradeoffs

Five hundred thousand steps is much smaller than the later approximately
six-million-step final training target, but it is over 240 complete PPO rollouts
and is large enough to test whether sparse reward ever appears across many
episodes. If the agent learns despite the hypothesis, training is not stopped or
reinterpreted to manufacture failure.

Squashed gSDE is retained rather than switching to the default clipped Gaussian
or a new custom distribution. That keeps actions bounded and transparent and
holds policy architecture constant across reward comparisons. Its exploration
behavior may affect learning, so the eventual analysis must name it rather than
claiming the reward is the only conceivable cause of the result.
