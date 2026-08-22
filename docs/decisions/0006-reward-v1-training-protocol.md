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
- vertical fall boundary: 10 units below the local reference-path height;
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

## Pre-run amendment: vertical fall detection

The first preliminary run was stopped at TensorBoard step 28,672 after visual
review showed repeated backward falls from the starting platform onto the grass
below. Because the environment only measured horizontal X/Z offset, 62 of 64
episodes waited for the 45-second timeout instead of recognizing the fall. That
run is preserved but invalidated in `reward_v1.md`.

Before restarting the formal budget from zero, add a truncation when the car is
more than 10 vertical units below the height of its nearest reference-path point.
This is a crash/invalid-state boundary, not reward shaping: sparse reward remains
zero for the truncation. Verify the threshold by deliberately reversing off the
A01 start and record the measured vertical offset before the formal run begins.

The first verification attempt did not reproduce the visible failure. Holding
raw action `[0, 0, 1]` applied gas `-65536`, but the car progressed approximately
115 path units over 100 steps instead of backing off the platform. Its minimum
vertical offset was only `-0.203`, so it timed out and correctly failed the probe.
The 10-unit threshold remains provisional; reproduce the stochastic PPO behavior
or add an explicit behind-start boundary before restarting formal training.

The stochastic-policy reproduction then completed 8,192 steps with the actual
seed-42 PPO/gSDE sampling path. It produced 87 episode boundaries: 78 crossed the
10-unit vertical threshold, nine timed out, and none finished. The minimum
vertical offset was `-12.061`. This reproduces the owner's visible grass-drop
observation and validates that a 10-unit local vertical boundary catches the
failure without classifying normal on-track motion from the fixed-action probe.
Its final summary initially failed because detailed action records lacked the
specific terminal-reason fields; preserve that schema failure and rerun the probe
after adding those fields before formal training.

The corrected probe reran the same 8,192-step seed-42 stochastic PPO path and
completed successfully. Of 118 episode boundaries, 110 were vertical falls and
eight were timeouts; there were zero horizontal off-track truncations and zero
finishes. Minimum vertical offset was `-12.196`, while every action remained
finite and in range. This clean reproduction verifies the 10-unit boundary for
the observed A01 grass drop. Formal reward-v1 training may now restart from zero.
