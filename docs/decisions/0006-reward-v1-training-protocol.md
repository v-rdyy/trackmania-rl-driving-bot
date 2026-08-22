# Decision 0006: Reward-v1 training protocol

Status: Completed with checkpointed host-sleep resume

Date: 2026-08-22

## Context

Reward v1 tests the pre-registered sparse finish-only hypothesis in
`reward_v1.md`. The run must be long enough to expose real behavior, must not be
redefined after observing results, and should avoid changing unrelated policy or
environment choices at the same time as the reward.

The Phase 1 smoke run at 6x sustained approximately 59 environment steps per
wall-clock second. The original 500,000-step plan therefore allowed roughly 2.35
hours, subject to training and machine variability. At the 450-step timeout,
that budget could contain about 1,111 complete episodes before shorter invalid-
state truncations.

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
- environment: 100x simulation, 100 ms control period, 45-second timeout, and
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

## Pre-run amendment: wall-clock simulation speed

The first valid 6x start was interrupted at TensorBoard step 10,240 when the
owner proposed increasing TMInterface speed instead of reducing the fixed
500,000-step budget. The interruption was motivated by wall-clock cost, not by a
change to the reward hypothesis or a post-hoc stopping rule. Its 221 completed
episodes contained 214 falls, seven timeouts, zero finishes, and zero reward.
The artifacts are preserved under `reward_v1_interrupted_speed_benchmark` and
excluded from the fresh formal run.

The existing accelerated-time probe verified finite telemetry and live input at
20x, 50x, and 100x. A new end-to-end benchmark then ran the exact PPO settings
for 2,048 steps at each speed. Throughput increased from `56.35` environment
steps/second at 6x to `166.81` at 20x, `231.76` at 50x, and `240.81` at 100x.
An 8,192-step 100x soak test improved to `313.92` steps/second after amortizing
startup overhead and completed 118 episodes with 112 falls, six timeouts, no
bridge/reset errors, and 8,192 finite, bounded, affine-consistent actions.

Use 100x for the formal restart. This changes wall-clock scheduling only: the
environment still advances by one 100 ms game-time control interval per agent
step, retains the same episode timeout and boundaries, and trains for the same
step budget. The sustained soak rate projects roughly 26.5 minutes for 500,000
steps before checkpoint and reporting overhead. Ignored evidence hashes:

- four-speed PPO benchmark: `21BB39779A974D6E719EEB4FC72F3DD421D9BCEA5B130A6B57DF40CB86AF2F19`;
- 100x PPO soak: `24B34136C4AC7317E99DE8CE6C95A526F3D06CB50194D98EF01F916ED437154C`;
- interrupted 6x manifest: `DCC08FE7908ED27414E4AD2F2C2C28A267F67CB991BDBD3EEFFDC402C0102A2F`;
- interrupted 6x Monitor CSV: `33829ED2D1C209B437DCEE811B898F0747D63E378ADE4CC77A4F9BC8E96DC4E4`;
- interrupted 6x TensorBoard event: `E31DF59DCBF8FCAE5D08D923BF5AEBB9CC33854518C8B638140C6A37CAFE389E`.

## Mid-run amendment: checkpointed bridge recovery

The first formal 100x attempt reached model timestep `450,560` before the PC
went to sleep. Suspending the host aborted the local TMInterface socket during a
synchronous response with Windows error `10053`. The latest periodic checkpoint
is at exactly `450,000`, limiting replay to 560 environment interactions. This is
an external host interruption, not a reward, policy, bridge, or 100x stability
failure; resume from that checkpoint without changing any experiment parameter.

Preserve the attempt-one failure manifest and TensorBoard event. The final
training summary must combine numbered formal TensorBoard segments, retain the
original start and cumulative wall time, and report replayed interactions rather
than silently counting the run as uninterrupted. Because the in-memory aggregate
action callback was lost with the process, label its final statistics as covering
the resumed segment; environment and PPO validation remained fail-fast during
the first segment. Ignored evidence hashes:

- attempt-one failure manifest: `178DD024296A81E88096DAB210C12CC4407CA229C4B436FB4E60647DBE0C3FD7`;
- 450,000-step checkpoint: `0C44268EFCA5445AEF796EBDC911CAC10D4A997037DA193C965381EB5D353528`;
- attempt-one TensorBoard event: `1F826C743FC7C77C2F8FED12C1208B6E6B16138F434A65F79740E212839EEE63`.

## Result

The resumed model completed at 501,200 timesteps. `reward_v1.md` records the
quantitative results, deterministic behavior, artifact hashes, and the partially
supported hypothesis verdict. No protocol or reward parameter changed across
the checkpoint boundary.

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
