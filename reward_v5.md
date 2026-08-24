# Reward v5: Steering-rate smoothness

Status: Pre-registered; not implemented or trained

Date pre-registered: 2026-08-24

Approved in scope by the project owner on 2026-08-24. The coefficient and
evaluation gate below are frozen before implementation or training.

## Pre-registered hypothesis

Penalizing steering-rate-of-change (jerk) rather than steering magnitude should reduce oscillation without impairing cornering ability, since V4's lack of any smoothness incentive is the most likely remaining cause after the clamp hypothesis was falsified.

This statement is recorded exactly as supplied by the project owner. It must not
be revised after observing training or evaluation results.

## Isolated reward change

V5 retains every V4 reward term and coefficient, then adds exactly one term:

```text
progress_delta = current_progress - previous_progress
signed_progress = clip(progress_delta, -20.0, 20.0) / 10.0
time_cost = 0.10
steering_rate_cost = 0.05 * abs(steer_t - steer_t_minus_1)

reward = signed_progress - time_cost - steering_rate_cost

if race_finished:
    reward += 50.0
else if verified_failure:
    reward -= 250.0
```

`steer_t` is the normalized steering action in `[-1, 1]` applied on the current
100 ms control frame. `steer_t_minus_1` is the action applied on the immediately
preceding control frame in the same episode. The first action after reset has no
preceding action and receives zero steering-rate cost. Episode reset clears this
history; it does not compare the first action with the prior episode's last
action or with an invented neutral action.

The new term penalizes change in steering, not absolute steering magnitude. A
steady large steering value through a legitimate corner receives no smoothness
cost. Throttle and brake changes are not penalized. V5 adds no lateral,
heading-error, checkpoint, inversion, speed, steering-magnitude, or direct
oscillation-detector term.

## Coefficient calibration

The coefficient is fixed at `0.05`. Across V4's preserved 100-episode action
log, excluding each episode's first action:

- mean absolute steering change per step: `0.129069`;
- median: `0.072284`;
- 95th percentile: `0.463231`;
- maximum observed: `1.164672`;
- mean per-episode steering total variation: `32.148600`.

Applied counterfactually to those actions, `0.05` would subtract an average
`0.006453` per evaluated step and `1.607430` per episode, about `0.65%` of V4's
typical `245.7` finish reward. At the theoretical maximum full reversal of
`2.0`, the new term subtracts `0.10`. This is intentionally small relative to
route progress and terminal outcomes while giving rapid reversals a measurable
cost.

## Frozen training protocol

- Initialize the complete PPO policy and optimizer state from
  `checkpoints/reward_v4/final_model.zip`.
- Required V4 checkpoint SHA-256:
  `6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2`.
- Required starting model timestep: `2,002,944`.
- Train for at least `1,000,000` additional interactions; Stable-Baselines3 may
  complete the active 2,048-step rollout, and the exact total must be reported.
- Seed lineage, PPO/gSDE hyperparameters, 100 ms control period, 100x training
  speed, A01 map, corrected pedal mapping, detector thresholds, and checkpoint
  cadence remain unchanged from V4.
- TensorBoard run name: `reward_v5_steering_rate_smoothness`.
- Save under distinct `runs/reward_v5/`, `checkpoints/reward_v5/`, and
  TensorBoard namespaces.
- Verify sleep and hibernation are disabled immediately before training.

No reward coefficient, environment threshold, initialization rule, training
budget, or checkpoint-selection rule may change during the formal run. The
final checkpoint remains the pre-registered selection; do not choose an earlier
checkpoint after seeing evaluation results.

## Frozen A01 evaluation and scale gate

First run exactly 20 deterministic A01 episodes at 6x using the same corrected
fall detector, action audit, replay retention, lap-time basis, and Decision 0008
precision metrics as V4. Compare directly against V4's original 20-run result
and its 100-run scale validation.

Report at minimum:

- finish rate and every terminal cause;
- best, mean, and worst TMNF race-clock lap time;
- oscillation-detected episodes, peak two-second sign crossings, total steering
  variation, significant direction reversals, and mean absolute steering;
- mean and tail lateral deviation, upside-down time, and stuck duration;
- every action finite/range check and all artifact hashes;
- visible cornering and final-approach behavior from preserved replays.

V5 qualifies as promising enough for the separately preserved 100-episode scale
validation only if all three pre-registered conditions hold in the 20-run test:

1. oscillation is detected in at most `10/20` episodes, at least a 50% reduction
   in prevalence from V4's `20/20`;
2. at least `19/20` episodes finish; and
3. mean successful TMNF race-clock lap time is at most `25.500s`.

If any condition fails, stop at 20 episodes and report the result. Do not weaken
the gate or modify V5 mid-experiment. If all conditions pass, run exactly 100
additional deterministic episodes under a distinct run tag and artifact set,
retaining every replay.

## Interpretation

The primary comparison is whether the single rate-of-change term reduces the
already frozen oscillation metric. Finish rate, lap time, lateral deviation, and
cornering behavior measure the tradeoff. A smoother policy that becomes slower
or less reliable is not an unqualified success; a reliable policy that remains
oscillatory falsifies or weakens the hypothesis at this coefficient and budget.

The second-track V4 evaluation requested alongside V5 is a separate
generalization measurement. It does not alter this reward, training, A01 gate,
or checkpoint selection.
