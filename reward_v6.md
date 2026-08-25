# Reward v6 proposal: Clustered steering-reversal frequency

Status: A01-only training complete; deterministic evaluation pending; hypothesis,
formula, and success gates remain frozen

Date pre-registered: 2026-08-25

Owner approval recorded: 2026-08-25. Option A (A01-only V6) was selected. The
pre-registered hypothesis, formula, and five success gates below are frozen
exactly as drafted and must not be adjusted after observing training or
evaluation results. Multi-track training remains a separate future phase.

## Motivation

V5's magnitude-based steering-rate penalty did not solve the fixed oscillation
gate. It reduced mean steering magnitude, full left/right hysteresis crossings,
and lateral deviation, but significant steering-slope reversals increased from
`35.0` to `37.1` per episode. The post-evaluation diagnostic found that increase
was localized: V5 added `3.1` reversals per episode from `10-30%` A01 progress
around the early turn exit, drop/landing, and following straight, plus one per
episode in the `90-100%` final-alignment region. It did not add reversals evenly
around the lap.

V6 therefore targets clustered reversal *frequency*, not steering magnitude or
the size of each steering change. It is a sibling experiment to V5, not a
stronger version of the same jerk penalty.

## Pre-registered hypothesis

A frequency-based penalty on steering-slope reversals inside a fixed short
rolling window will reduce the localized corrective wobble at A01's
drop/landing and final-alignment sections without V5's magnitude tradeoff of
smaller but more frequent corrections. Because isolated, sustained cornering
and isolated large corrections are not penalized, V6 is expected to preserve
V4-level cornering ability, finish reliability, and lap time while reducing
clustered reversals and the fixed oscillation prevalence.

This hypothesis is recorded before V6 implementation or training. It must not
be revised after observing a V6 run.

## Proposed isolated reward addition

V6 uses V4's complete reward unchanged and replaces V5's magnitude-based jerk
term with one frequency-based term. At each 100 ms control step:

```text
steering_delta_t = steer_t - steer_t_minus_1

if steering_delta_t >= 0.05:
    delta_direction_t = +1
else if steering_delta_t <= -0.05:
    delta_direction_t = -1
else:
    delta_direction_t = 0

reversal_t = 1 only when delta_direction_t is nonzero and differs from the
             previous nonzero delta direction; otherwise 0

window_reversals_t = number of reversal events with timestamps in
                     [time_t - 2.0 seconds, time_t], including reversal_t

frequency_cost_t = 0.05 * reversal_t * max(0, window_reversals_t - 3)

signed_progress = clip(current_progress - previous_progress, -20.0, 20.0) / 10.0
reward = signed_progress - 0.10 - frequency_cost_t

if race_finished:
    reward += 50.0
else if verified_failure:
    reward -= 250.0
```

The first three slope reversals inside a rolling 2.0-second window are free. A
new fourth reversal costs `0.05`, a fifth costs `0.10`, and a sixth costs
`0.15`. No cost remains active on later frames merely because prior reversals
are still inside the window; the cost fires only when a new reversal occurs.

The `0.05` steering-delta deadband and 2.0-second window match Decision 0008's
existing diagnostic protocol. A sub-deadband delta does not replace the last
nonzero delta direction. Episode reset clears the prior steering action, last
nonzero delta direction, and rolling event history. The first action receives
no frequency cost.

The cost is not multiplied by `abs(steering_delta_t)` or steering magnitude. A
steady large steering action through a corner is free. One isolated large
correction is also free. V6 adds no direct lateral, heading-error, checkpoint,
airtime, speed, inversion, or steering-magnitude term. Throttle, brake, failure
detection, and every V4 terminal term remain unchanged. V5's
`0.05 * abs(steer_t - steer_t_minus_1)` term is not stacked with this one.

## Coefficient calibration

The proposed `0.05` coefficient was calibrated counterfactually against the
preserved actions before V6 training. With the first three events in each
2.0-second window free:

- V4's 100-episode scale archive would receive a mean frequency cost of
  `1.7675` per episode, with a `1.65-2.20` range;
- V4's original comparable 20 episodes would receive `1.7200` per episode;
- V5's 20 episodes would receive `1.8700` per episode, with its one 39-reversal
  episode receiving `2.25`;
- V5's preregistered magnitude penalty was calibrated at about `1.6074` per V4
  episode.

The proposed V6 term is therefore on approximately the same episode-reward
scale as V5 while targeting clustered frequency directly. On V4's typical
roughly `245.7` finish reward, `1.7675` is about `0.72%`. The free allowance
avoids treating every normal steering correction as a failure, while the
increasing event cost distinguishes a burst of four, five, or six reversals.

## Isolation and initialization

The clean causal comparison is to initialize from the exact V4 checkpoint, as
V5 did, rather than continue from V5's already-adapted policy:

- checkpoint: `checkpoints/reward_v4/final_model.zip`
- SHA-256: `6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2`
- model timestep: `2,002,944`

Using the same parent checkpoint makes V5 and V6 sibling experiments: both add
one alternative smoothness mechanism to V4. Continuing from V5 would confound
V6's frequency term with a policy already shaped by the falsified jerk term.

The PPO/gSDE hyperparameters, seed lineage, 100 ms control period, failure and
stuck thresholds, action audit, checkpoint cadence, power gate, and minimum
`1,000,000` additional interactions should remain the same as V5. Exact run
names and the track scheduler cannot be frozen until the owner chooses the
training sequence below.

## Training-sequence gate

No V6 implementation or training starts until the owner chooses one of these
two protocols.

### Option A: A01-only V6

Train and evaluate on A01 exactly as V1-V5. This keeps checkpoint, training
distribution, budget, evaluator, and track geometry fixed, so the reversal-
frequency term is the only experimental change. The result is directly
comparable to V4 and V5 and gives the clearest answer about the hypothesis.

The cost is that it knowingly leaves the separate A02 overfitting result
unaddressed for one more reward iteration. Even a successful V6 would remain an
A01 specialist until a later multi-track experiment proves otherwise.

### Option B: Start A01+A02 multi-track training in V6

Training on both tracks begins addressing the 50% A02 zero-shot result
immediately, and geometric variety could itself discourage A01-specific
correction patterns. It is closer to the final goal of general driving.

However, it changes two independent variables at once: the reward and the
training distribution. If oscillation or A02 reliability improves, the result
cannot show whether the frequency term, multi-track experience, or their
interaction caused it. A01 lap-time regression could likewise be caused by the
new reward or by sharing policy capacity across tracks. This option also
requires a frozen A01/A02 sampling schedule and separate per-track evaluation;
after training, A02 is no longer a zero-shot test.

## Recommendation before training

Use **Option A: A01-only for V6**. It preserves the clean reward-iteration story
and directly tests the diagnosed mechanism. If V6 succeeds, begin multi-track
training as a distinct next phase using the winning reward. The strongest later
generalization design would keep reward and track distribution as separate
axes: the existing V4/A01 and proposed V6/A01 runs provide the single-track
side, while V4-reward/A01+A02 and V6-reward/A01+A02 runs would reveal whether
multi-track exposure and reversal penalization contribute independently or
interact.

This recommendation does not minimize the A02 result. Decision 0011 remains the
frozen evidence that V4 substantially overfit A01. It simply avoids obscuring
that finding by bundling the first generalization intervention with a new reward
mechanism.

## Training result

The formal A01-only V6 run completed on 2026-08-25 in one uninterrupted attempt.
It initialized from the checksum-pinned V4 checkpoint at timestep `2,002,944`
and PPO completed its active rollout at timestep `3,004,416`, producing
`1,001,472` additional interactions. The 1,472-step overshoot is the same
pre-approved rollout-boundary behavior used for V4 and V5; no steps were
discarded or repeated. Wall time was `2,624.889s` (about 43 minutes 45 seconds).

The stochastic training log contained `3,776` completed episodes and `3,673`
finishes. Terminal flags included `10` timeouts, `6` verified falls, `88` stuck
truncations, and no lateral off-track truncations; flags are reported directly
and are not assumed to be mutually exclusive. These are training-distribution
outcomes, not the frozen deterministic evaluation.

All `1,001,472` policy actions were finite, in range, and passed the affine
action-space audit with no hidden clipping. Sleep and hibernation were disabled
for AC and DC under the High performance power plan. The manifest records the
pre-training `reward_v6.md` SHA-256 as
`8DDA7C721946F3974C0FD37C8B6F45ED70EBFF1B4CA047EDC7D70846218EE8B1`,
so the hypothesis and gates can be verified as preceding the run.

The frozen final checkpoint is:

- `checkpoints/reward_v6/final_model.zip`
- SHA-256 `2825F18DA19C8C0438F490ABF7FED484808F734F363BE3EA11EAC4D3A99418A8`

Training evidence:

- summary SHA-256:
  `7FBE96E8123C39DC7C68D93CA7B71821EBC14EBCCBF6FF0FA88E19DC1C4893F9`
- completed manifest SHA-256:
  `F7DD2046321157F70AA761598794A761533CE58EF3167AC00E35824BEC3EE99C`

No conclusion about the V6 hypothesis is drawn from training episodes. The five
pre-registered gates below remain unchanged and are evaluated only against the
frozen final checkpoint's deterministic 20-episode run.

## Frozen evaluation protocol

For an A01-only V6, retain V5's 20-episode deterministic protocol and report the
same finish, race-clock, lateral-deviation, oscillation, orientation, stuck, and
action-audit metrics. Add the rolling reversal counts and the `10-30%` and
`90-100%` hotspot totals as first-class metrics. Preserve clean, overlay-free
best, closest-to-mean, and worst-finish videos.

Proposed success gates, to be frozen with the selected sequence before
training, are:

1. oscillation in at most `10/20` episodes, matching V5's unweakened gate;
2. at least `19/20` finishes;
3. mean successful lap time at most `25.500 s`;
4. mean significant slope reversals at most `28.0` per episode, a 20% reduction
   from V4's `35.0`; and
5. combined `10-30%` plus `90-100%` hotspot reversals at most `6.4` per episode,
   a 20% reduction from V4's `8.0` baseline.

These last two gates test the mechanism V6 actually targets rather than
declaring success from a smaller steering magnitude. If multi-track training is
selected instead, the A01 gates remain useful but A02 needs its own frozen
reliability and precision gates before training begins.
