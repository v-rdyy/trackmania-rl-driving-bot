# Reward v5: Steering-rate smoothness

Status: Complete; 20-episode gate failed, so no 100-episode scale run

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

## Training result

The uninterrupted formal run completed on 2026-08-24 in one attempt. PPO
finished its active rollout at model timestep `3,004,416`, for `1,001,472`
additional interactions after the pinned V4 timestep. This is the expected
rollout-boundary overshoot allowed by the protocol; no interactions were
discarded or replayed. Wall time was `2,609.184s` (about 43 minutes 29 seconds).

The run logged `3,850` training episodes: `3,733` finishes, `20` timeouts, `15`
verified falls, and `82` stuck truncations. These stochastic training outcomes
are operational evidence, not the pre-registered deterministic evaluation.
All `1,001,472` policy actions were finite, in range, and passed the affine
action-space audit with no hidden clipping. Host sleep and hibernation were
confirmed disabled immediately before the run.

The frozen final checkpoint is
`checkpoints/reward_v5/final_model.zip`, SHA-256
`6324DFC2047DC744474D12FB0D8F0000982434FBB48A221247C427EBA787E5C2`.
The training-summary SHA-256 is
`35ED39475C23013EBA18AE1951568CE7DB0B5412BF9C5028F27F3FA203BDD193`.
No conclusion about the oscillation hypothesis is drawn until the deterministic
evaluation below is complete.

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

## Deterministic evaluation result

The frozen final checkpoint completed all `20/20` deterministic A01 episodes.
There were zero falls, stuck truncations, timeouts, off-track terminations, or
inversions. Best/mean/worst TMNF race-clock lap times were `24.770s`, `24.784s`,
and `24.800s`, respectively. The best was `0.270s` slower than the real `24.5s`
human PB. The evaluator audited `4,946` active-race action records; every action
was finite, in range, and valid. It explicitly discarded `45` first-episode
startup records after detecting the countdown restart, so the apparently long
raw first episode did not contaminate its `24.770s` race-clock result.

The primary gate failed: oscillation was detected in `20/20`, versus the
required maximum of `10/20`. Peak sign crossings remained `5` in a two-second
window, the same as V4. The reliability gate passed (`20/20`, required at least
`19/20`) and the mean-lap gate passed (`24.784s`, required at most `25.500s`).
Because all three conditions were required, the pre-registered 100-episode V5
scale run was not performed.

The single added term still changed several secondary metrics relative to V4's
original comparable 20 episodes:

| Metric | V4 | V5 | Change |
| --- | ---: | ---: | ---: |
| finishes | 20/20 | 20/20 | unchanged |
| best lap | 24.900s | 24.770s | 0.130s faster |
| mean lap | 24.929s | 24.784s | 0.146s faster |
| oscillation detected | 20/20 | 20/20 | unchanged |
| mean steering total variation | 32.028 | 31.574 | 1.4% lower |
| mean absolute steering | 0.567 | 0.528 | 6.8% lower |
| mean hysteresis sign crossings | 24.6 | 19.6 | 20.3% lower |
| mean significant direction reversals | 35.0 | 37.1 | 6.0% higher |
| mean p95 absolute lateral offset | 10.628 | 9.063 | 14.7% lower |
| maximum absolute lateral offset | 17.833 | 11.954 | 33.0% lower |

The normal-speed review of preserved episode 4 showed a clean, upright finish:
the car negotiated the corners and final jump without the checkpoint collision
or flip seen in earlier versions. It still hugged the inside boundary in the
early turn, and the action trace retained enough rapid reversals to trip the
fixed detector. Clean, overlay-free normal-speed captures were rendered from
the exact preserved inputs for the best episode (`24.770s`, episode 4), the
closest-to-mean episode (`24.780s`, episode 7), and the worst episode
(`24.800s`, episode 2). They are stored under
`artifacts/videos/reward_v5_evaluation_clean/`; their SHA-256 values are
`0F31673831CB1E5EDAFCEF3C536D07C6A488A1E971BA742906A2F7BC8E9A8D25`,
`0E4F4D84F3E4C1F67BBA7C51E39C3997D153C91DEFE28885FF1329DA9E0F3644`,
and `1745085E5664BA368D025A8051A927045398FF6108BA203020431285DBF96ADA`,
respectively. Each video manifest records `overlay: false`.
All 20 input replays remain preserved under
`artifacts/replays/reward_v5_evaluation/`.

The evaluation-summary SHA-256 is
`8F4BB8671F42119E0231D707B40C9E72F265AFE84A75A2A8221A8F7DD3113361`;
the action-log SHA-256 is
`2C020376D805CA0E6F5A5604B052CB773EB4D0C0F6DDDE18C809E0BE63F9330A`.

## Hypothesis outcome

The hypothesis is not supported at coefficient `0.05` and this training budget.
The penalty modestly reduced steering effort, hysteresis crossings, and lateral
error without impairing cornering, finish rate, or lap time. It did **not**
reduce fixed oscillation prevalence at all, and significant direction reversals
increased slightly. This is a useful partial effect, but it does not qualify V5
as the requested oscillation fix and does not justify relaxing the frozen gate.
