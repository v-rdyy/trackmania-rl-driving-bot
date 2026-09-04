# Stage 2b: final-alignment and KL diagnosis

Date: 2026-09-03. Retrospective diagnosis, not a new reward hypothesis.
Training remains paused. No reward, zone, optimizer, or checkpoint was changed.

## Conclusion

The strongest measured mechanism is a changed launch direction for the long
final jump, followed by a worse landing line and an abrupt near-stop at the
final structure. It is not a new high-speed approach or an observed slide.
The line already differs before the bonus zone, changes within it, and diverges
again through the launch setup. This supports downstream trajectory coupling,
but does **not** establish that the graduated bonus caused the harmful change.
There was no matched continuation with the bonus disabled, and no saved
per-minibatch advantages or reward-component gradients to establish causality.

The KL guard performed its implemented early-stop correctly, but it is neither
a rollback nor a cumulative constraint against the original reliable policy.
Cumulative movement is demonstrated; consistently harmful movement at every
individual update is not. Stochastic training remained mostly successful while
the final deterministic policy failed all ten evaluation episodes.

## Evidence and method

Compare the reliable Stage 2 Gate 500k checkpoint, evaluated as Stage 2b target
zero, against Stage 2b Gate 50k (51,200 actual interactions). Both saved
evaluations contain ten deterministic episodes at 6x on the identical
checksum-pinned A01 reference path. Finishes were 9/10 versus 0/10.

All trajectory/dynamics measurements come from the original **direct-live
SimState logs**. No input replay was used to regenerate telemetry, no new game
episodes were run, and no Trackmania training occurred during this diagnosis.
Episodes are separate deterministic repeats, not matched-seed stochastic pairs.

Progress comparisons linearly interpolate the first forward crossing, excluding
clock resets and gaps, so a stalled car's later samples cannot dominate the
approach average. Lateral offsets are relative to the recorded reference path,
not a surveyed road center, structure boundary, or measured clearance margin.

Same-state probes rebuild the exact 26-value observation from each live state
and ask each frozen policy for its deterministic action. A log row contains the
state **after** its action: the reconstruction is validated against the next
row's action. Maximum error is below `8.4e-7` across both datasets. This avoids
an off-by-one comparison and separates changed decisions from different inputs.

## Trajectory and speed: where the change develops

Median first-crossing values across all ten episodes:

| Reference progress | Gate 500k offset | Gate 50k offset | Gate 500k speed | Gate 50k speed |
| --- | ---: | ---: | ---: | ---: |
| 1100: bonus entry | +0.164 | +0.669 | 432.79 | 433.35 |
| 1300: in the corner | -1.441 | +0.181 | 460.03 | 460.42 |
| 1410: bonus exit | -3.289 | -1.845 | 474.20 | 474.47 |
| 1700: launch setup | -6.063 | -5.028 | 460.06 | 460.19 |
| 1800: airborne | -9.263 | -9.199 | 419.15 | 418.82 |
| 2000: airborne approach | -16.263 | -18.227 | 365.98 | 364.16 |
| 2150: after landing, before abrupt near-stop | -19.885 | -23.486 | 363.34 | 358.85 |

The trained policy exits the bonus zone 1.44 reference units to the positive
side of the baseline, with only 0.27 displayed-speed difference. Its bonus-zone
traversal is about 15ms shorter by interpolated crossing estimates; both remain
2.5s under the original 100ms sampled gate metric. That sub-sample estimate is
not a new timing success claim. Both still have zero wheel-slide onset, and
total bonus over ten episodes is essentially unchanged: 10.1652 versus 10.1598.

The policies already differ at progress 300 (+0.30 offset difference), 900
(+0.93), and bonus entry (+0.50), so the bonus zone is not the first observable
trajectory divergence. The reward is localized, but the shared neural policy
has changed outside that zone as well.

### The decisive geometric difference: launch velocity

The longest pre-structure zero-wheel-contact run begins at the 20.2s sample
in every episode for both policies. Across episodes:

- Mean starting lateral offset is -6.33 for the base and -5.52 for Gate 50k.
  Gate 50k starts closer to the reference path, which is not itself proof of a
  better launch.
- Mean lateral velocity relative to the path at that sample changes from
  -4.06 to -5.23 world units/s: approximately 29% more sideways motion.
- Mean horizontal world-velocity heading changes by approximately 0.593 degrees.
  Within each flight, the measured horizontal velocity heading varies by less
  than 0.000003 degrees even while steering changes substantially.
- First wheel contact returns at 24.0s for the base and 23.9s for Gate 50k.
  First-zero-to-first-contact intervals are 3.8s and 3.7s; exact physical
  takeoff/landing lie between the 100ms samples.
- Mean lateral position at that first landing contact worsens from -21.42 to
  -24.40. At fixed progress 2150, the median difference is 3.60 units.

The measured flight direction carries the launch error downstream. Late
airborne steering differences should not be mistaken for evidence that the
policy can redirect its horizontal flight path at that moment: the recorded
velocity direction remains unchanged throughout this section.

At 24.4s, all ten Gate 50k episodes lose speed from 362-369 to 0-37 in a single
100ms step, at progress 2161.88-2166.92. All later terminate stuck. This is strong
telemetry evidence of impact consistent with the known final-structure problem,
but the logs are not a dedicated collision sensor and do not identify the exact
piece of frame struck. The baseline also had one severe near-stop/failure and
two smaller abrupt losses that still finished: this was an existing vulnerable
section, not an entirely new failure location.

### Is the bonus directly conflicting with the final structure?

The final positive bonus sample is at 17.9s in all twenty episodes. The major
jump begins about 2.3s later; the Gate 50k near-stop is 6.5s after the last bonus,
roughly 739-744 straight-line world units away. The two events are not adjacent
or overlapping reward windows in the measured trajectory, although the line
through the corner feeds the later launch.

On identical baseline states, Gate 50k's steering changes by a mean -0.0198
in the bonus zone (maximum absolute change 0.0929) and -0.0495 in the following
1410-1800 span (maximum 0.0993). On Gate 50k's own recorded states, corresponding
means are -0.0235 and -0.0462. These signs use the existing steering convention;
they do not by themselves identify a safer road line. Same-state throttle is
also lower on average, not higher. In the late 2000-2150 span, mean absolute
same-state steering change is only 0.0213 on baseline states despite the much
larger realized landing displacement.

Thus a changed corner/launch line is real, and a launch-angle mechanism is
strongly supported. A direct speed-for-bonus tradeoff is not observed. Calling
the line change an exploit or an intentional optimization of the graduated
bonus would go beyond this evidence: neither bonus yield nor slide precursor
improved, and the policy changed before and after the rewarded zone.

## KL: limited update work, but no cumulative anchor

The custom guard matches the installed Stable-Baselines3 implementation:
it measures a minibatch approximate KL against that rollout's behavior policy;
when it exceeds `1.5 * target_kl = 0.015`, it skips optimizing that minibatch
and stops the remaining epochs. Previous minibatch updates remain in place.
It does not line-search, undo an overshoot, cap every state's KL, or compare
against the original Stage 2 Gate 500k checkpoint on later rollouts.

A controlled offline toy-environment test forces an overshoot on minibatch two.
Both stock PPO and the audited implementation perform exactly one Adam step,
skip the triggering batch, retain the first step, and produce identical final
weights. No production training was used for this test.

The saved Adam counters provide an independent check: 547,840 at the base,
547,995 after 24 rollout updates, and 547,999 after 25. That is 159 actual
optimizer steps versus 8,000 scheduled (25 rollouts x 10 epochs x 32 batches),
with four steps in the last update. All 25 updates stopped in epoch zero.
Triggering minibatch KL ranged from 0.01562 to 0.04337: the threshold detects
overshoot rather than guaranteeing a strict 0.01 upper bound.

Fixed-state analytic KL from base to final, averaged on baseline live states,
is 1.476 nats in the bonus zone, 0.843 post-zone, and 1.029 in late alignment.
These sum all three action dimensions, use the pre-tanh diagonal Gaussians,
and are **not** the same measurement as the training minibatch estimate.
The same invertible squashing/scaling preserves distribution KL in theory;
we compute before numerical action saturation. Steering alone accounts for
0.987 nats in the bonus zone but only 0.054 in late alignment; throttle/brake
dominate the latter. These are descriptive policy distances, not safety scores.

The saved periodic checkpoint is after update 24; the final checkpoint is
after update 25. In the bonus zone and the next span, that last update moves
steering in the same direction as the accumulated first 24 on 92% and 84% of
baseline states. In the approach span this is only 1%, and in late alignment
49%. There are no saved policy snapshots for updates 1-23, so we cannot claim
that every individual update moved consistently toward the failure.

Stochastic training completed 184/195 episodes successfully (94.4%), including
7/8 episodes whose completions fell in the final rollout. Those episodes use
exploratory actions; some span update boundaries. They are not a substitute
for deterministic checkpoint evaluations, and the last rollout precedes the
final update. Their success rules out describing the available evidence as a
smooth, observed decline in *all* driving reliability. The final deterministic
mean-action trajectory can fail while nearby sampled trajectories still finish.
We have not established that exploration itself rescued those particular laps.

## Notable moments and limits

- The most informative difference is roughly half a degree of launch-velocity
  direction, not a large late steering error or new speed increase. Its effect
  accumulates across several seconds of flight.
- The KL guard was active and correct as an early-stop, but a "tight leash"
  analogy is incomplete: its reference resets every rollout and prior steps
  are not rolled back. Cumulative drift is real; uniformly harmful update
  direction remains unproven.
- The apparent good stochastic training performance masked a deterministic
  failure mode. This is why the original live deterministic gate remains the
  decision source, and why no continuation follows from these training finishes.
- Bonus causality, the exact collision surface, and each update's contribution
  remain unresolved. No zone, coefficient, or KL revision is proposed here.

## Reproduce and inspect

Run `.venv\Scripts\python.exe scripts/diagnose_wr_stage2b_alignment.py`.
It verifies input hashes, loads policies for inference only, and writes derived
JSON and a static comparison plot under
`artifacts/analysis/wr_chase_stage2b/alignment_diagnosis/`. It never connects to
the game or writes checkpoints. The five new tests cover progress alignment,
clock boundaries, flight selection, analytic KL, and exact guard behavior.

Derived JSON SHA-256:
`ECB0459010DC5154E1BA7DFFBF99BB7C5C9B8AFB5B54B3FBE6D833EEC9F157BD`.
It includes hashes of all input models/logs and selected replay files.

Baseline finish replay inputs: best episode 02 (24.790s), closest-to-mean
episode 05 (24.830s; actual mean 24.838s), worst episode 09 (24.980s), in
`artifacts/replays/wr_chase_stage2b/gate_00000000/`.

Gate 50k has no finishes: highest-, median-, and lowest-progress failures are
episodes 01, 07, and 04 in `artifacts/replays/wr_chase_stage2b/gate_00050000/`.
The JSON contains the exact filenames and checksums. These remain input
replays, not newly rendered videos; the prior MP4 capture blocker is unchanged.
