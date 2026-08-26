# A01 world-record chase: staged discovery plan

Status: Stage 1 pre-registered; no Stage 1 training result observed

Date pre-registered: 2026-08-26

This plan is a sibling to `reward-comparison-v1-v4.md`. It does not define a
new reward version. Stage 1 deliberately continues the exact V4 reward so the
project can distinguish techniques discovered by reinforcement learning from
techniques introduced through explicit shaping.

## Benchmarks and scope

- Owner's human PB: `24.500s`.
- Rollin's July 2025 classic-category record: `23.780s`.
- Current verified classic-category benchmark when this plan was written:
  Axell's `23.770s`, set on 2025-09-16.
- Frozen V4 checkpoint: `checkpoints/reward_v4/final_model.zip`, SHA-256
  `6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2`,
  model timestep `2,002,944`.

The project will report gaps to both the owner's PB and the current record. The
Rollin time remains useful as the historical benchmark referenced in the
project direction, but it is not described as the current record.

## Stage 0: V4 prerequisite measurement

Stage 0 replayed five representative successful V4 scale-validation laps: the
best, lower quartile, median, upper quartile, and worst finishes. Full
`SimState` telemetry was sampled every 100 ms from checksum-pinned TMInterface
input replays.

All five laps used effectively the same opening strategy:

| Measurement | Five-lap V4 result |
| --- | ---: |
| all-wheel-airborne time on the opening drop | `2,000ms +/- 100ms` |
| diagonal trajectory angle from the driven reference | mean `3.653 degrees` |
| minimum throttle from takeoff through landing | `0.894` |
| maximum brake from takeoff through landing | `0.000` |
| landing displayed speed | `306-307` |
| lateral offset at progress 560 | mean `-16.979` units |
| displayed speed at progress 560 | mean `325.602` |
| slip angle at progress 560 | mean `0.325 degrees` |

Positive lateral offset is car-right of the reference, so V4 is roughly 17
units car-left of that line at the fixed entry sample. This is relative to the
intentionally cautious `33.280s` driven reference in Decision 0003, not a
surveyed road center. It therefore does not prove the car is 17 units left of
the geometric road center.

V4 has a mild incidental diagonal trajectory, but not the documented optimized
drop strategy: it makes no meaningful accelerator release and shows almost no
variance across laps. Its first-turn entry sample is not a slide attempt. The
claim that the optimal entry must be closer to road center remains unverified;
the available written strategy instead describes a somewhat wider approach.
No lateral target will be added without a modern record-replay or track-geometry
baseline.

Stage 0 evidence is reproducible with
`scripts/analyze_v4_wr_prerequisites.py`. The local generated summary is
`artifacts/analysis/v4_wr_prerequisites/analysis_summary.json`, SHA-256
`8698EF85939EC32D16A1A7AD1CCDF263AD1406662359D4AA6EBC2B2FCCB9CAA2`.

## Stage 1: unchanged-reward pure discovery

### Frozen hypothesis

> Given sufficient training time, the existing progress-based reward may be sufficient for the agent to discover slip-angle/drift behavior near the first turn and/or final corner unassisted, similar to Yosh's AI, but is expected to plateau below world-record pace without targeted assistance.

This sentence is recorded exactly as approved before extended training starts
and must not be edited after seeing the result.

### Unchanged reward and environment

At each 100 ms action step, Stage 1 retains V4 exactly:

```text
progress_delta = current_progress - previous_progress
signed_progress = clip(progress_delta, -20.0, 20.0) / 10.0
reward = signed_progress - 0.10

if race_finished:
    reward += 50.0
else if verified_failure:
    reward -= 250.0
```

There is no drift, slip-angle, airtime, speed, throttle-release, brake, lateral,
heading, steering, checkpoint, or zone-specific term. V5's steering-rate term
and V6's reversal-frequency term are absent. V4's failure detector, 45-second
episode limit, 100 ms control period, PPO/gSDE hyperparameters, seed lineage,
and 100x training simulation speed remain unchanged.

Continuing the V4 policy rather than restarting is deliberate. Stage 1 asks
whether more experience under the already successful reward discovers a better
strategy; it does not retest route learning from scratch.

### Training cadence and time box

- Train in nominal `500,000`-interaction gates. PPO may pass a gate by up to one
  `2,048`-step rollout; actual model timesteps, not rounded labels, are reported.
- Save optimizer-bearing checkpoints every `250,000` nominal interactions and
  at every evaluation gate.
- Start with `5,000,000` additional interactions, five times a prior reward
  version's planned budget.
- Do not stop at `5,000,000` if the most recent `1,000,000` interactions still
  contain a new deterministic stage best by at least `0.050s`. Continue in
  `500,000` gates while that condition holds, up to a safety ceiling of
  `10,000,000` additional interactions.
- Reaching the safety ceiling without discovery or plateau is an inconclusive
  time-box result. It is not permission to redefine the gates.
- TensorBoard name: `wr_chase_stage1_v4_pure_discovery`.
- Run artifacts live under `runs/wr_chase_stage1/` and checkpoints under
  `checkpoints/wr_chase_stage1/`; existing V4 evidence is never overwritten.

Host sleep and hibernation must pass the existing disabled-power preflight
before each resumed training attempt.

### Evaluation cadence

After every nominal 500,000-interaction gate:

1. Freeze and checksum the optimizer-bearing checkpoint.
2. Run 10 deterministic A01 episodes using the V4 detector at 6x.
3. Preserve all 10 TMInterface input replays and the action audit.
4. Replay those inputs through full `SimState` telemetry at 100 ms resolution.
5. Record finish rate and best/mean/worst race-clock time, plus opening-drop
   airtime, throttle/brake behavior, diagonal angle, landing speed, first-turn
   entry line, wheel-slide flags, slip angle, and body-up yaw rate.
6. Preserve clean overlay-free videos for the best, closest-to-mean, and worst
   successful laps at the terminal Stage 1 gate.

The A01 zones used only for measurement are:

- first known speedslide region: reference progress `680-930`, encompassing
  the turn whose strong driven-reference curvature lies around `715-875`;
- final-corner speedslide region: reference progress `1100-1410`, encompassing
  the strong driven-reference curvature around `1140-1355`.

These ranges do not change reward. A track-wide scan also records candidate
events outside them. The transcript's third possible slide near the Uphill is
not localized reliably and is explicitly excluded from Stage 1 zone claims.

### Pre-registered slide measurements

A **candidate drift window** is at least two consecutive 100 ms samples with:

- at least three wheels grounded;
- displayed speed at least `350`;
- at least one wheel reporting `is_sliding`;
- absolute velocity slip angle at least `1.0 degree`; and
- absolute body-up yaw rate at least `0.25 rad/s`.

A **confirmed telemetry slide** is at least three consecutive samples
(`>=300ms`) meeting the same speed, contact, slip-angle, and yaw requirements,
with at least two wheels reporting `is_sliding` on every sample.

The thresholds deliberately sit well above V4's measured baselines: across the
five Stage 0 laps, both known regions had zero sliding wheels, maximum absolute
slip angle below `0.36 degrees`, and maximum absolute body-up yaw rate below
`0.83 rad/s`. A confirmed telemetry slide must appear in at least `3/10`
deterministic episodes at the same checkpoint to trigger the Stage 1 discovery
report. Visual review must then distinguish a useful speedslide from wall
contact, landing instability, or a crash. A single noisy event is reported as a
candidate, not declared a learned technique.

Discovery outside the two known regions is reported separately and does not
justify localized Stage 2 shaping until its usefulness is verified.

### Pre-registered plateau rule

Plateau assessment begins only after `2,000,000` additional interactions. A
clear plateau requires all of the following at three consecutive 500,000-step
evaluation gates:

1. no confirmed telemetry-slide discovery;
2. no new deterministic stage-best lap by at least `0.050s`;
3. less than `0.050s` improvement in deterministic mean successful lap time
   from the first to the third gate; and
4. less than `0.100s` improvement between the preceding and most recent 500
   successful stochastic training episodes.

If the policy is still improving at the latest gate, training continues within
the time box. Stage 1 reports immediately after a confirmed discovery or a
clear plateau, whichever occurs first. The fixed thresholds are not adjusted
after looking at results.

## Stage 2: localized assistance, held behind Stage 1

No Stage 2 reward is implemented or pre-registered yet. If Stage 1 clearly
plateaus without a useful slide, Stage 2 may add a localized slip/drift bonus
only in a region where replay analysis and established A01 strategy show that a
real speedslide saves time. The first-turn and final-corner regions are the only
current candidates. The Uphill claim remains an investigation item, not a
reward zone.

Stage 2 will be its own clean reward iteration, initialized from a checksum-
pinned Stage 1 checkpoint and pre-registered before training. It will not
hand-script steering, braking, or throttle sequences.

## Realistic expectation

Yosh's public result shows that pure progress reward can discover the drop
release, diagonal jump, and a useful speed-drift. It does not make discovery
likely on any particular compute budget. Those maneuvers require a narrow
sequence of approach speed, line, brake timing, steering, and slide angle.
This project acts at 10 Hz while Yosh described a 20 Hz controller, which makes
fine timing less expressive here. Stage 1 is therefore a genuine discovery
test, not a promised route to the record.
