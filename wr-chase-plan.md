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

### Observed Gate 1 result: 500,000 nominal interactions

This section was added after Gate 1 completed. The manifest preserves the
pre-training version of this document as SHA-256
`13D2ADE20C6FAB247899BE19C80B6C81753DDC2B35F9638F4E2797BFC9C244C9`.

PPO completed `501,760` actual interactions in one uninterrupted attempt, from
model timestep `2,002,944` to `2,504,704`, in `1,323.615s`. The rollout-boundary
overshoot is disclosed rather than rounded away. All `501,760` audited actions
were finite, within the declared action space, and passed the affine-action
check without hidden clipping. The stochastic training log contained `1,916`
episodes and `1,778` finishes (`92.80%`). Mean race-clock time among the last
500 stochastic finishes was `25.949s`, versus `26.041s` for the preceding 500.

The frozen checkpoint's ten deterministic episodes produced:

- `6/10` finishes, four verified stuck truncations, zero falls, zero timeouts,
  and zero lateral off-track truncations;
- best successful lap `24.750s`;
- mean successful lap `24.788s`;
- worst successful lap `24.870s`.

Successful pace improved over V4's 100-episode baseline by `0.150s` at the best
and about `0.142s` at the mean, but reliability regressed sharply from V4's
`98/100` result. Pace and reliability must therefore remain separate findings.

Full replay telemetry found no candidate or confirmed drift window anywhere on
the track. Both known zones had zero sliding-wheel samples. Maximum absolute
slip angle was `0.321 degrees` in the first-turn region and `0.369 degrees` in
the final-corner region.

The opening strategy changed in the wrong direction for the documented
downhill optimization: all ten runs remained airborne for `2.100s +/- 0.100s`,
minimum throttle rose to about `0.989`, and mean diagonal angle fell to
`1.388 degrees`. The lateral entry offset moved from V4's roughly `-16.98` to
about `-10.40` units relative to the driven reference, but this is not claimed
as a better road-center line because Decision 0003's limitation still applies.

The frozen decision is **continue**: no telemetry discovery occurred, plateau
assessment cannot begin before two million additional interactions, and the
new best improved by more than the registered `0.050s` threshold.

Gate 1 evidence:

- checkpoint SHA-256:
  `0C1E690A2EE07CE5170CC2B95370BA24E31F8C39E0D66B61687D7549158E4A71`;
- training summary SHA-256:
  `A4A037EE7EA340069C410722BAC5C882DADF4CFBED954BB46EE54DCFBD79005B`;
- deterministic evaluation summary SHA-256:
  `AC058C915EDD73B84B275CCE9AEE1D1B9683A0A5EF27C123C025B4B2A71FE5B7`;
- full replay-telemetry summary SHA-256:
  `2AB990C2D178602B6A14A365A1C56BBE4CE6FC424A575A13A4BED666CDA4E78D`;
- clean best (`24.750s`) video SHA-256:
  `617378B00DC41A8C3F53152778D599E13CE83EF5BA0EB3D71CDD2F5E29CA2695`;
- clean closest-to-mean (`24.760s`) video SHA-256:
  `07B6F3A87009523734ACE54EEDE7B77AAE60328785B37A0FB0946A03D6577122`;
- clean worst successful (`24.870s`) video SHA-256:
  `F981F1F057222C8203A5ECE8C7AFCA009350BB69635CBE70F929B52D53C401DC`.

### Observed Gate 2 result: 1,000,000 nominal interactions

Gate 2 resumed the complete Gate 1 optimizer state and ended at model timestep
`3,004,416`, or `1,001,472` actual Stage 1 interactions. Its uninterrupted
`499,712`-interaction segment took `1,312.289s`. Across the cumulative training
log, `3,590/3,808` stochastic episodes finished (`94.28%`). The most recent 500
stochastic successful laps averaged `26.754s`, regressing from `26.052s` in the
preceding 500 despite the deterministic result below.

The frozen ten-episode evaluation produced `10/10` finishes with no falls,
stuck truncations, timeouts, or off-track truncations:

- best: `24.850s`;
- mean: `24.865s`;
- worst: `24.880s`.

Gate 2 restored the observed ten-lap reliability lost at Gate 1, but was
`0.100s` slower than Gate 1's best and about `0.077s` slower at the mean. It
remained exactly `0.050s` faster than the original V4 best.

All ten input replays reproduced the live evaluation trajectory exactly inside
the two measured slide zones under the added fixed fidelity audit: maximum
position and progress error were zero at matched timestamps. This check was
added after a clean playback attempt failed to expose its finish flag within a
one-second allowance; rather than keep widening the finish tolerance, analysis
now stops at the original evaluation cutoff and permits a slide conclusion only
when the earlier measured zones match the live position log within two units.

The trusted telemetry again found zero candidate or confirmed drift windows,
with zero sliding-wheel samples. Maximum absolute slip angle was below
`0.354 degrees` in the first-turn region and below `0.350 degrees` in the final
corner. The opening drop returned to `2.000s` observed airtime, but still held
roughly `98.5%` throttle and used only about a `2.335-degree` mean diagonal, so
the documented release/diagonal optimization had not emerged.

The frozen decision remains **continue**. No discovery occurred, fewer than two
million additional interactions have elapsed, and the original V4 best was
still improved by the registered `0.050s` threshold.

Gate 2 evidence:

- checkpoint SHA-256:
  `BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044`;
- training summary SHA-256:
  `1CBA6EE72AACA7670AF0D95A320C20002555DF708BFE577EDFE6C1230CB9E0A5`;
- deterministic evaluation summary SHA-256:
  `894B5CF73538580F80031CB62EAE86687B201AE8BC0C080B37A0380B2A9B47C3`;
- full replay-telemetry summary SHA-256:
  `78CEFBC732C5934A26152EB289451B6F91A4094428D601FC2A0E3CB2FCAAE595`;
- clean best (`24.850s`) video SHA-256:
  `4A7DED8777B42A8DB30DE4BD2AAE9004A74B5BCF854BFC25CE6D3376A50DABC6`;
- clean closest-to-mean (`24.860s`) video SHA-256:
  `40328A74153730CFF268E0B1AF6BDE2B256BA96DA099C71BD26835439024E125`;
- clean worst (`24.880s`) video SHA-256:
  `6E308EB12F1D820DD2103758B8B654BC1D7FB4266ED6F7585DE01919CFC7B8FB`.

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
