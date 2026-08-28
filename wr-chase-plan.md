# A01 world-record chase: staged discovery plan

Status: Stage 2 paused at the pre-registered Gate 1,000,000 safety review;
the frozen policy-mode diagnostic found an unstable `7/10` stochastic result
and the Gate 750,000 diagnostic located the deterministic regression between
Gate 500,000 and Gate 750,000; no drift induction was observed

Date pre-registered: 2026-08-26

Stage 2 date pre-registered: 2026-08-27

This plan is a sibling to `reward-comparison-v1-v4.md`. Stage 1 deliberately
continued the exact V4 reward so the project could distinguish techniques
discovered by reinforcement learning from techniques introduced through
explicit shaping. Stage 2 is the separate localized-assistance iteration.

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

### Observed Gate 3 result: 1,500,000 nominal interactions

Gate 3 resumed Gate 2 and ended at model timestep `3,504,128`, or `1,501,184`
actual Stage 1 interactions. Its uninterrupted `499,712`-interaction segment
took `1,320.502s`. The cumulative stochastic log contained `5,215/5,650`
finishes (`92.30%`). Its most recent 500 stochastic successful laps improved to
`25.821s` from `26.252s` in the preceding 500, but that training-distribution
improvement did not predict the frozen deterministic result.

The formal ten-episode evaluation regressed to:

- `1/10` finishes;
- eight verified stuck truncations;
- one confirmed fall;
- no timeout or lateral off-track truncation; and
- a sole successful lap of `29.100s`.

The only successful lap is simultaneously Gate 3's best, closest-to-mean, and
worst finish. Gate 3 therefore lost both Gate 2's reliability and pace.

Input-replay telemetry was trustworthy in both measured zones for nine
episodes, but episode 1 diverged by up to `937.41` world units in the final zone
and showed four sliding wheels on the divergent trajectory. The fidelity gate
correctly rejected that apparent slide instead of counting it as discovery.

To resolve the blocked tenth episode, a separate ten-episode deterministic
diagnostic logged full wheel and dynamics state directly during live policy
steps. It did not replace the formal outcomes. The diagnostic independently
produced `1/10` finishes, this time with nine stuck truncations, confirming that
the reliability collapse was reproducible rather than a single unlucky batch.
All ten live trajectories had zero candidate or confirmed drift windows and
zero sliding-wheel samples in both known zones. Maximum absolute slip angle
remained below `0.355 degrees` at the first turn and `0.378 degrees` at the
final corner.

The live diagnostic's opening drop used `2.000-2.100s` observed airtime,
roughly `93.7-94.2%` minimum throttle, and an inconsistent trajectory angle:
about `+0.74` degrees in seven runs and `-0.98` degrees in three. It was not the
documented accelerator-release/diagonal optimization.

The frozen decision remains **continue**. Gate 3 contains no discovery and the
two-million-interaction minimum for plateau assessment has not yet been met.
Gate 4 is the first point where the three-gate plateau rule can stop Stage 1.

Gate 3 evidence:

- checkpoint SHA-256:
  `EA08FBD3769B15CCBCAF4963316CB3D863E59EC29E882C1B00AC2E4ACF725CE5`;
- training summary SHA-256:
  `0A4C5EF430293A047483F5F49C1D4F6BCB513BF8A4D3B486040711008E3A1099`;
- formal evaluation summary SHA-256:
  `DF39E053C217F08B531BCBE983FFD3836797AC098539BE869578B8451ED3961A`;
- replay telemetry summary SHA-256:
  `703A4C82DB9EBB5E49CEEA2884CCC61614B5CF2F7094B588310BE2FE5287A783`;
- direct live diagnostic summary SHA-256:
  `6C20BC0992573A1FBF862AD62DB0CA1C4B1B84A6E8D45C25BF7AD808B28AFC01`;
- clean sole-finish (`29.100s`) video SHA-256:
  `2BA82D0642A2E4714ED948F87905E9D9D75D68131E4B19C55081AE4E7135B5FC`;
- clean representative stuck-input video SHA-256:
  `54DD6381BC30F3C1069A9321736CC775C8E244D524DF99D14E6EBCE4A9589C76`.

The attempted clean capture of formal fall episode 10 did not reproduce the
fall after the saved input endpoint and is not presented as visual evidence of
that failure. Its non-reproduction is preserved rather than silently discarded.

### Observed Gate 4 and terminal Stage 1 result: 2,000,000 nominal interactions

Gate 4 resumed Gate 3 and ended at model timestep `4,003,840`, or `2,000,896`
actual Stage 1 interactions. Its uninterrupted `499,712`-interaction segment
took `1,327.243s`. All `499,712` actions were finite, in range, affine-correct,
and free of hidden clipping. Cumulative stochastic training produced
`6,985/7,532` finishes (`92.74%`). The last 500 successful stochastic laps
averaged `26.385s`, regressing by `0.119s` from `26.266s` in the preceding 500.

The frozen ten-episode deterministic evaluation recovered completely from Gate
3's reliability collapse:

- `10/10` finishes;
- zero falls, stuck truncations, timeouts, and lateral off-track truncations;
- best: `24.870s`;
- mean: `24.932s`; and
- worst: `25.190s`.

Gate 4 remained `0.120s` slower than Gate 1's Stage 1 best and was effectively
equal to the original V4 mean. It was `0.370s` slower than the owner's
`24.500s` PB and `1.100s` slower than Axell's `23.770s` benchmark at best.

All ten slide conclusions came from complete, terminal-outcome-matched live
`SimState` records captured while the policy drove, not from replayed physics.
The checksum-pinned input replays remain preserved for reproducibility and
video. This source choice prevents a divergent input replay from manufacturing
the kind of false four-wheel-slide signal observed at Gate 3.

No confirmed telemetry slide occurred anywhere on the track. Both registered
zones had zero sliding-wheel samples and zero candidate windows. Maximum
absolute slip angle was `0.422 degrees` in the first-turn zone and `0.375
degrees` in the final-corner zone. Two of ten laps had a separate `200ms`
four-wheel candidate near progress `2132`, close to the final landing/finish.
It was outside the registered zones, did not meet the frozen `300ms`
confirmation duration, and is not claimed as a useful technique.

The opening strategy also did not become the documented optimization. Every
lap recorded `2.000s +/- 0.100s` opening airtime. Mean minimum throttle through
the drop was `0.900`, mean diagonal angle was `1.377 degrees`, and mean landing
speed was `312.6`. First-turn entry at progress 560 averaged `-7.728` lateral
units relative to the cautious driven reference, displayed speed `329.420`,
and only `0.544 degrees` of slip. Longer unchanged-reward training changed the
line but did not discover a throttle-release diagonal drop or a speedslide.

The preregistered decision is **plateau**, so Stage 1 stops here. Every frozen
check passed across the Gate 2/Gate 3/Gate 4 window: at least two million
interactions elapsed; no confirmed discovery occurred; no checkpoint created a
new stage best by at least `0.050s`; Gate 2's `24.865s` mean did not improve at
Gate 4; and the latest stochastic 500-finish window regressed rather than
improving by `0.100s`. Reliability was intentionally not retrofitted into the
plateau rule, but Gate 4's `10/10` result independently shows that this terminal
decision is not merely the result of another Gate 3-style failure collapse.

The observed progression was non-monotonic:

| Checkpoint | Actual additional interactions | Deterministic finishes | Best | Mean successful | Confirmed slide |
| --- | ---: | ---: | ---: | ---: | ---: |
| frozen V4 baseline | `0` | `98/100` | `24.900s` | `24.931s` | none in Stage 0 sample |
| Gate 1 | `501,760` | `6/10` | `24.750s` | `24.788s` | no |
| Gate 2 | `1,001,472` | `10/10` | `24.850s` | `24.865s` | no |
| Gate 3 | `1,501,184` | `1/10` | `29.100s` | `29.100s` | no |
| Gate 4 | `2,000,896` | `10/10` | `24.870s` | `24.932s` | no |

The frozen hypothesis was only partly supported. The model did plateau below
record pace as expected, but sufficient training under this time box did not
produce the hypothesized unassisted slide. The useful positive result was a
`0.150s` Stage 1 best-lap gain at Gate 1; it came with worse reliability and
did not survive as a monotonic trend. Pure additional V4 training therefore
does not justify more compute under the same reward in this experiment.

This is a genuine plateau **within the tested budget**, not evidence that pure
discovery is impossible at greater scale. `2,000,896` decisions at this
project's 10 Hz control rate represent about `55.6` simulated control-hours.
Yosh described roughly `400` equivalent training-hours with a 20 Hz
controller, which would be about `28.8 million` action opportunities if
continuous. The figures are not directly comparable because his reset time,
algorithm, reward scale, and definition of equivalent hours are not disclosed.
They nevertheless show that this project's Stage 1 used far less exploration,
so its conclusion is deliberately limited to the registered two-million-step
budget.

Gate 4 and terminal Stage 1 evidence:

- checkpoint SHA-256:
  `E0F1C2113C57406A1E32A0A341572DEA7AFC705483D6BD5B17751AD990626609`;
- training summary SHA-256:
  `ED0871EB7A6E7F89B71F9F6C35C4A15B2CBD8986CAAA85F1D6599894C0E116FE`;
- deterministic evaluation summary SHA-256:
  `0AB9BD7D1B032D92327BA5B346657F13FC71C37ACCD7A449AA09493EF576449C`;
- direct live action log SHA-256:
  `CA4140D907D0A61CB20F90765CCC4E8CCA826CB06E7F320FE071F38C4C9946BB`;
- telemetry summary SHA-256:
  `795B5AFB1A263EC8FD9177ED918A4E1B6DDA0A4A708A7FF784B6288EB0B56591`;
- clean best (`24.870s`) video SHA-256:
  `E477A41E9D94645E5F1BDDD42F850B57CE0D4C91A2E4AD0606B2E6F4C25B37FC`;
- clean closest-to-mean (`24.900s`) video SHA-256:
  `4B10BAD555AD0B92C7479CB81983C34CFB9C0BECEDE7F5C9B45375DCBD220931`;
- clean worst (`25.190s`) video SHA-256:
  `E85BE3E34863B04DCE175EBFF23233F19903A180CD91C4A075DC5F21A5CB3D59`.

At the time this terminal result was committed, no Stage 2 reward or training
had started. The owner subsequently approved the separately pre-registered
localized-assistance experiment below.

## Stage 2: localized drift assistance

Stage 2 follows the structure of Yosh's intervention, not an undisclosed exact
formula. His transcript says that the AI received extra reward for drifting
inside one specific section, but gives no coefficient, physics threshold,
duration, or coordinate bounds. The exact formula below is this project's
pre-registered engineering choice. The owner's approved direction applies it
to both established A01 speedslide zones. It does not claim that Yosh's text
itself identifies or rewards both zones.

### Frozen Stage 2 hypothesis

> Adding a bounded reward for new forward progress made while live telemetry confirms drift-like dynamics, only inside the first-turn and final-corner zones, is expected to induce repeated drift attempts in one or both zones where unchanged V4 training produced none. Consistent with Yosh's result, the first induced attempts are expected to be imprecise and may initially worsen lap time or finish reliability before technique quality improves.

This sentence is recorded before the Stage 2 baseline or training is run and
must not be edited after observing a result.

### Initialization checkpoint

Stage 2 starts from Stage 1 Gate 2's optimizer-bearing checkpoint:

- path: `checkpoints/wr_chase_stage1/gate_01000000_model.zip`;
- SHA-256:
  `BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044`;
- model timestep: `3,004,416`; and
- frozen result: `10/10` finishes, `24.850s` best, `24.865s` mean,
  `24.880s` worst, with no falls or stuck episodes.

Gate 2 is the strongest pace/reliability compromise among Stage 1 checkpoints.
Gate 1 was `0.100s` faster at best but finished only `6/10`; Gate 3 collapsed
to `1/10`; Gate 4 recovered to `10/10` but was slower and much more variable
than Gate 2. Retaining Gate 2's optimizer state avoids combining the reward
change with an optimizer reset. Its Adam moments were learned under V4, but
that limitation applies to every optimizer-bearing Stage 1 candidate.

Before training, the unchanged Gate 2 policy will receive a fresh 10-episode
deterministic baseline through the direct-live Stage 2 telemetry path. This is
an instrumentation/comparability check, not another checkpoint-selection
contest; the preserved result above remains the reason Gate 2 was selected.

### Exact Stage 2 reward addition

The V4 reward and every terminal term remain unchanged. Let
`maximum_progress_before_step` be the greatest reference-path progress reached
earlier in the current episode:

```text
new_progress_delta = max(
    0,
    current_progress - maximum_progress_before_step,
)
new_progress_fraction = clip(new_progress_delta, 0, 20) / 20

in_assisted_zone = (
    680 <= current_progress <= 930
    or 1100 <= current_progress <= 1410
)

drift_attempt = (
    ground_contact_count >= 3
    and displayed_speed >= 350
    and sliding_wheel_count >= 1
    and abs(slip_angle_degrees) >= 1.0
    and abs(body_up_yaw_rate) >= 0.25
)

localized_drift_bonus = (
    0.50 * new_progress_fraction
    if in_assisted_zone and drift_attempt
    else 0.0
)

stage2_reward = v4_reward + localized_drift_bonus
```

The coefficient is capped at `0.50` per 100 ms step. Even if every unit of the
combined `560`-unit assisted range qualified on its first traversal, the
high-water-progress scaling bounds the approximate one-pass bonus to `14.0`.
That is material enough to distinguish drift attempts but remains below V4's
`+50` finish bonus and far below its `-250` verified-failure penalty.

Using episode-high-water progress is deliberate: reversing and repeatedly
crossing the same zone cannot earn the bonus again. One sliding wheel is enough
to reward an exploratory attempt, while the stricter measurement rule below
still requires two sliding wheels and sustained behavior before the project
claims that a slide was learned. The bonus is binary with respect to angle once
the minimum is met; it does not encode an optimal speedslide angle or reward
larger, crash-prone slip angles.

This design retains a real sparsity risk: Stage 1 recorded zero sliding-wheel
samples in both zones, so Stage 2 is identical to V4 until exploration first
crosses every binary eligibility threshold at once. If that never happens, the
registered interpretation is that this localized binary assistance remained
too sparse within the Stage 2 budget. The thresholds or coefficient will not be
softened after observing that outcome.

Wheel contact, wheel sliding, velocity slip angle, and body-up yaw rate must
come directly from the live `SimState` used for that policy step. They are
available only to reward calculation and audit logging, not added to the PPO
observation. Missing or incomplete live dynamics is a hard instrumentation
failure. Stage 2 never substitutes input-replay physics for a reward or slide
conclusion.

No reward is added outside the two registered zones. There is no Uphill zone,
track-wide drift reward, direct steering/brake/throttle term, steering
smoothness term, target path, target angle, or scripted action sequence.

### Frozen Stage 2 gates

- Training uses nominal `500,000`-interaction gates at 100x with the existing
  100 ms action period. PPO rollout overshoot is reported exactly.
- Save optimizer-bearing checkpoints every nominal `250,000` interactions and
  at each gate.
- At every gate, run 10 deterministic episodes at 6x, preserve all input
  replays and the complete action audit, and compute slide/prerequisite metrics
  exclusively from terminal-outcome-matched direct-live `SimState` records.
- Preserve finish rate; best, mean, and worst race-clock times; lateral and
  oscillation metrics; both zone-specific bonus totals; candidate/confirmed
  slide windows; opening-drop measurements; and clean videos for review gates.
- Candidate and confirmed windows retain Stage 1's frozen thresholds. A
  confirmed slide requires at least three consecutive 100 ms samples with at
  least three wheels grounded, speed `>=350`, absolute slip `>=1 degree`,
  absolute body-up yaw `>=0.25 rad/s`, and at least two sliding wheels.
- A repeated induction requires a confirmed slide in the same registered zone
  in at least `3/10` deterministic episodes. Stop and report that gate for
  visual review immediately; pace improvement is not required because early
  slowdown and imprecision are part of the hypothesis.
- Invalid/incomplete live telemetry stops the experiment as an instrumentation
  failure. There is no replay fallback.
- Safety review triggers after `0/10` finishes at one gate, or `<=5/10`
  finishes at two consecutive gates. One weak gate alone is retained because
  early reliability loss is pre-registered as plausible.
- The initial Stage 2 time box is `2,000,000` interactions. If no repeated
  induction occurs, report the bonus as ineffective within that budget and do
  not change its coefficient after the fact. No automatic extension is allowed.

TensorBoard name: `wr_chase_stage2_localized_drift`. Artifacts use distinct
`runs/wr_chase_stage2/`, `checkpoints/wr_chase_stage2/`,
`artifacts/replays/wr_chase_stage2/`, and
`artifacts/analysis/wr_chase_stage2/` roots. Stage 1 evidence is immutable.

### Target-zero direct-live baseline result

The untouched Stage 1 Gate 2 initialization was evaluated once through the
frozen Stage 2 path before training. This was a comparability and
instrumentation gate, not checkpoint selection. It completed `9/10` episodes:

- best finish: `24.850s`;
- mean of the nine finishes: `24.861s`;
- worst finish: `24.870s`; and
- one non-finish: stuck termination at `32.600s`, with no fall, off-track, or
  upside-down event.

The earlier Stage 1 Gate 2 evaluation finished `10/10`. The fresh `9/10`
result is retained rather than rerun away. The policy is deterministic, so the
difference is evidence of live simulator/reset-state variability rather than
stochastic policy sampling. It does not trigger Stage 2's trained-gate safety
rule because target zero was pre-registered as a baseline only.

There was no pre-existing assisted behavior: both the first-turn and
final-corner zones had zero candidate windows, zero confirmed windows, zero
eligible attempt steps, and exactly `0.0` localized bonus. Independent
reconstruction matched every logged reward and every high-water-progress delta
with maximum absolute error `0.0`. The result therefore passed the direct-live
gate with decision `continue`; any later repeated confirmed window can be
compared against a genuinely zero-attempt baseline. Had target zero already
shown a repeated confirmed slide in at least `3/10` episodes, the frozen logic
would have stopped for a baseline-comparability review instead of attributing
that behavior to Stage 2.

The prerequisite measurements were stable across all ten runs: opening-drop
airtime was `2,000ms`; the drop chord was `2.30` to `2.41` degrees diagonal to
the reference; first-turn entry lateral offset was `-11.13` to `-10.77` units;
and entry speed was `329.53` to `330.51`. Precision remained recognizably V4:
oscillation was detected in `10/10` episodes, mean significant steering
reversals were `32.3`, mean p95 absolute lateral offset was `12.209` units, and
maximum absolute lateral offset was `18.653` units. The stuck episode
contributed `2.1s` of detected stuck duration; upside-down duration was zero.

Episode zero contained a disclosed 46-row startup/reset prefix, producing 294
raw action rows for a `24.870s` finish. Lap time comes from the live terminal
race clock, not row count. Those prefix rows remain included in the exact
reward audit while the post-restart rows alone feed lap and trajectory metrics.

Evidence:

- evaluation summary SHA-256:
  `2D4C4B36F85FBFF5E6EC447ADE5732D9EFE87A2769CFC176E140DC58367218C2`;
- direct-live action log SHA-256:
  `DE0031B312CE6758DBE32BA71CC670BFC83159958575ACA277FF8C99EB8F11FF`;
- independent analysis SHA-256:
  `8FE691694FDE44DF6C70F1187645985F8F3DAAFB6FD2260C954D6276C5E9F9D5`;
- all ten nonempty input replays were checksum-bound to the evaluation before
  the gate could continue;
- clean best (`24.850s`) video SHA-256:
  `149A12529524FFA774D4BFE231E6097A58C1DD1A801B306483606B0775519462`;
- clean closest-to-mean (`24.860s`) video SHA-256:
  `5658BB4778C18FC54CFE33D88BE78DC9FC7322611E1B2E32E91EAB7ACF626BDC`;
- clean worst (`24.870s`) video SHA-256:
  `FA715DC67178BE634BDDF782932E2DC37E3B96067F3B466AE13CE5D9B5F01E98`;
  and
- every delivered video manifest records H.264, `overlay: false`, and a
  successful replay finish.

### Gate 500,000 result

The first Stage 2 segment completed `501,760` interactions rather than exactly
`500,000` because PPO finished its current rollout. The overshoot is disclosed
and retained. The resulting deterministic evaluation again finished `9/10`
episodes:

- best finish: `24.780s`;
- mean of the nine finishes: `24.816s`;
- worst finish: `24.900s`; and
- one non-finish: upright stuck termination at `26.400s`, maximum progress
  `2161.96`, with no fall, off-track, or upside-down event.

Compared with target zero, best time improved `0.070s` and mean finish time
improved about `0.046s`, while the worst finish was `0.030s` slower. This is
not evidence that localized assistance worked. The independent live audit
found zero eligible attempt steps, zero candidate windows, zero confirmed
windows, and exactly `0.0` bonus in both registered zones. The checkpoint only
received the unchanged V4 component of the Stage 2 reward during all observed
qualifying decisions, so the modest pace gain is attributed to continued PPO
training on that base signal.

Precision regressed despite the pace gain. Mean p95 absolute lateral offset
rose from `12.209` to `17.770` units and maximum absolute offset rose from
`18.653` to `24.176`. Mean significant steering reversals rose from `32.3` to
`39.7`, even though mean steering total variation fell from `39.981` to
`34.931` and mean absolute steering fell from `0.625` to `0.444`. That pattern
is consistent with more frequent, smaller corrections, not a learned slide.
Oscillation remained detected in `10/10` episodes.

The approach did change incidentally. First-turn entry moved closer to the
reference center, from the baseline's `-11.13..-10.77` range to
`-7.79..-7.53`, at essentially unchanged `329.35..330.35` entry speed. Opening
drop airtime remained `2,000ms` in nine runs and rose to `2,100ms` in one;
drop-chord diagonal angle narrowed to `1.26..1.84` degrees. Thus one claimed
speedslide prerequisite improved without generating even a candidate slide,
while airtime did not improve.

The live reward and progress-high-water reconstructions again had maximum
absolute error `0.0`. All ten raw input replays were checksum-bound and
preserved. No safety or induction threshold fired, so the frozen decision is
`continue` to Gate 1,000,000 without changing the coefficient or thresholds.

Evidence:

- checkpoint SHA-256:
  `8A06E00055886E8E671E988D5D6948C6688B74780EDB32A87C6CC213870C8326`;
- training summary SHA-256:
  `1632251A0B036EA776B73A8B400BCBF389A5440EDDDB06644A75E653946CAFC0`;
- evaluation summary SHA-256:
  `786F236AC7DA51BBF85F3E6678CCA0440DB24D718FFA59A14A5601C08D190C45`;
- direct-live action log SHA-256:
  `BA1126A9CF7EA8076E32DDDA602D8DCB6D155219B510B5839719D092236B6026`;
- independent analysis SHA-256:
  `B0CBED778FCC775DA6ACF8E40B48534EC54654E448B07A562E4920092874ADD7`;
- clean best (`24.780s`) video SHA-256:
  `E015AA97BD56B6E6E2F5E537E5B370FDCB2DD0E588788B4D0A483DC638951CD7`;
- clean closest-to-mean (`24.800s`) video SHA-256:
  `21B454B550A2A672304FF72F9A0D1ECDDBD4E60B09C4326D8809C7263F026914`;
- clean worst (`24.900s`) video SHA-256:
  `6B2206F73A726F045900EE3BC413B8300C10046460A0F68CBA84252FCC73941E`;
  and
- all three video records are H.264, successful finishes, and
  `overlay: false`.

### Gate 1,000,000 safety-review result

Gate 1,000,000 resumed the checksum-bound Gate 500,000 optimizer state and
trained for another `499,712` interactions. The cumulative Stage 2 total is
therefore `1,001,472`, and the model ended at timestep `4,005,888`; both small
overshoots come from completing PPO rollout boundaries. The resulting
checkpoint is `checkpoints/wr_chase_stage2/gate_01000000_model.zip`, SHA-256
`F1FCCB19F80192C7A6528BC183C9CD8317A7F7EAA35472E189A56699A85C92FF`.

The deterministic result was the pre-registered single-gate safety trigger:

- `0/10` finishes and therefore no best, mean, or worst lap time;
- `10/10` stuck terminations, with `0` falls, off-track terminations, or
  timeouts;
- terminal race clocks from `26.800s` to `30.300s`;
- final reference-path progress from `2167.61` to `2186.19` units, with best
  maximum progress `2186.25 / 2206.53`, about `20.27` units short; and
- upside-down periods in `9/10` episodes (`29.0s` total) plus detected stuck
  periods in all ten (`21.0s` total).

The clean replay review locates the failure at the finish structure rather
than earlier on the course. The closest run reaches the finish platform but is
inverted beside the finish trigger; the other preserved representatives are
wedged against or below the surrounding structure. This supports a final
alignment/collision failure, but it does not prove that one identical contact
caused all ten outcomes.

The direct-live reward audit remained exact and used no replay-telemetry
fallback. In the ten deterministic episodes, both registered zones had zero
eligible attempt steps, candidate windows, confirmed windows, rewarded steps,
and exactly `0.0` localized bonus. Reward and progress-high-water
reconstruction errors were both `0.0`. Gate 1,000,000 therefore provides no
evidence that Stage 2 induced a drift. The audit does not establish whether an
eligible bonus event occurred during stochastic training, because training
interactions were not recorded with the same full direct-live audit.

Reliability and precision worsened across the three direct comparisons:

| Metric | Target zero | Gate 500,000 | Gate 1,000,000 |
| --- | ---: | ---: | ---: |
| deterministic finishes | `9/10` | `9/10` | `0/10` |
| mean p95 absolute lateral offset | `12.209` | `17.770` | `23.585` |
| maximum absolute lateral offset | `18.653` | `24.176` | `30.594` |
| mean significant steering reversals | `32.3` | `39.7` | `42.8` |
| oscillation-detected episodes | `10/10` | `10/10` | `10/10` |

The training rollouts still reported a high stochastic cumulative finish rate
(`3623/3829`, or `94.62%`; `1769/1902`, or `93.01%`, during this segment),
while the frozen deterministic policy failed all ten gate episodes. That is a
real stochastic-versus-deterministic policy-mode divergence, not a reason to
replace or rerun the registered result.

The opening approach also changed again without producing a slide. All ten
runs measured `1,900ms` opening-drop airtime, a `1.64..1.74` degree drop chord,
and `308..313` landing speed. First-turn entry moved back left to
`-10.48..-10.18` units at `330.70..331.62` displayed speed. Maximum absolute
slip in either assisted zone remained below `0.367` degrees and no wheel was
reported sliding there.

Per the frozen `0/10` single-gate trigger, the decision is `safety_review`.
Gate 1,500,000 was not run. Stage 2 is not supported within the executed
one-million-interaction portion and failed its safety gate; this is not a claim
that drift discovery is impossible at greater scale, nor is it the registered
two-million-interaction time-box conclusion because that endpoint was never
reached.

Before this segment, one launch attempt ended before training when the original
`60s` readiness window expired before TMInterface exposed its bridge. Later
inspection showed port `8478` became available after that cutoff. No model
interaction occurred, the launcher allowance was raised to `120s` with a
regression test, and the successful retry loaded the exact same Gate 500,000
checksum. The failed-before-training record remains preserved rather than
being hidden.

Evidence:

- training summary SHA-256:
  `D894BB206BF4ED449CEACE4601FA054576D9342C810FCE0795926E82B0672969`;
- evaluation summary SHA-256:
  `F96A762148D448FBDF0D6D4B74EE277791C037527E6FE3DE31C30207CDAF34F1`;
- direct-live action log SHA-256:
  `3CB45B494A1695A120F2BB78B69B648E19A9C4BFAB64473A390DC30AAB77E015`;
- independent analysis SHA-256:
  `D76B2900A4E0A83753E3B7173E8878BB2BDFA4C3D76FEAAD919AE1D07BCAF6ED`;
- all ten nonempty input replays were checksum-bound to the evaluation;
- clean closest-progress failure video SHA-256:
  `C978EE87D4EA1382C9BC104F2CA844A913BD3EC18D0B859495F07E90320BC1C2`;
- clean lower-median terminal-progress failure video SHA-256:
  `E28892ED07A27764F5A6F573F96F02E8693C782E34D4D963487961EF69556428`;
- clean lowest-terminal-progress failure video SHA-256:
  `912C30F2063A386E5704A33DCA394ACCC6A4B4915711014E98FA2F84D39C0D80`;
  and
- every delivered video record is H.264 and `overlay: false`; none is labeled
  as a finish because this gate produced no finish.

### Frozen Gate 1,000,000 policy-mode diagnostic

The safety review must distinguish a genuinely collapsed final policy from a
bad deterministic mean action that was masked during training by generalized
state-dependent exploration (`gSDE`). Training sampled actions with
`use_sde=True`, a squashed policy, and new exploration noise every four policy
steps. The registered deterministic gates instead used the policy mean and no
exploration noise.

Before inspecting a stochastic run, perform exactly one diagnostic batch from
the frozen Gate 1,000,000 checkpoint:

- run `10` live episodes at `6x` with no training and no reward change;
- sample the policy stochastically with `gSDE` noise refreshed every `4` policy
  steps, matching training collection;
- fix the diagnostic random seed to `20260828`;
- retain the same `2,000ms` stuck cutoff, action period, snapshot reset,
  direct-live action/SimState logging, and raw input-replay preservation as the
  deterministic gate;
- write to a separate policy-mode diagnostic root and never replace Gate
  1,000,000 evidence; and
- report all ten outcomes without selecting or rerunning a favorable seed.

Interpretation is frozen as follows:

- `>=8/10` stochastic finishes after the deterministic `0/10` means the final
  policy distribution still contains a reliable route but its mean action is a
  finish-alignment failure. Continued unmodified training is still blocked;
  the next experiment must address deterministic robustness rather than claim
  that the whole sampled policy forgot the track.
- `<=2/10` stochastic finishes means the final sampled policy itself also
  collapsed, and the high cumulative training finish rate was primarily a
  historical aggregate rather than evidence about the final checkpoint.
- `3..7/10` is an unstable boundary result: exploration sometimes rescues the
  route, but neither policy mode is reliable enough to resume Stage 2.

This diagnostic does not override the registered Stage 2 safety stop, select a
new checkpoint, or count toward the two-million-interaction training budget.

### Gate 1,000,000 policy-mode diagnostic result

The single registered batch finished `7/10` stochastic episodes. This is the
top of the frozen `3..7/10` unstable-boundary range, not the `>=8/10` reliable
distribution result:

- best finish: `24.810s`;
- mean of the seven finishes: `29.171s`;
- worst finish: `37.440s`; and
- three stuck failures, with no fall, off-track, or timeout termination.

Four finishes remained in the familiar `24.810..24.920s` band, while the other
three took `33.280..37.440s`. Exploration can move the Gate 1,000,000 policy
off its deterministic collision line, but it neither makes that route reliable
nor consistently preserves pace. The stochastic batch still had oscillation
in `10/10` episodes, mean `47.1` significant steering reversals, mean p95
absolute lateral offset `15.206`, maximum absolute lateral offset `29.277`,
two episodes with an upside-down period, and three with a stuck period.

The direct-live audit covered all `2,913` action records, matched every
terminal outcome, and used no replay fallback. Neither assisted zone contained
an eligible bonus step, so the diagnostic again observed no localized drift
reward. It changed no model weight and consumed zero training interactions.

Offline reconstruction rules out an evaluator/checkpoint mismatch. Building
the documented 26-value observation from each preceding live state reproduced
the next logged deterministic action with maximum absolute error below
`7.16e-7` at Gate 1,000,000 (`5.97e-7` at Gate 500,000). The evaluator loaded
the intended policy and used the intended observation and action mapping.

The actual regression is a late-line shift. At fixed progress landmarks, mean
lateral offsets changed as follows:

| Progress | Gate 500,000 | Gate 1,000,000 | Additional negative offset |
| ---: | ---: | ---: | ---: |
| `2000` | `-16.50` | `-22.48` | `-5.98` |
| `2100` | `-20.72` | `-27.39` | `-6.67` |
| `2140` | `-20.40` | `-28.53` | `-8.13` |
| `2160` | `-16.92` | `-25.20` | `-8.27` |

On Gate 1,000,000's live states at progress `>=2100`, the Gate 500,000 and
Gate 1,000,000 deterministic means differ on average by `0.099` steering,
`0.099` throttle, and `0.052` brake; their p95 differences are `0.373`,
`0.414`, and `0.333`. This is substantial policy drift in the narrow final
alignment corridor.

The impact signature is correspondingly sharp. The largest late speed loss in
each deterministic Gate 1,000,000 episode averages progress `2166.58` and
lateral offset `-22.74`: displayed speed falls from mean `323.8` to `81.4` in
one 100 ms step, a `242.4` loss, while mean throttle is still `0.874` and brake
only `0.082`. Nine episodes share the tightly localized collision-like onset;
one enters the structure already rotating. Combined with the clean visual
review, the evidence supports a repeatable finish-structure impact caused by
the excessively negative line, followed by the flip—not deliberate braking or
an earlier fall.

The exploration distribution did not grow to cause the regression. Mean
latent gSDE standard deviation changed from Gate 500,000 to Gate 1,000,000 by
about `-1.2%` for steering, `-32.3%` for throttle, and `-22.1%` for brake.
However, the noise is state-dependent and persists for four steps, so a sampled
route can still receive a coherent lateral correction that the deterministic
mean lacks. PPO's `target_kl` was unset; this fact motivates examining the
intermediate checkpoint, but does not by itself prove which update caused the
line shift.

The training monitor remains consistent with this explanation. Its final
stochastic rollout windows finished `9/10`, `17/20`, `46/50`, `95/100`, and
`453/500`. Those rows came from evolving pre-update policies and are not a
post-training test of the saved mean. The frozen stochastic diagnostic supplies
that missing final-checkpoint measurement and shows a real, but unreliable,
rescue effect.

Decision: Stage 2 remains stopped. The next safe checkpoint-localization test
is a deterministic evaluation of the preserved nominal Gate 750,000 checkpoint
before proposing any optimizer or reward change. That will show whether the
mean collision line appeared between Gate 500,000 and Gate 750,000 or during
the final segment to Gate 1,000,000.

Evidence:

- stochastic diagnostic summary SHA-256:
  `9935EFE934463EAEB8BC1DE280768B45F341E6B368950C62B25AAC10D3EC51E6`;
- direct-live stochastic action log SHA-256:
  `84C07852D460B328B1087FBE562060FB8AB1340536354489EE5DDCEF2FED7103`;
- reproducible offline diagnostic SHA-256:
  `6AC632FF2635A62CD86060E6ACC4DBF97C03CEF86A2964DBCEF1D6A899D039A3`;
- all ten stochastic input replays are checksum-bound in the summary;
- clean best (`24.810s`) finish video SHA-256:
  `33145B1AE84CBF2ABAA788F97C7C597694A6520F7036BDABC4EE627BF1106E4A`;
- clean closest-to-mean (`33.280s`) finish video SHA-256:
  `99C5B70F7036383EC51DD31E5E1EB5296BA60B3DBB77621EACF144406BC0CA5D`;
- clean worst (`37.440s`) finish video SHA-256:
  `8A74C10A0BE7D3F35E19C1719B8CFBE1823659E58B6D1C403F5792080AF4913F`;
  and
- every delivered finish video is H.264, reports `overlay: false`, and
  reproduced its original live finish and race clock.

The first preservation attempt stopped the two slow finishes at the replay
tool's legacy `32.000s` ceiling, making them look like playback failures. Their
registered live finish times were `33.280s` and `37.440s`; extending capture to
the environment's full `45.000s` horizon made both finish at exactly those
times. The original truncated clips remain disclosed, and the corrected clean
clips are the delivered evidence.

### Frozen intermediate-checkpoint localization diagnostic

Before changing PPO settings or the reward, evaluate the preserved automatic
checkpoint at model timestep `3,756,176`, which is `751,760` Stage 2
interactions beyond the initialization. Its path is
`checkpoints/wr_chase_stage2/ppo_wr_stage2_3756176_steps.zip`, SHA-256
`0F0326F84B442D5F506B2FDD6A646D4B1168F9887BF74152B8CD695E3B4635B7`.

Run exactly `10` deterministic episodes at `6x`, using the same live snapshot,
100 ms action period, reward, stuck cutoff, complete action/SimState audit, and
raw replay preservation as the registered gates. This is a read-only
checkpoint-localization diagnostic: it performs no training, does not alter a
weight, does not replace a gate result, and writes to a separate artifact root.

Interpretation is frozen before the run:

- `>=8/10` finishes means the deterministic route was still reliable at this
  intermediate checkpoint, locating the collapse after `751,760` Stage 2
  interactions.
- `<=2/10` finishes with the same excessively negative final line and impact
  signature means the deterministic regression was already established by
  this checkpoint.
- `3..7/10` finishes means the route was already in an unstable transition at
  this checkpoint.

In every case, compare lateral offset at progress `2000`, `2100`, `2140`, and
`2160` against Gate 500,000 and Gate 1,000,000. Report the observed result once;
do not select a different automatic checkpoint after seeing it.

### Intermediate-checkpoint localization result

The preserved nominal Gate 750,000 checkpoint finished `0/10` deterministic
episodes. This satisfies the frozen `<=2/10` branch and locates the mean-policy
regression at or before `751,760` Stage 2 interactions:

- nine stuck terminations and one `45.100s` timeout;
- no fall or off-track termination;
- upside-down periods in `5/10` episodes and stuck periods in `9/10`;
- best maximum progress `2203.80 / 2206.53`, without triggering a finish;
- mean p95 absolute lateral offset `23.706` and maximum `32.025`;
- mean `38.1` significant steering reversals; and
- oscillation detected in `10/10` episodes.

There are no best, mean, or worst finish times because no episode finished.
The direct-live audit covered all `3,101` actions and matched every terminal
outcome without replay fallback. Both assisted zones again had zero eligible
bonus steps and zero sliding-wheel samples.

The localization is visible in the fixed progress comparison:

| Progress | Gate 500,000 | Gate 750,000 | Gate 1,000,000 |
| ---: | ---: | ---: | ---: |
| `2000` | `-16.50` | `-15.96` | `-22.48` |
| `2100` | `-20.72` | `-19.42` | `-27.39` |
| `2140` | `-20.40` | `-23.03` | `-28.53` |
| `2160` | `-16.92` | `-25.13` | `-25.20` |

Gate 750,000 still matches or slightly improves on Gate 500,000 through
progress `2100`, then fails to make Gate 500,000's corrective line change
between `2140` and `2160`. By progress `2160`, its mean lateral offset is
already only `0.07` units from Gate 1,000,000's collision line and `8.21` units
farther negative than Gate 500,000's.

The same impact signature follows. Gate 750,000's largest late speed loss
averages progress `2166.49` and lateral offset `-24.15`; displayed speed drops
from mean `323.8` to `91.2` in 100 ms. The action remains mixed rather than a
planned stop (mean throttle `0.792`, brake `0.275`), and the visual failures
remain concentrated around the finish structure.

Reconstructed observations reproduce its logged deterministic actions with
maximum absolute error `5.97e-7`. On its own live states at progress `>=2100`,
Gate 500,000 versus Gate 750,000 mean absolute action changes are `0.085`
steering, `0.096` throttle, and `0.144` brake; p95 changes are `0.492`, `0.533`,
and `0.813`. Thus the checkpoint itself contains a materially different final
correction policy; this is not reset noise or evaluator mapping error.

Conclusion: the stable mean route broke during the `250,000` interactions
between the successful Gate 500,000 checkpoint and this automatic checkpoint.
Gate 1,000,000 made the bad approach appear earlier, but it did not originate
there. Stage 2 remains stopped, and the checksum-pinned Gate 500,000 checkpoint
is the last registered reliable base. Any continuation now needs a separate,
pre-registered optimizer-stability experiment rather than resuming the failed
run or silently selecting a favorable later checkpoint.

Evidence:

- intermediate summary SHA-256:
  `2459366EACFA58476CAF8953A204F0C9F2D054AF699DF5636029AAB5C87EF909`;
- direct-live intermediate action log SHA-256:
  `E9F88E10492BD5EE01E9E964E4C105A1754573A0A30460778349FB2708B57089`;
- extended offline diagnostic SHA-256:
  `6AC632FF2635A62CD86060E6ACC4DBF97C03CEF86A2964DBCEF1D6A899D039A3`;
- all ten input replays are checksum-bound in the summary;
- clean closest-progress failure video SHA-256:
  `11DD61AE40C980842CF48EC41F8F1C6A893FB9F053D3E1E99502DEED4A0BDA4B`;
- clean source-evaluation lower-median-progress failure video SHA-256:
  `517BBB4DA98FEF9B0374418186E8CD697D777EE0393718552CA60D375C65D39B`;
- clean lowest-progress failure video SHA-256:
  `B117AF36703758BB81F212B4E992FE86060BE44ADE7BBF165A02EBF9E4F149FF`;
  and
- all three videos are H.264, `overlay: false`, and visibly preserve a
  non-finish. Labels use the registered source-evaluation progress ranking;
  replay inspection was allowed to continue to the full 45-second horizon.

## Realistic expectation

Yosh's public result shows that pure progress reward can discover the drop
release, diagonal jump, and a useful speed-drift. It does not make discovery
likely on any particular compute budget. Those maneuvers require a narrow
sequence of approach speed, line, brake timing, steering, and slide angle.
This project acts at 10 Hz while Yosh described a 20 Hz controller, which makes
fine timing less expressive here. Stage 1 is therefore a genuine discovery
test, not a promised route to the record.
