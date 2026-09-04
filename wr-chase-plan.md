# A01 world-record chase: staged discovery plan

Status: Stage 2b paused at the first trained gate under the frozen safety rule.
The target-zero baseline finished `9/10`, but Gate 50,000 finished `0/10`; all
ten deterministic episodes became stuck at the same final-structure approach.
Accepted slide onset and precursor p95 both remained flat. Gate 100,000 is not
authorized while this safety result is under review.

Date pre-registered: 2026-08-26

Stage 2 date pre-registered: 2026-08-27

Stage 2b date pre-registered: 2026-09-02

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

### Stage 2 bonus-reachability audit before Stage 2b

Before restarting from the reliable Gate 500,000 checkpoint, the frozen Stage
2 eligibility rule was audited against every preserved direct-live evaluation
available at this point: target zero, deterministic Gates 500,000, 750,000,
and 1,000,000, plus the seeded stochastic Gate 1,000,000 diagnostic. This is
`50` episode-passes through each assisted zone, with `1,203` first-turn samples
and `1,286` final-corner samples. All records were terminal-outcome matched,
required complete live `SimState`, and reproduced the registered reward and
progress-high-water calculations. No training was performed for this audit.

The position windows were not the blocker. Every one of the `50` episode-passes
entered both zones, and every in-zone sample made positive new high-water
progress. The individual threshold results were:

| Condition | First turn | Final corner |
| --- | ---: | ---: |
| At least three grounded wheels | `1,201/1,203` (`99.83%`) | `1,285/1,286` (`99.92%`) |
| Displayed speed `>=350` | `1,093/1,203` (`90.86%`) | `1,173/1,286` (`91.21%`) |
| Absolute yaw rate `>=0.25` | `1,119/1,203` (`93.02%`) | `1,093/1,286` (`84.99%`) |
| At least one sliding wheel | `2/1,203` (`0.17%`) | `3/1,286` (`0.23%`) |
| Absolute slip angle `>=1 degree` | `2/1,203` (`0.17%`) | `2/1,286` (`0.16%`) |
| All reward conditions together | `0/1,203` | `0/1,286` |

The important result is the joint behavior, not the isolated maxima. Among all
grounded samples at `>=350` speed, neither zone contained a single sliding
wheel or a single sample at `>=1 degree` slip. Maximum speed-qualified slip was
only `0.395 degrees` in the first turn and `0.426 degrees` in the final corner.
All four deterministic datasets combined contained zero sliding-wheel and zero
`>=1 degree` slip samples in either assisted zone.

Stochastic exploration produced the only two four-of-five near misses, both in
the same slow `33.280s` finish and both missing speed only:

- first turn: speed `305`, three grounded wheels, one sliding wheel,
  `1.562 degrees` slip, and `3.339` yaw rate; and
- final corner: speed `333`, three grounded wheels, three sliding wheels,
  `3.118 degrees` slip, and `3.069` yaw rate.

These samples show that the conjunction is not mathematically impossible, but
the recognized behavior appeared only after the policy had slowed below the
intended fast-driving regime. Across both zones, `2,031/2,489` samples
(`81.60%`) already had speed, ground contact, yaw, and positive high-water
progress but received no partial credit because sliding and slip were absent.

The physics fields themselves are live. Across the complete `13,926`-sample
evaluation corpus, the logs contain `1,336` sliding-wheel samples, `4,421`
samples at `>=1 degree` slip, and `20` samples satisfying all five dynamics
conditions outside the assisted zones. The zero in-zone eligibility count is
therefore not explained by a dead telemetry field, a parser error, or a zone
that the car never entered.

Conclusion: the exact sparsity risk pre-registered for Stage 2 materialized.
The binary bonus requires the policy to have already produced a high-speed
slide before it receives any shaping signal toward one. This is a reachability
failure within the measured policy distribution, separate from the
Gate-500,000-to-750,000 optimizer regression. A conservative KL guard and
shorter deterministic checkpoint intervals remain necessary for Stage 2b, but
they are not sufficient: restarting with the unchanged gate would again test a
bonus that supplied no observed learning signal.

Stage 2b must therefore be a separately pre-registered iteration that revises
eligibility before training. The recommended direction is to retain the
original `>=1` sliding-wheel plus `>=1 degree` criteria as the full-drift target
and measurement, while adding a bounded, continuous in-zone bridge that gives
partial credit for increasing live slip/yaw at competitive speed before the
wheel-sliding flag appears. The exact formula and safeguards must be frozen and
approved before training. Simply lowering the speed or slip thresholds is not
recommended: the only threshold-passing slide-like samples were slow and could
turn loss of control into a rewarded behavior.

The original `1 degree` and `0.25 rad/s` thresholds remain an operational
continuity metric, not physical ground truth. The owner cannot perform a
speedslide, Yosh's public material does not publish a universal telemetry
angle, and the rejected WR resimulation cannot supply trustworthy dynamics.
That does not block a test of whether exploration produces slide onset: the
simulator's direct per-wheel `is_sliding` state can define that narrower event,
while matched-control speed and time determine whether it helped. Stage 2b's
graduated reward still requires a separate frozen design after that diagnostic.

Scope limitation: full `SimState` was not retained for every stochastic
training interaction. This audit proves zero eligibility across the `50`
preserved evaluation episodes and `2,489` in-zone samples; it does not claim
that no transient eligible step could have occurred anywhere in training.

Evidence:

- reproducible analysis:
  `runs/wr_chase_stage2_bonus_reachability/analysis.json`;
- analysis SHA-256:
  `4118BF49151461A4AE575FBEB4EB82E49B47F314029A698D54722384617FB924`;
  and
- audit implementation and tests: commit `1438dcb`.

### External WR reference attempt: rejected by the frozen fidelity gate

An official-record reference was attempted as a possible dynamics-calibration
source. The source was Axell's `23.770s` A01 replay, record
`12847157`, downloaded from the TMNF-X A01 leaderboard. The source file is
`39,854` bytes, identifies ghost login `fwo_axell`, contains `815` control
entries and `268` embedded 100 ms position records, and has SHA-256
`1A253411D641D26CFC57F428DCD0F5F6EA8557FBEBC8AD30B363623EFA8F644F`.
The replay reports `TmForeverCompPatch.2.11.26(1.6)`. It is retained locally
for calibration only; download availability is not treated as permission to
redistribute the replay or derived footage.

Before the first playback, acceptance was frozen at both of these conditions:

- local finish time within `100 ms` of the source's `23.770s`; and
- no more than `1.0` world unit of position error at any exact 100 ms source
  sample, both overall and inside each assisted zone.

The extracted controls were replayed once at 1x through the same live
`SimState` pipeline used for policy evaluation. The trajectory began at zero
position error but diverged early, reached only `276.952` reference-progress
units, and never entered either assisted zone. It timed out unfinished at
`45.000s`. Across the `238` source timestamps covered by the source ghost, the
local position error had a `420.172`-unit median, `751.886`-unit p95, and
`762.970`-unit maximum. The attempt therefore failed both predeclared fidelity
conditions and is rejected as a physical WR reference.

This is evidence about replay portability, not about speedslide thresholds.
No slip, wheel-state, or yaw value from the divergent trajectory may be used to
validate or revise Stage 2. The original replay's embedded ghost positions and
controls may still support timing, zone, and approximate-speed facts because
those do not depend on the rejected local trajectory: its approximately 10 ms
brake taps occur near reference progress `781` and `1120`, at position-derived
speeds of roughly `400` and `454`, with near-full steering. They do not reveal
a trustworthy slip angle or wheel state.

The owner has already stated that they cannot perform the maneuver, so an
owner-driven reference is not a prerequisite. The replacement below uses live
engine slide state to recognize onset and paired local performance to recognize
possible usefulness. It deliberately does not claim to identify a perfect or
WR-quality speedslide. Stage 2b's reward remains untouched until the
exploration ablation is reported and the eligibility redesign is separately
pre-registered.

Rejected-reference evidence:

- TMNF-X leaderboard:
  `https://tmnf.exchange/trackreplayshow/2233`;
- source replay:
  `https://tmnf.exchange/recordgbx/12847157`;
- analysis:
  `artifacts/analysis/wr_reference_speedslide/analysis.json`;
- analysis SHA-256:
  `37FFAB61205943AC64BF3A47E1E4A21F0E34D4D8537D959ECA9647C2822FCEC7`;
- live telemetry SHA-256:
  `1FFFDBDB8D1AC48C29BB042D789A8C92A7B5BC1E0D91DBBC1A3B09F99099EB5A`;
  and
- capture and rejection tooling: commits `5faa24e` and `6ba3c75`.

### Stage 2b pre-design final-zone exploration ablation

This diagnostic is added before the graduated Stage 2b bonus is finalized. Its
purpose is to isolate whether the current PPO/gSDE policy can produce more
drift-adjacent behavior when exploration is increased only in the known final
corner, without any reward assistance or weight update. It is not a Stage 2b
training gate and cannot be reported as learning.

No human reference is required. The experiment asks the narrower question
"does zone-local exploration produce real slide onset?", not "does it reproduce
an optimal speedslide?" Yosh describes a speed-dependent, precise drift but
does not publish a universal slip-angle or yaw-rate cutoff. A fixed angle
invented from that description would therefore be false precision. The primary
detector instead uses the simulator's direct per-wheel slide state, sustained
duration, a road-speed regime supported by public mechanics guidance and the
original WR ghost, and safety guards. Every sampling, budget, comparison, and
decision rule below is frozen now and must not be adjusted after the ablation
starts.

#### Frozen non-human scoring contract

A sample is **valid for final-zone slide onset** only when all of these hold:

- reference progress is within `1100-1410`, inclusive;
- displayed speed is at least `400`;
- at least three wheels are grounded;
- new high-water progress is positive;
- upright cosine is at least `0.8`;
- absolute heading error is at most `pi/4`; and
- absolute lateral offset is at most `20` reference units.

The `400` speed floor is not presented as a universal Trackmania constant. It
is appropriate for this fixed-policy A01 final-zone diagnostic: all `250/250`
reliable Gate 500,000 final-zone samples are `433-473`, the original WR ghost's
final-zone brake tap is approximately `454`, and public road-speedslide
guidance places the technique in roughly the `400+` regime. It therefore
rejects the earlier slow loss-of-control near misses without making the target
unreachable for the source policy.

A **drift-adjacent slide-onset window**, the ablation's primary event, is at
least two consecutive valid 100 ms samples (`>=200ms`) with at least one wheel
reporting `is_sliding` on every sample and at least two sliding wheels on one or
more samples. This is engine-defined slide onset; it does not depend on a
guessed optimal angle. Absolute slip angle and body-up yaw rate are still
reported for every window but are not primary gates. Yaw cannot stand alone:
`84.99%` of the audited final-zone samples already exceeded `0.25 rad/s` during
ordinary cornering.

An **above-envelope kinematic near miss** is at least two consecutive valid
samples with absolute slip at least `0.50 degrees` but no qualifying slide-onset
window. The `0.50` boundary is only a diagnostic separator above the complete
observed fast-policy maximum of `0.426 degrees`; it is not called a physical
speedslide threshold and cannot pass the ablation's success gate.

The original Stage 1/2 **candidate drift window** and **confirmed telemetry
slide** definitions remain unchanged as secondary continuity metrics. A
**locally useful speedslide candidate** requires a primary slide-onset window,
a safe finish, traversal from progress `1100` through `1410` at least one 100 ms
step faster than its matched-seed control, and displayed speed at the `1410`
exit at least as high as that control. This remains a candidate rather than
proof of WR-quality technique.

Reject a primary or useful window if the window plus one sample on either side
contains fewer than three grounded wheels, upright cosine below `0.8`, absolute
heading error above `pi/4`, absolute lateral offset above `20`, non-positive
high-water progress, any terminal failure, or a one-step displayed-speed loss
of at least `25`. The `25` guard is beyond the worst `19`-unit in-zone one-step
loss in the preserved baseline. Every otherwise qualifying window still needs
direct-live and visual review because the current log has no authoritative
wall-contact label.

#### Technical design

The reliable source policy is the Stage 2 Gate 500,000 checkpoint:

- path: `checkpoints/wr_chase_stage2/gate_00500000_model.zip`;
- SHA-256:
  `8A06E00055886E8E671E988D5D6948C6688B74780EDB32A87C6CC213870C8326`;
- model timestep: `3,506,176`; and
- frozen deterministic result: `9/10` finishes, `24.780s` best,
  `24.816s` mean, and `24.900s` worst.

The checkpoint uses PPO with gSDE, a four-action noise-refresh cadence,
squashed actions, and a global entropy coefficient of `0.0`. Stock
Stable-Baselines3 applies one entropy coefficient to the whole minibatch; it
cannot make that coefficient final-zone-only. Temporarily changing `log_std`
only during rollout would also be invalid for training because PPO would not
reproduce the same distribution when recalculating action probabilities.

The diagnostic therefore appends one binary flag to the policy input:

```text
final_zone_flag = 1 if 1100 <= pre_action_progress <= 1410 else 0
zone_noise_scale = 1 + final_zone_flag
```

A parameter-free feature extractor removes the flag before the actor and value
networks, so both still receive the original 26-value observation and retain
the checkpoint's exact deterministic mean actions and values. The custom gSDE
distribution alone multiplies its state-dependent latent noise by
`zone_noise_scale`. This makes the treatment's action standard deviation
exactly `2x` inside the inclusive final-corner window and leaves it exactly
`1x` everywhere else. The same conditional distribution must be used for
action sampling, action-log-probability calculation, and PPO
`evaluate_actions`; no action-wrapper or post-sampling perturbation is allowed.

Both arms use this same policy implementation:

- control: the appended flag is fixed to zero for the whole lap; and
- treatment: the flag follows the exact pre-action progress condition above.

This keeps the learned policy, gSDE refresh timing, and random draws matched;
only final-zone noise amplitude differs. The original checkpoint remains
immutable. Each arm loads a fresh in-memory copy, never calls `learn`, never
saves a model, and uses the unchanged V4
`signed_progress_efficiency_reward` for logging. The old localized drift bonus
is not evaluated or paid.

#### Frozen execution protocol

- Run `20` control and `20` treatment episodes at 6x with the existing 100 ms
  action period.
- Use the same per-episode seeds in both arms: integers `20,260,830` through
  `20,260,849`, inclusive.
- Retain the native gSDE refresh cadence of once every four actions in both
  arms. The treatment changes amplitude, not refresh frequency.
- Record complete direct-live `SimState` for every action. Input-replay
  telemetry is forbidden for the slide conclusion.
- Verify before and after each arm that checkpoint SHA-256, model timesteps,
  optimizer state, reward identity, and original 26 actor/value inputs remain
  unchanged.
- Report `training_interactions: 0` and `diagnostic_only: true`. There is no
  TensorBoard run and no output checkpoint.
- The first-turn zone is a negative control: its exploration scale must remain
  `1x` in both arms.
- Any apparent candidate requires visual review to reject wall contact,
  landing instability, or airborne rotation masquerading as a useful slide.

The primary comparison is the frozen drift-adjacent slide-onset detector above.
For continuity, the original Stage 2 candidate and confirmed-window detectors
remain unchanged and are also reported. Each arm must additionally report
above-envelope near misses; final-zone sliding-wheel samples; p50, p95, and
maximum absolute slip angle and yaw rate; speed and sustained-window duration;
zone-entry and exit speed; final-zone traversal time; steering/action variance;
finish rate; best, mean, and worst finish; fall, stuck, off-track, and
upside-down outcomes; lateral deviation; and oscillation. First-turn versions
of the slide metrics are reported as the negative control.

The treatment counts as producing more attempts only if all of these are true:

- drift-adjacent slide-onset windows occur in at least `3/20` treatment
  episodes;
- the treatment has such windows in at least two more episodes than control;
  and
- the qualifying windows survive the direct-live and visual collision/airborne
  review.

Finish-rate or lap-time regression is reported as a separate tradeoff and
does not redefine the attempt gate. One isolated near miss is documented but
cannot pass it.

If the gate passes, the exact `2x`, final-zone-only conditional gSDE mechanism
will be included alongside the graduated bonus in the proposed Stage 2b
design. If it does not pass, Stage 2b will use the graduated bonus alone and
the result will be recorded as no demonstrated exploration effect within this
40-episode budget. In that case, PPO's on-policy, smooth-noise exploration
remains a plausible contributor to difficult discovery. Intrinsic-motivation
methods such as Random Network Distillation (RND) and off-policy algorithms
such as Soft Actor-Critic (SAC) are legitimate more
disruptive alternatives, but either would break the PPO-based comparability
across prior versions and requires a deliberate later decision rather than an
automatic substitution here.

Artifacts are isolated under
`runs/wr_chase_stage2b_zone_exploration/`,
`artifacts/analysis/wr_chase_stage2b_zone_exploration/`, and
`artifacts/replays/wr_chase_stage2b_zone_exploration/{control,boosted}/`.
All input replays are retained. Each arm also receives clean, overlay-free
best, closest-to-mean, and worst successful videos; if an arm has no finishes,
the delivered representatives use high, median, and low maximum progress and
are labeled as failures.

The resulting sequence is now fixed:

1. freeze the non-human live-physics detector above;
2. run and report this fixed-policy exploration ablation;
3. use the result to finalize the graduated Stage 2b eligibility design;
4. obtain approval for the complete Stage 2b preregistration; and
5. only then restart training with the approved KL guard and shorter gates.

#### Completed result: zone-local exploration did not produce slide onset

The fixed `20` control versus `20` treatment episode ablation is complete.
Both arms used seeds `20,260,830-20,260,849`, the same four-action gSDE
refresh cadence, the unchanged V4 reward, and fresh in-memory copies of the
Stage 2 Gate 500,000 checkpoint. The treatment applied exactly `2x` gSDE
standard deviation only when pre-action progress was within `1100-1410`;
the control applied none. There were `510` treatment-zone actions and zero
control-zone actions. Neither arm called `learn` or wrote a checkpoint, and
the source checkpoint remained SHA-256
`8A06E00055886E8E671E988D5D6948C6688B74780EDB32A87C6CC213870C8326`.

The frozen live-wheel detector recorded:

- control: `0/20` episodes and `0` windows with raw wheel-slide onset;
- treatment: `0/20` episodes and `0` windows with raw wheel-slide onset;
- treatment-minus-control raw-onset delta: `0` episodes;
- above-envelope near misses: `0` in both arms;
- original Stage 1/2 legacy candidates and confirmed windows: `0` in both
  arms; and
- first-turn negative-control onset: `0` in both arms.

The pre-registered attempt gate required onset in at least `3/20` treatment
episodes and at least two more treatment episodes than control. It therefore
fails before visual review, with decision `no_effect`. This is the raw attempt
result. It is intentionally separate from usefulness: because treatment
produced no qualifying onset at all, there were `0` useful candidates and `0`
genuine useful speedslides. Exit speed, traversal time, finish status, and
video cannot convert a non-onset into a useful slide.

The distribution stayed inside the previously observed ordinary-cornering
envelope. Final-zone absolute slip was `0.323 degrees` p50,
`0.339 degrees` p95, and `0.389 degrees` maximum in control versus
`0.323`, `0.339`, and `0.390 degrees` in treatment. No final-zone sample in
either arm reported a sliding wheel. Across the `18` pairs that entered the
zone within the frozen two-unit comparability tolerance, treatment changed
mean traversal time by only `-5.6ms` and mean exit speed by `0.0` displayed
speed units. These paired performance facts are secondary and do not alter
the zero-onset conclusion.

Safety and lap performance regressed slightly rather than improving:

- control: `17/20` finishes (`85%`), `24.730s` best, `25.412s` mean,
  `34.920s` worst, one fall, two stuck terminations, three upside-down
  episodes, and three stuck-detected episodes;
- treatment: `16/20` finishes (`80%`), `24.740s` best, `26.083s` mean,
  `34.590s` worst, one fall, three stuck terminations, four upside-down
  episodes, and four stuck-detected episodes; and
- oscillation remained universal (`20/20` episodes in both arms), while mean
  steering total variation increased from `36.80` to `38.11`.

Matched seeds do not make live Trackmania trajectories bit-identical. gSDE is
state-dependent, so small simulator-state differences before treatment
changed sampled actions even with the same random seed. Eighteen pairs reached
the zone within the frozen two-unit tolerance; seeds `20,260,830` and
`20,260,839` did not and are excluded from paired usefulness attribution.
They remain in the pre-registered arm-level `20` versus `20` raw-onset result.
Exact pre-zone action equality was an extra audit attempted by the runner, not
a pre-registered requirement; treating live action jitter as descriptive
pairing evidence avoids post-hoc deletion or rerunning of valid randomized
arms.

All `40` original input replays are retained. Clean overlay-free best and
worst control videos and best, closest-to-mean, and worst treatment videos
were reproduced. The closest-to-mean control input replay did not reproduce
its original finish during later TMInterface playback, and two attempted
replacement captures aborted the bridge. The original input replay and live
evaluation record remain hash-pinned; no failed playback is relabeled as a
successful average video. This replay-rendering limitation is separate from
the direct-live scoring result.

Evidence:

- comparison: `runs/wr_chase_stage2b_zone_exploration/analysis.json`, SHA-256
  `84A391EB92CB8B3FF53B969B59A14AEBB79C0BAD5BA07E34233125F396FDFF8E`;
- representative selection:
  `runs/wr_chase_stage2b_zone_exploration/representative_selection.json`,
  SHA-256
  `FE50AEE6931BA8F7EBC12BC2F60AEE1CDD0FD530C32D472985337698BB1536F2`;
- control summary SHA-256:
  `5572F9A54756C38081F77B4EDFE0E1055D4422212D47C37D20DA07B06192A61F`;
- treatment summary SHA-256:
  `DA8D02F2201E302385294298B8059FCE74BA3320B355710ADCE94BF35F55DC01`;
- live-physics scorer and tests: commit `e6ad5f4`;
- matched zero-learning runner: commit `7746204`;
- slow-map startup handling: commit `38c8bcc`;
- live gSDE pairing audit correction: commit `888accb`; and
- replay-capture hardening: commits `6c4f31a` and `ca36a8b`.

Within this `40`-episode fixed-policy budget, final-zone-only `2x` gSDE
exploration did not move the car closer to physical slide onset. Per the
frozen decision rule, it will not be carried into the proposed Stage 2b
design. The next design proposal should pre-register a graduated bonus alone,
while recording PPO/gSDE's smooth on-policy exploration as a
plausible contributor to difficult discovery. RND or SAC remain deliberate
future alternatives, not automatic changes to this PPO-comparable sequence.

### Stage 2b pre-registration: graduated final-corner precursor reward

Stage 2b is a separate reward iteration, not a continuation whose rules can be
changed in response to intermediate results. Training has not started. The
formula, initialization, measurement contract, optimizer guard, gates, and
time box below are frozen for owner review before any implementation or live
training.

The exploration ablation produced zero wheel-slide onset in both `20`-episode
arms. Within that fixed-policy, final-zone, `2x` gSDE experiment, insufficient
randomness is therefore ruled out as the primary barrier. This is not a claim
that no larger or fundamentally different exploration method could ever find
the maneuver. It means more of the same PPO/gSDE noise is not the supported
next lever. The current V4 reward pays progress and time efficiency but gives
no preference at all to a `0.4-degree` slide precursor over a `0.3-degree`
ordinary cornering state. A shaped, graduated approach signal is now the
primary remaining lever rather than secondary assistance layered on an
exploration change.

#### Exact frozen reward addition

Stage 2b retains `signed_progress_efficiency_reward` exactly, including signed
centerline progress clipped to `[-20, 20]` then divided by `10`, the `-0.10`
per-step time cost, `+50` finish bonus, and `-250` verified-failure penalty.
There is no exploration multiplier, first-turn bonus, finish-bonus change, or
other reward term.

For every 100 ms transition, let:

```text
new_progress_fraction = (
    clip(new_high_water_progress_delta, 0, 20) / 20
)

abs_slip = abs(slip_angle_degrees)
slip_progress = clip(abs_slip / 1.0, 0, 1)
wheel_slide_progress = clip(sliding_wheel_count / 2, 0, 1)

precursor_score = (
    slip_progress + wheel_slide_progress
) / 2

speed_loss = previous_display_speed - displayed_speed

safe_final_zone_step = (
    1100 <= current_progress <= 1410
    and displayed_speed >= 400
    and ground_contact_count >= 3
    and new_high_water_progress_delta > 0
    and upright_cosine >= 0.8
    and abs(heading_error) <= pi / 4
    and abs(lateral_offset) <= 20
    and speed_loss < 25
    and not terminated
    and not truncated
)

graduated_final_corner_bonus = (
    0.50 * new_progress_fraction * precursor_score
    if safe_final_zone_step
    else 0.0
)

stage2b_reward = (
    signed_progress_efficiency_reward
    + graduated_final_corner_bonus
)
```

`previous_display_speed` is the live displayed speed from the immediately
preceding policy transition. The first transition without a preceding speed
cannot earn this bonus. All position, orientation, speed, contact, slide, and
slip fields must come from the same direct-live `SimState` transition used to
calculate the reward. Missing, incomplete, nonfinite, or non-four-wheel state
is a hard instrumentation failure. Input-replay telemetry is never a reward
fallback.

The operational precursor target is `1.0 degree` absolute velocity slip and
two sliding wheels. The `1.0-degree` value preserves continuity with Stage 2;
it is not asserted to be a universal or physically optimal speedslide angle.
The simulator's wheel flags, not this angle, remain the primary onset detector.
Both component scores saturate at `1.0`: angles above `1.0 degree` and more
than two sliding wheels earn no additional credit. This removes any direct
incentive to seek progressively more extreme, crash-prone slip.

Equal weighting is deliberate and minimizes new knobs. The continuous slip
half supplies reward before any binary wheel-slide flag appears; the wheel
half then distinguishes actual engine-reported slide state from kinematic
yaw. The current final-zone median slip of approximately `0.323 degrees` with
zero sliding wheels would have a precursor score of approximately `0.162`, so
the source policy begins inside a nonzero reward slope instead of behind
another binary gate.

The coefficient and progress scaling retain Stage 2's original maximum of
`0.50` per step. Across the `310`-unit final zone, the high-water rule bounds
the approximate one-pass maximum to `7.75`. A car that maintains the current
`0.323-degree`, zero-wheel state would earn only about `1.25` across a complete
first traversal; a `1.0-degree`, zero-wheel precursor is capped near `3.875`;
and the full two-wheel target is capped near `7.75`. These are trajectory-level
bounds, subject to the per-step `20`-unit clamp. Reversing or repeatedly
crossing the zone cannot repay the same progress. Slowing creates additional
V4 time cost, and a verified failure still costs `250`, so the graduated term
cannot make a near-finish crash net positive.

The safety gate mirrors the frozen non-human scoring envelope and adds the
same `25`-speed-unit sudden-loss guard used to reject collisions. It is a
reward eligibility guard, not a declaration that every rewarded step is a
useful speedslide. No yaw-rate threshold is used in the reward because ordinary
final-corner driving already satisfied the old yaw threshold on `84.99%` of
audited samples. Yaw remains logged as a secondary diagnostic.

#### Pre-registered Stage 2b hypothesis

> A bounded, continuous final-corner reward for increasing high-speed slip angle and sliding-wheel count, added to V4 without an exploration boost, is expected to move the reliable Gate 500,000 policy beyond its approximately 0.4-degree ordinary-cornering envelope and produce frozen-contract wheel-slide onset in at least 3 of 10 deterministic episodes within 500,000 interactions. Early onset attempts are expected to be imprecise and may worsen lap time or finish rate before becoming useful. A speedslide will be called genuinely useful only if it separately passes the frozen exit-speed, traversal-time, safe-finish, and visual checks. If precursor score rises but onset remains below 3 of 10, the reward shaped approach behavior without demonstrating repeatable slide induction; if precursor score does not rise, the graduated formula is ineffective within this budget.

This hypothesis tests induction first and usefulness second. A higher bonus,
slip angle, yaw rate, or sliding-wheel count is not itself a lap-time success.
The outcome is reported honestly even if the learned attempts are slower,
unsafe, or visually unlike a real speedslide.

#### Initialization and optimizer guard

Stage 2b initializes from the last reliable optimizer-bearing Stage 2 model:

- checkpoint: `checkpoints/wr_chase_stage2/gate_00500000_model.zip`;
- SHA-256:
  `8A06E00055886E8E671E988D5D6948C6688B74780EDB32A87C6CC213870C8326`;
- model timestep: `3,506,176`; and
- frozen deterministic result: `9/10` finishes, `24.780s` best,
  `24.816s` mean, and `24.900s` worst.

The saved PPO policy, value function, optimizer moments, learning rate
`0.0003`, `2,048`-step rollouts, batch size `64`, `10` PPO epochs, gamma
`0.99`, GAE lambda `0.95`, clip range `0.2`, entropy coefficient `0.0`, value
coefficient `0.5`, gradient-norm cap `0.5`, gSDE enabled, and four-action gSDE
refresh cadence remain unchanged. The only optimizer safeguard added is
Stable-Baselines3 `target_kl = 0.01`. With the pinned local SB3 `2.9.0`
implementation, remaining PPO epochs for an update stop when minibatch
approximate KL exceeds `1.5 * target_kl`, or `0.015`. The prior Stage 2 run had
no KL target and logged median update KL near `0.054`; this is intentionally a
conservative guard against another rapid mean-policy shift.

Every update must log approximate KL, completed PPO epochs, clip fraction,
policy loss, value loss, entropy, and action-distribution statistics. A KL
early-stop is not a failed gate and does not change the `50,000`-interaction
evaluation cadence; it only limits that update's optimization epochs.

#### Short gates, frozen measurement, and stop rules

- Before learning, reconstruct the exact Stage 2b reward and precursor score
  over the preserved direct-live Gate 500,000 deterministic evaluation. If any
  required field is absent, run one fresh 10-episode deterministic target-zero
  evaluation instead. This is an instrumentation baseline, not checkpoint
  selection.
- Train at 100x with the existing 100 ms action period in nominal `50,000`
  interaction segments. PPO rollout overshoot is retained and exact model
  timesteps are reported.
- Save a complete optimizer-bearing checkpoint at every segment boundary.
- At every boundary, run `10` deterministic episodes at 6x through the same
  direct-live pipeline. Preserve all action logs and input replays. Produce
  clean overlay-free best, closest-to-mean, and worst successful videos; if a
  gate has no finishes, use high-, median-, and low-progress failures and label
  them as failures.
- The initial time box is `500,000` additional interactions, or ten nominal
  segments. There is no automatic extension and no coefficient change inside
  this iteration.
- Raw induction uses the frozen primary detector: within progress `1100-1410`
  at speed `>=400`, at least two consecutive valid 100 ms samples must have at
  least one engine-reported sliding wheel on every sample and at least two
  sliding wheels on one sample. All frozen grounded, progress, upright,
  heading, lateral, sudden-speed-loss, terminal, direct-live, and visual
  rejection rules remain in force.
- Pause and report immediately when accepted onset occurs in at least `3/10`
  deterministic episodes. Do not continue training while visual or usefulness
  review is pending.
- On induction, run the frozen matched-control usefulness protocol against the
  unchanged Gate 500,000 base: `20` base and `20` candidate deterministic
  episodes at 6x, paired on seeds `20,260,830-20,260,849`, with the same 100 ms
  cadence and frozen two-unit zone-entry comparability tolerance. A candidate
  must finish safely, traverse progress `1100-1410` at least `100ms` faster
  than its comparable control, exit at progress `1410` with displayed speed
  at least as high, and survive visual rejection of wall contact, airborne
  rotation, or loss of control. Report raw onset count and genuine-useful
  count separately.
- Pause for safety review after any single gate with `<=7/10` finishes, any
  `0/10` gate, or a repeated fixed-location collision/stuck failure in at least
  `3/10` episodes. This is deliberately stricter than Stage 2's old two-gate
  rule. Lap-time or reliability regression is never hidden by improved
  precursor metrics.
- If no gate reaches repeated induction by `500,000`, stop and report the
  formula as not demonstrating repeatable slide induction within the frozen
  budget. A rising precursor score is a secondary finding, not permission to
  move the success threshold or extend training automatically.

Every gate reports bonus total; precursor-score p50, p95, and maximum; slip
and yaw distributions; sliding-wheel samples; above-envelope near misses;
accepted onset windows; final-zone entry/exit speed and traversal time; finish
rate; best, mean, and worst lap; failure causes; lateral deviation;
oscillation; upside-down and stuck duration; and comparison against the
`24.500s` owner PB and `23.770s` benchmark. The first-turn zone is retained as
an unassisted negative control in measurement only.

TensorBoard run name:
`wr_chase_stage2b_graduated_final_corner`. Artifacts are isolated under
`runs/wr_chase_stage2b/`, `checkpoints/wr_chase_stage2b/`,
`artifacts/replays/wr_chase_stage2b/`,
`artifacts/analysis/wr_chase_stage2b/`, and
`artifacts/videos/wr_chase_stage2b/`. Stage 1, Stage 2, and the exploration
ablation remain immutable.

At preregistration time, implementation and training remained blocked until
the owner approved this exact Stage 2b contract. After approval, implementation is one isolated reward
function plus the transition fields and gate runner needed to enforce this
preregistration; no reward coefficient may be changed mid-run.

#### Stage 2b target-zero instrumentation baseline

The owner approved the frozen Stage 2b contract on 2026-09-02. Before any
learning, the checksum-pinned Stage 2 Gate 500,000 checkpoint was evaluated for
10 fresh deterministic episodes through the Stage 2b live pipeline:

- `9/10` finishes, with `24.790s` best, `24.838s` mean, and `24.980s` worst;
- zero raw or accepted slide-onset episodes and zero final-zone
  sliding-wheel samples;
- precursor score `0.1623` p50, `0.1687` p95, and `0.1690` maximum over the
  frozen valid final-zone envelope;
- `10.1652` total graduated bonus across all ten episodes, proving the shaped
  precursor credit is reachable before wheel-slide onset;
- steering oscillation in `10/10` episodes, no inversion, and one stuck
  episode; and
- exact reconstruction of the logged Stage 2b reward, with maximum absolute
  error below `1e-6`.

The baseline decision is `continue`. Its zero accepted-onset count and p95
precursor score are the frozen reference for the first 50,000-interaction gate.
No learning or checkpoint selection occurred during this baseline.

The first attempted baseline exposed a tooling-only artifact-path defect: the
shared evaluator still contained one literal Stage 2 replay directory. That
run was not analyzed or used as evidence. Its run and all ten replay exports
were preserved under `wr_chase_stage2b_path_mismatch_20260902` archives, the
literal was replaced with the experiment slug, and the complete baseline was
rerun once under the correct isolated roots.

Evidence:

- implementation commits `29dbd97`, `3560a97`, and `f3a1509`;
- live-baseline path correction commit `2704189`;
- evaluation summary SHA-256:
  `AA07F4F545710BFD7B41ADE4E96039FED91396A491AAF82C8D42BBE905E9556F`;
- direct-live action log SHA-256:
  `D86C871CF696E2BFFDBDBC00F03194B54107967BE13E6EEF006C222319C4D451`;
  and
- analysis summary SHA-256:
  `0F98307D67492A6B76C1D612C23C717F8ACB5A7A0C8DDEF48BE6EEEE094C9C67`.

The evaluation-summary checksum above identifies the final file after the
Stage 2b wrapper binds its audit metadata, correcting the earlier recorded
checksum of the generic evaluator's intermediate output.

#### Stage 2b Gate 50,000: immediate safety pause

The first accepted training gate ran for `51,200` actual interactions, the
minimum 25 complete 2,048-step rollouts that pass the nominal 50,000 boundary.
It loaded the exact registered Stage 2 Gate 500,000 checkpoint and saved a new
optimizer-bearing checkpoint at model timestep `3,557,376`.

The conservative optimizer guard was active rather than nominal. All `25/25`
rollout updates tripped the KL early-stop during the first scheduled PPO epoch;
no update completed a full epoch, versus ten scheduled epochs without the
guard. The triggering minibatch KL values ranged from `0.01562` to `0.04337`,
while SB3's logged per-update final-epoch mean approximate KL ranged from
`0.00434` to `0.01145`. Every update has a matching embedded audit row and
TensorBoard scalar row, including the final update.

The frozen deterministic evaluation nevertheless failed the safety gate:

- `0/10` finishes, so there are no best, mean, or worst finish times;
- all ten episodes terminated as stuck, with no fall or off-track termination;
- all ten failures share the registered fixed-location signature
  `stuck@progress_2150_2175`, reaching maximum progress `2161.92-2169.23` and
  terminal lateral offset `-19.08` to `-27.09`;
- accepted and raw slide onset stayed flat at `0/10`, with zero final-corner
  sliding-wheel samples and zero above-envelope near misses;
- precursor score was `0.1606` p50, `0.1681` p95, and `0.1685` maximum, versus
  baseline `0.1623`, `0.1687`, and `0.1690`;
- total graduated bonus was `10.1598`, versus baseline `10.1652`;
- final-zone traversal remained `2.500s` in every episode, with sampled entry
  speed `434` and exit speed `475-476`; and
- oscillation remained present in `10/10`, mean p95 absolute lateral offset
  worsened to `22.50`, one episode inverted, and all ten contained a detected
  stuck period totaling `21.0s`.

Thus the reward did not move the measured precursor distribution before the
mean policy lost its reliable final alignment. The result satisfies two
independent frozen pause conditions: `0/10` finishes and a repeated
fixed-location stuck failure in `10/10`. Stage 2b stops at Gate 50,000; Gate
100,000 must not run without a separately approved response to this result.

One earlier Gate 50,000 attempt was rejected before evaluation because its
final PPO update remained buffered instead of being written to TensorBoard.
That model was not selected or evaluated. Its model, monitor, training summary,
TensorBoard log, and manifest snapshot are preserved under
`wr_chase_stage2b_invalid_missing_final_audit_20260902` archives. The accepted
rerun added an explicit final logger flush and an embedded per-update audit,
then restarted from the original registered base.

The high-, median-, and low-progress input replays are preserved as episodes
1, 7, and 4 respectively. Clean MP4 rendering was attempted only after the
safety result. A process inventory found two leftover TMNF instances during
failed capture retries; whether they caused the earlier socket aborts remains
unresolved, because capture also failed after restarting with a single instance.
Matching the capture duration to the replay end did not resolve the abort,
either. A proposed bridge-disconnect capture mode then recorded the user's
foreground app while TMNF was tabbed out. Sampled frames verified that capture
problem, but did not establish why the bridge disconnected. That accidental
foreground footage and sampled frames were removed rather than retained. The failed capture mode remains
visible in commits `ec11e53` and `682dfef`, but is not active. Clean MP4s remain
pending resolution of the capture failure and a short interval in which TMNF
can stay foreground; this does not affect the direct-live result or replay evidence.

Evidence:

- complete per-update audit fix: commit `431c991`;
- Gate 50,000 checkpoint SHA-256:
  `1FD54344AA00DD4283FEF3F64150429806927F625CC93055D8D45F47EEE17425`;
- training summary SHA-256:
  `1E774950E9D029A20C9C832B71CDFD1F9C8C0C8BB5641BD34C1EDADE7B1F3F4E`;
- deterministic evaluation summary SHA-256:
  `B3CA91BFC1595A2387BFFA1956B514500E5B33251F76A637B8F96CEA66A0894C`;
- direct-live action log SHA-256:
  `54D10399FB290325DBADA58A5CD3C3FA0110F355C533600396C033103E3AF009`;
  and
- analysis summary SHA-256:
  `575233C8ABB78978B49159B7DD002338510BCCE626175F1338BCC53056D473C4`.

#### Stage 2b post-pause alignment diagnosis

The 2026-09-03 read-only diagnosis is recorded in
[the alignment and KL report](docs/stage2b-alignment-diagnosis.md).
Saved direct-live trajectories show a changed launch direction for the long
final jump: approximately 0.59 degrees of horizontal velocity heading and
29% greater lateral velocity at takeoff, followed by a worse landing line and
an abrupt near-stop at the final structure. Bonus-zone speed and bonus yield
remain essentially unchanged. The bonus is not proven to have caused the
line change; the policies already differ before entering its zone.

An offline test confirms the KL guard matches stock PPO's early-stop behavior.
It reduced optimization to 159 minibatch steps, but does not roll back earlier
steps or constrain cumulative distance from the original reliable policy.
Saved policy probes demonstrate cumulative drift; they cannot establish that
all 25 updates moved in a consistently harmful direction. Stochastic training
still finished 184/195 episodes, unlike the final deterministic 0/10 result.
No new training, zone change, bonus retuning, or KL change is authorized by this
diagnosis. The original safety pause and registered reward remain unchanged.

#### Stage 2b finer-checkpoint audit and recurring final-alignment fragility

The 2026-09-04 [sub-checkpoint and anchoring review](docs/stage2b-subcheckpoint-and-anchor-review.md)
found no historical 10k, 20k, 30k, or 40k policies: the accepted run saved every
50k interactions. Its surviving periodic snapshot contains 24 completed updates
(49,152 interactions learned through), and already shows pre-zone action drift
on identical baseline states. The final 25th update did not originate that
action difference, but gradual versus sudden onset earlier in the run is
unresolved. These offline probes are not an intermediate driven trajectory.
Scalar logs and rejected-attempt checkpoints cannot fill the missing timeline.

This is the **third distinct recurrence of fragility in the same final-alignment
section** in the requested sequence: V3's original collision/low-hoop issue,
Stage 2's optimizer-associated collapse, and now Stage 2b's changed launch line.
Treat this as recurring structural fragility in that part of the policy, not
three unrelated incidents or proof of one shared cause. Preserve the important
V3 qualification: its original finish deficit was confounded by the fall
detector, and the unchanged model finished 20/20 after correction. The recurrent
precision concern must not be rewritten as three identical verified collapses.

A frozen-Gate-500k action-matching auxiliary loss is technically feasible as a
candidate mitigation, but protecting only the late airborne approach would not
address the measured launch-direction error. The review discusses protected
grounded setup plus final correction, fixed rehearsal states, and the risk of
blocking useful slide-exit adaptations. This is a feasibility assessment only:
no anchor coefficient or boundary has been selected, no loss was implemented,
and training remains paused under the existing safety gate.

#### New direction: continuous pure discovery until owner stop

On 2026-09-04 the owner redirected the next experiment away from Stage 2b
bonus/anchoring work to a substantially longer uninterrupted run with unchanged
V4 reward. The [overnight preflight](docs/overnight-pure-discovery-preflight.md)
records the proposed separate run, suggested reliable Stage 1 Gate 2 base,
250k checkpoint spacing without evaluation/approval pauses, and a standalone
local process requiring no overnight Codex monitoring or API usage.

Power-setting read-back now confirms automatic sleep/hibernate, hybrid sleep,
display-off, and disk-idle are disabled on AC/DC; prior changed values are
preserved for restoration. Disk headroom is approximately 478 GiB. Historical
full-PPO pure-discovery throughput is about 379 interactions/s, supporting a
planning estimate of 8.6-10.9 million additional interactions over eight hours.
This is not a fresh benchmark or a claim of proven overnight uptime: a live
100x warm-up remains required. The owner subsequently approved running until
they request stop and confirmed readiness to launch. The separate continuous
runner retains original V4/Stage 1 PPO settings, checks current-host power and
the live connection before learning, saves after updates crossing nominal 250k
boundaries, and supports a local clean stop independent of Codex. The preflight
document freezes the carried-forward hypothesis and implementation details.
Actual launch results must be recorded after observation; this experiment must
not overwrite or resume the failed Stage 2b evidence.

## Realistic expectation

Yosh's public result shows that pure progress reward can discover the drop
release, diagonal jump, and a useful speed-drift. It does not make discovery
likely on any particular compute budget. Those maneuvers require a narrow
sequence of approach speed, line, brake timing, steering, and slide angle.
This project acts at 10 Hz while Yosh described a 20 Hz controller, which makes
fine timing less expressive here. Stage 1 is therefore a genuine discovery
test, not a promised route to the record.
