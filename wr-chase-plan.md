# A01 world-record chase: staged discovery plan

Status: Stage 2 pre-registered; no Stage 2 training result observed

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

## Realistic expectation

Yosh's public result shows that pure progress reward can discover the drop
release, diagonal jump, and a useful speed-drift. It does not make discovery
likely on any particular compute budget. Those maneuvers require a narrow
sequence of approach speed, line, brake timing, steering, and slide angle.
This project acts at 10 Hz while Yosh described a 20 Hz controller, which makes
fine timing less expressive here. Stage 1 is therefore a genuine discovery
test, not a promised route to the record.
