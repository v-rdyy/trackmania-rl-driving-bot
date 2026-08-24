# Reward v4: Signed progress with time and terminal outcomes

Status: Complete; trained, quantitatively evaluated, and visually reviewed

Date pre-registered: 2026-08-24

Approved by the project owner on 2026-08-24.

## Pre-registered hypothesis

A signed centerline-progress reward with an explicit per-step time cost, finish
bonus, and failure penalty is expected to preserve V3's reliability because the
underlying centerline-progress signal is unchanged, while directly incentivizing
speed through the time cost and eliminating the near-finish collision problem
because the verified-failure penalty outweighs near-complete progress.

It is an open question whether replacing V3's one-sided 10-unit clamp with a
signed 20-unit clamp will also reduce the observed steering oscillation. V4 does
not directly penalize steering or lateral deviation, so reduced oscillation is
an evaluation question, not a claimed or explicitly engineered outcome.

This statement was approved before implementation or training and must not be
reworded after results are observed.

## Fixed reward contract

At each 100 ms control step:

```text
progress_delta = current_progress - previous_progress
signed_progress = clip(progress_delta, -20.0, 20.0) / 10.0
reward = signed_progress - 0.10

if race_finished:
    reward += 50.0
else if verified_failure:
    reward -= 250.0
```

`verified_failure` means an environment truncation caused by one of the existing
safety outcomes: confirmed fall, stuck, off-track, or the 45-second timeout. A
fall is confirmed only when the car is more than 10 vertical units below the
reference path and the frozen 2.0-second no-progress/no-motion window also
confirms less than 1.0 unit of centerline progress and less than 2.0 units of
world motion. Merely taking A01's low final-jump trajectory is not a verified
failure while the car continues moving or progressing.

The time cost applies on every environment transition, including the terminal
transition. A finish and verified failure are mutually exclusive. There is no
direct speed, steering, lateral-offset, heading, inversion, checkpoint-specific,
or control-smoothness term. The reward remains an isolated callable injected
into `TrackmaniaEnv`.

## Failure-penalty scale check

V3's full route contains about 2,206 centerline-progress units, worth about
`+220.6` under the V4 normalization before time and terminal terms. A typical
near-finish collision after roughly 27-28 seconds would therefore score about:

```text
+220.6 progress - 28.0 time - 250.0 failure = -57.4
```

The approved `-250` penalty deliberately makes a near-complete failed episode
net negative. The rejected `-50` candidate would have left the same episode near
`+142.6`, contradicting the purpose of terminal failure shaping. A representative
finish at the same pace instead scores about `+242.6` after the `+50` finish
bonus.

## Fixed environment thresholds

- Control period: `100ms`.
- Maximum episode time: `45.0s`.
- Horizontal off-track threshold: `50.0` units from the reference path.
- Vertical fall candidate: more than `10.0` units below the reference path.
- Fall/stuck confirmation window: `2.0s`.
- Confirmation progress gain: `< 1.0` unit within that window.
- Confirmation world motion: `< 2.0` units within that window.
- Both confirmation conditions must hold before truncating a moving low
  trajectory.
- Corrected TMNF pedal semantics remain mandatory.

No threshold or reward coefficient may be changed during the formal run.

## Initialization and training protocol

- Initialize the full PPO policy and optimizer state from
  `checkpoints/reward_v3/final_model.zip`.
- Required V3 checkpoint SHA-256:
  `C9791B6ECF3E83146299376F2180D3132250061C8B33DAF294CF295F1996FE38`.
- Train for `1,000,000` additional environment timesteps.
- Seed: retain the checkpoint's seed-42 experiment lineage.
- PPO/gSDE hyperparameters: unchanged from V3.
- Simulation speed: `100x`.
- Checkpoint interval: `50,000` additional V4 timesteps.
- TensorBoard run name: `reward_v4_signed_progress_efficiency`.
- Host sleep and hibernation must be disabled immediately before training.
- Start the configured `TmForever/default` ModLoader profile and load A01 through
  the automated launcher rather than relying on an existing bridge session.

Loading V3 rather than training a fresh model is deliberate: V4 tests whether
the approved reward correction can refine an already reliable policy without
discarding the learned route-following behavior.

## Evaluation protocol

Evaluate the final V4 checkpoint for 20 deterministic episodes at 6x, using the
same corrected detector and fixed precision protocol as V3. Preserve the action
log and all 20 TMInterface input replays. Report:

- finish rate and every terminal cause;
- best, mean, and worst lap time against both V3's corrected result and the real
  `24.5s` human PB;
- episode reward and length curve shape;
- centerline progress, lateral-deviation distribution, and maximum deviation;
- fixed-threshold steering oscillation, inversion, and stuck-duration metrics;
- visible final-hoop behavior and any collision, recovery, or new exploit;
- finite/range action checks and checkpoint, summary, log, and replay hashes.

Specifically compare oscillation detection with V3's corrected 20/20 result.
If oscillation persists, document it as evidence for considering a direct
smoothness or lateral term in V5; do not add such a term to V4 mid-run.

## Actual outcome

### Training

V4 loaded the pinned V3 checkpoint and completed `1,001,472` additional PPO
interactions on its first attempt in `2,441.23s` (`40m 41s`) at 100x. No
interactions were discarded or replayed. All `1,001,472` audited actions were
finite, in range, and passed the exact affine mapping check without hidden
clipping.

The 3,806 stochastic training episodes contained 3,662 finishes (`96.22%`), 103
stuck truncations, 24 confirmed falls, 12 timeouts, and 5 off-track truncations.
The first and last 100 training episodes both finished 98 times, while mean time
among their successful laps improved from `28.282s` to `25.840s`. Mean reward
over those windows rose from `235.946` to `238.251`. The TensorBoard rolling
reward started at `192.809`, ended at `238.251`, and peaked at `244.870`.

Training pace and reliability fluctuated rather than improving monotonically.
The TensorBoard mean episode length ended above its first value because
occasional exploratory failures ran longer even while successful laps became
faster. The final checkpoint remained the pre-registered selection; no earlier
checkpoint was cherry-picked.

- final model: `checkpoints/reward_v4/final_model.zip`;
- final model SHA-256:
  `6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2`;
- training summary SHA-256:
  `8EE823F93F554402896D666D61868B1C29C75B33AF95CDFF0EE8A9E13F9C963A`.

### Deterministic evaluation

The final V4 checkpoint finished all 20 deterministic 6x episodes with zero
falls, stuck periods, timeouts, off-track truncations, or inversions. Every
episode used 249 control steps. TMNF terminal race-clock times, which are the
correct basis for comparison with the displayed human PB, were:

- best: `24.900s`, `0.400s` slower than the owner's `24.5s` PB;
- mean: `24.929s`;
- worst: `24.950s`.

The environment-controlled elapsed values were `24.800s` best, `24.829s` mean,
and `24.850s` worst. They subtract the 100 ms captured reset state and remain
useful for the exact V3/V4 control-loop comparison, but they are not used for the
human-PB claim. The first generated V4 summary used those elapsed values for the
PB comparison; replay terminal clocks exposed the mismatch. That summary was
preserved rather than overwritten silently, and the corrected summary was
rebuilt from the same completed log and 20 replays without rerunning an episode.

- corrected evaluation summary SHA-256:
  `36FBA22B9FD71C98EA9DFAC39859C215C31E27ED445FAAC5AE02A8DA4BEBB847`;
- pre-correction summary SHA-256:
  `D529B1A3BB3977FAD2838307B70466C4B7AD616DB5FE88640B5CEF4BD54DF247`;
- action log SHA-256:
  `EA8AAE76ED3EA946077191006993493C8C572BC2156F4CBE0918C9677307192D`;
- all 4,980 evaluated actions were finite and in range;
- all 20 input replays are preserved in
  `artifacts/replays/reward_v4_evaluation/`.

On the same TMNF race-clock basis, corrected V3 was `28.030s` best, `28.106s`
mean, and `28.660s` worst. V4 therefore improved best time by `3.130s`, mean by
`3.177s`, and worst by `3.710s` while retaining the 20/20 finish rate.

### Precision and oscillation result

The open oscillation question resolved negatively: the fixed detector still
flagged oscillation in 20/20 V4 episodes, versus 20/20 for corrected V3. Mean
peak sign crossings within two seconds changed only from `5.15` to `5.00`.

Lateral precision was mixed rather than uniformly better or worse:

| Metric | Corrected V3 | V4 |
|---|---:|---:|
| Mean absolute lateral offset | `3.846` | `3.749` |
| Mean p95 absolute lateral offset | `9.571` | `10.628` |
| Maximum absolute lateral offset | `13.568` | `17.833` |
| Mean seconds above 5 units | `9.755` | `6.185` |
| Mean seconds above 10 units | `1.245` | `1.640` |

V4 spent less time moderately off the reference line but had worse severe-tail
excursions. Mean absolute steering increased from `0.486` to `0.567`. Removing
V3's flat clamp region was not sufficient to remove oscillation: none of V4's
4,980 deterministic steps reached the raised +/-20 clamp, none moved backward
in projected progress, and the largest progress delta was `17.603` units.
Oscillation therefore cannot be attributed only to V3's active clamp. This is
direct evidence for considering an explicit smoothness or lateral term in V5,
not permission to add one retroactively to V4.

### Qualitative result and where the time came from

The widest V4 episode passed centrally through the final hoop without the
lower-right clip, low recovery, or inversion seen earlier. Its minimum vertical
offset was only `-2.251`, compared with a corrected V3 run that recovered after
reaching `-13.366`. Progress splits show that V4 was `0.100s` behind V3 at
progress 500, `0.415s` ahead at progress 1700, `0.770s` ahead at progress 1900,
`2.560s` ahead at progress 2100, and `3.160s` ahead at progress 2200. The large
gain therefore came mainly from the clean hoop/final section rather than the
opening drop.

The maximum `17.833` lateral excursion occurred around progress 545 after an
aggressive right-edge opening-drop trajectory. Visual review did not establish
that this line improved the early split, so it is documented as a precision risk
rather than claimed as a discovered advanced technique.

The first qualitative capture correctly replayed V4 but inherited a hard-coded
"V3" overlay from the shared inspector. That mislabeled take is preserved in
`artifacts/videos/reward_v4_evaluation/`. The inspector was fixed and the same
replay was recaptured without changing the policy or evaluation:

- labeled video:
  `artifacts/videos/reward_v4_evaluation_labeled/reward_v4_final_6df90018_ep_12.mp4`;
- video SHA-256:
  `5AEB244AB6149CC1FDB01A159F23760FA1360464EC396E23683A7D0651EC1BEC`;
- full-lap, opening-drop, and final-section contact sheets are preserved beside
  the video.

### Hypothesis assessment

The pre-registered hypothesis was supported on its primary claims: V4 preserved
V3's 20/20 deterministic reliability, substantially improved lap time, and had
no near-finish collision in 20 reviewed terminal outcomes. The clean final
section accounted for most of the measured speed gain. The stated open question
also produced a clear result: oscillation did not disappear, and severe lateral
excursions worsened despite the raised clamp never activating. No reward term,
threshold, initialization, or checkpoint-selection rule changed during the run.
