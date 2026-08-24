# Reward v4: Signed progress with time and terminal outcomes

Status: Pre-registered and approved; training not started

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

Pending. Complete this section from preserved training and evaluation artifacts
without modifying the hypothesis, coefficients, thresholds, or protocol above.
