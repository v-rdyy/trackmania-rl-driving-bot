# Reward v3: Clamped forward centerline progress

Status: Pre-registered and approved; training not started

Date pre-registered: 2026-08-22

Approved by the project owner on 2026-08-22.

## Pre-registered hypothesis

A clamped dense reward based only on positive forward progress along A01's
reference centerline will align PPO with route completion better than V2's
raw-speed reward. Clamping each 100 ms progress gain at 10 units is expected to
prevent large projection jumps from dominating learning, while truncating an
episode after a rolling 2.0-second window with less than 1.0 unit of centerline
progress and less than 2.0 units of world motion is expected to stop
unrecoverable stalls without prescribing steering. Expected outcome: V3 improves
finish rate and reduces final-checkpoint timeouts and stuck duration relative to
V2, but it may still oscillate or clip the final checkpoint because steering and
lateral error are measured only, not penalized.

This statement was approved before implementation or training and must not be
reworded after results are observed.

## Reliability-only goal

V3 is not a lap-time or speed optimization experiment. Its reward contains no
time or speed term. Its sole job is to test whether centerline progress plus
stuck truncation can produce an agent that finishes A01 reliably without the
final-checkpoint inversion and long stuck time seen in V2.

The owner's actual A01 human PB is `24.5s`. The earlier `33.280s` manual lap was
intentionally slow and cautious for telemetry capture. V3 lap times will be
reported after reliability evaluation, but being slower than `24.5s` is expected
and is not a V3 failure. The speed gap becomes a separate optimization problem
only after reliable finishing is established.

## Fixed reward contract

At each 100 ms control step:

```text
forward_progress = max(0, current_progress - previous_progress)
reward = min(forward_progress, 10.0) / 10.0
```

- Reward range is exactly `[0, 1]` per step.
- Backward progress produces zero, not a negative penalty.
- There is no finish bonus.
- There is no timeout, fall, off-track, stuck, crash, steering, lateral,
  inversion, time, or speed reward term.
- Termination or truncation only stops future rewards; the final step retains
  the same progress reward calculation.
- The reward remains an isolated callable injected into `TrackmaniaEnv`.

## Fixed stuck truncation

- Rolling window: `2.0s`, equal to 20 control steps at 100 ms.
- Centerline-progress condition: window gain is `< 1.0` unit.
- World-motion condition: cumulative three-dimensional path length inside the
  same window is `< 2.0` units.
- Both conditions must hold simultaneously.
- The episode is truncated with `stuck=True`; it is not marked as a finish and
  receives no extra penalty.
- The existing 45-second timeout, 50-unit horizontal off-track boundary, and
  10-unit vertical-fall boundary remain safety truncations, not reward terms.

World motion is required so a moving car with low projected progress is not
mistaken for physically stuck. No steering, lateral, or checkpoint-specific rule
is added to force a clean result.

## Control and training protocol

- Corrected TMNF pedal semantics are mandatory; V3 does not use the V0/V1/V2
  compatibility mapping.
- Seed: `42`.
- PPO/gSDE hyperparameters: unchanged from the formal V2 experiment.
- Training speed: `100x`; control period: `100ms`.
- Minimum training budget: `1,000,000` model timesteps.
- Checkpoint interval: `50,000` model timesteps.
- TensorBoard run name: `reward_v3_centerline_progress`.
- Host sleep and hibernation must be disabled immediately before training.

The pedal correction means V3 is not literally a reward-only transport-boundary
comparison with V2. That difference is already documented and must remain
visible in the final analysis.

## Evaluation protocol

Run 20 deterministic episodes at 6x from the final checkpoint and report:

- finish rate and every terminal cause;
- finish-time distribution and comparison with the real `24.5s` human PB,
  explicitly without treating pace as V3's optimization target;
- reward and episode-length curve shape;
- centerline progress and lateral deviation;
- fixed-threshold steering oscillation, inversion, and stuck-duration metrics;
- visible final-checkpoint behavior, especially whether lower-right contact,
  flipping, or recovery still occurs;
- all action range/finite checks and artifact hashes.

If V3 still clips the final checkpoint, document the result without changing the
reward or thresholds mid-run.

## Deferred context, not V3 work

After reliable finishing is solved, the planned next direction is a V4 that adds
time efficiency or speed on top of the winning reliability reward. Later
experiments may test whether advanced techniques emerge from reward shaping
alone: first optimizing the opening drop to reduce airtime and preserve
acceleration time, then attempting harder behavior such as speed sliding without
hand-coded controls. The research result is which techniques emerge unaided,
which need stronger shaping or curriculum, and which are not discovered.

None of that work begins during V3.

## Actual outcome

Pending. This section must be completed from the real training and evaluation
artifacts without modifying the hypothesis or fixed thresholds above.
