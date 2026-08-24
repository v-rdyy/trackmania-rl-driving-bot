# Reward v3: Clamped forward centerline progress

Status: Complete; trained and evaluated, including an unchanged-model
fall-detector correction study

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

## Pre-training implementation verification

The frozen reward and stuck cutoff passed 104 offline tests. A live A01 gate then
ran 20 consecutive snapshot-reset episodes and 400 action steps through the real
TMInterface bridge. Every action was finite and in range, all reset observations
were identical (`0.0` maximum delta), and all 20 neutral-action episodes stopped
at the 2.0-second stuck cutoff rather than a fall or timeout. The final observed
window contained only `0.108391` units of world motion, confirming that A01's
small start-platform settling does not cross the 2.0-unit motion threshold.

The first live attempt incorrectly used full brake as a stationary action. In
TMNF, brake from rest also selects reverse: the log showed signed Gas `+65536`,
speed rising to `75`, zero progress, and a fall `10.184` units below the start at
1.6 seconds. The smoke action was corrected to true neutral; no reward,
termination, or training parameter changed. Stationary projection jitter
produced a negligible maximum per-step reward of `0.0000334`, within the frozen
`[0, 1]` contract.

Ignored live evidence:

- action log SHA-256:
  `635803D2BD3E718BE5087F38D756285990BC103EC5A3EE02B73C064336D8748D`;
- smoke summary SHA-256:
  `5F8451FB46864D0B87D53FEF12113DEB0FF93EAD9D93E75EA32C0F027364A7EC`.

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

### Training

The final seed-42 run completed `1,001,472` PPO timesteps in `3,266.40s`
(`54m 26s`) at 100x, with no resume or discarded training interactions. The
TensorBoard mean episode reward rose from `1.539` to `189.427` and peaked at
`202.039`; mean episode length rose from `59.824` to `250.990` steps and peaked
at `332.990`. That is real learning signal and strongly differs from V1's flat
sparse result.

Training recorded 1,086 finishes across 3,787 episodes, but its terminal-cause
counts used the original immediate vertical-drop detector. The detector issue
found below means the 2,281 recorded training "falls" are not a trustworthy
measure of physical failure. The checkpoint itself was not modified:

- final model: `checkpoints/reward_v3/final_model.zip`;
- SHA-256:
  `C9791B6ECF3E83146299376F2180D3132250061C8B33DAF294CF295F1996FE38`.

### Original pre-registered evaluation

The original 20-episode deterministic evaluation at 6x finished 12/20 runs
(`60%`) and stopped the other eight as vertical falls at the final jump. The 12
recorded finishes ranged from `27.730s` to `28.210s`, averaging `27.958s`.
Steering oscillation was detected in 20/20 episodes; no inversion or stuck
period was observed before the early fall cutoffs.

This remains the immutable result of the pre-registered evaluator as it existed
when the run began. Its summary is `runs/reward_v3/evaluation_summary.json`,
SHA-256
`F8B90E9CC417D625D06C2CA2ACD2DBAD21DCEB6336E3B4C358EFA9C41C95E9B8`.

### Fall-detector diagnosis and one corrected re-evaluation

Replay inspection showed that crossing 10 units below the reference path was
not by itself proof of a fall on A01's final jump. A nominally failed replay was
still moving and gaining progress after the cutoff, landed on the lower course,
and only later stopped at a pillar. A corrected successful replay likewise
reached `-13.366` vertical offset, recovered onto the final straight, and
finished. The original detector therefore confused a low but recoverable final
jump trajectory with terminal failure.

The detector was corrected to require both conditions before declaring a fall:

1. the car is more than 10 vertical units below the reference path; and
2. the existing frozen 2.0-second V3 stuck window also confirms less than 1.0
   unit of progress and less than 2.0 units of world motion.

The same checkpoint was then evaluated once for 20 deterministic episodes at
6x. It finished 20/20 (`100%`) with zero falls, stuck truncations, timeouts,
off-track truncations, or inversions. Finish times were:

- best: `27.930s`, which is `3.430s` slower than the real `24.5s` human PB;
- mean: `28.016s`;
- worst: `28.560s`.

The completed live run produced all 20 replays and 5,659 raw action records.
The first summary pass then exposed a restarted-countdown prefix in episode 0.
The summary was rebuilt from the preserved completed log and replays instead of
rerunning the model: 45 startup records were disclosed and removed, leaving
5,614 evaluated actions. All evaluated actions were finite and in range.

Corrected summary: `runs/reward_v3/corrected_detector_final_summary.json`,
SHA-256
`DAA7C6FBB2DACF14D29017F3BC99DBF44B1850CF92456A39D24AACCEBB78853D`.

### Precision result and remaining reward flaw

The corrected result establishes reliable finishing, not clean driving:

- steering oscillation was still detected in 20/20 episodes;
- mean per-episode p95 absolute lateral offset was `9.571` units;
- maximum absolute lateral offset was `13.568` units;
- the widest run visibly took an awkward low final-hoop exit, became yawed and
  airborne, dropped below the old cutoff, recovered, and finished.

V3's clamp is a separate incentive problem from V2's raw-speed reward. In the
original 5,261-step evaluation, 1,640 steps (`31.17%`) exceeded the 10-unit
positive-progress cap. No evaluated step moved backward in projected progress,
and the maximum single-step delta was `13.469` units. The cap discarded `7.33%`
of otherwise valid progress value. Once a step earns the maximum reward, V3 has
no immediate preference for cleaner progress beyond 10 units, so oscillating or
taking a wider line can be equally rewarded. This is evidence of a flat reward
region that can support imprecision, not proof that the policy consciously
"chose" to exploit it. PPO's `gamma=0.99` also creates a mild implicit preference
for earlier progress even though V3 has no explicit time term.

The pre-registered hypothesis is therefore supported on reliability and dense
learning signal, while its stated oscillation/checkpoint-risk caveat also
materialized. No reward or threshold was changed during training. V3 is a
reliable foundation for V4, but it is not yet a precise or human-PB-level agent.

Qualitative evidence is preserved in:

- `artifacts/videos/reward_v3_final_jump/` for the original early-cutoff
  success/failure comparison;
- `artifacts/videos/reward_v3_corrected_detector/` for the widest corrected
  finish and contact sheets;
- `artifacts/replays/reward_v3_corrected_detector_final/` for all 20 corrected
  input replays.
