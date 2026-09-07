# Continuous pure-discovery retrospective

Date evaluated: 2026-09-04

Run: `wr_pure_continuous_20260904_062051`

Status: stopped by PPO numerical failure after `1,835,008` additional
interactions. The run was not stopped by Windows sleep, the monitor, a stop
request, or a TMInterface disconnection. No training has been restarted.

2026-09-05 infrastructure conclusion: this run is not a fair test of the full-
length pure-discovery hypothesis. Only its finite prefix is usable, and its
`100x` training/evaluation domain is not playable-physics faithful. The
[live fidelity and numerical-hardening report](simulation-speed-fidelity.md)
pins the replacement runner to `2x` and adds exact pre-update rollback. Pure
discovery remained the approved experiment direction; after a fresh launch
confirmation it began as the separate
[verified-2x run](continuous-pure-discovery-verified2x-launch.md).

2026-09-07 update: the verified-2x replacement was itself invalidated by an
unexpected whole-host shutdown after only `25,747` collected interactions and
before its first checkpoint. Its 12 optimizer rows remained finite and the
active numerical guard had no reason to intervene. The separate
[interrupted-run report](verified2x-interrupted-run.md) records the Windows
events, surviving evidence, and checkpoint-recovery recommendation. This second
attempt also remains insufficient to accept or reject long-budget discovery.

## Outcome

This longer unchanged-V4 experiment did not demonstrate useful pure discovery
within its completed budget. Across all eight deterministic 6x checkpoint
evaluations:

- no checkpoint produced a confirmed telemetry slide in the first-turn zone,
  final-corner zone, or elsewhere on the track;
- every evaluated episode still met the existing oscillation detector;
- the final checkpoint finished `10/10`, with `24.820s` best and `24.823s`
  mean, only `0.020s` and `0.035s` faster than the starting checkpoint;
- the project's previously preserved `24.730s` lap therefore remains the
  fastest observed driven run;
- deterministic reliability and pace changed discontinuously during training,
  including `0/10` and `2/10` checkpoints and a later `34.184s` mean; and
- the experiment ended in numerical divergence before the owner requested a
  stop.

The final checkpoint is `0.320s` behind the owner's `24.500s` PB and `1.050s`
behind the current `23.770s` benchmark recorded in `wr-chase-plan.md`. Its
`0.020s` best-time advantage over the starting checkpoint is smaller than the
starting checkpoint's own `0.040s` ten-lap spread, so this is not claimed as a
meaningful speed improvement.

This extends the Stage 1 pure-discovery evidence by `1,835,008` interactions,
for `3,835,904` interactions across the original Stage 1 and this continuous
extension. It remains a budget-limited result, not proof that discovery is
impossible. Yosh's reported experiment used roughly 400 training hours at
20 Hz; this project controls at 10 Hz and has used far less compute.

## Run health and failure cause

Training began at 02:21:54 EDT and failed at 03:58:13 EDT after about 96.3
minutes. It completed 895 PPO updates and learned through model timestep
`4,837,376`. The stochastic training stream contained 6,980 episodes and 6,626
finishes (`94.93%`); its most recent 500 episodes finished at `97.2%`.

The terminal error was not environmental. PPO's policy-distribution mean became
all NaN during an optimizer update and PyTorch rejected the invalid Normal
distribution. The run had no `target_kl` guard because its optimizer settings
were intentionally kept identical to the earlier V4/Stage 1 experiment.
TensorBoard shows the warning sequence before failure: approximate KL first
exceeded 1 around model timestep `4,685,824`, exceeded 100 by `4,700,160`,
exceeded 1,000 by `4,743,168`, and ultimately reached approximately
`4.329e11`.

The emergency archive faithfully preserves the terminal state but contains
NaN parameters and cannot be loaded. All seven periodic checkpoints load and
have finite parameters and optimizer state. The last periodic checkpoint,
`step_000001751040.zip`, was saved after KL divergence had already begun. It is
valid for deterministic evidence but is not treated as a safe optimizer state
for resumed training. The last checkpoint saved before approximate KL exceeded
1 is `step_000001501184.zip`; its deterministic 6x mean is `34.184s`, so it is
not a useful speed-chase resume point either.

## Comparable 6x checkpoint evaluation

Each checkpoint received ten deterministic episodes at the established 6x
formal-evaluation speed. Full live SimState, action logs, input replays, finish
times, safety outcomes, lateral deviation, steering reversals, and slide
detection were retained.

| Additional interactions | Finishes | Best | Mean | Worst | Mean p95 lateral offset | Mean significant reversals | Confirmed slides |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| starting checkpoint | `10/10` | `24.840s` | `24.858s` | `24.880s` | `11.68` | `32.0` | `0` |
| `251,904` | `0/10` | - | - | - | `23.44` | `36.6` | `0` |
| `501,760` | `10/10` | `24.930s` | `24.943s` | `24.990s` | `9.81` | `36.3` | `0` |
| `751,616` | `10/10` | `25.040s` | `25.058s` | `25.070s` | `11.43` | `37.8` | `0` |
| `1,001,472` | `2/10` | `26.350s` | `27.385s` | `28.420s` | `21.19` | `46.6` | `0` |
| `1,251,328` | `9/10` | `24.920s` | `24.942s` | `25.050s` | `16.54` | `41.7` | `0` |
| `1,501,184` | `10/10` | `32.490s` | `34.184s` | `35.750s` | `14.64` | `63.1` | `0` |
| `1,751,040` | `10/10` | `24.820s` | `24.823s` | `24.830s` | `8.31` | `37.2` | `0` |

There were 61 finishes among the 80 formal episodes. The closest observed
finish to their global `26.523s` mean was `26.350s`; this aggregate mean is
pulled upward by the slow `1,501,184` checkpoint and is not a model-selection
metric.

The stochastic training stream did not expose this deterministic fragility.
Its seven 250k windows reported finish rates from `92.76%` to `97.72%`, while
the corresponding deterministic checkpoints ranged from `0/10` to `10/10`.
Stochastic episode statistics are useful training-health evidence, but they
cannot substitute for frozen deterministic evaluation.

## Simulation-speed sensitivity

The same audit was first run at the 100x speed used for training. It produced
`80/80` finishes, including `10/10` for the checkpoint that later finished
`0/10` at 6x. The starting checkpoint was `25.490s` best and `25.589s` mean at
100x, versus `24.840s` and `24.858s` at 6x. Identical initial state/action
samples diverged under the two physics rates within the first few control
steps and produced materially different trajectories.

The 100x evaluation remains preserved as a real domain-sensitivity finding,
but its lap times and finish rates are not used as PB-comparable evidence. The
established 6x rerun is the formal comparison above. This also means that long
100x training can optimize a behavior distribution whose reliability differs
from the 6x presentation/evaluation environment.

## Evidence and replay set

- Formal 6x aggregate:
  `runs/wr_pure_continuous_20260904_062051/retrospective_6x/retrospective.json`,
  SHA-256
  `717648418582A937C7DA3C05F837F6E447B8BB59078694D3FB5BE7B480500B67`.
- Preserved 100x sensitivity aggregate:
  `runs/wr_pure_continuous_20260904_062051/retrospective_live/retrospective.json`,
  SHA-256
  `FA0A4C57EE8FBEE81296E32E99B3662471683FEB9108DDE8EB93B1C057FD5C70`.
- Formal 6x input replays:
  `artifacts/replays/wr_pure_continuous_20260904_062051/retrospective_6x/`.
- Periodic model checkpoints:
  `checkpoints/wr_pure_continuous_20260904_062051/`.

The formal set's best finish is `24.820s`, its finish closest to the global
mean is `26.350s`, and its worst finish is `35.750s`. All three input replays
are retained in their checkpoint directories. The no-finish `251,904`
checkpoint is also preserved and can be rendered separately when a failure
progression video is wanted.

## Recommendation before more training (resolved 2026-09-05)

Do not resume the emergency checkpoint or the final checkpoint's optimizer
state. The two infrastructure choices are now resolved:

1. use `2x`, the highest contiguous speed that passed the new `1x` fidelity
   contract; and
2. use pre-update policy/optimizer snapshots, finite rollout/loss/gradient/
   state/output checks, PPO `target_kl=0.20`, hard mean KL `0.50`, exact
   rollback, and a controlled guarded stop.

A safety guard changes the exact optimizer protocol, but continuing without
one would repeat a known risk. If pure discovery is retried,
the most defensible base is the original checksum-pinned Stage 1 Gate 2 model,
not a post-divergence optimizer state. If the objective is fastest progress
toward the record rather than one more pure-discovery budget, this result is
also evidence for returning to a separately pre-registered technique-specific
experiment.

## Notable moments

- The overnight stop initially looked like another host interruption, but the
  status heartbeat, uninterrupted game process, absent stop request, and
  TensorBoard KL explosion isolated an optimizer failure. The runner correctly
  retained seven recoverable checkpoints even though the emergency state was
  already contaminated by NaNs.
- The first reused-game evaluator counted the negative race countdown toward
  stuck time and truncated before the car could start. Switching stuck timing
  to active race time fixed the false failure and now has a regression test.
- The first completed 100x audit looked reassuring at `80/80`, but replaying
  the same models at the established 6x exposed `0/10` and `2/10` checkpoints.
  That unexpected difference revealed simulation speed as a genuine physics
  domain variable, not merely a wall-clock accelerator.
- The final checkpoint recovered to `10/10` and `24.820s` even though it was
  saved after the KL warning sequence began. It is useful behavioral evidence,
  but selecting its optimizer for further training would confuse a good
  evaluation snapshot with a safe learning state.
