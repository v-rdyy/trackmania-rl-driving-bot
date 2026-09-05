# Simulation-speed fidelity and unattended-training hardening

Date completed: 2026-09-05

Status: both infrastructure blockers from the failed continuous run are fixed.
Training has not been restarted.

## Physics result

TMInterface simulation speed is an observable dynamics variable for this agent,
not a transparent wall-clock multiplier. The live study ran two checksum-pinned
policies for five deterministic episodes at each of `1x`, `2x`, `4x`, `6x`,
`10x`, `20x`, `50x`, and `100x`. It retained 80 direct-live SimState logs and
80 input replays. Playable `1x` was registered as ground truth.

| Speed | Reliable base | Mean finish | Speed-sensitive checkpoint | Failure profile |
| ---: | ---: | ---: | ---: | --- |
| `1x` | `5/5` | `24.876s` | `0/5` | 5 stuck |
| `2x` | `5/5` | `24.870s` | `0/5` | 5 stuck |
| `4x` | `5/5` | `24.854s` | `1/5` | 4 stuck |
| `6x` | `5/5` | `24.856s` | `0/5` | 5 stuck |
| `10x` | `4/5` | `24.855s` | `0/5` | 4 stuck, 1 timeout |
| `20x` | `5/5` | `24.860s` | `0/5` | 4 stuck, 1 fall |
| `50x` | `5/5` | `25.450s` | `5/5` | none |
| `100x` | `5/5` | `25.576s` | `5/5` | none |

The speed-sensitive checkpoint is the decisive canary. The same model that is
`0/5` at playable speed becomes `5/5` at `50x` and `100x`. At `100x`, its
median trajectory is as much as `48.14` world units and `48.02` progress units
from `1x` at the registered 5/10/15/20-second anchors. This is meaningful
policy/domain miscalibration, not evaluator noise or a missed control period.

The first summary rule considered only finish count and lap time. That could
incorrectly accept two zero-finish arms even when one changed from five stuck
runs to four stuck plus one fall. The corrected frozen rule also requires the
same fall/stuck counts and bounded trajectory, speed, and action differences.
It then selects the highest speed in the *contiguous* passing range from `1x`.
The passing speeds were `1x`, `2x`, and `6x`, but `4x` failed first; accelerated
behavior is non-monotonic, so it would be unsafe to jump over that failure and
call `6x` verified. The unattended runner is therefore pinned to `2x`.

The reliable base's `2x` median path matched `1x` at all registered anchors,
its mean lap differed by `-6ms`, and both policies had identical outcome and
failure counts. A separate no-learning runner preflight then completed `20/20`
finishes and 4,960 finite interactions at `2x` in `248.828s` (`19.933`
interactions/second). It performed zero optimizer updates.

At that measured collection rate, eight hours represents about `574,000`
interactions before PPO optimization overhead. Use `0.52-0.57M` as the honest
planning range; six hours is roughly `0.39-0.43M`, and ten hours is roughly
`0.65-0.72M`. The next real run remains owner-stop-controlled rather than
silently converted to a fixed budget.

## Trust impact on earlier work

The result does not erase V1-V4 or the later reward-comparison story. Those were
real training experiments in the `100x` domain, and several resulting policies,
including the selected reliable base, still work at playable speed. It does
invalidate treating performance observed only at `100x` as evidence of playable
physics. The original continuous run is now classified as infrastructure-
confounded: it had only about two hours of finite training before numerical
divergence, and its initial `100x` retrospective masked failures seen at lower
speed. It is not a fair rejection of the long-budget pure-discovery hypothesis.

Any future PB, reliability, or slide claim must be evaluated at `1x` or the
pinned `2x` fidelity domain. Historical `6x` evaluations remain preserved as
comparative evidence, but should not be silently relabeled as the new playable-
physics baseline. A historical checkpoint whose conclusion matters should be
rechecked at `1x/2x` before that conclusion is reused.

## Numerical divergence protection

The continuous runner now wraps every PPO update in a transactional guard:

- validate rollout data and the incoming policy/Adam state are finite;
- snapshot the policy, optimizer, and update counter in memory;
- reject nonfinite loss or gradients before an optimizer step;
- run the update with `target_kl=0.20`, which gives PPO a minibatch early-stop
  threshold of `0.30`;
- reject nonfinite policy/optimizer tensors, nonfinite deterministic outputs,
  missing/nonfinite training metrics, or mean approximate KL above `0.50`;
- restore and revalidate the exact pre-update state on any violation; and
- save a uniquely labeled `rollback_model.zip`, write a `guarded_stop` record,
  and stop instead of continuing from contaminated weights.

The thresholds are deliberately less intrusive than Stage 2b's `0.01` guard.
The failed run's stable prefix had approximate-KL p99 near `0.254`; the terminal
sequence first exceeded `1` and eventually reached roughly `4.329e11`.
`0.20/0.50` should preserve ordinary historical updates while stopping the
known explosion pattern. Tests inject both NaN damage and a finite `0.75` KL
spike and verify exact rollback. The guard has not yet had to fire during a
real TrackMania update; that is intentional, because this infrastructure pass
does not authorize new learning.

## Evidence

- Full local study: `runs/simulation_speed_fidelity_20260904/summary.json`,
  SHA-256
  `11D269D9250DF49C23385D05586C44A9D268C4DC119FF5D61BA6A1C5D1643A38`.
- Initial, insufficient gate retained as
  `runs/simulation_speed_fidelity_20260904/summary_initial_gate.json`.
- Successful no-learning preflight:
  `runs/wr_verified_2x_preflight_20260905b/run_manifest.json`, SHA-256
  `8731B8DB41C81AF65E5B0645EDD7452DCEEF9473181B1F23E77BC4AA7F01CA22`.
- The first failed 2x preflight is retained under
  `runs/wr_verified_2x_preflight_20260905/`; it changed no weights.
- The tracked speed pin is `config/verified_simulation_speed.json`; the runner
  refuses to start if its local evidence is absent, changed, or selects a speed
  other than `2x`.

## Notable moments

- The original `100x` retrospective looked perfect at `80/80`, but the canary
  is `0/5` at `1x` and `5/5` at `100x`. Comparing identical policies and
  registered race-time anchors isolated simulation speed itself as the cause.
- The first improved classifier still had a conceptual hole: fail/fail could
  pass despite a changed failure mechanism. Adding failure-profile and
  trajectory checks fixed it; using a contiguous range prevents cherry-picking
  the non-monotonic `6x` pass after `4x` failed.
- A sandboxed launcher could not see the healthy game window, spawned a hidden
  duplicate, and timed out. Running the desktop-dependent study with desktop
  access attached to the original bridge; the exact hidden duplicate was
  identified by path/window handle and removed.
- The first pinned-`2x` preflight stopped at three minutes because its timeout
  still assumed `100x`. The diagnostic showed zero optimizer updates and a
  clean timeout. Scaling the ceiling by episode duration fixed it, and the
  rerun passed `20/20` in `248.828s`.
- Selecting `2x` is conservative. `6x` is close on the two measured policies,
  but accepting it after a `4x` outcome change would prioritize throughput over
  the user's requirement that the next run reproduce playable physics.
