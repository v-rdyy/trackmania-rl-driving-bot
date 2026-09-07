# Verified-2x interrupted-run report

Date inspected: 2026-09-07

Run: `wr_pure_continuous_verified2x_20260905_045450`

Outcome: infrastructure-invalidated. An unexpected whole-host shutdown ended
the run after approximately 21.5 minutes of learning, before its first model
checkpoint. This is not evidence for or against long-budget pure discovery.

## What survived

- `25,747` collected interactions.
- `24,576` interactions learned through in `12` complete PPO updates.
- Model timestep `3,028,992` at the last complete update.
- `98` stochastic training episodes, including `66` finishes (`67.35%`).
- `24.800s` stochastic best and `25.093s` mean among finishes.
- Five checksum-pinned input trajectories, the monitor log, TensorBoard, status,
  and launch manifest.

The next rollout contained `1,171` collected but unlearned interactions when
the machine stopped. There is no periodic, final, or rollback checkpoint for
this run because the first nominal 250k boundary was never reached. The original
Stage 1 Gate 2 base remains intact; every updated policy state from this attempt
is lost.

## Cause and numerical-health evidence

Windows event 6008 reports that the previous system shutdown at 05:20:40 EDT on
2026-09-05 was unexpected. Kernel-Power event 41 at the next boot reports that
Windows restarted without a clean shutdown and identifies a hang, crash, or
power loss as the class of event. The evidence does not distinguish among those
host-level causes.

The training status heartbeat was last buffered at about 05:21 EDT. Its small
offset from the Windows event timestamp is treated as timestamp/buffering
uncertainty, not a basis for ordering events to the second. There is no clean
sleep/shutdown event, Python exception, TMInterface disconnect, `STOP.request`,
guard event, `guarded_stop`, or rollback artifact.

All 12 optimizer rows in TensorBoard are finite:

- mean approximate KL: minimum `0.013014`, maximum `0.043511`, last `0.017301`;
- loss: minimum `14.961`, maximum `1448.450`, last `1129.018`;
- clip fraction: minimum `0.1141`, maximum `0.3526`, last `0.1407`; and
- entropy and value-loss rows are finite.

These values do not resemble the prior numerical explosion. The rollback/NaN
guard was active and correctly did not fire. An in-process optimizer guard
cannot save a process from loss of the whole host.

## What cannot be concluded

The stochastic finish figures are not a deterministic checkpoint evaluation.
There is no saved updated model, so the registered checkpoint-progression and
live-SimState slide evaluation cannot be run. This attempt therefore provides
no reliable pace progression, finish-rate progression, plateau, or global
slide-onset result.

Five selected input trajectories remain and can be replayed through TMInterface
for live SimState. The audit tool verifies both local and TMInterface checksums,
captures full wheel/contact/slip dynamics, and applies the frozen slide-onset
contract. After reboot, however, the bridge listened while A01 produced no
race-step callback, and three connection attempts timed out before any replay
ran. The audit is explicitly pending until A01 is in an active race. Even a
successful audit is only a five-trajectory sample, not a census of the 98
training episodes.

## Recovery recommendation

Keep the formal 250k comparison/evaluation cadence for continuity, but consider
adding recovery-only checkpoints every 25k-50k interactions before the next
overnight attempt. At the measured `19.95` interactions/second, that limits an
external-outage loss to roughly 21-42 minutes instead of roughly 3.5 hours.
Model archives are small relative to the verified disk headroom. This is a
protocol recommendation, not an unapproved restart or silent cadence change.

Host-level resilience is a separate decision. Windows sleep was already
disabled, so an automatic reboot-resume task or a UPS would address different
failure modes; neither is inferred as authorized here.

## Evidence

- status SHA-256:
  `A41AC1CC2DCEE98BAF21809696BAED19F5077D7952F554F39425B3924BB51D80`;
- run-manifest SHA-256:
  `A4EE03136F2811033FC94BCB6850A938911E02212D9461AE72F2A20FD555E46B`;
- selected-replay catalog SHA-256:
  `8648600EAD0EEE394CE22FCEAE13CBDB1479C26BF5F9DA1FF61FD0B3E999C2B`;
- status and manifest:
  `runs/wr_pure_continuous_verified2x_20260905_045450/`; and
- TensorBoard:
  `tensorboard/wr_pure_continuous_verified2x_20260905_045450_0/`.

## Notable moments

- The run originally appeared simply stale. The missing trainer/game processes,
  matching Windows unexpected-shutdown events, empty checkpoint directory, and
  finite final TensorBoard rows isolated a whole-host failure rather than a
  repeat optimizer explosion.
- The numerical guard's non-intervention is meaningful: it was active, all
  audited values stayed finite and below its KL ceiling, and there was no bad
  update to roll back. Its protection boundary ends at host power/process life.
- A 250k cadence was reasonable at the former 100x throughput but costs about
  3.5 hours between recoverable states at 2x. Separating frequent recovery
  snapshots from formal evaluation gates preserves comparability while reducing
  outage loss.
- Slide discovery is unresolved. The five inputs are recoverable evidence, but
  scoring them before A01 emits live callbacks—or generalizing them to all 98
  episodes—would overstate what survived.
