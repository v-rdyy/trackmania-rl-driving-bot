# Verified-2x continuous pure-discovery launch

Launched: 2026-09-05 04:55 EDT

Run: `wr_pure_continuous_verified2x_20260905_045450`

Final observed status: externally interrupted by an unexpected whole-host
shutdown after about 21.5 minutes of learning. This run used unchanged V4
reward, no slide bonus, no steering term, and no action anchor. It did not run
long enough to test the overnight pure-discovery hypothesis.

## Frozen launch contract

- Base: Stage 1 Gate 2 at model timestep 3,004,416, SHA-256
  `BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044`.
- Simulation speed: checksum-pinned `2x`, backed by the
  [live fidelity study](simulation-speed-fidelity.md).
- Checkpoint cadence: after each completed PPO update crossing a nominal
  250,000 additional interactions, plus a final or rollback checkpoint.
- Duration: one uninterrupted eight-hour overnight session unless a safety
  failure stops it first. No performance or approval gate interrupts training.
- Local process only: no OpenAI API/model calls or Codex polling loop is part
  of training.

The run manifest records `target_kl=0.20`, PPO's resulting `0.30` minibatch
early-stop threshold, and the hard post-update mean-KL ceiling of `0.50`.
Every update snapshots policy and Adam state first; nonfinite rollout/loss/
gradient/state/output data or excess KL restores that exact state, saves a
distinct rollback model, records `guarded_stop`, and ends the run.

## Observed launch

The runner independently rechecked High Performance power settings, all five
AC/DC idle settings at zero, about 475 GiB free, the base checksum, and the
fidelity-evidence checksum. Its no-learning preflight completed `20/20`
finishes, 4,960 finite interactions, and zero optimizer updates in `248.890s`
at `19.928` interactions/second.

The first real PPO update completed at model timestep 3,006,464. TensorBoard
recorded finite mean approximate KL `0.019714`, loss `40.4343`, and clip
fraction `0.227344`. The hard guard accepted it; no rollback or stderr event
occurred. At 2,395 additional interactions, live throughput was `19.945`
interactions/second and the stochastic stream had `7/9` finishes. These are
health signals, not deterministic evaluation results.

Projected collection before PPO overhead is about 574,000 interactions in
eight hours. The honest planning range after optimizer overhead is
`0.52-0.57M`; six hours is `0.39-0.43M`, and ten hours is `0.65-0.72M`.

The one-time 13:00 EDT follow-up could not run because the machine remained
powered down. It was deleted after the reboot so it cannot fire against stale
state.

## Observed interruption

The last status heartbeat was written at approximately 05:21 EDT with `25,747`
collected interactions, `12` completed PPO updates, and model timestep
`3,028,992`. Of those interactions, `24,576` were learned through; the remaining
`1,171` were in the next rollout when the host stopped. Windows event 6008 says
the prior shutdown at 05:20:40 EDT was unexpected, and event 41 on the next boot
says Windows restarted without a clean shutdown. The small timestamp mismatch
between the buffered status file and Windows event is not used to claim a more
precise failure order.

There is no Python traceback, TMInterface socket error, owner stop request,
guard event, rollback model, or `guarded_stop`. All 12 TensorBoard update rows
are finite. Mean approximate KL ranged from `0.0130` to `0.0435`, far below the
`0.50` hard ceiling. The numerical guard therefore did not intervene and shows
no evidence that it should have.

The stochastic collection stream recorded `66/98` finishes (`67.35%`), a
`24.800s` best finish, and `25.093s` mean among finishes. These are health data,
not a deterministic policy evaluation. The process died before the first
nominal 250k checkpoint, so all 12 updated policy states were lost and no valid
checkpoint progression or global slide-onset conclusion can be produced.

Five selected input trajectories survived. A live-SimState audit was prepared,
but after the reboot TMInterface listened without A01 producing race-step
callbacks. Three clean connection attempts timed out before replay playback.
The audit remains pending; input replay telemetry alone is not substituted for
the frozen live-SimState slide contract. Even after it runs, it will describe
only those five trajectories, not all 98 episodes.

## Local evidence

- Status: `runs/wr_pure_continuous_verified2x_20260905_045450/status.json`.
- Manifest: `runs/wr_pure_continuous_verified2x_20260905_045450/run_manifest.json`.
- Checkpoints: `checkpoints/wr_pure_continuous_verified2x_20260905_045450/`.
- TensorBoard: `tensorboard/wr_pure_continuous_verified2x_20260905_045450_0/`.
- Process logs:
  `artifacts/logs/continuous_launcher/wr_pure_continuous_verified2x_20260905_045450.*.log`.
- Forensic report:
  [verified2x-interrupted-run.md](verified2x-interrupted-run.md).

## Notable moments

- The launch preflight repeated the earlier 20/20 result rather than assuming
  yesterday's bridge state still held. It matched the expected 2x throughput.
- The first completed optimizer update is a live proof that the configured
  guarded path is executing. A non-intervention is the expected healthy result;
  its finite `0.0197` KL is well below both guard levels.
- The much slower verified domain means the first nominal checkpoint should
  arrive after roughly 3.5-3.9 hours, not every 11-14 minutes as at 100x.
- That cadence became a real recovery weakness: an external host failure before
  250k preserved logs and selected inputs but no learned policy. A future retry
  should consider 25k-50k recovery checkpoints, while retaining 250k formal
  evaluation spacing, as a separately approved protocol change.
