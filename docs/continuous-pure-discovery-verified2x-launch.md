# Verified-2x continuous pure-discovery launch

Launched: 2026-09-05 04:55 EDT

Run: `wr_pure_continuous_verified2x_20260905_045450`

Status at handoff: training continuously after a successful live preflight and
first guarded PPO update. This run uses unchanged V4 reward, no slide bonus,
no steering term, and no action anchor.

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

A one-time local-thread morning follow-up is scheduled for 13:00 EDT, eight
hours after launch. If training is still healthy, it will request a graceful
stop, wait for the final checkpoint, and run the registered deterministic
live-SimState progression evaluation at `2x`. It must report slide onset and
safety from that evaluator, not infer either from stochastic training logs or
input replays.

## Local evidence

- Status: `runs/wr_pure_continuous_verified2x_20260905_045450/status.json`.
- Manifest: `runs/wr_pure_continuous_verified2x_20260905_045450/run_manifest.json`.
- Checkpoints: `checkpoints/wr_pure_continuous_verified2x_20260905_045450/`.
- TensorBoard: `tensorboard/wr_pure_continuous_verified2x_20260905_045450_0/`.
- Process logs:
  `artifacts/logs/continuous_launcher/wr_pure_continuous_verified2x_20260905_045450.*.log`.

## Notable moments

- The launch preflight repeated the earlier 20/20 result rather than assuming
  yesterday's bridge state still held. It matched the expected 2x throughput.
- The first completed optimizer update is a live proof that the configured
  guarded path is executing. A non-intervention is the expected healthy result;
  its finite `0.0197` KL is well below both guard levels.
- The much slower verified domain means the first nominal checkpoint should
  arrive after roughly 3.5-3.9 hours, not every 11-14 minutes as at 100x.
