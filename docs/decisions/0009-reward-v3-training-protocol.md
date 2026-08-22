# Decision 0009: Reward V3 reliability-only training protocol

Status: Accepted before implementation and training

Date: 2026-08-22

## Context

V2 learned near-complete A01 driving but its raw-speed reward tolerated poor
precision. The fixed retrospective found steering oscillation in 20/20 final
evaluation episodes, inversion in 19/20, and stuck periods in all 14 timeouts.
Live review identified repeated lower-right contact at the final checkpoint/hoop
as the cause of the flip.

The next experiment must isolate whether rewarding route progress fixes
reliability. Adding speed, time, steering, or lateral terms now would test several
changes at once and make the result harder to interpret.

## Decision

Use the isolated V3 reward and stuck thresholds pre-registered in
`reward_v3.md`:

- `reward = min(max(progress_delta, 0), 10) / 10`;
- a 2.0-second rolling stuck window;
- stuck only when progress gain is below 1.0 unit and world path length is below
  2.0 units;
- stuck truncates without a penalty;
- no finish bonus or speed, time, crash, steering, lateral, or inversion term.

Train from a new seed-42 PPO model for at least 1,000,000 timesteps at 100x using
the V2 PPO/gSDE hyperparameters, 100 ms control period, 45-second episode limit,
50,000-step checkpoints, and the distinct TensorBoard name
`reward_v3_centerline_progress`. Use corrected pedal semantics.

Evaluate the final checkpoint for 20 deterministic episodes at 6x using the
fixed Decision 0008 precision metrics. Do not tune the reward or thresholds
mid-run to eliminate final-checkpoint contact.

## Interpretation

This is a reliability experiment, not speed optimization. Lap times must be
reported against the owner's real `24.5s` A01 PB, but a slower reliable agent is
an expected intermediate result. The intentionally cautious `33.280s` telemetry
lap is not a performance baseline.

V4 time/speed efficiency and later reward-shaped technique-discovery experiments
are deferred until V3 training and evaluation are complete.

## Operational gate

The pre-V3 source/evidence archive and checksum in `docs/pre-v3-backup.md` must
remain present. Verify host sleep and hibernation are disabled immediately before
training. Preserve failures, resume only from real checkpoints, and disclose all
discarded/replayed interactions as in V2.
