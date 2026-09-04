# Continuous pure-discovery launch

Launched on 2026-09-04 following the owner's explicit readiness confirmation.
Run: `wr_pure_continuous_20260904_062051`.

## Frozen experiment

- Base: Stage 1 Gate 2, model timestep 3,004,416, SHA-256
  `BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044`.
- Reward and PPO settings are unchanged from V4 / pure-discovery Stage 1.
  No zone bonus, steering term, anchoring loss, or Stage 2b KL guard.
- Run until the owner requests stop, or a health failure prevents continuation.
  No fixed duration, interaction ceiling, or performance/evaluation pause.
- Save after completed updates crossing nominal 250k-interaction boundaries.
  Retain all intervals, selected input replays, Monitor data, and TensorBoard.
- The runner and tests were committed as `27b4aae` before launch.
- The pre-launch protocol document was pinned in the manifest as SHA-256
  `04356E2E3C17CD7AF559A3FBF4F9D147CC893529341432BA6569BEE6BD8B216F`.
  See [the approved protocol](overnight-pure-discovery-preflight.md).

## Live preflight evidence

The independently launched background process passed a 60-second no-learning
warm-up at simulation speed 100x: 25,998 interactions and 101 complete episodes,
all finishing. Repeated post-finish snapshot resets worked. This is a connection,
finite-data, and reset check, not a new formal reliability or slide evaluation.
No warm-up replay set was retained, and no lap-time or slide result is inferred.

Warm-up inference throughput was 433.3 interactions/second; that excludes PPO
optimization and must not be substituted for end-to-end training throughput.
Automatic sleep, hibernate, hybrid sleep, display-off, and disk-idle settings
were all zero on AC and DC. The process acquired a Windows keep-awake request.
Initial free disk space was 513,039,687,680 bytes (about 478 GiB).

Training began at 06:21:54 UTC / 02:21:54 EDT. The actual trainer PID recorded
in `status.json` was 1344; the launching virtual-environment parent PID was
15372. PID numbers are historical evidence, not durable stop targets.
The first selected input replay was recovered and copied successfully, and
TensorBoard output was created. There is no Codex automation or API loop.

At 60 seconds of actual training, the live status showed 18,054 additional
interactions, eight completed PPO updates, and 300.9 interactions/second.
The error log was empty and episode/replay files were advancing. This early
rate implies roughly 8.7 million additional interactions in eight hours if
sustained; historical 379/second remains an estimate, not the current measured
rate. The first nominal 250k checkpoint was not yet due at this observation.

## Local controls and artifacts

- Status: `runs/wr_pure_continuous_20260904_062051/status.json`.
- Manifest: `runs/wr_pure_continuous_20260904_062051/run_manifest.json`.
- Checkpoints: `checkpoints/wr_pure_continuous_20260904_062051/`.
- Replay index: `runs/wr_pure_continuous_20260904_062051/replays.jsonl`.
- Replays: `artifacts/replays/wr_pure_continuous_20260904_062051/`.
- TensorBoard: `tensorboard/wr_pure_continuous_20260904_062051_0/`.
- Process logs: `artifacts/logs/continuous_launcher/` with the run-name prefix.

Double-click `stop-training.cmd` in the repository root for an independent
stop-and-save request, or ask Codex to stop training. Wait for `status.json` to
say `stopped` and the final checkpoint to be published; do not force-kill Python
or close the game during the save. A crash can still lose work since the most
recent completed checkpoint. No automatic rollback/restart merges histories.

Switching off the monitor is distinct from sleeping the PC. Keep Windows and
the game running. Monitor-off behavior has not yet been separately verified on
this display; a display disconnect or graphics-device change remains possible.

## Notable moments

The Windows UI tool could read the launcher but repeatedly rejected its returned
element IDs, and screenshot capture failed. The owner opened A01 manually; the
trainer itself then used only the existing live bridge, without UI input.

An offline exact-comparison test matched the continuous loop's policy updates
against stock PPO. A Windows-only file-flush failure was caught by the stop/save
test: the archive was reopened read-only before fsync. Reopening with write
access fixed it, and the saved model reloaded with optimizer state intact.

This launch does not prove unattended overnight uptime or beneficial slide
discovery. Those require subsequent artifacts and direct-live SimState review;
selected input replays alone remain video evidence, not mechanics calibration.
