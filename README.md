# Trackmania RL Driving Bot

This repository follows [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md). **Phases 0 and 1
are complete.** The live Gymnasium loop and bounded PPO/TensorBoard smoke test
have passed. Phase 2 reward v1-v4 experiments are complete: sparse finish-only reward
produced zero finishes and a flat curve, while dense speed reward produced a
30-35% deterministic finish rate but an oscillatory line and unreliable final
checkpoint/hoop approach. Reward v3 trained a reliability-focused progress
policy; reward v4 retained its 20/20 deterministic finish rate while improving
the TMNF race-clock best from `28.030s` to `24.900s`.
Development history follows the rules in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Current status

- TrackMania Nations Forever is installed through Steam.
- The official TMNF 2.11.26 compatibility update is applied with a verified rollback backup.
- TrackMania ModLoader is installed.
- TMInterface 2.2.1 is installed and pinned in the active ModLoader profile.
- Python 3.11.9 and the pinned Phase 0 dependencies are installed in `.venv`.
- The audited `python_link.as` bridge is live on `127.0.0.1:8478`; its Python handshake, immediate command path, graceful disconnect, and reconnect are verified.
- Full-lap A01 telemetry is verified with finite, changing position, velocity, orientation, and speed data.
- Scripted acceleration and steering are verified from resulting motion telemetry.
- The exact 6x target is verified with telemetry and scripted acceleration active; the measured effective rate was 6.400x.
- Decision 0002's normalized continuous controls are verified end to end. A
  repeated direction probe corrected TMNF Gas polarity to negative-forward and
  positive-backward; V0/V1/V2 retain their original behavior through an explicit
  legacy compatibility mode.
- The A01 driven reference line is provenance-pinned and resampled into 443 fixed-spacing points per Decision 0003.
- The documented 26-value engineered observation is finite and progress-consistent across the full manual lap.
- The owner's actual A01 human PB is `24.5s`. The `33.280s` manual telemetry lap
  was intentionally slow and cautious for reference-path capture, not a
  performance attempt or evidence that an agent is near human pace.
- TMInterface snapshot rewind is live-verified as the reliable Phase 1 episode-reset primitive.
- The Gymnasium wrapper completed 20 consecutive live episodes and 100 finite
  steps with deterministic resets and per-step raw-action auditing.
- A 2,048-step PPO smoke run completed at roughly 59 environment steps per
  second, wrote verified TensorBoard reward/episode-length metrics, and audited
  every policy and environment action without hidden clipping per Decision 0005.
- Reward v1 completed 501,200 PPO timesteps at 100x simulation speed with zero
  finishes. Its 20-episode deterministic evaluation repeatedly reversed hard
  left and fell from the A01 start, partially supporting the sparse-reward
  hypothesis while exposing an unexpected harmful behavior change.
- Reward v2 completed 1,001,120 PPO timesteps at 100x. Its 20-episode
  deterministic evaluation finished seven times, timed out 13 times, and showed
  no in-place loop exploit. Visible review instead found left/right oscillation
  after the first major turn and a right-edge hoop collision after the second.
  A distinct fixed-metric 20-episode retrospective finished six times; all 20
  runs oscillated, 19 went upside down, and all 14 timeouts became stuck.
- Reward v3 completed `1,001,472` timesteps. Its original evaluator reported
  12/20 finishes because a low but recoverable final-jump path crossed an
  immediate vertical cutoff. After requiring stalled-motion confirmation, one
  unchanged-checkpoint re-evaluation finished 20/20 with zero terminal failures:
  TMNF race-clock best `28.030s`, mean `28.106s`, versus the owner's `24.5s` PB.
- Reward v4 continued from that pinned V3 model for `1,001,472` additional
  timesteps. Its 20-episode deterministic evaluation also finished 20/20, with
  best `24.900s`, mean `24.929s`, and worst `24.950s`: only `0.400s` off the PB.
- V4 fixed the observed final-hoop collision in the reviewed runs but did not
  solve precision. Oscillation remained 20/20, and maximum absolute lateral
  deviation increased from V3's `13.568` to `17.833` units. A precision-focused
  next reward is now evidence-backed but not yet approved.

Run the Phase 1 PPO smoke test while A01 is loaded. The wrapper automatically
respawns through TMInterface, observes the full pre-race countdown transition,
then validates and captures a deterministic start snapshot:

```powershell
.\.venv\Scripts\python.exe .\scripts\train_phase1_smoke.py
```

Preserve the staged V0-V2 checkpoint progression as compact TMInterface input
replays before selecting website clips:

```powershell
.\.venv\Scripts\python.exe .\scripts\capture_progress_replays.py
```

Round-trip representative saved inputs and compare their playback with the
original outcomes:

```powershell
.\.venv\Scripts\python.exe .\scripts\validate_progress_replays.py
```

The fixed matrix and replay/video evidence protocol are documented in
[`docs/video-evidence.md`](docs/video-evidence.md).

The checksum-pinned V2 evidence/source bundle used before V3 is documented in
[`docs/pre-v3-backup.md`](docs/pre-v3-backup.md). The completed V3 protocol,
results, detector correction, and hashes are in [`reward_v3.md`](reward_v3.md).
The completed V4 hypothesis, fixed formula, training result, V3 comparison, and
qualitative evidence are in [`reward_v4.md`](reward_v4.md).

Live V3/V4 evaluation and replay-inspection entry points now start the configured
`TmForever/default` ModLoader profile and load A01 themselves. Manual game/menu
setup is no longer required; `--reuse-game` is an explicit opt-in for attaching
to an already-running session.

Run the non-mutating local audit from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\phase0_preflight.ps1
```

Verified facts and pending hands-on checks are tracked in [`phase0_notes.md`](phase0_notes.md).
