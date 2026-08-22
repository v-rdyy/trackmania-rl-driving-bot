# Checkpoint progression video evidence

The website footage begins as replay evidence, not a hand-picked highlight reel.
`config/video_progression.json` fixes the checkpoint matrix, seed, deterministic
policy setting, and 6x simulation speed. The primary capture tool uses
TMInterface's `recover_inputs` command after every episode, preserving the exact
input stream even when the normal finish flow is suppressed. Its take manifest
records model, input-replay, and raw-action-log SHA-256 hashes plus the telemetry
outcome.

Replay output is local under `artifacts/replays/progression/`; later H.264 renders
are under `artifacts/videos/progression/`. Each rerun creates `take_001`,
`take_002`, and so on, and neither capture tool replaces an earlier take.

## Progression matrix

- V0 / Phase 1 integration: reconstructed seed-42 untrained reference and the
  real 2,048-step smoke checkpoint. No intermediate V0 checkpoint ever existed,
  so the project must not imply a finer historical progression.
- V1 sparse finish reward: real 50,000-step, 250,000-step, and final 501,200-step
  checkpoints. Two final episodes preserve simulator variation in the repeated
  backward/fall behavior.
- V2 dense speed reward: real 50,000-step, 250,000-step, 500,000-step,
  750,000-step, and final 1,001,120-step checkpoints. Five final episodes make it
  likely that both a finish and the observed timeout/jump failure are retained;
  the manifest, not manual selection, reports the actual mix.

The zero-step V0 model is explicitly marked as a reconstructed untrained
reference created from the documented seed and policy architecture. Every other
stage loads a model saved during the original experiment.

## First live-capture failure

The first V2 50,000-step smoke take coupled `ImageGrab` and H.264 encoding to the
same Python process as the synchronous TMInterface client. The first full-screen
grab held that process long enough for the bridge to abort with Windows socket
error `10053` before one action completed. The partial MP4 and empty action log
remain in `v2_speed_050000/take_001`; they are failed evidence, not a website
clip. Capture now runs in a spawned process so screen acquisition cannot hold the
control loop's Python interpreter lock.

## Replay-first capture

With A01 Race loaded and TMInterface enabled:

```powershell
.\.venv\Scripts\python.exe .\scripts\capture_progress_replays.py
```

The resulting `.txt` files are TMInterface input replays. Keeping them is faster
and smaller than recording every checkpoint directly to video, and they can be
loaded later through TMInterface's `load` command for consistent rendering.
Before every checkpoint stage, the collector focuses the verified `TmForever`
window and sends TrackMania's Delete restart key. This avoids an intermittent
handoff where a new bridge client could not start the pre-race countdown after
the preceding stage's intercepted finish. Snapshot rewind remains responsible
for repeated episodes within one stage.

## Optional direct video capture

With A01 Race loaded and TMInterface enabled:

```powershell
.\.venv\Scripts\python.exe .\scripts\capture_progress_videos.py
```

Use `--stage <stage-id>` to capture only one matrix entry. Use `--dry-run` to
list the selected entries without connecting to TrackMania or creating files.
The tool brings the TrackMania window forward, captures only its client area,
encodes H.264/yuv420p with fast-start metadata for browsers, and decodes a frame
from every completed MP4 before accepting it.
