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
`capture_catalog.json` inventories every successful and failed take across all
invocations, including checkpoint/manifest hashes and episode outcomes.

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
Use `--catalog-only` to rebuild the aggregate index without connecting to the
game.
Before every checkpoint stage, the collector focuses the verified `TmForever`
window and sends TrackMania's Delete restart key. This avoids an intermittent
handoff where a new bridge client could not start the pre-race countdown after
the preceding stage's intercepted finish. Snapshot rewind remains responsible
for repeated episodes within one stage.

## Replay round-trip verification

`scripts/validate_progress_replays.py` checks that preserved input files are
actually reusable rather than trusting successful writes. It verifies the local
replay hash against its capture manifest and the exact copy in TMInterface's
Scripts directory, loads the raw input file, restarts from a full countdown,
records 100 ms telemetry, and compares outcome, elapsed time, and maximum path
progress with the original capture. TMInterface's official command reference
defines `load [filename.txt]` as loading inputs from the configured Scripts
folder: https://donadigo.com/tminterface/commands.

The representative V0/V1/V2 gate passed in one uninterrupted sequence:

| Case | Original / playback outcome | Time delta | Maximum-progress delta |
| --- | --- | ---: | ---: |
| V0 Phase 1, 2,048 steps | timeout / timeout | `0 ms` | `0.000` |
| V1 final, episode 1 | fall / fall | `+100 ms` | `0.000` |
| V2 final, episode 4 | finish / finish | `+100 ms` | `0.000` |
| V2 final, episode 1 | timeout / timeout | `0 ms` | `-0.001` |

The one-step timing differences are within the fixed 100 ms callback period.
The raw replay files contain TMInterface `steer` and signed `gas` commands, so
their playback does not depend on the later correction to Python's pedal labels.
The ignored representative summary SHA-256 is
`F1889C965FBA93F08ACA03F9343FB8A86DD4D85F6490AAE2FE9B83BE7668607E`.

Several earlier attempts reproduced an intermittent handoff after an intercepted
finish: the next connection remained frozen at race time `30100` despite bridge
`GiveUp` requests, so no next-replay telemetry was accepted. Stronger synthetic
Delete input and a two-second connection delay were individually insufficient.
The fix is architectural: the validator now holds one live bridge connection,
captures one clean A01 start snapshot, loads each input file without overriding
its controls, and rewinds that same snapshot between cases. The requested live
nine-replay V2 batch then passed all five finishes and four timeouts, including
repeated finish-to-finish and finish-to-timeout transitions. Outcome, elapsed
time, and maximum progress all matched; finish playback differed by one 100 ms
callback step and timeout playback by 0 ms. The ignored nine-case summary SHA-256
is `C8A52200B98B25A74E10B4750B9492E2E1DEBDC076B7FB6F67FD5F517600DE21`.

The fixed Decision 0008 precision metrics were then applied to every V2 replay,
not only selected failures. Eight of nine oscillated, five went upside down, and
four became stuck. All five final-model replays oscillated and went upside down;
four were stuck for 12.9 to 15.4 seconds, while the one finish still spent 2.0
seconds inverted. The ignored precision-summary SHA-256 is
`86AD9C5F50716CFAE3E917ADFBE5C642997BE8C8EB459AC891A22AE4E2F9E9FD`.

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
