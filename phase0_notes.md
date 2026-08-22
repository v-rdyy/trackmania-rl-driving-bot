# Phase 0 notes

Last updated: 2026-08-22

## Local audit

| Item | Status | Evidence |
| --- | --- | --- |
| TrackMania Nations Forever installed | Confirmed | Steam app `11020`, build `9531569`, installed at `C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever` |
| Game launches and runs | Confirmed | The patched Steam installation launches through the pinned TMLoader profile in windowed mode |
| TrackMania ModLoader installed | Confirmed | TMLoader `1.0.1` at `C:\Users\Vardhan\AppData\Local\TMLoader\TMLoader.exe` |
| TMInterface installed/enabled | Confirmed live | Runtime title is `TrackMania Modded Forever (2.12.0) [default]: TMInterface (2.2.1), CoreMod (1.0.11)`; installed `TMInterface.dll` SHA-256 is `C986CA9BC1F8FD208FCD59DA7A1BECE8386BA0ACD7E3FF20D3E2F4F9404D027B` |
| TMInterface Python bridge present | Confirmed live | Installed at `C:\Users\Vardhan\Documents\TMInterface\Plugins\python_link.as`; it listens only on `127.0.0.1:8478`, accepts the Python client, completes the connect handshake, executes commands, and accepts a clean reconnect |
| Python 3.10/3.11 available | Confirmed | Official Python install manager `26.3` installed Python `3.11.9`; project `.venv` imports all pinned Phase 0 packages successfully |

## Compatibility research

- The upstream `TMInterfaceClientPython` project says its native server API works only with TMInterface versions below 2.0, recommends TMInterface 1.4.3 for that API, and labels the Python client legacy and unmaintained.
- Current TMInterface uses an AngelScript plugin API. On 2026-08-21, the official TMInterface changelog and the locally installed TMLoader catalog both identified 2.2.1 as the current release.
- The actively developed `dersiwi/trackmania-gym` project uses current TMInterface with an AngelScript `python_link.as` plugin that exposes a TCP bridge compatible with the older Python protocol. Its repository was updated in July 2026 and documents telemetry plus input injection.
- Stable-Baselines3 2.9.0 is the current stable release and requires Python 3.10 or newer. Gymnasium 1.3.0 also requires Python 3.10 or newer.

## Steam compatibility patch

- TMInterface's official installation guide requires the Steam copy of TMNF to be updated from `2.11.16` to `2.11.26` before using TMLoader/TMInterface.
- The official update installer was downloaded from `files2.trackmaniaforever.com`; its SHA-256 is `F410BA92219F993AB224A4A672D0637649E2D5D36D003C9F1CE7F84F889FC9C5`.
- The installer payload was extracted and audited before installation. It contained 315 files: 19 existing files to replace and 296 new files.
- All replaced files were copied to `C:\Users\Vardhan\Documents\TMNF-backup-pre-2.11.26`, together with `patch-manifest.csv` recording the old and new hashes. The backup is sufficient to restore overwritten files; the manifest also identifies newly added files.
- Every installed file was checked against the extracted payload. The patched `TmForever.exe` SHA-256 is `4B6A7B31D86766409E94101F1256CD61DFFECB23EA497A40B6617506B2CED2D4` and contains the embedded version string `2.11.26`.
- `scripts/apply_tmforever_21126_patch.ps1` constrains the destination to the known Steam directory, pins the patched executable hash, writes a rollback manifest, and re-verifies every installed file.

## TMInterface bridge provenance

- Source repository: `dersiwi/trackmania-gym`.
- Source commit: `1d066ee742e736f6388818df2e07a4a89329f598`.
- Upstream `python_link.as` SHA-256: `63305AB927B0D199DA1DB7D011F7720C5F6397BF93C2EE2102C5AEC1CD224365`.
- Initial vendored SHA-256: `A47CCE075B3020BE9234C95135A170A58C1B950811B69963D63831813DC6366E`; at that point it differed only by one final LF byte added by repository line-ending normalization.
- Phase 0 hardened vendored SHA-256: `C815698A027D0BD7959E4D216051958EE828C930E3435BAF5126C91C874C3C5B`.
- Current Phase 1 vendored SHA-256: `7F9C8F839DB462E8102431E4AE82AC7C10DC40788C73020097C45377207FF716`; it preserves the Phase 0 protocol, adds Decision 0002's analog steer/gas message as ID `24`, suppresses redundant nested checkpoint/lap socket callbacks during snapshot rewind, and intercepts finishes in TMInterface's checkpoint callback before result UI can block the simulator.
- TMInterface 2.2.1 runtime hardening binds the default loopback port during plugin startup, reports listener failures, services graceful shutdown messages from menus, replaces an inactive client after an interrupted Python process, and executes Python-supplied console commands immediately through `CommandList`.
- License: GNU GPL v3.0, preserved in `vendor/tminterface/LICENSE`.
- Network exposure: the audited bridge binds to `127.0.0.1`; Phase 0 uses port `8478`.

## Live bridge verification

- TrackMania was launched through the pinned `default` profile in 640x480 windowed mode, and the TMInterface safety prompt was kept offline.
- The runtime loaded TMInterface `2.2.1`, CoreMod `1.0.11`, and the hardened `python_link.as` plugin without a bind failure.
- The plugin log recorded a Python client connecting from `127.0.0.1`, receiving its connect callback, executing configuration and map commands, disconnecting cleanly, and reconnecting.
- `A01-Race.Challenge.Gbx` was copied from the built-in Nations campaign to the user's local `Tracks\Challenges` directory with matching SHA-256 `F0A870809BE99DA2CB36AD5DF43A2CF63D8F74FE4AC3470ECAC68B9E97625DC3`. TMInterface now accepts and queues that map command.
- The global main menu does not transition into Solo mode in response to the bridge's queued map command. One manual `Play Solo` transition was needed; after that, A01 loaded and the full-lap capture completed.

## Manual-lap telemetry evidence

- The first full A01 manual lap produced 361 JSON Lines records in `artifacts/telemetry/phase0_manual_lap.jsonl`; the ignored local evidence file SHA-256 is `C8469209C0A91A1421F052DBF055A900C9A092E27AA8561C230D2A650CBDB0CC`.
- All position, velocity, rotation-matrix, and yaw/pitch/roll values are finite. Race times are monotonic from `-2600` through `33280` ms, with 360 unique timestamps across 361 samples.
- Position spans are approximately `[967.077, 104.026, 554.008]`, traced path length is `2206.540`, velocity magnitude ranges from `0.000` to `123.046`, and displayed speed ranges from `0` to `442`.
- Yaw/pitch/roll spans are approximately `[6.257191, 1.078489, 0.353209]`, and the maximum rotation-matrix change from the starting orientation is `2.828427`.
- A live post-lap bridge query returned `race_finished=True`, confirming the sample reaches a completed race rather than only a partial drive.

## Scripted-input evidence

- The input probe held acceleration for 2.5 seconds, then held acceleration plus left steering for 1.5 seconds while sampling the same bridge telemetry.
- It captured 41 finite driving samples in `artifacts/telemetry/phase0_input.jsonl`; the ignored local evidence file SHA-256 is `5C8903706E7EB25A0A4DE4B3BA2622B2FD38C6006CC9C6B4BE834DF7B68D9953`.
- The car moved `85.562` units, displayed speed rose from `0` to a maximum of `157`, and yaw changed by `2.147260` radians during the steering phase.
- `scripts/probe_input.py` rejects runs with less than 5 units of movement, a maximum displayed speed below 20, a steering yaw change below 0.05 radians, or nonfinite telemetry.

## Accelerated-time evidence

- A 4x warm-up advanced four in-game seconds in `0.953` wall seconds (`4.197x` effective), captured 41 finite telemetry samples, moved the car `111.227` units, and reached displayed speed `177`.
- The first exact-6x attempt began with the car wedged after the earlier steering probe. Its clock advanced two in-game seconds in `0.266` wall seconds, but the car moved only `0.789` units and reached speed `9`; the movement threshold correctly rejected that run instead of treating clock speed alone as success.
- After a clean manual restart, the exact-6x verification advanced four in-game seconds in `0.625` wall seconds (`6.400x` effective), captured 41 finite telemetry samples, moved the car `111.330` units, and reached displayed speed `177` while scripted acceleration remained active.
- The successful local evidence file is `artifacts/telemetry/phase0_accelerated_6x.jsonl`, with SHA-256 `58B0AFB3379EF280634098BA1D3AD85DC7C8F2D484C27FAFF66B53F867E55C35`.
- `scripts/probe_accelerated_time.py` restores neutral input and 1x speed in a `finally` block even when validation fails.

## Python environment

- Runtime: CPython `3.11.9` (64-bit), installed by the official Python install manager `26.3`.
- Project environment: `.venv` (excluded from Git).
- Verified imports: Stable-Baselines3 `2.9.0`, Gymnasium `1.3.0`, NumPy `2.4.6`, TensorBoard `2.21.0`, PyTorch `2.13.0+cpu`, and the TMInterface protocol client `1.0.2`.
- The Python manager's signed online index resolves `3.11` on Windows to `3.11.9`. Newer 3.11 security releases are source-only, so `3.11.9` remains the current official prebuilt Windows runtime in that line.
- The installed PyTorch wheel is CPU-only (`torch.cuda.is_available() == False`). Phase 0 does not require GPU training; revisit the PyTorch build before any later workload that would materially benefit from CUDA.

The owner approved the modern integration on 2026-08-21. The decision and version verification are documented in `docs/decisions/0001-tminterface-bridge.md`.

## Exit criteria

- [x] Raw telemetry is reliable over a full manual lap, with a captured sample reviewed for changing position, velocity, orientation, and speed and no NaN/stale values.
- [x] Scripted accelerate and steer inputs move the car as expected.
- [x] Accelerated game speed works with telemetry and input at the target 6x multiplier.
- [ ] A short, low-complexity training track is selected and the reason is documented.
- [x] Python environment is installed and the Phase 0 dependency set is pinned.

## Phase 1 handoff

All Phase 0 exit criteria are satisfied. On 2026-08-22, the owner selected continuous steer/throttle/brake control for Phase 1 and required raw per-step action logging with finite/range validation. The contract is recorded in `docs/decisions/0002-continuous-action-space.md`.

## Track choice

`A01-Race` is the Phase 0 track because it is short, flat, built into the Nations campaign, and simple enough to diagnose telemetry and input independently of difficult driving. The first manual lap confirmed that the local copy loads, drives, and completes correctly.

## Gotchas

- The Steam executable exposes no useful Windows file-version metadata. The patched binary was therefore verified by cryptographic hash and its embedded `2.11.26` string.
- `Nadeo.ini` still reports `2.11.16` after the official compatibility update, so it is not a reliable patch-level check.
- The ModLoader catalog and the standalone TMInterface download do not expose the same latest version; this project should use one recorded, reproducible ModLoader profile.
- TMLoader needed a one-time game-location setting and its official install control before the command-line profile launch would work. The original profile is preserved as `default.before-tminterface.yaml` in TMLoader's profile directory.
- The TMInterface safety prompt appears on startup; Phase 0 uses `Stay offline` and does not authorize online play or leaderboard submission.
- The plugin registers `custom_port` after TMLoader processes its startup config string, so setting that variable in the profile produced a harmless `Unknown variable` warning. The Phase 0 profile now relies on the audited bridge's pinned default `8478` instead.
- TMInterface's `GiveUp()` respawns the car but does not reset the absolute race clock in this post-finish flow. Input probes use elapsed time from their first race callback rather than assuming race time begins at zero.
