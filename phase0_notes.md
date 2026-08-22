# Phase 0 notes

Last updated: 2026-08-22

## Local audit

| Item | Status | Evidence |
| --- | --- | --- |
| TrackMania Nations Forever installed | Confirmed | Steam app `11020`, build `9531569`, installed at `C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever` |
| Game launches and runs | Confirmed | The patched Steam installation launches through the pinned TMLoader profile in windowed mode |
| TrackMania ModLoader installed | Confirmed | TMLoader `1.0.1` at `C:\Users\Vardhan\AppData\Local\TMLoader\TMLoader.exe` |
| TMInterface installed/enabled | Confirmed statically | Active `default` profile pins `2.2.1`; installed `TMInterface.dll` SHA-256 is `C986CA9BC1F8FD208FCD59DA7A1BECE8386BA0ACD7E3FF20D3E2F4F9404D027B`. Runtime title/version still needs a live launch check |
| TMInterface Python bridge present | Confirmed | Installed at `C:\Users\Vardhan\Documents\TMInterface\Plugins\python_link.as`; SHA-256 matches the vendored bridge |
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
- Vendored SHA-256: `A47CCE075B3020BE9234C95135A170A58C1B950811B69963D63831813DC6366E`; it differs only by one final LF byte added by repository line-ending normalization.
- License: GNU GPL v3.0, preserved in `vendor/tminterface/LICENSE`.
- Network exposure: the audited bridge binds to `127.0.0.1`; Phase 0 uses port `8478`.

## Python environment

- Runtime: CPython `3.11.9` (64-bit), installed by the official Python install manager `26.3`.
- Project environment: `.venv` (excluded from Git).
- Verified imports: Stable-Baselines3 `2.9.0`, Gymnasium `1.3.0`, NumPy `2.4.6`, TensorBoard `2.21.0`, PyTorch `2.13.0+cpu`, and the TMInterface protocol client `1.0.2`.
- The Python manager's signed online index resolves `3.11` on Windows to `3.11.9`. Newer 3.11 security releases are source-only, so `3.11.9` remains the current official prebuilt Windows runtime in that line.
- The installed PyTorch wheel is CPU-only (`torch.cuda.is_available() == False`). Phase 0 does not require GPU training; revisit the PyTorch build before any later workload that would materially benefit from CUDA.

The owner approved the modern integration on 2026-08-21. The decision and version verification are documented in `docs/decisions/0001-tminterface-bridge.md`.

## Exit criteria

- [ ] Raw telemetry is reliable over a full manual lap, with a captured sample reviewed for changing position, velocity, orientation, and speed and no NaN/stale values.
- [ ] Scripted accelerate and steer inputs move the car as expected.
- [ ] Accelerated game speed works with telemetry and input.
- [ ] A short, low-complexity training track is selected and the reason is documented.
- [x] Python environment is installed and the Phase 0 dependency set is pinned.

## Next hands-on checks

1. Launch the pinned TMLoader `default` profile with TMInterface `custom_port` set to `8478`.
2. Confirm the running TMInterface title reports `2.2.1` and record the game version shown at runtime.
3. Run separate telemetry and scripted-input probes.

## Track choice

Pending human selection. No track has been chosen silently.

## Gotchas

- The Steam executable exposes no useful Windows file-version metadata. The patched binary was therefore verified by cryptographic hash and its embedded `2.11.26` string.
- `Nadeo.ini` still reports `2.11.16` after the official compatibility update, so it is not a reliable patch-level check.
- The ModLoader catalog and the standalone TMInterface download do not expose the same latest version; this project should use one recorded, reproducible ModLoader profile.
- TMLoader needed a one-time game-location setting and its official install control before the command-line profile launch would work. The original profile is preserved as `default.before-tminterface.yaml` in TMLoader's profile directory.
