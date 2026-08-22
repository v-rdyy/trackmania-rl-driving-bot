# Phase 0 notes

Last updated: 2026-08-21

## Local audit

| Item | Status | Evidence |
| --- | --- | --- |
| TrackMania Nations Forever installed | Confirmed | Steam app `11020`, build `9531569`, installed at `C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever` |
| Game launches and runs | Not yet confirmed | Steam reports no prior launch and no game process was running during the audit |
| TrackMania ModLoader installed | Confirmed | TMLoader `1.0.1` at `C:\Users\Vardhan\AppData\Local\TMLoader\TMLoader.exe` |
| TMInterface installed/enabled | Confirmed statically | Active `default` profile pins `2.2.1`; installed `TMInterface.dll` SHA-256 is `C986CA9BC1F8FD208FCD59DA7A1BECE8386BA0ACD7E3FF20D3E2F4F9404D027B`. Runtime title/version still needs a live launch check |
| TMInterface Python bridge present | Confirmed | Installed at `C:\Users\Vardhan\Documents\TMInterface\Plugins\python_link.as`; SHA-256 matches the vendored bridge |
| Python 3.10/3.11 available | Confirmed | Official Python install manager `26.3` installed Python `3.11.9`; project `.venv` imports all pinned Phase 0 packages successfully |

## Compatibility research

- The upstream `TMInterfaceClientPython` project says its native server API works only with TMInterface versions below 2.0, recommends TMInterface 1.4.3 for that API, and labels the Python client legacy and unmaintained.
- Current TMInterface uses an AngelScript plugin API. On 2026-08-21, the official TMInterface changelog and the locally installed TMLoader catalog both identified 2.2.1 as the current release.
- The actively developed `dersiwi/trackmania-gym` project uses current TMInterface with an AngelScript `python_link.as` plugin that exposes a TCP bridge compatible with the older Python protocol. Its repository was updated in July 2026 and documents telemetry plus input injection.
- Stable-Baselines3 2.9.0 is the current stable release and requires Python 3.10 or newer. Gymnasium 1.3.0 also requires Python 3.10 or newer.

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

- The Steam executable exposes no useful Windows file-version metadata, so compatibility must be verified from the running launcher/game rather than inferred from the executable properties.
- The ModLoader catalog and the standalone TMInterface download do not expose the same latest version; this project should use one recorded, reproducible ModLoader profile.
- TMLoader needed a one-time game-location setting and its official install control before the command-line profile launch would work. The original profile is preserved as `default.before-tminterface.yaml` in TMLoader's profile directory.
