# Phase 0 notes

Last updated: 2026-08-21

## Local audit

| Item | Status | Evidence |
| --- | --- | --- |
| TrackMania Nations Forever installed | Confirmed | Steam app `11020`, build `9531569`, installed at `C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever` |
| Game launches and runs | Not yet confirmed | Steam reports no prior launch and no game process was running during the audit |
| TrackMania ModLoader installed | Confirmed | TMLoader `1.0.1` at `C:\Users\Vardhan\AppData\Local\TMLoader\TMLoader.exe` |
| TMInterface installed/enabled | Not yet confirmed | Official changelog and TMLoader catalog both identify `2.2.1` as current, but the active `default` profile currently contains only `TmForever` |
| TMInterface Python bridge present | Missing | `C:\Users\Vardhan\Documents\TMInterface` and `Plugins\python_link.as` were absent |
| Python 3.10/3.11 available | Confirmed | Official Python install manager `26.3` installed Python `3.11.9`; project `.venv` imports all pinned Phase 0 packages successfully |

## Compatibility research

- The upstream `TMInterfaceClientPython` project says its native server API works only with TMInterface versions below 2.0, recommends TMInterface 1.4.3 for that API, and labels the Python client legacy and unmaintained.
- Current TMInterface uses an AngelScript plugin API. On 2026-08-21, the official TMInterface changelog and the locally installed TMLoader catalog both identified 2.2.1 as the current release.
- The actively developed `dersiwi/trackmania-gym` project uses current TMInterface with an AngelScript `python_link.as` plugin that exposes a TCP bridge compatible with the older Python protocol. Its repository was updated in July 2026 and documents telemetry plus input injection.
- Stable-Baselines3 2.9.0 is the current stable release and requires Python 3.10 or newer. Gymnasium 1.3.0 also requires Python 3.10 or newer.

## Python environment

- Runtime: CPython `3.11.9` (64-bit), installed by the official Python install manager `26.3`.
- Project environment: `.venv` (excluded from Git).
- Verified imports: Stable-Baselines3 `2.9.0`, Gymnasium `1.3.0`, NumPy `2.4.6`, TensorBoard `2.21.0`, and PyTorch `2.13.0+cpu`.
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

1. Enable TMInterface 2.2.1 in TMLoader and launch the game through that profile.
2. Confirm the launcher/game version shown by the running game and record it here.
3. Add the approved bridge and run separate telemetry and scripted-input probes.

## Track choice

Pending human selection. No track has been chosen silently.

## Gotchas

- The Steam executable exposes no useful Windows file-version metadata, so compatibility must be verified from the running launcher/game rather than inferred from the executable properties.
- The ModLoader catalog and the standalone TMInterface download do not expose the same latest version; this project should use one recorded, reproducible ModLoader profile.
