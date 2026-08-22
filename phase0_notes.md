# Phase 0 notes

Last updated: 2026-08-21

## Local audit

| Item | Status | Evidence |
| --- | --- | --- |
| TrackMania Nations Forever installed | Confirmed | Steam app `11020`, build `9531569`, installed at `C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever` |
| Game launches and runs | Not yet confirmed | Steam reports no prior launch and no game process was running during the audit |
| TrackMania ModLoader installed | Confirmed | TMLoader `1.0.1` at `C:\Users\Vardhan\AppData\Local\TMLoader\TMLoader.exe` |
| TMInterface installed/enabled | Not yet confirmed | TMLoader catalog contains TMInterface through `2.2.1`, but the active `default` profile currently contains only `TmForever` |
| TMInterface Python bridge present | Missing | `C:\Users\Vardhan\Documents\TMInterface` and `Plugins\python_link.as` were absent |
| Python 3.10/3.11 available | Missing | `py` is absent; `python.exe` and `python3.exe` resolve only to Microsoft Store aliases |

## Compatibility research

- The upstream `TMInterfaceClientPython` project says its native server API works only with TMInterface versions below 2.0, recommends TMInterface 1.4.3 for that API, and labels the Python client legacy and unmaintained.
- Current TMInterface uses an AngelScript plugin API. The locally installed TMLoader catalog offers TMInterface 2.2.1.
- The actively developed `dersiwi/trackmania-gym` project uses current TMInterface with an AngelScript `python_link.as` plugin that exposes a TCP bridge compatible with the older Python protocol. Its repository was updated in July 2026 and documents telemetry plus input injection.
- Stable-Baselines3 2.9.0 is the current stable release and requires Python 3.10 or newer. Gymnasium 1.3.0 also requires Python 3.10 or newer.

The proposed integration is documented in `docs/decisions/0001-tminterface-bridge.md`. It is intentionally not installed or coded against until the owner approves the choice.

## Exit criteria

- [ ] Raw telemetry is reliable over a full manual lap, with a captured sample reviewed for changing position, velocity, orientation, and speed and no NaN/stale values.
- [ ] Scripted accelerate and steer inputs move the car as expected.
- [ ] Accelerated game speed works with telemetry and input.
- [ ] A short, low-complexity training track is selected and the reason is documented.
- [ ] Python environment is installed and the final dependency set is pinned.

## Next hands-on checks

1. Approve the TMInterface bridge decision.
2. Enable the chosen TMInterface version in TMLoader and launch the game through that profile.
3. Confirm the launcher/game version shown by the running game and record it here.
4. Install Python and create `.venv` from `requirements.txt`.
5. Add the approved bridge and run separate telemetry and scripted-input probes.

## Track choice

Pending human selection. No track has been chosen silently.

## Gotchas

- The Steam executable exposes no useful Windows file-version metadata, so compatibility must be verified from the running launcher/game rather than inferred from the executable properties.
- The ModLoader catalog and the standalone TMInterface download do not expose the same latest version; this project should use one recorded, reproducible ModLoader profile.

