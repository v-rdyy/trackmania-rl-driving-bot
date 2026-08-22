# Trackmania RL Driving Bot

This repository follows [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md). **Phase 0: environment and tooling verification is complete.** Phase 1 has started with the owner-approved continuous action contract in [`Decision 0002`](docs/decisions/0002-continuous-action-space.md). Development history follows the rules in [`CONTRIBUTING.md`](CONTRIBUTING.md).

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
- Decision 0002's normalized continuous controls are verified end to end against TMInterface's analog steer/gas inputs.
- The A01 driven reference line is provenance-pinned and resampled into 443 fixed-spacing points per Decision 0003.
- The documented 26-value engineered observation is finite and progress-consistent across the full manual lap.

Run the non-mutating local audit from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\phase0_preflight.ps1
```

Verified facts and pending hands-on checks are tracked in [`phase0_notes.md`](phase0_notes.md).
