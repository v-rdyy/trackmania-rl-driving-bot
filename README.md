# Trackmania RL Driving Bot

This repository follows [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md). Work is currently limited to **Phase 0: environment and tooling verification**. No Gymnasium wrapper or reinforcement-learning training code should be added until live telemetry, scripted input, and accelerated-time operation are proven. Development history follows the rules in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Current status

- TrackMania Nations Forever is installed through Steam.
- The official TMNF 2.11.26 compatibility update is applied with a verified rollback backup.
- TrackMania ModLoader is installed.
- TMInterface 2.2.1 is installed and pinned in the active ModLoader profile.
- Python 3.11.9 and the pinned Phase 0 dependencies are installed in `.venv`.
- The audited `python_link.as` loopback bridge and Python protocol client are installed; live game, telemetry, input, and accelerated-time verification are next.

Run the non-mutating local audit from PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\phase0_preflight.ps1
```

Verified facts and pending hands-on checks are tracked in [`phase0_notes.md`](phase0_notes.md).
