# Decision 0001: TMInterface-to-Python bridge

Status: Approved

Date: 2026-08-21

Approved by the project owner on 2026-08-21.

## Context

The project requires live telemetry and deterministic external control from Python. The original TMInterface Python client no longer connects directly to current TMInterface 2.x releases; its upstream repository labels that direct API legacy and says it requires TMInterface 1.4.3.

A current TMNF Gymnasium implementation demonstrates a different route: run modern TMInterface through TMLoader, load an AngelScript `python_link.as` plugin, and communicate over its local TCP protocol from Python. This retains the modern TMInterface plugin model while exposing the simulation state and input controls needed by this project.

## Decision

Use:

- TMLoader with a pinned TMInterface 2.2.1 profile;
- an audited and version-pinned `python_link.as` TCP bridge derived from a maintained public implementation;
- the Python protocol client only as a transport/protocol dependency, wrapped behind this project's own small adapter;
- loopback-only networking for the bridge.

Before vendoring the bridge, confirm its source and license permit the intended reuse. Record its exact upstream commit and file hash in `phase0_notes.md`.

## Version verification

On 2026-08-21, the official TMInterface changelog listed `2.2.1` (released 2026-01-05) as the latest release. The local TMLoader catalog independently contains the same version. On 2026-08-22, the launched game confirmed the active runtime as `TrackMania Modded Forever (2.12.0) [default]: TMInterface (2.2.1), CoreMod (1.0.11)`, so Decision 0001 now reflects the version actually installed and running rather than only the catalog target.

Sources:

- https://donadigo.com/tminterface/
- https://donadigo.com/tminterface/installation

## Why this is recommended

- It avoids deliberately downgrading the game integration to the unmaintained TMInterface 1.4.3 stack.
- It has recent, public evidence of use for Gymnasium-compatible TMNF control and telemetry.
- It preserves the project's intended architecture: Python owns observations, rewards, Gymnasium, and PPO; the game plugin only transports state and inputs.

## Alternative

Install TMInterface 1.4.3 and use `tminterface==1.0.2` directly. This is simpler and historically proven, but upstream explicitly calls it legacy and unmaintained. Choose it only if the modern bridge fails Phase 0 telemetry or input tests.

## Approval record

The owner approved the modern TMInterface 2.2.1 plus `python_link.as` bridge after reviewing the maintained-versus-legacy tradeoff. Phase 0 implementation may proceed, subject to the live telemetry, input, and accelerated-game exit checks.
