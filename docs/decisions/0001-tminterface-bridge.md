# Decision 0001: TMInterface-to-Python bridge

Status: Proposed — human approval required

Date: 2026-08-21

## Context

The project requires live telemetry and deterministic external control from Python. The original TMInterface Python client no longer connects directly to current TMInterface 2.x releases; its upstream repository labels that direct API legacy and says it requires TMInterface 1.4.3.

A current TMNF Gymnasium implementation demonstrates a different route: run modern TMInterface through TMLoader, load an AngelScript `python_link.as` plugin, and communicate over its local TCP protocol from Python. This retains the modern TMInterface plugin model while exposing the simulation state and input controls needed by this project.

## Proposed decision

Use:

- TMLoader with a pinned TMInterface 2.2.1 profile;
- an audited and version-pinned `python_link.as` TCP bridge derived from a maintained public implementation;
- the Python protocol client only as a transport/protocol dependency, wrapped behind this project's own small adapter;
- loopback-only networking for the bridge.

Before adoption, confirm the bridge source and license permit the intended reuse. Record its exact commit and file hash in `phase0_notes.md`.

## Why this is recommended

- It avoids deliberately downgrading the game integration to the unmaintained TMInterface 1.4.3 stack.
- It has recent, public evidence of use for Gymnasium-compatible TMNF control and telemetry.
- It preserves the project's intended architecture: Python owns observations, rewards, Gymnasium, and PPO; the game plugin only transports state and inputs.

## Alternative

Install TMInterface 1.4.3 and use `tminterface==1.0.2` directly. This is simpler and historically proven, but upstream explicitly calls it legacy and unmaintained. Choose it only if the modern bridge fails Phase 0 telemetry or input tests.

## Approval needed

The owner must approve the proposed modern bridge before implementation or dependency installation proceeds.

