# Phase 1 notes

Last updated: 2026-08-22

## Current status

- Decision 0002 defines continuous `[steer, throttle, brake]` actions and fail-fast raw-action logging.
- The live bridge applies true analog `Steer` and `Gas` values without changing the Phase 0 digital protocol.
- Decision 0003 defines the provenance-pinned A01 driven reference path.
- The engineered 26-value observation builder is implemented and validated against the full manual lap.
- Snapshot rewind and 100 ms synchronous callback control are verified live for episode resets.
- The Gymnasium `reset()`/`step()` wrapper and PPO smoke test have not yet been implemented.

## Action contract

The action space is `Box(low=[-1, 0, 0], high=[1, 1, 1], dtype=float32)` in `[steer, throttle, brake]` order. The bridge maps steer directly to signed analog `Steer` and maps `throttle - brake` to signed analog `Gas`, both scaled by `65536`. Invalid or nonfinite values fail before transport.

## Observation contract

The observation is a 26-value `float32` vector:

| Indices | Values | Normalization |
| --- | --- | --- |
| `0` | displayed speed | `/ 1000` |
| `1:4` | car-frame velocity `[forward, right, up]` | `/ 100` |
| `4` | signed heading error | `/ pi` |
| `5` | signed lateral offset, positive to car-right | `/ 50` |
| `6:26` | ten `[forward, right]` reference-path points at 10–100 units ahead | `/ 100` |

TMInterface rotation columns are car `[right, up, forward]`; the transpose converts world vectors to the car frame. Reference projection, tangent heading, and lateral offset use X/Z horizontally. Y is elevation and remains available to the velocity transform and source reference data.

## Full-lap observation audit

- Input: all 361 verified Phase 0 manual-lap records.
- Output shape: `(361, 26)`; every value is finite.
- Projected progress: `0.0` through `2206.528335`, with zero backsteps larger than one unit.
- Maximum absolute reference-line offset: `0.279431` units.
- Maximum absolute heading error: `0.489704` radians (`0.155878` after normalization).
- Maximum normalized magnitudes: speed `0.442`, car-frame velocity `1.230450`, lateral offset `0.005589`, look-ahead coordinates `1.000000`.

## Temporary smoke-test reward

Decision 0004 records the owner-approved disposable reward: `display_speed / 1000.0` each step, minus `1.0` on timeout/off-track truncation, with no finish bonus. This infrastructure reward is not one of the later pre-registered reward experiments.

## Reset primitive

The wrapper will capture one known start snapshot and restore its raw TMInterface simulation bytes on every reset. A live round-trip moved the car `17.554` units under analog throttle, then restored the captured position with `0.000000` measured error. The next callback advanced from captured race time `7343600` to `7343700`, matching the configured 100 ms game-time step period.
