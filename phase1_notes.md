# Phase 1 notes

Last updated: 2026-08-22

## Current status

- Decision 0002 defines continuous `[steer, throttle, brake]` actions and fail-fast raw-action logging.
- The live bridge applies true analog `Steer` and `Gas` values without changing the Phase 0 digital protocol.
- Decision 0003 defines the provenance-pinned A01 driven reference path.
- The engineered 26-value observation builder is implemented and validated against the full manual lap.
- Snapshot rewind and 100 ms synchronous callback control are verified live for episode resets.
- The Gymnasium `reset()`/`step()` wrapper passed its 20-episode live reliability check.
- The PPO and TensorBoard smoke test is still pending.

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

The wrapper captures one known start snapshot and restores its raw TMInterface simulation bytes on every reset. Before capture it refuses states above 5 displayed speed, beyond 25 path-progress units, or more than 10 lateral units from the reference path. A live round-trip moved the car `17.554` units under analog throttle, then restored the captured position with `0.000000` measured error. The next callback advanced from captured race time `7343600` to `7343700`, matching the configured 100 ms game-time step period.

The reliability smoke test ran 20 consecutive live episodes with a deliberately
short 500 ms timeout, producing 100 finite `reset()`/`step()` interactions. All
100 raw `[steer, throttle, brake]` actions were finite and in range before the
bridge applied steer `0` and gas `22938`. Every episode timed out normally after
five steps. All 20 post-rewind reset observations were identical, with maximum
absolute delta `0.000000000`. The ignored local action log SHA-256 is
`DA74F4A99E507E8EC0B393400453B25FFC2FBF5CA9FE0DB914A7217F5FF29C3D`; the
ignored summary SHA-256 is
`D0607E9FD2E02BE3F943892822E316BB28BB912733770569DB70FBFC5EA5B1E4`.

## Notable moments

- The first reliability run technically completed 20 episodes, but its raw
  action log showed that episode 0 accepted gas `22938` while remaining at speed
  zero for all five steps. Episodes after the first accelerated normally. The
  reset implementation revealed the lifecycle difference: the first episode
  returned the freshly captured state directly, while later episodes rewound the
  snapshot and advanced one synchronized neutral tick. Routing the first episode
  through the same rewind path removed the asymmetry; its speeds became 7, 16,
  23, 30, and 36, and all 20 reset observations became identical.
- Snapshot rewind was selected over issuing a normal respawn because the earlier
  live probe showed that `GiveUp()` did not reliably reset the absolute race
  clock. Raw snapshot restore is more tightly coupled to TMInterface state bytes,
  but it gives deterministic episode starts and fast resets at 6x simulation.
- The 500 ms reliability timeout is a test override chosen to exercise 20 reset
  boundaries quickly. The owner-approved production safety limit remains 45
  seconds; the shorter value is not a reward-design change.
