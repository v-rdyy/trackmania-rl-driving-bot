# Phase 1 notes

Last updated: 2026-08-22

## Current status

- Decision 0002 defines continuous `[steer, throttle, brake]` actions and fail-fast raw-action logging.
- The live bridge applies true analog `Steer` and `Gas` values without changing the Phase 0 digital protocol.
- Decision 0003 defines the provenance-pinned A01 driven reference path.
- The engineered 26-value observation builder is implemented and validated against the full manual lap.
- Snapshot rewind and 100 ms synchronous callback control are verified live for episode resets.
- The Gymnasium `reset()`/`step()` wrapper passed its 20-episode live reliability check.
- The bounded PPO/TensorBoard smoke test completed 2,048 live steps without crashing.
- All Phase 1 exit criteria are satisfied; Phase 2 has not started.

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

## Bounded PPO smoke test

Decision 0005 records why the smoke runner uses PPO with squashed gSDE instead
of Stable-Baselines3's default unbounded Gaussian followed by action clipping.
The callback logs the tanh-bounded policy sample and its environment-unit action,
then verifies their exact affine relationship. The environment independently
logs the received action before sending analog inputs to TMInterface.

The live seed-42 run completed 2,048 steps in approximately 34 seconds at about
59 environment steps per second on CPU. It used 256-step rollouts, batch size 64,
and four optimization epochs. Both action logs contain exactly 2,048 records;
all values are finite and in range, and all policy-to-environment conversions
match the expected affine transform without hidden action clipping.

Four episode boundaries occurred: the first three reached the 450-step/45-second
timeout, and the fourth hit the off-track boundary after 398 steps at lateral
offset `-50.306451`. No episode finished the race, which is acceptable for this
integration-only run and is not evidence that the disposable reward learned a
useful driver.

TensorBoard contains seven `rollout/ep_rew_mean` points spanning `7.247250` to
`10.078000` and seven `rollout/ep_len_mean` points spanning `437` to `450`.
The changing reward and recorded episode-length series verify the required
logging path. Local artifact hashes:

- checkpoint: `88AD79435FB1287762A17A6965F72836B2B5F669FB268E11B40F0A83DDEFF202`;
- environment action log: `2FDFEE8BAFB4AEBFA15D4B41FE5A3B46706AF51D07E5130D4F28BE1B4C684AA5`;
- policy action log: `FBE5896A4412C5BFA394DA16DC73C1119C9B4EACBD9B5261F5ED8DCAA06ADC59`;
- TensorBoard event: `0D47E2A33A2F444E54989E12BC700111003ECD0067803F0488076454EF6645B0`;
- smoke summary: `F1B5EE8AFBBE93BABB977EFC2E47DF3158E760EC7EA54A8095E12BBE0644B72B`.

## Reset primitive

The wrapper captures one known start snapshot and restores its raw TMInterface simulation bytes on every reset. Before capture it refuses states above 5 displayed speed, beyond 25 path-progress units, or more than 10 lateral units from the reference path. A live round-trip moved the car `17.554` units under analog throttle, then restored the captured position with `0.000000` measured error. The next callback advanced from captured race time `7343600` to `7343700`, matching the configured 100 ms game-time step period.

After Phase 1, the first initial-respawn attempt made `prepare()` issue
TMInterface `GiveUp()`, apply neutral analog input, and advance one synchronized
step before snapshot capture. Its 20-episode live test completed all 100 steps,
but correctly failed the strict reset comparison: maximum observation drift was
`0.000083494`, above the `0.00001` threshold.

The raw action log identified the lifecycle cause. `GiveUp()` placed episode 0
inside the pre-race countdown (`-2500` through `-2100` ms), so gas `22938` did
not move the car. Snapshot rewind restored the car state for episode 1 but did
not restore the absolute clock, which remained near `201200` ms; the same gas
then accelerated normally. The failed attempt's ignored action log contains 100
valid finite actions and has SHA-256
`0F1C475DD6534025FB18B7CCFD9715EA6EE6F099E2B657485ADDE6E0D3C02635`.
A second attempt waited until race time was nonnegative, but it exposed a more
subtle transition: the first callback after `GiveUp()` still carried the old
positive clock, so the readiness check returned before observing the new
countdown. The following callback then entered `-2500` ms. Episode 0 stayed in
that countdown, while rewound episodes restored a start state facing backward
(`heading_error` approximately pi) at the old positive clock. The maximum reset
observation delta was `2.258129597`. Its ignored 100-action log has SHA-256
`20D22DD39F27B3340FC5786D037C4D4CD6DDAA68A7C1581FD61256981F206CF6`.
The next implementation must observe the negative countdown transition before
accepting a later nonnegative callback, rather than merely testing `>= 0` or
weakening the determinism threshold.

The transition-aware implementation passed the same 20-episode live test without
a manual Delete press. It ignored the stale positive callback, observed negative
race time, held neutral input until a later callback reached zero, and captured
the snapshot there. Every reset then began at race time `100`, speed `0`, and
effectively zero progress/lateral offset. All 100 actions were finite and valid,
and the maximum reset observation delta was `0.000000000`. The ignored evidence
hashes are
`4724F8676263B9FC982B8B53D717794EF112C13F3BA77ADB23E47CFBB3D4B87D`
for the action log and
`905BAC759BD0BC6B0F167BEDDFD196A36A0870264197BBF124BBF5658BB124C3`
for the summary. Manual Delete-key setup is no longer required once A01 is loaded.

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
- Stable-Baselines3's default PPO path would have clipped an unbounded Gaussian
  action before the environment could audit it. Squashed gSDE was chosen for the
  smoke run because it stays within SB3's supported policy paths while making
  every transformation observable. Its different exploration behavior remains
  a judgment call to revisit before long Phase 2 runs.
- During the PPO run, the first three episodes timed out, so an interim progress
  report expected that pattern to continue. The fourth instead crossed the
  lateral safety boundary after 398 steps. The raw terminal record identified
  the exact cause (`lateral_offset=-50.306451`) and explains why TensorBoard's
  mean episode length changed from 450 to 437.
- SB3 reported nonzero `train/clip_fraction`, but this measures PPO probability-
  ratio clipping during optimization, not control-action clipping. The separate
  2,048-record affine audit is the evidence that no action clipping occurred.
