# Decision 0002: Continuous action space

Status: Amended after live direction verification

Date: 2026-08-22

Approved by the project owner on 2026-08-22.

## Context

Phase 1 needs an action space for steering, acceleration, and braking. Discrete actions would simplify early debugging but would permanently limit the policy's control resolution unless the environment were redesigned later.

TMInterface 2.2.1 supports true analog inputs. Its official `SimulationManager::SetInputState` reference specifies integer values from `-65536` through `65536` for `InputType::Steer` and `InputType::Gas`.

## Decision

Use a three-value Gymnasium `Box` action in this order:

1. `steer` in `[-1.0, 1.0]`;
2. `throttle` in `[0.0, 1.0]`;
3. `brake` in `[0.0, 1.0]`.

The environment maps normalized steering to TMInterface's analog `Steer` range.
TMNF exposes acceleration and braking as the signed analog `Gas` axis. A live
identical-snapshot direction probe established that negative `Gas` drives
forward and positive `Gas` drives backward. The corrected applied value is
therefore `(brake - throttle) * 65536`, rounded to the nearest integer and
constrained to the documented range.

Models created before this direction check (V0, reward v1, and reward v2) retain
their exact original behavior through an explicit
`legacy_reversed_pedal_mapping` compatibility flag. New models, beginning with
reward v3, use the corrected default. The action-vector contract remains
`[steer, throttle, brake]` in both cases; compatibility swaps the two pedal
channels only at the transport boundary.

The Phase 1 smoke test must log every raw action before conversion. Each record must include the step index, raw three-value action, and applied integer steer/gas values. The smoke test must fail immediately if any raw action is nonfinite or outside the declared action space; it must not silently hide a policy or wrapper defect by clipping invalid values.

## Why

- The policy retains fine steering and pedal resolution from the first environment version.
- Raw per-step logging and fail-fast validation recover most of the debuggability advantage of a discrete action space.
- The mapping uses TMInterface's maintained analog API instead of synthesizing continuous behavior from digital key thresholds.

## Sources

- https://donadigo.com/tminterface/plugins/api/global/SimulationManager/SetInputState
- https://donadigo.com/tminterface/plugins/api/global/InputType
- https://donadigo.com/tminterface/plugins/api/global/InputState

## Approval record

The owner selected continuous control and explicitly required per-step raw-action logging plus range and NaN sanity checks during the Phase 1 smoke test.

## Initial live verification (insufficient for direction)

On 2026-08-22, the extended bridge applied the then-labeled half throttle as
gas `32768` and normalized half steering as steer `32768`. A 3.5-second probe
captured 36 finite records, moved the car `51.570` units, reached displayed
speed `86`, and changed yaw by `1.081477` radians. That probe proved analog input
worked, but did not compare motion against the start-path direction and therefore
did not establish whether the pedal labels were correct. The ignored evidence
file `artifacts/telemetry/phase1_analog.jsonl` has SHA-256
`58FA784C277CEBD6ED387C2B8B75299E39A10AD3D34D40CA3E3947112E5EA56E`.

## Corrective live direction verification

On 2026-08-22, four 20-step trials were run from the same A01 snapshot: two at
each full `Gas` sign. Both `+65536` trials moved `19.176` units backward along
the start tangent and fell `23.855` vertical units. Both `-65536` trials moved
`31.560` units forward, gained `31.075` path-progress units, and remained near
the reference height. The repeated result resolved the protocol polarity as
negative-forward/positive-backward and proved that the original throttle/brake
labels were reversed.

Ignored local evidence:

- samples SHA-256:
  `4CFB82A21827E7F2192E863A69E1DA2B23B7C6A99B8B1440892FA355BEDCA3E1`;
- pre-fix summary SHA-256:
  `4A56E193ED866FFF96D4949FBBDA9CDF4469EDF42247D5A47DC50332944021AA`;
- corrected-mapping samples SHA-256:
  `6D0F97F8FE4752CB7C41A0D6F41BD2AD9F3FC38CF80A1BBD982ECC5940D34484`;
- corrected-mapping summary SHA-256:
  `18EC872F39D13A7D0829A3E93D73907A220AA6E27A529B23FDC885396415DBF5`.
