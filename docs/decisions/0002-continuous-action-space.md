# Decision 0002: Continuous action space

Status: Approved

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

The environment maps normalized steering to TMInterface's analog `Steer` range. TMNF exposes acceleration and braking as the signed analog `Gas` axis, so the applied gas value is `(throttle - brake) * 65536`, rounded to the nearest integer and constrained to the documented range.

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
