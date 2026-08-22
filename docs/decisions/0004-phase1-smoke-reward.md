# Decision 0004: Disposable Phase 1 smoke-test reward

Status: Approved

Date: 2026-08-22

Approved by the project owner on 2026-08-22.

## Context

Phase 1 needs a trivial reward only to prove that the Gymnasium and PPO loop runs. Reward design remains owner-controlled and the project later requires separately documented, pre-registered reward experiments.

## Decision

Use this temporary reward during the Phase 1 infrastructure smoke test:

- each normal step: `display_speed / 1000.0`;
- timeout or off-track truncation: subtract `1.0` on the final step;
- completed race: no additional finish bonus.

Initial safety limits are 45 seconds of in-game episode time and 50 horizontal units of absolute reference-path offset. These are conservative wrapper-safety limits, not claims about optimal episode design.

## Scope boundary

This reward exists only to exercise transport, observations, actions, episode boundaries, logging, and Stable-Baselines3 integration. It is not counted as one of the later formal reward iterations because it has no learning hypothesis and will not be evaluated as a candidate driving objective.

Any change from this formula before the Phase 1 smoke test completes requires another owner decision.

## Approval record

The owner approved the proposed normalized display-speed reward, `-1.0` timeout/off-track penalty, and absence of a finish bonus.
