# Overnight pure-discovery preflight

Date: 2026-09-04. Requested direction: uninterrupted overnight training with
the unchanged V4 reward; no localized bonus, action anchor, or approval/evaluation
stop at each checkpoint. Training must not start until the owner confirms readiness.

2026-09-05 amendment: the original `100x` launch was infrastructure-confounded
and is complete, not resumable. The replacement runner is pinned to live-
verified `2x` physics and has transactional optimizer rollback protection. See
[the fidelity and hardening report](simulation-speed-fidelity.md). No replacement
training has started; fresh owner readiness is still required.

## Host checks before launch

Read-only checks found the High Performance plan active. Automatic sleep and
hibernate timers are zero (never) on both AC and DC. These are timer settings,
not removal of Windows' manual Sleep/Hibernate capability.

Remaining idle settings observed before preparation:

| Setting | Original AC value | Original DC value |
| --- | ---: | ---: |
| Hybrid sleep allowed | 1 (on) | 1 (on) |
| Display-off timer | 900 seconds | 600 seconds |
| Disk-idle timer | 1200 seconds | 1200 seconds |

Their original values are retained here so these changes can be reversed after
the experiment. Power preparation completed: hybrid sleep, display-off, and
disk-idle settings were set to zero on AC and DC, then the existing High
Performance plan was reapplied. Read-back verified all five settings
(STANDBYIDLE, HIBERNATEIDLE, HYBRIDSLEEP, VIDEOIDLE, DISKIDLE) are zero on
both power sources. Manual sleep/hibernate and Windows restarts remain possible.

The C: drive had approximately 478 GiB free. No TMForever or Python trainer
process was found at this check. The live speed preflight has not run today.

## Throughput and disk estimate

The four completed Stage 1 gates used the unchanged V4 reward at the 100x
simulation setting. They processed 2,000,896 interactions in 5,283.649 seconds,
or approximately 379 interactions/second including PPO optimization. These were
four separate roughly 22-minute sessions, not one overnight stability trial.
Historical results support 100x as the candidate setting; they do not prove
current eight-hour reliability. A live no-learning connection/reset/100x
warm-up must pass before the overnight learner is launched.

Planning allowance: 300-380 interactions/second, not a fresh benchmark:

| Runtime | Additional interactions |
| --- | ---: |
| 6 hours | 6.48-8.21 million |
| 8 hours | 8.64-10.94 million |
| 10 hours | 10.80-13.68 million |

At 250k checkpoint spacing, eight hours means approximately 35-44 interval
checkpoints, plus final/emergency files. Existing optimizer-bearing models are
about 0.2 MB each, so weights are only about 10 MB. Compact episode logs,
TensorBoard summaries, and selected input replays should fit comfortably in a
1 GiB planning allowance; full per-step SimState dumps or continuous video are
excluded from that estimate. Retain original checkpoints and all new intervals.
An abrupt crash can still lose interactions since the last valid checkpoint;
250k spacing bounds that normal recovery gap to roughly 11-14 minutes.

Those estimates are retained as the historical basis for the first attempt and
are superseded for any replacement run. The live `2x` no-learning preflight
measured `19.933` interactions/second across `20/20` finishes. Plan for roughly
`0.52-0.57M` interactions in eight hours after PPO overhead, with a nominal
250k checkpoint approximately every 3.5-3.9 hours rather than every 11-14
minutes.

## Approved launch contract

The owner subsequently approved running **until they request stop**, not until a
fixed morning time, and confirmed readiness to launch on 2026-09-04. The start
still depends on the current-host live preflight succeeding.

- Selected base: the last reliable pure-discovery checkpoint, Stage 1 Gate 2,
  `checkpoints/wr_chase_stage1/gate_01000000_model.zip`, rather than either
  Stage 2b model. It was trained with V4 reward only and measured 10/10
  deterministic finishes. This is a pre-run selection, not a newly selected
  checkpoint after overnight results.
  SHA-256: `BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044`.
- Keep the original Stage 1 PPO/gSDE settings and V4 reward. Do not silently
  inherit the Stage 2b graduated bonus, anchor, or optimizer changes.
- Use the checksum-pinned `2x` simulation-speed contract. `100x` is prohibited
  because the same checkpoint changed from `0/5` at `1x` to `5/5` at `100x`.
- Use a separate run directory and a continuous stop-request-controlled learner, not the
  existing gate runner (which intentionally requires evaluation between gates).
- Save every nominal 250k interactions without evaluation pauses; include
  optimizer state, hashes, and completed-update timing. Also save on clean exit.
- Write local progress/lap-time/finish summaries and preserve useful replay
  evidence without a live Codex reporting loop. Evaluate checkpoints after the
  overnight run; a successful stochastic training curve is not sufficient proof
  of deterministic reliability.
- No performance-gate pause overnight. Still fail safely on invalid/nonfinite
  actions, unrecoverable bridge/game failure, checkpoint-write failure, or low
  disk space. A local error record must survive; no Codex approval is required
  to write it. Do not silently restart from an older model and merge histories.
- Add a run-scoped Windows keep-awake request and release it on process exit.
  Manual sleep, power loss, Windows restart, and game crashes remain risks.
- Run as a standalone local Python process with file-backed logs. No OpenAI
  API calls, Codex polling loop, recurring Codex automation, or model inference
  service is needed overnight. Codex usage is for setup and later analysis,
  not the local PPO computation itself. Keep the host and game running.

## Continuous implementation and frozen hypothesis

Hypothesis carried forward unchanged: "Given sufficient training time, the
existing progress-based reward may be sufficient for the agent to discover
slip-angle/drift behavior near the first turn and/or final corner unassisted,
similar to Yosh's AI, but is expected to plateau below world-record pace without
targeted assistance."

This extends the budget, not the reward or observation/action definitions. The
prior 2M-interaction plateau remains a valid budget-limited result. No guaranteed
discovery, lap-time target, or training-performance stopping threshold is added.

`scripts/train_wr_continuous.py` uses a separate run directory and the stock PPO
rollout collector in an end-condition-free loop. Fixed
learning rate 0.0003, clip 0.2, 2048 rollout steps, batch 64, 10 epochs, gamma 0.99,
GAE 0.95, entropy coefficient 0, value coefficient 0.5, gradient norm 0.5,
and gSDE refresh every 4 steps are inherited from the pinned base. Nonconstant
schedules are rejected because an indefinite run has no percentage of budget
remaining. The runner now sets `target_kl=0.20`, rejects a mean KL above `0.50`,
and transactionally restores the pre-update policy and optimizer if finite/KL
validation fails. This is a new infrastructure guard, not the Stage 2b reward
or its aggressive `0.01` early stop.

Periodic saves occur immediately after the update crossing each nominal 250k
boundary (at most 2047 interactions late). Archives include optimizer state,
are CRC-checked, and have SHA-256 sidecars recording collected versus actually
learned-through timesteps. A stop during rollout preserves learned weights and
records the discarded, unlearned partial rollout honestly. Checkpoint writes use
a temporary archive before publishing the finished file; older evidence is never
overwritten. A health failure attempts a distinctly labeled emergency save,
without automatically restarting or rolling back.

The no-learning live warm-up requires at least 60 seconds and 20 completed
episodes, at least one finish, finite data, and successful reset/rewind cycles
at the pinned `2x`. Its wall-clock timeout scales with simulation speed. The
observed preflight completed `20/20` in `248.828s`. This verifies present
operation, not guaranteed overnight stability. Game/A01
setup occurs before the runner, so the long-running process needs no UI inputs.

`stop-training.cmd` in the project root locates the one matching live trainer
and writes its stop request. The owner can double-click it without Codex. A
chat request to stop will use the same mechanism; neither method force-kills
Python. Local `status.json` is refreshed every 30 seconds, with compact Monitor
and TensorBoard logs. There is no recurring Codex task or overnight API call.

Selected real game input replays are saved before reset: improving finishes in
each 250k block, a sample every 1000 episodes, and the first failure per block.
These support later videos, not slide verification. Future discovery evaluation
still requires the frozen direct-live SimState contract; stochastic training
finishes or input replay telemetry alone cannot prove a beneficial slide.

Launch status and fresh results will be recorded below only after observation.

## Notable moments

The previous Stage 2b KL early-stop greatly reduced optimizer work; its rate
would not be a defensible estimate for restored full-PPO training. The estimate
above uses the original pure-discovery workload. The 100x setting is a simulation
target, not a promise of 1000 interactions/second end-to-end.

That statement is historical only. The live fidelity study proved `100x` can
change outcomes and trajectories, so the replacement run deliberately trades
most of that throughput for the verified `2x` domain.

Skipping evaluation gates meets the requested uninterrupted experiment but
removes early detection of deterministic collapse. Preserving every checkpoint
is therefore important; the last checkpoint may not be the best one.

The OpenAI Docs check informed the low-credit design: avoid repeated agent
messages/tool-result ingestion while a local process does the work. Official
[usage guidance](https://learn.chatgpt.com/docs/pricing) describes token-based
usage for prompts, context, tool results, and responses. Independence of this
trainer is a code-level conclusion: the local PPO/environment workflow does
not call an OpenAI model or API.
