# Overnight pure-discovery preflight

Date: 2026-09-04. Requested direction: uninterrupted overnight training with
the unchanged V4 reward; no localized bonus, action anchor, or approval/evaluation
stop at each checkpoint. Training must not start until the owner confirms readiness.

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

## Proposed launch contract, awaiting readiness confirmation

- Suggested base: the last reliable pure-discovery checkpoint, Stage 1 Gate 2,
  `checkpoints/wr_chase_stage1/gate_01000000_model.zip`, rather than either
  Stage 2b model. It was trained with V4 reward only and measured 10/10
  deterministic finishes. This is a proposed base, not a newly selected
  checkpoint after overnight results.
  SHA-256: `BA056E0B42D7CAEE4D02B6AB8E0D592BE6A363068E75AAEBC3ED8487791B4044`.
- Keep the original Stage 1 PPO/gSDE settings and V4 reward. Do not silently
  inherit the Stage 2b graduated bonus, anchor, or optimizer changes.
- Use a separate run directory and a continuous time-bounded learner, not the
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

The owner still needs to confirm readiness and the intended duration/morning
stop time. No overnight runner has been implemented or launched in this
preflight turn, and no fresh throughput or overnight stability result is claimed.

## Notable moments

The previous Stage 2b KL early-stop greatly reduced optimizer work; its rate
would not be a defensible estimate for restored full-PPO training. The estimate
above uses the original pure-discovery workload. The 100x setting is a simulation
target, not a promise of 1000 interactions/second end-to-end.

Skipping evaluation gates meets the requested uninterrupted experiment but
removes early detection of deterministic collapse. Preserving every checkpoint
is therefore important; the last checkpoint may not be the best one.

The OpenAI Docs check informed the low-credit design: avoid repeated agent
messages/tool-result ingestion while a local process does the work. Official
[usage guidance](https://learn.chatgpt.com/docs/pricing) describes token-based
usage for prompts, context, tool results, and responses. Independence of this
trainer is a code-level conclusion: the local PPO/environment workflow does
not call an OpenAI model or API.
