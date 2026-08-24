# Decision 0010: Validate V4 reliability over 100 episodes

**Status:** Complete
**Date:** 2026-08-24

## Context

The first formal V4 evaluation finished 20/20 deterministic episodes, with a
best TMNF race-clock lap of 24.900 seconds and a mean of 24.929 seconds. That is
strong evidence that V4 fixed the final-hoop failure observed in earlier reward
versions, but 20 episodes is still a small sample for making a reliability
claim.

V5 is deliberately on hold. This validation must test the already-trained V4
policy without changing its checkpoint, reward, environment, detector, or
action processing.

## Decision

Run exactly 100 deterministic episodes from the pinned V4 final checkpoint:

- checkpoint: `checkpoints/reward_v4/final_model.zip`
- SHA-256: `6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2`
- map: A01-Race
- game speed: 6x
- corrected geometry-aware fall detector
- the same finish, lap-time, lateral-deviation, steering-oscillation,
  upside-down, stuck-duration, and action-validity metrics used by the 20-run
  V4 evaluation
- a clean TrackMania/TMInterface lifecycle before the first episode

The 100-run output must be separate from the original 20-run evidence:

- summary: `runs/reward_v4/evaluation_100_summary.json`
- action log: `runs/reward_v4/evaluation_100_actions.jsonl`
- input replays: `artifacts/replays/reward_v4_evaluation_100/`
- replay tag: `scale100`

All 100 input replays will be retained for audit and future website/video use.
No failed episode may be discarded or rerun selectively. If postprocessing
fails after the live episodes complete, rebuild the summary from the preserved
action log and replay set rather than repeating the policy evaluation.

## Acceptance and reporting

Report the observed finish rate, a 95% Wilson confidence interval, terminal
causes, best/mean/worst TMNF race-clock lap times, action validity, and the same
precision metrics as the original V4 evaluation. Compare the 100-run result to
the earlier 20-run result without treating repeated deterministic snapshot
episodes as proof of robustness to every possible starting condition.

Do not begin V5 after this validation. The next reward direction remains a user
decision after reviewing the 100-episode result and the V1-V4 comparison.

## Outcome

The unchanged checkpoint finished `98/100` episodes. Its 95% Wilson finish-rate
interval was `93.00%` to `99.45%`. There were two stuck terminations and zero
falls, timeouts, off-track terminations, or inversions. The 98 finish times were
`24.900s` best, `24.930510s` mean, and `24.980s` worst on TMNF's race clock.

All 100 input replays were retained. Replay review found two distinct late
failures: episode 31 braked to a stop about eight progress units short of the
finish, while episode 61 hit the left-side final-approach structure at high
speed and remained wedged upright. Oscillation was detected in `100/100`
episodes.

The first summary pass exposed a bookkeeping bug when a clean-launch countdown
restart contributed 45 disclosed startup action records. The live evaluation
itself was complete. The count invariant was corrected and regression-tested,
then the summary was rebuilt from the preserved log and replays without
rerunning any episode.

Full interpretation and cross-version context are in
[`docs/reward-comparison-v1-v4.md`](../reward-comparison-v1-v4.md).
