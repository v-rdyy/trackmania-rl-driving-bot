# Reward comparison: V1 through V4

**Status:** Complete through the V4 100-episode scale validation
**Date:** 2026-08-24

This document puts the four pre-registered reward experiments in one place:
what each version predicted, what actually happened, and why the evidence
motivated the next version. The underlying experiment records remain in
[`reward_v1.md`](../reward_v1.md), [`reward_v2.md`](../reward_v2.md),
[`reward_v3.md`](../reward_v3.md), and [`reward_v4.md`](../reward_v4.md).

## At a glance

| Version | Reward signal | Training | Main deterministic result | What the result exposed |
|---|---|---:|---|---|
| V1 | `+1` only on finish; otherwise `0` | `501,200` PPO steps | `0/20` finishes; all rewards `0` | PPO had no useful external gradient and converged to a repeatable reverse/hard-left fall. |
| V2 | `displayed_speed / 1000`; no terminal shaping | `1,001,120` PPO steps | `7/20` finishes in the original evaluation; `6/20` in the fixed-metric retrospective | Dense signal learned most of the route, but raw speed did not value precise placement or completion. |
| V3 | Positive forward centerline progress, capped at `10` units and divided by `10`; stuck termination | `1,001,472` PPO steps | Unchanged-checkpoint corrected evaluation: `20/20`, best `28.030s`, mean `28.106s` | Route-aligned reward solved observed reliability, but a flat clamp and no precision term left oscillation and wide lines. |
| V4 | Signed progress clipped to `[-20, 20] / 10`, `-0.10` per step, `+50` finish, `-250` verified failure | `1,001,472` additional PPO steps from V3 | Initial `20/20`; scale validation `98/100`, best `24.900s`, mean `24.931s` | Time and terminal shaping nearly closed the PB gap, but oscillation and rare late-course control failures remained. |

All lap times in this document use TMNF's terminal race clock, the basis directly
comparable to the owner's real `24.5s` A01 PB.

## V1: sparse finish-only

**Pre-registered hypothesis.** “A reward that only fires on finishing the track
(sparse) will be too sparse for PPO to find gradient signal in a reasonable
number of training steps. Expected outcome: the agent never learns to finish,
reward stays near zero/flat, with no meaningful behavior change across
training.”

**Result.** The central prediction held. Training recorded zero finishes across
`501,200` model timesteps, and all 245 TensorBoard reward points were exactly
zero. The 20 deterministic evaluation episodes also produced zero finishes;
each repeated the same 1.6-second reverse-and-hard-left fall. The “no meaningful
behavior change” prediction did not hold: episode length collapsed and PPO
settled on a harmful repeatable behavior even though every outcome was equally
rewarded. The hypothesis was therefore partially, not completely, supported.

**What motivated V2.** V1 established that a finish-only signal was too sparse
for this setup. V2 deliberately tested the simplest dense alternative—raw
speed—to find out whether abundant feedback alone was enough or whether proxy
alignment would become the next problem.

## V2: dense speed

**Pre-registered hypothesis.** “A dense speed reward will provide learning
signal (unlike v1's flat zero), but because it rewards raw speed rather than
progress toward finishing, the agent is expected to find a way to exploit it,
e.g. looping or oscillating in place to farm speed, rather than learning to
drive the track toward completion.”

**Result.** Dense feedback worked, but the exact predicted exploit did not. The
rolling mean reward rose from `2.883` to `65.696`; the final policy drove almost
the full route and finished `7/20` original deterministic episodes. It did not
loop in place. The separate fixed-metric retrospective finished `6/20`, while
all 20 episodes oscillated, 19 inverted, and all 14 timeouts contained a stuck
period. Visible review tied the failures to left/right steering oscillation and
a lower-right final-hoop collision that could leave the car upside down. V2
supported the broader proxy-misalignment concern, but contradicted the predicted
in-place speed-farming mechanism.

**What motivated V3.** Speed was learnable but not equivalent to advancing and
finishing. V3 replaced it with forward progress along the A01 centerline and
added a geometry-and-motion-based stuck termination, while continuing to measure
precision instead of prescribing a steering solution.

## V3: clamped forward centerline progress

**Pre-registered hypothesis.** Positive forward centerline progress, capped at
`10` units per 100 ms step, was expected to align PPO with route completion
better than V2's speed proxy. A 2.0-second stuck window requiring both less than
`1.0` unit of progress and less than `2.0` units of world motion was expected to
end unrecoverable stalls. V3 was expected to improve finish rate and stuck time,
while possibly retaining oscillation or final-checkpoint contact because neither
steering nor lateral error was penalized.

**Result.** The original evaluator reported `12/20` finishes and eight final-jump
falls, but replay evidence showed that its immediate 10-unit vertical cutoff
also stopped valid low trajectories that were still moving and recoverable. The
fall detector was corrected to require the frozen no-progress/no-motion window,
then the unchanged checkpoint was evaluated once more: `20/20` finishes, no
terminal failures or inversions, best `28.030s`, mean `28.106s`, and worst
`28.660s`. Reliability improved dramatically over V2, but oscillation remained
`20/20`; maximum absolute lateral deviation was `13.568` units. In the original
evaluation, `31.17%` of steps reached V3's positive-progress clamp, creating a
flat region with no immediate preference for cleaner progress beyond the cap.

**What motivated V4.** V3 had a reliable route but was `3.530s` behind the human
PB and did not make a near-complete failure net worse than a finish. V4 retained
the route-aligned progress foundation, made progress signed, raised the clamp,
added an explicit time cost, and made verified failure decisively negative. It
deliberately added no steering or lateral penalty so oscillation reduction would
remain an evaluated question rather than a built-in answer.

## V4: signed progress, time cost, and terminal outcomes

**Pre-registered hypothesis.** Signed centerline progress plus a per-step time
cost, finish bonus, and `-250` verified-failure penalty was expected to preserve
V3 reliability, improve speed, and eliminate the near-finish collision because
a failed near-complete run would be net negative. Whether the raised signed
clamp would reduce oscillation was explicitly left open; V4 had no direct
smoothness or lateral term.

**Result.** V4 completed `1,001,472` additional steps from the pinned V3 model.
The initial 20-run evaluation finished `20/20`; best time improved by `3.130s`
to `24.900s`, mean improved by `3.177s` to `24.929s`, and the reviewed widest
run passed centrally through the final hoop. Oscillation nevertheless remained
`20/20`, and maximum lateral deviation increased to `17.833` units even though
no evaluation step reached the raised clamp.

Decision 0010 then tested the unchanged checkpoint over 100 more deterministic
episodes at 6x. It finished `98/100` (`98.0%`), with a 95% Wilson interval of
`93.00%` to `99.45%`. There were two stuck terminations and zero falls, timeouts,
off-track terminations, or inversions. Among the 98 finishes:

- best: `24.900s`;
- mean: `24.930510s`;
- worst: `24.980s`;
- 95th percentile: `24.950s`;
- population standard deviation: `0.013s`.

The lap time therefore held up: the 100-run mean was only about `0.002s` slower
than the 20-run mean, and the best time was identical. Reliability did not remain
literally perfect at the larger scale. The two preserved failures were distinct:

- episode 31 gradually braked from displayed speed `80` to zero and stopped at
  progress `2198.6/2206.5`, about eight progress units short of the finish;
- episode 61 hit the left-side final-approach structure at displayed speed
  roughly `363`, dropped to speed `1` in one step, and remained wedged upright
  at progress `2162.3/2206.5`.

Fixed metrics still detected oscillation in `100/100` episodes. Mean peak
sign-crossings in a two-second window was `5.01`; mean per-episode p95 absolute
lateral offset was `11.127` units, and maximum absolute lateral offset was
`18.642` units. No episode inverted. The 24,963 evaluated actions were all
finite, in range, and valid. All 100 input replays are retained in
`artifacts/replays/reward_v4_evaluation_100/`; the two failure videos and
telemetry are in `artifacts/videos/reward_v4_scale100_failures/`.

**What the V4 evidence motivates.** V4 is already a strong resume-scale result:
a custom live-game PPO system progressed from zero finishes to a measured 98%
finish rate and a best lap `0.400s` behind the owner's PB. The remaining evidence
also gives a concrete case for a precision-focused V5: oscillation was universal,
tail lateral deviation worsened, and one of 100 runs still made a high-speed
final-approach impact. The other reasonable choice is to stop reward iteration
and finalize the project around the V1-V4 story. V5 remains on hold pending that
decision.

## Corrections that affect interpretation

- **Pedal labels:** V1 and V2 used a bridge mapping whose throttle/brake labels
  were reversed relative to TMNF's signed Gas polarity. Their physical behavior,
  rewards, and outcome counts remain reproducible, but causal descriptions use
  the corrected interpretation. V3 and V4 use corrected semantics.
- **V3 fall detector:** the original `12/20` V3 result measured an invalid
  immediate vertical cutoff. Both that result and the one corrected
  unchanged-checkpoint `20/20` evaluation are preserved; only the latter is used
  for the reliable-finishing claim.
- **Race clock:** environment-controlled elapsed time is 100 ms lower than
  TMNF's terminal clock because it subtracts the captured reset state. Human-PB
  comparisons use the game clock consistently.
- **V4 scale postprocessing:** episode 1 included 45 disclosed pre-race records
  from a restarted countdown. A summary accounting check initially rejected the
  completed log, then was corrected to require `raw = evaluated + startup`.
  The summary was rebuilt from the original 100 replays and action log; no live
  episode was rerun or discarded.
- **Scope of reliability:** these are repeated deterministic episodes from the
  fixed A01 start snapshot. The Wilson interval describes observed repeatability
  under that protocol, not generalization to new tracks, randomized starts, or
  every machine state.

## Evidence pointers

- V4 100-run summary: `runs/reward_v4/evaluation_100_summary.json`, SHA-256
  `0A97C8E5B0E91F6A0256D3A1CFBD24575E86E1A7833F412CF46F0EEF75A17CB8`.
- V4 100-run action log: `runs/reward_v4/evaluation_100_actions.jsonl`, SHA-256
  `2E96B85A9CAA30688C64B8DE7287485C907D3434AF3EFDFC8C2BC680B20EBF05`.
- V4 final checkpoint: `checkpoints/reward_v4/final_model.zip`, SHA-256
  `6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2`.
- Fixed scale-validation protocol: [Decision 0010](decisions/0010-reward-v4-100-episode-validation.md).
