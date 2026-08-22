# Reward v1: Sparse finish-only

Status: Experiment complete; hypothesis partially supported

Date pre-registered: 2026-08-22

Approved by the project owner on 2026-08-22.

## Pre-registered hypothesis

A reward that only fires on finishing the track (sparse) will be too sparse for PPO to find gradient signal in a reasonable number of training steps. Expected outcome: the agent never learns to finish, reward stays near zero/flat, with no meaningful behavior change across training.

This statement is recorded exactly as supplied by the project owner. It must not
be revised after observing training or evaluation results.

## Reward contract

- Normal step: `0.0`.
- Timeout or off-track truncation: `0.0`.
- Finished race: `1.0`.

The implementation will be an isolated reward function passed into the
environment, not a conditional hardcoded into `TrackmaniaEnv`.

## Implementation verification

`sparse_finish_reward` is implemented in `src/trackmania_rl/rewards.py` as a
callable over an immutable transition record. `TrackmaniaEnv` accepts a
`reward_function` argument; its default remains the separate Phase 1 smoke
reward. Nonfinite reward outputs fail before Gymnasium returns the step.

The reward-v1 wiring check completed 20 live timeout-truncated episodes and 100
steps. Every reward was exactly `0.0`, every action was finite and in range, and
all reset observations were identical. The `+1.0` finish branch is covered by an
isolated unit test. Ignored local evidence hashes:

- action log: `71C2A4A7685A590D23138AD50CAC69B1A70B054C89ACA752A6A52655F904D513`;
- summary: `08FDF010A1A996BBBD2BC95E0207D817FA19B287DB6374443111CFA0BAA86A88`.

`scripts/train_reward_v1.py` implements Decision 0006 with periodic checkpoints,
a distinct `reward_v1_sparse` TensorBoard run, Monitor episode outcomes, exact
artifact hashes, failure manifests, and resumable checkpoints. Every action is
still checked for finite/bounded/affine behavior, but the long run stores compact
aggregate extrema instead of flushing two JSON records per step.

`scripts/evaluate_reward_v1.py` fixes the post-training evaluation at 20 or more
deterministic episodes. It records every evaluation action and episode outcome,
finish rate, episode lengths, best/average progress, speeds, terminal causes, and
a telemetry-based behavior description. Visible behavior during the run remains
an explicit human observation rather than being invented from numbers alone.

## Invalidated preliminary training attempt

The first long run was deliberately stopped and excluded from the formal result
at TensorBoard step `28,672`. It completed 64 episodes with zero finishes and a
flat zero reward, but the project owner visually observed the car reversing off
the starting platform onto the grass below on essentially every run. The Monitor
classified 62 episodes as 45-second timeouts and only two as off-track.

The discrepancy identified an environment bug rather than a reward result:
off-track detection measured only horizontal X/Z distance from the reference
path. A car directly below the start could remain horizontally close and spend
the rest of the episode on the lower grass. Those episode lengths therefore mix
sparse-reward behavior with missing vertical crash detection. The partial run is
preserved under `reward_v1_failed_vertical_drop` and must not be counted toward
the fixed 500,000-step budget.

Ignored local artifact hashes:

- failed-run manifest: `2B4E0356BCD43C11FC3FA4C6A4A91720BC0253722E541110EE5EE651E9C591FE`;
- Monitor CSV: `6C0895098369B27F60C9909247C92F4F64EBBF56F2EC02A77765015D9D510330`;
- TensorBoard event: `7D33E411F11F636761E104330BE9A3B8292F3689CB4CD89A6ED44664F03DD1CA`.

A first fixed-action reproduction also failed honestly: `[steer=0, throttle=0,
brake=1]` applied gas `-65536` but moved about 115 path units forward rather than
backing off the start. Over 100 steps its minimum vertical offset was only
`-0.203`, so the proposed 10-unit fall boundary did not fire. The ignored action
log SHA-256 is
`DCEE08E43643EBB4753F51CEB833A54BA5CDE995E4E6A17C72CDDDDC5ABD04A0`.
Formal training remains stopped while the visible PPO trajectory is reproduced.

The actual stochastic PPO reproduction completed 8,192 steps and matched the
visual report. Its 87 episode boundaries contained 78 vertical falls, nine
timeouts, and zero finishes; minimum vertical offset was `-12.061`, maximum path
progress was `173.242`, and maximum displayed speed was `123`. The 10-unit local
vertical boundary therefore detects the repeated grass drop.

Post-processing failed after collection with `KeyError: 'timeout'` because the
action log recorded generic truncation plus telemetry but omitted explicit
terminal-reason fields. The full 8,192-record log is intact with SHA-256
`490B73E375FF52F800C107B74234F55C6DDC5108082E6FEAF006823468C7E3B7`.
This is a reporting-schema bug, not a lost trajectory.

After adding explicit `timeout`, `off_track`, `fallen`, and `race_finished`
fields to every action record, the same seed-42 stochastic PPO probe reran
cleanly for 8,192 steps. It produced 118 episode boundaries: 110 vertical falls,
eight timeouts, zero horizontal off-track truncations, and zero finishes. The
minimum vertical offset was `-12.196`; maximum path progress was `119.072`, and
maximum displayed speed was `102`. Every action was finite, in range, and
affine-consistent, with no hidden clipping.

This clean reproduction validates the 10-unit local vertical boundary for the
A01 start drop. Falling remains a zero-reward truncation, so this is an
environment validity fix rather than reward shaping. Ignored local evidence
hashes:

- action log: `69030B4ADC0294BD27B87AEC57128F10E797E8534BC5D1DE8284A059DF04295A`;
- summary: `FF6329BED2C227F0B7D3CCDA82DD83E428AF5455208B2C0215078B2FBA0C5879`.

## Interrupted 6x start and speed optimization

A valid formal start at 6x was interrupted at TensorBoard step 10,240 solely to
benchmark the owner's proposal to increase simulation speed. Its 221 episodes
contained 214 vertical falls, seven timeouts, zero finishes, and reward fixed at
zero. It is preserved under `reward_v1_interrupted_speed_benchmark` and excluded
from the fresh formal budget; this is not an early-stopped reward result.

End-to-end PPO benchmarking measured `56.35` steps/second at 6x and `240.81` at
100x. A longer 8,192-step 100x soak sustained `313.92` steps/second, completed
118 episode resets without a bridge failure, and validated every action. Per
Decision 0006, the formal run will restart from zero at 100x while retaining the
same 100 ms game-time control interval and 500,000-step minimum. This is a
wall-clock optimization, not a reward or sample-budget change.

## Checkpointed 100x bridge interruption

The first formal 100x attempt reached model timestep `450,560` before the PC
went to sleep, aborting the local bridge socket with Windows error `10053`. This
was an external host interruption, not evidence of 100x instability. The last
checkpoint at `450,000` is preserved, so the fixed experiment resumes from there
and replays only 560 interactions. No reward, environment, policy, seed, or
training parameter changes for the resumed segment. Decision 0006 records the
failure artifacts and requires the final summary to combine both TensorBoard
segments and disclose cumulative time and replayed steps.

## Evaluation plan

- Use a distinct reward-v1 TensorBoard run name and checkpoint namespace.
- Train for at least 500,000 environment steps under the fixed hyperparameters
  in Decision 0006. Stable-Baselines3 may complete its current 2,048-step rollout,
  so record the exact actual total.
- Evaluate finish rate and episode-length trend quantitatively.
- Inspect the TensorBoard reward curve for sparse/flat behavior or unexpected
  learning signal.
- Watch deterministic evaluation episodes and describe actual behavior,
  including any finish or meaningful behavioral change.

## Actual outcome

The fixed formal model completed `501,200` PPO timesteps against the 500,000-step
minimum. Including 560 interactions replayed after host sleep, the environment
processed `501,760` interactions over a cumulative `1,758.399` wall-clock
seconds (29.31 minutes).

Training completed 29,917 episodes with zero finishes, 29,900 vertical falls,
17 timeouts, and zero horizontal off-track truncations. Episode reward was
exactly `0.0` throughout. TensorBoard's 245 reward points were flat at zero.
Mean episode length fell from `240.375` at the first rollout report to `16.0` at
the last, with observed report extrema of `16.0` and `240.375`.

The 20-episode deterministic evaluation also produced zero finishes. Every
episode followed the same 16-step (1.6 game-second) trajectory, reached displayed
speed 63, made effectively zero forward path progress, and crossed the vertical
fall boundary. Its mean normalized action was approximately
`[steer=-0.8403, throttle=0.9192, brake=0.0070]`: the final policy accelerated
while steering hard left, rotated to about 1.625 radians of heading error, and
fell 11.166 units below the local path. This explains why accelerated training
looked like a stationary rapid-reset loop on screen; the full crash occurred
between rendered frames.

The hypothesis is partially supported. Its central prediction held: sparse PPO
never discovered a finish, received no gradient signal from the external reward,
and produced a perfectly flat reward curve. The prediction of "no meaningful
behavior change" did not hold. Episode length collapsed and the deterministic
policy developed a repeatable full-throttle, hard-left fall. This is a harmful
rather than useful behavior change and is documented as an unexpected result.

The experiment does not isolate why PPO settled on this particular zero-return
action pattern; initialization, critic transients, and optimization dynamics are
plausible contributors. Root-causing that drift is deferred because reward v1
provides no preference among zero-return behaviors and the pre-registered sparse-
signal question is answered. Reward v2 should pre-register dense forward-progress
feedback and an explicit cost for falls before its implementation begins.

Ignored local result hashes:

- completed run manifest: `6B30F4662C37B4166DBFDC924B310EB61D6D5F35801A87BD27B9018779D60131`;
- training summary: `C4A9A209512FC73FB4CD62E04E7C5723E4219EE66E0B7592468122DACF73FF9B`;
- Monitor CSV: `9B5FFCEE5EFA212A4089B9E5B6FF2B15B817345ED53C3F5211E7A4A23BD1D827`;
- final model: `E19E7009E2CB94FEE309DA20AA201EA23EA852A435FB2648E91C2FD26D1C7D08`;
- evaluation action log: `4AE09AF842929DE3D094EE26A33F08FFB5E3117709341620255E43827FD45268`;
- evaluation summary: `72E1D0989CFA3E0DF37C1CF33D74F178DDF3015EB8212DB09CAC6BACFDE18737`.
