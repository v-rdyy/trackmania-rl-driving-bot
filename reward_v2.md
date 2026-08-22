# Reward v2: Dense speed

Status: Training, deterministic evaluation, and precision retrospective complete

Date pre-registered: 2026-08-22

Approved by the project owner on 2026-08-22.

## Pre-registered hypothesis

A dense speed reward will provide learning signal (unlike v1's flat zero), but because it rewards raw speed rather than progress toward finishing, the agent is expected to find a way to exploit it, e.g. looping or oscillating in place to farm speed, rather than learning to drive the track toward completion.

This statement is recorded exactly as supplied by the project owner. It must not
be revised after observing training or evaluation results.

## Reward contract

- Every step: `displayed_speed / 1000`.
- No finish bonus.
- No explicit crash, fall, off-track, or timeout penalty.
- Termination and truncation only stop future reward collection.

The implementation must be an isolated reward function passed into
`TrackmaniaEnv`, not a conditional hardcoded into the environment.

## Corrective control-label interpretation

After V1 and V2 were completed, an identical-snapshot live direction probe found
that TMNF's signed `Gas` polarity is negative-forward and positive-backward. The
original bridge used `(throttle - brake) * 65536`, so the two pedal labels were
reversed throughout V2. The final policy's mean raw output was approximately
`[steer=-0.1334, throttle=0.0425, brake=0.9749]`; under the legacy mapping, its
large nominal brake value generated strong negative Gas and was physically
forward acceleration. V2 therefore learned to drive through the mislabeled brake
channel rather than by braking around the course.

Finish rate, reward curves, route progress, oscillation, and hoop-contact
observations remain empirically valid because evaluation reproduces the exact
mapping used during training. V1 and V2 also shared the same mapping, so their
reward-signal comparison remains meaningful. Historical V0/V1/V2 scripts now
opt into an explicit compatibility mode; V3 and future models use corrected
pedal semantics. That necessary control correction means V3 is not literally a
reward-only change at the transport boundary, and the comparison will disclose
it rather than hiding it.

The four-trial direction evidence is recorded in Decision 0002; its ignored
samples hash is
`4CFB82A21827E7F2192E863A69E1DA2B23B7C6A99B8B1440892FA355BEDCA3E1`.

## Implementation verification

`dense_speed_reward` is implemented in `src/trackmania_rl/rewards.py` as an
isolated callable over `RewardTransition`. Unit coverage verifies that normal,
finished, and truncated transitions all return exactly `display_speed / 1000`
with no terminal adjustment.

The live wiring check completed 20 timeout-truncated episodes and 100 steps.
Every returned reward matched the same step's logged displayed speed divided by
1000. Rewards ranged from `0.005` to `0.036`; terminal timeout steps retained
their positive speed reward. All actions were finite/in range and reset
observations were identical. Ignored local evidence hashes:

- action log: `A3FF2542A2C9570824AEDA87F7F501F87D4475C1380EFD575441B9324F6D5FB6`;
- summary: `51DE7425B47298247C2A7E395458C10670E0D2BF1138E755740F6E27E7E5CE57`.

## Host power preflight

Before implementation or training, the active Windows plan was verified as
High performance (`8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`). Automatic sleep and
hibernate timers are both `0` (Never) on AC and battery. No battery or lid-action
setting was exposed on this host. Hybrid sleep is enabled, but cannot trigger
automatically while automatic sleep is disabled; an explicit manual sleep action
could still suspend the host and must be avoided during training.

The display-only timeout remains 900 seconds on AC and 600 seconds on battery.
Turning off the display does not suspend the host or the training process, so it
does not require a system-setting change.

## Training protocol

Decision 0007 fixes a 1,000,000-step minimum with the same seed-42 PPO/gSDE,
environment, and 100x training settings used for the valid reward-v1 run. Only
the injected reward changes. Checkpoints are written every 50,000 steps and
TensorBoard uses the distinct `reward_v2_dense_speed` run name.

After training, evaluate 20 deterministic episodes at 6x. Record finish rate,
episode lengths, reward curve shape, path progress, world distance/displacement,
speed, steering changes, and terminal causes. Review the visible behavior for
looping or oscillation rather than inferring it from reward alone.

## Checkpointed callback-timeout interruption

The first formal attempt stopped at model timestep `117,423` when the bridge
socket remained open but delivered no new synchronous simulation callback for 30
seconds. The project owner subsequently confirmed that the host had slept. This
explains why no automatic timer or application-fault evidence appeared: the
automatic sleep and hibernation timers were still disabled, but those settings
cannot prevent a manual or externally triggered suspend. `TmForever` remained
alive and responsive after wake, while its synchronous callback state did not.

Resume unchanged from the valid 100,000-step checkpoint, replaying 17,423 model
interactions. The preserved attempt-one Monitor contains 357 episodes: one
finish, 151 timeouts, 197 falls, and eight horizontal off-tracks. The finish
occurred after the saved checkpoint, so it is evidence from a discarded training
window and does not establish that the resumed 100k model retains a finishing
policy. Final accounting must disclose the replay and include both TensorBoard
segments. Ignored evidence hashes:

- attempt-one failure manifest: `F7E0865B9EE4843CB54E149FA117CBC06BC6DAF69D7A685504DFE6A0807850A4`;
- attempt-one Monitor copy: `83E4020424E5718AFB9842E39FDB2DAB02E55F4C26BD111B35E0F7E62D16F99D`;
- 100,000-step checkpoint: `F9606CA5F114E48D98BFC47A02A68E69C13B97F301C5B5C6B143CFCDEF6BFF46`;
- attempt-one TensorBoard event: `BF8C2E7BA2E56E1CD5288341AD876E1D4736CD33FE164A7E42B89034CD70D220`.

The first reconnect attempt confirmed that the server-side callback remained
stalled: a new TCP connection opened, but no initial simulation callback arrived
within 30 seconds, so the model collected zero new steps. Cleanup then raised a
secondary `AttributeError` because Stable-Baselines3 had not initialized its
logger before environment reset failed. The original timeout is intact in the
manifest, and logger cleanup now tolerates this pre-setup failure.

Two increasingly explicit callback-reset attempts also collected zero steps:
first sending `Delete` directly to the verified `TmForever` process, then sending
`Escape` followed by `Delete`. Both opened a fresh TCP connection but timed out
before the first callback. This shows that post-sleep recovery requires a full
game/TMInterface restart rather than only restarting the race callback. These
zero-step retries do not add replayed interactions or change the protocol.

- stalled-reconnect manifest:
  `F6C4DB1DFF7D4675BAA6AA297CB333BA29970F4B1D13A35C68F844A4144E7C1B`;
- Delete-only recovery manifest:
  `916465C2F0F7BAF276B804F3E3CCEA5A5326F7BC787805C323840C5D00316EA3`;
- Escape-plus-Delete recovery manifest:
  `0B2D98A240884779979F97A23BCE000FD808395ABA790F385C639BB321EF7E84`.

After the full game/TMInterface restart, attempt five re-established the bridge
and collected `6,144` new interactions before Windows reported socket error
`10053` at model timestep `106,144`. The boundary is diagnostic: `106,144` is
exactly three `2,048`-step PPO rollouts after the checkpoint, and the bridge's
default synchronous response timeout was only two seconds. Environment traffic
stops while PPO optimizes a completed rollout, so a sufficiently long optimizer
pass can be mistaken for a dead Python client. The bridge response timeout is
now set to 30 seconds during its connect handshake. This is a transport
reliability setting; it does not change the 45-second episode timeout or any
reward, termination, action, policy, seed, or training-budget setting. The
`6,144` unsaved interactions will be replayed from the 100k checkpoint.

- rollout-boundary abort manifest:
  `0B43310DFF0068D8D3375F58279F506FAB994C05E280243B609DC69773CFF6A9`;
- cumulative Monitor through attempt five:
  `06ED82229AFD4EE1CB372A4B2556FE62BF79B9AB38B4B07196ADA64D4B1B9B85`;
- attempt-five TensorBoard event:
  `7D2656E33EE8DD00D5CAD51DEF700D6F771A92A8A712E72BDEB6B456167D4F68`.

Attempt six proved the 30-second response timeout across four complete PPO
rollouts, then collected `8,564` interactions and another finish before timing
out at model timestep `108,564`. The visible game state supplied the real cause:
TrackMania had left the active simulation for its post-race results screen. The
environment detected and logged the finish but did not invoke the bridge's
existing `C_PREVENT_SIMULATION_FINISH` command before its next snapshot rewind.
The first corrective attempt invoked that command after Python observed a
finish. This preserved the terminal classification and reward contract, but
later evidence showed that this placement was too late to prevent every game UI
transition. The `8,564` unsaved interactions will also be replayed from the 100k
checkpoint.

- finish-results-screen manifest:
  `FE65A8EB60363B32C808506D8BBAFBC9EE787EB27A449A9B81D46D815625ED06`;
- cumulative Monitor through attempt six:
  `3193A1237CBA4AB2A0B87F3386CE34215A2F6461001A62B9F71ADC8BF374D1BB`;
- attempt-six TensorBoard event:
  `9A6301CF2ACAFC4CEAE44C7CE36A0B4DFB6E62B6C17BABDF764E7DA154B4BEC8`.

Attempt seven initially appeared to validate the late finish protection: it
logged two finishes and continued training after both. It later
timed out inside `reset()` at model timestep `106,308`, after `6,308` unsaved
interactions. TMInterface's log showed lap-count messages during snapshot
rewinds. The bridge was opening those as nested synchronous exchanges while an
outer run-step acknowledgement was already pending; the nested wait can consume
that outer acknowledgement, leaving Python and the plugin waiting on different
message types. Checkpoint and lap counts already exist in every full simulation
snapshot and are not used as separate environment observations, so the bridge
now suppresses those redundant callback exchanges. Run-step and connect
callbacks remain synchronous. Attempt seven's unsaved interactions will be
replayed and included in final accounting.

- nested-callback reset manifest:
  `052F90C0FA08B86055D8DC72D7354A8BC6CD9771E1C032205BDC397DF67B418B`;
- cumulative Monitor through attempt seven:
  `AC5372442FA46A05A05783B9DA38C7EC49027C7F90E6AEB65BF9355D5DAAAC49`;
- attempt-seven TensorBoard event:
  `E6398F5D01B284B0192ECF70E10039DF9A090EF6C42F119BC9D37100569DD1F2`.

Attempt eight connected while the no-opponent dialog was only highlighted, not
confirmed, and collected zero steps. Attempt nine began from a genuinely active
A01 race with the nested callbacks suppressed, collected `3,042` unsaved
interactions, and logged another finish. It then stalled with a visible "new
personal record" modal. This disproved the assumption that a
post-`race_finished` Python command was early enough. TMInterface's API requires
`PreventSimulationFinish` inside `OnCheckpointCountChanged`, where it invalidates
the last checkpoint time before the game stops simulation. The plugin now calls
it there and stores a one-shot finish flag for Python, so Gymnasium still receives
the terminal event while TrackMania never creates results/record UI. Lap and
checkpoint callbacks remain free of nested socket exchanges.

- opponent-dialog manifest:
  `5B69B79B7F7E72D4CB7752CCA77244CF8760A1D986EE822CD8992F06B7A0E678`;
- late-finish-prevention manifest:
  `F349323C757A90AFD0621402D0CFC143E5F6739632A3258EBDB44034606EC19A`;
- cumulative Monitor through attempt nine:
  `5BD2B3F7A8F7D3578DCF8F0DE03A055930D9DA81459867F014D968D9E4FA5181`;
- attempt-nine TensorBoard event:
  `AEB931A694277E3D93177E2F5E3123180C1219447DC2BB3E0B9D8182539E2885`.

## Actual outcome

Training completed at `1,001,120` model timesteps. Across the original run,
replayed recovery windows, and final uninterrupted attempt, the host executed
`1,042,601` observed environment interactions. The final checkpoint is
`checkpoints/reward_v2/final_model.zip`, SHA-256
`400D64EEC5E9AB305A2B26FD1225FC2987A678735AD93CC7694FEFDFBF4BB070`.

The canonical Monitor contains 3,694 completed episodes from every attempt: 725
finishes, 360 timeouts, 2,581 falls, and 28 horizontal off-tracks. Because failed
windows were replayed from the 100,000-step checkpoint, those totals describe
all observed interactions, not the retained model path alone. Counting only the
308 episodes completed before that checkpoint and the 3,273 episodes from the
successful final attempt gives 3,581 retained-path completed episodes: 720
finishes (`20.106%`), 330 timeouts, 2,508 falls, and 23 horizontal off-tracks.
The remaining 113 completed episodes belong to discarded or checkpoint-crossing
windows: five finishes, 30 timeouts, 73 falls, and five horizontal off-tracks.
One 450-step timeout began at step 99,997 and ended at 100,447, crossing the
100,000-step checkpoint; it is not counted as retained because the successful
resume started from the checkpoint before that episode completed. The audit
reconciles 1,001,120 retained model-path interactions, 41,481 replayed/discarded
interactions, and 1,042,601 interactions observed by the host.

The TensorBoard reward curve supplied the dense signal v1 lacked. Its 100-episode
rolling mean rose from `2.883` at the beginning to `65.696` at the end, with a
recorded minimum of `1.413` and maximum of `68.501`. Mean episode length did not
improve monotonically: it started at `179.0`, peaked at `427.52`, and ended at
`272.12` steps. Within the uninterrupted 100k-to-finish segment, the first 100
completed episodes averaged `58.258` reward, `353.51` steps, and six finishes;
the last 100 averaged `65.696` reward, `272.12` steps, and 11 finishes. All
901,120 actions audited during that final segment were finite, inside the declared
space, and applied without hidden clipping.

The final deterministic evaluation ran 20 episodes at 6x. It finished seven
(`35%`), timed out 13, and had no fall or horizontal off-track termination. Finish
times ranged from `29.570` to `32.910` seconds and averaged `30.267` seconds.
Timeouts still reached between `2129.057` and `2179.228` of the reference path's
`2206.528` progress units. All 7,972 evaluation actions were valid. Automated
trajectory analysis classified zero of 20 episodes as an in-place loop or
steering-oscillation speed-farming exploit because average forward-progress
efficiency remained high and the car drove almost the entire route.

A separate 20-episode precision retrospective preserved the original evaluation
and wrote distinct action/summary files. It finished six (`30%`) and timed out 14,
again with no fall or horizontal off-track termination and zero in-place
speed-farming candidates. The fixed precision metrics flagged steering
oscillation in all 20 episodes, inversion in 19, and a physically stuck period in
all 14 timeout episodes. Across the sample the car spent `303.22` seconds upside
down and `209.90` seconds stuck; the longest stuck period was `17.20` seconds.
The mean episode 95th-percentile absolute reference-line offset was `13.507`
units and the worst absolute offset was `18.595` units. Five of six finishes also
included an inversion, showing that finish rate alone hides substantial control
instability. Descriptively combining the two separately preserved evaluation
samples gives 13 finishes in 40 runs (`32.5%`); this is not treated as a new
pre-registered evaluation protocol.

Visible review by the project owner found two repeatable quality failures. On the
straight after the first major 90-degree left turn, the car oscillates left and
right instead of holding a straight line. At the final checkpoint/hoop after the
second left turn, it approaches too far right and clips the structure's
lower-right edge, causing the inversion seen in telemetry. The upside-down car
can remain above the vertical-fall boundary until the 45-second timeout,
explaining why these failures appear as timeouts rather than falls. This
qualitative cause and the orientation/stuck metrics are complementary: telemetry
measures frequency and duration, while the live review identifies the collision
that initiates the failure.

The exact pre-registered prediction was therefore not supported: v2 did not
settle into an in-place loop and did learn a policy capable of finishing A01.
The broader reward-alignment concern was supported. Raw speed alone produced a
fast, near-complete route with steering jitter and unreliable jump alignment,
because it provides no direct incentive for smooth controls, centerline
placement, successful obstacle traversal, or completion. Reward v3 should target
forward centerline progress and stuck detection while retaining explicit
measurement of steering smoothness and jump failures as evaluation metrics rather
than silently changing this completed v2 experiment.

Ignored local evaluation evidence:

- action log SHA-256:
  `B05274B78E918602A06F74A6FE156857406CD2DBAF61ED692CBD3A071DD6490D`;
- evaluation summary SHA-256:
  `B3ABB48ADA4F25B679DD191B7AD7FDF69E36E480630D3499519C0DDA499FACE1`;
- training summary SHA-256:
  `EDE04C118284FF59BF9FB088C84BC0C2D1BB4F56FA8A4339B7E4B00C6F4B8F7A`;
- precision-retrospective action log SHA-256:
  `221B9AE3B5784227DA208D150B43884D1EFE9C9F08F94E4B251EA46DCD2FE970`;
- precision-retrospective evaluation summary SHA-256:
  `9D393B5BEA832B721A98C6762E05BF4E552088392E9351997B4933B81708BA19`;
- nine-replay precision summary SHA-256:
  `86AD9C5F50716CFAE3E917ADFBE5C642997BE8C8EB459AC891A22AE4E2F9E9FD`;
- reconciled V2 retrospective summary SHA-256:
  `7187DEF6E4C79C87B5522158AE92EBDC87B91FC034D42AC05DADA29246D85CCB`.
