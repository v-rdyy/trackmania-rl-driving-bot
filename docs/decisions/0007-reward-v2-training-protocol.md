# Decision 0007: Reward-v2 training and evaluation protocol

Status: Completed

Date: 2026-08-22

## Context

Reward v2 tests the owner-approved dense speed hypothesis and exact contract in
`reward_v2.md`. It must be trained long enough for a useful or exploitative
behavior to emerge, while changing no policy or environment parameter that would
confound comparison with reward v1.

Reward v1 completed more than 500,000 model steps and developed a stable harmful
behavior. Dense reward supplies signal on every moving step, but may require more
experience to optimize and expose a repeatable speed-farming strategy. The
verified 100x environment has sustained roughly 286 to 314 environment steps per
wall-clock second over long runs.

## Decision

Run reward v2 with this fixed protocol:

- minimum environment-step budget: `1,000,000`;
- Stable-Baselines3 may finish the current 2,048-step rollout, so actual steps
  may exceed the minimum and must be reported exactly;
- seed: `42`;
- algorithm/policy: PPO `MlpPolicy` with bounded squashed gSDE per Decision 0005;
- rollout steps: `2,048`;
- batch size: `64`;
- optimization epochs per rollout: `10`;
- learning rate: `0.0003`;
- gamma: `0.99`;
- GAE lambda: `0.95`;
- PPO clip range: `0.2`;
- environment: 100x simulation, 100 ms control period, 45-second timeout,
  50-unit lateral boundary, and 10-unit local vertical fall boundary;
- reward: exactly `displayed_speed / 1000` on every step, with no finish bonus
  or terminal penalty;
- checkpoint interval: every `50,000` callback steps, plus a final checkpoint;
- TensorBoard run name: `reward_v2_dense_speed`;
- post-training evaluation: 20 deterministic episodes at 6x simulation speed.

The training summary must include reward and episode-length curve extrema,
finish/fall/off-track/timeout counts, checkpoint and TensorBoard hashes, and
bounded-action validation. The deterministic evaluator must log every action and
world position, then report finish rate, episode length, cumulative reward,
progress, distance traveled, start-to-end displacement, speed, steering changes,
and terminal causes.

Qualitative review must specifically look for looping or oscillation. Telemetry
may identify a candidate exploit when the car travels substantial world distance
while gaining little net/path progress, repeatedly changes steering direction,
or ends near a previously visited location. This automated description supplements
rather than replaces the owner's visible observation of the 6x evaluation.

## Tradeoffs

One million steps is twice the reward-v1 model budget and approximately 488 PPO
rollouts. At verified throughput it plans for roughly 55 to 65 minutes. This is
long enough to test for stable dense-reward behavior while reserving the later
multi-million-step budget for the winning reward. Do not stop early merely
because the behavior appears exploitative; the fixed budget keeps the curve and
comparison honest.

Keeping seed, policy, hyperparameters, control period, simulation speed, and
termination rules fixed isolates reward as the intended change. Using 6x for
evaluation makes behavior visible while preserving the same 100 ms game-time
action interval.

## Power preflight

Immediately before training, re-query the active Windows plan and confirm that
automatic sleep and hibernation remain disabled on AC and battery. Do not start
if either timer is nonzero. Manual sleep can still suspend the host and must be
avoided for the duration of the run.

## Mid-run amendment: checkpointed callback-timeout recovery

Attempt one reached model timestep `117,423`, then timed out after 30 seconds
without a new TMInterface simulation callback. The project owner later confirmed
that the host had slept. Automatic sleep/hibernate remained disabled and Windows
recorded no automatic timer or TrackMania application fault, but those controls
cannot prevent a manual or externally triggered suspend. Preserve the failed
segment and resume from the valid 100,000-step checkpoint without changing the
reward, environment, policy, seed, timeout, or training budget.

The 17,423 interactions after the checkpoint must be reported as replayed. The
attempt-one Monitor includes one finish after the checkpoint; preserve it as
observed discarded-window evidence, but do not claim that the resumed checkpoint
contains that behavior. Combine numbered TensorBoard segments and retain the
appended Monitor history in the final quantitative report.

The first resume connected at the TCP layer but received no initial simulation
callback and timed out before collecting a step. This confirms the game-side
synchronous state remained stalled after attempt one. Restart the race callback,
retain the same 100,000-step checkpoint, and retry without a protocol change.
Also make logger cleanup safe when environment reset fails before SB3 initializes
its logger; this reporting fix does not alter training behavior.

Sending `Delete` to the verified game process and then, on a separate retry,
sending `Escape` followed by `Delete` also produced fresh TCP connections but no
simulation callback. Each attempt timed out before collecting a model step. A
full game/TMInterface restart is therefore required after this sleep event; the
failed zero-step recovery attempts remain preserved as experiment evidence and
do not change the training protocol or replay count.

The first post-restart resume collected exactly three more 2,048-step rollouts,
then aborted at model timestep `106,144` as the next rollout began. The bridge's
two-second synchronous response timeout can expire while PPO pauses environment
traffic to optimize a completed rollout. Set the bridge response timeout to 30
seconds during the connect handshake. This operational setting only controls how
long the plugin waits for Python between callbacks; the fixed 45-second episode
timeout and every experimental parameter above remain unchanged. Replay the
6,144 unsaved attempt-five interactions from the same 100,000-step checkpoint
and include them in discarded-step accounting.

Attempt six crossed four rollout boundaries with the longer response timeout,
then exposed a separate success-path defect after its first retained-window
finish: TrackMania eventually entered the post-race results screen, where no
simulation callbacks exist. The initial mitigation invoked
`PreventSimulationFinish` after Python's positive `race_finished` query. PPO
still observed an ordinary terminated episode and the reward contract remained
exact, but later attempts showed that the command could arrive too late to stop
all finish UI. Replay the 8,564 unsaved attempt-six interactions and retain its
finish as discarded-window evidence.

Attempt seven confirmed the finish protection by continuing after two finishes,
then exposed a nested callback deadlock during a later snapshot reset. TMInterface
can invoke checkpoint/lap callbacks from inside `RewindToState` while the plugin
is already synchronously waiting for the outer run-step acknowledgement. The
nested wait can consume that outer acknowledgement and leave each side waiting
for a different response. Suppress the plugin's redundant checkpoint/lap socket
callbacks; their state remains available in every simulation snapshot and the
environment does not use them as separate observations. Preserve and replay the
6,308 unsaved attempt-seven interactions without changing the experiment.

Attempt eight was a zero-step setup failure: the no-opponent choice was
highlighted but not confirmed, so A01 never entered active simulation. Attempt
nine used the installed nested-callback fix, collected 3,042 interactions and a
finish, then visibly stopped on the "new personal record" modal. Per
TMInterface's API contract, move `PreventSimulationFinish` into the plugin's
`OnCheckpointCountChanged` callback, before the game creates finish UI. Store a
one-shot finish flag before invalidating the checkpoint and consume it from the
next `race_finished` request. This retains Gymnasium termination semantics while
keeping TrackMania in active simulation. Replay attempt nine and preserve both
failed manifests.

## Outcome

The final attempt reached `1,001,120` model timesteps and saved checkpoint
`400D64EEC5E9AB305A2B26FD1225FC2987A678735AD93CC7694FEFDFBF4BB070`.
It completed 3,273 episodes after resuming from 100,000 steps and recorded 720
finishes. Combined with the 308 complete episodes retained before that checkpoint,
the retained model path contains 3,581 completed episodes and a `20.106%` training
finish rate. The 41,481 unsaved interactions from interrupted attempts remain
preserved and disclosed but were not learned by the final model.

The fixed 20-episode deterministic evaluation at 6x produced seven finishes,
13 timeouts, no fall/off-track terminations, and zero telemetry-qualified
in-place speed-farming candidates. Visible review found oscillatory steering on
the straight after the first major left and a too-far-right approach after the
second left that sometimes clips the hoop jump and flips the car. Thus the exact
looping prediction was not supported, but the experiment still demonstrated that
speed alone is insufficient for smooth, reliable completion.
