# Trackmania RL Driving Bot — Full Project Scope

**Status:** Phases 0 and 1 complete. Phase 2 reward v1-v4 experiments are
complete. V4 retained V3's reliable finishing and nearly matched the human PB,
but oscillation and severe lateral excursions remain; the next reward-design
decision is not yet approved.
**Owner workflow:** Claude/human as decision layer and reward design, Codex as implementation layer. Codex should not make reward-design or scope decisions unilaterally, flag and ask instead.

**Current evidenced project narrative:**

> Trained a PPO agent to drive TrackMania Nations Forever through a custom Gymnasium environment over TMInterface, using engineered state instead of pixels. Iterated through 4 pre-registered rewards: sparse finish-only produced zero finishes; dense speed reached the final hoop but drove imprecisely; clamped centerline progress trained a reliable finisher; signed progress with time and terminal shaping retained 20/20 deterministic reliability and improved the TMNF race-clock best from 28.030s to 24.900s versus a 24.5s human PB. V4 trained for 1,001,472 additional steps at 100x from the pinned V3 checkpoint. Oscillation still occurred in 20/20 runs and severe lateral deviation worsened, so precision remains active work rather than being hidden by the finish rate.

The paragraph above uses observed artifacts, not projected resume numbers. The
`24.5s` human PB is a real measured comparison baseline. Future training totals
and performance claims must continue to use actual results even when they differ
from earlier planning targets.

The approximately 33-second manual lap used to capture the Phase 0 reference
path was intentionally driven slowly and cautiously for clean telemetry. It was
not a PB attempt and must never be used to claim that an agent nearly matched
human performance.

---

## Phase 0 — Environment & Tooling Verification

**Goal:** Confirm every piece of the stack actually works in isolation before any RL code is written. No training, no env wrapper yet. Just proving the plumbing works.

**Tasks:**

1. Confirm TrackMania Nations Forever is installed and running.
2. Install and confirm TMInterface is working: check current version compatibility against Nations Forever (verify this now, don't assume, TMInterface versioning has shifted before). Flag to human if compatibility is ambiguous or the maintained wrapper landscape has changed.
3. Confirm TMInterface can read live telemetry from a manually-driven lap: speed, position (x/y/z or whatever coordinate system it exposes), car orientation/yaw. Log a few seconds of raw output to a file and manually eyeball it for sanity (values changing as expected, no obvious garbage/NaN/stale values).
4. Confirm TMInterface can accept external input commands (steer, accelerate, brake) and that a scripted "drive straight" test actually moves the car. This is the single riskiest unknown in the whole project, get it working and confirmed before anything else.
5. Set up Python environment: Python 3.10 or 3.11 (verify against current stable-baselines3 compatibility, don't assume), `stable-baselines3`, `gymnasium`, `numpy`, `tensorboard`, TMInterface Python client/wrapper (identify current actively-maintained option, flag choice to human before committing to it).
6. Pick the training track: something short, low-complexity, minimal jumps/walls, ideally a straightish loop. Document the track choice and why (this matters for the "second track = generalization bonus" framing later, phase 1 track should stay easy on purpose).
7. Confirm the game can run at accelerated speed (the target scope assumes 6x accelerated game speed for training efficiency) and that TMInterface telemetry/input still works correctly at that speed, not just 1x.

**Exit criteria (must all be true before Phase 1 starts):**

- [x] Raw telemetry read confirmed reliable over a full manual lap (no dropouts, no garbage values)
- [x] Scripted input successfully drives the car (at minimum: accelerate + steer produces expected car movement)
- [x] Accelerated game speed confirmed working with both telemetry and input
- [x] Track chosen and documented
- [x] Python env installed with pinned versions in a requirements file

**Deliverable:** A short markdown note (`phase0_notes.md`) documenting what was confirmed, what wrapper/library was used and why, TMInterface version, and any gotchas hit. This becomes the setup section of the eventual README.

---

## Phase 1 — Environment Wrapper (Prove the Loop Runs)

**Goal:** A working Gymnasium environment wrapping TMInterface, with a trivial reward, that PPO can train against end-to-end without crashing. Not a good agent, just a real loop.

**State space (engineered, not pixels, per target scope):**

- Speed
- Car-frame velocity (velocity decomposed into the car's own reference frame, not world frame, this matters for the model to learn turning behavior correctly)
- Heading error (angle between car's current heading and the track's intended direction at that point)
- Lateral offset (distance from track centerline)
- 10 look-ahead centerline points (sampled points along the upcoming track centerline, giving the agent a sense of what's coming, similar to how a human looks ahead while driving)

Document the exact shape/normalization of this observation vector in code comments, this will need to be explained in interviews.

**Action space:**

- Continuous steer/throttle/brake, approved by the owner and specified in `docs/decisions/0002-continuous-action-space.md`. The Phase 1 smoke test logs and validates every raw action before analog conversion.

**Tasks:**

1. Build `reset()`: restart/respawn the car at track start, clear any internal state (stuck counters, episode timers).
2. Build `step(action)`: send input to TMInterface, read new telemetry, compute observation vector, compute reward (trivial for now, e.g. +speed each tick), check termination (crash, timeout, or finish).
3. Build the centerline representation: this needs a reference path for the chosen track (recorded from a manual clean lap, or extracted from track geometry, decide approach and document it) so lateral offset and look-ahead points are computable.
4. Wire the owner-approved disposable smoke reward from `docs/decisions/0004-phase1-smoke-reward.md`: normalized display speed with a timeout/off-track penalty and no finish bonus. This is explicitly not a formal reward iteration, just enough for PPO to have gradient signal.
5. Connect stable-baselines3 PPO to the env, run a short training job (a few thousand steps, not the full 6M) purely to confirm nothing crashes, tensorboard logs show up, and reward is changing over time (even if it's not learning anything good yet).

**Exit criteria:**

- [x] `reset()` and `step()` work reliably over at least 20 consecutive episodes without the env hanging or erroring
- [x] PPO training runs for a short smoke-test duration without crashing
- [x] Tensorboard shows reward/episode-length curves, confirming logging works
- [x] Observation vector shape and each component documented in code

**Deliverable:** Working env code, committed in small pieces (telemetry reader, input sender, centerline/reference path builder, env wrapper, training smoke-test script), each with its own commit. No giant single commit.

---

## Phase 2 — Reward Iteration (the actual project)

**Goal:** Produce the 3 documented reward versions from the target scope, each with a pre-registered hypothesis, a real training run, and a documented failure or success analysis. This phase is the core resume story, don't rush it.

**Reward v1 — Sparse finish-only**

- Hypothesis (write this down before training, not after): reward only on finishing the track will be too sparse for PPO to find signal in a reasonable number of steps.
- Train it. Expected/target outcome per the resume scope: never learns (never finishes, reward stays near zero/flat).
- Document actual outcome, whether or not it matches the hypothesis. If it does learn, that's also worth documenting, the scope doesn't get to just be right by default, real results win.

**Reward v2 — Dense speed reward**

- Hypothesis: rewarding speed every tick will give denser signal but risks reward hacking, since speed isn't the same as progress.
- Train it. Target outcome per scope: agent exploits it by looping/oscillating to farm speed without ever finishing.
- Actual outcome: the final policy finished 7/20 deterministic episodes and did
  not loop in place. It oscillated on the straight after the first major left and
  sometimes clipped the lower-right edge of the hoop jump after the second left,
  flipping onto its roof. Preserve this mismatch between prediction and result.
- Document actual outcome and, if reward hacking occurs, capture a specific example (a short clip, telemetry snippet, or description of the exploit behavior). This is the best interview story in the whole project, don't lose it.

**Reward v3 — Clamped progress-along-centerline with stuck detection**

- Hypothesis: rewarding progress along the track's centerline (not raw speed) directly targets what you actually want (progress toward finishing) and clamping prevents exploiting large single-tick jumps; stuck detection (penalize/terminate if no forward progress for N ticks) prevents the v2 failure mode of oscillating in place.
- Train it. Target outcome: this is the version that actually produces a finishing agent.
- V3 is explicitly reliability-only. It has no speed or time term and is not
  expected to approach the `24.5s` human PB yet. First determine whether it can
  finish without flipping or remaining stuck; pace is a separate later problem.
- Actual outcome: the original detector reported 12/20 finishes and eight final
  jump falls. Replay inspection proved that the immediate vertical cutoff also
  stopped moving, recoverable trajectories. One corrected re-evaluation of the
  unchanged checkpoint finished 20/20 (race-clock best `28.030s`, mean
  `28.106s`) with no falls, stuck periods, timeouts, or inversions. Steering
  oscillation remained in 20/20 runs, so V3 solved reliability but not precision.

**Tasks per version:**

1. Write the hypothesis in a versioned doc (`reward_v1.md`, `reward_v2.md`, `reward_v3.md`) before training starts.
2. Implement the reward function as its own isolated, swappable component (don't hardcode into the env, make it a passed-in function/config so versions are easy to diff and re-run).
3. Train for a meaningful number of steps, enough to see real behavior emerge, not just a smoke test. Log everything to tensorboard under a distinct run name per version.
4. After training, evaluate qualitatively (watch it drive, does it finish, does it exploit something weird) and quantitatively (finish rate over N episodes, average episode length, reward curve shape).
5. Write the failure/success analysis in the versioned doc: what happened, why, and what it motivates for the next version.

**Exit criteria:**

- [x] All 3 reward versions implemented as swappable, isolated functions
- [x] All 3 trained with logged runs
- [x] All 3 have a pre-registered hypothesis doc and a post-hoc analysis doc
- [x] The actual failure modes are documented with specifics (numbers, behavior descriptions), not vague summaries

**Deliverable:** `reward_v1.md`, `reward_v2.md`, `reward_v3.md`, reward function code for each version, tensorboard logs per run, and a short comparison summary once all 3 are done.

### Deferred direction after reliable finishing

Do not begin this work during V3. Once a reward reliably finishes A01, test a V4
that reintroduces time efficiency or speed on top of the winning reliability
reward. After that, investigate which advanced techniques can emerge through
reward shaping rather than scripted controls: begin with optimizing the first
drop to minimize airtime and maximize acceleration time, then consider a harder
speed-slide experiment. Record which behaviors emerge unaided, which require
stronger shaping or curriculum, and which are not discovered; that comparison is
a project result even if some techniques never emerge.

Actual V4 outcome: signed progress, a per-step time cost, finish bonus, and
decisive failure penalty retained 20/20 deterministic finishes and produced a
`24.900s` race-clock best (`24.929s` mean), `0.400s` behind the human PB. It
cleaned up the reviewed final-hoop approach but did not remove oscillation; fixed
metrics still flagged 20/20 episodes and maximum lateral deviation rose to
`17.833` units. The next precision experiment should be scoped from that evidence
rather than treating V4 as a fully solved driver.

---

## Phase 3 — Full Training, Evaluation, Reproducibility

**Goal:** Take the best evidence-backed reward forward after the V4 precision
decision and produce honest final numbers.

**Tasks:**

1. Full training run at target scale: aim for ~6M steps at 6x accelerated game speed per the target scope. Treat 6M/6x/~14h as a planning target, not a hard requirement, actual numbers depend on hardware and how training goes. If it converges earlier or needs more, that's fine, just document what actually happened.
2. Checkpoint regularly during training (not just at the end) and tag checkpoints so any result is reproducible from a specific saved model.
3. Run a deterministic evaluation: target scope uses a 20-episode deterministic eval for finish rate. Also record best lap time achieved.
4. Compare against the owner's actual `24.5s` A01 personal best. Do not use the
   intentionally cautious approximately 33-second telemetry lap as a performance
   benchmark.
5. Write up final real results: finish rate, best lap vs human PB, total training steps, wall-clock time, hardware used.
6. Confirm reproducibility: from a tagged checkpoint, can you reload the model and get consistent eval results? Document the process for doing so.

**Exit criteria:**

- [ ] Full training run completed and logged
- [ ] Deterministic eval run (20 episodes minimum) with finish rate recorded
- [ ] Best lap time recorded and compared to a real human PB lap
- [ ] Checkpoints tagged and reload/reproducibility confirmed
- [ ] Final numbers are real, not copied from the projected resume, if they differ from the projection, the real numbers are what go on the resume

**Deliverable:** Final training run artifacts, eval results doc, and an updated resume bullet written from actual numbers (human should do this last step, not Codex).

---

## Phase 4 (Bonus, optional, do not block on this) — Second Track Generalization

Only pursue if time allows after Phase 3 is genuinely done. Train or evaluate the Phase 3 agent on a second, different track to test generalization. This was explicitly scoped as a bonus, not a requirement, in the original project scope. Skipping it does not compromise the core resume claim.

---

## Cross-cutting rules for every phase

- **Commit discipline:** small, scoped commits per component. Commit messages describe what works, not "wip" or "changes."
- **No fabricated numbers, ever:** every number in the target end state paragraph is a target, not a script. Codex should report real results even when they don't match the projection, and flag clearly if a phase's actual outcome diverges from what was hypothesized.
- **Flag ambiguity, don't guess silently:** TMInterface version compatibility, wrapper library choice, continuous vs. discrete action space, and reward hyperparameters are all decisions that should be surfaced to the human, not made silently.
- **Reward design stays collaborative:** Phase 2 is the core intellectual work of this project. Codex should implement and train what's specified, but hypothesis-writing and interpretation of failures should loop the human in, not be resolved unilaterally.
