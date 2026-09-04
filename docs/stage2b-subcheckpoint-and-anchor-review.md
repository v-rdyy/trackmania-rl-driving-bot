# Stage 2b: missing sub-checkpoint timeline and anchoring feasibility

Date: 2026-09-04. Read-only follow-up; training remains paused.

## 1. What historical timeline is actually available?

The requested 10k, 20k, 30k, and 40k Stage 2b policies do not exist in the
inspected project evidence. The accepted run's save interval was 50,000
interactions, not 10,000. This is an observability limitation of the completed
experiment, not an unsuccessful attempt to load existing files.

The inventory covered the checkpoint tree, workspace model/archive filenames,
Stage 2b run files and TensorBoard event contents, and the existing backup ZIP.
The latter is the pre-V3 V2 backup and contains no Stage 2b members. TensorBoard
contains scalar update statistics but no tensors, histograms, or policy history.
The rejected missing-final-audit run also has only 50k/final files; those are
different training-attempt artifacts and were not substituted into this run.

| Additional interactions | Historical policy available? | What can be concluded |
| --- | --- | --- |
| 0 | Yes: original Stage 2 Gate 500k | Frozen reliable baseline |
| 10k / 20k / 30k / 40k | No | No trajectory or action timeline at these points |
| 50,000 observed; learned through 49,152 | Yes: periodic checkpoint, 24 updates | Same-state pre-zone action drift is already present |
| 51,200 | Yes: final checkpoint, 25 updates | Previously measured shifted trajectory and 0/10 deterministic finishes |

The distinction at 50,000 matters: the callback saves during rollout collection,
before its PPO update. The snapshot's last audit row is timestep 3,555,328,
or 49,152 interactions learned beyond the 3,506,176 starting point. The final
checkpoint incorporates rollout 25. The filename's observation count alone
must not be treated as completed optimization.

### Additional measurement on the surviving checkpoint

Using the original direct-live baseline states, rebuild each observation and
query all three frozen policies. The baseline prediction matches the next-row
logged action to within `7.16e-7`. No new rollouts or replay-generated telemetry
are involved. Mean absolute steering differences from the original policy:

| Pre-zone progress band | After update 24 | After update 25 |
| --- | ---: | ---: |
| 0-300 | 0.03795 | 0.03463 |
| 300-600 | 0.02041 | 0.03519 |
| 600-900 | 0.01319 | 0.01852 |
| 900-1100 | 0.01590 | 0.02584 |
| All pre-zone states, sample-weighted | 0.02669 | 0.03075 |

These are action units on the steering range [-1, 1], **not trajectory offsets**.
The final update did not originate pre-zone action drift. It reduced the
average discrepancy in the earliest band and increased it in the others;
the path of earlier updates remains unknown. The after-update24 policy has not
been driven in this follow-up, so no new finish rate or trajectory is claimed.

**Timeline conclusion: onset of pre-zone action change is bounded only to
somewhere in the first 24 updates. Gradual drift versus a sharp break cannot
be resolved from the retained evidence.** Aggregate losses, rewards, KL values,
or endpoint Adam moments cannot reconstruct the missing weights. Weight
interpolation would not be a historical checkpoint. Repeating training would
be a separately labeled replication, not recovery of the original timeline,
and has not been authorized or performed here.

Even a genuinely gradual action timeline would not alone establish shared
representation interference: gradual control changes can cross a narrow
geometric safety boundary abruptly. The present evidence supports global policy
change and recurring alignment fragility, not a proven gradient-level cause.

## 2. Is a localized action anchor feasible?

**Yes, as an auxiliary optimization loss, not a new in-game reward or a scripted
control override.** A frozen copy of Gate 500k can act as the teacher. Unlike
the rollout-relative KL early-stop, this teacher stays fixed across updates.
Policy distillation provides precedent for using a trained policy as a target;
the localized protection described here is our proposed adaptation, not a
published guarantee for Trackmania. See [Policy Distillation](https://arxiv.org/abs/1511.06295)
and [Distilling Policy Distillation](https://arxiv.org/abs/1902.02186).

A candidate objective, **not pre-registered or implemented**, is:

```text
L_total = L_existing_PPO + lambda_anchor * L_anchor
L_anchor = mean_over_protected_states(
    mean_over_actions((a_current(state) - a_Gate500k(state))^2)
)
```

Here `a` is the deterministic policy output, with steering, throttle, and brake
all normalized to [-1, 1]. Targets are the teacher's actions on the **same
observation**, not a prerecorded steering sequence indexed by race time.
The teacher is frozen; only the student receives gradients. The V4/graduated
environment reward would remain unchanged, but the training objective would
change, so this must be labeled a new controlled variant.

### Where protection must apply

Anchoring only the late 2000-2150 approach is poorly matched to the diagnosed
mechanism. Most of that span is airborne: launch direction was set earlier,
and the original live logs show almost unchanged horizontal velocity heading
through the flight. Copying the old action on a new, already-misaligned airborne
state cannot be assumed to repair that trajectory.

A useful candidate therefore protects the post-corner **grounded launch setup**
as well as the landing/final correction. The measured 1410-to-about-1710 setup
and roughly 2120-2175 landing/structure spans are candidate localization clues,
not frozen training boundaries. The existing 1100-1410 bonus zone would not
receive direct imitation loss. No boundary or coefficient is selected here.

Even with non-overlapping masks, shared network weights receive both losses,
so this is not a guarantee that learning stays confined to a geographic zone.
The aim is to oppose forgetting at selected states while permitting learning
elsewhere, not to independently freeze a spatial part of the neural network.

### State coverage and integration

- A fixed rehearsal buffer of protected states from the original successful
  live episodes would keep the original behavior represented even if the
  student stops visiting those states. Existing logs reconstruct the required
  26-value observation accurately; no human driving demonstration is needed.
- Protected states encountered by the student could also be queried against
  the teacher, to cover deviations from that original route. Teacher behavior
  on unfamiliar states is not validated merely because it can be queried.
  The buffer mixture, state coverage, held-out episodes, and safety gates would
  require an explicit protocol before a training test.
- The current audited PPO implementation already constructs its differentiable
  loss before `backward()`. The additional term can be integrated there using
  the policy distribution's differentiable deterministic output, not the
  no-gradient `predict()` convenience method. No algorithm replacement or
  environment protocol change is required. The teacher and small state buffer
  add computation, but do not require extra simulator interactions per update.

This is a code-level feasibility assessment; no regularization term has been
added to the optimizer and no efficacy experiment has been run.

### Why action matching rather than another distribution KL?

Direct deterministic-action matching targets the observed mean-action failure
and can retain more freedom to explore than matching the teacher's entire
distribution. Full teacher KL is also technically possible, but would constrain
the gSDE distribution as well, potentially resisting the exploration/behavior
change being sought. The candidate action loss does **not** explicitly preserve
exploration variance, which would still need auditing. Squashed outputs also
have smaller gradients near saturation. These are deliberate tradeoffs, not
claims that mean-action matching is guaranteed superior.

Both differ from the existing rolling KL guard, whose reference is the current
rollout's behavior policy; its early-stop is not a persistent teacher anchor.
The installed implementation was inspected directly; the corresponding
[official PPO source](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html)
shows the same distinction.

### Important limits

1. Matching actions does not match entry position, velocity, or heading. A
   changed slide exit can carry a different state into the protected section.
2. A soft loss cannot guarantee clearance. Live launch-vector, landing-line,
   and deterministic finishing measurements remain essential.
3. The teacher itself finished 9/10, not 10/10 in this baseline. Anchoring can
   preserve an existing weakness; it does not establish perfect reliability.
4. Tight anchoring can prevent the changed corner exit or launch trajectory
   needed for a useful speedslide. Loose anchoring may not protect enough.
   This is the central discovery-versus-preservation tradeoff.
5. It would be a mitigation test, not proof that the graduated bonus caused
   the original failure. No next training configuration is proposed here.

## 3. Recurring structural fragility, with historical qualifications

This is the **third distinct recurrence** in the requested sequence at the
same final-alignment/structure section: V3's original collision/low-hoop issue,
Stage 2's optimizer-associated collapse, and Stage 2b's launch-line regression.
It warrants treatment as recurring structural fragility in the policy's handling
of that section, not three unrelated incidents. This does not mean three
identical causal mechanisms have been established.

In particular, V3's original 12/20 finish result was confounded by the faulty
fall detector; the unchanged V3 policy finished 20/20 after detector correction.
The awkward low-hoop path and original collision concern remain relevant, but
V3 must not be relabeled as a confirmed optimizer collapse or a genuine 40%
physical-failure rate. Stage 2 and Stage 2b have separate, directly measured
deterministic collapses at the same vulnerable final section.

## Notable moments and evidence

- Finer reporting cadence does not imply finer saved models. The 50k checkpoint
  setting prevents the requested 10k-40k retrospective timeline; no evidence
  was synthesized to fill it.
- The intermediate filename overstates how far optimization had progressed:
  50,000 observed interactions versus 49,152 learned through. The new timing
  tests distinguish those counts.
- Protecting only the visually obvious impact section would be too late for
  the measured launch-angle mechanism. Earlier setup coverage is a substantive
  design choice, not a boundary tweak to make a result look better.

Reproduce the inventory and surviving-policy probes with
`.venv\Scripts\python.exe scripts/audit_wr_stage2b_subcheckpoints.py`.
Derived audit: `artifacts/analysis/wr_chase_stage2b/subcheckpoint_audit.json`.
SHA-256: `456E22EB5C660F5C4045786E342D09B822661B7316047893E4B3AB028B036B2F`.
The earlier alignment diagnosis and its hash-pinned artifact are unchanged.
No new driving episodes, finish times, replays, or videos were produced.
