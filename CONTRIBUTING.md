# Contributing

## Commit discipline

The Git history is part of the project's evidence. It must show the real sequence of working components, decisions, and failed experiments.

1. Make one commit per logical unit of work. For example, a telemetry reader, input sender, and configuration file are separate commits when they are separate completed units.
2. Write plain-language commit messages that say what changed and why. Avoid generic messages such as `updates`, `wip`, or `phase 1 stuff`.
3. Commit at natural checkpoints after a component works and its relevant checks pass. Do not create trivial commits to simulate activity.
4. Preserve failed experiments. Commit them on a clearly labeled branch or with an honest message, especially for the three planned reward iterations.
5. Use real commit times. Never backdate commits or manufacture an artificial development cadence.
6. Reference the relevant phase or decision document when it explains the implementation, such as `per Decision 0001` for the approved TMInterface bridge.

Do not combine a session's unrelated completed units into a single catch-all commit.

## Progress reporting discipline

Every progress report to the project owner includes a concise **Notable moments**
section. It may say that nothing qualifies, but it must not be silently omitted.
Capture enough specifics to reconstruct the project narrative later:

1. Debugging stories with the observed failure, its actual cause, the diagnostic
   evidence that identified it, and the fix.
2. Design decisions that weighed reasonable alternatives, including why the
   selected tradeoff fit this project.
3. Behavior that differed from expectations, whether helpful or harmful.
4. Uncertainty or judgment calls that could benefit from another perspective,
   with one sentence explaining why.

Commit messages still describe the code or documentation change. Notable moments
record why the work took its eventual shape and what was learned along the way.
