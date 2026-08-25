# Decision 0011: V4 A02 zero-shot generalization protocol

## Status

Blocked before measurement on 2026-08-24. No V4 episode ran on A02, so there is
no generalization score and no result to interpret.

## Question

Does the finished V4 policy transfer at all from A01-Race to a different,
similarly introductory TMNF track without retraining or adaptation?

## Decision

Use the built-in `A02-Race` as the second track. It is adjacent to A01 in the
Nations White campaign but has different turn geometry. Run the already-finished
V4 checkpoint for 20 deterministic episodes at 6x simulation speed. Do not train,
fine-tune, select checkpoints, or change actions based on A02 results.

The model checkpoint is frozen as:

- `checkpoints/reward_v4/final_model.zip`
- SHA-256 `6DF90018CEC877796F6865BB6CB8D1A86929D84B6826642D26001DC5871C63F2`
- model timestep `2,002,944`

Build A02's reference path from the installed official Nadeo author replay, not
from an agent attempt. Pin the installed source files before extraction:

- `A02-Race.Challenge.Gbx`: SHA-256
  `DCBB1376DCBD10A6018E26D6991EF0155717FA27A57AED27236B95A5876A9D80`
- `A02-Race.Replay.gbx`: SHA-256
  `7546E19CE9CA0D36E074406256A547A256F20B23602985085E98082EE7D178E3`

TMInterface 2.2.1's direct replay-file `dump_inputs` support will extract the
author inputs. Replay those inputs at 100 ms telemetry intervals, require a real
finish and a monotonic race clock, then resample the driven positions at the same
5-unit spacing used for A01. Preserve the extracted inputs, raw telemetry,
reference path, hashes, and provenance manifests.

The evaluation will retain the same action audit, replay retention, finish/fall/
stuck classification, lateral-deviation metrics, and oscillation detection used
for V4 on A01. Results are descriptive; there is no pass gate and no A02 retry
based on outcome.

## Interpretation boundary

This is zero-shot **policy transfer over the same engineered observation type**.
The policy receives A02-relative centerline progress, heading, lateral offset,
and look-ahead geometry just as it received A01-relative values during training.
It is not raw-pixel generalization and it is not navigation without a supplied
reference line. A failure still demonstrates A01 specialization; a success only
supports transfer under this explicitly shared representation.

## Why A02

A02 is a stronger controlled comparison than a distant or advanced track:

- beginner campaign placement limits the difficulty jump;
- its geometry differs enough to expose A01-specific steering behavior;
- the installed official replay supplies reproducible, first-party geometry;
- no external leaderboard run or manually chosen racing line is required.

No V5 checkpoint will be used for this measurement. V5 remains the separate,
single-variable steering-smoothness experiment requested by the owner.

## Blocked setup record

The installed challenge and replay hashes matched the pre-registration, and the
bridge was live-verified at the main menu. The extraction workflow then tried
TMInterface 2.2.1's direct replay-file command with both forms below:

```text
dump_inputs a02_nadeo_author.Replay.Gbx a02_nadeo_author.txt
dump_inputs "C:\Program Files (x86)\Steam\steamapps\common\TrackMania Nations Forever\GameData\Tracks\Campaigns\Nations\White\A02-Race.Replay.gbx" a02_nadeo_author.txt
```

In both cases the bridge accepted and acknowledged `SCOnConnectSync`, but
TMInterface created no output file in `Documents\TMInterface\Scripts` within
the fixed 30-second verification window. Earlier versions of the extractor also
waited unnecessarily for a race-step callback; replacing that with a menu-only
command connection resolved the handshake ambiguity but exposed the same
output-file failure. No reference telemetry or A02 centerline was fabricated,
and the frozen V4 evaluator was therefore not started.

Several launcher hardening findings were kept separately in git history: a TCP
readiness probe was consuming a real `python_link.as` client connection, fresh
launches injected focus-dependent Enter presses at the main menu, and a proposed
queued-handshake plugin recovery did not fix the live failure. The unsuccessful
plugin change was removed from the active installed copy. These issues are not
reported as A02 policy performance.

To resume this decision, first establish the exact TMInterface 2.2.1 replay-path
syntax or obtain a known-good A02 input script, then create and checksum the A02
reference path before any V4 episode. Do not substitute A01's centerline or let
V4 attempt A02 without track-relative geometry, because either would change the
meaning of the test.
