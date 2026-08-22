# Decision 0008: Precision and failure metrics

Status: Pre-registered before retrospective V2 replay analysis

Date: 2026-08-22

## Context

Reward V2's original evaluator measured route efficiency and screened for
in-place speed farming. The owner then identified subtler visible failures: left
and right steering oscillation on a straight, excessive right-side deviation at
the hoop, occasional inversion, and long timeouts while the car could no longer
make useful progress. Finish rate and terminal labels do not quantify those
failures.

The saved TMInterface input replays preserve exact signed steer/Gas commands.
Checksum-verified round-trip playback can add orientation and motion telemetry
without changing the completed V2 policy or regenerating its actions.

## Decision

Add the following evaluator-only metrics and fix their thresholds before running
the full V2 replay archive:

- steering oscillation: normalize TMInterface steer to `[-1, 1]`, use `0.10`
  hysteresis for left/right sign crossings, ignore steering deltas below `0.05`
  for direction reversals, and flag oscillation at four or more hysteresis
  crossings inside any 2.0-second window;
- lateral deviation: report mean absolute, RMS, 95th-percentile absolute, and
  maximum absolute reference-line offset, plus seconds beyond 5, 10, and 20
  units;
- upside down: use the world-Y component of the car's up axis and count periods
  where its cosine is below `0.0`, reporting entries, total time, and longest
  duration;
- stuck: use a rolling 2.0-second window and mark a candidate only when
  centerline-progress gain is below 1.0 unit and actual world path length is
  below 2.0 units. Report each reconstructed period, total time, and longest
  duration.

Run these metrics against every existing successful V2 input replay: the 50k,
250k, 500k, and 750k checkpoints plus all five final-model episodes. Preserve
per-episode results and checkpoint-stage aggregates. Do not select only visually
interesting replays.

## Tradeoffs

The oscillation flag intentionally measures repeated control direction changes,
not whether oscillation harmed the lap. Lateral metrics provide the trajectory
context separately. The stuck definition requires low world motion as well as
low progress, so a fast loop is not mislabeled as physically stuck; looping
remains covered by the original route-efficiency metric.

The reference path is a driven centerline proxy, not a surveyed road center, so
lateral offsets are comparative precision measurements rather than claims about
legal track limits. Upside-down detection uses orientation instead of a visual
guess. All thresholds are frozen for this retrospective pass and will not be
retuned after viewing the V2 numbers.
