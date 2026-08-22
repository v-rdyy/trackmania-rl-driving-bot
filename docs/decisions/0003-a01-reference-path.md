# Decision 0003: A01 reference path from the verified manual lap

Status: Accepted

Date: 2026-08-22

## Context

Phase 1 observations require heading error, lateral offset, and ten look-ahead path points. The project scope permits either extracting track geometry or recording a clean manual lap.

The verified Phase 0 A01 lap contains 361 finite samples over a completed
`33.280`-second run. It was intentionally driven slowly and cautiously for clean
telemetry, not as a PB attempt; the owner's actual A01 PB is `24.5s`. Its 360
consecutive segments are continuous: the largest is `12.355` units, the median
is `7.182`, and there are no respawn or teleport jumps. The source JSON Lines
file SHA-256 is
`C8469209C0A91A1421F052DBF055A900C9A092E27AA8561C230D2A650CBDB0CC`.

## Decision

Build the initial A01 reference path from the verified manual lap. Remove stationary countdown/finish samples and resample the remaining three-dimensional driven line at 5-unit arc-length intervals. Commit the resampled path and provenance metadata so the environment does not depend on a local ignored telemetry artifact at runtime.

Use the horizontal X/Z plane for nearest-path distance, heading, lateral offset, and look-ahead coordinates. Preserve Y in the reference file for future slope-aware observations, but do not treat elevation change as lateral error.

## Tradeoff

This is a driven reference line, not an extracted claim about the map's exact geometric centerline. It is sufficient to prove the Phase 1 loop and matches the simplest approach explicitly allowed by the scope. If the policy later overfits quirks in the human line or accurate road widths become necessary, replace it with geometry-derived data in a new documented decision.

## Reproducibility

`scripts/build_reference_path.py` verifies the source hash, rejects nonfinite/nonmonotonic/teleporting input, and writes both the fixed-spacing CSV and a metadata JSON file.

The verified build produced 443 points over `2206.528335` units. `data/tracks/a01_reference_path.csv` has SHA-256 `0BDD4FDD5FF65E3905A32055CFF34D4200B1A0EA7107618EABD28E699CB19B58`; its metadata file has SHA-256 `D6D40478E4449FD834A7AB8ACE4583D5F4FD97B3B2F4D22FE44E69015572A152`.
