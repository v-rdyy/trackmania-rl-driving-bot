"""Frozen live-physics scoring for the Stage 2b exploration ablation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

FIRST_TURN_ZONE = (680.0, 930.0)
FINAL_CORNER_ZONE = (1_100.0, 1_410.0)
MIN_DISPLAY_SPEED = 400.0
MIN_GROUND_CONTACTS = 3
MIN_UPRIGHT_COSINE = 0.8
MAX_ABS_HEADING_ERROR = math.pi / 4.0
MAX_ABS_LATERAL_OFFSET = 20.0
MAX_ALLOWED_ONE_STEP_SPEED_LOSS = 25.0
NEAR_MISS_MIN_ABS_SLIP_DEGREES = 0.50
LEGACY_MIN_DISPLAY_SPEED = 350.0
LEGACY_MIN_ABS_SLIP_DEGREES = 1.0
LEGACY_MIN_ABS_YAW_RATE = 0.25
SAMPLE_PERIOD_MS = 100

Record = Mapping[str, Any]
Run = tuple[int, int]


def _finite(record: Record, *fields: str) -> bool:
    try:
        return all(math.isfinite(float(record[field])) for field in fields)
    except (KeyError, TypeError, ValueError):
        return False


def _in_zone(record: Record, zone: tuple[float, float]) -> bool:
    return zone[0] <= float(record["progress"]) <= zone[1]


def valid_slide_onset_sample(
    record: Record,
    *,
    zone: tuple[float, float] = FINAL_CORNER_ZONE,
) -> bool:
    """Return whether one row satisfies the frozen grounded-forward envelope."""

    scalar_fields = (
        "progress",
        "display_speed",
        "new_high_water_progress_delta",
        "upright_cosine",
        "heading_error",
        "lateral_offset",
    )
    if not _finite(record, *scalar_fields):
        return False
    return bool(
        _in_zone(record, zone)
        and float(record["display_speed"]) >= MIN_DISPLAY_SPEED
        and int(record.get("ground_contact_count", -1)) >= MIN_GROUND_CONTACTS
        and float(record["new_high_water_progress_delta"]) > 0.0
        and float(record["upright_cosine"]) >= MIN_UPRIGHT_COSINE
        and abs(float(record["heading_error"])) <= MAX_ABS_HEADING_ERROR
        and abs(float(record["lateral_offset"])) <= MAX_ABS_LATERAL_OFFSET
    )


def _is_consecutive(previous: Record, current: Record) -> bool:
    try:
        return (
            int(current["race_time_ms"]) - int(previous["race_time_ms"])
            == SAMPLE_PERIOD_MS
        )
    except (KeyError, TypeError, ValueError):
        return False


def _contiguous_runs(
    records: Sequence[Record], predicate: Callable[[Record], bool]
) -> list[Run]:
    runs: list[Run] = []
    start: int | None = None
    previous_index: int | None = None
    for index, record in enumerate(records):
        if predicate(record):
            if (
                start is None
                or previous_index is None
                or not _is_consecutive(records[previous_index], record)
            ):
                if start is not None and previous_index is not None:
                    runs.append((start, previous_index))
                start = index
            previous_index = index
        elif start is not None and previous_index is not None:
            runs.append((start, previous_index))
            start = None
            previous_index = None
    if start is not None and previous_index is not None:
        runs.append((start, previous_index))
    return runs


def _window_summary(records: Sequence[Record], run: Run) -> dict[str, Any]:
    start, end = run
    window = records[start : end + 1]
    return {
        "start_index": start,
        "end_index": end,
        "sample_count": len(window),
        "duration_ms": len(window) * SAMPLE_PERIOD_MS,
        "start_race_time_ms": int(window[0]["race_time_ms"]),
        "end_race_time_ms": int(window[-1]["race_time_ms"]),
        "start_progress": float(window[0]["progress"]),
        "end_progress": float(window[-1]["progress"]),
        "minimum_speed": min(float(row["display_speed"]) for row in window),
        "maximum_speed": max(float(row["display_speed"]) for row in window),
        "maximum_sliding_wheels": max(
            int(row.get("sliding_wheel_count", 0)) for row in window
        ),
        "maximum_abs_slip_angle_degrees": max(
            abs(float(row["slip_angle_degrees"])) for row in window
        ),
        "maximum_abs_body_up_yaw_rate": max(
            abs(float(row["body_up_yaw_rate"])) for row in window
        ),
    }


def _failure_flag(record: Record) -> bool:
    return any(
        bool(record.get(field, False))
        for field in ("timeout", "off_track", "fallen", "stuck", "truncated")
    )


def _passes_surrounding_safety(records: Sequence[Record], run: Run) -> bool:
    start, end = run
    context_start = max(0, start - 1)
    context_end = min(len(records) - 1, end + 1)
    for index in range(context_start, context_end + 1):
        record = records[index]
        if not _finite(
            record,
            "display_speed",
            "new_high_water_progress_delta",
            "upright_cosine",
            "heading_error",
            "lateral_offset",
        ):
            return False
        if (
            int(record.get("ground_contact_count", -1)) < MIN_GROUND_CONTACTS
            or float(record["new_high_water_progress_delta"]) <= 0.0
            or float(record["upright_cosine"]) < MIN_UPRIGHT_COSINE
            or abs(float(record["heading_error"])) > MAX_ABS_HEADING_ERROR
            or abs(float(record["lateral_offset"])) > MAX_ABS_LATERAL_OFFSET
            or _failure_flag(record)
        ):
            return False
        if index > 0 and _is_consecutive(records[index - 1], record):
            speed_delta = float(record["display_speed"]) - float(
                records[index - 1]["display_speed"]
            )
            if speed_delta <= -MAX_ALLOWED_ONE_STEP_SPEED_LOSS:
                return False
    return True


def _raw_slide_onset_runs(
    records: Sequence[Record], zone: tuple[float, float]
) -> list[Run]:
    runs = _contiguous_runs(
        records,
        lambda row: valid_slide_onset_sample(row, zone=zone)
        and int(row.get("sliding_wheel_count", 0)) >= 1,
    )
    return [
        run
        for run in runs
        if run[1] - run[0] + 1 >= 2
        and max(
            int(records[index].get("sliding_wheel_count", 0))
            for index in range(run[0], run[1] + 1)
        )
        >= 2
    ]


def _overlaps(left: Run, right: Run) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _legacy_runs(
    records: Sequence[Record],
    *,
    zone: tuple[float, float],
    minimum_sliding_wheels: int,
    minimum_samples: int,
) -> list[Run]:
    runs = _contiguous_runs(
        records,
        lambda row: bool(
            _finite(
                row,
                "progress",
                "display_speed",
                "slip_angle_degrees",
                "body_up_yaw_rate",
            )
            and _in_zone(row, zone)
            and float(row["display_speed"]) >= LEGACY_MIN_DISPLAY_SPEED
            and int(row.get("ground_contact_count", -1)) >= MIN_GROUND_CONTACTS
            and int(row.get("sliding_wheel_count", 0))
            >= minimum_sliding_wheels
            and abs(float(row["slip_angle_degrees"]))
            >= LEGACY_MIN_ABS_SLIP_DEGREES
            and abs(float(row["body_up_yaw_rate"])) >= LEGACY_MIN_ABS_YAW_RATE
        ),
    )
    return [run for run in runs if run[1] - run[0] + 1 >= minimum_samples]


def score_zone(
    records: Sequence[Record],
    *,
    zone: tuple[float, float],
) -> dict[str, Any]:
    """Score one episode in one measurement zone."""

    raw_onsets = _raw_slide_onset_runs(records, zone)
    accepted_onsets = [
        run for run in raw_onsets if _passes_surrounding_safety(records, run)
    ]
    slip_runs = [
        run
        for run in _contiguous_runs(
            records,
            lambda row: valid_slide_onset_sample(row, zone=zone)
            and abs(float(row["slip_angle_degrees"]))
            >= NEAR_MISS_MIN_ABS_SLIP_DEGREES,
        )
        if run[1] - run[0] + 1 >= 2
    ]
    near_misses = [
        run
        for run in slip_runs
        if not any(_overlaps(run, onset) for onset in raw_onsets)
    ]
    candidate_runs = _legacy_runs(
        records,
        zone=zone,
        minimum_sliding_wheels=1,
        minimum_samples=2,
    )
    confirmed_runs = _legacy_runs(
        records,
        zone=zone,
        minimum_sliding_wheels=2,
        minimum_samples=3,
    )
    valid_rows = [row for row in records if valid_slide_onset_sample(row, zone=zone)]
    sliding_rows = [
        row for row in valid_rows if int(row.get("sliding_wheel_count", 0)) >= 1
    ]
    return {
        "zone": [zone[0], zone[1]],
        "valid_sample_count": len(valid_rows),
        "sliding_sample_count": len(sliding_rows),
        "one_sample_or_longer_slide_run_count": len(
            _contiguous_runs(
                records,
                lambda row: valid_slide_onset_sample(row, zone=zone)
                and int(row.get("sliding_wheel_count", 0)) >= 1,
            )
        ),
        "raw_slide_onset_count": len(raw_onsets),
        "accepted_slide_onset_count": len(accepted_onsets),
        "raw_slide_onsets": [
            _window_summary(records, run) for run in raw_onsets
        ],
        "accepted_slide_onsets": [
            _window_summary(records, run) for run in accepted_onsets
        ],
        "above_envelope_near_miss_count": len(near_misses),
        "above_envelope_near_misses": [
            _window_summary(records, run) for run in near_misses
        ],
        "legacy_candidate_count": len(candidate_runs),
        "legacy_confirmed_count": len(confirmed_runs),
        "legacy_candidates": [
            _window_summary(records, run) for run in candidate_runs
        ],
        "legacy_confirmed": [
            _window_summary(records, run) for run in confirmed_runs
        ],
        "maximum_abs_slip_angle_degrees": max(
            (abs(float(row["slip_angle_degrees"])) for row in valid_rows),
            default=None,
        ),
        "maximum_abs_body_up_yaw_rate": max(
            (abs(float(row["body_up_yaw_rate"])) for row in valid_rows),
            default=None,
        ),
        "maximum_sliding_wheels": max(
            (int(row.get("sliding_wheel_count", 0)) for row in valid_rows),
            default=0,
        ),
    }


def zone_traversal_metrics(records: Sequence[Record]) -> dict[str, Any]:
    """Measure sampled final-zone entry, exit, and local elapsed time."""

    entry_index = next(
        (
            index
            for index, row in enumerate(records)
            if float(row["progress"]) >= FINAL_CORNER_ZONE[0]
        ),
        None,
    )
    if entry_index is None:
        return {
            "entered": False,
            "exited": False,
            "entry_race_time_ms": None,
            "exit_race_time_ms": None,
            "traversal_time_ms": None,
            "entry_speed": None,
            "exit_speed": None,
        }
    exit_index = next(
        (
            index
            for index in range(entry_index, len(records))
            if float(records[index]["progress"]) >= FINAL_CORNER_ZONE[1]
        ),
        None,
    )
    entry = records[entry_index]
    if exit_index is None:
        return {
            "entered": True,
            "exited": False,
            "entry_race_time_ms": int(entry["race_time_ms"]),
            "exit_race_time_ms": None,
            "traversal_time_ms": None,
            "entry_speed": float(entry["display_speed"]),
            "exit_speed": None,
        }
    exit_record = records[exit_index]
    return {
        "entered": True,
        "exited": True,
        "entry_race_time_ms": int(entry["race_time_ms"]),
        "exit_race_time_ms": int(exit_record["race_time_ms"]),
        "traversal_time_ms": int(exit_record["race_time_ms"])
        - int(entry["race_time_ms"]),
        "entry_speed": float(entry["display_speed"]),
        "exit_speed": float(exit_record["display_speed"]),
    }


def score_episode(records: Sequence[Record]) -> dict[str, Any]:
    if not records:
        raise ValueError("episode scoring requires at least one live record")
    final = records[-1]
    safe_finish = bool(final.get("race_finished", False)) and not _failure_flag(final)
    return {
        "episode": int(final.get("episode", 0)),
        "seed": final.get("policy_seed"),
        "safe_finish": safe_finish,
        "terminal_race_time_ms": int(final["race_time_ms"]),
        "final_corner": score_zone(records, zone=FINAL_CORNER_ZONE),
        "first_turn_negative_control": score_zone(records, zone=FIRST_TURN_ZONE),
        "final_zone_traversal": zone_traversal_metrics(records),
    }


def paired_usefulness(
    control_score: Mapping[str, Any], treatment_score: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen local-usefulness test to one matched episode pair."""

    control = control_score["final_zone_traversal"]
    treatment = treatment_score["final_zone_traversal"]
    has_onset = (
        int(treatment_score["final_corner"]["accepted_slide_onset_count"]) > 0
    )
    comparable = bool(
        control["exited"]
        and treatment["exited"]
        and control["traversal_time_ms"] is not None
        and treatment["traversal_time_ms"] is not None
        and control["exit_speed"] is not None
        and treatment["exit_speed"] is not None
    )
    traversal_delta_ms = (
        int(treatment["traversal_time_ms"]) - int(control["traversal_time_ms"])
        if comparable
        else None
    )
    exit_speed_delta = (
        float(treatment["exit_speed"]) - float(control["exit_speed"])
        if comparable
        else None
    )
    useful = bool(
        has_onset
        and treatment_score["safe_finish"]
        and comparable
        and traversal_delta_ms is not None
        and traversal_delta_ms <= -SAMPLE_PERIOD_MS
        and exit_speed_delta is not None
        and exit_speed_delta >= 0.0
    )
    return {
        "has_accepted_slide_onset": has_onset,
        "comparable": comparable,
        "safe_finish": bool(treatment_score["safe_finish"]),
        "traversal_time_delta_ms_treatment_minus_control": traversal_delta_ms,
        "exit_speed_delta_treatment_minus_control": exit_speed_delta,
        "useful_candidate_before_visual_review": useful,
    }

