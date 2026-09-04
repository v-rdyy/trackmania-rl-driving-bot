"""Read-only Stage 2b alignment diagnosis from frozen live logs and policies.

No game connection, learning, replay telemetry, or checkpoint writes. Outputs
are derived evidence only. Progress profiles use first forward crossings, not
post-crash dwell samples. Policy probes use identical reconstructed live states.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from trackmania_rl.observations import ReferencePath, build_observation
from trackmania_rl.rewards import graduated_final_corner_precursor_bonus
from analyze_wr_stage2b_gate import transition_from_record

OUT = ROOT / "artifacts/analysis/wr_chase_stage2b/alignment_diagnosis"
RUN = ROOT / "runs/wr_chase_stage2b"
ZONES = {"bonus_zone": (1100, 1410), "post_zone": (1410, 1800),
         "approach": (1800, 2000), "final_alignment": (2000, 2150)}
FIELDS = ("race_time_ms", "lateral_offset", "heading_error", "display_speed",
          "input_steer", "input_throttle", "input_brake", "vertical_offset")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def stats(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    return {"n": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "min": float(values.min()),
            "max": float(values.max()), "p95": float(np.quantile(values, .95))}


def crossing(records, progress):
    """Interpolate the first forward crossing; never extrapolate or cross resets."""
    for a, b in zip(records, records[1:]):
        if (b["race_time_ms"] - a["race_time_ms"] != 100
                or b.get("race_clock_boundary", False)):
            continue
        if a["progress"] <= progress < b["progress"]:
            f = (progress - a["progress"]) / (b["progress"] - a["progress"])
            result = {k: a[k] + f * (b[k] - a[k]) for k in FIELDS}
            result["position"] = (np.asarray(a["position"]) + f * (
                np.asarray(b["position"]) - a["position"])).tolist()
            return result
    return None


def distribution_snapshot(model, observations):
    with torch.no_grad():
        obs, _ = model.policy.obs_to_tensor(observations)
        dist = model.policy.get_distribution(obs).distribution
        return dist.loc.detach().cpu().numpy().copy(), dist.scale.detach().cpu().numpy().copy()


def normal_kl(a, b):
    """Per-state KL(a || b) for the pre-tanh diagonal Gaussian."""
    return normal_kl_components(a, b).sum(axis=1)


def normal_kl_components(a, b):
    mu_a, sd_a = (np.asarray(x, dtype=np.float64) for x in a)
    mu_b, sd_b = (np.asarray(x, dtype=np.float64) for x in b)
    return np.log(sd_b / sd_a) + (sd_a**2 + (mu_a - mu_b)**2) / (2 * sd_b**2) - .5


def optimizer_steps(model):
    return {name: int(model.policy.optimizer.state[param]["step"].item())
            for name, param in model.policy.named_parameters()
            if "step" in model.policy.optimizer.state.get(param, {})}


def final_airborne_run(records):
    """Longest observed zero-contact run in the pre-structure 1500..2150 span."""
    runs, current = [], []
    for i, r in enumerate(records):
        if 1500 <= r["progress"] < 2150 and r["ground_contact_count"] == 0:
            if current and r["race_time_ms"] - records[current[-1]]["race_time_ms"] != 100:
                runs.append(current)
                current = []
            current.append(i)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    if not runs:
        return None
    indices = max(runs, key=len)
    return indices[0], indices[-1]


def analyze():
    torch.set_num_threads(1)
    manifest = json.loads((RUN / "run_manifest.json").read_text())
    assert manifest["status"] == "safety_review_required"
    evidence = {}
    datasets = {}
    summaries = {}
    for label, gate in zip(("base", "gate50"), manifest["gates"], strict=True):
        for field in ("checkpoint", "evaluation_summary", "evaluation_action_log"):
            path = ROOT / gate[field]
            assert sha(path) == gate[field + "_sha256"]
            evidence[str(path.relative_to(ROOT))] = sha(path)
        summary = json.loads((ROOT / gate["evaluation_summary"]).read_text())
        assert summary["measurement_source"] == "direct_live_evaluation_simstate"
        assert not summary["replay_telemetry_fallback_used"]
        assert summary["deterministic"] and summary["simulation_speed"] == 6.0
        summaries[label] = summary
        rows = [json.loads(line) for line in (
            ROOT / gate["evaluation_action_log"]).read_text().splitlines()]
        assert all(r["full_simstate_available"] for r in rows)
        datasets[label] = {i: [r for r in rows if r["episode"] == i] for i in range(10)}
    path_file = ROOT / summaries["base"]["reference_path"]
    assert sha(path_file) == summaries["base"]["reference_path_sha256"]
    assert summaries["gate50"]["reference_path_sha256"] == sha(path_file)
    reference = ReferencePath.from_csv(path_file)
    evidence[str(path_file.relative_to(ROOT))] = sha(path_file)
    model_paths = {"base": ROOT / manifest["gates"][0]["checkpoint"],
                   "pre_final_update": ROOT / "checkpoints/wr_chase_stage2b/ppo_wr_stage2b_3556176_steps.zip",
                   "gate50": ROOT / manifest["gates"][1]["checkpoint"]}
    models = {k: PPO.load(p, device="cpu") for k, p in model_paths.items()}
    for key, path in model_paths.items():
        evidence[str(path.relative_to(ROOT))] = sha(path)
        models[key].policy.set_training_mode(False)
    output = {"source": "saved_direct_live_SimState_no_replay_fallback",
              "training_performed": False, "input_sha256": evidence,
              "caveat": "Retrospective descriptive diagnosis; unpaired deterministic episodes. No bonus-free learning control or per-update policy snapshots 1-23.",
              "checkpoints": {}, "crossings": {}, "episodes": {}, "policy_probes": {}}
    for key, model in models.items():
        output["checkpoints"][key] = {"timesteps": model.num_timesteps,
            "optimizer_steps": optimizer_steps(model),
            "audit_updates": len(getattr(model, "kl_update_audit", [])),
            "gamma": model.gamma, "gae_lambda": model.gae_lambda,
            "target_kl": model.target_kl, "batch_size": model.batch_size,
            "n_steps": model.n_steps, "learning_rate": model.policy.optimizer.param_groups[0]["lr"]}

    anchors = [300, 600, 900, 1000, 1100, 1200, 1300, 1410, 1500, 1600, 1700, 1800, 1900, 2000, 2050, 2100, 2125, 2150]
    for label, episodes in datasets.items():
        output["crossings"][label] = {}
        for p in anchors:
            samples = [crossing(rows, p) for rows in episodes.values()]
            samples = [r for r in samples if r is not None]
            output["crossings"][label][p] = {k: stats([r[k] for r in samples]) for k in FIELDS}
        output["episodes"][label] = []
        for episode, rows in episodes.items():
            decels = [r for r in rows if r["previous_progress"] > 1800
                      and r["previous_display_speed"] - r["display_speed"] >= 25]
            bonus_rows = [r for r in rows if graduated_final_corner_precursor_bonus(transition_from_record(r)) > 0]
            detail = {"episode": episode, "finished": any(r["race_finished"] for r in rows),
                "first_abrupt_deceleration": ({k: decels[0][k] for k in (
                    "race_time_ms", "progress", "position", "display_speed", "previous_display_speed",
                    "input_steer", "lateral_offset", "heading_error", "ground_contact_count")} if decels else None),
                "bonus_total": sum(graduated_final_corner_precursor_bonus(transition_from_record(r)) for r in rows),
                "last_bonus_time_ms": bonus_rows[-1]["race_time_ms"] if bonus_rows else None,
                "last_bonus_position": bonus_rows[-1]["position"] if bonus_rows else None,
                "crossings": {p: crossing(rows, p) for p in anchors}}
            if decels and bonus_rows:
                detail["last_bonus_to_deceleration_ms"] = decels[0]["race_time_ms"] - bonus_rows[-1]["race_time_ms"]
                detail["last_bonus_to_deceleration_world_distance"] = float(np.linalg.norm(
                    np.asarray(decels[0]["position"]) - bonus_rows[-1]["position"]))
            airborne = final_airborne_run(rows)
            if airborne:
                start, end = airborne
                flight = rows[start:end + 1]
                fields = ("race_time_ms", "progress", "position", "velocity",
                          "lateral_offset", "heading_error", "display_speed", "input_steer")
                detail["final_flight"] = {
                    "first_zero_contact": {k: flight[0][k] for k in fields},
                    "last_zero_contact": {k: flight[-1][k] for k in fields},
                    "first_contact_after": {k: rows[end + 1][k] for k in fields} if end + 1 < len(rows) else None,
                    "zero_contact_samples": len(flight),
                    "sample_span_ms": flight[-1]["race_time_ms"] - flight[0]["race_time_ms"],
                    "velocity_heading_degrees": stats([float(np.degrees(np.arctan2(r["velocity"][2], r["velocity"][0]))) for r in flight]),
                    "start_reference_lateral_velocity": float(np.dot(
                        [reference.project(np.asarray(flight[0]["position"])).tangent_xz[1],
                         -reference.project(np.asarray(flight[0]["position"])).tangent_xz[0]],
                        np.asarray(flight[0]["velocity"])[[0, 2]]))}
            output["episodes"][label].append(detail)

        # Each row describes the POST-action state. Verify reconstructed row N
        # against action N+1, excluding reset, terminal, and non-100ms boundaries.
        probe_rows, next_actions = [], []
        for rows in episodes.values():
            for a, b in zip(rows, rows[1:]):
                if (not a["terminated"] and not a["truncated"]
                        and b["race_time_ms"] - a["race_time_ms"] == 100
                        and not b.get("race_clock_boundary", False)):
                    probe_rows.append(a)
                    next_actions.append(b["raw_action"])
        observations = np.asarray([build_observation(SimpleNamespace(**r), reference)[0] for r in probe_rows])
        actions = {k: model.predict(observations, deterministic=True)[0] for k, model in models.items()}
        replay_error = np.abs(actions[label] - next_actions)
        assert float(replay_error.max()) < 2e-5, float(replay_error.max())
        distributions = {k: distribution_snapshot(model, observations) for k, model in models.items()}
        probes = {"action_reconstruction_max_abs_error": float(replay_error.max()), "zones": {}}
        progress = np.asarray([r["progress"] for r in probe_rows])
        for zone, (lo, hi) in ZONES.items():
            mask = (progress >= lo) & (progress < hi)
            delta = actions["gate50"][mask] - actions["base"][mask]
            first24 = actions["pre_final_update"][mask, 0] - actions["base"][mask, 0]
            last = actions["gate50"][mask, 0] - actions["pre_final_update"][mask, 0]
            probes["zones"][zone] = {
                "samples": int(mask.sum()), "steer_delta": stats(delta[:, 0]),
                "abs_steer_delta": stats(abs(delta[:, 0])),
                "throttle_delta": stats(delta[:, 1]), "brake_delta": stats(delta[:, 2]),
                "first24_steer_delta": stats(first24), "last_update_steer_delta": stats(last),
                "last_update_same_direction_fraction": float(np.mean(first24 * last > 0)),
                "base_to_final_kl": stats(normal_kl(distributions["base"], distributions["gate50"])[mask]),
                "base_to_final_kl_by_action": {name: stats(normal_kl_components(
                    distributions["base"], distributions["gate50"])[mask, i])
                    for i, name in enumerate(("steer", "throttle", "brake"))},
                "pre_final_to_final_kl": stats(normal_kl(distributions["pre_final_update"], distributions["gate50"])[mask])}
        # Retain action probes at each original live state for the diagnostic plot.
        probes["sample_rows"] = [{"episode": r["episode"], "progress": r["progress"],
            "race_time_ms": r["race_time_ms"], "steer_base": float(actions["base"][i, 0]),
            "steer_pre_final": float(actions["pre_final_update"][i, 0]),
            "steer_gate50": float(actions["gate50"][i, 0])} for i, r in enumerate(probe_rows)]
        output["policy_probes"][label] = probes

    training_file = RUN / "gates/gate_00050000_training.json"
    training = json.loads(training_file.read_text())
    assert sha(training_file) == manifest["gates"][1]["training_summary_sha256"]
    evidence[str(training_file.relative_to(ROOT))] = sha(training_file)
    output["update_audit"] = training["ppo_update_audit_this_gate"]
    monitor = RUN / "training.monitor.csv"
    evidence[str(monitor.relative_to(ROOT))] = sha(monitor)
    with monitor.open() as stream:
        next(stream)
        monitor_rows = list(csv.DictReader(stream))
    total_steps = 0
    training_rows = []
    for i, row in enumerate(monitor_rows):
        total_steps += int(row["l"])
        training_rows.append({"episode": i, "completed_at_step": total_steps,
            "collecting_rollout": (total_steps - 1) // 2048 + 1,
            "finished": row["race_finished"] == "True", "reward": float(row["r"]),
            "race_time_ms": int(row["race_time_ms"]), "progress": float(row["progress"])})
    output["stochastic_training_episodes"] = training_rows
    output["stochastic_training_by_rollout"] = [{"rollout": n,
        "episodes": sum(r["collecting_rollout"] == n for r in training_rows),
        "finishes": sum(r["collecting_rollout"] == n and r["finished"] for r in training_rows)} for n in range(1, 26)]
    output["training_summary"] = {"completed_episodes": len(training_rows),
        "finishes": sum(r["finished"] for r in training_rows),
        "completed_episode_steps": total_steps,
        "unfinished_episode_steps": training["actual_gate_interactions"] - total_steps}
    output["representative_replays"] = {}
    for label, summary in summaries.items():
        details = summary["episodes_detail"]
        finishes = [r for r in details if r["finished"]]
        if finishes:
            mean_time = np.mean([r["terminal_race_time_ms"] for r in finishes])
            chosen = {"best_finish": min(finishes, key=lambda r: r["terminal_race_time_ms"]),
                      "closest_to_mean_finish": min(finishes, key=lambda r: abs(r["terminal_race_time_ms"] - mean_time)),
                      "worst_finish": max(finishes, key=lambda r: r["terminal_race_time_ms"])}
        else:
            ordered = sorted(details, key=lambda r: max(x["progress"] for x in datasets[label][r["episode"]]))
            chosen = {"highest_progress_failure": ordered[-1],
                      "median_progress_failure": ordered[(len(ordered)-1)//2],
                      "lowest_progress_failure": ordered[0]}
        output["representative_replays"][label] = {}
        for category, row in chosen.items():
            path = ROOT / row["input_replay"]
            assert sha(path) == row["input_replay_sha256"]
            output["representative_replays"][label][category] = {
                "path": str(path.relative_to(ROOT)), "sha256": sha(path),
                "episode": row["episode"], "race_time_ms": row["terminal_race_time_ms"],
                "finished": row["finished"]}
    output["methods"] = {
        "crossings": "Linear interpolation of first forward crossing; steering is the action ending at each logged state (not a policy probe).",
        "policy_probe": "Same 26-value observation reconstructed from live state; next-row action validates timing. Separate base-state and Gate50-state pools.",
        "kl": "Analytic pre-tanh diagonal Gaussian KL summed over 3 actions; same invertible squashing/scaling, not the minibatch sample estimator. Fixed evaluation states, not training occupancy.",
        "monitor": "Stochastic episode outcomes grouped by rollout containing completion; some episodes span updates. Not deterministic checkpoint evaluations.",
        "collision_proxy": "First >=25 displayed-speed loss per 100ms after progress1800. Abrupt deceleration is not a collision/contact sensor.",
        "lateral": "Signed displacement from driven reference path, not surveyed road center or measured structure clearance."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "diagnosis.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    plot(datasets, output)
    print(f"Saved {OUT / 'diagnosis.json'}")
    return output


def plot(datasets, result):
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    grid = np.arange(1050, 2151, 5)
    for label, color in (("base", "#2468b4"), ("gate50", "#d25335")):
        profiles = [[crossing(rows, p) for p in grid] for rows in datasets[label].values()]
        for ax, field in zip(axes[:3], ("lateral_offset", "display_speed", "input_steer")):
            values = np.asarray([[r[field] if r else np.nan for r in profile] for profile in profiles])
            ax.fill_between(grid, np.nanmin(values, axis=0), np.nanmax(values, axis=0), color=color, alpha=.15)
            ax.plot(grid, np.nanmedian(values, axis=0), color=color, label=label)
    points = result["policy_probes"]["base"]["sample_rows"]
    points = [r for r in points if r["episode"] == 0 and 1050 <= r["progress"] <= 2150]
    axes[3].plot([r["progress"] for r in points], [r["steer_pre_final"] - r["steer_base"] for r in points], label="After 24 updates - base", color="#8b63ad")
    axes[3].plot([r["progress"] for r in points], [r["steer_gate50"] - r["steer_base"] for r in points], label="After 25 updates - base", color="#d25335")
    for ax, label in zip(axes, ("Signed reference offset", "Displayed speed", "Applied steering", "Same-state steering change\n(base episode 1)")):
        ax.axvspan(1100, 1410, alpha=.12, color="green")
        ax.axvspan(2000, 2150, alpha=.09, color="red")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.legend(loc="upper left")
    axes[3].set_xlabel("Reference progress (green: bonus zone; red: late alignment)")
    fig.suptitle("Stage 2b diagnosis — saved live telemetry, no retraining\nTop panels: median + full range across 10 episodes; first crossings only")
    fig.tight_layout()
    fig.savefig(OUT / "alignment_comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    analyze()
