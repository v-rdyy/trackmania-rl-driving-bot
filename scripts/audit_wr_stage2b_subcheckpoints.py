"""Inventory historical Stage 2b snapshots and probe surviving pre-zone actions.

No learning, game connection, policy interpolation, or reconstructed history.
Writes only a separate derived audit, preserving the previous diagnosis.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
import torch
from stable_baselines3 import PPO
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from diagnose_wr_stage2b_alignment import sha, stats
from trackmania_rl.observations import ReferencePath, build_observation

START = 3_506_176
OUT = ROOT / "artifacts/analysis/wr_chase_stage2b/subcheckpoint_audit.json"


def snapshot_timing(data):
    audit = data.get("kl_update_audit", [])
    if not isinstance(audit, list):
        raise ValueError("expected an ordinary JSON audit list")
    return {
        "model_timesteps": int(data["num_timesteps"]),
        "observed_additional_interactions": int(data["num_timesteps"]) - START,
        "completed_rollout_updates": len(audit),
        "learned_through_additional_interactions": (
            int(audit[-1]["model_timesteps"]) - START if audit else None),
    }


def main():
    torch.set_num_threads(1)
    manifest = json.loads((ROOT / "runs/wr_chase_stage2b/run_manifest.json").read_text())
    assert manifest["status"] == "safety_review_required"
    paths = sorted((ROOT / "checkpoints").rglob("*.zip"))
    inventory = []
    for path in paths:
        if "wr_chase_stage2b" not in str(path):
            continue
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
            row = {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                   "archive_members": archive.namelist(), **snapshot_timing(data)}
        row["accepted_run"] = path.parent.name == "wr_chase_stage2b"
        inventory.append(row)
    backups = []
    for path in sorted((ROOT / "artifacts/backups").glob("*.zip")):
        with zipfile.ZipFile(path) as archive:
            matches = [n for n in archive.namelist() if "stage2b" in n.lower()]
        backups.append({"path": str(path.relative_to(ROOT)), "stage2b_members": matches})
    tensorboard = []
    for path in sorted((ROOT / "tensorboard/wr_chase_stage2b_graduated_final_corner_0").glob("events.*")):
        events = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
        tensorboard.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                            "tags": events.Tags(),
                            "kl_rows": len(events.Scalars("train/kl_early_stop"))})
    result = {"checkpoint_inventory": inventory, "backups": backups,
              "tensorboard": tensorboard,
              "requested_10k_20k_30k_40k_snapshots_found": [],
              "trajectory_timeline_recoverable": False,
              "policy_probe_is_live_rollout": False, "training_performed": False}
    for row in inventory:
        if row["accepted_run"] and row["observed_additional_interactions"] in (10000, 20000, 30000, 40000):
            result["requested_10k_20k_30k_40k_snapshots_found"].append(row["path"])
    assert not result["requested_10k_20k_30k_40k_snapshots_found"]

    baseline = manifest["gates"][0]
    action_log = ROOT / baseline["evaluation_action_log"]
    assert sha(action_log) == baseline["evaluation_action_log_sha256"]
    summary = json.loads((ROOT / baseline["evaluation_summary"]).read_text())
    reference_file = ROOT / summary["reference_path"]
    assert sha(reference_file) == summary["reference_path_sha256"]
    reference = ReferencePath.from_csv(reference_file)
    rows = [json.loads(line) for line in action_log.read_text().splitlines()]
    states, actual_next_actions = [], []
    for a, b in zip(rows, rows[1:]):
        if (a["episode"] == b["episode"] and not a["terminated"] and not a["truncated"]
                and b["race_time_ms"] - a["race_time_ms"] == 100
                and not b["race_clock_boundary"] and a["progress"] < 1100):
            assert a["full_simstate_available"]
            states.append(a)
            actual_next_actions.append(b["raw_action"])
    observations = np.asarray([build_observation(SimpleNamespace(**r), reference)[0] for r in states])
    model_paths = {
        "base": ROOT / baseline["checkpoint"],
        "after_update24": ROOT / "checkpoints/wr_chase_stage2b/ppo_wr_stage2b_3556176_steps.zip",
        "after_update25": ROOT / manifest["gates"][1]["checkpoint"],
    }
    assert sha(model_paths["base"]) == baseline["checkpoint_sha256"]
    assert sha(model_paths["after_update25"]) == manifest["gates"][1]["checkpoint_sha256"]
    predictions = {}
    for label, path in model_paths.items():
        model = PPO.load(path, device="cpu")
        model.policy.set_training_mode(False)
        predictions[label] = model.predict(observations, deterministic=True)[0]
    error = float(np.abs(predictions["base"] - actual_next_actions).max())
    assert error < 2e-5
    result["baseline_action_reconstruction_max_abs_error"] = error
    result["baseline_action_log_sha256"] = sha(action_log)
    result["reference_path_sha256"] = sha(reference_file)
    result["prezone_fixed_state_action_probes"] = {}
    progress = np.asarray([r["progress"] for r in states])
    for lo, hi in ((0, 300), (300, 600), (600, 900), (900, 1100), (0, 1100)):
        mask = (progress >= lo) & (progress < hi)
        bands = {}
        for name, left, right in (("first24_vs_base", "base", "after_update24"),
                                  ("first25_vs_base", "base", "after_update25"),
                                  ("update25_only", "after_update24", "after_update25")):
            delta = predictions[right][mask] - predictions[left][mask]
            bands[name] = {channel: {"signed": stats(delta[:, i]), "absolute": stats(abs(delta[:, i]))}
                           for i, channel in enumerate(("steer", "throttle", "brake"))}
        result["prezone_fixed_state_action_probes"][f"{lo}_{hi}"] = bands
    result["limits"] = [
        "No pre-zone trajectories at intermediate 10k-40k policies exist in the inspected project evidence.",
        "The after-update24 probe is a same-state inference comparison, not a driven trajectory or finish-rate measurement.",
        "Rejected-run models are listed but not loaded or substituted into the accepted run.",
        "Scalar metrics and Adam moments do not recover missing historical policies.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Audit saved: {OUT}\nSHA-256: {sha(OUT)}")


if __name__ == "__main__":
    main()
