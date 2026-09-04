"""Request a clean save/stop from the one live continuous trainer; never kill it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trackmania_rl.continuous_training import atomic_json


def main() -> int:
    matches = []
    for status in (ROOT / "runs").glob("*/status.json"):
        data = json.loads(status.read_text(encoding="utf-8"))
        if data.get("status") not in {"preflight", "training"}:
            continue
        try:
            command = psutil.Process(int(data["pid"])).cmdline()
        except (psutil.Error, KeyError):
            continue
        if any(Path(arg).name == "train_wr_continuous.py" for arg in command):
            if status.parent.name in command:
                matches.append(status.parent)
    if len(matches) != 1:
        print(f"Found {len(matches)} matching live trainers; no stop request sent.")
        return 1
    atomic_json(matches[0] / "STOP.request", {"reason": "owner local stop control"})
    print("Stop requested. Wait for status.json to say 'stopped' and final_model.zip to exist.")
    print("Do not close the game or force-kill Python while it saves.")
    print(matches[0] / "status.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
