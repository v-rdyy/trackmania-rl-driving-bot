from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import imageio_ffmpeg
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "src"))

from trackmania_rl.video_capture import probe_video, scaled_even_size, sha256

SCRIPT_PATH = WORKSPACE_ROOT / "scripts" / "capture_progress_videos.py"
SPEC = importlib.util.spec_from_file_location("capture_progress_videos", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class VideoCaptureTests(unittest.TestCase):
    def test_scaled_size_preserves_aspect_and_encoder_safe_dimensions(self) -> None:
        self.assertEqual(scaled_even_size(2560, 1400, 1280), (1280, 700))
        width, height = scaled_even_size(1919, 1079, 1280)
        self.assertEqual(width % 2, 0)
        self.assertEqual(height % 2, 0)
        self.assertAlmostEqual(width / height, 1919 / 1079, places=2)

    def test_sha256_is_uppercase_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"trackmania")
            self.assertEqual(
                sha256(path),
                "4AF122F69E3F9D12357D97A7EBC3E6861E24E0A169D484D33DA16DBA75047319",
            )

    def test_h264_output_can_be_decoded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.mp4"
            writer = imageio_ffmpeg.write_frames(
                str(path),
                (64, 48),
                fps=10,
                codec="libx264",
                macro_block_size=2,
            )
            writer.send(None)
            for value in range(5):
                writer.send(np.full((48, 64, 3), value * 40, dtype=np.uint8))
            writer.close()
            metadata = probe_video(path)
            self.assertEqual(metadata["size"], [64, 48])
            self.assertAlmostEqual(float(metadata["fps"]), 10.0)

    def test_plan_has_unique_valid_progression_stages(self) -> None:
        plan = MODULE.load_plan(WORKSPACE_ROOT / "config" / "video_progression.json")
        stages = MODULE.selected_stages(
            plan,
            ["v0_untrained_reference", "v2_speed_final"],
        )
        self.assertEqual(
            [stage["id"] for stage in stages],
            ["v0_untrained_reference", "v2_speed_final"],
        )
        self.assertIsNone(stages[0]["checkpoint"])
        self.assertEqual(stages[1]["episodes"], 5)

    def test_take_directories_are_append_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = MODULE.next_take_directory(root, "v2_speed_final")
            second = MODULE.next_take_directory(root, "v2_speed_final")
            self.assertEqual(first.name, "take_001")
            self.assertEqual(second.name, "take_002")

    def test_replay_catalog_aggregates_successes_and_failures(self) -> None:
        replay_script = WORKSPACE_ROOT / "scripts" / "capture_progress_replays.py"
        spec = importlib.util.spec_from_file_location(
            "capture_progress_replays",
            replay_script,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            success_dir = root / "stage" / "take_001"
            failure_dir = root / "stage" / "take_002"
            success_dir.mkdir(parents=True)
            failure_dir.mkdir(parents=True)
            (success_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "stage": {"id": "stage", "model_timesteps": 50},
                        "checkpoint_sha256": "ABC",
                        "episodes": [
                            {
                                "finished": True,
                                "fallen": False,
                                "off_track": False,
                                "timeout": False,
                                "input_replay": "replay.txt",
                                "input_replay_bytes": 123,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (failure_dir / "failure_manifest.json").write_text(
                json.dumps(
                    {
                        "stage": {"id": "stage"},
                        "error": "countdown failed",
                    }
                ),
                encoding="utf-8",
            )

            catalog = module.build_catalog(root, WORKSPACE_ROOT / "plan.json")

            self.assertEqual(catalog["successful_take_count"], 1)
            self.assertEqual(catalog["failed_take_count"], 1)
            self.assertEqual(catalog["input_replay_count"], 1)
            self.assertEqual(catalog["input_replay_bytes"], 123)
            self.assertEqual(catalog["successful_takes"][0]["outcomes"], ["finish"])


if __name__ == "__main__":
    unittest.main()
