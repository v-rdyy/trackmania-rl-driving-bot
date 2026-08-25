from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from trackmania_rl.track_assets import ensure_pinned_track_copy


class TrackAssetTests(unittest.TestCase):
    def test_installs_and_reuses_a_pinned_track(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "installed" / "A02-Race.Challenge.Gbx"
            destination = root / "user" / "A02-Race.Challenge.Gbx"
            source.parent.mkdir()
            source.write_bytes(b"pinned-track")
            expected = hashlib.sha256(b"pinned-track").hexdigest()

            self.assertTrue(
                ensure_pinned_track_copy(source, destination, expected)
            )
            self.assertEqual(destination.read_bytes(), b"pinned-track")
            self.assertFalse(
                ensure_pinned_track_copy(source, destination, expected)
            )

    def test_refuses_to_overwrite_a_different_user_track(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.gbx"
            destination = root / "destination.gbx"
            source.write_bytes(b"expected")
            destination.write_bytes(b"owner-file")
            expected = hashlib.sha256(b"expected").hexdigest()

            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                ensure_pinned_track_copy(source, destination, expected)


if __name__ == "__main__":
    unittest.main()
