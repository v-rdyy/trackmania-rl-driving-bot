from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.build_pre_v3_backup import collect_evidence_files, sha256


class PreV3BackupTests(unittest.TestCase):
    def test_collect_evidence_files_is_sorted_unique_and_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            first = root / "b.txt"
            second = nested / "a.txt"
            first.write_text("first", encoding="utf-8")
            second.write_text("second", encoding="utf-8")

            files = collect_evidence_files((root, second))

            self.assertEqual(files, sorted({first, second}, key=lambda path: str(path).casefold()))

    def test_sha256_is_uppercase_and_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "evidence.bin"
            path.write_bytes(b"v2 evidence")

            digest = sha256(path)

            self.assertEqual(digest, digest.upper())
            self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
