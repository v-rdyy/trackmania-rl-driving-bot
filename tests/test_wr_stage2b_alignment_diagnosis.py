from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_wr_stage2b_alignment import FIELDS, crossing, final_airborne_run, normal_kl


class AlignmentDiagnosisTests(unittest.TestCase):
    def row(self, t, p, value, contacts=4):
        row = {k: value for k in FIELDS}
        row.update(race_time_ms=t, progress=p, position=[value]*3, ground_contact_count=contacts)
        return row

    def test_first_crossing_not_later_post_crash_pass(self):
        rows = [self.row(0, 0, 0), self.row(100, 10, 10),
                self.row(200, 0, 100), self.row(300, 10, 200)]
        self.assertEqual(crossing(rows, 5)["lateral_offset"], 5)
        self.assertIsNone(crossing(rows, 11))

    def test_crossing_rejects_reset_or_skipped_samples(self):
        for t in (0, 200):
            self.assertIsNone(crossing([self.row(0, 0, 0), self.row(t, 10, 10)], 5))
        rows = [self.row(0, 0, 0), self.row(100, 10, 10)]
        rows[1]["race_clock_boundary"] = True
        self.assertIsNone(crossing(rows, 5))

    def test_normal_kl_matches_known_values(self):
        sd = np.ones((1, 3))
        a = (np.zeros((1, 3)), sd)
        self.assertEqual(normal_kl(a, a)[0], 0)
        self.assertAlmostEqual(normal_kl(a, (np.ones((1, 3)), sd))[0], 1.5)

    def test_longest_flight_excludes_post_impact_and_small_hop(self):
        rows = [self.row(19000, 1600, 0, 0), self.row(19100, 1610, 0),
                self.row(19200, 1700, 0, 0), self.row(19300, 1710, 0, 0),
                self.row(19400, 1720, 0), self.row(24400, 2160, 0, 0)]
        self.assertEqual(final_airborne_run(rows), (2, 3))


if __name__ == "__main__":
    unittest.main()
