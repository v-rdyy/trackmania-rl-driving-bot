from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_wr_stage2b_subcheckpoints import START, snapshot_timing


class SubcheckpointTimingTests(unittest.TestCase):
    def test_mid_rollout_checkpoint_does_not_claim_unperformed_updates(self):
        timing = snapshot_timing({"num_timesteps": START + 50000,
            "kl_update_audit": [{"model_timesteps": START + i * 2048} for i in range(1, 25)]})
        self.assertEqual(timing["observed_additional_interactions"], 50000)
        self.assertEqual(timing["learned_through_additional_interactions"], 49152)
        self.assertEqual(timing["completed_rollout_updates"], 24)

    def test_final_checkpoint_contains_last_complete_update(self):
        timing = snapshot_timing({"num_timesteps": START + 51200,
            "kl_update_audit": [{"model_timesteps": START + i * 2048} for i in range(1, 26)]})
        self.assertEqual(timing["learned_through_additional_interactions"], 51200)

    def test_absent_audit_does_not_invent_training_timing(self):
        timing = snapshot_timing({"num_timesteps": START + 50000})
        self.assertIsNone(timing["learned_through_additional_interactions"])


if __name__ == "__main__":
    unittest.main()
