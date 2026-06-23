import unittest

from engine.modules.root_cause_engine import AutoRootCauseEngine


class TestDriftCountConsistency(unittest.TestCase):

    def setUp(self):
        self.engine = AutoRootCauseEngine(verbose=False)

    def test_canonical_drift_count_used_everywhere(self):
        # Build per_feature list: 3 DRIFT, 1 WARN
        per_feature = [
            {"feature": "f1", "status": "DRIFT", "psi": 0.3, "ks_stat": 0.25},
            {"feature": "f2", "status": "DRIFT", "psi": 0.28, "ks_stat": 0.22},
            {"feature": "f3", "status": "DRIFT", "psi": 0.26, "ks_stat": 0.21},
            {"feature": "f4", "status": "WARN",  "psi": 0.12, "ks_stat": 0.11},
        ]
        drift_report = {"findings": {"per_feature": per_feature}}

        result = self.engine.run(drift_report=drift_report)

        # Evidence summary should report 3 features show drift
        evidence_summary = result.get("evidence", [""])[0]
        self.assertIn("3 features show drift", evidence_summary)

        # Risk explanation should reference 3 drifted feature(s)
        risk_expl = result.get("risk_explanation", "")
        self.assertIn("3 drifted feature(s) detected", risk_expl)

        # Root causes should include exactly 3 feature_drift entries
        root_causes = result.get("root_causes", [])
        drift_causes = [rc for rc in root_causes if rc.get("category") == "feature_drift"]
        self.assertEqual(len(drift_causes), 3)

        # Canonical count stored in metadata
        metadata = result.get("metadata", {})
        self.assertEqual(metadata.get("evidence_summary", ""), evidence_summary)
        # canonical_drift_count should be present and equal to 3
        self.assertIn("canonical_drift_count", metadata)
        self.assertEqual(metadata.get("canonical_drift_count"), 3)


if __name__ == "__main__":
    unittest.main()
