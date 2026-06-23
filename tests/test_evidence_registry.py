import unittest

from engine.modules.evidence_registry import EvidenceRegistry


class TestEvidenceRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = EvidenceRegistry()

    def test_register_and_get(self):
        rec = self.reg.register(
            claim="Feature drift detected in age",
            category="feature_drift",
            evidence=["PSI=0.32"],
            source_module="drift_engine",
            confidence=82,
        )
        self.assertIsInstance(rec, dict)
        self.assertEqual(rec['claim'], "Feature drift detected in age")
        self.assertEqual(len(self.reg.get_claims()), 1)

    def test_find_duplicates(self):
        self.reg.register("X issue", "feature_drift", ["e1"], "modA", 50)
        self.reg.register("X issue", "feature_drift", ["e2"], "modB", 60)
        dupes = self.reg.find_duplicates()
        self.assertIn("X issue", dupes)

    def test_clear_and_len(self):
        self.reg.register("A", "feature_drift", ["e"], "mod", 10)
        self.assertEqual(len(self.reg), 1)
        self.reg.clear()
        self.assertEqual(len(self.reg), 0)


if __name__ == '__main__':
    unittest.main()
