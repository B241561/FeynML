import unittest

from engine.modules.evidence_registry import EvidenceRegistry
from engine.modules.fact_validator import validate


class TestFactValidator(unittest.TestCase):
    def setUp(self):
        self.reg = EvidenceRegistry()

    def test_valid_record_passes(self):
        self.reg.register("Feature drift in X", "feature_drift", ["PSI=0.2"], "drift_engine", 70)
        res = validate(self.reg.get_claims())
        self.assertTrue(res['valid'])
        self.assertEqual(len(res['issues']), 0)

    def test_missing_evidence_detected(self):
        self.reg.register("Weird claim", "feature_drift", [], "drift_engine", 10)
        res = validate(self.reg.get_claims())
        self.assertFalse(res['valid'])
        self.assertTrue(any('missing evidence' in s.lower() for s in res['issues']))

    def test_missing_source_detected(self):
        self.reg.register("Another claim", "feature_drift", ["e1"], None, 20)
        res = validate(self.reg.get_claims())
        self.assertFalse(res['valid'])
        self.assertTrue(any('missing source_module' in s for s in res['issues']))

    def test_duplicate_detection(self):
        self.reg.register("Dup", "feature_drift", ["e"], "m", 10)
        self.reg.register("Dup", "feature_drift", ["e2"], "m2", 20)
        res = validate(self.reg.get_claims())
        self.assertFalse(res['valid'])
        self.assertTrue(any('duplicate claim detected' in s.lower() for s in res['issues']))


if __name__ == '__main__':
    unittest.main()
