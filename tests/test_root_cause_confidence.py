import unittest

from engine.modules.root_cause_engine import AutoRootCauseEngine


class TestRootCauseConfidence(unittest.TestCase):

    def setUp(self):
        self.engine = AutoRootCauseEngine(verbose=False)

    def test_no_causes(self):
        health, conf = self.engine._calculate_health_status([])
        self.assertEqual(health, "Healthy")
        self.assertEqual(conf, 50)
        self.assertLessEqual(conf, 95)

    def test_one_high_issue(self):
        causes = [{"cause": "X", "score": 60, "severity": "HIGH", "evidence": [], "category": "drift"}]
        health, conf = self.engine._calculate_health_status(causes)
        self.assertIn(health, ("Warning", "Critical", "Healthy"))
        # Expect modest confidence boost: 50 + min(20,1*4)=54, + criticals 0 => 54, + avg_score_all/10 ~6 => ~60
        self.assertGreaterEqual(conf, 55)
        self.assertLessEqual(conf, 65)
        self.assertLessEqual(conf, 95)

    def test_three_critical_issues(self):
        causes = [
            {"cause": f"C{i}", "score": 90, "severity": "CRITICAL", "evidence": [], "category": "leakage"}
            for i in range(3)
        ]
        health, conf = self.engine._calculate_health_status(causes)
        # high confidence expected but capped
        self.assertGreaterEqual(conf, 75)
        self.assertLessEqual(conf, 90)
        self.assertLessEqual(conf, 95)

    def test_many_critical_issues_capped(self):
        causes = [
            {"cause": f"C{i}", "score": 95, "severity": "CRITICAL", "evidence": [], "category": "leakage"}
            for i in range(10)
        ]
        health, conf = self.engine._calculate_health_status(causes)
        self.assertLessEqual(conf, 95)


if __name__ == "__main__":
    unittest.main()
