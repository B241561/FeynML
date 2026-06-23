import unittest

from engine.modules.root_cause_engine import AutoRootCauseEngine


class TestRootCauseDeduplication(unittest.TestCase):
    def setUp(self):
        self.engine = AutoRootCauseEngine(verbose=False)

    def test_consolidate_drift_recommendation(self):
        # Create five drift causes
        features = ['Income', 'DebtRatio', 'LoanAmount', 'ApprovalFlag', 'EmploymentYears']
        scored_causes = []
        for i, f in enumerate(features):
            scored_causes.append({
                'cause': f + ' feature drift',
                'score': 75 - i,  # descending scores
                'severity': 'HIGH',
                'evidence': [f'PSI=0.1'],
                'category': 'feature_drift'
            })

        recs = self.engine._generate_recommendations(scored_causes, leakage_evidence=[])

        # Expect a single consolidated drift recommendation
        # (no duplicates per-feature)
        self.assertIsInstance(recs, list)
        self.assertGreaterEqual(len(recs), 1)

        # The first recommendation should summarize drift
        first = recs[0]
        self.assertIn('features show significant drift', first)
        # Should mention count 5
        self.assertIn('5 features', first)
        # Should list top 3 features
        self.assertIn('Income', first)
        self.assertIn('DebtRatio', first)
        self.assertIn('LoanAmount', first)

    def test_generate_investigation_summary_dedupes(self):
        # Build a fake result containing the five drift root_causes
        features = ['Income', 'DebtRatio', 'LoanAmount', 'ApprovalFlag', 'EmploymentYears']
        root_causes = []
        for i, f in enumerate(features):
            root_causes.append({
                'cause': f + ' feature drift',
                'score': 75 - i,
                'severity': 'HIGH',
                'category': 'feature_drift'
            })

        result = {
            'health_status': 'Warning',
            'confidence': 64,
            'evidence_summary': '5 features show drift.',
            'root_causes': root_causes,
            'recommended_actions': ['dummy']
        }

        summary = self.engine.generate_investigation_summary(result)
        # Should contain consolidated primary concern
        self.assertIn('Primary concern: Feature drift affecting 5 features', summary)
        self.assertIn('Most impacted', summary)
        # Not contain 5 separate cause lines
        self.assertNotIn('Income feature drift', summary)


if __name__ == '__main__':
    unittest.main()
