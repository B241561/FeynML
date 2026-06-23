import unittest

from engine.modules.risk_scorer import RiskScorer


class TestRiskScoringBreakdown(unittest.TestCase):
    def test_contributions_and_details(self):
        scorer = RiskScorer()

        # Prepare component reports with explicit severities
        leakage_report = {'severity': 'NONE'}
        label_noise_report = {'severity': 'HIGH'}
        drift_report = {'severity': 'CRITICAL'}
        calibration_report = {'severity': 'MEDIUM'}
        missing_data_report = {'severity': 'NONE'}
        fairness_report = {'severity': 'NONE'}

        # Root causes list to set root_cause severity to CRITICAL
        root_causes = [{'severity': 'CRITICAL'}]

        res = scorer.score(
            leakage_report=leakage_report,
            label_noise_report=label_noise_report,
            drift_report=drift_report,
            calibration_report=calibration_report,
            missing_data_report=missing_data_report,
            fairness_report=fairness_report,
            root_causes=root_causes,
        )

        # Basic fields
        self.assertIn('component_details', res)
        comp_details = res['component_details']

        # 1) Contributions sum to total score (allow small float tolerance)
        total_contrib = sum(d['contribution'] for d in comp_details.values())
        self.assertAlmostEqual(total_contrib, res['score'], places=3)

        # 2) No negative contributions
        for d in comp_details.values():
            self.assertGreaterEqual(d['contribution'], 0)

        # 3) Contribution never exceeds max_contribution
        for d in comp_details.values():
            self.assertLessEqual(d['contribution'], d['max_contribution'] + 1e-6)

        # 4) score_pct remains correctly computed
        # Recompute pct to compare
        max_sev = max(scorer.severity_map.values())
        max_possible = sum(max_sev * scorer.component_weights.get(comp, 1.0) for comp in comp_details.keys())
        recomputed_pct = round((res['score'] / max_possible * 100.0) if max_possible > 0 else 0.0, 2)
        self.assertAlmostEqual(recomputed_pct, res['score_pct'], places=2)


if __name__ == '__main__':
    unittest.main()
