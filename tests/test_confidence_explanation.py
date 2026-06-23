import unittest

from engine.modules.ai_investigator import AIInvestigator
from engine.modules.investigation import Investigation, RootCause


class TestConfidenceExplanation(unittest.TestCase):
    def setUp(self):
        self.inv = Investigation()
        self.inv.confidence = 84
        # Add 6 drift causes, 5 high severity among them
        features = ['Income', 'DebtRatio', 'LoanAmount', 'ApprovalFlag', 'EmploymentYears', 'OtherFeature']
        for i, f in enumerate(features):
            sev = 'HIGH' if i < 5 else 'LOW'
            rc = RootCause(cause=f + ' feature drift', score=70 - i, severity=sev, evidence=[f'PSI=0.15'], category='feature_drift', source_modules=['drift_engine'])
            self.inv.root_causes.append(rc)
        # Add metadata for label noise and calibration
        self.inv.metadata['label_noise'] = {'findings': {'estimated_noise_fraction': 0.349}}
        self.inv.metadata['calibration'] = {'findings': {'ece': 0.1325}}
        # No leakage
        self.inv.metadata['leakage'] = {}

    def test_confidence_explanation_content(self):
        ai = AIInvestigator()
        explanation = ai._generate_confidence_explanation(self.inv)
        # Check expected lines
        self.assertIn('84%', explanation)
        self.assertIn('5 high-severity findings', explanation)
        self.assertIn('6 drifted features detected', explanation)
        self.assertIn('34.9% estimated label noise', explanation)
        self.assertIn('Calibration ECE of 13.25%', explanation)
        self.assertIn('No target leakage detected', explanation)
        self.assertIn('Evidence quality is moderate-to-strong', explanation)


if __name__ == '__main__':
    unittest.main()
