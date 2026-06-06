"""
Statistical Causal Tests — Causal Inference
===========================================
Tests for ATE/ATT estimation, DiD, and DoWhy integration.
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P4_CAUSAL = os.path.join(_ROOT, "scratch", "phase4", "causal")
for p in [_ROOT, _P4_CAUSAL]:
    if p not in sys.path:
        sys.path.insert(0, p)

from potential_outcomes import propensity_score, ipw_ate, matching_ate
from diff_in_diff import did_estimate
from dowhy_integration import DOWHY_AVAILABLE, full_causal_analysis

class TestCausalInference(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.N = 300
        # Z is a confounder
        self.z = np.random.normal(size=self.N)
        self.ps = 1 / (1 + np.exp(-self.z))
        self.t = (np.random.random(self.N) < self.ps).astype(int)
        # True ATE = 2.0
        self.y = 2.0 * self.t + 1.5 * self.z + np.random.normal(0, 0.1, self.N)
        self.df = pd.DataFrame({'T': self.t, 'Y': self.y, 'Z': self.z})

    def test_ipw_ate(self):
        ps_est = propensity_score(self.df[['Z']], self.df['T'])
        ate = ipw_ate(self.df['Y'], self.df['T'], ps_est)
        # True is 2.0. IPW should be close.
        self.assertAlmostEqual(ate, 2.0, delta=0.5)

    def test_matching_ate(self):
        ate = matching_ate(self.df[['Z']], self.df['Y'], self.df['T'])
        self.assertAlmostEqual(ate, 2.0, delta=0.5)

    def test_did_estimate(self):
        # Y_pre_t=10, Y_post_t=15 (diff=5)
        # Y_pre_c=10, Y_post_c=11 (diff=1)
        # DiD = 5 - 1 = 4
        ate = did_estimate([10], [15], [10], [11])
        self.assertEqual(ate, 4.0)

    @unittest.skipUnless(DOWHY_AVAILABLE, "DoWhy not installed")
    def test_dowhy_integration(self):
        res = full_causal_analysis(self.df, 'T', 'Y', common_causes=['Z'])
        self.assertAlmostEqual(res['value'], 2.0, delta=0.5)

if __name__ == "__main__":
    unittest.main()
