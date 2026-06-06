"""
Structural Causal Tests — Causal Thinking
=========================================
Tests for DAG building, node role detection, and Simpson's Paradox.
"""

import unittest
import sys
import os

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_P4_CAUSAL = os.path.join(_ROOT, "scratch", "phase4", "causal")
for p in [_ROOT, _P4_CAUSAL]:
    if p not in sys.path:
        sys.path.insert(0, p)

from dag_builder import DAG
from confounder_detector import classify_node_role, find_all_confounders
from simpsons_paradox import detect_simpsons_paradox, generate_paradox_example

class TestCausalThinking(unittest.TestCase):
    def setUp(self):
        self.dag = DAG()
        # Classical Confounder: Z -> X, Z -> Y
        self.dag.add_edge("Z", "X")
        self.dag.add_edge("Z", "Y")
        self.dag.add_edge("X", "Y")

    def test_dag_cycles(self):
        with self.assertRaises(ValueError):
            self.dag.add_edge("Y", "Z")

    def test_role_detection(self):
        role = classify_node_role(self.dag, "X", "Y", "Z")
        self.assertEqual(role, "CONFOUNDER")
        
        # Add a mediator: X -> M -> Y
        self.dag.add_edge("X", "M")
        self.dag.add_edge("M", "Y")
        role_m = classify_node_role(self.dag, "X", "Y", "M")
        self.assertEqual(role_m, "MEDIATOR")

    def test_confounder_identification(self):
        confounders = find_all_confounders(self.dag, "X", "Y")
        self.assertIn("Z", confounders)

    def test_simpsons_paradox_detection(self):
        X, y, Z = generate_paradox_example(seed=42)
        res = detect_simpsons_paradox(X, y, Z)
        self.assertTrue(res['detected'])
        self.assertGreater(res['aggregate_correlation'], 0)
        for c in res['stratum_correlations'].values():
            self.assertLess(c, 0)

if __name__ == "__main__":
    unittest.main()
