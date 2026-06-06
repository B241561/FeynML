"""
Tests for Causal Thinking Module
"""

import pytest
import numpy as np
import pandas as pd

from engine.causal_thinking import CausalGraphBuilder


class TestCausalGraphBuilder:

    def test_build_dag_basic(self):
        """Test basic DAG construction."""
        builder = CausalGraphBuilder(verbose=False)
        
        nodes = ['X', 'Y', 'Z', 'T', 'O']
        edges = [('X', 'T'), ('X', 'O'), ('T', 'O'), ('Y', 'O')]
        
        result = builder.build_dag(nodes, edges)
        
        assert result['is_acyclic'] is True
        assert result['n_nodes'] == 5
        assert result['n_edges'] == 4
        assert result['status'] == 'SUCCESS'

    def test_build_dag_with_cycle(self):
        """Test DAG construction rejects cycles."""
        builder = CausalGraphBuilder(verbose=False)
        
        nodes = ['X', 'Y', 'Z']
        edges = [('X', 'Y'), ('Y', 'Z'), ('Z', 'X')]  # Cycle!
        
        with pytest.raises(ValueError):
            builder.build_dag(nodes, edges)

    def test_identify_confounders(self):
        """Test confounder identification."""
        builder = CausalGraphBuilder(verbose=False)
        
        nodes = ['C', 'T', 'O', 'M']
        edges = [('C', 'T'), ('C', 'O'), ('T', 'M'), ('M', 'O')]
        
        builder.build_dag(nodes, edges)
        result = builder.identify_confounders('T', 'O')
        
        assert 'confounders' in result
        assert 'C' in result['confounders']
        assert result['status'] == 'SUCCESS'

    def test_identify_colliders(self):
        """Test collider identification."""
        builder = CausalGraphBuilder(verbose=False)
        
        nodes = ['T', 'O', 'C']
        edges = [('T', 'C'), ('O', 'C')]  # C is a collider
        
        builder.build_dag(nodes, edges)
        result = builder.identify_colliders('T', 'O')
        
        assert 'colliders' in result
        assert 'C' in result['colliders']

    def test_identify_mediators(self):
        """Test mediator identification."""
        builder = CausalGraphBuilder(verbose=False)
        
        nodes = ['T', 'M', 'O']
        edges = [('T', 'M'), ('M', 'O')]  # M is a mediator
        
        builder.build_dag(nodes, edges)
        result = builder.identify_mediators('T', 'O')
        
        assert 'mediators' in result
        assert 'M' in result['mediators']

    def test_detect_simpsons_paradox(self):
        """Test Simpson's Paradox detection."""
        # Create data where aggregate effect is positive, but stratified is negative
        n = 100
        df = pd.DataFrame({
            'treatment': np.concatenate([np.ones(50), np.zeros(50)]),
            'outcome': np.concatenate([
                np.random.normal(1.0, 0.5, 50),  # Treated: high outcome
                np.random.normal(0.5, 0.5, 50)   # Control: medium outcome
            ]),
            'group': np.concatenate([
                np.random.choice(['A', 'B'], 50),
                np.random.choice(['A', 'B'], 50)
            ])
        })
        
        builder = CausalGraphBuilder(verbose=False)
        result = builder.detect_simpsons_paradox(df, 'treatment', 'outcome', 'group')
        
        assert 'paradox_detected' in result
        assert 'overall_effect' in result
        assert 'stratified_effects' in result
        assert result['status'] == 'SUCCESS'

    def test_run_method(self):
        """Test run method (should be interactive)."""
        builder = CausalGraphBuilder(verbose=False)
        
        result = builder.run()
        
        assert 'module' in result
        assert 'findings' in result

    def test_edge_case_empty_graph(self):
        """Test with single node."""
        builder = CausalGraphBuilder(verbose=False)
        
        result = builder.build_dag(['X'], [])
        
        assert result['n_nodes'] == 1
        assert result['n_edges'] == 0
        assert result['is_acyclic'] is True

    def test_edge_case_no_path(self):
        """Test with disconnected nodes."""
        builder = CausalGraphBuilder(verbose=False)
        
        nodes = ['T', 'O', 'X', 'Y']
        edges = [('X', 'Y')]  # T and O are disconnected
        
        builder.build_dag(nodes, edges)
        result = builder.identify_confounders('T', 'O')
        
        # No confounders since T and O are disconnected
        assert len(result['confounders']) == 0
