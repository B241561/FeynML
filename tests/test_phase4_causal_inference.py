"""
Tests for Causal Inference Engine
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

from engine.causal_inference import CausalInferenceEngine


@pytest.fixture
def treatment_data():
    """Generate data with treatment and outcome."""
    n = 200
    
    # Confounders
    C = np.random.normal(0, 1, n)
    
    # Treatment (influenced by confounder)
    T = (C + np.random.normal(0, 1, n) > 0).astype(int)
    
    # Outcome (influenced by both treatment and confounder)
    Y = 2 * T + 1.5 * C + np.random.normal(0, 1, n)
    
    df = pd.DataFrame({
        'confounder': C,
        'treatment': T,
        'outcome': Y
    })
    
    return df


class TestCausalInferenceEngine:

    def test_estimate_ate_ipw(self, treatment_data):
        """Test ATE estimation with IPW."""
        engine = CausalInferenceEngine(verbose=False)
        
        result = engine.estimate_ate(
            treatment_data,
            'treatment',
            'outcome',
            ['confounder'],
            method='ipw'
        )
        
        assert 'ate_estimate' in result
        assert 'confidence_interval' in result
        assert 'std_error' in result
        assert result['status'] == 'SUCCESS'
        # True effect is ~2, should be in ballpark
        assert 0 < result['ate_estimate'] < 4

    def test_estimate_ate_adjustment(self, treatment_data):
        """Test ATE estimation with adjustment."""
        engine = CausalInferenceEngine(verbose=False)
        
        result = engine.estimate_ate(
            treatment_data,
            'treatment',
            'outcome',
            ['confounder'],
            method='adjustment'
        )
        
        assert result['status'] == 'SUCCESS'
        assert 'ate_estimate' in result
        assert 0 < result['ate_estimate'] < 4

    def test_estimate_att(self, treatment_data):
        """Test ATT estimation."""
        engine = CausalInferenceEngine(verbose=False)
        
        result = engine.estimate_att(
            treatment_data,
            'treatment',
            'outcome',
            ['confounder']
        )
        
        assert 'att_estimate' in result
        assert 'confidence_interval' in result
        assert result['status'] == 'SUCCESS'

    def test_difference_in_differences(self):
        """Test DiD estimator."""
        # Create panel data
        n_units = 50
        n_time = 4
        
        df_list = []
        for unit in range(n_units):
            for t in range(n_time):
                treated = 1 if (unit > n_units // 2 and t > 1) else 0
                outcome = 1.0 + 0.5 * t + 2.0 * treated + np.random.normal(0, 0.5)
                df_list.append({
                    'unit': unit,
                    'time': t,
                    'treatment': treated,
                    'outcome': outcome
                })
        
        df = pd.DataFrame(df_list)
        
        engine = CausalInferenceEngine(verbose=False)
        result = engine.difference_in_differences(df, 'treatment', 'time', 'outcome')
        
        assert 'did_estimate' in result
        assert result['status'] == 'SUCCESS'
        # True effect is 2.0, should be in ballpark
        assert 0 < result['did_estimate'] < 4

    def test_natural_experiment_analysis(self):
        """Test IV (2SLS) analysis."""
        n = 200
        
        # Instrument: exogenous shock
        Z = np.random.normal(0, 1, n)
        
        # Endogenous treatment
        T = Z + np.random.normal(0, 0.5, n)
        
        # Outcome
        Y = 2 * T + np.random.normal(0, 1, n)
        
        df = pd.DataFrame({
            'instrument': Z,
            'treatment': T,
            'outcome': Y
        })
        
        engine = CausalInferenceEngine(verbose=False)
        result = engine.natural_experiment_analysis(
            df,
            'instrument',
            'treatment',
            'outcome'
        )
        
        # IV might not be available or skip if statsmodels missing
        if result['status'] != 'SKIPPED':
            assert 'iv_estimate' in result
            assert 'first_stage_f_stat' in result

    def test_run_method(self, treatment_data):
        """Test main run method."""
        engine = CausalInferenceEngine(verbose=False)
        
        result = engine.run(
            treatment_data,
            'treatment',
            'outcome',
            ['confounder']
        )
        
        assert 'module' in result
        assert 'findings' in result
        assert 'severity' in result

    def test_edge_case_no_covariates(self, treatment_data):
        """Test with no covariates (may be biased)."""
        engine = CausalInferenceEngine(verbose=False)
        
        result = engine.estimate_ate(
            treatment_data,
            'treatment',
            'outcome',
            []  # No covariates
        )
        
        assert result['status'] == 'SUCCESS'
        # Result may be biased but should still compute

    def test_edge_case_all_treated(self):
        """Test with all units treated."""
        df = pd.DataFrame({
            'treatment': np.ones(100),
            'outcome': np.random.randn(100),
            'confounder': np.random.randn(100)
        })
        
        engine = CausalInferenceEngine(verbose=False)
        
        with pytest.raises(ValueError):
            engine.estimate_ate(df, 'treatment', 'outcome', [])

    def test_edge_case_continuous_outcome(self):
        """Test with continuous outcome (regression)."""
        df = pd.DataFrame({
            'treatment': np.random.randint(0, 2, 100),
            'outcome': np.random.normal(5, 2, 100),  # Continuous
            'confounder': np.random.randn(100)
        })
        
        engine = CausalInferenceEngine(verbose=False)
        result = engine.estimate_ate(df, 'treatment', 'outcome', ['confounder'])
        
        assert result['status'] == 'SUCCESS'
