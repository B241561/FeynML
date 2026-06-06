"""
Tests for Leakage Detector
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from engine.leakage_detector import LeakageDetector


@pytest.fixture
def clean_data():
    """Generate clean classification data."""
    X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                               n_redundant=3, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    df['target'] = y
    return df


@pytest.fixture
def leaky_data():
    """Generate data with obvious leakage."""
    X, y = make_classification(n_samples=200, n_features=8, n_informative=5,
                               n_redundant=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
    df['target'] = y
    
    # Add leaky feature: directly derived from target
    df['leaky_feature'] = df['target'] * 10 + np.random.normal(0, 0.1, len(df))
    
    return df


class TestLeakageDetector:

    def test_detect_target_leakage_clean(self, clean_data):
        """Test leakage detection on clean data."""
        detector = LeakageDetector(verbose=False)
        
        results = detector.detect_target_leakage(clean_data, 'target')
        
        assert 'leakage_suspects' in results
        assert 'n_suspects' in results
        assert results['status'] == 'SUCCESS'
        # Clean data should have few suspects
        assert results['n_suspects'] < 3

    def test_detect_target_leakage_leaky(self, leaky_data):
        """Test leakage detection detects added leakage."""
        detector = LeakageDetector(verbose=False)
        
        results = detector.detect_target_leakage(leaky_data, 'target')
        
        assert results['n_suspects'] > 0
        assert 'leaky_feature' in [s['feature'] for s in results['leakage_suspects']]

    def test_detect_temporal_leakage(self, clean_data):
        """Test temporal leakage detection."""
        clean_data['date'] = pd.date_range('2020-01-01', periods=len(clean_data))
        
        detector = LeakageDetector(verbose=False)
        results = detector.detect_temporal_leakage(clean_data, 'target', 'date')
        
        assert 'temporal_leakage_detected' in results
        assert 'suspicious_features' in results
        assert results['status'] == 'SUCCESS'

    def test_permutation_importance_spike(self, clean_data):
        """Test permutation importance spike detection."""
        detector = LeakageDetector(verbose=False)
        
        results = detector.permutation_importance_spike(clean_data, 'target')
        
        assert 'importance_scores' in results
        assert 'spike_flags' in results
        assert results['status'] == 'SUCCESS'

    def test_run_method(self, clean_data):
        """Test main run method."""
        detector = LeakageDetector(verbose=False)
        
        result = detector.run(clean_data, 'target')
        
        assert 'module' in result
        assert 'findings' in result
        assert 'severity' in result

    def test_edge_case_missing_target(self, clean_data):
        """Test error handling for missing target column."""
        detector = LeakageDetector(verbose=False)
        
        with pytest.raises(ValueError):
            detector.detect_target_leakage(clean_data, 'nonexistent_target')

    def test_edge_case_categorical_features(self):
        """Test with categorical features."""
        df = pd.DataFrame({
            'cat_feature': ['A', 'B', 'A', 'C', 'B'] * 40,
            'num_feature': np.random.randn(200),
            'target': np.random.randint(0, 2, 200)
        })
        
        detector = LeakageDetector(verbose=False)
        results = detector.detect_target_leakage(df, 'target')
        
        assert results['status'] == 'SUCCESS'

    def test_edge_case_single_feature(self):
        """Test with only one feature."""
        df = pd.DataFrame({
            'feature': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        detector = LeakageDetector(verbose=False)
        results = detector.detect_target_leakage(df, 'target')
        
        assert 'n_suspects' in results
