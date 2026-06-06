"""
Tests for Label Noise Analyzer
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from engine.label_noise import LabelNoiseAnalyzer


@pytest.fixture
def synthetic_data():
    """Generate synthetic classification data."""
    X, y = make_classification(n_samples=200, n_features=10, n_informative=8,
                               n_redundant=2, random_state=42)
    return X, y


@pytest.fixture
def data_with_noise(synthetic_data):
    """Add label noise to synthetic data."""
    X, y = synthetic_data
    y_noisy = y.copy()
    # Flip 10% of labels
    noise_indices = np.random.choice(len(y), size=20, replace=False)
    y_noisy[noise_indices] = 1 - y_noisy[noise_indices]
    return X, y_noisy, y


class TestLabelNoiseAnalyzer:

    def test_confident_learning_basic(self, synthetic_data):
        """Test basic confident learning functionality."""
        X, y = synthetic_data
        analyzer = LabelNoiseAnalyzer(verbose=False)
        
        results = analyzer.confident_learning(X, y)
        
        assert 'label_issues' in results
        assert 'label_issues_df' in results
        assert 'n_issues' in results
        assert 'estimated_noise_rate' in results
        assert results['status'] == 'SUCCESS'
        assert isinstance(results['label_issues'], np.ndarray)
        assert len(results['label_issues']) == len(X)

    def test_confident_learning_with_noise(self, data_with_noise):
        """Test confident learning detects added noise."""
        X, y_noisy, y_true = data_with_noise
        analyzer = LabelNoiseAnalyzer(verbose=False)
        
        results = analyzer.confident_learning(X, y_noisy)
        
        assert results['estimated_noise_rate'] > 0.0
        assert results['n_issues'] > 0

    def test_asymmetric_noise_matrix_no_clean(self, synthetic_data):
        """Test noise matrix estimation without clean labels."""
        X, y = synthetic_data
        analyzer = LabelNoiseAnalyzer(verbose=False)
        
        results = analyzer.asymmetric_noise_matrix(y)
        
        assert 'noise_matrix' in results
        assert 'noise_rates' in results
        assert results['status'] == 'SUCCESS'

    def test_asymmetric_noise_matrix_with_clean(self, data_with_noise):
        """Test noise matrix estimation with clean labels."""
        X, y_noisy, y_true = data_with_noise
        analyzer = LabelNoiseAnalyzer(verbose=False)
        
        results = analyzer.asymmetric_noise_matrix(y_noisy, y_clean=y_true)
        
        assert 'noise_matrix' in results
        assert 'noise_rates' in results
        # Diagonal should be > off-diagonal (mostly correct labels)
        assert np.mean(np.diag(results['noise_matrix'])) > 0.5

    def test_noise_summary(self, synthetic_data):
        """Test summary generation."""
        X, y = synthetic_data
        analyzer = LabelNoiseAnalyzer(verbose=False)
        
        results = analyzer.confident_learning(X, y)
        summary = analyzer.noise_summary(results, y)
        
        assert 'total_samples' in summary
        assert 'estimated_noise_rate_pct' in summary
        assert 'n_issues_found' in summary
        assert 'summary_text' in summary

    def test_run_method(self, synthetic_data):
        """Test main run method."""
        X, y = synthetic_data
        analyzer = LabelNoiseAnalyzer(verbose=False)
        
        result = analyzer.run(X, y)
        
        assert 'module' in result
        assert 'findings' in result
        assert 'severity' in result
        assert 'status' in result['findings']

    def test_edge_case_small_dataset(self):
        """Test with very small dataset."""
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        
        analyzer = LabelNoiseAnalyzer(verbose=False)
        results = analyzer.confident_learning(X, y)
        
        # Should still work, maybe warn
        assert 'label_issues' in results

    def test_edge_case_binary_classification(self):
        """Test binary classification explicitly."""
        X, y = make_classification(n_samples=100, n_features=5, n_classes=2,
                                   random_state=42)
        analyzer = LabelNoiseAnalyzer(verbose=False)
        
        results = analyzer.confident_learning(X, y)
        
        assert results['status'] == 'SUCCESS'

    def test_edge_case_all_same_label(self):
        """Test with all samples having same label."""
        X = np.random.randn(50, 5)
        y = np.zeros(50)
        
        analyzer = LabelNoiseAnalyzer(verbose=False)
        results = analyzer.confident_learning(X, y)
        
        # Should handle gracefully
        assert 'label_issues' in results
