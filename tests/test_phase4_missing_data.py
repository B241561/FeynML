"""
Tests for Missing Data Analyzer
"""

import pytest
import numpy as np
import pandas as pd

from engine.missing_data import MissingDataAnalyzer


@pytest.fixture
def data_no_missing():
    """Complete data with no missing values."""
    return pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })


@pytest.fixture
def data_with_missing():
    """Data with missing values."""
    df = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    # Introduce MCAR-like missingness
    missing_idx = np.random.choice(len(df), 20, replace=False)
    df.loc[missing_idx, 'feature_1'] = np.nan
    
    # Introduce MAR-like missingness
    df.loc[df['feature_2'] > 1, 'feature_2'] = np.nan
    
    return df


class TestMissingDataAnalyzer:

    def test_classify_missingness_no_missing(self, data_no_missing):
        """Test classification with no missing values."""
        analyzer = MissingDataAnalyzer(verbose=False)
        
        results = analyzer.classify_missingness(data_no_missing)
        
        assert results['status'] == 'SUCCESS'
        assert len(results['missingness_by_column']) == 0
        assert results['summary']['no_missing_values'] is True

    def test_classify_missingness_with_missing(self, data_with_missing):
        """Test classification with missing values."""
        analyzer = MissingDataAnalyzer(verbose=False)
        
        results = analyzer.classify_missingness(data_with_missing)
        
        assert results['status'] == 'SUCCESS'
        assert len(results['missingness_by_column']) > 0
        assert results['summary']['no_missing_values'] is False
        assert results['summary']['total_missing_cells'] > 0

    def test_missingness_as_signal(self, data_with_missing):
        """Test missingness as signal analysis."""
        analyzer = MissingDataAnalyzer(verbose=False)
        
        results = analyzer.missingness_as_signal(data_with_missing, 'target')
        
        assert 'signal_columns' in results
        assert 'n_signal_columns' in results
        assert results['status'] == 'SUCCESS'

    def test_missingness_heatmap_data(self, data_with_missing):
        """Test co-occurrence analysis."""
        analyzer = MissingDataAnalyzer(verbose=False)
        
        results = analyzer.missingness_heatmap_data(data_with_missing)
        
        assert 'missing_correlation' in results
        assert 'co_missing_pairs' in results
        assert results['status'] == 'SUCCESS'

    def test_run_method(self, data_with_missing):
        """Test main run method."""
        analyzer = MissingDataAnalyzer(verbose=False)
        
        result = analyzer.run(data_with_missing, target_col='target')
        
        assert 'module' in result
        assert 'findings' in result
        assert 'severity' in result

    def test_edge_case_all_missing_column(self):
        """Test column that is all NaN."""
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'all_missing': [np.nan] * 100,
            'target': np.random.randint(0, 2, 100)
        })
        
        analyzer = MissingDataAnalyzer(verbose=False)
        results = analyzer.classify_missingness(df)
        
        assert 'all_missing' in results['missingness_by_column']
        assert results['missingness_by_column']['all_missing']['missing_rate'] == 1.0

    def test_edge_case_sparse_missing(self):
        """Test with very sparse missing values."""
        df = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Only 1 missing value
        df.loc[0, 'feature_1'] = np.nan
        
        analyzer = MissingDataAnalyzer(verbose=False)
        results = analyzer.classify_missingness(df)
        
        assert len(results['missingness_by_column']) == 1
