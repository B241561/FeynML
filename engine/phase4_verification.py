"""
Phase 4 Verification Script
============================

Runs all Phase 4 modules on synthetic data to verify implementation.

Usage:
    python engine/phase4_verification.py
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Add project root to path
_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.label_noise import LabelNoiseAnalyzer
from engine.leakage_detector import LeakageDetector
from engine.missing_data import MissingDataAnalyzer
from engine.causal_thinking import CausalGraphBuilder
from engine.causal_inference import CausalInferenceEngine


def generate_synthetic_data():
    """Generate synthetic dataset for all modules."""
    print("Generating synthetic data...")
    
    n_samples = 200
    n_features = 10
    
    # Classification data
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    
    # Add some missing values
    missing_idx = np.random.choice(len(df), 30, replace=False)
    df.loc[missing_idx[:10], 'feature_0'] = np.nan
    df.loc[missing_idx[10:20], 'feature_1'] = np.nan
    df.loc[missing_idx[20:], 'feature_2'] = np.nan
    
    # Add date column for temporal analysis
    df['date'] = pd.date_range('2020-01-01', periods=len(df))
    
    # Add treatment column
    df['treatment'] = np.random.randint(0, 2, len(df))
    
    return df


def test_label_noise():
    """Test Label Noise Analyzer."""
    print("\n" + "="*70)
    print("MODULE 1: LABEL NOISE ANALYZER")
    print("="*70)
    
    try:
        # Generate data with noise
        X, y = make_classification(n_samples=150, n_features=8, random_state=42)
        y_noisy = y.copy()
        noise_idx = np.random.choice(len(y), 15, replace=False)
        y_noisy[noise_idx] = 1 - y_noisy[noise_idx]
        
        analyzer = LabelNoiseAnalyzer(verbose=False)
        result = analyzer.run(X, y_noisy)
        
        print(f"[OK] Status: {result['severity']}")
        print(f"[OK] Detected issues: {result['findings']['label_issues']['n_issues']}")
        print(f"[OK] Estimated noise rate: {result['findings']['label_issues']['estimated_noise_rate']*100:.1f}%")
        
        if result['findings']['label_issues']['status'] == 'SUCCESS':
            print("[PASS] Label Noise Analyzer")
            return True
        else:
            print("[FAIL] Label Noise Analyzer")
            return False
    
    except Exception as e:
        print(f"[FAIL] Label Noise Analyzer: {e}")
        return False


def test_leakage_detection():
    """Test Leakage Detector."""
    print("\n" + "="*70)
    print("MODULE 2: LEAKAGE DETECTOR")
    print("="*70)
    
    try:
        df = generate_synthetic_data()
        
        # Add obvious leakage
        df['leaky'] = df['target'] * 5 + np.random.normal(0, 0.1, len(df))
        
        detector = LeakageDetector(verbose=False)
        result = detector.run(df, 'target', date_col='date')
        
        # Check result structure
        if not isinstance(result, dict):
            print(f"[FAIL] Leakage Detector: Invalid result type {type(result)}")
            return False
        
        if 'findings' not in result:
            print(f"[FAIL] Leakage Detector: No findings in result. Keys: {result.keys()}")
            return False
        
        findings = result['findings']
        
        if 'target_leakage' not in findings:
            error_msg = findings.get('error', 'No error message')
            print(f"[FAIL] Leakage Detector: No target_leakage in findings")
            print(f"       Error: {error_msg}")
            print(f"       Available keys: {findings.keys()}")
            return False
        
        n_suspects = findings['target_leakage'].get('n_suspects', 0)
        print(f"[OK] Target leakage suspects: {n_suspects}")
        
        if n_suspects > 0:
            most_leaky = findings['target_leakage'].get('most_leaky_feature', 'unknown')
            print(f"[OK] Most leaky feature: {most_leaky}")
        
        if 'permutation_importance' in findings:
            n_spikes = findings['permutation_importance'].get('n_spikes', 0)
            print(f"[OK] Permutation importance spikes: {n_spikes}")
        
        if findings['target_leakage'].get('status') == 'SUCCESS':
            print("[PASS] Leakage Detector")
            return True
        else:
            print("[FAIL] Leakage Detector")
            return False
    
    except Exception as e:
        import traceback
        print(f"[FAIL] Leakage Detector: {e}")
        traceback.print_exc()
        return False


def test_missing_data():
    """Test Missing Data Analyzer."""
    print("\n" + "="*70)
    print("MODULE 3: MISSING DATA ANALYZER")
    print("="*70)
    
    try:
        df = generate_synthetic_data()
        
        analyzer = MissingDataAnalyzer(verbose=False)
        result = analyzer.run(df, target_col='target')
        
        n_missing_cols = result['findings']['classification']['summary']['total_missing_columns']
        total_missing_cells = result['findings']['classification']['summary']['total_missing_cells']
        
        print(f"[OK] Columns with missing values: {n_missing_cols}")
        print(f"[OK] Total missing cells: {total_missing_cells}")
        
        if n_missing_cols > 0:
            for col, info in result['findings']['classification']['missingness_by_column'].items():
                print(f"  - {col}: {info['mechanism']} ({info['missing_rate']*100:.1f}%)")
        
        print(f"[OK] Signal columns: {result['findings']['signal_analysis']['n_signal_columns']}")
        
        if result['findings']['classification']['status'] == 'SUCCESS':
            print("[PASS] Missing Data Analyzer")
            return True
        else:
            print("[FAIL] Missing Data Analyzer")
            return False
    
    except Exception as e:
        print(f"[FAIL] Missing Data Analyzer: {e}")
        return False


def test_causal_thinking():
    """Test Causal Graph Builder."""
    print("\n" + "="*70)
    print("MODULE 4: CAUSAL THINKING")
    print("="*70)
    
    try:
        builder = CausalGraphBuilder(verbose=False)
        
        # Example causal graph
        nodes = ['Confounder', 'Treatment', 'Outcome', 'Mediator']
        edges = [
            ('Confounder', 'Treatment'),
            ('Confounder', 'Outcome'),
            ('Treatment', 'Mediator'),
            ('Mediator', 'Outcome')
        ]
        
        build_result = builder.build_dag(nodes, edges)
        print(f"[OK] DAG built: {build_result['n_nodes']} nodes, {build_result['n_edges']} edges")
        
        # Identify structures
        confounders = builder.identify_confounders('Treatment', 'Outcome')
        print(f"[OK] Confounders: {confounders['confounders']}")
        
        mediators = builder.identify_mediators('Treatment', 'Outcome')
        print(f"[OK] Mediators: {mediators['mediators']}")
        
        colliders = builder.identify_colliders('Treatment', 'Outcome')
        print(f"[OK] Colliders: {colliders['colliders']}")
        
        # Simpson's Paradox test
        df = generate_synthetic_data()
        paradox = builder.detect_simpsons_paradox(
            df, 'treatment', 'target', 'feature_0'
        )
        print(f"[OK] Simpson's Paradox detected: {paradox['paradox_detected']}")
        
        print("[PASS] Causal Thinking")
        return True
    
    except Exception as e:
        print(f"[FAIL] Causal Thinking: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_causal_inference():
    """Test Causal Inference Engine."""
    print("\n" + "="*70)
    print("MODULE 5: CAUSAL INFERENCE")
    print("="*70)
    
    try:
        # Generate treatment data
        n = 200
        C = np.random.normal(0, 1, n)
        T = (C + np.random.normal(0, 1, n) > 0).astype(int)
        Y = 2 * T + 1.5 * C + np.random.normal(0, 1, n)
        
        df = pd.DataFrame({
            'confounder': C,
            'treatment': T,
            'outcome': Y
        })
        
        engine = CausalInferenceEngine(verbose=False)
        
        # ATE estimation
        ate_result = engine.estimate_ate(
            df, 'treatment', 'outcome',
            ['confounder'],
            method='ipw'
        )
        print(f"[OK] ATE (IPW): {ate_result['ate_estimate']:.4f}")
        print(f"  95% CI: [{ate_result['confidence_interval'][0]:.4f}, {ate_result['confidence_interval'][1]:.4f}]")
        
        # Adjustment method
        ate_adj = engine.estimate_ate(
            df, 'treatment', 'outcome',
            ['confounder'],
            method='adjustment'
        )
        print(f"[OK] ATE (Adjustment): {ate_adj['ate_estimate']:.4f}")
        
        # ATT estimation
        att_result = engine.estimate_att(
            df, 'treatment', 'outcome',
            ['confounder']
        )
        print(f"[OK] ATT: {att_result['att_estimate']:.4f}")
        
        # DiD test
        did_data = generate_synthetic_data()
        did_data['post_treatment'] = (did_data.index > len(did_data)//2).astype(int)
        
        did_result = engine.difference_in_differences(
            did_data, 'treatment', 'post_treatment', 'target'
        )
        print(f"[OK] DiD estimate: {did_result['did_estimate']:.4f}")
        
        print("[PASS] Causal Inference")
        return True
    
    except Exception as e:
        print(f"[FAIL] Causal Inference: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_verification():
    """Run all Phase 4 verifications."""
    print("\n" + "="*70)
    print("PHASE 4 VERIFICATION SUITE")
    print("Advanced Data Quality Analysis")
    print("="*70)
    
    results = {}
    
    # Test each module
    results['label_noise'] = test_label_noise()
    results['leakage'] = test_leakage_detection()
    results['missing_data'] = test_missing_data()
    results['causal_thinking'] = test_causal_thinking()
    results['causal_inference'] = test_causal_inference()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for module_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {module_name}")
    
    passed_count = sum(1 for p in results.values() if p)
    total_count = len(results)
    
    print("\n" + "-"*70)
    print(f"Result: {passed_count}/{total_count} modules passed")
    
    if passed_count == total_count:
        print("\nALL PHASE 4 MODULES VERIFIED SUCCESSFULLY!")
        return True
    else:
        print(f"\nWARNING: {total_count - passed_count} module(s) failed verification")
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
