"""
DoWhy Integration Module
========================
Wrapper around the DoWhy library for production causal inference.

Four-step DoWhy workflow:
1. Model — define causal graph
2. Identify — find estimand using do-calculus
3. Estimate — compute effect
4. Refute — sensitivity analysis
"""

import pandas as pd
import numpy as np

try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False

def build_causal_model(df, treatment, outcome, dag_str=None, common_causes=None):
    """
    Build DoWhy CausalModel.
    """
    if not DOWHY_AVAILABLE:
        return None
        
    model = CausalModel(
        data=df,
        treatment=treatment,
        outcome=outcome,
        graph=dag_str,
        common_causes=common_causes
    )
    return model

def identify_effect(model):
    """Identify causal estimand."""
    if model is None: return None
    return model.identify_effect(proceed_when_unidentifiable=True)

def estimate_effect(model, estimand, method="backdoor.propensity_score_weighting"):
    """Estimate causal effect."""
    if model is None or estimand is None: return None
    return model.estimate_effect(estimand, method_name=method)

def refute_estimate(model, estimand, estimate, method="placebo_treatment_refuter"):
    """Run refutation tests on estimate."""
    if model is None or estimate is None: return None
    return model.refute_estimate(estimand, estimate, method_name=method)

def full_causal_analysis(df, treatment, outcome, dag_str=None, method="backdoor.linear_regression"):
    """Run complete causal analysis pipeline."""
    if not DOWHY_AVAILABLE:
        return {"error": "DoWhy not installed. Please install with 'pip install dowhy'."}
        
    model = build_causal_model(df, treatment, outcome, dag_str)
    estimand = identify_effect(model)
    estimate = estimate_effect(model, estimand, method=method)
    
    res = {
        "estimand": str(estimand),
        "value": float(estimate.value),
        "method": method
    }
    
    # Run a simple refutation
    try:
        refutation = refute_estimate(model, estimand, estimate)
        res["refutation"] = str(refutation)
    except:
        res["refutation"] = "Refutation failed or not applicable."
        
    return res

def run_verification():
    """Run module verification."""
    print("--- DoWhy Integration Verification ---")
    if not DOWHY_AVAILABLE:
        print("DoWhy not available. Skipping full verification.")
        return
        
    # Simple synthetic data
    N = 200
    z = np.random.normal(size=N)
    d = np.where(z + np.random.normal(size=N) > 0, 1, 0)
    y = d + z + np.random.normal(size=N)
    df = pd.DataFrame({'d': d, 'y': y, 'z': z})
    
    # Analysis
    res = full_causal_analysis(df, 'd', 'y', common_causes=['z'])
    print(f"Causal Effect: {res['value']:.4f} (True: 1.00)")
    print(f"Estimand: {res['estimand'][:100]}...")

if __name__ == "__main__":
    run_verification()
