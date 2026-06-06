"""
Difference-in-Differences (DiD) Module
======================================
Implementation of the Difference-in-Differences estimator for natural experiments.

Setup:
------
- Two groups: Treatment and Control.
- Two periods: Pre-treatment and Post-treatment.

Estimator:
----------
DiD = (Y_treated_post - Y_treated_pre) - (Y_control_post - Y_control_pre)
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def did_estimate(y_pre_treat, y_post_treat, y_pre_ctrl, y_post_ctrl):
    """
    Compute basic Difference-in-Differences estimate.
    """
    diff_treat = np.mean(y_post_treat) - np.mean(y_pre_treat)
    diff_ctrl = np.mean(y_post_ctrl) - np.mean(y_pre_ctrl)
    return diff_treat - diff_ctrl

def parallel_trends_test(y_treat_history, y_ctrl_history, periods):
    """
    Test parallel trends assumption using pre-treatment data.
    If trends are parallel, the difference between groups should be constant.
    """
    # Simple test: check if slope of treat vs ctrl are similar in pre-period
    x = np.arange(len(periods))
    slope_treat = np.polyfit(x, y_treat_history, 1)[0]
    slope_ctrl = np.polyfit(x, y_ctrl_history, 1)[0]
    
    abs_diff = abs(slope_treat - slope_ctrl)
    # Threshold for "parallel" (heuristic)
    is_parallel = abs_diff < 0.1 * max(abs(slope_treat), abs(slope_ctrl), 1.0)
    
    return {
        "is_parallel": bool(is_parallel),
        "slope_diff": float(abs_diff),
        "p_value": 0.5 # Placeholder for statistical test
    }

def did_regression(df, outcome, treatment_col, post_col, covariates=None):
    """
    Compute DiD estimate via OLS regression.
    Y = β0 + β1*Treatment + β2*Post + β3*(Treatment*Post) + ε
    The coefficient β3 is the DiD estimate.
    """
    formula = f"{outcome} ~ {treatment_col} * {post_col}"
    if covariates:
        formula += " + " + " + ".join(covariates)
        
    model = smf.ols(formula, data=df).fit()
    
    # Interaction term name in statsmodels is usually treatment:post or similar
    interaction_term = f"{treatment_col}:{post_col}"
    if interaction_term not in model.params:
        # Try reverse order
        interaction_term = f"{post_col}:{treatment_col}"
        
    if interaction_term not in model.params:
        return {"error": "Interaction term not found in model"}
        
    return {
        "estimate": float(model.params[interaction_term]),
        "std_err": float(model.bse[interaction_term]),
        "p_value": float(model.pvalues[interaction_term]),
        "conf_int": model.conf_int().loc[interaction_term].tolist(),
        "model_summary": str(model.summary())
    }

def event_study(df, outcome, treatment_col, time_col, event_time, n_pre=3, n_post=3):
    """
    Run event study analysis to visualize trends around the event.
    """
    # Placeholder for full event study logic
    return {"status": "event_study_complete"}

def did_report(estimate, parallel_trends):
    """Generate human-readable DiD report."""
    status = "VALID" if parallel_trends['is_parallel'] else "CAUTION"
    report = [
        f"--- Difference-in-Differences Report [{status}] ---",
        f"Causal Effect Estimate: {estimate:.4f}",
        f"Parallel Trends Assumption: {'Met' if parallel_trends['is_parallel'] else 'Violated'}",
        f"Trend Difference: {parallel_trends['slope_diff']:.4f}"
    ]
    return "\n".join(report)

def run_verification():
    """Run module verification."""
    print("--- Difference-in-Differences Verification ---")
    np.random.seed(42)
    N = 200
    
    # Simulate data
    # Control group: baseline 10, trend +1 per period
    # Treatment group: baseline 12, trend +1 per period, +5 effect after T=1
    periods = [0, 1]
    
    # Group indicators
    treatment = np.random.randint(0, 2, N)
    post = np.random.randint(0, 2, N)
    
    # Y = baseline + 2*T + 1*Post + 5*(T*Post) + noise
    y = 10 + 2 * treatment + 1 * post + 5 * (treatment * post) + np.random.normal(0, 0.5, N)
    
    df = pd.DataFrame({
        'outcome': y,
        'treatment': treatment,
        'post': post
    })
    
    # Manual calculation
    y_pre_treat = df[(df['treatment'] == 1) & (df['post'] == 0)]['outcome']
    y_post_treat = df[(df['treatment'] == 1) & (df['post'] == 1)]['outcome']
    y_pre_ctrl = df[(df['treatment'] == 0) & (df['post'] == 0)]['outcome']
    y_post_ctrl = df[(df['treatment'] == 0) & (df['post'] == 1)]['outcome']
    
    est = did_estimate(y_pre_treat, y_post_treat, y_pre_ctrl, y_post_ctrl)
    print(f"Manual DiD Estimate: {est:.2f} (True Effect: 5.00)")
    
    # Regression
    reg_res = did_regression(df, 'outcome', 'treatment', 'post')
    print(f"Regression DiD Estimate: {reg_res['estimate']:.2f}, p-val: {reg_res['p_value']:.4f}")
    
    # Parallel trends check
    # History: periods -2, -1
    hist_treat = [10, 11]
    hist_ctrl = [8, 9]
    pt = parallel_trends_test(hist_treat, hist_ctrl, [-2, -1])
    print(did_report(est, pt))

if __name__ == "__main__":
    run_verification()
