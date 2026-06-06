"""
Causal Inference Module
=======================

Estimates causal effects from observational data:
  • Average Treatment Effect (ATE)
  • Average Treatment Effect on Treated (ATT)
  • Difference-in-Differences (DiD)
  • Instrumental Variables (IV)
  • DoWhy framework integration

Educational Focus:
  Causal inference turns correlations into estimated causal effects:
  1. Requires assumptions (confounding, overlap, SUTVA)
  2. Different methods for different data structures
  3. Always report confidence intervals and sensitivity analysis
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import warnings

from engine.base_module import BaseModule

try:
    from dowhy import CausalModel
    HAS_DOWHY = True
except ImportError:
    HAS_DOWHY = False

try:
    import statsmodels.api as sm
    from statsmodels.regression.linear_model import RegressionResults
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class CausalInferenceEngine(BaseModule):
    """
    Estimates causal effects using multiple methods.
    
    Educational Philosophy:
      Different causal inference methods suit different data structures.
      This module teaches when and how to use each approach.
    """

    def estimate_ate(self, df, treatment_col, outcome_col, covariates, method="ipw", ci=0.95):
        """
        Estimate Average Treatment Effect (ATE).
        
        WHY THIS MATTERS:
          ATE answers: "On average, what is the causal effect of treatment?"
          ATE is the parameter of interest for policy decisions.
        
        Parameters
        ----------
        df : pandas.DataFrame
        treatment_col : str
            Binary treatment (0/1)
        outcome_col : str
        covariates : list of str
            Confounding variables to adjust for
        method : str
            'ipw' (inverse propensity weighting) or 'adjustment'
        ci : float
            Confidence level (0.95 = 95% CI)
        
        Returns
        -------
        dict with keys:
          'ate_estimate' : float
          'confidence_interval' : (lower, upper)
          'std_error' : float
          'method' : str
          'assumptions' : str
          'interpretation' : str
        """
        self._log(f"Estimating ATE: {treatment_col} → {outcome_col}, method={method}")
        
        if treatment_col not in df.columns or outcome_col not in df.columns:
            raise ValueError("Treatment or outcome column not found")
        
        # Prepare data
        T = df[treatment_col].values.astype(float)
        Y = df[outcome_col].values.astype(float)

        # BUG FIX 1: Check if treatment is continuous
        n_unique_t = len(np.unique(T))
        if method == "ipw" and n_unique_t > 10:
            self._log(f"Treatment '{treatment_col}' has {n_unique_t} unique values. Switching from IPW to linear regression adjustment (backdoor.linear_regression).")
            method = "adjustment"
        
        # Prepare covariates
        if len(covariates) == 0:
            self._warn("No covariates provided—ATE may be biased due to confounding")
            X = np.ones((len(df), 1))
        else:
            X_list = []
            for col in covariates:
                if col in df.columns:
                    X_list.append(df[col].values.astype(float))
            X = np.column_stack(X_list) if X_list else np.ones((len(df), 1))
        
        # Standardize X
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if method == "ipw":
            # Inverse Propensity Weighting
            # 1. Estimate propensity score P(T=1|X)
            lr = LogisticRegression(random_state=42, max_iter=1000)
            lr.fit(X_scaled, T)
            propensity = lr.predict_proba(X_scaled)[:, 1]
            
            # Avoid extreme propensity scores
            propensity = np.clip(propensity, 0.01, 0.99)
            
            # 2. Compute IPW weights
            weights = np.where(T == 1, 1 / propensity, 1 / (1 - propensity))
            
            # 3. Weighted difference in means
            Y_treated = Y[T == 1]
            Y_control = Y[T == 0]
            W_treated = weights[T == 1]
            W_control = weights[T == 0]
            
            if len(Y_treated) == 0 or len(Y_control) == 0:
                raise ValueError("No treated or control observations")
            
            ate = np.sum(W_treated * Y_treated) / np.sum(W_treated) - \
                  np.sum(W_control * Y_control) / np.sum(W_control)
            
            # SE approximation
            var_treated = np.sum(W_treated**2 * (Y_treated - np.mean(Y_treated))**2) / (np.sum(W_treated)**2)
            var_control = np.sum(W_control**2 * (Y_control - np.mean(Y_control))**2) / (np.sum(W_control)**2)
            se = np.sqrt(var_treated + var_control)
        
        else:  # method == "adjustment"
            # Linear regression adjustment
            X_design = np.column_stack([T, X_scaled])
            lr_model = LinearRegression()
            lr_model.fit(X_design, Y)
            ate = lr_model.coef_[0]
            
            # SE approximation via residuals
            Y_pred = lr_model.predict(X_design)
            residuals = Y - Y_pred
            mse = np.mean(residuals**2)
            X_T = X_design.T @ X_design
            try:
                var_ate = mse * np.linalg.inv(X_T)[0, 0]
                se = np.sqrt(var_ate)
            except:
                se = np.nan
        
        # Confidence interval
        z_score = norm.ppf(1 - (1 - ci) / 2)
        ci_lower = ate - z_score * se
        ci_upper = ate + z_score * se
        
        assumptions = (
            "Key assumptions:\n"
            "  • No unmeasured confounding\n"
            "  • Positivity (0 < P(T=1|X) < 1)\n"
            "  • Consistency (no interference)\n"
            "  • SUTVA (no hidden variations)\n"
        )
        
        interpretation = (
            f"Average Treatment Effect ({method}):\n"
            f"  • Point estimate: {ate:.4f}\n"
            f"  • {ci*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
            f"  • Std error: {se:.4f}\n"
            f"  • Interpretation: Treatment increases outcome by {ate:.4f} units on average\n"
        )
        
        return {
            'ate_estimate': float(ate),
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'std_error': float(se),
            'method': method,
            'assumptions': assumptions,
            'interpretation': interpretation,
            'status': 'SUCCESS'
        }

    def estimate_att(self, df, treatment_col, outcome_col, covariates, ci=0.95):
        """
        Estimate Average Treatment Effect on the Treated (ATT).
        
        WHY THIS MATTERS:
          ATT answers: "For those who received treatment, what was the effect?"
          Useful for cost-benefit analysis of actual users.
        
        Parameters
        ----------
        df : pandas.DataFrame
        treatment_col : str
        outcome_col : str
        covariates : list of str
        ci : float
        
        Returns
        -------
        dict with keys:
          'att_estimate' : float
          'confidence_interval' : tuple
          'interpretation' : str
        """
        self._log(f"Estimating ATT: {treatment_col} → {outcome_col}")
        
        T = df[treatment_col].values.astype(float)
        Y = df[outcome_col].values.astype(float)
        
        if len(covariates) == 0:
            X = np.ones((len(df), 1))
        else:
            X_list = [df[col].values.astype(float) for col in covariates if col in df.columns]
            X = np.column_stack(X_list) if X_list else np.ones((len(df), 1))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Propensity score matching variant for ATT
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_scaled, T)
        propensity = lr.predict_proba(X_scaled)[:, 1]
        
        # For treated units, find matched controls
        treated_mask = T == 1
        control_mask = T == 0
        
        Y_treated = Y[treated_mask]
        
        # Simple ATT: treated minus weighted controls
        propensity_controls = propensity[control_mask]
        Y_controls = Y[control_mask]
        
        # Weight controls by propensity
        weights_controls = propensity_controls / (1 - propensity_controls)
        weights_controls = weights_controls / np.sum(weights_controls) * len(Y_controls)
        
        att = np.mean(Y_treated) - np.sum(weights_controls * Y_controls) / len(Y_controls)
        
        # SE approximation
        se = np.std(Y_treated) / np.sqrt(len(Y_treated)) + np.std(Y_controls) / np.sqrt(len(Y_controls))
        
        z_score = norm.ppf(1 - (1 - ci) / 2)
        ci_lower = att - z_score * se
        ci_upper = att + z_score * se
        
        interpretation = (
            f"Average Treatment Effect on Treated:\n"
            f"  • ATT estimate: {att:.4f}\n"
            f"  • {ci*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]\n"
            f"  • Applies only to those who received treatment\n"
        )
        
        return {
            'att_estimate': float(att),
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'interpretation': interpretation,
            'status': 'SUCCESS'
        }

    def difference_in_differences(self, df, treatment_col, time_col, outcome_col):
        """
        Difference-in-Differences estimator for panel data.
        
        WHY THIS MATTERS:
          DiD compares trends before/after treatment:
          1. Assumes parallel trends (would have same trajectory without treatment)
          2. Removes time-invariant confounding
          3. Works even with unobserved confounders
        
        Parameters
        ----------
        df : pandas.DataFrame
            Panel data (multiple observations per unit over time)
        treatment_col : str
            Binary treatment indicator
        time_col : str
            Time period identifier
        outcome_col : str
        
        Returns
        -------
        dict with keys:
          'did_estimate' : float
          'parallel_trends_test' : bool
          'interpretation' : str
        """
        self._log(f"Computing DID: {treatment_col} × time → {outcome_col}")
        
        # Create interaction term
        df_temp = df.copy()
        df_temp['treated'] = df[treatment_col].astype(float)
        df_temp['post'] = (df[time_col] > df[time_col].median()).astype(float)
        df_temp['interaction'] = df_temp['treated'] * df_temp['post']
        
        # Regress: Y ~ treatment + post + interaction
        X = df_temp[['treated', 'post', 'interaction']].values
        X = np.column_stack([np.ones(len(X)), X])
        Y = df_temp[outcome_col].values
        
        model = LinearRegression()
        model.fit(X, Y)
        did_estimate = model.coef_[3]  # Coefficient on interaction
        
        # Parallel trends test: regress pre-treatment difference on group
        df_pre = df_temp[df_temp['post'] == 0]
        if len(df_pre) > 0:
            X_pre = df_pre[['treated']].values
            X_pre = np.column_stack([np.ones(len(X_pre)), X_pre])
            Y_pre = df_pre[outcome_col].values
            
            model_pre = LinearRegression()
            model_pre.fit(X_pre, Y_pre)
            
            # If pre-treatment trend differs, parallel trends violated
            parallel_trends_ok = abs(model_pre.coef_[1]) < abs(did_estimate)
        else:
            parallel_trends_ok = True
        
        interpretation = (
            f"Difference-in-Differences Estimator:\n"
            f"  • DiD estimate: {did_estimate:.4f}\n"
            f"  • Parallel trends assumption: {'likely OK' if parallel_trends_ok else 'VIOLATED'}\n"
            f"  • Interpretation: {did_estimate:.4f} units change in outcome due to treatment\n"
        )
        
        return {
            'did_estimate': float(did_estimate),
            'parallel_trends_assumption': bool(parallel_trends_ok),
            'interpretation': interpretation,
            'status': 'SUCCESS'
        }

    def natural_experiment_analysis(self, df, instrument_col, treatment_col, outcome_col):
        """
        Instrumental Variable approach (2SLS).
        
        WHY THIS MATTERS:
          IV methods work when you have an exogenous shock that affects treatment.
          Example: lottery winner status as IV for wealth → health.
        
        Parameters
        ----------
        df : pandas.DataFrame
        instrument_col : str
            Exogenous variable that affects treatment but not outcome directly
        treatment_col : str
        outcome_col : str
        
        Returns
        -------
        dict with keys:
          'iv_estimate' : float
          'first_stage_f_stat' : float
          'interpretation' : str
        """
        if not HAS_STATSMODELS:
            self._warn("statsmodels not available—skipping 2SLS")
            return {'status': 'SKIPPED', 'reason': 'statsmodels not installed'}
        
        self._log(f"Computing IV: {instrument_col} → {treatment_col} → {outcome_col}")
        
        # First stage: instrument → treatment
        Z = sm.add_constant(df[instrument_col].values)
        T = df[treatment_col].values
        
        model_fs = sm.OLS(T, Z).fit()
        first_stage_f = model_fs.fvalue
        
        # Second stage: predicted treatment → outcome
        T_hat = model_fs.predict(Z)
        X_2s = sm.add_constant(T_hat)
        Y = df[outcome_col].values
        
        model_ss = sm.OLS(Y, X_2s).fit()
        iv_estimate = model_ss.params[1]
        
        interpretation = (
            f"Instrumental Variable (2SLS) Estimate:\n"
            f"  • IV: {instrument_col}\n"
            f"  • First-stage F-stat: {first_stage_f:.2f} (>10 desired)\n"
            f"  • IV estimate: {iv_estimate:.4f}\n"
            f"  • Valid if: F-stat >10 AND instrument exogenous\n"
        )
        
        return {
            'iv_estimate': float(iv_estimate),
            'first_stage_f_stat': float(first_stage_f),
            'interpretation': interpretation,
            'status': 'SUCCESS'
        }

    def dowhy_analysis(self, df, treatment_col, outcome_col, common_causes, method="backdoor"):
        """
        DoWhy framework for causal inference.
        
        Parameters
        ----------
        df : pandas.DataFrame
        treatment_col : str
        outcome_col : str
        common_causes : list of str
            Confounding variables
        method : str
            'backdoor' or 'frontdoor'
        
        Returns
        -------
        dict with keys:
          'effect_estimate' : float
          'refutation_results' : dict
          'status' : str
        """
        if not HAS_DOWHY:
            self._warn("DoWhy not installed—skipping causal model")
            return {'status': 'SKIPPED', 'reason': 'DoWhy not installed'}
        
        self._log(f"Running DoWhy analysis")
        
        try:
            # Define causal graph
            gml_graph = f"""
            digraph {{
            {treatment_col} [label="{treatment_col}"];
            {outcome_col} [label="{outcome_col}"];
            {'; '.join(f'{c} [label="{c}"]' for c in common_causes)};
            {'; '.join(f'{c} -> {treatment_col}' for c in common_causes)};
            {'; '.join(f'{c} -> {outcome_col}' for c in common_causes)};
            {treatment_col} -> {outcome_col};
            }}
            """
            
            model = CausalModel(
                data=df,
                treatment=treatment_col,
                outcome=outcome_col,
                common_causes=common_causes,
                graph=gml_graph
            )
            
            # Identify causal effect
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
            
            # Estimate
            estimate = model.estimate_effect(
                identified_estimand,
                method_name="backdoor.linear_regression"
            )
            
            return {
                'effect_estimate': float(estimate.value),
                'dowhy_summary': str(estimate),
                'status': 'SUCCESS'
            }
        
        except Exception as e:
            self._error(f"DoWhy analysis failed: {e}")
            return {'status': 'FAILED', 'error': str(e)}

    def run(self, df, treatment_col, outcome_col, covariates=None, **kwargs):
        """
        Main entry point for causal inference.
        
        Parameters
        ----------
        df : pandas.DataFrame
        treatment_col : str
        outcome_col : str
        covariates : list of str, optional
        **kwargs : additional arguments
        
        Returns
        -------
        dict (wrapped by BaseModule._result)
        """
        if covariates is None:
            covariates = []
        
        try:
            findings = {}
            
            # Resolve columns
            treatment_col = self._resolve_target(df, treatment_col)
            outcome_col = self._resolve_target(df, outcome_col)
            
            # Estimate ATE
            ate_result = self.estimate_ate(df, treatment_col, outcome_col, covariates)
            
            # Estimate ATT
            att_result = self.estimate_att(df, treatment_col, outcome_col, covariates)
            
            severity = 'NONE'
            if abs(ate_result['ate_estimate']) > 0:
                severity = 'LOW'  # Just reporting results
            
            findings = self._serialize_findings({
                'ate': ate_result,
                'att': att_result,
                'treatment_col': treatment_col,
                'outcome_col': outcome_col,
                'covariates': covariates
            })

            return self._result(findings, severity=severity, module_name="CausalInferenceEngine")
        
        except Exception as e:
            self._error(f"Causal inference failed: {e}")
            error_findings = self._serialize_findings({'error': str(e), 'status': 'FAILED'})
            return self._result(error_findings, severity='CRITICAL')
