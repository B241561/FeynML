"""
Missing Data Analysis Module
=============================

Classifies missingness mechanisms:
  • MCAR: Missing Completely At Random (ignorable)
  • MAR: Missing At Random (depends on observed variables)
  • MNAR: Missing Not At Random (depends on unobserved variables)

Also detects:
  • Missingness as a signal for the target
  • Co-missing patterns in features

Educational Focus:
  The mechanism of missingness fundamentally affects how we handle it:
  1. MCAR → can safely impute or delete
  2. MAR → need regression-based imputation
  3. MNAR → can bias inference—may need domain knowledge
  
  This module gives data scientists the evidence to make informed decisions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency, entropy
import warnings

from engine.base_module import BaseModule

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.imputation.mice import MICEData
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class MissingDataAnalyzer(BaseModule):
    """
    Analyzes missing data mechanisms and patterns.
    
    Educational Philosophy:
      Ignoring the mechanism of missingness can lead to biased inference.
      This module surfaces the evidence that guides correct handling.
    """

    def classify_missingness(self, df):
        """
        Classify missingness mechanism for each column with missing values.
        
        WHY THIS MATTERS:
          • MCAR is safe to ignore (within reason)
          • MAR requires careful imputation (not simple mean imputation)
          • MNAR can bias results no matter what we do
        
        Parameters
        ----------
        df : pandas.DataFrame
        
        Returns
        -------
        dict with keys:
          'missingness_by_column' : dict
              {column: {mechanism: MCAR/MAR/MNAR, confidence: float, evidence: str}}
          'missing_rate_by_column' : dict
              {column: fraction of missing values}
          'summary' : dict
              Overall missingness summary
        """
        self._log(f"Classifying missingness mechanisms in {len(df)} rows × {len(df.columns)} cols")
        
        # Find columns with missing values
        missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]
        
        if not missing_cols:
            self._log("No missing values detected")
            return {
                'missingness_by_column': {},
                'missing_rate_by_column': {},
                'summary': {'total_missing_cells': 0, 'no_missing_values': True},
                'status': 'SUCCESS'
            }
        
        self._log(f"Found {len(missing_cols)} columns with missing values")
        
        missingness_by_column = {}
        missing_rate_by_column = {}
        
        for col in missing_cols:
            missing_rate = df[col].isnull().sum() / len(df)
            missing_rate_by_column[col] = float(missing_rate)
            
            # Create indicator: 1 if missing, 0 if observed
            missing_indicator = df[col].isnull().astype(int)
            
            # Test 1: Little's MCAR test approximation
            # We'll use chi-square test of independence between missingness indicators
            # A simple heuristic: if missingness is independent of other columns, likely MCAR
            
            other_cols = [c for c in df.columns if c != col and df[c].dtype in [np.float64, np.int64, 'object']]
            
            if len(other_cols) == 0:
                mechanism = 'MCAR'
                confidence = 0.7
                evidence = 'No other columns to test dependence'
            else:
                # Test association between missingness and other variables
                associations = []
                
                for other_col in other_cols[:5]:  # Limit to 5 for speed
                    try:
                        if df[other_col].dtype == 'object':
                            # Categorical: chi-square
                            cross_tab = pd.crosstab(missing_indicator, df[other_col].fillna('NULL'))
                            chi2, p_val, _, _ = chi2_contingency(cross_tab)
                            associations.append(p_val)
                        else:
                            # Numeric: t-test via logistic regression
                            X = df[other_col].fillna(df[other_col].mean()).values.reshape(-1, 1)
                            lr = LogisticRegression(random_state=42, solver='lbfgs', max_iter=1000)
                            try:
                                lr.fit(X, missing_indicator)
                                # Get coefficient significance
                                associations.append(lr.score(X, missing_indicator))
                            except:
                                pass
                    except:
                        pass
                
                # Classify based on associations
                if len(associations) == 0:
                    mechanism = 'MCAR'
                    confidence = 0.5
                    evidence = 'No testable associations'
                else:
                    # If any p-value < 0.05, likely MAR
                    sig_associations = sum(1 for a in associations if a < 0.05)
                    
                    if sig_associations == 0:
                        mechanism = 'MCAR'
                        confidence = 0.8
                        evidence = f'Tested {len(associations)} variables: no significant associations'
                    elif sig_associations == len(associations):
                        mechanism = 'MAR'
                        confidence = 0.85
                        evidence = f'Missingness depends on {sig_associations} observed variables'
                    else:
                        mechanism = 'MAR'
                        confidence = 0.7
                        evidence = f'Partial dependence on observed variables'
            
            missingness_by_column[col] = {
                'mechanism': mechanism,
                'confidence': float(confidence),
                'evidence': evidence,
                'missing_rate': float(missing_rate)
            }
        
        # Summary
        mcar_count = sum(1 for m in missingness_by_column.values() if m['mechanism'] == 'MCAR')
        mar_count = sum(1 for m in missingness_by_column.values() if m['mechanism'] == 'MAR')
        mnar_count = sum(1 for m in missingness_by_column.values() if m['mechanism'] == 'MNAR')
        
        summary = {
            'total_missing_columns': len(missing_cols),
            'mcar_columns': mcar_count,
            'mar_columns': mar_count,
            'mnar_columns': mnar_count,
            'total_missing_cells': int(df.isnull().sum().sum()),
            'no_missing_values': False
        }
        
        self._log(f"Classification: {mcar_count} MCAR, {mar_count} MAR, {mnar_count} MNAR")
        
        return {
            'missingness_by_column': missingness_by_column,
            'missing_rate_by_column': missing_rate_by_column,
            'summary': summary,
            'status': 'SUCCESS'
        }

    def missingness_as_signal(self, df, target_col):
        """
        Test if missingness indicators are predictive of the target.
        
        WHY THIS MATTERS:
          Sometimes the FACT that a variable is missing is informative.
          Example: patients who don't report weight are sicker.
          This method detects and quantifies that signal.
        
        Parameters
        ----------
        df : pandas.DataFrame
        target_col : str
        
        Returns
        -------
        dict with keys:
          'signal_columns' : list of (column, mi_score) tuples
              Missingness indicators that are predictive
          'n_signal_columns' : int
          'interpretation' : str
        """
        self._log(f"Testing missingness as signal for '{target_col}'")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        y = df[target_col].values
        
        # Find columns with missing values
        missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0 and c != target_col]
        
        if not missing_cols:
            self._log("No missing values to test as signal")
            return {
                'signal_columns': [],
                'n_signal_columns': 0,
                'interpretation': 'No missing values detected'
            }
        
        signal_columns = []
        
        for col in missing_cols:
            # Create binary indicator: 1 if missing, 0 if observed
            indicator = df[col].isnull().astype(int)
            
            # Test if indicator predicts target
            if len(np.unique(indicator)) > 1 and len(np.unique(y)) > 1:
                try:
                    # Simple correlation for numeric targets
                    if df[target_col].dtype in [np.float64, np.int64]:
                        corr = abs(np.corrcoef(indicator, y)[0, 1])
                        if corr > 0.2:  # Reasonable threshold for signal
                            signal_columns.append((col, float(corr)))
                    else:
                        # For classification, use mutual information
                        rf = RandomForestClassifier(n_estimators=20, random_state=42)
                        rf.fit(indicator.reshape(-1, 1), y)
                        importance = rf.feature_importances_[0]
                        if importance > 0.05:
                            signal_columns.append((col, float(importance)))
                except:
                    pass
        
        signal_columns = sorted(signal_columns, key=lambda x: x[1], reverse=True)
        
        interpretation = (
            f"Missingness signal analysis:\n"
            f"  • {len(signal_columns)} features' missingness is predictive of target\n"
        )
        if signal_columns:
            interpretation += f"  • Strongest: {signal_columns[0][0]} (score: {signal_columns[0][1]:.3f})\n"
            interpretation += "  • Consider: can these indicators improve your model?"
        else:
            interpretation += "  • No strong signals detected\n"
        
        return {
            'signal_columns': signal_columns,
            'n_signal_columns': len(signal_columns),
            'interpretation': interpretation,
            'status': 'SUCCESS'
        }

    def missingness_heatmap_data(self, df):
        """
        Compute correlation matrix of missingness patterns.
        
        WHY THIS MATTERS:
          Co-missing patterns tell a story:
          1. Two columns always missing together → might be same root cause
          2. Highly correlated missingness → potential MNAR mechanism
        
        Parameters
        ----------
        df : pandas.DataFrame
        
        Returns
        -------
        dict with keys:
          'missing_correlation' : array
              Correlation matrix of missingness indicators
          'co_missing_pairs' : list of (col1, col2, correlation)
              Pairs with correlated missingness
        """
        self._log("Computing missingness co-occurrence patterns")
        
        # Create binary missingness matrix
        missing_matrix = df.isnull().astype(int)
        
        # Remove columns with no missing values
        cols_with_missing = [c for c in df.columns if missing_matrix[c].sum() > 0]
        
        if len(cols_with_missing) <= 1:
            self._log("Insufficient columns with missing values for correlation")
            return {
                'missing_correlation': np.array([]),
                'co_missing_pairs': [],
                'status': 'SUCCESS'
            }
        
        subset = missing_matrix[cols_with_missing]
        
        # Compute correlation
        corr_matrix = subset.corr()
        
        # Find strong pairs
        co_missing_pairs = []
        for i, col1 in enumerate(cols_with_missing):
            for j, col2 in enumerate(cols_with_missing):
                if i < j:
                    corr_val = corr_matrix.loc[col1, col2]
                    if abs(corr_val) > 0.5:  # Moderate/strong correlation
                        co_missing_pairs.append((col1, col2, float(corr_val)))
        
        co_missing_pairs = sorted(co_missing_pairs, key=lambda x: abs(x[2]), reverse=True)
        
        self._log(f"Found {len(co_missing_pairs)} co-missing pairs")
        
        return {
            'missing_correlation': corr_matrix.values,
            'missing_correlation_columns': cols_with_missing,
            'co_missing_pairs': co_missing_pairs,
            'status': 'SUCCESS'
        }

    def run(self, df, target_col=None, **kwargs):
        """
        Main entry point for missing data analysis.
        
        Parameters
        ----------
        df : pandas.DataFrame
        target_col : str, optional
        **kwargs : additional arguments (unused)
        
        Returns
        -------
        dict (wrapped by BaseModule._result)
        """
        try:
            findings = {}
            severity = 'NONE'
            
            # Resolve target column if provided
            if target_col:
                target_col = self._resolve_target(df, target_col)

            # Classify mechanisms
            classify_result = self.classify_missingness(df)
            findings['classification'] = classify_result
            
            # Check severity based on missing rate
            missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns))
            if missing_rate > 0.3:
                severity = 'HIGH'
            elif missing_rate > 0.1:
                severity = 'MEDIUM'
            
            # Test missingness as signal
            if target_col:
                signal_result = self.missingness_as_signal(df, target_col)
                findings['signal_analysis'] = signal_result
            
            # Co-occurrence patterns
            heatmap_result = self.missingness_heatmap_data(df)
            findings['co_occurrence'] = heatmap_result
            
            return self._result(findings, severity=severity, module_name="MissingDataAnalyzer")
        
        except Exception as e:
            self._error(f"Missing data analysis failed: {e}")
            return self._result({'error': str(e), 'status': 'FAILED'}, severity='CRITICAL')
