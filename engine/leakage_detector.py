"""
Feature Leakage Detection Module
=================================

Detects data leakage in features:
  • Target leakage: features that are suspiciously predictive of target
  • Temporal leakage: features that use future information
  • Permutation importance anomalies: features with unnaturally high importance

Educational Focus:
  Data leakage is one of the most insidious ML bugs:
  1. Models perform well in development but fail in production
  2. The "leaky" feature is often harmless-looking
  3. Regular train/test split won't catch it—need domain knowledge
  
  This module uses statistical methods to surface suspicious features
  that warrant human review.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr, spearmanr
from scipy.stats import entropy as scipy_entropy
import warnings

from engine.base_module import BaseModule

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    HAS_MI = True
except ImportError:
    HAS_MI = False


class LeakageDetector(BaseModule):
    """
    Detects various forms of data leakage in feature sets.
    
    Educational Philosophy:
      Leakage detection requires both statistical rigor and domain intuition.
      This module flags statistically suspicious features, but human review is essential.
    """

    def detect_target_leakage(self, df, target_col, threshold=0.95, n_jobs=-1):
        """
        Identify features that are suspiciously predictive of the target.
        
        WHY THIS MATTERS:
          Features with very high correlation or MI with target might be:
          1. Direct leakage: target is computed from these features
          2. Temporal leakage: these features come from the future
          3. Label leakage: these features contain information only in training set
        
        Parameters
        ----------
        df : pandas.DataFrame
        target_col : str
            Target column name
        threshold : float
            Flag features with MI/correlation > threshold * MI(target, target)
            Default 0.95 means flag features with MI > 95% of max possible
        n_jobs : int
            Jobs for parallel processing
        
        Returns
        -------
        dict with keys:
          'leakage_suspects' : list of dicts
              Each dict: {feature, mutual_info, correlation, leakage_risk}
          'n_suspects' : int
          'most_leaky_feature' : str or None
          'threshold_used' : float
          'detailed_findings' : DataFrame
        """
        self._log(f"Detecting target leakage in {len(df)} samples, {len(df.columns)} features")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Prepare target
        y = df[target_col].values
        
        # Infer if classification or regression
        n_unique = len(np.unique(y))
        is_classification = n_unique <= 20  # Heuristic
        
        # Get feature columns (exclude target) and filter for numeric/categorical only
        # Skip datetime, timedelta, and other complex types
        feature_cols = []
        for c in df.columns:
            if c != target_col:
                dtype = df[c].dtype
                # Include numeric types and object (which we'll encode)
                if dtype in [np.float64, np.int64, np.float32, np.int32, 'float', 'int', 'object']:
                    feature_cols.append(c)
        
        X = df[feature_cols].copy()
        
        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Remove NaN
        X = X.fillna(X.mean(numeric_only=True))
        
        suspects = []
        
        # Method 1: Mutual Information
        if HAS_MI:
            try:
                if is_classification:
                    mi_scores = mutual_info_classif(X, y, random_state=42, n_jobs=n_jobs)
                else:
                    mi_scores = mutual_info_regression(X, y, random_state=42, n_jobs=n_jobs)
                
                max_mi = np.max(mi_scores) if len(mi_scores) > 0 else 1.0
                threshold_val = threshold * max_mi
                
                self._log(f"Max MI: {max_mi:.4f}, threshold: {threshold_val:.4f}")
                
                for feature, mi in zip(feature_cols, mi_scores):
                    if mi > threshold_val:
                        suspects.append({
                            'feature': feature,
                            'mutual_info': float(mi),
                            'correlation': np.nan,
                            'leakage_risk': 'CRITICAL' if mi > 0.95 * max_mi else 'HIGH'
                        })
            except Exception as e:
                self._warn(f"MI calculation failed: {e}")
        
        # Method 2: Correlation (for numeric features)
        try:
            for col in feature_cols:
                if X[col].dtype in [np.float64, np.int64]:
                    try:
                        corr, _ = pearsonr(X[col].dropna(), y[:len(X[col].dropna())])
                        corr = abs(corr)
                        
                        if corr > 0.9:  # Very high correlation
                            existing = [s for s in suspects if s['feature'] == col]
                            if existing:
                                existing[0]['correlation'] = float(corr)
                                existing[0]['leakage_risk'] = 'CRITICAL'
                            else:
                                suspects.append({
                                    'feature': col,
                                    'mutual_info': np.nan,
                                    'correlation': float(corr),
                                    'leakage_risk': 'CRITICAL' if corr > 0.95 else 'HIGH'
                                })
                    except (ValueError, RuntimeWarning):
                        pass
        except Exception as e:
            self._warn(f"Correlation calculation failed: {e}")
        
        # Remove duplicates and sort
        suspect_dict = {s['feature']: s for s in suspects}
        suspects = list(suspect_dict.values())
        suspects = sorted(suspects, key=lambda x: x['mutual_info'] if not np.isnan(x['mutual_info']) else x['correlation'], reverse=True)
        
        findings = {
            'leakage_suspects': suspects,
            'n_suspects': len(suspects),
            'most_leaky_feature': suspects[0]['feature'] if suspects else None,
            'threshold_used': float(threshold),
            'status': 'SUCCESS'
        }
        
        self._log(f"Found {len(suspects)} leakage suspects")
        
        return findings

    def detect_temporal_leakage(self, df, target_col, date_col, window_size=10):
        """
        Detect temporal leakage: features that use future information.
        
        WHY THIS MATTERS:
          Time-series models often suffer from temporal leakage:
          1. Features contain future information that wouldn't be available at prediction time
          2. Common example: 30-day rolling average computed on all data, not forward-looking
        
        Parameters
        ----------
        df : pandas.DataFrame
            Must be sorted by date_col
        target_col : str
        date_col : str
            Datetime column for temporal ordering
        window_size : int
            Rows to check for correlation stability
        
        Returns
        -------
        dict with keys:
          'temporal_leakage_detected' : bool
          'suspicious_features' : list
          'evidence' : str
              Explanation of findings
        """
        self._log(f"Detecting temporal leakage using {date_col}")
        
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        # Ensure sorted by date
        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        
        feature_cols = [c for c in df.columns if c not in [target_col, date_col]]
        y = df_sorted[target_col].values
        
        suspicious = []
        evidence_list = []
        
        # Check each feature
        for col in feature_cols:
            if df_sorted[col].dtype not in [np.float64, np.int64]:
                continue
            
            x = df_sorted[col].fillna(df_sorted[col].mean()).values
            
            try:
                # Compute correlation on first half vs second half
                mid = len(x) // 2
                if mid < window_size:
                    continue
                
                corr_first = abs(np.corrcoef(x[:mid], y[:mid])[0, 1])
                corr_second = abs(np.corrcoef(x[mid:], y[mid:])[0, 1])
                
                # Spike in second correlation suggests future information
                if corr_second > 2 * corr_first and corr_second > 0.7:
                    suspicious.append(col)
                    evidence_list.append(
                        f"  • {col}: correlation jumps from {corr_first:.3f} to {corr_second:.3f}"
                    )
            except (ValueError, RuntimeWarning):
                pass
        
        findings = {
            'temporal_leakage_detected': len(suspicious) > 0,
            'suspicious_features': suspicious,
            'evidence': '\n'.join(evidence_list) if evidence_list else 'No temporal leakage patterns detected',
            'status': 'SUCCESS'
        }
        
        self._log(f"Temporal leakage check: {len(suspicious)} suspicious features")
        
        return findings

    def permutation_importance_spike(self, df, target_col, model=None, n_repeats=10):
        """
        Detect features with unusually high permutation importance.
        
        WHY THIS MATTERS:
          Permutation importance measures "how much worse does the model perform
          if we scramble this feature?" High importance suggests the feature
          is crucial—which is normal. But ANOMALOUSLY high importance can indicate:
          1. Leakage: feature contains target information
          2. Unbalanced importance: one feature carries too much signal
        
        Parameters
        ----------
        df : pandas.DataFrame
        target_col : str
        model : estimator, optional
            Fitted model. If None, trains a RandomForest.
        n_repeats : int
            Permutation repetitions
        
        Returns
        -------
        dict with keys:
          'importance_scores' : DataFrame
              Columns: feature, importance, relative_importance
          'spike_flags' : list
              Features with importance > 3× median
          'detailed_findings' : dict
        """
        self._log(f"Computing permutation importance for {len(df)} samples")
        
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found")
        
        feature_cols = [c for c in df.columns if c != target_col]
        
        # Prepare data - only select numeric and categorical columns that can be encoded
        X = df[feature_cols].copy()
        y = df[target_col].values
        
        # Remove datetime and other non-numeric columns
        numeric_cols = []
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64, np.float32, np.int32, 'float', 'int']:
                numeric_cols.append(col)
            elif X[col].dtype == 'object':
                numeric_cols.append(col)  # Will be encoded below
            # Skip datetime and other complex types
        
        X = X[numeric_cols].copy()
        feature_cols = numeric_cols
        
        # Encode categoricals
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Fill NaN
        X = X.fillna(X.mean(numeric_only=True))
        
        # Train model if not provided
        if model is None:
            n_unique = len(np.unique(y))
            is_classification = n_unique <= 20
            
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            model.fit(X, y)
            self._log(f"Trained {model.__class__.__name__} model")
        
        # Compute permutation importance
        try:
            perm_importance = permutation_importance(
                model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=-1
            )
            
            importances_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': perm_importance.importances_mean,
                'std': perm_importance.importances_std
            }).sort_values('importance', ascending=False)
            
            importances_df['relative_importance'] = (
                importances_df['importance'] / importances_df['importance'].sum()
            )
            
            # Detect spikes: importance > 3× median
            median_importance = importances_df['importance'].median()
            spike_threshold = 3 * median_importance
            spike_flags = importances_df[
                importances_df['importance'] > spike_threshold
            ]['feature'].tolist()
            
            findings = {
                'importance_scores': importances_df.to_dict('records'),
                'spike_flags': spike_flags,
                'median_importance': float(median_importance),
                'spike_threshold': float(spike_threshold),
                'n_spikes': len(spike_flags),
                'status': 'SUCCESS'
            }
            
            self._log(f"Permutation importance: {len(spike_flags)} potential anomalies")
            
            return findings
        
        except Exception as e:
            self._error(f"Permutation importance failed: {e}")
            return {
                'importance_scores': [],
                'spike_flags': [],
                'n_spikes': 0,
                'status': 'FAILED',
                'error': str(e)
            }

    def run(self, df, target_col, date_col=None, **kwargs):
        """
        Main entry point for leakage detection.
        
        Parameters
        ----------
        df : pandas.DataFrame
        target_col : str
        date_col : str, optional
            If provided, run temporal leakage check
        **kwargs : passed to detection methods
        
        Returns
        -------
        dict (wrapped by BaseModule._result)
        """
        try:
            findings = {}
            severity = 'NONE'
            
            # Resolve target column
            target_col = self._resolve_target(df, target_col)
            
            # Target leakage
            target_leakage = self.detect_target_leakage(df, target_col, **kwargs)
            findings['target_leakage'] = target_leakage
            
            if target_leakage.get('n_suspects', 0) > 0:
                severity = 'HIGH'
            
            # Temporal leakage (if date_col provided)
            if date_col:
                temporal_leakage = self.detect_temporal_leakage(df, target_col, date_col)
                findings['temporal_leakage'] = temporal_leakage
                
                if temporal_leakage.get('temporal_leakage_detected', False):
                    severity = 'CRITICAL'
            
            # Permutation importance
            perm_importance = self.permutation_importance_spike(df, target_col)
            findings['permutation_importance'] = perm_importance
            
            if perm_importance.get('n_spikes', 0) > 0:
                severity = 'MEDIUM' if severity == 'NONE' else severity
            
            return self._result(findings, severity=severity, module_name="LeakageDetector")
        
        except Exception as e:
            self._error(f"Leakage detection failed: {e}")
            return self._result({'error': str(e), 'status': 'FAILED'}, severity='CRITICAL')
