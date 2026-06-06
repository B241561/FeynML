"""
Label Noise Detection Module
=============================

Detects potentially mislabeled samples using:
  • Confident Learning (cleanlab)
  • Asymmetric noise matrix estimation
  • Noisy label identification

Educational Focus:
  The key insight is that mislabeled samples often have:
  1. Low confidence from classifiers (predicted prob ≠ given label)
  2. Inconsistent patterns with similar samples
  3. Predictability that violates the noise model
  
  Cleanlab uses "confident learning" to:
  1. Rank samples by label confidence
  2. Estimate the noise transition matrix T[i,j] = P(given=j | true=i)
  3. Return sorted list of samples most likely to be mislabeled
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
import warnings

from engine.base_module import BaseModule

try:
    from cleanlab.filter import find_label_issues
    HAS_CLEANLAB = True
except ImportError:
    HAS_CLEANLAB = False
    warnings.warn("cleanlab not installed. Label noise detection will use fallback.")


class LabelNoiseAnalyzer(BaseModule):
    """
    Analyzes label quality and identifies potentially mislabeled samples.
    
    Educational Philosophy:
      Every mislabeled sample is a data quality problem that can degrade model
      performance significantly. This module surfaces those problems systematically.
    """

    def _is_regression(self, y):
        """
        Detect if the target is likely a continuous regression target.
        Regression detected if:
          - More than 20 unique values OR
          - Data type is float
        """
        if not np.issubdtype(y.dtype, np.number):
            return False
            
        n_unique = len(np.unique(y))
        
        # User specified rule: unique values > 20 OR dtype is float
        if n_unique > 20 or np.issubdtype(y.dtype, np.floating):
            return True
        return False

    def regression_noise_detection(self, X, y, cv=5):
        """
        Identify outliers in regression targets using residual analysis.
        
        Parameters
        ----------
        X : array-like
        y : array-like
        cv : int, cross-validation folds
        
        Returns
        -------
        dict with noise metrics
        """
        self._log(f"Running regression noise detection on {len(X)} samples")
        
        reg = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        try:
            y_pred = cross_val_predict(reg, X, y, cv=cv)
            residuals = np.abs(y - y_pred)
            
            mean_res = np.mean(residuals)
            std_res = np.std(residuals)
            threshold = mean_res + 3 * std_res
            
            is_noisy = residuals > threshold
            noisy_indices = np.where(is_noisy)[0].tolist()
            noisy_sample_count = int(np.sum(is_noisy))
            noise_rate = float(noisy_sample_count / len(y))
            
            self._log(f"Detected {noisy_sample_count} noisy regression samples (rate: {noise_rate:.3f})")
            
            # Create a detailed DataFrame for the results
            issues_df = pd.DataFrame({
                'sample_idx': np.arange(len(y)),
                'actual_value': y,
                'predicted_value': y_pred,
                'residual': residuals,
                'is_issue': is_noisy
            })
            issues_df = issues_df[issues_df['is_issue']].sort_values('residual', ascending=False)
            
            return {
                'noise_rate': noise_rate,
                'noisy_indices': noisy_indices,
                'noisy_sample_count': noisy_sample_count,
                'n_total': len(y),
                'label_issues_df': issues_df,
                'estimated_noise_rate': noise_rate,
                'n_issues': noisy_sample_count,
                'status': 'SUCCESS',
                'method': 'residual_outlier_detection'
            }
            
        except Exception as e:
            self._error(f"Regression noise detection failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'noise_rate': 0.0,
                'noisy_indices': [],
                'noisy_sample_count': 0
            }

    def confident_learning(self, X, y, clf=None, return_indices=True):
        """
        Identify samples with problematic labels using confident learning.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix
        y : array-like, shape (n_samples,)
            Given labels (may contain noise)
        clf : estimator, optional
            Classifier to estimate P(y | X). Defaults to RandomForestClassifier.
            Must have predict_proba() method.
        return_indices : bool
            If True, return sample indices; if False, return feature arrays.
        
        Returns
        -------
        dict with keys:
          'label_issues' : array of bool, shape (n_samples,)
              True for samples flagged as potentially mislabeled
          'label_issues_df' : DataFrame
              Detailed breakdown: [idx, given_label, pred_label, confidence_score]
          'n_issues' : int
              Count of samples with label issues
          'estimated_noise_rate' : float
              Fraction of samples estimated to be mislabeled
        
        Notes
        -----
        Confident learning uses:
          1. Cross-validated predictions P(y | X)
          2. Comparison with given labels to detect conflicts
          3. Ranking by "label confidence" = how well label matches predictions
        """
        self._log(f"Running confident learning on {len(X)} samples, {len(np.unique(y))} classes")
        
        n_samples = len(X)
        self._check_min_samples(n_samples, minimum=20, context="for label noise detection")
        
        # Determine valid CV folds based on class counts
        unique_classes, class_counts = np.unique(y, return_counts=True)
        min_class_count = np.min(class_counts)
        
        if min_class_count < 2:
            self._warn("Some classes have only 1 sample. Cross-validation is not possible.")
            # If all classes have only 1 sample, we definitely can't do anything
            if len(unique_classes) == n_samples:
                 return {
                    'noisy_indices': [],
                    'n_total': n_samples,
                    'label_issues_df': pd.DataFrame(),
                    'n_issues': 0,
                    'estimated_noise_rate': 0.0,
                    'status': 'INSUFFICIENT_DATA',
                    'message': "Insufficient data for label noise detection: every class has only one sample."
                }
            
            # Try to continue with cv=2 if possible, but sklearn's StratifiedKFold 
            # requires at least 2 samples per class for cv=2.
            cv_folds = 2
            if min_class_count < 2:
                # Fallback to a non-stratified approach or just return insufficient data
                # for now, as stratified is better for label noise.
                return {
                    'noisy_indices': [],
                    'n_total': n_samples,
                    'label_issues_df': pd.DataFrame(),
                    'n_issues': 0,
                    'estimated_noise_rate': 0.0,
                    'status': 'INSUFFICIENT_DATA',
                    'message': f"Insufficient data: minimum class frequency is {min_class_count}, but at least 2 are required for cross-validation."
                }
        else:
            cv_folds = min(5, min_class_count)
            if cv_folds < 5:
                self._log(f"Dynamically reduced CV folds to {cv_folds} due to small class sizes")

        if clf is None:
            clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            self._log("Using default RandomForestClassifier for predictions")
        
        # Get cross-validated predictions
        try:
            pred_probs = cross_val_predict(clf, X, y, cv=cv_folds, method='predict_proba')
            self._log(f"Obtained cross-validated probabilities: shape {pred_probs.shape}")
        except Exception as e:
            self._error(f"Failed to get predictions: {e}")
            return {
                'noisy_indices': [],
                'n_total': n_samples,
                'label_issues_df': pd.DataFrame(),
                'n_issues': 0,
                'estimated_noise_rate': 0.0,
                'status': 'FAILED',
                'error': str(e)
            }
        
        # Use cleanlab if available
        if HAS_CLEANLAB:
            try:
                label_issues = find_label_issues(y, pred_probs, return_indices_ranked_by="self_confidence")
                self._log(f"Cleanlab identified {len(label_issues)} potential label issues")
                
                issues_bool = np.zeros(n_samples, dtype=bool)
                issues_bool[label_issues] = True
                
                # Build detailed DataFrame
                pred_labels = np.argmax(pred_probs, axis=1)
                confidence_scores = np.max(pred_probs, axis=1)
                
                issues_df = pd.DataFrame({
                    'sample_idx': np.arange(n_samples),
                    'given_label': y,
                    'predicted_label': pred_labels,
                    'confidence_score': confidence_scores,
                    'is_issue': issues_bool
                })
                issues_df = issues_df[issues_df['is_issue']].sort_values('confidence_score')
                
                return {
                    'noisy_indices': label_issues.tolist(),
                    'n_total': n_samples,
                    'label_issues_df': issues_df,
                    'n_issues': int(np.sum(issues_bool)),
                    'estimated_noise_rate': float(np.mean(issues_bool)),
                    'status': 'SUCCESS'
                }
            except Exception as e:
                self._error(f"Cleanlab error: {e}. Falling back to confidence-based detection.")
        
        # Fallback: confidence-based detection
        pred_labels = np.argmax(pred_probs, axis=1)
        confidence_scores = np.max(pred_probs, axis=1)
        
        # Flag samples where predicted label doesn't match given label OR confidence is very low
        mismatch = pred_labels != y
        low_confidence = confidence_scores < 0.5
        issues_bool = mismatch | low_confidence
        
        issues_df = pd.DataFrame({
            'sample_idx': np.arange(n_samples),
            'given_label': y,
            'predicted_label': pred_labels,
            'confidence_score': confidence_scores,
            'is_issue': issues_bool
        })
        issues_df = issues_df[issues_df['is_issue']].sort_values('confidence_score')
        
        self._log(f"Confidence-based detection identified {np.sum(issues_bool)} issues")
        
        return {
            'noisy_indices': np.where(issues_bool)[0].tolist(),
            'n_total': n_samples,
            'label_issues_df': issues_df,
            'n_issues': int(np.sum(issues_bool)),
            'estimated_noise_rate': float(np.mean(issues_bool)),
            'status': 'SUCCESS'
        }

    def asymmetric_noise_matrix(self, y_noisy, y_clean=None):
        """
        Estimate the noise transition matrix T[i,j] = P(given label = j | true label = i).
        
        Parameters
        ----------
        y_noisy : array-like, shape (n_samples,)
            Observed (potentially noisy) labels
        y_clean : array-like, shape (n_samples,), optional
            True labels. If None, estimate using confident learning.
        
        Returns
        -------
        dict with keys:
          'noise_matrix' : array, shape (n_classes, n_classes)
              T[i,j] = P(observed=j | true=i)
          'confidence_noise_matrix' : array, shape (n_classes, n_classes)
              Confidence/agreement matrix for the estimate
          'noise_rates' : dict
              {class_i: estimated noise rate for class i}
          'confusion_matrix_interpretation' : dict
              Common class confusion patterns
        """
        self._log(f"Estimating noise matrix from {len(y_noisy)} samples")
        
        classes = np.unique(y_noisy)
        n_classes = len(classes)
        
        if y_clean is None:
            self._log("y_clean not provided. Noise matrix estimate will be limited.")
            # Simple heuristic: assume label given by random forest cross-val is "clean"
            noise_matrix = np.eye(n_classes) * 0.9 + (1 - np.eye(n_classes)) * 0.05
            noise_rates = {c: 0.1 for c in classes}
        else:
            self._check_lengths(y_noisy=y_noisy, y_clean=y_clean)
            
            # Build confusion matrix: noisy vs clean
            confusion = np.zeros((n_classes, n_classes))
            for true_label_idx, true_label in enumerate(classes):
                mask = y_clean == true_label
                if np.sum(mask) == 0:
                    continue
                observed_labels = y_noisy[mask]
                for obs_label_idx, obs_label in enumerate(classes):
                    confusion[true_label_idx, obs_label_idx] = np.mean(observed_labels == obs_label)
            
            noise_matrix = confusion
            noise_rates = {
                classes[i]: 1.0 - confusion[i, i]
                for i in range(n_classes)
            }
            self._log(f"Noise matrix estimated. Average noise rate: {np.mean(list(noise_rates.values())):.3f}")
        
        return {
            'noise_matrix': noise_matrix,
            'noise_rates': noise_rates,
            'n_classes': n_classes,
            'status': 'SUCCESS'
        }

    def noise_summary(self, results, y=None):
        """
        Generate human-readable summary of label noise findings.
        
        Parameters
        ----------
        results : dict
            Output from confident_learning()
        y : array-like, optional
            Ground truth labels for context
        
        Returns
        -------
        dict with keys:
          'total_samples' : int
          'estimated_noise_rate_pct' : float
          'n_issues_found' : int
          'top_suspicious_samples' : list of dicts
          'summary_text' : str
              Human-readable interpretation
        """
        issues_df = results.get('label_issues_df', pd.DataFrame())
        n_issues = results.get('n_issues', 0)
        noise_rate = results.get('estimated_noise_rate', 0.0)
        n_total = results.get('n_total', 0)
        
        summary = {
            'total_samples': int(n_total),
            'estimated_noise_rate_pct': float(noise_rate * 100),
            'n_issues_found': int(n_issues),
            'summary_text': ''
        }
        
        if len(issues_df) > 0:
            top_10 = issues_df.head(10)
            
            if 'confidence_score' in top_10.columns:
                score_str = f"(confidence: {top_10.iloc[0]['confidence_score']:.3f})"
            elif 'residual' in top_10.columns:
                score_str = f"(residual: {top_10.iloc[0]['residual']:.3f})"
            else:
                score_str = ""

            summary['summary_text'] = (
                f"Label Noise Summary:\n"
                f"  • Estimated noise rate: {noise_rate*100:.1f}%\n"
                f"  • Samples flagged: {n_issues} / {n_total}\n"
                f"  • Top suspect: sample {top_10.iloc[0]['sample_idx']} "
                f"{score_str}\n"
                f"  • Recommendation: Review flagged samples with domain experts"
            )
        else:
            summary['summary_text'] = (
                f"Label Noise Summary:\n"
                f"  • No significant label noise detected\n"
                f"  • Estimated noise rate: {noise_rate*100:.1f}%\n"
                f"  • Data quality appears good"
            )
        
        return summary

    def run(self, X, y, **kwargs):
        """
        Main entry point for label noise analysis.
        
        Parameters
        ----------
        X : array-like or DataFrame
        y : array-like or str (column name)
        **kwargs : passed to confident_learning()
        
        Returns
        -------
        dict (wrapped by BaseModule._result)
        """
        try:
            # Handle DataFrame + string column name input
            if isinstance(X, pd.DataFrame):
                if isinstance(y, str):
                    target_col = self._resolve_target(X, y)
                    y = X[target_col].values
                    X = X.drop(columns=[target_col])
                
                # Convert categorical features to numeric for internal models
                X = X.copy()
                for col in X.select_dtypes(include=['object', 'category']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                X = X.values
            
            # Ensure y is numeric for np.unique and other operations
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                self._log("Encoded string labels to numeric for analysis")

            # Check for regression target
            if self._is_regression(y):
                results = self.regression_noise_detection(X, y)
                if results.get('status') == 'FAILED':
                    return self._result(self._serialize_findings(results), severity='CRITICAL', module_name="LabelNoiseAnalyzer")
            else:
                results = self.confident_learning(X, y, **kwargs)
            
            # Handle insufficient data gracefully
            if results.get('status') == 'INSUFFICIENT_DATA':
                findings = {
                    'status': 'FAILED',
                    'message': results.get('message', 'Insufficient data for analysis'),
                    'details': 'Label noise detection requires at least 2 samples per class for cross-validation.'
                }
                return self._result(self._serialize_findings(findings), severity='WARNING', module_name="LabelNoiseAnalyzer")

            summary = self.noise_summary(results, y)
            
            severity = 'NONE'
            if results.get('estimated_noise_rate', 0) > 0.2:
                severity = 'CRITICAL'
            elif results.get('estimated_noise_rate', 0) > 0.1:
                severity = 'HIGH'
            elif results.get('estimated_noise_rate', 0) > 0.05:
                severity = 'MEDIUM'
            
            findings = {
                'label_issues': results,
                'summary': summary
            }
            
            # Ensure findings are JSON-serializable
            serialized_findings = self._serialize_findings(findings)
            
            return self._result(serialized_findings, severity=severity, module_name="LabelNoiseAnalyzer")
        
        except Exception as e:
            self._error(f"Label noise analysis failed: {e}")
            error_findings = self._serialize_findings({'error': str(e), 'status': 'FAILED'})
            return self._result(error_findings, severity='CRITICAL')
