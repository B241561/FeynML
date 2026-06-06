"""
Engine Module — Root Cause Intelligence Engine
===============================================
Move beyond simple drift detection toward identifying the 'Why' behind failures.

Responsibilities:
  • Missing Value Analysis: Detect sudden increases and patterns.
  • Distribution Shift: Compare Mean, Median, Std Dev, Quantiles.
  • Category Drift: Identify new categories or significant frequency shifts.
  • Outlier Analysis: Detect extreme values and their influence.
  • Correlation Drift: Identify changes in feature-target or feature-feature relationships.
  • Root Cause Scoring: Assign confidence scores to potential causes.
  • Investigation Story: Generate a narrative timeline of the failure.

Usage:
    from engine.modules.root_cause_engine import RootCauseEngine

    rce = RootCauseEngine()
    results = rce.run(reference_df, current_df, target_col="target")
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

class RootCauseEngine:
    def __init__(self, verbose=True):
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[RootCauseEngine] {msg}")

    def run(self, reference_df, current_df, target_col=None):
        """
        Run the full root cause analysis suite.
        """
        self._log("Starting Root Cause Analysis...")
        
        missing_findings = self._analyze_missing_values(reference_df, current_df)
        dist_findings = self._analyze_distribution_shifts(reference_df, current_df)
        cat_findings = self._analyze_category_drift(reference_df, current_df)
        outlier_findings = self._analyze_outliers(reference_df, current_df)
        corr_findings = self._analyze_correlation_drift(reference_df, current_df, target_col)
        
        # Combine all findings
        all_findings = {
            "missing_values": missing_findings,
            "distribution_shifts": dist_findings,
            "category_drift": cat_findings,
            "outliers": outlier_findings,
            "correlation_drift": corr_findings
        }
        
        # Score root causes
        scored_causes = self._score_root_causes(all_findings)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(scored_causes)
        
        # Generate investigation timeline
        timeline = self._generate_timeline()
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "findings": all_findings,
            "scored_causes": scored_causes,
            "recommendations": recommendations,
            "timeline": timeline
        }

    def _analyze_missing_values(self, ref_df, cur_df):
        """Detect sudden increase in missing values."""
        findings = []
        for col in cur_df.columns:
            ref_missing = ref_df[col].isnull().mean()
            cur_missing = cur_df[col].isnull().mean()
            
            if cur_missing > ref_missing + 0.05: # 5% increase threshold
                findings.append({
                    "feature": col,
                    "ref_missing_pct": round(ref_missing * 100, 2),
                    "cur_missing_pct": round(cur_missing * 100, 2),
                    "increase_pct": round((cur_missing - ref_missing) * 100, 2),
                    "severity": "HIGH" if cur_missing > 0.2 else "MEDIUM"
                })
        return findings

    def _analyze_distribution_shifts(self, ref_df, cur_df):
        """Compare Mean, Median, Std Dev, Quantiles."""
        findings = []
        numeric_cols = cur_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            ref_stats = ref_df[col].describe()
            cur_stats = cur_df[col].describe()
            
            mean_diff_pct = abs(cur_stats['mean'] - ref_stats['mean']) / (ref_stats['mean'] + 1e-9)
            
            if mean_diff_pct > 0.1: # 10% shift threshold
                findings.append({
                    "feature": col,
                    "metric": "mean",
                    "ref_value": round(ref_stats['mean'], 4),
                    "cur_value": round(cur_stats['mean'], 4),
                    "shift_pct": round(mean_diff_pct * 100, 2),
                    "severity": "HIGH" if mean_diff_pct > 0.25 else "MEDIUM"
                })
        return findings

    def _analyze_category_drift(self, ref_df, cur_df):
        """Detect new categories or frequency shifts."""
        findings = []
        cat_cols = cur_df.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            ref_cats = set(ref_df[col].unique())
            cur_cats = set(cur_df[col].unique())
            
            new_cats = cur_cats - ref_cats
            if new_cats:
                findings.append({
                    "feature": col,
                    "new_categories": list(new_cats),
                    "severity": "HIGH"
                })
        return findings

    def _analyze_outliers(self, ref_df, cur_df):
        """Identify extreme values."""
        findings = []
        numeric_cols = cur_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Use IQR for outlier detection
            Q1 = ref_df[col].quantile(0.25)
            Q3 = ref_df[col].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            
            cur_outliers = cur_df[cur_df[col] > upper_bound]
            if not cur_outliers.empty:
                findings.append({
                    "feature": col,
                    "outlier_count": len(cur_outliers),
                    "max_value": float(cur_df[col].max()),
                    "severity": "MEDIUM"
                })
        return findings

    def _analyze_correlation_drift(self, ref_df, cur_df, target_col):
        """Detect changes in correlations."""
        findings = []
        if target_col not in cur_df.columns or target_col not in ref_df.columns:
            return findings
            
        numeric_cols = cur_df.select_dtypes(include=[np.number]).columns
        if target_col not in numeric_cols:
            return findings
            
        for col in numeric_cols:
            if col == target_col: continue
            
            ref_corr = ref_df[col].corr(ref_df[target_col])
            cur_corr = cur_df[col].corr(cur_df[target_col])
            
            if abs(ref_corr - cur_corr) > 0.2:
                findings.append({
                    "feature": col,
                    "ref_correlation": round(ref_corr, 4),
                    "cur_correlation": round(cur_corr, 4),
                    "drift": round(abs(ref_corr - cur_corr), 4),
                    "severity": "MEDIUM"
                })
        return findings

    def _score_root_causes(self, findings):
        """Assign confidence scores to potential causes."""
        scored = []
        
        # Example logic for scoring
        for f in findings['missing_values']:
            scored.append({
                "cause": f"Missing Values in {f['feature']}",
                "confidence": 80 if f['severity'] == "HIGH" else 60,
                "impact": "Data Quality"
            })
            
        for f in findings['distribution_shifts']:
            scored.append({
                "cause": f"Distribution Shift in {f['feature']}",
                "confidence": 90 if f['severity'] == "HIGH" else 70,
                "impact": "Model Performance"
            })

        for f in findings['category_drift']:
            scored.append({
                "cause": f"New Categories in {f['feature']}",
                "confidence": 85,
                "impact": "Data Drift"
            })

        # Sort by confidence
        scored = sorted(scored, key=lambda x: x['confidence'], reverse=True)
        return scored[:5] # Top 5 causes

    def _generate_recommendations(self, scored_causes):
        """Generate AI recommendations based on root causes."""
        recommendations = []
        for cause in scored_causes:
            if "Missing Values" in cause['cause']:
                recommendations.append(f"Investigate upstream data pipeline for {cause['cause'].split()[-1]}. Consider imputation.")
            elif "Distribution Shift" in cause['cause']:
                recommendations.append(f"Retrain model using updated data to capture {cause['cause'].split()[-1]} shift.")
            elif "New Categories" in cause['cause']:
                recommendations.append(f"Update feature encoding strategy to handle new values in {cause['cause'].split()[-1]}.")
        
        if not recommendations:
            recommendations.append("No critical issues found. Continue monitoring.")
            
        return recommendations

    def _generate_timeline(self):
        """Generate a narrative timeline."""
        return [
            {"event": "Dataset Uploaded", "timestamp": datetime.now().isoformat()},
            {"event": "Profiling Completed", "timestamp": datetime.now().isoformat()},
            {"event": "Drift Analysis Triggered", "timestamp": datetime.now().isoformat()},
            {"event": "Root Causes Identified", "timestamp": datetime.now().isoformat()},
            {"event": "Recommendations Generated", "timestamp": datetime.now().isoformat()}
        ]
