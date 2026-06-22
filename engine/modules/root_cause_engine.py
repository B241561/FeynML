"""
Engine Module — Auto Root Cause Engine
=======================================
Production wrapper for automated root cause analysis.

Collects findings from:
  • Drift Engine
  • Slice Analysis Engine
  • Calibration Engine
  • Data Quality Checks
  • Explainability Engine (if available)

Analyzes all findings, ranks likely root causes, and generates:
  • Risk Level
  • Confidence Score
  • Root Cause Ranking
  • Recommended Actions

Returns structured JSON for integration with Report Engine.

Usage:
    from engine.modules.root_cause_engine import AutoRootCauseEngine

    arce = AutoRootCauseEngine()
    results = arce.run(
        drift_report=drift_result,
        slice_report=slice_result,
        calibration_report=calib_result,
        data_quality_report=quality_result
    )
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json

try:
    from .investigation import Investigation, RootCause
except ImportError:
    from investigation import Investigation, RootCause


class AutoRootCauseEngine:
    """
    Auto Root Cause Engine for ML Failure Investigation.
    
    Aggregates findings from multiple analysis engines and identifies
    the most likely root causes of model degradation.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the Auto Root Cause Engine.
        
        Args:
            verbose: Enable logging
        """
        self.verbose = verbose
        self._scoring_rules = self._init_scoring_rules()
        self._excluded_features = set()
        self._audit_log = []

    def _log(self, msg: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[AutoRootCauseEngine] {msg}")

    def _is_identifier_column(self, feature_name: str) -> bool:
        """
        Detect if a column is an identifier column that should be excluded from analysis.

        Args:
            feature_name: Name of the feature column

        Returns:
            True if the column is an identifier, False otherwise
        """
        if not feature_name:
            return False

        feature_lower = feature_name.lower()

        # Direct matches
        if feature_lower in ["id", "studentid", "userid", "customerid", "recordid"]:
            return True

        # Pattern matches: *_id
        if feature_lower.endswith("_id"):
            return True

        # Pattern matches: id_*
        if feature_lower.startswith("id_"):
            return True

        return False
    
    def _init_scoring_rules(self) -> Dict[str, Dict]:
        """
        Initialize scoring rules for different failure types.
        
        Priority order (highest to lowest):
        1. Leakage (CRITICAL)
        2. Label Noise
        3. Calibration
        4. Drift
        
        Returns:
            Dictionary of scoring rules
        """
        return {
            "leakage": {
                "leakage_confidence_gt_050": {"score": 90, "severity": "CRITICAL"},
                "leakage_confidence_gt_070": {"score": 95, "severity": "CRITICAL"},
                "leakage_confidence_gt_090": {"score": 100, "severity": "CRITICAL"},
            },
            "feature_drift": {
                "psi_gt_025": {"score": 50, "severity": "HIGH"},
                "psi_gt_050": {"score": 75, "severity": "CRITICAL"},
                "ks_gt_010": {"score": 30, "severity": "MEDIUM"},
                "ks_gt_020": {"score": 50, "severity": "HIGH"},
            },
            "slice_degradation": {
                "accuracy_drop_gt_10": {"score": 40, "severity": "MEDIUM"},
                "accuracy_drop_gt_20": {"score": 60, "severity": "HIGH"},
                "effect_size_gt_05": {"score": 45, "severity": "MEDIUM"},
                "effect_size_gt_08": {"score": 65, "severity": "HIGH"},
            },
            "calibration": {
                "ece_increase_gt_005": {"score": 30, "severity": "MEDIUM"},
                "ece_increase_gt_010": {"score": 50, "severity": "HIGH"},
                "brier_increase_gt_010": {"score": 35, "severity": "MEDIUM"},
            },
            "missing_values": {
                "missing_rate_gt_10": {"score": 25, "severity": "MEDIUM"},
                "missing_rate_gt_20": {"score": 40, "severity": "HIGH"},
            },
            "outliers": {
                "outlier_increase_gt_20": {"score": 20, "severity": "MEDIUM"},
                "outlier_increase_gt_50": {"score": 35, "severity": "HIGH"},
            },
            "importance_drift": {
                "importance_change_gt_010": {"score": 30, "severity": "MEDIUM"},
                "importance_change_gt_020": {"score": 50, "severity": "HIGH"},
                "importance_change_gt_030": {"score": 70, "severity": "CRITICAL"},
            },
        }
    
    def run(self, 
            drift_report: Optional[Dict] = None,
            slice_report: Optional[Dict] = None,
            calibration_report: Optional[Dict] = None,
            data_quality_report: Optional[Dict] = None,
            explainability_report: Optional[Dict] = None,
            training_importance: Optional[Dict] = None,
            production_importance: Optional[Dict] = None,
            leakage_report: Optional[Dict] = None) -> Dict:
        """
        Run the auto root cause analysis.
        
        Args:
            drift_report: Output from DriftEngine
            slice_report: Output from SlicerEngine
            calibration_report: Output from CalibrationEngine
            data_quality_report: Output from data quality checks
            explainability_report: Output from ExplainabilityEngine
            training_importance: Training SHAP importance (dict: feature -> importance)
            production_importance: Production SHAP importance (dict: feature -> importance)
            leakage_report: Output from LeakageEngine
        
        Returns:
            Dictionary with health_status, confidence, root_causes, recommended_actions
        """
        self._log("Starting Auto Root Cause Analysis...")
        
        # Collect evidence from all sources
        evidence = self._collect_evidence(
            drift_report, slice_report, calibration_report,
            data_quality_report, explainability_report, leakage_report
        )
        
        # Apply conflict resolution: leakage supersedes drift
        evidence = self._resolve_conflicts(evidence)
        
        # Detect SHAP importance drift if both training and production importance provided
        if training_importance and production_importance:
            importance_drift = self._detect_importance_drift(training_importance, production_importance)
            evidence["importance_drift"] = importance_drift
        
        # Score potential root causes
        scored_causes = self._score_root_causes(evidence)
        
        # Calculate overall health status and confidence
        health_status, confidence = self._calculate_health_status(scored_causes)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(scored_causes, evidence.get("leakage", []))
        
        # Build Investigation object
        investigation = Investigation(
            health_status=health_status,
            confidence=confidence,
            evidence=[self._summarize_evidence(evidence)],
            recommendations=recommendations,
            metadata={
                "evidence_summary": self._summarize_evidence(evidence),
                "excluded_features": list(self._excluded_features),
                "audit_log": self._audit_log
            }
        )
        
        # Add root causes to investigation
        for cause in scored_causes:
            investigation.add_root_cause(
                cause=cause["cause"],
                score=cause["score"],
                severity=cause["severity"],
                evidence=cause["evidence"],
                category=cause["category"],
                source_modules=cause.get("source_modules", [])
            )
        
        self._log(f"Analysis complete. Health: {health_status}, Confidence: {confidence}")
        self._log(f"Excluded {len(self._excluded_features)} identifier columns: {list(self._excluded_features)}")
        
        result = investigation.to_dict()
        result["excluded_features"] = list(self._excluded_features)
        result["audit_log"] = self._audit_log
        
        return result
    
    def _collect_evidence(self,
                        drift_report: Optional[Dict],
                        slice_report: Optional[Dict],
                        calibration_report: Optional[Dict],
                        data_quality_report: Optional[Dict],
                        explainability_report: Optional[Dict],
                        leakage_report: Optional[Dict]) -> Dict:
        """
        Collect evidence from all analysis engines.
        
        Returns:
            Dictionary of evidence by category
        """
        evidence = {
            "feature_drift": [],
            "slice_degradation": [],
            "calibration": [],
            "missing_values": [],
            "outliers": [],
            "explainability": [],
            "importance_drift": [],
            "leakage": []
        }
        
        # Process drift report with error handling
        try:
            if drift_report and isinstance(drift_report, dict):
                findings = drift_report.get("findings", {})
                if findings and isinstance(findings, dict):
                    per_feature = findings.get("per_feature", [])
                    if per_feature and isinstance(per_feature, list):
                        for feat in per_feature:
                            if isinstance(feat, dict) and feat.get("status") in ["DRIFT", "WARN"]:
                                feature_name = feat.get("feature", "unknown")
                                
                                # Skip identifier columns
                                if self._is_identifier_column(feature_name):
                                    if feature_name not in self._excluded_features:
                                        self._excluded_features.add(feature_name)
                                        self._audit_log.append(f"[Feature Excluded] {feature_name} (Identifier Column)")
                                        self._log(f"Excluding identifier column from root cause: {feature_name}")
                                    continue
                                
                                psi_val = feat.get("psi", 0)
                                ks_val = feat.get("ks_stat", 0)
                                evidence["feature_drift"].append({
                                    "feature": feature_name,
                                    "psi": psi_val,
                                    "ks_stat": ks_val,
                                    "status": feat.get("status"),
                                    "evidence": f"PSI={psi_val:.3f}, KS={ks_val:.3f}"
                                })
        except Exception as e:
            self._log(f"Warning: Error processing drift report: {e}")
        
        # Process slice report with error handling
        try:
            if slice_report and isinstance(slice_report, dict):
                findings = slice_report.get("findings", {})
                if findings and isinstance(findings, dict):
                    slices = findings.get("slices", [])
                    if slices and isinstance(slices, list):
                        for sl in slices:
                            if isinstance(sl, dict):
                                effect_size = sl.get("effect_size", 0)
                                if effect_size > 0.2:
                                    slice_loss = sl.get("slice_loss", 0)
                                    overall_loss = findings.get("overall_loss", 0)
                                    loss_gap = slice_loss - overall_loss
                                    evidence["slice_degradation"].append({
                                        "slice": sl.get("description", "unknown"),
                                        "effect_size": effect_size,
                                        "slice_loss": slice_loss,
                                        "loss_gap": loss_gap,
                                        "evidence": f"Effect size={effect_size:.2f}, loss gap={loss_gap:.3f}"
                                    })
        except Exception as e:
            self._log(f"Warning: Error processing slice report: {e}")
        
        # Process calibration report with error handling
        try:
            if calibration_report and isinstance(calibration_report, dict):
                ece = calibration_report.get("ece", 0)
                brier = calibration_report.get("brier_score", 0)
                if ece > 0.05:
                    evidence["calibration"].append({
                        "ece": ece,
                        "brier_score": brier,
                        "evidence": f"ECE={ece:.4f}, Brier={brier:.4f}"
                    })
        except Exception as e:
            self._log(f"Warning: Error processing calibration report: {e}")
        
        # Process data quality report with error handling
        try:
            if data_quality_report and isinstance(data_quality_report, dict):
                checks = data_quality_report.get("checks", [])
                if checks and isinstance(checks, list):
                    for check in checks:
                        if isinstance(check, dict):
                            check_type = check.get("check", "")
                            if "missing" in str(check_type).lower():
                                missing_rate = check.get("missing_rate", 0)
                                if missing_rate > 0.1:
                                    evidence["missing_values"].append({
                                        "feature": check.get("feature", "overall"),
                                        "missing_rate": missing_rate,
                                        "evidence": f"Missing rate={missing_rate:.2%}"
                                    })
                            elif "outlier" in str(check_type).lower():
                                outlier_rate = check.get("outlier_rate", 0)
                                if outlier_rate > 0.2:
                                    evidence["outliers"].append({
                                        "feature": check.get("feature", "overall"),
                                        "outlier_rate": outlier_rate,
                                        "evidence": f"Outlier rate={outlier_rate:.2%}"
                                    })
        except Exception as e:
            self._log(f"Warning: Error processing data quality report: {e}")
        
        # Process explainability report with error handling
        try:
            if explainability_report and isinstance(explainability_report, dict):
                findings = explainability_report.get("findings", {})
                if findings:
                    evidence["explainability"].append({
                        "summary": "Explainability analysis available",
                        "evidence": "SHAP/LIME analysis completed"
                    })
        except Exception as e:
            self._log(f"Warning: Error processing explainability report: {e}")
        
        # Process leakage report with error handling
        try:
            if leakage_report and isinstance(leakage_report, dict):
                findings = leakage_report.get("findings", {})
                if findings and isinstance(findings, dict):
                    suspects = findings.get("suspects", [])
                    if suspects and isinstance(suspects, list):
                        for suspect in suspects:
                            if isinstance(suspect, dict):
                                feature_name = suspect.get("feature", "unknown")
                                leakage_confidence = suspect.get("leakage_confidence", 0)
                                
                                # Skip identifier columns
                                if self._is_identifier_column(feature_name):
                                    if feature_name not in self._excluded_features:
                                        self._excluded_features.add(feature_name)
                                        self._audit_log.append(f"[Feature Excluded] {feature_name} (Identifier Column)")
                                        self._log(f"Excluding identifier column from leakage: {feature_name}")
                                    continue
                                
                                # Add to leakage evidence
                                evidence["leakage"].append({
                                    "feature": feature_name,
                                    "leakage_confidence": leakage_confidence,
                                    "evidence": f"Leakage confidence={leakage_confidence:.3f}",
                                    "category": "target_leakage"
                                })
                                self._audit_log.append(f"[Leakage Detected] {feature_name} (Confidence: {leakage_confidence:.3f})")
        except Exception as e:
            self._log(f"Warning: Error processing leakage report: {e}")
        
        return evidence
    
    def _resolve_conflicts(self, evidence: Dict) -> Dict:
        """
        Resolve conflicts between different issue types.
        
        Priority order:
        1. Leakage (highest priority)
        2. Label Noise
        3. Calibration
        4. Drift (lowest priority)
        
        If a feature appears as leakage, remove it from drift evidence.
        
        Args:
            evidence: Dictionary of evidence by category
        
        Returns:
            Evidence dictionary with conflicts resolved
        """
        # Get features that have leakage
        leakage_features = set()
        for leak in evidence.get("leakage", []):
            if isinstance(leak, dict):
                feature_name = leak.get("feature", "")
                if feature_name:
                    leakage_features.add(feature_name)
        
        # Remove leakage features from drift evidence
        if leakage_features:
            original_drift_count = len(evidence.get("feature_drift", []))
            evidence["feature_drift"] = [
                drift for drift in evidence.get("feature_drift", [])
                if isinstance(drift, dict) and drift.get("feature", "") not in leakage_features
            ]
            removed_count = original_drift_count - len(evidence["feature_drift"])
            
            if removed_count > 0:
                self._log(f"Conflict Resolution: Removed {removed_count} feature(s) from drift due to leakage detection")
                for feature in leakage_features:
                    self._audit_log.append(f"[Conflict Resolution] {feature} removed from drift (leakage takes priority)")
        
        return evidence
    
    def _detect_importance_drift(self, training_importance: Dict, production_importance: Dict,
                                threshold: float = 0.10) -> List[Dict]:
        """
        Detect SHAP importance drift between training and production.

        Args:
            training_importance: Dict of feature -> importance (training)
            production_importance: Dict of feature -> importance (production)
            threshold: Minimum importance change to flag as drift

        Returns:
            List of features with importance drift
        """
        importance_drift = []

        # Get common features
        common_features = set(training_importance.keys()) & set(production_importance.keys())

        for feature in common_features:
            # Skip identifier columns
            if self._is_identifier_column(feature):
                if feature not in self._excluded_features:
                    self._excluded_features.add(feature)
                    self._audit_log.append(f"[Feature Excluded] {feature} (Identifier Column)")
                    self._log(f"Excluding identifier column from importance drift: {feature}")
                continue
            
            train_imp = training_importance[feature]
            prod_imp = production_importance[feature]

            # Calculate absolute change
            change = abs(prod_imp - train_imp)

            if change >= threshold:
                importance_drift.append({
                    "feature": feature,
                    "training_importance": train_imp,
                    "production_importance": prod_imp,
                    "change": change,
                    "evidence": f"Training={train_imp:.3f}, Production={prod_imp:.3f}, Change={change:.3f}"
                })

        return importance_drift
    
    def _calculate_severity(self, psi: float, ks_stat: float) -> tuple[str, int]:
        """
        Calculate severity and score for feature drift.
        
        Args:
            psi: Population Stability Index
            ks_stat: Kolmogorov-Smirnov statistic
        
        Returns:
            Tuple of (severity: str, score: int)
        """
        if psi >= 0.2 and ks_stat >= 0.3:
            return "CRITICAL", 100
        elif psi >= 0.2 or (psi >= 0.1 and ks_stat >= 0.2):
            return "HIGH", 75
        elif psi >= 0.1 or ks_stat >= 0.2:
            return "MEDIUM", 50
        elif ks_stat >= 0.1:
            return "LOW", 25
        else:
            return "NONE", 0

    def _score_root_causes(self, evidence: Dict) -> List[Dict]:
        """
        Score potential root causes based on evidence.
        
        Args:
            evidence: Dictionary of evidence by category
        
        Returns:
            List of scored root causes, sorted by score
        """
        scored_causes = []
        
        # Score leakage (highest priority)
        for leak in evidence["leakage"]:
            leakage_confidence = leak.get("leakage_confidence", 0)
            score = 0
            severity = "CRITICAL"
            
            if leakage_confidence >= 0.90:
                score = self._scoring_rules["leakage"]["leakage_confidence_gt_090"]["score"]
            elif leakage_confidence >= 0.70:
                score = self._scoring_rules["leakage"]["leakage_confidence_gt_070"]["score"]
            elif leakage_confidence >= 0.50:
                score = self._scoring_rules["leakage"]["leakage_confidence_gt_050"]["score"]
            
            if score > 0:
                scored_causes.append({
                    "cause": f"{leak['feature']} target leakage",
                    "score": score,
                    "severity": severity,
                    "evidence": [leak["evidence"]],
                    "category": "target_leakage",
                    "source_modules": ["leakage_engine"]
                })
        
        # Score feature drift
        for drift in evidence["feature_drift"]:
            psi = drift.get("psi", 0)
            ks = drift.get("ks_stat", 0)
            
            severity, score = self._calculate_severity(psi, ks)
            
            if severity != "NONE":
                scored_causes.append({
                    "cause": f"{drift['feature']} feature drift",
                    "score": score,
                    "severity": severity,
                    "evidence": [drift["evidence"]],
                    "category": "feature_drift",
                    "source_modules": ["drift_engine"]
                })
        
        # Score slice degradation
        for sl in evidence["slice_degradation"]:
            score = 0
            severity = "LOW"
            source_modules = ["slicer_engine"]
            
            effect_size = sl.get("effect_size", 0)
            loss_gap = sl.get("loss_gap", 0)
            
            if effect_size >= 0.8:
                score += self._scoring_rules["slice_degradation"]["effect_size_gt_08"]["score"]
                severity = self._scoring_rules["slice_degradation"]["effect_size_gt_08"]["severity"]
            elif effect_size >= 0.5:
                score += self._scoring_rules["slice_degradation"]["effect_size_gt_05"]["score"]
                severity = self._scoring_rules["slice_degradation"]["effect_size_gt_05"]["severity"]
            
            if loss_gap >= 0.20:
                score += self._scoring_rules["slice_degradation"]["accuracy_drop_gt_20"]["score"]
                severity = max(severity, self._scoring_rules["slice_degradation"]["accuracy_drop_gt_20"]["severity"],
                               key=lambda x: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(x))
            elif loss_gap >= 0.10:
                score += self._scoring_rules["slice_degradation"]["accuracy_drop_gt_10"]["score"]
                severity = max(severity, self._scoring_rules["slice_degradation"]["accuracy_drop_gt_10"]["severity"],
                               key=lambda x: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(x))
            
            if score > 0:
                scored_causes.append({
                    "cause": f"Slice failure: {sl['slice']}",
                    "score": min(score, 100),
                    "severity": severity,
                    "evidence": [sl["evidence"]],
                    "category": "slice_degradation",
                    "source_modules": source_modules
                })
        
        # Score calibration issues
        for cal in evidence["calibration"]:
            score = 0
            severity = "LOW"
            source_modules = ["calibration_engine"]
            
            ece = cal.get("ece", 0)
            
            if ece >= 0.10:
                score += self._scoring_rules["calibration"]["ece_increase_gt_010"]["score"]
                severity = self._scoring_rules["calibration"]["ece_increase_gt_010"]["severity"]
            elif ece >= 0.05:
                score += self._scoring_rules["calibration"]["ece_increase_gt_005"]["score"]
                severity = self._scoring_rules["calibration"]["ece_increase_gt_005"]["severity"]
            
            if score > 0:
                scored_causes.append({
                    "cause": "Calibration degradation",
                    "score": min(score, 100),
                    "severity": severity,
                    "evidence": [cal["evidence"]],
                    "category": "calibration",
                    "source_modules": source_modules
                })
        
        # Score missing values
        for mv in evidence["missing_values"]:
            score = 0
            severity = "LOW"
            source_modules = ["missing_data_engine"]
            
            missing_rate = mv.get("missing_rate", 0)
            
            if missing_rate >= 0.20:
                score += self._scoring_rules["missing_values"]["missing_rate_gt_20"]["score"]
                severity = self._scoring_rules["missing_values"]["missing_rate_gt_20"]["severity"]
            elif missing_rate >= 0.10:
                score += self._scoring_rules["missing_values"]["missing_rate_gt_10"]["score"]
                severity = self._scoring_rules["missing_values"]["missing_rate_gt_10"]["severity"]
            
            if score > 0:
                scored_causes.append({
                    "cause": f"Missing values in {mv['feature']}",
                    "score": min(score, 100),
                    "severity": severity,
                    "evidence": [mv["evidence"]],
                    "category": "missing_values",
                    "source_modules": source_modules
                })
        
        # Score outliers
        for out in evidence["outliers"]:
            score = 0
            severity = "LOW"
            source_modules = ["missing_data_engine"]
            
            outlier_rate = out.get("outlier_rate", 0)
            
            if outlier_rate >= 0.50:
                score += self._scoring_rules["outliers"]["outlier_increase_gt_50"]["score"]
                severity = self._scoring_rules["outliers"]["outlier_increase_gt_50"]["severity"]
            elif outlier_rate >= 0.20:
                score += self._scoring_rules["outliers"]["outlier_increase_gt_20"]["score"]
                severity = self._scoring_rules["outliers"]["outlier_increase_gt_20"]["severity"]
            
            if score > 0:
                scored_causes.append({
                    "cause": f"Outlier increase in {out['feature']}",
                    "score": min(score, 100),
                    "severity": severity,
                    "evidence": [out["evidence"]],
                    "category": "outliers",
                    "source_modules": source_modules
                })
        
        # Score importance drift
        for imp_drift in evidence["importance_drift"]:
            score = 0
            severity = "LOW"
            source_modules = ["explainability_engine"]
            
            change = imp_drift.get("change", 0)
            
            if change >= 0.30:
                score += self._scoring_rules["importance_drift"]["importance_change_gt_030"]["score"]
                severity = self._scoring_rules["importance_drift"]["importance_change_gt_030"]["severity"]
            elif change >= 0.20:
                score += self._scoring_rules["importance_drift"]["importance_change_gt_020"]["score"]
                severity = self._scoring_rules["importance_drift"]["importance_change_gt_020"]["severity"]
            elif change >= 0.10:
                score += self._scoring_rules["importance_drift"]["importance_change_gt_010"]["score"]
                severity = self._scoring_rules["importance_drift"]["importance_change_gt_010"]["severity"]
            
            if score > 0:
                scored_causes.append({
                    "cause": f"Feature importance drift: {imp_drift['feature']}",
                    "score": min(score, 100),
                    "severity": severity,
                    "evidence": [imp_drift["evidence"]],
                    "category": "importance_drift",
                    "source_modules": source_modules
                })
        
        # Sort by score descending
        scored_causes.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top 10 causes
        return scored_causes[:10]
    
    def _calculate_health_status(self, scored_causes: List[Dict]) -> tuple:
        """
        Calculate overall health status and confidence score.
        
        Args:
            scored_causes: List of scored root causes
        
        Returns:
            Tuple of (health_status, confidence)
        """
        if not scored_causes:
            return "Healthy", 100
        
        # Calculate weighted score based on top causes
        top_causes = scored_causes[:3]
        total_score = sum(c["score"] for c in top_causes)
        max_possible = len(top_causes) * 100
        avg_score = total_score / max(len(top_causes), 1)
        
        # Determine health status
        if avg_score < 30:
            health_status = "Healthy"
        elif avg_score < 60:
            health_status = "Warning"
        else:
            health_status = "Critical"
        
        # Confidence is based on number of high-severity causes
        high_severity_count = sum(1 for c in scored_causes if c["severity"] in ["HIGH", "CRITICAL"])
        confidence = min(100, 50 + high_severity_count * 10)
        
        return health_status, int(confidence)
    
    def _generate_recommendations(self, scored_causes: List[Dict], leakage_evidence: List[Dict]) -> List[str]:
        """
        Generate context-aware recommended actions based on root causes.
        
        Args:
            scored_causes: List of scored root causes
            leakage_evidence: List of leakage evidence items
        
        Returns:
            List of recommended actions
        """
        recommendations = []
        
        if not scored_causes:
            return ["No critical issues detected. Continue monitoring model performance."]
        
        # Get top causes
        top_causes = scored_causes[:5]
        
        # Check if there are any leakage issues
        has_leakage = any(cause.get("category") == "target_leakage" for cause in top_causes)
        
        for cause in top_causes:
            category = cause.get("category", "")
            cause_text = cause.get("cause", "")
            severity = cause.get("severity", "LOW")
            evidence = cause.get("evidence", [])
            
            if category == "target_leakage":
                # Extract feature name from cause text
                feature = cause_text.replace(" target leakage", "")
                evidence_str = evidence[0] if evidence else ""
                recommendations.append(
                    f"Exclude {feature} from training immediately ({evidence_str}). "
                    f"Retrain model without this feature and revalidate metrics."
                )
            elif category == "feature_drift":
                # Extract feature name from cause text
                feature = cause_text.replace(" feature drift", "")
                evidence_str = evidence[0] if evidence else ""
                recommendations.append(
                    f"{feature} distribution shifted significantly ({evidence_str}). "
                    f"Collect recent samples from the affected segment before retraining."
                )
            elif category == "slice_degradation":
                # Extract slice description
                slice_desc = cause_text.replace("Slice failure: ", "")
                evidence_str = evidence[0] if evidence else ""
                recommendations.append(
                    f"Slice failure detected: {slice_desc} ({evidence_str}). "
                    f"Investigate affected customer segments and consider targeted data collection."
                )
            elif category == "calibration":
                evidence_str = evidence[0] if evidence else ""
                recommendations.append(
                    f"Calibration degradation detected ({evidence_str}). "
                    f"Apply post-hoc calibration (Platt scaling or isotonic regression) to improve probability estimates."
                )
            elif category == "missing_values":
                feature = cause_text.replace("Missing values in ", "")
                evidence_str = evidence[0] if evidence else ""
                recommendations.append(
                    f"Missing value increase in {feature} ({evidence_str}). "
                    f"Investigate upstream data pipeline and implement imputation strategy."
                )
            elif category == "outliers":
                feature = cause_text.replace("Outlier increase in ", "")
                evidence_str = evidence[0] if evidence else ""
                recommendations.append(
                    f"Outlier increase in {feature} ({evidence_str}). "
                    f"Review data ingestion process and consider outlier detection and filtering."
                )
            elif category == "importance_drift":
                feature = cause_text.replace("Feature importance drift: ", "")
                evidence_str = evidence[0] if evidence else ""
                recommendations.append(
                    f"Feature importance changed significantly for {feature} ({evidence_str}). "
                    f"Model behavior has shifted - investigate data distribution changes and consider retraining."
                )
        
        # Add general recommendation if multiple issues
        if len(top_causes) >= 3:
            recommendations.append(
                f"Multiple issues detected (severity: {top_causes[0]['severity']}). "
                f"Prioritize addressing the highest-scoring root causes first."
            )
        
        return recommendations[:5]
    
    def _summarize_evidence(self, evidence: Dict) -> str:
        """
        Generate a human-readable summary of evidence.
        
        Args:
            evidence: Dictionary of evidence by category
        
        Returns:
            Summary string
        """
        summary_parts = []
        
        if evidence["feature_drift"]:
            n_drift = len(evidence["feature_drift"])
            summary_parts.append(f"{n_drift} features show drift")
        
        if evidence["slice_degradation"]:
            n_slices = len(evidence["slice_degradation"])
            summary_parts.append(f"{n_slices} problematic data slices")
        
        if evidence["calibration"]:
            summary_parts.append("calibration degradation detected")
        
        if evidence["missing_values"]:
            n_missing = len(evidence["missing_values"])
            summary_parts.append(f"{n_missing} features with missing value increase")
        
        if evidence["outliers"]:
            n_outliers = len(evidence["outliers"])
            summary_parts.append(f"{n_outliers} features with outlier increase")
        
        if not summary_parts:
            return "No significant issues detected."
        
        return ". ".join(summary_parts) + "."
    
    def generate_investigation_summary(self, result: Dict) -> str:
        """
        Generate a formatted investigation summary.
        
        Args:
            result: Result from run() method
        
        Returns:
            Formatted summary string
        """
        lines = [
            "INVESTIGATION SUMMARY",
            "",
            f"Model Health: {result['health_status']}",
            f"Confidence: {result['confidence']}%",
            "",
            "Evidence:"
        ]
        
        for evidence in result.get("evidence_summary", "").split(". "):
            if evidence:
                lines.append(f"  • {evidence}")
        
        lines.append("")
        lines.append("Most Likely Causes:")
        
        for i, cause in enumerate(result.get("root_causes", [])[:3], 1):
            lines.append(f"  {i}. {cause['cause']} (score: {cause['score']}, severity: {cause['severity']})")
            for ev in cause.get("evidence", []):
                lines.append(f"     - {ev}")
        
        lines.append("")
        lines.append("Recommended Actions:")
        
        for action in result.get("recommended_actions", []):
            lines.append(f"  • {action}")
        
        return "\n".join(lines)
