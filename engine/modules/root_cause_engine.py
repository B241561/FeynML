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
    from .risk_scorer import RiskScorer
except ImportError:
    from risk_scorer import RiskScorer

try:
    from .investigation import Investigation, RootCause
except ImportError:
    from investigation import Investigation, RootCause

# Evidence registry and validator (Sprint 3A Phase 2 integration)
try:
    from .evidence_registry import EvidenceRegistry
    from .fact_validator import validate as validate_facts
except Exception:
    # Fallback for environments where module path resolution differs
    try:
        from evidence_registry import EvidenceRegistry
        from fact_validator import validate as validate_facts
    except Exception:
        EvidenceRegistry = None
        validate_facts = None


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
        # Evidence registry for Sprint 3A Phase 2 (non-blocking)
        if EvidenceRegistry is not None:
            self._evidence_registry = EvidenceRegistry()
        else:
            self._evidence_registry = None
        # Store last validation result for introspection (do not expose in public APIs)
        self._last_claim_validation: Optional[Dict[str, Any]] = None

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
            leakage_report: Optional[Dict] = None,
            label_noise_report: Optional[Dict] = None,
            fairness_report: Optional[Dict] = None) -> Dict:
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
                "audit_log": self._audit_log,
                "canonical_drift_count": evidence.get('canonical_drift_count', 0)
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
        
        # Compute unified risk using RiskScorer so all systems share the same source
        scorer = RiskScorer()
        risk = scorer.score(
            leakage_report=leakage_report,
            label_noise_report=label_noise_report,
            drift_report=drift_report,
            calibration_report=calibration_report,
            missing_data_report=data_quality_report,
            fairness_report=fairness_report,
            root_causes=[{"severity": c.get('severity')} for c in scored_causes]
        )

        # Attach risk into investigation metadata and result envelope
        investigation.metadata = {**investigation.metadata, 'risk': risk}

        # Validate registered claims from EvidenceRegistry (non-blocking)
        try:
            if validate_facts is not None and self._evidence_registry is not None:
                claims = self._evidence_registry.get_claims()
                self._last_claim_validation = validate_facts(claims)
        except Exception:
            # Store a minimal failure record internally; do not change external outputs
            self._last_claim_validation = {"valid": False, "issues": ["validation_error"]}

        result = investigation.to_dict()
        # Build a concise human-readable "why" explanation using available evidence
        try:
            why_lines = []
            # Leakage
            leaks = [e.get('feature') for e in evidence.get('leakage', []) if isinstance(e, dict) and e.get('feature')]
            if leaks:
                why_lines.append(f"Target leakage detected in {', '.join(leaks[:5])}.")

            # Label noise
            try:
                ln = None
                if label_noise_report and isinstance(label_noise_report, dict):
                    ln_find = label_noise_report.get('findings') or label_noise_report.get('findings', {}) or label_noise_report
                    if isinstance(ln_find, dict):
                        ln_val = ln_find.get('estimated_noise_fraction') or ln_find.get('estimated_noise_rate')
                        if ln_val is not None:
                            if isinstance(ln_val, (int, float)) and ln_val <= 1:
                                ln = f"{round(float(ln_val)*100,1)}%"
                            else:
                                ln = f"{ln_val}%" if isinstance(ln_val, (int, float)) else str(ln_val)
                if ln:
                    why_lines.append(f"Estimated label noise rate is {ln}.")
            except Exception:
                pass

            # Drift - use canonical drift count (only explicit DRIFT status)
            n_drift = evidence.get('canonical_drift_count', len(evidence.get('feature_drift', [])))
            if n_drift > 0:
                why_lines.append(f"{n_drift} drifted feature(s) detected.")

            # Calibration
            try:
                cal_metric = None
                if calibration_report and isinstance(calibration_report, dict):
                    findings = calibration_report.get('findings') or calibration_report
                    if isinstance(findings, dict):
                        val = findings.get('ece') or findings.get('best_ece') or findings.get('raw_ece')
                        if val is not None:
                            if isinstance(val, (int, float)) and val <= 1:
                                cal_metric = f"{round(float(val)*100,2)}%"
                            else:
                                cal_metric = f"{val}%"
                if cal_metric:
                    why_lines.append(f"Calibration reported ECE={cal_metric}.")
            except Exception:
                pass

            # Fairness
            try:
                fair_msg = None
                if fairness_report and isinstance(fairness_report, dict):
                    sev = fairness_report.get('severity') or fairness_report.get('status')
                    if sev and str(sev).upper() in ('NONE', 'OK'):
                        fair_msg = "No fairness concerns detected"
                    else:
                        fair_msg = f"Fairness severity: {sev}"
                if fair_msg:
                    why_lines.append(fair_msg)
            except Exception:
                pass

            # If we couldn't assemble evidence-specific lines, fall back to the
            # unified risk breakdown produced by RiskScorer so the dashboard
            # always has a concise explanation.
            if not why_lines and isinstance(risk, dict):
                rb = risk.get('breakdown', {}) or {}
                # List human-friendly bullets
                fb = []
                if rb.get('leakage') and rb.get('leakage') != 'NONE':
                    leaks = [e.get('feature') for e in evidence.get('leakage', []) if isinstance(e, dict) and e.get('feature')]
                    fb.append(f"Target leakage detected in {', '.join(leaks[:3])}" if leaks else "Target leakage detected")
                if rb.get('label_noise') and rb.get('label_noise') != 'NONE':
                    ln_val = None
                    try:
                        ln_find = label_noise_report.get('findings') if label_noise_report and isinstance(label_noise_report, dict) else None
                        if ln_find:
                            ln_val = ln_find.get('estimated_noise_fraction') or ln_find.get('estimated_noise_rate')
                    except Exception:
                        ln_val = None
                    if ln_val is not None:
                        try:
                            ln_pct = float(ln_val) * 100 if float(ln_val) <= 1 else float(ln_val)
                            fb.append(f"Label noise rate {ln_pct:.1f}%")
                        except Exception:
                            fb.append(f"Label noise detected: {ln_val}")
                    else:
                        fb.append("Label noise detected")
                if rb.get('drift') and rb.get('drift') != 'NONE':
                    n_drift = evidence.get('canonical_drift_count', len(evidence.get('feature_drift', [])))
                    fb.append(f"{n_drift} drifted feature(s) detected")
                if rb.get('calibration') and rb.get('calibration') != 'NONE':
                    try:
                        cal_val = None
                        cal_find = calibration_report.get('findings') if calibration_report and isinstance(calibration_report, dict) else calibration_report
                        if isinstance(cal_find, dict):
                            val = cal_find.get('ece') or cal_find.get('best_ece') or cal_find.get('raw_ece')
                            if val is not None:
                                cal_val = (float(val)*100) if float(val) <= 1 else float(val)
                        if cal_val is not None:
                            fb.append(f"Calibration ECE={cal_val:.2f}%")
                        else:
                            fb.append("Calibration concerns detected")
                    except Exception:
                        fb.append("Calibration concerns detected")
                # Fairness
                if rb.get('fairness') and rb.get('fairness') != 'NONE':
                    fb.append(f"Fairness severity: {rb.get('fairness')}")
                else:
                    fb.append("No fairness concerns")

                if fb:
                    # Compose concise explanation
                    risk_explanation = "; ".join(fb) + "."
                else:
                    risk_explanation = ""
            else:
                risk_explanation = " ".join(why_lines) if why_lines else ""

            result['risk_explanation'] = risk_explanation
            investigation.metadata['risk_explanation'] = risk_explanation
        except Exception:
            pass
        result['risk'] = risk
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
                            if isinstance(feat, dict):
                                status = feat.get("status")
                                if status not in ("DRIFT", "WARN"):
                                    continue
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

                                # Only treat explicit DRIFT as canonical drift evidence
                                if status == "DRIFT":
                                    evidence["feature_drift"].append({
                                        "feature": feature_name,
                                        "psi": psi_val,
                                        "ks_stat": ks_val,
                                        "status": status,
                                        "evidence": f"PSI={psi_val:.3f}, KS={ks_val:.3f}"
                                    })
                                else:
                                    # Preserve WARN entries separately to avoid losing metadata
                                    evidence.setdefault("feature_warn", []).append({
                                        "feature": feature_name,
                                        "psi": psi_val,
                                        "ks_stat": ks_val,
                                        "status": status,
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
            # After processing per_feature, compute canonical drift count
            try:
                canonical_drift_count = sum(1 for f in per_feature if isinstance(f, dict) and f.get("status") == "DRIFT")
                evidence["canonical_drift_count"] = canonical_drift_count
            except Exception:
                evidence["canonical_drift_count"] = 0
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
        Score is scaled by PSI magnitude within each severity tier so that
        features with higher drift score higher and appear first in prioritization.

        Args:
            psi: Population Stability Index
            ks_stat: Kolmogorov-Smirnov statistic

        Returns:
            Tuple of (severity: str, score: int)
        """
        if psi >= 0.2 and ks_stat >= 0.3:
            # Scale 80-100 by PSI magnitude (PSI=0.2 → 80, PSI=2.5+ → 100)
            score = int(min(100, 80 + (min(psi, 2.5) / 2.5) * 20))
            return "CRITICAL", score
        elif psi >= 0.2 or (psi >= 0.1 and ks_stat >= 0.2):
            # Scale 55-79 by PSI magnitude
            score = int(min(79, 55 + (min(psi, 0.5) / 0.5) * 24))
            return "HIGH", score
        elif psi >= 0.1 or ks_stat >= 0.2:
            # Scale 30-54 by PSI magnitude
            score = int(min(54, 30 + (min(psi, 0.2) / 0.2) * 24))
            return "MEDIUM", score
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
                rec = {
                    "cause": f"{leak['feature']} target leakage",
                    "score": score,
                    "severity": severity,
                    "evidence": [leak["evidence"]],
                    "category": "target_leakage",
                    "source_modules": ["leakage_engine"]
                }
                scored_causes.append(rec)
                # Register claim in EvidenceRegistry if available
                try:
                    if self._evidence_registry is not None:
                        self._evidence_registry.register(
                            claim=rec['cause'],
                            category=rec['category'],
                            evidence=rec['evidence'],
                            source_module='root_cause_engine',
                            confidence=rec.get('score')
                        )
                except Exception:
                    # non-blocking: do not change main behavior on registry errors
                    pass
        
        # Score feature drift
        for drift in evidence["feature_drift"]:
            psi = drift.get("psi", 0)
            ks = drift.get("ks_stat", 0)
            
            severity, score = self._calculate_severity(psi, ks)

            if severity != "NONE":
                rec = {
                    "cause": f"{drift['feature']} feature drift",
                    "score": score,
                    "severity": severity,
                    "evidence": [drift["evidence"]],
                    "category": "feature_drift",
                    "source_modules": ["drift_engine"]
                }
                scored_causes.append(rec)
                try:
                    if self._evidence_registry is not None:
                        self._evidence_registry.register(
                            claim=rec['cause'],
                            category=rec['category'],
                            evidence=rec['evidence'],
                            source_module='root_cause_engine',
                            confidence=rec.get('score')
                        )
                except Exception:
                    pass
        
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
                rec = {
                    "cause": f"Slice failure: {sl['slice']}",
                    "score": min(score, 100),
                    "severity": severity,
                    "evidence": [sl["evidence"]],
                    "category": "slice_degradation",
                    "source_modules": source_modules
                }
                scored_causes.append(rec)
                try:
                    if self._evidence_registry is not None:
                        self._evidence_registry.register(
                            claim=rec['cause'],
                            category=rec['category'],
                            evidence=rec['evidence'],
                            source_module='root_cause_engine',
                            confidence=rec.get('score')
                        )
                except Exception:
                    pass
        
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
                rec = {
                    "cause": "Calibration degradation",
                    "score": min(score, 100),
                    "severity": severity,
                    "evidence": [cal["evidence"]],
                    "category": "calibration",
                    "source_modules": source_modules
                }
                scored_causes.append(rec)
                try:
                    if self._evidence_registry is not None:
                        self._evidence_registry.register(
                            claim=rec['cause'],
                            category=rec['category'],
                            evidence=rec['evidence'],
                            source_module='root_cause_engine',
                            confidence=rec.get('score')
                        )
                except Exception:
                    pass
        
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
                rec = {
                    "cause": f"Missing values in {mv['feature']}",
                    "score": min(score, 100),
                    "severity": severity,
                    "evidence": [mv["evidence"]],
                    "category": "missing_values",
                    "source_modules": source_modules
                }
                scored_causes.append(rec)
                try:
                    if self._evidence_registry is not None:
                        self._evidence_registry.register(
                            claim=rec['cause'],
                            category=rec['category'],
                            evidence=rec['evidence'],
                            source_module='root_cause_engine',
                            confidence=rec.get('score')
                        )
                except Exception:
                    pass
        
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
                rec = {
                    "cause": f"Outlier increase in {out['feature']}",
                    "score": min(score, 100),
                    "severity": severity,
                    "evidence": [out["evidence"]],
                    "category": "outliers",
                    "source_modules": source_modules
                }
                scored_causes.append(rec)
                try:
                    if self._evidence_registry is not None:
                        self._evidence_registry.register(
                            claim=rec['cause'],
                            category=rec['category'],
                            evidence=rec['evidence'],
                            source_module='root_cause_engine',
                            confidence=rec.get('score')
                        )
                except Exception:
                    pass
        
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
                rec = {
                    "cause": f"Feature importance drift: {imp_drift['feature']}",
                    "score": min(score, 100),
                    "severity": severity,
                    "evidence": [imp_drift["evidence"]],
                    "category": "importance_drift",
                    "source_modules": source_modules
                }
                scored_causes.append(rec)
                try:
                    if self._evidence_registry is not None:
                        self._evidence_registry.register(
                            claim=rec['cause'],
                            category=rec['category'],
                            evidence=rec['evidence'],
                            source_module='root_cause_engine',
                            confidence=rec.get('score')
                        )
                except Exception:
                    pass
        
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
        # If no causes detected, preserve backward compatibility for test contracts
        # Historically, the engine returned full confidence (100%) when no causes were found.
        if not scored_causes:
            return "Healthy", 100

        # Calculate weighted score based on top causes
        top_causes = scored_causes[:3]
        total_score = sum(c["score"] for c in top_causes)
        max_possible = len(top_causes) * 100
        avg_score = total_score / max(len(top_causes), 1)
        
        # Determine health status from average top-cause strength
        if avg_score < 30:
            health_status = "Healthy"
        elif avg_score < 60:
            health_status = "Warning"
        else:
            health_status = "Critical"
        
        # Confidence calibration — avoid unrealistic certainty
        # Evidence considered:
        #  - number of high-severity findings (HIGH or CRITICAL)
        #  - number of critical findings
        #  - average root-cause score (strength)
        high_severity_count = sum(1 for c in scored_causes if c["severity"] in ["HIGH", "CRITICAL"])
        critical_count = sum(1 for c in scored_causes if c["severity"] == "CRITICAL")
        avg_score_all = (sum(c.get("score", 0) for c in scored_causes) / max(len(scored_causes), 1))

        confidence = 50
        # Evidence strength from number of serious findings (cap contribution)
        confidence += min(20, high_severity_count * 4)
        # Additional confidence from critical findings (cap contribution)
        confidence += min(15, critical_count * 3)
        # Additional confidence from average root-cause strength (score in 0-100)
        confidence += min(10, avg_score_all / 10)
        # Never claim certainty — cap at 95
        confidence = min(95, confidence)

        return health_status, int(confidence)
    
    def _generate_recommendations(self, scored_causes: List[Dict], leakage_evidence: List[Dict]) -> List[str]:
        """
        Generate context-aware recommended actions based on root causes.

        This implementation deduplicates causes by category and consolidates
        recommendations to avoid repetition (e.g., multiple drift findings ->
        single consolidated drift recommendation).
        """
        from collections import defaultdict

        recommendations = []

        if not scored_causes:
            return ["No critical issues detected. Continue monitoring model performance."]

        # Group causes by category for consolidation
        causes_by_category = defaultdict(list)
        for c in scored_causes:
            cat = c.get('category') or 'other'
            causes_by_category[cat].append(c)

        # Consolidate feature drift recommendations
        if causes_by_category.get('feature_drift'):
            drifts = causes_by_category['feature_drift']
            n = len(drifts)
            drifts_sorted = sorted(drifts, key=lambda x: x.get('score', 0), reverse=True)
            feats_sorted = [d.get('cause', '').replace(' feature drift', '') for d in drifts_sorted]
            top3 = feats_sorted[:3]
            features_str = ', '.join(top3)
            top_feature = feats_sorted[0] if feats_sorted else 'unknown'
            top_score = drifts_sorted[0].get('score', 0) if drifts_sorted else 0

            recommendations.append(
                f"{n} features show significant drift.\n\n"
                f"Highest priority: {top_feature} (score={top_score}) — fix this first.\n\n"
                f"All affected: {features_str}.\n\n"
                f"Collect fresh production samples, validate feature distributions, and retrain the model using recent data."
            )

        # Consolidate leakage recommendations
        if causes_by_category.get('target_leakage'):
            leaks = causes_by_category['target_leakage']
            n = len(leaks)
            recommendations.append(
                f"Target leakage detected in {n} feature(s).\n\nRemove leakage sources, retrain the model, and revalidate all performance metrics."
            )

        # Consolidate missing values recommendations
        if causes_by_category.get('missing_values'):
            mvs = causes_by_category['missing_values']
            n = len(mvs)
            recommendations.append(
                f"Missing value increases detected in {n} feature(s).\n\nInvestigate upstream data pipelines and implement consistent imputation."
            )

        # Consolidate outliers recommendations
        if causes_by_category.get('outliers'):
            outs = causes_by_category['outliers']
            n = len(outs)
            recommendations.append(
                f"Outlier growth detected across {n} feature(s).\n\nReview data ingestion quality and strengthen anomaly detection."
            )

        # Consolidate importance drift recommendations
        if causes_by_category.get('importance_drift'):
            imps = causes_by_category['importance_drift']
            n = len(imps)
            top3 = [d.get('cause', '').replace('Feature importance drift: ', '') for d in sorted(imps, key=lambda x: x.get('score',0), reverse=True)][:3]
            top3_str = ', '.join(top3)
            recommendations.append(
                f"Model behavior shifted across {n} important feature(s).\n\nInvestigate concept drift and consider model retraining.\nTop impacted: {top3_str}."
            )

        # Consolidate calibration recommendations (single message)
        if causes_by_category.get('calibration'):
            recommendations.append(
                "Calibration degradation detected.\n\nApply post-hoc calibration (Platt scaling or isotonic regression) and validate probability estimates."
            )

        # Consolidate slice_degradation as individual recommendations but limit spam
        if causes_by_category.get('slice_degradation'):
            slices = causes_by_category['slice_degradation']
            for sl in slices[:3]:
                desc = sl.get('cause') if sl.get('cause') else sl.get('evidence', 'Slice issue')
                recommendations.append(f"Slice issue: {desc}. Investigate affected segments and consider targeted data collection.")

        # If nothing consolidated above (other categories), fall back to per-cause recommendations
        other_categories = set(c.get('category') for c in scored_causes) - set(['feature_drift','target_leakage','missing_values','outliers','importance_drift','calibration','slice_degradation'])
        for cat in other_categories:
            for c in causes_by_category.get(cat, [])[:3]:
                recommendations.append(f"{c.get('cause')}: {', '.join(c.get('evidence', []))}")

        # Limit total recommendations to avoid overwhelming the user
        # Prefer consolidated messages; do not add extra generic 'multiple issues' text when a single consolidated category exists
        if len(recommendations) > 5:
            recommendations = recommendations[:5]

        return recommendations
    
    def _summarize_evidence(self, evidence: Dict) -> str:
        """
        Generate a human-readable summary of evidence.
        
        Args:
            evidence: Dictionary of evidence by category
        
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Use canonical drift count (explicit DRIFT status) for summaries
        n_drift = evidence.get("canonical_drift_count", len(evidence.get("feature_drift", [])))
        if n_drift:
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
        Generate a formatted investigation summary with deduplication of root causes by category.

        If multiple root causes belong to the same category (e.g., many feature drifts),
        summarize them rather than listing each individually to avoid repetition.
        """
        from collections import defaultdict

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
        # Backwards-compatible heading expected by legacy tests
        lines.append("Most Likely Causes:")
        lines.append("")
        lines.append("Primary concerns:")

        # Group root causes by category
        causes = result.get('root_causes', []) or []
        by_cat = defaultdict(list)
        for c in causes:
            cat = c.get('category', 'other')
            by_cat[cat].append(c)

        # If feature_drift dominates, produce a consolidated block
        if by_cat.get('feature_drift') and len(by_cat['feature_drift']) > 1:
            n = len(by_cat['feature_drift'])
            top3 = [c.get('cause','').replace(' feature drift','') for c in sorted(by_cat['feature_drift'], key=lambda x: x.get('score',0), reverse=True)[:3]]
            lines.append(f"  Primary concern: Feature drift affecting {n} features.")
            lines.append(f"  Most impacted: {', '.join(top3)}")
        else:
            # Otherwise list up to three most likely causes (deduplicated by category)
            listed = 0
            for cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
                if listed >= 3:
                    break
                if len(items) == 1:
                    c = items[0]
                    lines.append(f"  - {c.get('cause')} (score: {c.get('score')}, severity: {c.get('severity')})")
                    listed += 1
                else:
                    # Consolidated category summary
                    lines.append(f"  - {cat.replace('_',' ').title()}: {len(items)} related findings")
                    listed += 1

        lines.append("")
        lines.append("Recommended Actions:")

        # Deduplicated recommendations already provided in recommended_actions
        for action in result.get("recommended_actions", []):
            lines.append(f"  • {action}")

        return "\n".join(lines)
