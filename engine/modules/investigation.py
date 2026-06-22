"""
Investigation Module - Unified Investigation Result Object
=========================================================
Defines the unified investigation result object used across all downstream systems.

This object provides a consistent interface for:
- Report generation
- UI display
- Monitoring systems
- Alerting
- Historical tracking

Usage:
    from engine.modules.investigation import Investigation
    
    investigation = Investigation(
        investigation_id="inv_12345",
        health_status="Warning",
        confidence=75,
        root_causes=[...],
        evidence=[...],
        recommendations=[...]
    )
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid


@dataclass
class RootCause:
    """
    Represents a single root cause with full traceability.
    """
    cause: str
    score: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    evidence: List[str]
    category: str
    source_modules: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "cause": self.cause,
            "score": self.score,
            "severity": self.severity,
            "evidence": self.evidence,
            "category": self.category,
            "source_modules": self.source_modules
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RootCause':
        """Create from dictionary."""
        return cls(
            cause=data.get("cause", ""),
            score=data.get("score", 0),
            severity=data.get("severity", "LOW"),
            evidence=data.get("evidence", []),
            category=data.get("category", ""),
            source_modules=data.get("source_modules", [])
        )


@dataclass
class Investigation:
    """
    Unified investigation result object.
    
    This object aggregates findings from all engines and provides
    a consistent interface for downstream consumption.
    """
    investigation_id: str = field(default_factory=lambda: f"inv_{uuid.uuid4().hex[:12]}")
    health_status: str = "Unknown"  # Healthy, Warning, Critical
    confidence: int = 0  # 0-100
    root_causes: List[RootCause] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    module: str = "AutoRootCauseEngine"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of the investigation
        """
        return {
            "investigation_id": self.investigation_id,
            "health_status": self.health_status,
            "confidence": self.confidence,
            "root_causes": [rc.to_dict() for rc in self.root_causes],
            "evidence": self.evidence,
            "recommended_actions": self.recommendations,
            "generated_at": self.generated_at,
            "module": self.module,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Investigation':
        """
        Create from dictionary.
        
        Args:
            data: Dictionary representation of investigation
        
        Returns:
            Investigation object
        """
        root_causes = [
            RootCause.from_dict(rc) if isinstance(rc, dict) else rc
            for rc in data.get("root_causes", [])
        ]
        
        return cls(
            investigation_id=data.get("investigation_id", f"inv_{uuid.uuid4().hex[:12]}"),
            health_status=data.get("health_status", "Unknown"),
            confidence=data.get("confidence", 0),
            root_causes=root_causes,
            evidence=data.get("evidence", []),
            recommendations=data.get("recommended_actions", data.get("recommendations", [])),
            generated_at=data.get("generated_at", datetime.now().isoformat()),
            module=data.get("module", "AutoRootCauseEngine"),
            metadata=data.get("metadata", {})
        )
    
    def add_root_cause(self, cause: str, score: float, severity: str, 
                      evidence: List[str], category: str, source_modules: List[str]):
        """
        Add a root cause to the investigation.
        
        Args:
            cause: Description of the root cause
            score: Confidence score (0-100)
            severity: Severity level (LOW, MEDIUM, HIGH, CRITICAL)
            evidence: List of evidence strings
            category: Category of the root cause
            source_modules: List of source modules that identified this cause
        """
        root_cause = RootCause(
            cause=cause,
            score=score,
            severity=severity,
            evidence=evidence,
            category=category,
            source_modules=source_modules
        )
        self.root_causes.append(root_cause)
        # Sort by score descending
        self.root_causes.sort(key=lambda x: x.score, reverse=True)
    
    def add_recommendation(self, recommendation: str):
        """
        Add a recommendation to the investigation.
        
        Args:
            recommendation: Recommendation text
        """
        self.recommendations.append(recommendation)
    
    def add_evidence(self, evidence: str):
        """
        Add evidence to the investigation.
        
        Args:
            evidence: Evidence text
        """
        self.evidence.append(evidence)
    
    def get_critical_causes(self) -> List[RootCause]:
        """Get all critical severity root causes."""
        return [rc for rc in self.root_causes if rc.severity == "CRITICAL"]
    
    def get_high_severity_causes(self) -> List[RootCause]:
        """Get all high and critical severity root causes."""
        return [rc for rc in self.root_causes if rc.severity in ["HIGH", "CRITICAL"]]
    
    def get_causes_by_category(self, category: str) -> List[RootCause]:
        """Get all root causes for a specific category."""
        return [rc for rc in self.root_causes if rc.category == category]
    
    def get_causes_by_source_module(self, module: str) -> List[RootCause]:
        """Get all root causes identified by a specific source module."""
        return [rc for rc in self.root_causes if module in rc.source_modules]
    
    def is_healthy(self) -> bool:
        """Check if the investigation indicates a healthy system."""
        return self.health_status == "Healthy"
    
    def is_critical(self) -> bool:
        """Check if the investigation indicates a critical issue."""
        return self.health_status == "Critical"
    
    def __repr__(self) -> str:
        return (f"Investigation(id={self.investigation_id}, "
                f"health={self.health_status}, "
                f"confidence={self.confidence}, "
                f"causes={len(self.root_causes)})")
