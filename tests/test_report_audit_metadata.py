import pytest
from engine.modules.report_engine import ReportEngine


def test_audit_metadata_present():
    re = ReportEngine(project_name="P", model_name="M")
    # Simulate a root_cause section that carries precomputed validation result in metadata
    validation = {"valid": True, "issues": []}
    re.add_section("root_cause", {
        "health_status": "Healthy",
        "confidence": 100,
        "root_causes": [],
        "metadata": {
            "evidence_validation": validation
        }
    })

    findings = re.get_findings()
    assert "audit_metadata" in findings
    assert "evidence_validation" in findings["audit_metadata"]
    assert findings["audit_metadata"]["evidence_validation"] == validation


def test_audit_metadata_absent():
    re = ReportEngine(project_name="P", model_name="M")
    re.add_section("root_cause", {
        "health_status": "Healthy",
        "confidence": 100,
        "root_causes": [],
    })

    findings = re.get_findings()
    assert "audit_metadata" not in findings


def test_existing_structure_unchanged():
    re = ReportEngine(project_name="X", model_name="Y")
    re.add_section("evaluation", {"accuracy": 0.9, "severity": "OK"})
    out = re.get_findings()
    # core structure still present
    assert out["project"] == "X"
    assert out["model"] == "Y"
    assert "sections" in out
    assert "evaluation" in out["sections"]
    # No audit_metadata by default
    assert ("audit_metadata" not in out) or isinstance(out.get("audit_metadata"), dict)
