from scratch.phase2.fairness_audit import audit_by_group
from engine.modules.fairness_engine import FairnessEngine
from engine.modules.risk_scorer import RiskScorer
import json

# Construct extreme disparity case: group A all positive predictions, group B all negative
# Create 100 samples per group
n=100
# ground truth: alternate but we can craft so selection rates differ
# For dp_gap=1.0, selection_rate A=1.0, B=0.0
y_true = [1]*n + [0]*n
# set predictions to all 1 for group A, all 0 for group B
y_pred = [1]*n + [0]*n
# groups labels
groups = ['A']*n + ['B']*n

print('\n=== fairness_audit.audit_by_group ===')
res = audit_by_group(y_true, y_pred, groups, group_name='group')
print('dp_gap:', res['summary']['dp_gap'])
print('eo_gap:', res['summary']['eo_max_gap'])
print('di_ratio:', res['summary']['di_ratio'])
print('overall_sev:', res['severity'])
print('selection_rates:', res['demographic_parity']['selection_rates'])
print('group_counts:', {g: len([1 for x in groups if x==g]) for g in set(groups)})
print('\nFULL audit_by_group JSON:')
print(json.dumps(res, indent=2))

print('\n=== FairnessEngine.run ===')
fe = FairnessEngine()
fe.register_axis('group', groups)
fe_res = fe.run(y_true, y_pred)
print('FairnessEngine raw output keys:', list(fe_res.keys()))
print('FairnessEngine severity:', fe_res.get('severity'))
print('FairnessEngine summary:', fe_res.get('summary'))
print('\nFairnessEngine per_axis keys:', list(fe_res.get('per_axis', {}).keys()))
print('per_axis[group].summary:', fe_res.get('per_axis', {}).get('group', {}).get('summary'))
print('\nFULL FairnessEngine JSON:')
print(json.dumps(fe_res, indent=2))

print('\n=== Dashboard read simulation ===')
# Dashboard expects data.fairness.get('findings', {}).get('disparity_ratio'), 'groups', 'max_disparity'
fake_results = {'fairness': fe_res}
findings = fake_results['fairness'].get('findings', {})
print('dashboard findings object (expected):', findings)
print('dashboard Detected Bias value:', findings.get('disparity_ratio', 0.0))
print('dashboard Impacted Groups value:', len(findings.get('groups', [])) if findings.get('groups') else 0)
print('dashboard Disparity Index value:', (findings.get('max_disparity', 0) * 100) if findings.get('max_disparity') else 0)

print('\n=== RiskScorer normalization ===')
rs = RiskScorer()
# Check how RiskScorer extracts severity from fairness_report
print('_normalize_report_severity(fairness_report):', rs._normalize_report_severity(fe_res))
print('\nScoring with fairness only:')
score = rs.score(fairness_report=fe_res)
print(json.dumps(score, indent=2))
