from scratch.phase2.fairness_audit import audit_by_group
import json

y_true=[0,1,0,1]
y_pred=[0,1,0,1]
groups=['A','A','B','B']
res=audit_by_group(y_true,y_pred,groups)
print('per_group selection rates:', {k:v['selection_rate'] for k,v in res['per_group_rates'].items()})
print('summary.di_ratio (transformed):', res['summary']['di_ratio'])
print('dp_gap:', res['summary']['dp_gap'])
print('severity:', res['severity'])
print('\nFULL JSON:')
print(json.dumps(res, indent=2))
