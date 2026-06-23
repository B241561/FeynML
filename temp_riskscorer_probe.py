from engine.modules.risk_scorer import RiskScorer
from engine.modules.fairness_engine import FairnessEngine

# Case: all zero base_rate and selection_rate
N=50
y_t=[0]*N
y_p=[0]*N
groups=['A']*N
fe=FairnessEngine()
fe.register_axis('group', groups)
fe_res = fe.run(y_t, y_p)
print('FairnessEngine severity:', fe_res.get('severity'))

rs=RiskScorer()
print('_normalize_report_severity:', rs._normalize_report_severity(fe_res))
sc = rs.score(fairness_report=fe_res)
print('RiskScorer breakdown.fairness:', sc['breakdown']['fairness'])
