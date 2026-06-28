from engine.modules.fairness_engine import FairnessEngine

fe = FairnessEngine()
N = 50
# No bias case: all predictions and true labels the same
y_t = [0] * N
y_p = [0] * N
groups = ['A'] * N
fe.register_axis('group', groups)
report = fe.run(y_t, y_p, threshold=0.1)
print('REPORT_SEVERITY:', report.get('severity'))
per = report.get('per_axis') or {}
print('PER_AXIS_KEYS:', list(per.keys()))
for k,v in per.items():
    print('AXIS:', k)
    try:
        print('  summary:', v.get('summary'))
        print('  severity:', v.get('severity'))
        print('  group_names:', v.get('group_names'))
        print('  per_group_rates:', v.get('per_group_rates'))
    except Exception as e:
        print('  error reading axis details:', e)

