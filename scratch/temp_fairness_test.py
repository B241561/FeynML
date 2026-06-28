from engine.modules.fairness_engine import FairnessEngine
import json

# create balanced data
n=200
# half privileged 'A', half 'B'
groups = ['A']*(n//2)+['B']*(n//2)
# y_true balanced half ones
y_true = [1]*(n//2)+[0]*(n//2)
# predictions mirror true labels to avoid bias
y_pred = y_true.copy()
fe = FairnessEngine()
fe.register_axis('group', groups)
report = fe.run(y_true, y_pred, y_prob=[0.9 if y==1 else 0.1 for y in y_true])
print(json.dumps(report, indent=2))
