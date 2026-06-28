from engine.modules.drift_engine import DriftEngine
from engine.modules.root_cause_engine import AutoRootCauseEngine
import numpy as np
import pandas as pd

# create synthetic datasets similar to comprehensive_validation
np.random.seed(42)
N=500
student_ids = list(range(1000,1000+N))
attendance_train = np.random.normal(85,10,N)
study_time_train = np.random.normal(10,3,N)
grade_train = np.random.normal(85,10,N)
midterm_train = grade_train + np.random.normal(0,5,N)

target_train = (attendance_train*0.3 + study_time_train*0.2 + grade_train*0.5 + np.random.normal(0,5,N))
target_train = (target_train - target_train.min()) / (target_train.max()-target_train.min())*100

train = pd.DataFrame({
    'StudentID': student_ids,
    'Attendance': attendance_train,
    'StudyTime': study_time_train,
    'Grade': grade_train,
    'MidtermScore': midterm_train,
    'FinalGrade': target_train
})

# production drift
student_ids_prod = list(range(2000,2000+N))
attendance_prod = np.random.normal(75,12,N)
study_time_prod = np.random.normal(8,4,N)
grade_prod = np.random.normal(80,12,N)
midterm_prod = grade_prod + np.random.normal(0,5,N)

target_prod = (attendance_prod*0.3 + study_time_prod*0.2 + grade_prod*0.5 + np.random.normal(0,5,N))
target_prod = (target_prod - target_prod.min()) / (target_prod.max()-target_prod.min())*100

prod = pd.DataFrame({
    'StudentID': student_ids_prod,
    'Attendance': attendance_prod,
    'StudyTime': study_time_prod,
    'Grade': grade_prod,
    'MidtermScore': midterm_prod,
    'FinalGrade': target_prod
})

feature_names = ['StudentID','Attendance','StudyTime','Grade','MidtermScore']

d = DriftEngine(verbose=False)
d.set_reference(train[feature_names].values.tolist(), feature_names)
drift_result = d.run(prod[feature_names].values.tolist())

arce = AutoRootCauseEngine(verbose=False)
root_cause_result = arce.run(drift_report=drift_result)

import json
print('--- risk in result ---')
print(json.dumps(root_cause_result.get('risk', {}), indent=2))
print('--- risk_explanation ---')
print(root_cause_result.get('risk_explanation'))
print('--- investigation.metadata.risk_explanation ---')
# Build investigation object from dict to inspect metadata
from engine.modules.investigation import Investigation
inv = Investigation.from_dict(root_cause_result)
print(inv.metadata.get('risk_explanation'))

print('--- fairness in result ---')
print(json.dumps(root_cause_result.get('fairness', {}), indent=2))
