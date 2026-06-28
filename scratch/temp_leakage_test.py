import importlib, sys, types
# Import module
le = importlib.import_module('engine.modules.leakage_engine')
# Ensure module-level functions exist for testing
le._MODULES_LOADED = True
# Mock scan, rank_leakage_suspects, leakage_summary

def mock_scan(X,y,df,model,feature_names,time_col,train_idx,test_idx):
    return {'mock':'report'}

def mock_rank(report):
    # Create two suspects with high scores
    return [
        {'feature':'ApprovalFlag','score':0.882,'type':'target_leakage','severity':'MEDIUM'},
        {'feature':'LoanApproved','score':0.882,'type':'target_leakage','severity':'MEDIUM'}
    ]

def mock_summary(report):
    return {'n':2}

le.scan = mock_scan
le.rank_leakage_suspects = mock_rank
le.leakage_summary = mock_summary

eng = le.LeakageEngine()
# Call _run
res = eng._run(None, None, df=None, model=None, feature_names=None, time_col=None, train_idx=None, test_idx=None)
print('Returned severity:', res.get('severity'))
print('Findings keys:', list(res.get('result', {}).keys()) if isinstance(res, dict) else res)
print('Full result:', res)
