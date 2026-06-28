import sys, types
# Create fake sklearn package with required submodules and symbols
sklearn = types.ModuleType('sklearn')
sklearn.ensemble = types.ModuleType('sklearn.ensemble')
sklearn.ensemble.RandomForestClassifier = type('RFC', (), {})
sklearn.ensemble.RandomForestRegressor = type('RFR', (), {})
sklearn.model_selection = types.ModuleType('sklearn.model_selection')
sklearn.model_selection.cross_val_predict = lambda *a, **k: []
sklearn.preprocessing = types.ModuleType('sklearn.preprocessing')
sklearn.preprocessing.LabelEncoder = type('LabelEncoder', (), {})
# Insert into sys.modules
sys.modules['sklearn'] = sklearn
sys.modules['sklearn.ensemble'] = sklearn.ensemble
sys.modules['sklearn.model_selection'] = sklearn.model_selection
sys.modules['sklearn.preprocessing'] = sklearn.preprocessing

# Now import the module
import importlib
m = importlib.import_module('webapp.services.analysis_runner')
print('analysis_runner imported OK')
