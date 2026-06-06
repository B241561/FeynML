
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.modules.leakage_engine import LeakageEngine

def test_categorical_leakage():
    print("--- Testing Categorical Leakage ---")
    n = 100
    np.random.seed(42)
    
    # Create dataset with categorical strings
    df = pd.DataFrame({
        'gender': np.random.choice(['male', 'female'], n),
        'married': np.random.choice(['yes', 'no'], n),
        'target_copy': ['yes' if i > 0.5 else 'no' for i in np.random.rand(n)],
        'random_cat': np.random.choice(['A', 'B', 'C'], n)
    })
    
    # Target is derived from target_copy
    y = (df['target_copy'] == 'yes').astype(int)
    
    # Run LeakageEngine
    engine = LeakageEngine()
    X = df.drop(columns=['target_copy']) # This would be the input X
    
    # In reality, X is usually passed as values or dataframe
    try:
        result = engine.run(X, y, df=df, feature_names=X.columns.tolist())
        print(f"Severity: {result['severity']}")
        print(f"Suspects Found: {result['findings']['num_suspects']}")
        for s in result['findings']['suspects']:
            print(f"  - {s['feature']}: {s['type']} (Score: {s['score']:.4f}, Severity: {s['severity']})")
            
        if result['findings']['num_suspects'] > 0:
            print("\nSUCCESS: Categorical leakage detected without crash.")
        else:
            print("\nFAILURE: No leakage detected (check thresholds).")
            
    except Exception as e:
        print(f"\nCRASH DETECTED: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_categorical_leakage()
