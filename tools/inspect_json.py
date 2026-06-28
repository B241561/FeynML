import json
import os

report_path = r'c:\Users\arman\OneDrive\Desktop\DATA ANALYTICS\ML_investigator\ml_failure_engine_reorganized\webapp\reports\report_1780648840.json'

with open(report_path, 'r') as f:
    report = json.load(f)

charts = report.get('charts', {})

for chart_name, chart_json in charts.items():
    print(f"=== {chart_name} ===")
    try:
        data = json.loads(chart_json)
        trace = data['data'][0]
        
        x = trace.get('x', [])
        y = trace.get('y', [])
        z = trace.get('z', [])
        
        print(f"x: {x[:5]} (len: {len(x)})")
        print(f"y: {y[:5]} (len: {len(y)})")
        if z:
            import numpy as np
            z_np = np.array(z)
            print(f"z shape: {z_np.shape}")
            print(f"z first 5: {z_np.flatten()[:5]}")
            
        # Check for serialization issues
        s_data = json.dumps(data)
        for issue in ["bdata", "ndarray", "float64", "int64", "dtype", "NaN", "Infinity"]:
            if issue in s_data:
                print(f"FOUND ISSUE: {issue}")
                
        # Check for nulls
        if any(v is None for v in x): print("x contains None")
        if any(v is None for v in y): print("y contains None")
        if z:
            if any(None in row for row in z): print("z contains None")
            
    except Exception as e:
        print(f"Error parsing {chart_name}: {e}")
    print("\n")
