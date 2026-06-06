import json

report_path = r'c:\Users\arman\OneDrive\Desktop\DATA ANALYTICS\ML_investigator\ml_failure_engine_reorganized\webapp\reports\report_1780648840.json'

with open(report_path, 'r') as f:
    report = json.load(f)

charts = report.get('charts', {})

def compare_layouts(c1_name, c2_name):
    print(f"Comparing {c1_name} vs {c2_name}")
    d1 = json.loads(charts[c1_name])
    d2 = json.loads(charts[c2_name])
    
    l1 = d1.get('layout', {})
    l2 = d2.get('layout', {})
    
    # Check for template presence
    print(f" - {c1_name} has template: {'template' in l1}")
    print(f" - {c2_name} has template: {'template' in l2}")
    
    # Check for axes
    print(f" - {c1_name} xaxis: {l1.get('xaxis')}")
    print(f" - {c2_name} xaxis: {l2.get('xaxis')}")
    
    # Check for margins
    print(f" - {c1_name} margin: {l1.get('margin')}")
    print(f" - {c2_name} margin: {l2.get('margin')}")

compare_layouts('correlation_heatmap', 'ks_ranked')
compare_layouts('correlation_heatmap', 'psi_heatmap')
compare_layouts('correlation_heatmap', 'leakage_scores')
