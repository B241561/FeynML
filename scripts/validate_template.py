import sys
try:
    import jinja2
except Exception:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'jinja2'])
    import jinja2
p = r"C:\Users\arman\OneDrive\Desktop\DATA ANALYTICS\ML_investigator\ml_failure_engine_reorganized\webapp\templates\dashboard.html"
with open(p, 'r', encoding='utf-8') as f:
    s = f.read()
env = jinja2.Environment()
try:
    env.parse(s)
    print('JINJA_PARSE_OK')
except Exception as e:
    print('JINJA_PARSE_ERROR')
    print(e)
    sys.exit(1)
