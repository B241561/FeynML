import os
import sys
import pandas as pd
print("Python Version:", sys.version)
print("Pandas Version:", pd.__version__)
print("Current Directory:", os.getcwd())
try:
    import plotly
    print("Plotly Version:", plotly.__version__)
except ImportError:
    print("Plotly not found")
