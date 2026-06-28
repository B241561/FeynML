import pandas as pd
import plotly.express as px
import numpy as np

def verify_one():
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print("Testing simple Histogram...")
    try:
        fig = px.histogram(df, x='A')
        print("✓ Histogram OK")
    except Exception as e:
        print(f"✗ Histogram FAILED: {e}")

if __name__ == "__main__":
    verify_one()
