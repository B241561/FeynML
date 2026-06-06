import pandas as pd
import plotly.express as px
import numpy as np

def verify_charts():
    # Sample data
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': ['Type1', 'Type2'] * 50,
        'D': np.random.randint(1, 10, 100)
    })
    
    x_axis = 'A'
    y_axis = 'B'
    color = 'C'
    
    print("Testing Histogram...")
    try:
        fig = px.histogram(df, x=x_axis, color=color, nbins=30)
        print("✓ Histogram OK")
    except Exception as e:
        print(f"✗ Histogram FAILED: {e}")

    print("Testing Scatter...")
    try:
        fig = px.scatter(df, x=x_axis, y=y_axis, color=color)
        print("✓ Scatter OK")
    except Exception as e:
        print(f"✗ Scatter FAILED: {e}")

    print("Testing Bar...")
    try:
        # Grouped bar logic from app.py
        grouped = df.groupby(x_axis)[y_axis].mean().reset_index()
        fig = px.bar(grouped, x=x_axis, y=y_axis)
        print("✓ Bar OK")
    except Exception as e:
        print(f"✗ Bar FAILED: {e}")

    print("Testing Box Plot...")
    try:
        fig = px.box(df, x=x_axis, y=y_axis, color=color)
        print("✓ Box Plot OK")
    except Exception as e:
        print(f"✗ Box Plot FAILED: {e}")

if __name__ == "__main__":
    verify_charts()
