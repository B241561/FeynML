import sys
import os
import json

# Add project root to path
sys.path.append(os.path.abspath('.'))

from webapp.app import app, db
from webapp.models import User, Report, Dataset

def test_explorer_logic():
    print("--- Visual Explorer Forensic Test ---")
    
    with app.test_request_context():
        # 1. Test Dataset Lookup
        # Use a real report ID from the system
        report_id = 'report_1780648486'
        
        # We need a user context for @login_required but let's just test the helper _get_df
        from webapp.app import _get_df
        
        print(f"Testing _get_df with report_id={report_id}...")
        
        # Test with just report_id (should fallback)
        df, err = _get_df(report_id)
        if df is not None:
            print(f"SUCCESS: Loaded dataset with shape {df.shape}")
            print(f"Columns: {df.columns.tolist()[:5]}")
        else:
            print(f"FAILED: {err}")

        # 2. Test chart generation logic (mocking the API)
        # We'll call the logic parts directly to avoid auth issues in this script
        from webapp.app import generate_chart
        # Actually, let's just check if we can generate a Plotly fig with the loaded df
        import plotly.express as px
        
        if df is not None:
            cols = df.columns.tolist()
            x = cols[0]
            y = cols[1] if len(cols) > 1 else None
            
            print(f"Testing Histogram for {x}...")
            fig = px.histogram(df, x=x)
            print("SUCCESS: Histogram generated.")
            
            if y:
                print(f"Testing Scatter for {x} vs {y}...")
                fig = px.scatter(df, x=x, y=y)
                print("SUCCESS: Scatter generated.")
                
                print(f"Testing Bar for {x} vs {y}...")
                # Bar usually needs numeric y
                if 'float' in str(df[y].dtype) or 'int' in str(df[y].dtype):
                    fig = px.bar(df, x=x, y=y)
                    print("SUCCESS: Bar chart generated.")
                else:
                    print("SKIPPED: Bar chart (y is not numeric)")

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    test_explorer_logic()
