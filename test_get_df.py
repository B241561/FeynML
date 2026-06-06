import os
import pandas as pd
import glob
import json

UPLOAD_FOLDER = r'c:\Users\arman\OneDrive\Desktop\DATA ANALYTICS\ML_investigator\ml_failure_engine_reorganized\webapp\uploads'
_df_cache = {}

def _get_df(report_id: str, x_col: str = None, dataset_path: str = None):
    if report_id in _df_cache:
        return _df_cache[report_id], None

    if dataset_path and os.path.exists(dataset_path):
        try:
            if dataset_path.endswith('.csv'):
                df = pd.read_csv(dataset_path)
            else:
                try:
                    df = pd.read_json(dataset_path, lines=True)
                except Exception:
                    df = pd.read_json(dataset_path)
            print(f"[_get_df] ✓ LOADED from provided path → {os.path.basename(dataset_path)} shape={df.shape}")
            _df_cache[report_id] = df
            return df, None
        except Exception as exc:
            print(f"[_get_df] ⚠ Failed to load from provided path {dataset_path}: {exc}")

    csv_candidates = glob.glob(os.path.join(UPLOAD_FOLDER, '*.csv'))
    json_candidates = glob.glob(os.path.join(UPLOAD_FOLDER, '*.json'))
    all_candidates = csv_candidates + json_candidates

    if dataset_path:
        filename = os.path.basename(dataset_path)
        for cand in all_candidates:
            if os.path.basename(cand) == filename:
                try:
                    if cand.endswith('.csv'):
                        df = pd.read_csv(cand)
                    else:
                        try:
                            df = pd.read_json(cand, lines=True)
                        except Exception:
                            df = pd.read_json(cand)
                    print(f"[_get_df] ✓ RECOVERED by filename → {os.path.basename(cand)} shape={df.shape}")
                    _df_cache[report_id] = df
                    return df, None
                except Exception:
                    continue

    print(f"\n[_get_df] report_id={repr(report_id)}  x_col hint={repr(x_col)}")
    if not all_candidates:
        return None, "No dataset files found."

    if x_col:
        for path in sorted(all_candidates):
            try:
                if path.endswith('.csv'):
                    headers = pd.read_csv(path, nrows=0).columns.tolist()
                else:
                    headers = pd.read_json(path, lines=True, nrows=0).columns.tolist()
                if x_col in headers:
                    df = pd.read_csv(path) if path.endswith('.csv') else pd.read_json(path, lines=True)
                    print(f"[_get_df] ✓ MATCHED '{x_col}' → {os.path.basename(path)}")
                    return df, None
            except Exception:
                continue

    best_path = max(all_candidates, key=os.path.getsize)
    df = pd.read_csv(best_path) if best_path.endswith('.csv') else pd.read_json(best_path, lines=True)
    print(f"[_get_df] ⚠ FALLBACK → {os.path.basename(best_path)}")
    return df, None

if __name__ == "__main__":
    print("Testing standalone _get_df...")
    df, err = _get_df('test_report', x_col='species')
    if df is not None:
        print(f"Final shape: {df.shape}")
    else:
        print(f"Error: {err}")
