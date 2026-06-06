"""
Multiple Imputation by Chained Equations (MICE)
==============================================
Implementation of MICE from scratch for missing data imputation.

Algorithm:
----------
1. Initial fill: replace NaNs with column mean/mode.
2. For each feature with missingness (in sequence):
   a. Treat it as target variable.
   b. Use all other features as predictors.
   c. Fit a model on rows where the feature is observed.
   d. Predict missing values for rows where it is missing.
3. Repeat step 2 for N cycles until convergence.
4. For Multiple Imputation, repeat the whole process M times with different 
   random initializations or by sampling from the predictive distribution.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

def initial_fill(X):
    """
    Perform initial mean/mode fill.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    X_filled = X.copy()
    for col in X.columns:
        if X[col].dtype.kind in 'iuf': # Numeric
            X_filled[col] = X[col].fillna(X[col].mean())
        else: # Categorical
            mode = X[col].mode()
            if not mode.empty:
                X_filled[col] = X[col].fillna(mode[0])
            else:
                X_filled[col] = X[col].fillna("MISSING")
                
    return X_filled

def mice_cycle(X, n_cycles=5, random_state=42):
    """
    Run MICE cycles to produce one completed dataset.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
        
    N, K = X.shape
    X_work = initial_fill(X)
    
    # Identify missing masks
    masks = {col: X[col].isnull() for col in X.columns if X[col].isnull().any()}
    if not masks:
        return X_work
        
    np.random.seed(random_state)
    
    for cycle in range(n_cycles):
        for target_col in masks.keys():
            mask = masks[target_col]
            
            # Observed rows for this column
            obs_mask = ~mask
            
            # Predictors: all columns except target
            predictors = [c for c in X.columns if c != target_col]
            
            X_train = X_work.loc[obs_mask, predictors]
            y_train = X_work.loc[obs_mask, target_col]
            
            X_miss = X_work.loc[mask, predictors]
            
            # Simple model: Linear for numeric, Logistic for categorical/discrete
            if X[target_col].dtype.kind in 'iuf' and len(X[target_col].unique()) > 10:
                model = LinearRegression()
                model.fit(X_train, y_train)
                # For multiple imputation, we should sample from the distribution,
                # but for this scratch version, we'll use deterministic predictions
                # plus some noise to simulate the distribution.
                preds = model.predict(X_miss)
                # Add residual noise
                rmse = np.sqrt(np.mean((y_train - model.predict(X_train))**2))
                preds += np.random.normal(0, rmse * 0.1, size=len(preds))
            else:
                # Handle categorical/binary
                # Simplified: use Logistic for binary, skip multiclass for now
                if len(y_train.unique()) <= 2:
                    model = LogisticRegression(max_iter=100)
                    model.fit(X_train, y_train.astype(str))
                    preds = model.predict(X_miss)
                else:
                    # Fallback to mean/mode if complex
                    preds = X_work[target_col].mean() if X[target_col].dtype.kind in 'iuf' else X_work[target_col].mode()[0]
            
            X_work.loc[mask, target_col] = preds
            
    return X_work

def mice_impute(X, m=5, n_cycles=5, random_state=42):
    """
    Generate m complete datasets using MICE.
    """
    datasets = []
    for i in range(m):
        ds = mice_cycle(X, n_cycles=n_cycles, random_state=random_state + i)
        datasets.append(ds)
    return datasets

def rubin_rules(estimates, variances):
    """
    Pool results from multiple datasets using Rubin's rules.
    """
    m = len(estimates)
    # 1. Combined Estimate (Mean)
    q_bar = np.mean(estimates)
    
    # 2. Within-imputation variance
    u_bar = np.mean(variances)
    
    # 3. Between-imputation variance
    b = np.sum((estimates - q_bar)**2) / (m - 1)
    
    # 4. Total variance
    t = u_bar + (1 + 1/m) * b
    
    return {
        "pooled_estimate": q_bar,
        "total_variance": t,
        "std_error": np.sqrt(t)
    }

def run_verification():
    """Run module verification."""
    print("--- MICE Imputation Verification ---")
    np.random.seed(42)
    N = 100
    X = np.random.randn(N, 3)
    # f2 depends on f0 and f1
    X[:, 2] = 0.5 * X[:, 0] + 0.8 * X[:, 1] + np.random.normal(0, 0.1, N)
    
    df = pd.DataFrame(X, columns=['f0', 'f1', 'f2'])
    
    # Inject missingness in f2
    df.loc[np.random.choice(N, 20), 'f2'] = np.nan
    
    print(f"Initial NaNs in f2: {df['f2'].isnull().sum()}")
    
    # Run MICE
    df_imputed = mice_cycle(df, n_cycles=10)
    print(f"NaNs after imputation: {df_imputed['f2'].isnull().sum()}")
    
    # Check correlation recovery
    orig_corr = np.corrcoef(X[:, 0], X[:, 2])[0, 1]
    imputed_corr = df_imputed['f0'].corr(df_imputed['f2'])
    
    print(f"Original f0-f2 Correlation: {orig_corr:.4f}")
    print(f"Imputed f0-f2 Correlation:  {imputed_corr:.4f}")

if __name__ == "__main__":
    run_verification()
