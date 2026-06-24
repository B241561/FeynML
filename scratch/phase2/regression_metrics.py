"""
Phase 2.2 — Regression Metrics
================================
"""
import math


def mse(y_true, y_pred):
    n = len(y_true)
    return sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n

def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(y_true[i] - y_pred[i]) for i in range(n)) / n

def mape(y_true, y_pred, eps=1e-8):
    n = len(y_true)
    return 100 * sum(abs(y_true[i] - y_pred[i]) / max(abs(y_true[i]), eps)
                     for i in range(n)) / n

def smape(y_true, y_pred, eps=1e-8):
    """Symmetric MAPE — handles near-zero targets better than MAPE."""
    n = len(y_true)
    return 100 * sum(2 * abs(y_true[i] - y_pred[i]) /
                     (abs(y_true[i]) + abs(y_pred[i]) + eps)
                     for i in range(n)) / n

def r2_score(y_true, y_pred):
    mean_y  = sum(y_true) / len(y_true)
    ss_res  = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true)))
    ss_tot  = sum((y - mean_y)**2 for y in y_true)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

def adjusted_r2(y_true, y_pred, n_features):
    n  = len(y_true)
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

def residual_analysis(y_true, y_pred):
    residuals = [y_true[i] - y_pred[i] for i in range(len(y_true))]
    mean_res  = sum(residuals) / len(residuals)
    std_res   = math.sqrt(sum((r - mean_res)**2 for r in residuals) / (len(residuals)-1))
    return {
        "residuals":        residuals,
        "mean_residual":    round(mean_res, 6),
        "std_residual":     round(std_res, 6),
        "max_error":        round(max(abs(r) for r in residuals), 6),
        "bias_detected":    abs(mean_res) > 2 * std_res / math.sqrt(len(residuals)),
    }

def regression_summary(y_true, y_pred, n_features=1):
    return {
        "mse":          round(mse(y_true, y_pred),  6),
        "rmse":         round(rmse(y_true, y_pred), 6),
        "mae":          round(mae(y_true, y_pred),  6),
        "mape":         round(mape(y_true, y_pred), 4),
        "smape":        round(smape(y_true, y_pred),4),
        "r2":           round(r2_score(y_true, y_pred), 6),
        "adj_r2":       round(adjusted_r2(y_true, y_pred, n_features), 6),
        **{k: v for k, v in residual_analysis(y_true, y_pred).items() if k != "residuals"},
    }

def run_verification():
    import sklearn.metrics as skm, random
    random.seed(0)
    n = 300
    y_t = [2*random.random() + x*0.5 for x in range(n)]
    y_p = [y + random.gauss(0, 0.3) for y in y_t]

    checks = [
        ("MSE",   mse(y_t,y_p),   skm.mean_squared_error(y_t,y_p)),
        ("MAE",   mae(y_t,y_p),   skm.mean_absolute_error(y_t,y_p)),
        ("R²",    r2_score(y_t,y_p), skm.r2_score(y_t,y_p)),
    ]
    print("=" * 55)
    print("Phase 2.2 — Regression Metrics Verification")
    print("=" * 55)
    for name, ours, ref in checks:
        ok = abs(ours - ref) < 1e-4
        print(f"  {name:<6} ours={ours:.5f}  sklearn={ref:.5f}  [{'✓ PASS' if ok else '✗ FAIL'}]")
    print()


if __name__ == "__main__":
    run_verification()


# Compatibility layer: expose sklearn-like function names expected by tests
mean_squared_error = mse
root_mean_squared_error = rmse
mean_absolute_error = mae
r_squared = r2_score


def mean_absolute_percentage_error(y_true, y_pred, eps=1e-8):
    """Return MAPE as a floating-point percentage (keeps original behaviour)."""
    return mape(y_true, y_pred, eps=eps)


def median_absolute_error(y_true, y_pred):
    errors = sorted(abs(a - b) for a, b in zip(y_true, y_pred))
    n = len(errors)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return errors[n // 2]
    return 0.5 * (errors[n // 2 - 1] + errors[n // 2])


def max_error(y_true, y_pred):
    return max(abs(a - b) for a, b in zip(y_true, y_pred)) if y_true else 0.0


def explained_variance(y_true, y_pred):
    """Explained variance = 1 - Var(residuals) / Var(y_true).

    Returns 1.0 when variance of y_true is zero (perfect/explained by bias).
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    mean_y = sum(y_true) / n
    var_y = sum((yi - mean_y) ** 2 for yi in y_true) / n
    residuals = [y_true[i] - y_pred[i] for i in range(n)]
    mean_res = sum(residuals) / n
    var_res = sum((r - mean_res) ** 2 for r in residuals) / n
    if var_y == 0:
        return 1.0
    return 1.0 - (var_res / var_y)
