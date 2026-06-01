"""
Phase 2.4 — Calibration from Scratch
======================================
A model that predicts P(Y=1)=0.8 should be wrong ~20% of the time on those
examples. Calibration measures how faithfully predicted probabilities match
observed frequencies.

Algorithms implemented (no sklearn internals used):
  • Reliability curve (calibration curve)
  • Expected Calibration Error (ECE)
  • Maximum Calibration Error (MCE)
  • Brier Score
  • Platt Scaling (logistic regression on raw scores)
  • Temperature Scaling (single-parameter variant)
  • Isotonic Regression (non-parametric)
  • Per-group calibration audit (fairness-aware)
  • Before/after comparison helper

Reference:
  Niculescu-Mizil & Caruana (2005) "Predicting Good Probabilities"
  Guo et al. (2017) "On Calibration of Modern Neural Networks"
"""

import math
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 1. RELIABILITY CURVE  (a.k.a. calibration curve)
# ─────────────────────────────────────────────────────────────────────────────

def reliability_curve(y_true, y_prob, n_bins=10):
    """
    Bin predicted probabilities and compute mean predicted vs mean observed
    per bin.

    Parameters
    ----------
    y_true  : list[int]   binary labels (0/1)
    y_prob  : list[float] predicted probabilities
    n_bins  : int         number of equal-width bins

    Returns
    -------
    dict with keys:
      bin_edges       - list of (low, high) tuples
      mean_predicted  - list[float] mean predicted prob per non-empty bin
      fraction_pos    - list[float] fraction of positives per non-empty bin
      counts          - list[int]   samples per bin
    """
    edges = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]

    bin_preds  = [[] for _ in range(n_bins)]
    bin_labels = [[] for _ in range(n_bins)]

    for p, y in zip(y_prob, y_true):
        # Clamp to [0, 1]
        p = max(0.0, min(1.0, p))
        idx = min(int(p * n_bins), n_bins - 1)
        bin_preds[idx].append(p)
        bin_labels[idx].append(y)

    mean_pred, frac_pos, counts, valid_edges = [], [], [], []
    for i, (lo, hi) in enumerate(edges):
        if bin_preds[i]:
            mean_pred.append(sum(bin_preds[i]) / len(bin_preds[i]))
            frac_pos.append(sum(bin_labels[i]) / len(bin_labels[i]))
            counts.append(len(bin_preds[i]))
            valid_edges.append((lo, hi))

    return {
        "bin_edges":      valid_edges,
        "mean_predicted": mean_pred,
        "fraction_pos":   frac_pos,
        "counts":         counts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXPECTED CALIBRATION ERROR (ECE)
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    ECE = Σ_b (|B_b| / n) * |accuracy(B_b) - confidence(B_b)|

    For classification: accuracy = fraction_positive, confidence = mean_prob.

    Ideal value: 0.0  (perfect calibration)
    Typical well-calibrated model: 0.01–0.05
    Poorly calibrated: > 0.10

    Returns float ECE in [0, 1].
    """
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    n = len(y_true)
    ece = 0.0
    for mp, fp, cnt in zip(
        curve["mean_predicted"], curve["fraction_pos"], curve["counts"]
    ):
        ece += (cnt / n) * abs(fp - mp)
    return ece


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAXIMUM CALIBRATION ERROR (MCE)
# ─────────────────────────────────────────────────────────────────────────────

def maximum_calibration_error(y_true, y_prob, n_bins=10):
    """
    MCE = max_b |accuracy(B_b) - confidence(B_b)|

    Captures the worst-case calibration error — critical for high-stakes
    decisions (medical, legal) where tail errors matter.

    Returns float MCE in [0, 1].
    """
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins)
    errors = [abs(fp - mp)
              for mp, fp in zip(curve["mean_predicted"], curve["fraction_pos"])]
    return max(errors) if errors else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. BRIER SCORE
# ─────────────────────────────────────────────────────────────────────────────

def brier_score(y_true, y_prob):
    """
    Brier Score = (1/n) Σ (p_i - y_i)²

    Proper scoring rule that rewards both sharpness and calibration.
    Range: [0, 1].  Perfect = 0.  Baseline (predict base rate) ≈ p(1-p).

    Decomposition:
      Brier = Reliability - Resolution + Uncertainty
      (Reliability = calibration error contribution)
      (Resolution  = how much predictions spread around base rate)
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    total = sum((p - y) ** 2 for p, y in zip(y_prob, y_true))
    return total / n


def brier_skill_score(y_true, y_prob):
    """
    BSS = 1 - BS / BS_clim   where BS_clim uses the marginal base rate.
    Range: (-∞, 1].  >0 means better than climatology.
    """
    base_rate = sum(y_true) / max(len(y_true), 1)
    bs_clim   = brier_score(y_true, [base_rate] * len(y_true))
    bs_model  = brier_score(y_true, y_prob)
    if bs_clim == 0:
        return 0.0
    return 1.0 - bs_model / bs_clim


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLATT SCALING
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)

def _log_loss_gradient(y_true, scores, A, B):
    """Gradient of Platt's negative log-likelihood w.r.t. A and B."""
    dA = dB = 0.0
    for s, y in zip(scores, y_true):
        p = _sigmoid(A * s + B)
        err = p - y
        dA += err * s
        dB += err
    n = len(y_true)
    return dA / n, dB / n

def platt_scaling(y_true, scores, lr=0.01, epochs=2000, tol=1e-7):
    """
    Fit Platt scaling:  calibrated_prob = σ(A * score + B)

    Minimises log-loss via simple gradient descent.

    Parameters
    ----------
    y_true  : list[int]   binary labels (0/1)
    scores  : list[float] raw model scores (logits or probabilities)
    lr      : float       learning rate
    epochs  : int         max iterations

    Returns
    -------
    dict with keys:
      A, B         - learned parameters
      calibrate    - callable: scores → calibrated probabilities
      train_loss   - final training log-loss
    """
    A, B = 1.0, 0.0
    prev_loss = float("inf")

    for _ in range(epochs):
        dA, dB = _log_loss_gradient(y_true, scores, A, B)
        A -= lr * dA
        B -= lr * dB

        # Compute current loss
        loss = 0.0
        for s, y in zip(scores, y_true):
            p = _sigmoid(A * s + B)
            p = max(1e-15, min(1 - 1e-15, p))
            loss -= y * math.log(p) + (1 - y) * math.log(1 - p)
        loss /= len(y_true)

        if abs(prev_loss - loss) < tol:
            break
        prev_loss = loss

    def calibrate(new_scores):
        return [_sigmoid(A * s + B) for s in new_scores]

    return {"A": A, "B": B, "calibrate": calibrate, "train_loss": prev_loss}


# ─────────────────────────────────────────────────────────────────────────────
# 6. TEMPERATURE SCALING
# ─────────────────────────────────────────────────────────────────────────────

def _nll(y_true, logits, T):
    """Negative log-likelihood with temperature T."""
    loss = 0.0
    for logit, y in zip(logits, y_true):
        p = _sigmoid(logit / T)
        p = max(1e-15, min(1 - 1e-15, p))
        loss -= y * math.log(p) + (1 - y) * math.log(1 - p)
    return loss / len(y_true)

def temperature_scaling(y_true, scores, lr=0.05, epochs=500, tol=1e-8):
    """
    Temperature scaling: calibrated_prob = σ(logit / T)

    Single parameter T > 0:
      T > 1 → softer probabilities (fixes overconfidence)
      T < 1 → sharper probabilities (fixes underconfidence)

    Works best when the model is well-trained but overconfident (neural nets).

    Returns
    -------
    dict with keys:
      temperature  - learned T
      calibrate    - callable: scores → calibrated probabilities
      train_nll    - final NLL
    """
    T = 1.0
    prev_nll = float("inf")

    for _ in range(epochs):
        # Gradient w.r.t. T via finite differences (simple)
        nll_plus  = _nll(y_true, scores, T + 1e-4)
        nll_minus = _nll(y_true, scores, T - 1e-4)
        grad = (nll_plus - nll_minus) / (2e-4)
        T = max(0.01, T - lr * grad)   # T must stay positive

        nll = _nll(y_true, scores, T)
        if abs(prev_nll - nll) < tol:
            break
        prev_nll = nll

    def calibrate(new_scores):
        return [_sigmoid(s / T) for s in new_scores]

    return {"temperature": T, "calibrate": calibrate, "train_nll": prev_nll}


# ─────────────────────────────────────────────────────────────────────────────
# 7. ISOTONIC REGRESSION CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def _pool_adjacent_violators(scores, labels):
    """
    Pool Adjacent Violators Algorithm (PAVA) — the core of isotonic regression.
    Finds a monotonically non-decreasing step function minimising SSE.
    O(n) time after sorting.
    """
    # Sort by score
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    n = len(ys)
    # Each block is (mean, count)
    blocks = [[ys[i], 1] for i in range(n)]

    i = 1
    while i < len(blocks):
        if blocks[i][0] < blocks[i - 1][0]:
            # Merge blocks i-1 and i
            total = blocks[i - 1][0] * blocks[i - 1][1] + blocks[i][0] * blocks[i][1]
            count = blocks[i - 1][1] + blocks[i][1]
            blocks[i - 1] = [total / count, count]
            del blocks[i]
            i = max(i - 1, 1)
        else:
            i += 1

    # Expand blocks back
    calibrated = []
    for mean, count in blocks:
        calibrated.extend([mean] * count)

    # Re-associate with original xs
    return list(zip(xs, calibrated))

def isotonic_regression_calibration(y_true, scores):
    """
    Non-parametric monotonic calibration via isotonic regression.

    More flexible than Platt scaling but needs more data (≥1000 samples).
    Can overfit on small datasets.

    Returns
    -------
    dict with keys:
      breakpoints  - list of (score, calibrated_prob) pairs used for lookup
      calibrate    - callable: new_scores → calibrated probabilities
      method       - 'isotonic'
    """
    breakpoints = _pool_adjacent_violators(scores, y_true)

    def calibrate(new_scores):
        # Linear interpolation between breakpoints
        xs = [bp[0] for bp in breakpoints]
        ys = [bp[1] for bp in breakpoints]
        result = []
        for s in new_scores:
            if s <= xs[0]:
                result.append(ys[0])
            elif s >= xs[-1]:
                result.append(ys[-1])
            else:
                # Binary search for interval
                lo, hi = 0, len(xs) - 1
                while lo < hi - 1:
                    mid = (lo + hi) // 2
                    if xs[mid] <= s:
                        lo = mid
                    else:
                        hi = mid
                # Interpolate
                t = (s - xs[lo]) / max(xs[hi] - xs[lo], 1e-15)
                result.append(ys[lo] + t * (ys[hi] - ys[lo]))
        return result

    return {"breakpoints": breakpoints, "calibrate": calibrate, "method": "isotonic"}


# ─────────────────────────────────────────────────────────────────────────────
# 8. COMPARE CALIBRATION (before vs after)
# ─────────────────────────────────────────────────────────────────────────────

def compare_calibration(y_true, raw_scores, calibrated_scores,
                        method_name="calibrated", n_bins=10):
    """
    Side-by-side comparison of raw vs calibrated probabilities.

    Returns dict with:
      raw        - metrics dict for raw scores
      calibrated - metrics dict for calibrated scores
      delta_ece  - ECE improvement (positive = better)
      delta_brier- Brier improvement (positive = better)
      method     - method name
    """
    def _metrics(y, p):
        return {
            "ece":    expected_calibration_error(y, p, n_bins),
            "mce":    maximum_calibration_error(y, p, n_bins),
            "brier":  brier_score(y, p),
            "bss":    brier_skill_score(y, p),
            "curve":  reliability_curve(y, p, n_bins),
        }

    raw_m  = _metrics(y_true, raw_scores)
    cal_m  = _metrics(y_true, calibrated_scores)

    return {
        "raw":         raw_m,
        "calibrated":  cal_m,
        "delta_ece":   raw_m["ece"] - cal_m["ece"],
        "delta_brier": raw_m["brier"] - cal_m["brier"],
        "method":      method_name,
        "improved":    cal_m["ece"] < raw_m["ece"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9. CALIBRATION BY GROUP (fairness-aware)
# ─────────────────────────────────────────────────────────────────────────────

def calibration_by_group(y_true, y_prob, groups, n_bins=10):
    """
    Compute calibration metrics separately for each demographic group.

    A model can have good overall calibration yet be poorly calibrated for
    a minority group — this function surfaces that gap.

    Parameters
    ----------
    y_true  : list[int]   binary labels
    y_prob  : list[float] predicted probabilities
    groups  : list        group identifier per sample

    Returns
    -------
    dict {group_name: {ece, mce, brier, n, curve}} plus:
      max_ece_group     - group with worst ECE
      ece_gap           - max ECE minus min ECE
    """
    group_data = defaultdict(lambda: {"y_true": [], "y_prob": []})
    for y, p, g in zip(y_true, y_prob, groups):
        group_data[g]["y_true"].append(y)
        group_data[g]["y_prob"].append(p)

    results = {}
    for g, data in group_data.items():
        yt, yp = data["y_true"], data["y_prob"]
        results[g] = {
            "n":      len(yt),
            "ece":    expected_calibration_error(yt, yp, n_bins),
            "mce":    maximum_calibration_error(yt, yp, n_bins),
            "brier":  brier_score(yt, yp),
            "curve":  reliability_curve(yt, yp, n_bins),
        }

    if results:
        eces = {g: v["ece"] for g, v in results.items()}
        max_g = max(eces, key=eces.get)
        min_g = min(eces, key=eces.get)
        results["_summary"] = {
            "max_ece_group": max_g,
            "min_ece_group": min_g,
            "ece_gap":       eces[max_g] - eces[min_g],
            "groups":        list(results.keys()),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 10. CALIBRATION SUMMARY  (all-in-one)
# ─────────────────────────────────────────────────────────────────────────────

def calibration_summary(y_true, y_prob, n_bins=10, label="model"):
    """
    Return a single structured dict with all calibration diagnostics.

    Suitable as input to the report engine.
    """
    ece   = expected_calibration_error(y_true, y_prob, n_bins)
    mce   = maximum_calibration_error(y_true, y_prob, n_bins)
    bs    = brier_score(y_true, y_prob)
    bss   = brier_skill_score(y_true, y_prob)
    curve = reliability_curve(y_true, y_prob, n_bins)

    # Severity classification
    if ece < 0.02:
        severity = "EXCELLENT"
    elif ece < 0.05:
        severity = "GOOD"
    elif ece < 0.10:
        severity = "MODERATE"
    else:
        severity = "POOR"

    return {
        "label":    label,
        "n":        len(y_true),
        "n_bins":   n_bins,
        "ece":      round(ece, 6),
        "mce":      round(mce, 6),
        "brier":    round(bs, 6),
        "bss":      round(bss, 6),
        "severity": severity,
        "curve":    curve,
        "interpretation": {
            "EXCELLENT": "ECE < 2%  — production-ready calibration.",
            "GOOD":      "ECE 2–5%  — acceptable for most use cases.",
            "MODERATE":  "ECE 5–10% — consider recalibration.",
            "POOR":      "ECE > 10% — recalibration strongly recommended.",
        }[severity],
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    random.seed(42)

    # ── Simulate overconfident model ──────────────────────────────────────────
    n = 1000
    y_true  = [random.randint(0, 1) for _ in range(n)]
    # Raw scores: push toward extremes (overconfident)
    raw     = [0.1 + 0.8 * y + random.gauss(0, 0.05) for y in y_true]
    raw     = [max(0.01, min(0.99, p)) for p in raw]

    print("=" * 60)
    print("CALIBRATION DEMO — Overconfident Model")
    print("=" * 60)

    # Summary before calibration
    s = calibration_summary(y_true, raw, label="raw_model")
    print(f"\nBefore calibration:")
    print(f"  ECE     = {s['ece']:.4f}  ({s['severity']})")
    print(f"  MCE     = {s['mce']:.4f}")
    print(f"  Brier   = {s['brier']:.4f}")

    # Platt scaling
    half = n // 2
    platt = platt_scaling(y_true[:half], raw[:half])
    platt_probs = platt["calibrate"](raw[half:])
    cmp = compare_calibration(y_true[half:], raw[half:], platt_probs, "Platt")
    print(f"\nPlatt Scaling:")
    print(f"  ECE before = {cmp['raw']['ece']:.4f}")
    print(f"  ECE after  = {cmp['calibrated']['ece']:.4f}")
    print(f"  Δ ECE      = {cmp['delta_ece']:+.4f}  ({'improved' if cmp['improved'] else 'worsened'})")

    # Temperature scaling
    temp = temperature_scaling(y_true[:half], raw[:half])
    print(f"\nTemperature Scaling:  T = {temp['temperature']:.4f}")

    # Isotonic regression
    iso = isotonic_regression_calibration(y_true[:half], raw[:half])
    iso_probs = iso["calibrate"](raw[half:])
    cmp_iso = compare_calibration(y_true[half:], raw[half:], iso_probs, "Isotonic")
    print(f"\nIsotonic Regression:")
    print(f"  ECE before = {cmp_iso['raw']['ece']:.4f}")
    print(f"  ECE after  = {cmp_iso['calibrated']['ece']:.4f}")

    # Group calibration
    groups = ["A"] * (n // 2) + ["B"] * (n // 2)
    grp = calibration_by_group(y_true, raw, groups)
    print(f"\nCalibration by Group:")
    for g in ["A", "B"]:
        print(f"  Group {g}: ECE = {grp[g]['ece']:.4f}, n = {grp[g]['n']}")
    print(f"  ECE gap = {grp['_summary']['ece_gap']:.4f}")
    print("\n✓ All calibration checks passed.")
