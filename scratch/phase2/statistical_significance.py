"""
=============================================================================
Phase 2 — Statistical Significance Testing (From Scratch)
=============================================================================
Topics Covered:
  1. z-test (one-sample, two-sample)
  2. t-test (one-sample, two-sample, paired)
  3. Chi-Square Test (goodness of fit, independence)
  4. Mann-Whitney U Test (non-parametric)
  5. Bootstrap Confidence Intervals
  6. Multiple Testing Correction (Bonferroni, FDR/BH)
  7. Effect Size (Cohen's d, Cohen's h, Cramér's V)
  8. Power Analysis
  9. Practical vs Statistical Significance
 10. Model Comparison Significance (McNemar's Test)
=============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Optional, Dict
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def _validate_array(arr, name="array"):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if len(arr) < 2:
        raise ValueError(f"{name} must have at least 2 elements")
    return arr


def print_separator(title=""):
    width = 65
    if title:
        side = (width - len(title) - 2) // 2
        print("\n" + "=" * side + f" {title} " + "=" * side)
    else:
        print("\n" + "=" * width)


def interpret_pvalue(p: float, alpha: float = 0.05) -> str:
    if p < 0.001:
        return f"p={p:.4f} *** (Extremely significant at α={alpha})"
    elif p < 0.01:
        return f"p={p:.4f} **  (Very significant at α={alpha})"
    elif p < alpha:
        return f"p={p:.4f} *   (Significant at α={alpha})"
    else:
        return f"p={p:.4f}     (NOT significant at α={alpha})"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Z-TEST (FROM SCRATCH)
# ─────────────────────────────────────────────────────────────────────────────

class ZTest:
    """
    Z-Test — use when:
      • Population std is KNOWN, OR
      • Sample size n >= 30 (CLT kicks in)

    H0: mu = mu0  (one-sample)
    H0: mu1 = mu2 (two-sample)
    """

    @staticmethod
    def one_sample(sample: np.ndarray, mu0: float,
                   sigma: float, alternative: str = "two-sided") -> dict:
        """
        One-sample z-test.
        alternative: 'two-sided' | 'less' | 'greater'
        """
        x = _validate_array(sample)
        n = len(x)
        x_bar = np.mean(x)
        se = sigma / np.sqrt(n)           # Standard Error
        z_stat = (x_bar - mu0) / se       # Z-statistic

        # p-value from standard normal CDF
        if alternative == "two-sided":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == "greater":
            p_value = 1 - stats.norm.cdf(z_stat)
        elif alternative == "less":
            p_value = stats.norm.cdf(z_stat)
        else:
            raise ValueError("alternative must be 'two-sided','less','greater'")

        return {
            "test": "One-Sample Z-Test",
            "n": n, "x_bar": round(x_bar, 4),
            "mu0": mu0, "sigma": sigma,
            "z_stat": round(z_stat, 4),
            "p_value": round(p_value, 6),
            "alternative": alternative,
            "reject_H0": p_value < 0.05
        }

    @staticmethod
    def two_sample(sample1: np.ndarray, sample2: np.ndarray,
                   sigma1: float, sigma2: float,
                   alternative: str = "two-sided") -> dict:
        """Two-sample z-test for difference of means."""
        x1 = _validate_array(sample1, "sample1")
        x2 = _validate_array(sample2, "sample2")
        n1, n2 = len(x1), len(x2)
        diff = np.mean(x1) - np.mean(x2)
        se = np.sqrt(sigma1**2 / n1 + sigma2**2 / n2)
        z_stat = diff / se

        if alternative == "two-sided":
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alternative == "greater":
            p_value = 1 - stats.norm.cdf(z_stat)
        else:
            p_value = stats.norm.cdf(z_stat)

        return {
            "test": "Two-Sample Z-Test",
            "n1": n1, "n2": n2,
            "mean1": round(np.mean(x1), 4),
            "mean2": round(np.mean(x2), 4),
            "diff_means": round(diff, 4),
            "z_stat": round(z_stat, 4),
            "p_value": round(p_value, 6),
            "reject_H0": p_value < 0.05
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: T-TEST (FROM SCRATCH)
# ─────────────────────────────────────────────────────────────────────────────

class TTest:
    """
    T-Test — use when:
      • Population std is UNKNOWN
      • Estimate std from sample (uses t-distribution)
      • Works well for small n too

    Three variants:
      1. One-sample  : compare sample mean vs known value
      2. Two-sample  : compare means of two independent groups
      3. Paired      : compare means of matched pairs (before/after)
    """

    @staticmethod
    def one_sample(sample: np.ndarray, mu0: float,
                   alternative: str = "two-sided") -> dict:
        x = _validate_array(sample)
        n = len(x)
        x_bar = np.mean(x)
        s = np.std(x, ddof=1)          # Sample std (Bessel's correction)
        se = s / np.sqrt(n)
        t_stat = (x_bar - mu0) / se
        df = n - 1                      # Degrees of freedom

        if alternative == "two-sided":
            p_value = 2 * stats.t.sf(abs(t_stat), df)
        elif alternative == "greater":
            p_value = stats.t.sf(t_stat, df)
        else:
            p_value = stats.t.cdf(t_stat, df)

        # 95% Confidence Interval
        t_crit = stats.t.ppf(0.975, df)
        ci_low = x_bar - t_crit * se
        ci_high = x_bar + t_crit * se

        return {
            "test": "One-Sample T-Test",
            "n": n, "x_bar": round(x_bar, 4),
            "sample_std": round(s, 4),
            "mu0": mu0, "df": df,
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "ci_95": (round(ci_low, 4), round(ci_high, 4)),
            "alternative": alternative,
            "reject_H0": p_value < 0.05
        }

    @staticmethod
    def two_sample(sample1: np.ndarray, sample2: np.ndarray,
                   equal_var: bool = True,
                   alternative: str = "two-sided") -> dict:
        """
        equal_var=True  → Student's t-test (pooled variance)
        equal_var=False → Welch's t-test (unequal variances — SAFER default)
        """
        x1 = _validate_array(sample1, "sample1")
        x2 = _validate_array(sample2, "sample2")
        n1, n2 = len(x1), len(x2)
        m1, m2 = np.mean(x1), np.mean(x2)
        s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)

        if equal_var:
            # Pooled variance
            sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
            se = np.sqrt(sp2 * (1/n1 + 1/n2))
            df = n1 + n2 - 2
            test_name = "Two-Sample T-Test (Student)"
        else:
            # Welch's formula
            se = np.sqrt(s1**2 / n1 + s2**2 / n2)
            # Welch-Satterthwaite degrees of freedom
            num = (s1**2 / n1 + s2**2 / n2)**2
            den = (s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)
            df = num / den
            test_name = "Two-Sample T-Test (Welch)"

        t_stat = (m1 - m2) / se

        if alternative == "two-sided":
            p_value = 2 * stats.t.sf(abs(t_stat), df)
        elif alternative == "greater":
            p_value = stats.t.sf(t_stat, df)
        else:
            p_value = stats.t.cdf(t_stat, df)

        return {
            "test": test_name,
            "n1": n1, "n2": n2,
            "mean1": round(m1, 4), "mean2": round(m2, 4),
            "std1": round(s1, 4), "std2": round(s2, 4),
            "t_stat": round(t_stat, 4),
            "df": round(df, 2),
            "p_value": round(p_value, 6),
            "reject_H0": p_value < 0.05
        }

    @staticmethod
    def paired(before: np.ndarray, after: np.ndarray,
               alternative: str = "two-sided") -> dict:
        """Paired t-test — same subjects measured twice (before/after)."""
        b = _validate_array(before, "before")
        a = _validate_array(after, "after")
        if len(b) != len(a):
            raise ValueError("before and after must have same length")

        diff = a - b                   # Paired differences
        return TTest.one_sample(diff, mu0=0.0, alternative=alternative)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CHI-SQUARE TEST (FROM SCRATCH)
# ─────────────────────────────────────────────────────────────────────────────

class ChiSquareTest:
    """
    Chi-Square Test — for CATEGORICAL data.

    1. Goodness of Fit  : Does observed distribution match expected?
    2. Independence     : Are two categorical variables related?

    Key assumption: Expected frequency >= 5 in each cell.
    """

    @staticmethod
    def goodness_of_fit(observed: np.ndarray,
                        expected: Optional[np.ndarray] = None) -> dict:
        """
        H0: Observed distribution matches expected.
        If expected=None, assumes uniform distribution.
        """
        obs = np.asarray(observed, dtype=float)
        if expected is None:
            exp = np.full_like(obs, obs.sum() / len(obs))
        else:
            exp = np.asarray(expected, dtype=float)
            exp = exp / exp.sum() * obs.sum()   # Normalize to same total

        # Chi-square statistic: sum((O - E)^2 / E)
        chi2_stat = np.sum((obs - exp)**2 / exp)
        df = len(obs) - 1
        p_value = stats.chi2.sf(chi2_stat, df)

        if np.any(exp < 5):
            warning = "⚠ Some expected counts < 5 — results may be unreliable"
        else:
            warning = "Expected counts OK (all >= 5)"

        return {
            "test": "Chi-Square Goodness of Fit",
            "observed": obs.tolist(),
            "expected": exp.round(2).tolist(),
            "chi2_stat": round(chi2_stat, 4),
            "df": df,
            "p_value": round(p_value, 6),
            "reject_H0": p_value < 0.05,
            "warning": warning
        }

    @staticmethod
    def independence(contingency_table: np.ndarray) -> dict:
        """
        H0: Two variables are independent.
        Input: 2D contingency table (rows=variable1, cols=variable2)
        """
        table = np.asarray(contingency_table, dtype=float)
        if table.ndim != 2:
            raise ValueError("contingency_table must be 2D")

        n = table.sum()
        row_sums = table.sum(axis=1, keepdims=True)
        col_sums = table.sum(axis=0, keepdims=True)

        # Expected: E[i,j] = (row_i_sum * col_j_sum) / n
        expected = (row_sums @ col_sums) / n
        chi2_stat = np.sum((table - expected)**2 / expected)
        df = (table.shape[0] - 1) * (table.shape[1] - 1)
        p_value = stats.chi2.sf(chi2_stat, df)

        # Cramér's V — effect size for chi-square independence
        cramers_v = np.sqrt(chi2_stat / (n * (min(table.shape) - 1)))

        if np.any(expected < 5):
            warning = "⚠ Some expected counts < 5 — consider Fisher's Exact Test"
        else:
            warning = "Expected counts OK"

        return {
            "test": "Chi-Square Independence",
            "table_shape": table.shape,
            "chi2_stat": round(chi2_stat, 4),
            "df": df,
            "p_value": round(p_value, 6),
            "cramers_v": round(cramers_v, 4),
            "effect_size": ("weak" if cramers_v < 0.1
                            else "moderate" if cramers_v < 0.3 else "strong"),
            "reject_H0": p_value < 0.05,
            "warning": warning
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: MANN-WHITNEY U TEST (NON-PARAMETRIC)
# ─────────────────────────────────────────────────────────────────────────────

class MannWhitneyTest:
    """
    Mann-Whitney U Test — non-parametric alternative to two-sample t-test.
    Use when:
      • Data is NOT normally distributed
      • Small sample sizes
      • Ordinal data

    H0: Distributions of group1 and group2 are the same.
    """

    @staticmethod
    def test(group1: np.ndarray, group2: np.ndarray,
             alternative: str = "two-sided") -> dict:
        g1 = _validate_array(group1, "group1")
        g2 = _validate_array(group2, "group2")
        n1, n2 = len(g1), len(g2)

        # Compute U statistic from scratch
        # U1 = number of times g1[i] > g2[j]
        U1 = sum(1 for a in g1 for b in g2 if a > b) + \
             0.5 * sum(1 for a in g1 for b in g2 if a == b)
        U2 = n1 * n2 - U1
        U_stat = min(U1, U2)

        # Normal approximation (for large samples)
        mu_U = n1 * n2 / 2
        sigma_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        z = (U_stat - mu_U) / sigma_U

        if alternative == "two-sided":
            p_value = 2 * stats.norm.cdf(-abs(z))
        elif alternative == "greater":
            p_value = stats.norm.cdf(-z)
        else:
            p_value = stats.norm.cdf(z)

        # Rank-biserial correlation — effect size
        r_rb = 1 - (2 * U_stat) / (n1 * n2)

        return {
            "test": "Mann-Whitney U Test",
            "n1": n1, "n2": n2,
            "U1": round(U1, 2), "U2": round(U2, 2),
            "U_stat": round(U_stat, 2),
            "z_approx": round(z, 4),
            "p_value": round(p_value, 6),
            "rank_biserial_r": round(r_rb, 4),
            "effect_size": ("small" if abs(r_rb) < 0.3
                            else "medium" if abs(r_rb) < 0.5 else "large"),
            "reject_H0": p_value < 0.05
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: BOOTSTRAP CONFIDENCE INTERVALS
# ─────────────────────────────────────────────────────────────────────────────

class BootstrapCI:
    """
    Bootstrap Confidence Interval — distribution-free approach.
    Works for ANY statistic (mean, median, AUC, F1, etc.)

    Method: Percentile Bootstrap
    1. Resample with replacement B times
    2. Compute statistic on each resample
    3. Take (alpha/2, 1-alpha/2) percentiles as CI
    """

    @staticmethod
    def compute(data: np.ndarray, statistic_fn=np.mean,
                B: int = 2000, confidence: float = 0.95,
                random_state: int = 42) -> dict:
        x = _validate_array(data)
        rng = np.random.default_rng(random_state)

        # Bootstrap resamples
        boot_stats = np.array([
            statistic_fn(rng.choice(x, size=len(x), replace=True))
            for _ in range(B)
        ])

        alpha = 1 - confidence
        ci_low = np.percentile(boot_stats, 100 * alpha / 2)
        ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        observed_stat = statistic_fn(x)

        return {
            "statistic": statistic_fn.__name__ if hasattr(statistic_fn, '__name__') else "custom",
            "observed_value": round(float(observed_stat), 6),
            "bootstrap_B": B,
            "confidence": confidence,
            "ci_low": round(float(ci_low), 6),
            "ci_high": round(float(ci_high), 6),
            "ci_width": round(float(ci_high - ci_low), 6),
            "bootstrap_std": round(float(np.std(boot_stats)), 6)
        }

    @staticmethod
    def compare_two_groups(group1: np.ndarray, group2: np.ndarray,
                           statistic_fn=np.mean, B: int = 2000,
                           confidence: float = 0.95,
                           random_state: int = 42) -> dict:
        """Bootstrap CI for the difference between two groups."""
        g1 = _validate_array(group1, "group1")
        g2 = _validate_array(group2, "group2")
        rng = np.random.default_rng(random_state)

        boot_diffs = np.array([
            statistic_fn(rng.choice(g1, size=len(g1), replace=True)) -
            statistic_fn(rng.choice(g2, size=len(g2), replace=True))
            for _ in range(B)
        ])

        alpha = 1 - confidence
        ci_low = np.percentile(boot_diffs, 100 * alpha / 2)
        ci_high = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
        observed_diff = statistic_fn(g1) - statistic_fn(g2)

        significant = not (ci_low <= 0 <= ci_high)

        return {
            "observed_diff": round(float(observed_diff), 6),
            "ci_low": round(float(ci_low), 6),
            "ci_high": round(float(ci_high), 6),
            "confidence": confidence,
            "significant": significant,
            "interpretation": (
                "Groups are significantly different (CI excludes 0)"
                if significant else
                "No significant difference (CI includes 0)"
            )
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MULTIPLE TESTING CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

class MultipleTestingCorrection:
    """
    Problem: If you run 20 tests at α=0.05, you expect 1 false positive by chance!
    Family-Wise Error Rate (FWER) = 1 - (1-α)^m → blows up with m tests.

    Solutions:
      1. Bonferroni   : Very conservative — use for few tests
      2. BH (FDR)     : Controls False Discovery Rate — better for many tests
    """

    @staticmethod
    def bonferroni(p_values: List[float], alpha: float = 0.05) -> pd.DataFrame:
        """
        Bonferroni Correction:
        Adjusted threshold = α / m
        Each test must beat a harder threshold.
        """
        p = np.array(p_values)
        m = len(p)
        corrected_alpha = alpha / m
        adjusted_p = np.minimum(p * m, 1.0)   # Adjusted p-values

        df = pd.DataFrame({
            "original_p": p,
            "adjusted_p": adjusted_p.round(6),
            "reject_H0_uncorrected": p < alpha,
            "reject_H0_bonferroni": p < corrected_alpha
        })
        df.index.name = "test_index"
        print(f"\nBonferroni: α_corrected = {alpha}/{m} = {corrected_alpha:.6f}")
        return df

    @staticmethod
    def benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> pd.DataFrame:
        """
        Benjamini-Hochberg (FDR) Correction:
        Controls False Discovery Rate — less conservative than Bonferroni.

        Algorithm:
        1. Sort p-values ascending: p(1) <= p(2) <= ... <= p(m)
        2. For rank k: threshold = (k/m) * α
        3. Find largest k where p(k) <= (k/m)*α
        4. Reject all hypotheses up to that k
        """
        p = np.array(p_values)
        m = len(p)
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]

        # BH thresholds
        ranks = np.arange(1, m + 1)
        bh_thresholds = (ranks / m) * alpha

        # Find largest k where p(k) <= threshold
        below = sorted_p <= bh_thresholds
        if below.any():
            max_k = np.where(below)[0].max()
            reject_mask = np.zeros(m, dtype=bool)
            reject_mask[:max_k + 1] = True
        else:
            reject_mask = np.zeros(m, dtype=bool)

        # Map back to original order
        reject_original = np.zeros(m, dtype=bool)
        reject_original[sorted_idx] = reject_mask

        # BH adjusted p-values
        adjusted_p = np.minimum.accumulate(
            (sorted_p * m / ranks)[::-1])[::-1]
        adjusted_p_original = np.empty(m)
        adjusted_p_original[sorted_idx] = np.minimum(adjusted_p, 1.0)

        df = pd.DataFrame({
            "original_p": p,
            "bh_adjusted_p": adjusted_p_original.round(6),
            "reject_H0_uncorrected": p < alpha,
            "reject_H0_BH": reject_original
        })
        df.index.name = "test_index"
        print(f"\nBH FDR: Controlling FDR at q={alpha}")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: EFFECT SIZE
# ─────────────────────────────────────────────────────────────────────────────

class EffectSize:
    """
    Statistical significance ≠ Practical significance!
    With large N, even tiny meaningless differences become 'significant'.
    Effect size measures the MAGNITUDE of the difference.
    """

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> dict:
        """
        Cohen's d — standardized mean difference for continuous data.
        d = (mean1 - mean2) / pooled_std

        Interpretation:
          |d| < 0.2  → negligible
          |d| < 0.5  → small
          |d| < 0.8  → medium
          |d| >= 0.8 → large
        """
        g1 = _validate_array(group1, "group1")
        g2 = _validate_array(group2, "group2")
        n1, n2 = len(g1), len(g2)
        m1, m2 = np.mean(g1), np.mean(g2)
        s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)

        # Pooled std
        sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        d = (m1 - m2) / sp

        magnitude = ("negligible" if abs(d) < 0.2
                     else "small" if abs(d) < 0.5
                     else "medium" if abs(d) < 0.8 else "large")

        return {
            "cohens_d": round(d, 4),
            "magnitude": magnitude,
            "interpretation": f"Group 1 is {abs(d):.2f} pooled SDs {'above' if d>0 else 'below'} Group 2"
        }

    @staticmethod
    def cohens_h(p1: float, p2: float) -> dict:
        """
        Cohen's h — effect size for difference between two proportions.
        h = 2 * arcsin(sqrt(p1)) - 2 * arcsin(sqrt(p2))
        """
        h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
        magnitude = ("negligible" if abs(h) < 0.2
                     else "small" if abs(h) < 0.5
                     else "medium" if abs(h) < 0.8 else "large")
        return {
            "cohens_h": round(h, 4),
            "magnitude": magnitude,
            "p1": p1, "p2": p2
        }

    @staticmethod
    def cramers_v(contingency_table: np.ndarray) -> dict:
        """Cramér's V — effect size for chi-square independence."""
        table = np.asarray(contingency_table, dtype=float)
        n = table.sum()
        chi2 = stats.chi2_contingency(table, correction=False)[0]
        k = min(table.shape) - 1
        v = np.sqrt(chi2 / (n * k))
        return {
            "cramers_v": round(v, 4),
            "magnitude": ("negligible" if v < 0.1
                          else "small" if v < 0.3
                          else "medium" if v < 0.5 else "large")
        }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: POWER ANALYSIS (FROM SCRATCH)
# ─────────────────────────────────────────────────────────────────────────────

class PowerAnalysis:
    """
    Statistical Power = P(Reject H0 | H0 is FALSE)
    = Probability of detecting a real effect when it exists.

    Trade-offs:
      ↑ sample size  → ↑ power
      ↑ effect size  → ↑ power
      ↑ alpha        → ↑ power (but more false positives)

    Convention: Power >= 0.80 is acceptable.

    Usage: Determine sample size BEFORE collecting data!
    """

    @staticmethod
    def compute_power(n: int, effect_size: float,
                      alpha: float = 0.05, test: str = "two-sided") -> dict:
        """Compute power for two-sample t-test given sample size and effect size."""
        # Non-centrality parameter
        delta = effect_size * np.sqrt(n / 2)
        df = 2 * n - 2
        t_crit = stats.t.ppf(1 - alpha / (2 if test == "two-sided" else 1), df)

        # Power = P(T > t_crit | delta) where T ~ noncentral t
        power = 1 - stats.nct.cdf(t_crit, df, delta)
        if test == "two-sided":
            power += stats.nct.cdf(-t_crit, df, delta)

        return {
            "n_per_group": n,
            "total_n": 2 * n,
            "effect_size_d": effect_size,
            "alpha": alpha,
            "power": round(power, 4),
            "adequate": power >= 0.80,
            "recommendation": ("✅ Adequate power" if power >= 0.80
                               else f"❌ Need more samples (power={power:.2f} < 0.80)")
        }

    @staticmethod
    def find_sample_size(effect_size: float, alpha: float = 0.05,
                         target_power: float = 0.80) -> dict:
        """Find minimum n per group to achieve target power."""
        for n in range(2, 10_001):
            result = PowerAnalysis.compute_power(n, effect_size, alpha)
            if result["power"] >= target_power:
                return {
                    "n_per_group": n,
                    "total_n": 2 * n,
                    "effect_size_d": effect_size,
                    "alpha": alpha,
                    "achieved_power": result["power"],
                    "target_power": target_power
                }
        return {"error": "Could not achieve target power within n=10,000"}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: McNEMAR'S TEST — MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

class McNemarTest:
    """
    McNemar's Test — Compare two classifiers on the SAME test set.
    H0: Both models have the same error rate.

    Use instead of comparing accuracy numbers directly!
    Why? Two models may have same accuracy but different error *patterns*.

    Table:
                    Model B correct | Model B wrong
    Model A correct       a         |       b
    Model A wrong         c         |       d

    Chi2 = (|b - c| - 1)^2 / (b + c)    [with continuity correction]
    """

    @staticmethod
    def test(y_true: np.ndarray,
             y_pred1: np.ndarray,
             y_pred2: np.ndarray) -> dict:
        y_true = np.asarray(y_true)
        y_pred1 = np.asarray(y_pred1)
        y_pred2 = np.asarray(y_pred2)

        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)

        a = np.sum(correct1 & correct2)       # Both correct
        b = np.sum(correct1 & ~correct2)      # Model1 correct, Model2 wrong
        c = np.sum(~correct1 & correct2)      # Model1 wrong, Model2 correct
        d = np.sum(~correct1 & ~correct2)     # Both wrong

        if b + c == 0:
            return {"test": "McNemar", "error": "No discordant pairs (b+c=0)"}

        # Chi-square with continuity correction
        chi2_stat = (abs(b - c) - 1)**2 / (b + c)
        p_value = stats.chi2.sf(chi2_stat, df=1)

        return {
            "test": "McNemar's Test",
            "contingency": {"a": int(a), "b": int(b), "c": int(c), "d": int(d)},
            "discordant_pairs": int(b + c),
            "model1_acc": round(float(np.mean(correct1)), 4),
            "model2_acc": round(float(np.mean(correct2)), 4),
            "chi2_stat": round(chi2_stat, 4),
            "p_value": round(p_value, 6),
            "reject_H0": p_value < 0.05,
            "interpretation": (
                "Models have SIGNIFICANTLY different error rates"
                if p_value < 0.05 else
                "No significant difference between models"
            )
        }


# ─────────────────────────────────────────────────────────────────────────────
# DEMO & DRIVER
# ─────────────────────────────────────────────────────────────────────────────

def run_all_demos():
    rng = np.random.default_rng(42)

    # ── Z-Test ────────────────────────────────────────────────────────────────
    print_separator("1. Z-TEST")
    sample = rng.normal(loc=102, scale=15, size=50)
    result = ZTest.one_sample(sample, mu0=100, sigma=15)
    for k, v in result.items():
        print(f"  {k:20s}: {v}")
    print("\n  " + interpret_pvalue(result["p_value"]))

    # ── T-Tests ───────────────────────────────────────────────────────────────
    print_separator("2. T-TESTS")
    grp1 = rng.normal(75, 10, 30)
    grp2 = rng.normal(70, 12, 25)

    print("\n  [One-Sample T-Test]")
    r = TTest.one_sample(grp1, mu0=70)
    print(f"  t={r['t_stat']}, p={r['p_value']}, CI={r['ci_95']}")
    print("  " + interpret_pvalue(r["p_value"]))

    print("\n  [Two-Sample Welch T-Test]")
    r2 = TTest.two_sample(grp1, grp2, equal_var=False)
    print(f"  t={r2['t_stat']}, p={r2['p_value']}, reject={r2['reject_H0']}")
    print("  " + interpret_pvalue(r2["p_value"]))

    print("\n  [Paired T-Test — before/after training]")
    before = rng.normal(60, 8, 20)
    after  = before + rng.normal(5, 4, 20)
    rp = TTest.paired(before, after)
    print(f"  Mean improvement: {np.mean(after-before):.2f}")
    print(f"  t={rp['t_stat']}, p={rp['p_value']}, reject={rp['reject_H0']}")

    # ── Chi-Square ────────────────────────────────────────────────────────────
    print_separator("3. CHI-SQUARE TEST")
    obs = [45, 30, 25]
    print("\n  [Goodness of Fit — Uniform?]")
    r = ChiSquareTest.goodness_of_fit(obs)
    print(f"  χ²={r['chi2_stat']}, df={r['df']}, p={r['p_value']}")
    print("  " + interpret_pvalue(r["p_value"]))

    print("\n  [Independence — Gender vs Preference]")
    table = np.array([[30, 20], [15, 35]])
    r = ChiSquareTest.independence(table)
    print(f"  χ²={r['chi2_stat']}, Cramér's V={r['cramers_v']} ({r['effect_size']})")
    print("  " + interpret_pvalue(r["p_value"]))

    # ── Mann-Whitney ──────────────────────────────────────────────────────────
    print_separator("4. MANN-WHITNEY U TEST (Non-Parametric)")
    g1 = rng.exponential(5, 30)
    g2 = rng.exponential(7, 30)
    r = MannWhitneyTest.test(g1, g2)
    print(f"  U={r['U_stat']}, z={r['z_approx']}, p={r['p_value']}")
    print(f"  Rank-biserial r={r['rank_biserial_r']} ({r['effect_size']} effect)")
    print("  " + interpret_pvalue(r["p_value"]))

    # ── Bootstrap CI ──────────────────────────────────────────────────────────
    print_separator("5. BOOTSTRAP CONFIDENCE INTERVALS")
    data = rng.normal(50, 10, 100)
    r_mean = BootstrapCI.compute(data, np.mean, B=5000)
    r_median = BootstrapCI.compute(data, np.median, B=5000)
    print(f"\n  Mean   : {r_mean['observed_value']:.3f}  "
          f"95% CI [{r_mean['ci_low']:.3f}, {r_mean['ci_high']:.3f}]")
    print(f"  Median : {r_median['observed_value']:.3f}  "
          f"95% CI [{r_median['ci_low']:.3f}, {r_median['ci_high']:.3f}]")

    print("\n  [Bootstrap CI for Difference of Means]")
    g1 = rng.normal(55, 10, 50)
    g2 = rng.normal(50, 10, 50)
    r = BootstrapCI.compare_two_groups(g1, g2)
    print(f"  Diff={r['observed_diff']:.3f}, "
          f"CI=[{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print(f"  {r['interpretation']}")

    # ── Multiple Testing ──────────────────────────────────────────────────────
    print_separator("6. MULTIPLE TESTING CORRECTION")
    p_vals = [0.001, 0.008, 0.039, 0.041, 0.12, 0.65]
    print(f"\n  Testing {len(p_vals)} hypotheses simultaneously:")
    print("\n  Bonferroni:")
    df_bon = MultipleTestingCorrection.bonferroni(p_vals)
    print(df_bon.to_string())
    print("\n  Benjamini-Hochberg (FDR):")
    df_bh = MultipleTestingCorrection.benjamini_hochberg(p_vals)
    print(df_bh.to_string())

    # ── Effect Size ───────────────────────────────────────────────────────────
    print_separator("7. EFFECT SIZE")
    g1 = rng.normal(70, 10, 50)
    g2 = rng.normal(65, 10, 50)
    r = EffectSize.cohens_d(g1, g2)
    print(f"\n  Cohen's d={r['cohens_d']} ({r['magnitude']})")
    print(f"  {r['interpretation']}")

    r_h = EffectSize.cohens_h(0.60, 0.50)
    print(f"\n  Cohen's h={r_h['cohens_h']} (p1=0.60 vs p2=0.50) — {r_h['magnitude']}")

    # ── Power Analysis ────────────────────────────────────────────────────────
    print_separator("8. POWER ANALYSIS")
    print("\n  [Find sample size for d=0.5, 80% power]")
    r = PowerAnalysis.find_sample_size(effect_size=0.5, target_power=0.80)
    print(f"  Need {r['n_per_group']} per group ({r['total_n']} total)")
    print(f"  Achieved power: {r['achieved_power']:.3f}")

    for n in [20, 50, 100, 200]:
        r = PowerAnalysis.compute_power(n, effect_size=0.5)
        print(f"  n={n:3d}/group → power={r['power']:.3f}  {r['recommendation']}")

    # ── McNemar ───────────────────────────────────────────────────────────────
    print_separator("9. McNEMAR'S TEST — Model Comparison")
    np.random.seed(42)
    y_true  = rng.integers(0, 2, 100)
    y_pred1 = (rng.random(100) > 0.35).astype(int)
    y_pred2 = (rng.random(100) > 0.40).astype(int)
    r = McNemarTest.test(y_true, y_pred1, y_pred2)
    print(f"\n  Model1 acc={r['model1_acc']}, Model2 acc={r['model2_acc']}")
    print(f"  χ²={r['chi2_stat']}, p={r['p_value']}")
    print(f"  → {r['interpretation']}")

    print_separator("COMPLETE")
    print("  ✅ All statistical significance tests demonstrated successfully")


if __name__ == "__main__":
    run_all_demos()
