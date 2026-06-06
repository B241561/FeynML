"""
Phase 3.4 — SliceFinder: Automated Failure Slice Detection
============================================================
Primary source: Chung, Kraska, Polyzotis et al. (ICDE 2019) — arxiv 1807.06068v3

CORE PROBLEM (Section 1 of paper):
  Overall model metrics can MASK poor performance on subgroups.
  Example from paper Table 1:
    Overall log-loss = 0.35 (acceptable)
    BUT  Sex=Male slice log-loss = 0.41
    AND  Education=Doctorate slice loss = 0.56  ← completely hidden!

DEFINITION 1 (from paper):
  A SLICE S is a conjunction of feature-value predicates:
    e.g. "Sex=Male AND Education=Bachelors"
  
  A PROBLEMATIC SLICE satisfies:
    (a) effect_size ≥ threshold T
    (b) statistically significant (Welch t-test)
    (c) not subsumed by a simpler problematic slice

EFFECT SIZE (Section 2.3):
  φ = √2 × (ψ(S,h) - ψ(S',h)) / √(σ²_S + σ²_{S'})
  
  where S' = D - S (counterpart = everything NOT in S)
  This measures HOW LARGE the performance gap is (not just whether it exists).

ALGORITHM (Algorithm 1 from paper — Lattice Searching):
  1. Start with depth-1 slices (single predicates)
  2. For each level:
     a. Filter by effect_size ≥ T → candidate queue
     b. Test statistical significance via Welch t-test
     c. Keep significant slices → stop expanding them
     d. Expand non-significant slices by adding one more predicate
  3. α-investing for false discovery control

OUR IMPLEMENTATION:
  - Lattice search (Section 3.1.3) with effect size + Welch t-test
  - Bonferroni correction for multiple comparisons (simpler than α-investing)
  - Supports categorical and discretized numeric features
"""

import math
import random
from collections import defaultdict
from itertools import combinations


# ─────────────────────────────────────────────────────────────────────────────
# 1. PER-SAMPLE LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def log_loss_per_sample(y_true, y_probs, eps=1e-15):
    """Per-sample log-loss (not averaged). Used for effect size computation."""
    losses = []
    for yt, p in zip(y_true, y_probs):
        p = max(eps, min(1 - eps, p))
        losses.append(-(yt * math.log(p) + (1 - yt) * math.log(1 - p)))
    return losses

def accuracy_loss_per_sample(y_true, y_pred):
    """Per-sample 0-1 loss (1=wrong, 0=correct)."""
    return [0 if yt == yp else 1 for yt, yp in zip(y_true, y_pred)]

def squared_error_per_sample(y_true, y_pred):
    """Per-sample squared error for regression."""
    return [(yt - yp)**2 for yt, yp in zip(y_true, y_pred)]


# ─────────────────────────────────────────────────────────────────────────────
# 2. SLICE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class Slice:
    """
    A data slice: conjunction of (feature, value) predicates.
    e.g. Slice([("sex", "Male"), ("education", "Bachelors")])
    """
    def __init__(self, predicates):
        self.predicates = tuple(sorted(predicates))

    def matches(self, row_dict):
        """Does a data row (dict) satisfy ALL predicates in this slice?"""
        return all(row_dict.get(feat) == val for feat, val in self.predicates)

    def n_literals(self):
        return len(self.predicates)

    def is_subsumed_by(self, other_slice):
        """
        S is subsumed by S' if S' has fewer literals and S' ⊆ S.
        (every example in S also satisfies S', so S is more specific)
        """
        return set(other_slice.predicates).issubset(set(self.predicates))

    def __repr__(self):
        parts = " AND ".join(f"{f}={v}" for f, v in self.predicates)
        return f"Slice({parts})"

    def __hash__(self):
        return hash(self.predicates)

    def __eq__(self, other):
        return self.predicates == other.predicates


# ─────────────────────────────────────────────────────────────────────────────
# 3. EFFECT SIZE & STATISTICAL TEST (Section 2.3 of paper)
# ─────────────────────────────────────────────────────────────────────────────

def _mean(data):
    return sum(data) / len(data) if data else 0.0

def _var(data):
    if len(data) < 2:
        return 0.0
    m = _mean(data)
    return sum((x - m)**2 for x in data) / (len(data) - 1)

def compute_effect_size(slice_losses, counterpart_losses):
    """
    Effect size φ from Definition in paper Section 2.3:
    φ = √2 × (mean_S - mean_S') / √(var_S + var_S')
    
    Captures MAGNITUDE of performance difference.
    Cohen's rule: 0.2=small, 0.5=medium, 0.8=large, 1.3=very large
    """
    if not slice_losses or not counterpart_losses:
        return 0.0
    ms  = _mean(slice_losses)
    mc  = _mean(counterpart_losses)
    vs  = _var(slice_losses)
    vc  = _var(counterpart_losses)
    denom = math.sqrt(vs + vc)
    if denom < 1e-12:
        return 0.0
    return math.sqrt(2) * (ms - mc) / denom

def welch_ttest_pvalue(slice_losses, counterpart_losses):
    """
    Welch's t-test (Section 2.3 of paper):
    H₀: ψ(S,h) ≤ ψ(S',h)   [slice is NOT worse]
    Hₐ: ψ(S,h) > ψ(S',h)   [slice IS worse]
    
    Returns one-tailed p-value.
    """
    n1, n2 = len(slice_losses), len(counterpart_losses)
    m1, m2 = _mean(slice_losses), _mean(counterpart_losses)
    v1, v2 = _var(slice_losses), _var(counterpart_losses)

    se = math.sqrt(v1 / n1 + v2 / n2)
    if se < 1e-12:
        return 1.0

    t_stat = (m1 - m2) / se

    # Welch-Satterthwaite degrees of freedom
    num = (v1/n1 + v2/n2)**2
    den = (v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1)
    df  = num / den if den > 1e-12 else min(n1, n2) - 1

    # One-tailed p-value via normal approximation (works for df > 30)
    try:
        from scipy.stats import t as t_dist
        p_value = float(t_dist.sf(t_stat, df))
    except ImportError:
        # Normal approximation fallback
        p_value = 0.5 * math.erfc(t_stat / math.sqrt(2))

    return p_value

def is_problematic(slice_losses, counterpart_losses,
                   effect_threshold=0.3, alpha=0.05):
    """
    Definition 1 from paper:
    A slice is problematic if:
      (a) effect_size ≥ T
      (b) p-value < α (statistically significant)
    """
    if len(slice_losses) < 5:   # too small to be meaningful
        return False, 0.0, 1.0
    eff = compute_effect_size(slice_losses, counterpart_losses)
    if eff < effect_threshold:
        return False, eff, 1.0
    p = welch_ttest_pvalue(slice_losses, counterpart_losses)
    return p < alpha, eff, p


# ─────────────────────────────────────────────────────────────────────────────
# 4. SLICE FINDER — LATTICE SEARCH (Algorithm 1 from paper)
# ─────────────────────────────────────────────────────────────────────────────

class SliceFinder:
    """
    Implements Algorithm 1 (Lattice Searching) from Chung et al. (2019).
    
    Finds top-k interpretable, large, statistically significant failure slices.
    
    Usage:
      sf = SliceFinder(data_rows, per_sample_losses, feature_cols)
      results = sf.find_slices(k=5, effect_threshold=0.3, alpha=0.05)
    """

    def __init__(self, data_rows, per_sample_losses, feature_cols,
                 max_literals=3, min_slice_size=20):
        """
        Args:
          data_rows:         list of dicts, one per validation sample
          per_sample_losses: list of per-sample loss values (same length)
          feature_cols:      list of feature names to slice on
          max_literals:      max depth of predicates in one slice
          min_slice_size:    ignore slices smaller than this
        """
        assert len(data_rows) == len(per_sample_losses)
        self.data          = data_rows
        self.losses        = per_sample_losses
        self.features      = feature_cols
        self.max_literals  = max_literals
        self.min_size      = min_slice_size
        self.n             = len(data_rows)

        # Pre-compute feature value catalogue
        self.feature_values = defaultdict(set)
        for row in data_rows:
            for feat in feature_cols:
                if feat in row:
                    self.feature_values[feat].add(row[feat])

    def _get_slice_losses(self, sl):
        """Get losses for samples in slice and its counterpart."""
        in_slice = [i for i, row in enumerate(self.data) if sl.matches(row)]
        out_slice = [i for i in range(self.n) if i not in set(in_slice)]
        return (
            [self.losses[i] for i in in_slice],
            [self.losses[i] for i in out_slice],
            len(in_slice),
        )

    def _expand_slices(self, parent_slices, depth):
        """
        Generate child slices by adding one more predicate (ExpandSlices in paper).
        Only expand NON-problematic slices (problematic ones are leaves).
        """
        children = set()
        for sl in parent_slices:
            existing_feats = {f for f, v in sl.predicates}
            for feat in self.features:
                if feat in existing_feats:
                    continue
                for val in self.feature_values[feat]:
                    new_preds = list(sl.predicates) + [(feat, val)]
                    children.add(Slice(new_preds))
        return children

    def find_slices(self, k=5, effect_threshold=0.3, alpha=0.05,
                    bonferroni=True):
        """
        Algorithm 1 from Chung et al. (2019) — Lattice Searching.
        
        Returns top-k problematic slices sorted by effect size descending.
        """
        found_slices   = []   # confirmed problematic
        non_problematic = []  # to be expanded next level
        n_tests_total  = 0

        # Level 1: single-feature predicates
        current_level = set()
        for feat in self.features:
            for val in self.feature_values[feat]:
                current_level.add(Slice([(feat, val)]))

        for depth in range(1, self.max_literals + 1):
            candidates  = []
            to_expand   = []

            for sl in current_level:
                sl_losses, cp_losses, sl_size = self._get_slice_losses(sl)
                if sl_size < self.min_size:
                    continue

                eff = compute_effect_size(sl_losses, cp_losses)

                if eff >= effect_threshold:
                    candidates.append((sl, sl_losses, cp_losses, eff, sl_size))
                else:
                    to_expand.append(sl)

            # Sort candidates by size DESC, effect DESC (ordering ≺ from paper)
            candidates.sort(key=lambda t: (-t[4], -t[3]))

            # Multiple comparisons correction
            alpha_adj = alpha / max(1, len(candidates)) if bonferroni else alpha

            for sl, sl_losses, cp_losses, eff, sl_size in candidates:
                if len(found_slices) >= k:
                    break
                n_tests_total += 1
                p = welch_ttest_pvalue(sl_losses, cp_losses)
                if p < alpha_adj:
                    # Check not subsumed by already-found simpler slice
                    subsumed = any(sl.is_subsumed_by(fs["slice"])
                                   for fs in found_slices
                                   if fs["slice"].n_literals() < sl.n_literals())
                    if not subsumed:
                        found_slices.append({
                            "slice":         sl,
                            "description":   str(sl),
                            "n_literals":    sl.n_literals(),
                            "slice_size":    sl_size,
                            "effect_size":   round(eff, 4),
                            "p_value":       round(p, 6),
                            "slice_loss":    round(_mean(sl_losses), 6),
                            "overall_loss":  round(_mean(self.losses), 6),
                            "loss_gap":      round(_mean(sl_losses) - _mean(self.losses), 6),
                        })
                else:
                    to_expand.append(sl)

            if len(found_slices) >= k:
                break

            # Expand non-problematic slices to next depth
            if depth < self.max_literals:
                current_level = self._expand_slices(to_expand, depth + 1)
                if not current_level:
                    break
            else:
                break

        # Sort final results by effect size
        found_slices.sort(key=lambda x: x["effect_size"], reverse=True)

        return {
            "slices":        found_slices[:k],
            "n_found":       len(found_slices),
            "n_tests":       n_tests_total,
            "overall_loss":  round(_mean(self.losses), 6),
            "n_samples":     self.n,
            "alpha_used":    alpha,
            "effect_threshold": effect_threshold,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MANUAL SLICE DISCOVERY INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

class ManualSliceExplorer:
    """
    Interactive interface for manual slice discovery.
    
    Allows users to:
    - Define custom slice predicates
    - Evaluate slice performance
    - Compare slices against overall performance
    - Explore slice combinations
    """
    
    def __init__(self, data_rows, per_sample_losses, feature_cols):
        """
        Args:
            data_rows: list of dicts, one per validation sample
            per_sample_losses: list of per-sample loss values
            feature_cols: list of feature names to slice on
        """
        assert len(data_rows) == len(per_sample_losses)
        self.data = data_rows
        self.losses = per_sample_losses
        self.features = feature_cols
        self.n = len(data_rows)
        self.overall_loss = sum(per_sample_losses) / max(len(per_sample_losses), 1)
        
        # Pre-compute feature value catalogue
        self.feature_values = defaultdict(set)
        for row in data_rows:
            for feat in feature_cols:
                if feat in row:
                    self.feature_values[feat].add(row[feat])
    
    def evaluate_slice(self, predicates):
        """
        Evaluate a manually defined slice.
        
        Args:
            predicates: dict of {feature: value} or {feature: [values]} for multiple values
        
        Returns:
            dict with slice statistics
        """
        # Build slice object
        pred_list = []
        for feat, val in predicates.items():
            if isinstance(val, list):
                for v in val:
                    pred_list.append((feat, v))
            else:
                pred_list.append((feat, val))
        
        sl = Slice(pred_list)
        
        # Get slice losses
        in_slice = [i for i, row in enumerate(self.data) if sl.matches(row)]
        out_slice = [i for i in range(self.n) if i not in set(in_slice)]
        
        slice_losses = [self.losses[i] for i in in_slice]
        counterpart_losses = [self.losses[i] for i in out_slice]
        
        slice_size = len(in_slice)
        
        if slice_size < 5:
            return {
                "predicates": predicates,
                "size": slice_size,
                "error": "Slice too small (min 5 samples)",
                "slice_loss": None,
                "effect_size": None,
            }
        
        slice_loss = sum(slice_losses) / slice_size
        counterpart_loss = sum(counterpart_losses) / max(len(counterpart_losses), 1)
        
        # Compute effect size
        eff = compute_effect_size(slice_losses, counterpart_losses)
        
        return {
            "predicates": predicates,
            "description": str(sl),
            "size": slice_size,
            "slice_loss": round(slice_loss, 5),
            "rest_loss": round(counterpart_loss, 5),
            "overall_loss": round(self.overall_loss, 5),
            "effect_size": round(eff, 4),
            "loss_gap": round(slice_loss - self.overall_loss, 5),
            "interpretation": (
                f"Slice has {slice_size} samples with loss {slice_loss:.4f} "
                f"vs overall {self.overall_loss:.4f}. "
                f"Effect size: {eff:.3f}."
            ),
        }
    
    def suggest_slices(self, max_slices=10, effect_threshold=0.2):
        """
        Suggest potential slices based on single-feature analysis.
        
        Returns list of promising single-feature slices to explore.
        """
        suggestions = []
        
        for feat in self.features:
            for val in sorted(self.feature_values[feat]):
                result = self.evaluate_slice({feat: val})
                if "error" not in result and result["effect_size"] >= effect_threshold:
                    suggestions.append(result)
        
        suggestions.sort(key=lambda x: x["effect_size"], reverse=True)
        return suggestions[:max_slices]
    
    def explore_combinations(self, base_predicates, max_combinations=5):
        """
        Explore combinations with a base slice.
        
        Args:
            base_predicates: dict of base predicates
            max_combinations: max number of combinations to return
        
        Returns:
            list of refined slices by adding one more predicate
        """
        base_result = self.evaluate_slice(base_predicates)
        
        if "error" in base_result:
            return [base_result]
        
        # Get features not in base predicates
        used_feats = set(base_predicates.keys())
        available_feats = [f for f in self.features if f not in used_feats]
        
        combinations = []
        for feat in available_feats[:3]:  # Limit to avoid explosion
            for val in sorted(self.feature_values[feat]):
                new_pred = {**base_predicates, feat: val}
                result = self.evaluate_slice(new_pred)
                if "error" not in result:
                    combinations.append(result)
        
        combinations.sort(key=lambda x: x["effect_size"], reverse=True)
        return combinations[:max_combinations]


# ─────────────────────────────────────────────────────────────────────────────
# 6. DISCRETIZER — for numeric features
# ─────────────────────────────────────────────────────────────────────────────

def discretize_numeric_features(data_rows, numeric_cols, n_bins=4):
    """
    Convert numeric features to categorical bins for SliceFinder.
    Uses quantile-based binning so each bin has equal sample count.
    """
    bin_maps = {}
    new_rows = [dict(row) for row in data_rows]

    for col in numeric_cols:
        values = sorted([row[col] for row in data_rows if col in row])
        n = len(values)
        boundaries = [values[int(i * n / n_bins)] for i in range(1, n_bins)]

        def bin_label(v, boundaries=boundaries, col=col):
            for i, b in enumerate(boundaries):
                if v < b:
                    return f"{col}_bin{i}"
            return f"{col}_bin{n_bins-1}"

        bin_maps[col] = bin_label
        for row in new_rows:
            if col in row:
                row[col + "_binned"] = bin_label(row[col])

    return new_rows, bin_maps


# ─────────────────────────────────────────────────────────────────────────────
# 6. VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def run_verification():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    import numpy as np

    print("=" * 65)
    print("Phase 3.4 — SliceFinder Verification")
    print("=" * 65)

    # Synthetic dataset with injected failure slices
    np.random.seed(42)
    rng = random.Random(42)
    n   = 600

    # Features
    age_groups = ["young", "middle", "senior"]
    genders    = ["male", "female"]
    education  = ["low", "high"]

    rows   = []
    y_true = []
    y_pred = []

    for i in range(n):
        age = rng.choice(age_groups)
        gen = rng.choice(genders)
        edu = rng.choice(education)

        rows.append({"age": age, "gender": gen, "education": edu})

        # Inject failure: senior + low education → model performs poorly
        if age == "senior" and edu == "low":
            label = rng.randint(0, 1)
            pred  = 1 - label if rng.random() < 0.75 else label   # 75% wrong
        elif gen == "female" and age == "young":
            label = rng.randint(0, 1)
            pred  = 1 - label if rng.random() < 0.60 else label   # 60% wrong
        else:
            label = rng.randint(0, 1)
            pred  = label if rng.random() < 0.80 else 1 - label   # 80% correct

        y_true.append(label)
        y_pred.append(pred)

    # Per-sample losses
    per_sample_losses = accuracy_loss_per_sample(y_true, y_pred)
    overall_accuracy  = 1 - sum(per_sample_losses) / n
    print(f"\n  Overall accuracy: {overall_accuracy:.3f} (hides failure slices)")

    # Run SliceFinder
    sf = SliceFinder(
        data_rows=rows,
        per_sample_losses=per_sample_losses,
        feature_cols=["age", "gender", "education"],
        min_slice_size=15,
    )

    results = sf.find_slices(k=5, effect_threshold=0.3, alpha=0.05)

    print(f"\n  Found {results['n_found']} problematic slices ({results['n_tests']} tests):")
    print(f"  {'Slice':<40} {'Size':>5} {'Effect':>7} {'Loss':>7} {'p-val':>8}")
    print(f"  {'-'*70}")

    found_injected = False
    for sl in results["slices"]:
        flag = ""
        desc = sl["description"]
        if "senior" in desc and "low" in desc:
            flag = "⚠ INJECTED"
            found_injected = True
        elif "female" in desc and "young" in desc:
            flag = "⚠ INJECTED"
            found_injected = True
        print(f"  {desc:<40} {sl['slice_size']:>5} "
              f"{sl['effect_size']:>7.3f} {sl['slice_loss']:>7.3f} "
              f"{sl['p_value']:>8.4f}  {flag}")

    print(f"\n  Detected injected failure slices: "
          f"[{'✓ PASS' if found_injected else '✗ FAIL — increase n or lower threshold'}]")

    # Effect size thresholds
    print(f"\n  Effect size reference (Cohen's rule):")
    print(f"    0.2 = small | 0.5 = medium | 0.8 = large | 1.3 = very large")

    print("=" * 65)


if __name__ == "__main__":
    run_verification()
