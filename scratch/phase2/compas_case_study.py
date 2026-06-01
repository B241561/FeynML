"""
=============================================================================
Phase 2 — COMPAS Case Study: ML Fairness Analysis
=============================================================================
The COMPAS Story:
  COMPAS (Correctional Offender Management Profiling for Alternative Sanctions)
  is a risk assessment tool used in US courts to predict recidivism
  (likelihood of re-offending). ProPublica's 2016 investigation found:

    • Black defendants were nearly twice as likely to be falsely flagged
      as HIGH risk compared to white defendants.
    • White defendants were more often incorrectly flagged as LOW risk
      even when they went on to re-offend.

  Yet the model had EQUAL overall accuracy for both groups!
  This is the "Accuracy Paradox" in fairness — the COMPAS Impossibility Theorem.

Topics Covered:
  1. Synthetic COMPAS dataset generation
  2. Fairness metrics by demographic group
  3. Confusion matrix per subgroup
  4. Disparate Impact analysis
  5. Equalized Odds vs Equal Opportunity
  6. Calibration by subgroup
  7. The Impossibility Theorem (Chouldechova, 2017)
  8. Mitigation strategies overview
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: GENERATE SYNTHETIC COMPAS-LIKE DATASET
# ─────────────────────────────────────────────────────────────────────────────

def generate_compas_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic COMPAS-like dataset with built-in bias.

    Features:
      - age               : defendant age
      - priors_count      : number of prior offenses
      - charge_degree     : misdemeanor(0) or felony(1)
      - race              : 0=White, 1=Black
      - sex               : 0=Female, 1=Male
      - juv_fel_count     : juvenile felony count
      - juv_misd_count    : juvenile misdemeanor count

    Target:
      - two_year_recid : 1 if re-offended within 2 years

    Bias introduced: Black defendants have systematically higher
    predicted risk scores even controlling for priors.
    """
    rng = np.random.default_rng(seed)
    n_black = int(n * 0.51)         # ~51% Black in ProPublica dataset
    n_white = n - n_black

    def make_group(size, is_black, base_recid_rate):
        race = np.ones(size, dtype=int) if is_black else np.zeros(size, dtype=int)
        age = rng.integers(18, 70, size)
        priors = rng.negative_binomial(1.5, 0.4, size)
        charge = rng.binomial(1, 0.55 if is_black else 0.45, size)
        sex = rng.binomial(1, 0.80, size)
        juv_fel = rng.poisson(0.3 if is_black else 0.15, size)
        juv_misd = rng.poisson(0.5 if is_black else 0.25, size)

        # True recidivism probability — driven by actual risk factors
        log_odds = (
            -1.5
            + 0.03 * priors
            - 0.02 * (age - 25)
            + 0.4 * charge
            + 0.2 * juv_fel
            + 0.1 * juv_misd
            + (0.3 if is_black else 0.0)   # ← Systemic bias component
        )
        p_recid = 1 / (1 + np.exp(-log_odds))
        recid = rng.binomial(1, p_recid, size)

        # COMPAS score (1-10) — contains additional racial bias
        compas_base = (
            1
            + 0.8 * priors
            + 3 * charge
            + 1.5 * juv_fel
            + (1.5 if is_black else 0.0)   # Bias in the score itself
        )
        compas_score = np.clip(
            compas_base + rng.normal(0, 1.5, size), 1, 10
        ).astype(int)

        return pd.DataFrame({
            "race": race,
            "race_label": ["Black" if b else "White" for b in race],
            "age": age,
            "sex": sex,
            "priors_count": priors,
            "charge_degree": charge,
            "juv_fel_count": juv_fel,
            "juv_misd_count": juv_misd,
            "compas_score": compas_score,
            "two_year_recid": recid
        })

    df_black = make_group(n_black, is_black=True,  base_recid_rate=0.52)
    df_white = make_group(n_white, is_black=False, base_recid_rate=0.39)
    df = pd.concat([df_black, df_white], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: FAIRNESS METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_fairness_metrics(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              group: np.ndarray) -> pd.DataFrame:
    """
    Compute per-group fairness metrics.

    Key metrics:
      TPR (True Positive Rate)  = Recall = TP / (TP + FN)
      FPR (False Positive Rate) = FP / (FP + TN)  ← MOST IMPORTANT FOR COMPAS
      TNR (True Negative Rate)  = TN / (TN + FP)
      FNR (False Negative Rate) = FN / (TP + FN)
      PPV (Positive Predictive Value) = Precision = TP / (TP + FP)
      Accuracy = (TP + TN) / N

    Fairness Definitions:
      Equalized Odds   : TPR and FPR equal across groups
      Equal Opportunity: TPR equal across groups
      Calibration      : Among those with same score, P(recid)=same across groups
    """
    groups = np.unique(group)
    rows = []

    for g in groups:
        mask = group == g
        yt = y_true[mask]
        yp = y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp).ravel()
        n = len(yt)

        rows.append({
            "group": g,
            "n": n,
            "base_rate": round(yt.mean(), 4),      # P(actually recidivates)
            "predicted_pos_rate": round(yp.mean(), 4),
            "accuracy": round(accuracy_score(yt, yp), 4),
            "TPR": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),   # Recall
            "FPR": round(fp / (fp + tn) if (fp + tn) > 0 else 0, 4),   # KEY metric
            "TNR": round(tn / (tn + fp) if (tn + fp) > 0 else 0, 4),
            "FNR": round(fn / (tp + fn) if (tp + fn) > 0 else 0, 4),
            "PPV": round(precision_score(yt, yp, zero_division=0), 4),
            "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn)
        })

    return pd.DataFrame(rows).set_index("group")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: DISPARATE IMPACT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def disparate_impact(y_pred: np.ndarray,
                     group: np.ndarray,
                     privileged_group: str,
                     unprivileged_group: str) -> dict:
    """
    Disparate Impact (DI) = P(ŷ=1 | unprivileged) / P(ŷ=1 | privileged)

    Legal 4/5ths Rule: DI < 0.8 indicates illegal discrimination.

    Note: Here ŷ=1 means predicted HIGH risk (bad outcome).
    So we want DI close to 1.0 (equal treatment).
    """
    priv_mask = group == privileged_group
    unpriv_mask = group == unprivileged_group

    p_priv = y_pred[priv_mask].mean()
    p_unpriv = y_pred[unpriv_mask].mean()

    if p_priv == 0:
        return {"error": "Privileged group has 0 positive rate"}

    di = p_unpriv / p_priv

    return {
        "privileged_group": privileged_group,
        "unprivileged_group": unprivileged_group,
        "positive_rate_privileged": round(p_priv, 4),
        "positive_rate_unprivileged": round(p_unpriv, 4),
        "disparate_impact_ratio": round(di, 4),
        "4_5ths_rule": "VIOLATED (DI < 0.8)" if di < 0.8 else "OK (DI >= 0.8)",
        "interpretation": (
            f"Unprivileged group ({unprivileged_group}) is predicted positive "
            f"at {di:.2f}x the rate of the privileged group ({privileged_group})"
        )
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: CALIBRATION BY SUBGROUP
# ─────────────────────────────────────────────────────────────────────────────

def calibration_by_group(y_true: np.ndarray,
                          y_score: np.ndarray,
                          group: np.ndarray,
                          n_bins: int = 5) -> pd.DataFrame:
    """
    Check if predicted probabilities mean the same thing across groups.

    Calibration: Among defendants with predicted risk 0.6, do ~60% actually recidivate?

    If calibration holds: A score of 0.6 means the same for Black and White defendants.
    If calibration fails: The score is systematically biased for one group.

    COMPAS Finding: COMPAS IS roughly calibrated, BUT
    this is mathematically incompatible with equal FPR/FNR between groups
    when base rates differ! (The Impossibility Theorem)
    """
    groups = np.unique(group)
    rows = []

    bins = np.linspace(0, 1, n_bins + 1)
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(n_bins)]

    for g in groups:
        mask = group == g
        yt_g = y_true[mask]
        ys_g = y_score[mask]

        for i in range(n_bins):
            bin_mask = (ys_g >= bins[i]) & (ys_g < bins[i+1])
            if i == n_bins - 1:
                bin_mask = (ys_g >= bins[i]) & (ys_g <= bins[i+1])
            n_in_bin = bin_mask.sum()
            if n_in_bin > 0:
                rows.append({
                    "group": g,
                    "score_bin": bin_labels[i],
                    "n": n_in_bin,
                    "mean_predicted": round(ys_g[bin_mask].mean(), 3),
                    "actual_recid_rate": round(yt_g[bin_mask].mean(), 3),
                    "calibration_error": round(
                        abs(ys_g[bin_mask].mean() - yt_g[bin_mask].mean()), 3
                    )
                })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: THE IMPOSSIBILITY THEOREM
# ─────────────────────────────────────────────────────────────────────────────

def explain_impossibility_theorem(metrics_df: pd.DataFrame) -> None:
    """
    Chouldechova (2017) Impossibility Theorem:

    When base rates differ between groups, it is MATHEMATICALLY IMPOSSIBLE
    to simultaneously satisfy:
      1. Calibration (scores mean same thing across groups)
      2. Equal FPR across groups
      3. Equal FNR across groups

    You can satisfy at MOST two of these three simultaneously.

    COMPAS situation:
      - Black defendants have higher base rate (~52%) vs White (~39%)
      - COMPAS optimized for calibration
      - Therefore: FPR and FNR must differ between groups
      - Result: Black defendants face higher FPR (wrongly flagged high risk)

    Proof sketch:
      Let b1, b2 = base rates of groups 1, 2 (b1 ≠ b2)
      Let PPV, FPR, FNR be the fairness targets.
      From confusion matrix algebra:
        FNR = (1 - PPV) * PPV * prevalence / ((1 - PPV) * prevalence + ...)
      These constraints are OVERDETERMINED when b1 ≠ b2.
    """
    print("\n" + "─" * 65)
    print("  THE COMPAS IMPOSSIBILITY THEOREM")
    print("─" * 65)

    if "base_rate" in metrics_df.columns:
        black_row = metrics_df.loc["Black"] if "Black" in metrics_df.index else None
        white_row = metrics_df.loc["White"] if "White" in metrics_df.index else None

        if black_row is not None and white_row is not None:
            print(f"\n  Base Rates:")
            print(f"    Black defendants: {black_row['base_rate']:.1%}")
            print(f"    White defendants: {white_row['base_rate']:.1%}")
            print(f"\n  Base rate difference: "
                  f"{abs(black_row['base_rate'] - white_row['base_rate']):.1%}")

    print("""
  ┌──────────────────────────────────────────────────────────┐
  │  WHEN BASE RATES DIFFER (which they do):                │
  │                                                          │
  │  You CANNOT simultaneously have:                        │
  │  ✓ Calibration (PPV equal across groups)               │
  │  ✓ Equal FPR (equal false accusation rates)            │
  │  ✓ Equal FNR (equal miss rates)                        │
  │                                                          │
  │  Pick AT MOST TWO. This is mathematics, not policy.    │
  │                                                          │
  │  COMPAS chose calibration → Black FPR is 2x White FPR │
  └──────────────────────────────────────────────────────────┘

  Policy choices:
    • If we equalize FPR: model becomes LESS calibrated → scores
      mean different things for different groups
    • If we equalize FNR: model releases more dangerous individuals
      from one group to achieve "fairness"
    • Pre-processing bias: remove race as a feature
      → Proxies (ZIP code, prior arrests) still encode race

  There is no perfect solution. Fairness requires human judgment
  about which metric matters most given the social context.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: TRAIN LOGISTIC REGRESSION MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_model(df: pd.DataFrame,
                features: list,
                target: str = "two_year_recid") -> Tuple:
    """Train a logistic regression model — with and without race feature."""
    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    idx_test = np.arange(len(df))[len(X_train):]   # approximate

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_score = model.predict_proba(X_test_s)[:, 1]

    return model, scaler, X_test, y_test, y_pred, y_score


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: MITIGATION STRATEGIES
# ─────────────────────────────────────────────────────────────────────────────

def mitigation_strategies_overview() -> None:
    """Summary of bias mitigation approaches."""
    strategies = {
        "Pre-processing (Data Level)": [
            "Remove protected attribute (race) from features",
            "Reweigh training samples by group membership",
            "Learning fair representations (adversarial)",
            "Disparate impact remover (transform features)",
        ],
        "In-processing (Model Level)": [
            "Fairness constraints in optimization (Zafar 2017)",
            "Adversarial debiasing (learning to fool fairness auditor)",
            "Prejudice remover regularization",
            "Meta-fair algorithms",
        ],
        "Post-processing (Prediction Level)": [
            "Equalized odds post-processing (Hardt 2016)",
            "Reject option classification (near decision boundary)",
            "Calibrated equalized odds",
            "Group-specific thresholds",
        ],
    }

    print("\n" + "─" * 65)
    print("  BIAS MITIGATION STRATEGIES")
    print("─" * 65)
    for stage, methods in strategies.items():
        print(f"\n  [{stage}]")
        for m in methods:
            print(f"    • {m}")

    print("""
  IMPORTANT: Each mitigation strategy trades off one fairness
  metric against another. Always define which fairness
  criterion matters most BEFORE training the model.

  Key Libraries:
    • AI Fairness 360 (IBM AIF360)
    • Fairlearn (Microsoft)
    • What-If Tool (Google)
    • Themis-ml
    """)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: FULL ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_compas_analysis():
    print("=" * 65)
    print("  COMPAS FAIRNESS ANALYSIS")
    print("=" * 65)

    # 1. Generate data
    print("\n[1] Generating synthetic COMPAS dataset...")
    df = generate_compas_dataset(n=5000, seed=42)
    print(f"    Total records: {len(df)}")
    print(f"    Race distribution:\n{df['race_label'].value_counts().to_string()}")
    print(f"    Overall recidivism rate: {df['two_year_recid'].mean():.1%}")

    by_race = df.groupby("race_label")["two_year_recid"].agg(["mean", "count"])
    by_race.columns = ["recid_rate", "count"]
    print(f"\n    Recidivism by Race:\n{by_race.round(4).to_string()}")

    # 2. COMPAS score analysis (score >= 5 = high risk)
    print("\n[2] COMPAS Score Analysis (threshold=5)...")
    df["compas_high_risk"] = (df["compas_score"] >= 5).astype(int)

    group_arr = df["race_label"].values
    y_true_arr = df["two_year_recid"].values
    y_pred_arr = df["compas_high_risk"].values

    metrics = compute_fairness_metrics(y_true_arr, y_pred_arr, group_arr)
    print("\n    Fairness Metrics by Group:")
    print(metrics[["n", "base_rate", "accuracy", "TPR", "FPR",
                    "FNR", "PPV"]].to_string())

    # 3. Disparate impact
    print("\n[3] Disparate Impact Analysis...")
    di = disparate_impact(y_pred_arr, group_arr, "White", "Black")
    for k, v in di.items():
        print(f"    {k}: {v}")

    # 4. ProPublica-style findings
    print("\n[4] ProPublica-Style Key Findings...")
    if "Black" in metrics.index and "White" in metrics.index:
        fpr_black = metrics.loc["Black", "FPR"]
        fpr_white = metrics.loc["White", "FPR"]
        fnr_black = metrics.loc["Black", "FNR"]
        fnr_white = metrics.loc["White", "FNR"]

        print(f"""
    ┌─────────────────────────────────────────────────────────┐
    │  FALSE POSITIVE RATE (falsely labelled HIGH risk)      │
    │  Black defendants: {fpr_black:.1%}                             │
    │  White defendants: {fpr_white:.1%}                             │
    │  Ratio: {fpr_black/fpr_white:.2f}x — Black defendants are more    │
    │         likely to be WRONGLY flagged as high risk       │
    │                                                         │
    │  FALSE NEGATIVE RATE (wrongly labelled LOW risk)       │
    │  Black defendants: {fnr_black:.1%}                             │
    │  White defendants: {fnr_white:.1%}                             │
    │  Ratio: {fnr_white/fnr_black:.2f}x — White defendants more often  │
    │         escape HIGH risk label even when they reoffend  │
    └─────────────────────────────────────────────────────────┘
        """)

    # 5. Train logistic regression (with and without race)
    print("[5] Training Logistic Regression (with race)...")
    features_with_race = ["age", "priors_count", "charge_degree",
                          "race", "sex", "juv_fel_count", "juv_misd_count"]
    features_no_race = ["age", "priors_count", "charge_degree",
                        "sex", "juv_fel_count", "juv_misd_count"]

    _, _, X_test, y_test, y_pred_with, y_score_with = train_model(
        df, features_with_race)
    _, _, X_test2, y_test2, y_pred_no, y_score_no = train_model(
        df, features_no_race)

    print(f"    With race    — Accuracy: {accuracy_score(y_test, y_pred_with):.3f}, "
          f"AUC: {roc_auc_score(y_test, y_score_with):.3f}")
    print(f"    Without race — Accuracy: {accuracy_score(y_test2, y_pred_no):.3f}, "
          f"AUC: {roc_auc_score(y_test2, y_score_no):.3f}")
    print("    → Removing 'race' barely changes accuracy, but DOES change")
    print("      who bears the cost of errors (proxy variables take over)")

    # 6. Calibration by group
    print("\n[6] Calibration by Group...")
    y_score_full = df["compas_score"].values / 10.0  # normalize 1-10 → 0-1
    cal_df = calibration_by_group(y_true_arr, y_score_full, group_arr, n_bins=5)
    print(cal_df.to_string(index=False))

    # 7. Impossibility theorem
    explain_impossibility_theorem(metrics)

    # 8. Mitigation strategies
    mitigation_strategies_overview()

    # 9. Summary
    print("\n" + "=" * 65)
    print("  TAKEAWAYS FOR ML PRACTITIONERS")
    print("=" * 65)
    print("""
  1. ALWAYS disaggregate metrics by protected attributes
     before deploying any high-stakes model.

  2. Overall accuracy is meaningless for fairness evaluation.
     Look at FPR and FNR separately for each group.

  3. There is NO single "fair" metric — different definitions
     of fairness are mutually exclusive when base rates differ.

  4. Removing protected features does NOT remove bias.
     Proxy variables (zip code, prior arrests) encode it.

  5. Fairness is a policy decision, not just a technical one.
     Involve ethicists, lawyers, and affected communities.

  6. Document fairness trade-offs explicitly in model cards.
  """)


if __name__ == "__main__":
    run_compas_analysis()
