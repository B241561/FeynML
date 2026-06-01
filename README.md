# ML Failure Investigation Engine

A complete, from-scratch implementation of the ML Failure Investigation syllabus.
Every algorithm is built without high-level ML libraries in core implementations,
then verified against scikit-learn.

---

## 📁 Folder Structure

```
ml_failure_engine/
│
├── scratch/                        ← Learning implementations (from scratch)
│   ├── phase0/                     ← Prerequisites: Math & Python foundations
│   │   ├── linear_algebra.py       Vector ops, matrix ops, Gaussian elimination,
│   │   │                           eigenvectors (power iteration), covariance matrix
│   │   ├── calculus_optimization.py Derivatives, gradient descent, chain rule,
│   │   │                           Adam optimizer, numerical differentiation
│   │   ├── probability_statistics.py Distributions, Bayes theorem, hypothesis testing,
│   │   │                           t-test, chi-squared, MLE from scratch
│   │   └── statistics.py           Descriptive stats, correlation, confidence intervals,
│   │                               bootstrapping, effect sizes
│   │
│   ├── phase1/                     ← Core ML Algorithms (all from scratch)
│   │   ├── linear_regression.py    OLS, gradient descent, normal equation,
│   │   │                           Ridge (L2), Lasso (L1 coordinate descent)
│   │   ├── logistic_regression.py  Sigmoid (log-odds derivation), BCE loss (MLE),
│   │   │                           binary + L2 regularized + softmax multiclass
│   │   ├── decision_tree.py        CART, Gini & entropy, recursive splitting,
│   │   │                           classifier + regressor, feature importance
│   │   ├── ensemble_models.py      Random Forest (bagging + OOB score),
│   │   │                           Gradient Boosting (residual fitting),
│   │   │                           AdaBoost (sample reweighting)
│   │   ├── kmeans_pca.py           K-Means++ init, elbow method, silhouette score,
│   │   │                           PCA via eigendecomposition + deflation
│   │   └── run_phase1.py           ← Run all Phase 1 verifications
│   │
│   └── phase2/                     ← Model Evaluation & Failure Analysis
│       ├── classification_metrics.py Confusion matrix, precision/recall/F1,
│       │                            ROC-AUC, PR-AUC, calibration, ECE
│       ├── regression_metrics.py   MSE, RMSE, MAE, MAPE, SMAPE, R², adj-R²,
│       │                           residual analysis, bias detection
│       ├── validation_strategy.py  K-fold CV, stratified CV, nested CV,
│       │                           leakage detection, temporal split
│       ├── statistical_significance.py McNemar's test, bootstrap CI, permutation test,
│       │                           effect sizes, multiple testing correction
│       ├── compas_case_study.py    Full COMPAS fairness investigation:
│       │                           disparate FPR, calibration, impossibility theorem
│       └── (fairness_audit.py)     ← Covered inside compas_case_study.py
│
├── engine/                         ← Production pipeline modules
│   ├── failure_report_generator.py Automated HTML report generator
│   └── modules/                    ← Pluggable analysis modules
│       └── (metric_dashboard.py)   ← Add your Streamlit dashboard here
│
├── reports/                        ← Generated outputs
│   ├── compas_findings.json        Structured findings (machine-readable)
│   └── compas_investigation_report.html  Full HTML investigation report
│
├── tests/                          ← Unit tests
│   └── (add test_phase1.py, test_phase2.py here)
│
├── requirements.txt                ← All dependencies
└── README.md                       ← This file
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Phase 0 (math foundations)
python scratch/phase0/linear_algebra.py
python scratch/phase0/calculus_optimization.py
python scratch/phase0/statistics.py
python scratch/phase0/probability_statistics.py

# 3. Run Phase 1 (all ML algorithms)
python scratch/phase1/run_phase1.py

# 4. Run Phase 2 (evaluation & fairness)
python scratch/phase2/classification_metrics.py
python scratch/phase2/compas_case_study.py

# 5. Generate COMPAS investigation report
python engine/failure_report_generator.py
```

---

## 📚 Phase Descriptions

### Phase 0 — Prerequisites
Pure Python/NumPy implementations of all required math:
- **linear_algebra.py** — vectors, matrices, Gaussian elimination, eigendecomposition
- **calculus_optimization.py** — derivatives, gradient descent variants (SGD, Adam, RMSProp)
- **statistics.py** — descriptive stats, correlation, confidence intervals, bootstrapping
- **probability_statistics.py** — distributions, Bayes, t-test, chi-squared from scratch

### Phase 1 — Core ML Algorithms
Every algorithm built from scratch, verified against scikit-learn:
- **linear_regression.py** — OLS, GD, Normal Equation, Ridge, Lasso
- **logistic_regression.py** — sigmoid from log-odds, BCE from MLE, softmax
- **decision_tree.py** — full CART implementation, Gini, entropy, feature importance
- **ensemble_models.py** — Random Forest with OOB, Gradient Boosting, AdaBoost
- **kmeans_pca.py** — K-Means++ with silhouette, PCA via eigendecomposition

### Phase 2 — Model Evaluation & Failure Investigation
Complete evaluation framework:
- **classification_metrics.py** — all classification metrics, ROC/PR curves, calibration
- **regression_metrics.py** — all regression metrics, residual analysis
- **validation_strategy.py** — cross-validation, leakage detection, temporal splits
- **statistical_significance.py** — hypothesis tests, bootstrap CI, McNemar's test
- **compas_case_study.py** — real-world fairness investigation (COMPAS model)

---

## 🏗️ Design Principles

1. **From scratch first** — understand before abstracting
2. **Verify always** — every implementation checked against sklearn
3. **Intuition comments** — every non-obvious step explained geometrically
4. **Fail explicitly** — edge cases handled, errors meaningful
5. **Production-ready** — engine/ modules follow software engineering best practices

---

## 📊 Key Outputs

After running the full pipeline:
- `reports/compas_findings.json` — structured fairness findings
- `reports/compas_investigation_report.html` — open in browser for full visual report

---

## ⚠️ Important Notes

- This uses a **synthetic COMPAS dataset** for educational purposes only
- The real COMPAS dataset raises serious ethical concerns documented by ProPublica (2016)
- The fairness impossibility theorem (Chouldechova 2017) applies: you cannot simultaneously
  achieve equal TPR, FPR, and PPV across groups when base rates differ
