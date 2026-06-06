\# Phase 4 Architecture Design Document

\## ML Failure Engine — Root Cause Analysis



\---



\## 1. COMPLETE FOLDER STRUCTURE



```

ml\_failure\_engine/

│

├── scratch/

│   └── phase4/

│       ├── \_\_init\_\_.py

│       │

│       ├── label\_noise/

│       │   ├── \_\_init\_\_.py

│       │   ├── confident\_learning.py        ← Northcutt et al. 2021

│       │   ├── cleanlab\_integration.py      ← cleanlab library wrapper

│       │   └── asymmetric\_noise.py          ← noise transition matrices

│       │

│       ├── feature\_leakage/

│       │   ├── \_\_init\_\_.py

│       │   ├── target\_leakage.py            ← correlation + permutation spike

│       │   ├── temporal\_leakage.py          ← time-ordering violations

│       │   └── leakage\_scanner.py           ← unified leakage detector

│       │

│       ├── missing\_data/

│       │   ├── \_\_init\_\_.py

│       │   ├── missingness\_classifier.py    ← MCAR / MAR / MNAR detection

│       │   ├── little\_mcar\_test.py          ← Little's test from scratch

│       │   ├── multiple\_imputation.py       ← MICE from scratch

│       │   └── missingness\_as\_signal.py     ← missingness indicator features

│       │

│       └── causal/

│           ├── \_\_init\_\_.py

│           ├── dag\_builder.py               ← DAG construction + validation

│           ├── confounder\_detector.py       ← confounders, colliders, mediators

│           ├── simpsons\_paradox.py          ← detection + demonstration

│           ├── potential\_outcomes.py        ← ATE, ATT from scratch

│           ├── diff\_in\_diff.py              ← DiD estimator from scratch

│           └── dowhy\_integration.py         ← DoWhy library wrapper

│

├── engine/

│   └── modules/

│       ├── label\_noise\_engine.py            ← production label audit

│       ├── leakage\_engine.py                ← production leakage scanner

│       ├── missing\_data\_engine.py           ← production missingness audit

│       └── causal\_engine.py                 ← production causal analysis

│

└── tests/

&#x20;   ├── test\_label\_noise.py

&#x20;   ├── test\_feature\_leakage.py

&#x20;   ├── test\_missing\_data.py

&#x20;   └── test\_causal.py

```



\---



\## 2. MODULE DESIGN



\### 2.1 Label Noise Modules



\---



\#### `confident\_learning.py`

\*\*Purpose:\*\* Implement Confident Learning algorithm (Northcutt, Jiang, Chuang 2021 — JAIR) from scratch without cleanlab dependency.



\*\*Theory:\*\*

Confident Learning estimates the joint distribution `Q\[s,y]` between noisy labels `s` and true labels `y` using out-of-fold predicted probabilities. It identifies label errors as instances where the model is confidently predicting a different class than the given label.



\*\*Key equation:\*\*

```

C̃\[s,y] = #{x : p̂(y|x) >= t\_y AND s\_x = s}

where t\_y = (1/|Xs|) Σ p̂(y|x)   ← per-class threshold

```



\*\*Design principle:\*\* Stateless functions only. No global state. Every function takes data in, returns results out.



\---



\#### `cleanlab\_integration.py`

\*\*Purpose:\*\* Wrap cleanlab library with project-standard API. Provide fallback to `confident\_learning.py` when cleanlab unavailable.



\*\*Design principle:\*\* Adapter pattern. The rest of the project never imports cleanlab directly — only through this module. This isolates the dependency.



\---



\#### `asymmetric\_noise.py`

\*\*Purpose:\*\* Model and detect asymmetric label noise (some classes more likely to be mislabeled than others).



\*\*Theory:\*\*

Symmetric noise: each label flipped with equal probability `ε`.

Asymmetric noise: label `i` is flipped to label `j` with probability `T\[i,j]` where T is the noise transition matrix.



Real-world example: "cat" mislabeled as "dog" more often than "dog" as "cat".



\*\*Key output:\*\* Noise transition matrix `T` where `T\[i,j] = P(ŝ=j | y=i)`.



\---



\### 2.2 Feature Leakage Modules



\---



\#### `target\_leakage.py`

\*\*Purpose:\*\* Detect features that encode the target variable, either directly or through a proxy.



\*\*Three detection strategies:\*\*

1\. Mutual information spike — I(feature; target) >> I(other features; target)

2\. Permutation importance spike — one feature dominates after permutation

3\. Temporal availability check — feature computed using future information



\*\*Design principle:\*\* Every check returns a severity score 0–1 plus a human-readable explanation. The scanner aggregates these.



\---



\#### `temporal\_leakage.py`

\*\*Purpose:\*\* Detect time-ordering violations in train/test splits.



\*\*Key scenarios to detect:\*\*

\- Feature value computed using data from after the prediction timestamp

\- Test set contains timestamps that precede training set timestamps

\- Rolling window features that look into the future

\- Target encoding computed on the full dataset before splitting



\---



\#### `leakage\_scanner.py`

\*\*Purpose:\*\* Unified entry point. Runs all leakage detectors, aggregates results, produces ranked report.



\*\*Design principle:\*\* Orchestrator only. Contains no detection logic — delegates entirely to `target\_leakage.py` and `temporal\_leakage.py`.



\---



\### 2.3 Missing Data Modules



\---



\#### `missingness\_classifier.py`

\*\*Purpose:\*\* Classify which missingness mechanism is present.



\*\*Three mechanisms (Rubin 1976):\*\*



| Mechanism | Definition | Example | Implication |

|---|---|---|---|

| MCAR | P(missing) independent of all data | Random equipment failure | Safe to drop rows |

| MAR | P(missing) depends on observed data only | Older patients skip income question | Must impute carefully |

| MNAR | P(missing) depends on the missing value itself | High earners omit salary | Most dangerous — selection bias |



\*\*Design principle:\*\* MCAR can be tested (Little's test). MAR vs MNAR cannot be distinguished from data alone — requires domain knowledge. The module clearly communicates this limitation.



\---



\#### `little\_mcar\_test.py`

\*\*Purpose:\*\* Implement Little's (1988) test for MCAR from scratch.



\*\*Theory:\*\*

Partitions data into groups with identical missingness patterns. Tests whether group means differ significantly from overall means under the null hypothesis of MCAR. Uses chi-square distribution.



\*\*Why from scratch:\*\* This test is rarely implemented in standard libraries correctly. Building it forces understanding of the underlying statistics.



\---



\#### `multiple\_imputation.py`

\*\*Purpose:\*\* Implement MICE (Multiple Imputation by Chained Equations) from scratch.



\*\*Algorithm:\*\*

```

1\. Initial fill: replace NaN with column mean/mode

2\. For each feature with missingness (in sequence):

&#x20;  a. Treat it as target variable

&#x20;  b. Use all other features as predictors

&#x20;  c. Fit simple model on observed rows

&#x20;  d. Predict missing values

3\. Repeat step 2 for N cycles until convergence

4\. Produce M complete datasets (M=5 typical)

5\. Analysis: run model on all M datasets, pool results (Rubin's rules)

```



\*\*Design principle:\*\* Each imputation cycle is a pure function. State between cycles passed explicitly. This makes the algorithm testable at each step.



\---



\#### `missingness\_as\_signal.py`

\*\*Purpose:\*\* Treat the pattern of missingness itself as a predictive feature.



\*\*Key insight:\*\* Whether data is missing is often MORE informative than the imputed value. A patient skipping a lab test may indicate the doctor didn't think it was needed — itself a clinical signal.



\*\*Outputs:\*\* Binary indicator matrix `M` where `M\[i,j] = 1` if `X\[i,j]` is missing. Correlation between indicators and target. Clustering of missingness patterns.



\---



\### 2.4 Causal Modules



\---



\#### `dag\_builder.py`

\*\*Purpose:\*\* Construct and validate Directed Acyclic Graphs representing causal assumptions.



\*\*Core operations:\*\*

\- Add nodes (variables), add directed edges (causal claims)

\- Detect cycles (a DAG must be acyclic — cycle = logical contradiction)

\- Find paths between nodes (direct, backdoor, frontdoor)

\- Identify adjustment sets for a given query (X → Y)



\*\*Design principle:\*\* Pure graph operations. No statistics here. The DAG represents ASSUMPTIONS, not data-derived facts.



\---



\#### `confounder\_detector.py`

\*\*Purpose:\*\* Identify confounders, colliders, and mediators given a DAG and a causal query.



\*\*Three structural roles:\*\*



| Role | Structure | Effect of conditioning | Action |

|---|---|---|---|

| Confounder | Z → X, Z → Y | Blocks backdoor path | MUST adjust for Z |

| Collider | X → Z ← Y | OPENS path (selection bias) | NEVER adjust for Z |

| Mediator | X → Z → Y | Blocks direct effect | Depends on query |



\*\*Design principle:\*\* These are graph-theoretic properties, not statistical ones. Computed entirely from DAG structure using d-separation rules.



\---



\#### `simpsons\_paradox.py`

\*\*Purpose:\*\* Detect and demonstrate Simpson's Paradox — when a trend present in subgroups reverses when groups are combined.



\*\*Detection algorithm:\*\*

1\. Compute correlation(X, Y) in aggregate

2\. For each possible confounder Z, compute correlation(X, Y) within each level of Z

3\. Paradox detected if sign(aggregate correlation) ≠ sign(all stratum correlations)



\*\*Design principle:\*\* Include a synthetic data generator that produces a clean paradox example. This is used in tests and as a teaching tool.



\---



\#### `potential\_outcomes.py`

\*\*Purpose:\*\* Implement Rubin's Potential Outcomes framework from scratch.



\*\*Core concepts:\*\*

\- `Y(1)` = outcome if treated (counterfactual for untreated units)

\- `Y(0)` = outcome if untreated (counterfactual for treated units)

\- Fundamental problem: we observe only one potential outcome per unit

\- ATE = E\[Y(1) - Y(0)] — average treatment effect across all units

\- ATT = E\[Y(1) - Y(0) | T=1] — effect specifically on treated units



\*\*Estimators implemented:\*\*

\- Naive difference in means (biased without randomization)

\- IPW (Inverse Probability Weighting) using propensity scores

\- Matching estimator (nearest-neighbor)



\---



\#### `diff\_in\_diff.py`

\*\*Purpose:\*\* Implement Difference-in-Differences estimator from scratch.



\*\*Setup:\*\* Two groups (treatment, control), two time periods (pre, post).



\*\*Estimator:\*\*

```

DiD = (Ȳ\_treated\_post - Ȳ\_treated\_pre) - (Ȳ\_control\_post - Ȳ\_control\_pre)

```



\*\*Key assumption:\*\* Parallel trends — in the absence of treatment, treated and control groups would have followed parallel trajectories. This assumption is UNTESTABLE but can be assessed via pre-treatment trend analysis.



\*\*Design principle:\*\* The module explicitly tests and reports on the parallel trends assumption. A DiD estimate without this check is incomplete.



\---



\#### `dowhy\_integration.py`

\*\*Purpose:\*\* Wrap DoWhy library for production causal inference. Provides fallback to `potential\_outcomes.py` when DoWhy unavailable.



\*\*Four-step DoWhy workflow:\*\*

1\. Model — define causal graph

2\. Identify — find estimand using do-calculus

3\. Estimate — compute effect

4\. Refute — sensitivity analysis (placebo test, random common cause, data subset)



\---



\## 3. FUNCTION LIST



\### `confident\_learning.py`

```

compute\_thresholds(y\_proba, y\_noisy)          → per-class thresholds t\_y

estimate\_joint\_distribution(y\_proba, y\_noisy) → C̃\[s,y] matrix

identify\_label\_errors(y\_proba, y\_noisy)       → indices of suspected errors

label\_error\_fraction(y\_proba, y\_noisy)        → estimated noise rate ε

rank\_by\_label\_quality(y\_proba, y\_noisy)       → instances ranked by confidence

noise\_transition\_matrix(y\_proba, y\_noisy)     → T\[i,j] = P(label=j | true=i)

run\_verification()

```



\### `cleanlab\_integration.py`

```

find\_label\_issues(X, y, model)                → label issue indices + scores

label\_quality\_scores(X, y, model)             → per-sample quality score 0-1

clean\_dataset(X, y, model, threshold)         → filtered X, y

run\_verification()

```



\### `asymmetric\_noise.py`

```

inject\_asymmetric\_noise(y, T, seed)           → noisy labels y\_tilde

estimate\_noise\_matrix(y\_proba, y\_noisy)       → estimated T

detect\_asymmetric\_noise(y\_proba, y\_noisy)     → noise type + severity

noise\_matrix\_report(T)                        → human-readable report

run\_verification()

```



\### `target\_leakage.py`

```

mutual\_information\_score(X, y, feature\_idx)   → MI score for feature

permutation\_importance\_spike(model, X, y)     → spike detection result

availability\_check(df, feature\_col, time\_col) → temporal validity flag

target\_correlation\_scan(X, y, feat\_names)     → ranked correlation report

leakage\_score(X, y, model, feat\_names)        → per-feature leakage score 0-1

run\_verification()

```



\### `temporal\_leakage.py`

```

check\_split\_ordering(timestamps, train\_idx, test\_idx)  → ordering violation

detect\_future\_features(df, feature\_cols, time\_col)     → future-feature list

rolling\_window\_audit(df, window\_cols, time\_col)        → window validity

target\_encoding\_leakage(df, cat\_col, target\_col)       → encoding leakage flag

temporal\_leakage\_report(df, time\_col, target\_col)      → full temporal audit

run\_verification()

```



\### `leakage\_scanner.py`

```

scan(X, y, df, model, feature\_names, time\_col)  → unified leakage report

rank\_leakage\_suspects(report)                    → prioritised suspect list

leakage\_summary(report)                          → plain-English summary

run\_verification()

```



\### `little\_mcar\_test.py`

```

missingness\_pattern\_groups(X)                → groups by missingness pattern

little\_test\_statistic(X)                     → chi-square statistic

little\_mcar\_test(X, alpha)                   → {statistic, pvalue, mcar\_likely}

run\_verification()

```



\### `missingness\_classifier.py`

```

compute\_missingness\_rate(X)                  → per-feature miss rate

missingness\_indicator\_matrix(X)              → binary M matrix

correlate\_missingness\_with\_observed(X)       → MAR evidence matrix

correlate\_missingness\_with\_target(X, y)      → MNAR evidence vector

classify\_mechanism(X, y, alpha)              → {feature: mechanism} dict

run\_verification()

```



\### `multiple\_imputation.py`

```

initial\_fill(X)                              → mean/mode filled X

mice\_cycle(X, n\_cycles)                      → one complete MICE pass

mice\_impute(X, m, n\_cycles)                  → m complete datasets

rubin\_rules(estimates, variances)            → pooled estimate + CI

run\_verification()

```



\### `missingness\_as\_signal.py`

```

create\_indicator\_features(X, feature\_names)          → X\_augmented with M cols

missingness\_pattern\_clusters(X, n\_clusters)          → cluster assignments

indicator\_target\_correlation(M, y)                   → sorted correlation list

should\_include\_indicators(M, y, threshold)            → {feature: include} dict

run\_verification()

```



\### `dag\_builder.py`

```

DAG class:

&#x20; add\_node(name, label)

&#x20; add\_edge(from\_node, to\_node)

&#x20; has\_cycle()                                → bool

&#x20; get\_parents(node)                          → list

&#x20; get\_children(node)                         → list

&#x20; get\_ancestors(node)                        → set

&#x20; all\_paths(source, target)                  → list of paths

&#x20; backdoor\_paths(X, Y)                       → list of backdoor paths

&#x20; is\_d\_separated(X, Y, given\_set)            → bool

&#x20; adjustment\_set(X, Y)                       → minimal adjustment set

&#x20; to\_dict()                                  → serialisable dict

run\_verification()

```



\### `confounder\_detector.py`

```

classify\_node\_role(dag, X, Y, Z)             → confounder|collider|mediator|none

find\_all\_confounders(dag, X, Y)              → list of confounders

find\_all\_colliders(dag, X, Y)                → list of colliders

find\_mediators(dag, X, Y)                    → list of mediators

should\_adjust\_for(dag, X, Y, Z)              → bool + reason

adjustment\_warning(dag, X, Y, proposed\_set)  → warnings list

run\_verification()

```



\### `simpsons\_paradox.py`

```

aggregate\_correlation(X, y)                       → float

stratum\_correlations(X, y, Z)                     → {stratum: correlation}

detect\_simpsons\_paradox(X, y, Z)                  → {detected, aggregate, strata}

generate\_paradox\_example(seed)                    → X, y, Z with known paradox

paradox\_explanation(result)                       → plain-English explanation

run\_verification()

```



\### `potential\_outcomes.py`

```

naive\_ate(y, treatment)                           → naive mean difference

propensity\_score(X, treatment)                    → P(T=1|X) per unit

ipw\_ate(y, treatment, propensity)                 → IPW ATE estimate

matching\_ate(X, y, treatment, k)                  → matched ATE estimate

matching\_att(X, y, treatment, k)                  → matched ATT estimate

overlap\_check(propensity)                         → positivity violation report

run\_verification()

```



\### `diff\_in\_diff.py`

```

did\_estimate(y\_pre\_treat, y\_post\_treat,

&#x20;            y\_pre\_ctrl,  y\_post\_ctrl)            → DiD estimate

parallel\_trends\_test(y\_treat\_history,

&#x20;                    y\_ctrl\_history,  periods)    → {parallel, p\_value}

did\_regression(df, outcome, treatment,

&#x20;              time, covariates)                  → {estimate, se, ci, p}

event\_study(df, outcome, treatment,

&#x20;           time\_col, n\_pre, n\_post)              → per-period estimates

did\_report(estimate, parallel\_trends)             → plain-English report

run\_verification()

```



\### `dowhy\_integration.py`

```

build\_causal\_model(df, treatment, outcome, dag\_str)  → CausalModel

identify\_effect(model)                               → identified estimand

estimate\_effect(model, estimand, method)             → CausalEstimate

refute\_estimate(model, estimand, estimate)           → refutation results

full\_causal\_analysis(df, treatment, outcome,

&#x20;                    dag\_str, method)                → complete report

run\_verification()

```



\---



\## 4. TEST STRATEGY



\### Philosophy

Every test module follows the same three-layer structure:



```

Layer 1: Unit tests    — each function in isolation with known inputs/outputs

Layer 2: Integration   — functions working together on synthetic data

Layer 3: Oracle tests  — inject known ground truth, verify recovery

```



Oracle testing is the most important layer for Phase 4. Because we are testing things like label noise and causal effects, we can synthetically inject known quantities and verify recovery.



\---



\### `test\_label\_noise.py`



\*\*Oracle tests (most important):\*\*

\- Inject 15% symmetric noise. Verify `identify\_label\_errors` recovers ≥ 80% of injected errors.

\- Inject known asymmetric noise matrix T. Verify `noise\_transition\_matrix` estimates match within tolerance 0.05.

\- Generate clean labels, zero noise. Verify `label\_error\_fraction` returns < 0.02.



\*\*Edge cases:\*\*

\- Single-class dataset (should raise informative error)

\- All labels identical

\- n\_samples < 50 (small dataset warning)

\- Perfect model (all probabilities = 1.0)



\*\*Regression tests:\*\*

\- Verify cleanlab integration returns same order of label issues as our implementation on standard dataset (allow rank correlation ≥ 0.7)



\---



\### `test\_feature\_leakage.py`



\*\*Oracle tests:\*\*

\- Inject a feature that is literally `y + noise(0.01)`. Verify leakage score ≥ 0.9.

\- Inject a feature computed from future timestamps. Verify temporal check flags it.

\- Five random noise features. Verify none score above 0.3.



\*\*Edge cases:\*\*

\- Dataset with no timestamps (temporal leakage should return gracefully)

\- All features perfectly correlated with each other (not with target)

\- Single-feature dataset



\---



\### `test\_missing\_data.py`



\*\*Oracle tests (MCAR):\*\*

\- Generate complete data. Drop 10% entries uniformly at random (true MCAR). Verify `little\_mcar\_test` returns `mcar\_likely=True`.



\*\*Oracle tests (MAR):\*\*

\- Make missingness in column B depend on values of column A. Verify `correlate\_missingness\_with\_observed` flags A-B correlation.



\*\*Oracle tests (MNAR):\*\*

\- Make high-value entries of column C missing with higher probability. Verify `classify\_mechanism` flags C as suspected MNAR.



\*\*MICE verification:\*\*

\- Generate data from known linear model. Inject 20% MCAR. Impute via MICE. Verify imputed coefficient estimates within 10% of true values.



\*\*Edge cases:\*\*

\- No missing data (all functions should return gracefully)

\- Entire column missing (should flag as MNAR candidate)

\- Single row missing



\---



\### `test\_causal.py`



\*\*DAG tests:\*\*

\- Build DAG with known cycle. Verify `has\_cycle()` returns True.

\- Build chain A→B→C. Verify `is\_d\_separated(A, C, {B})` returns True.

\- Build fork A←B→C. Verify `is\_d\_separated(A, C, {B})` returns True.

\- Build collider A→B←C. Verify `is\_d\_separated(A, C, {})` returns True AND `is\_d\_separated(A, C, {B})` returns False.



\*\*Simpson's Paradox tests:\*\*

\- Use `generate\_paradox\_example`. Verify `detect\_simpsons\_paradox` flags it correctly.

\- Generate data with no paradox. Verify function returns `detected=False`.



\*\*DiD oracle tests:\*\*

\- Simulate known treatment effect of +5. Verify `did\_estimate` recovers 5 ± 0.5.

\- Simulate parallel trends pre-treatment. Verify `parallel\_trends\_test` passes.

\- Violate parallel trends. Verify test flags it.



\*\*ATE oracle tests:\*\*

\- Simulate RCT (random treatment assignment). Verify `naive\_ate` and `ipw\_ate` give similar estimates.

\- Simulate confounded observational study. Verify `naive\_ate` is biased and `ipw\_ate` is closer to truth.



\---



\## 5. ENGINE INTEGRATION PLAN



\### Design Contract

Each engine module exposes exactly one class following this interface:



```python

class PhaseNEngine:

&#x20;   def \_\_init\_\_(self, config: dict)

&#x20;   def fit(self, X, y, \*\*kwargs) → self

&#x20;   def analyze(self) → dict          # returns structured report

&#x20;   def report(self) → str            # returns plain-English string

&#x20;   def to\_json(self) → dict          # returns JSON-serialisable result

```



This contract means `report\_engine.py` from Phase 2 can consume any Phase 4 engine without modification.



\---



\### `label\_noise\_engine.py`

\*\*Wraps:\*\* `confident\_learning.py` + `cleanlab\_integration.py` + `asymmetric\_noise.py`



\*\*Inputs:\*\* model, X\_train, y\_train (cross-val probabilities computed internally)



\*\*Output report sections:\*\*

\- Estimated noise rate

\- Top-N suspected label errors with per-instance quality score

\- Noise transition matrix (if asymmetric)

\- Recommended action: relabel / drop / investigate



\*\*Production behaviour:\*\* Falls back to `confident\_learning.py` if cleanlab not installed. Logs which backend was used.



\---



\### `leakage\_engine.py`

\*\*Wraps:\*\* `target\_leakage.py` + `temporal\_leakage.py` + `leakage\_scanner.py`



\*\*Inputs:\*\* X\_train, y\_train, feature\_names, timestamps (optional), model (optional)



\*\*Output report sections:\*\*

\- Leakage risk per feature (0–1 score)

\- Severity: CRITICAL / WARNING / NONE

\- Explanation of each flagged feature

\- Recommended action per feature



\*\*Production behaviour:\*\* Runs without model for pre-training checks. Runs with model for post-training permutation checks.



\---



\### `missing\_data\_engine.py`

\*\*Wraps:\*\* `missingness\_classifier.py` + `little\_mcar\_test.py` + `multiple\_imputation.py` + `missingness\_as\_signal.py`



\*\*Inputs:\*\* X (with NaN), y, feature\_names



\*\*Output report sections:\*\*

\- Missingness rate per feature

\- Mechanism classification (MCAR / MAR / MNAR / unknown)

\- Indicator-target correlations

\- Imputation recommendation



\*\*Production behaviour:\*\* Never imputes silently. Always logs what mechanism was assumed and why. If MNAR suspected, raises explicit warning that imputation may be biased.



\---



\### `causal\_engine.py`

\*\*Wraps:\*\* `dag\_builder.py` + `confounder\_detector.py` + `simpsons\_paradox.py` + `potential\_outcomes.py` + `diff\_in\_diff.py` + `dowhy\_integration.py`



\*\*Inputs:\*\* df, treatment\_col, outcome\_col, dag\_edges (optional), time\_col (optional)



\*\*Output report sections:\*\*

\- DAG structure summary

\- Identified confounders, colliders, mediators

\- Simpson's paradox check

\- ATE / ATT estimate with confidence interval

\- Parallel trends test result (if DiD)

\- Refutation test results (if DoWhy)



\*\*Production behaviour:\*\* If no DAG provided, operates in exploratory mode only (no causal claims made). Clearly distinguishes association from causation in all outputs.



\---



\## 6. VERIFICATION PLAN



\### Verification runner: `run\_phase4.py`



Structure mirrors `run\_phase2.py` and `run\_phase3.py` exactly.



```

run\_phase4.py

&#x20; ├── verify\_label\_noise()

&#x20; ├── verify\_feature\_leakage()

&#x20; ├── verify\_missing\_data()

&#x20; ├── verify\_causal()

&#x20; └── print\_phase4\_summary()

```



\### Verification datasets



| Dataset | Purpose | Source |

|---|---|---|

| Synthetic clean → inject noise | Label noise oracle tests | `confident\_learning.py::generate\_noisy\_dataset` |

| UCI Adult (Census) | Slicing + leakage baseline | sklearn fetch |

| Synthetic with MCAR/MAR/MNAR | Missing data oracle tests | `missingness\_classifier.py::generate\_missing\_dataset` |

| Simulated RCT | ATE oracle test | `potential\_outcomes.py::generate\_rct` |

| Simulated observational study | Confounding bias test | `potential\_outcomes.py::generate\_confounded` |

| Classic paradox table | Simpson's detection | `simpsons\_paradox.py::generate\_paradox\_example` |



\### Pass criteria (minimum for phase completion)



| Module | Pass Criterion |

|---|---|

| `confident\_learning` | Recall ≥ 0.75 on injected noise oracle |

| `asymmetric\_noise` | Matrix estimation error < 0.05 per cell |

| `target\_leakage` | F1 ≥ 0.85 on injected leakage oracle |

| `little\_mcar\_test` | Correct classification on 3 known-mechanism datasets |

| `multiple\_imputation` | Coefficient recovery within 10% on linear oracle |

| `dag\_builder` | All d-separation tests pass |

| `did\_estimate` | Effect recovery within 10% on simulation oracle |

| `ipw\_ate` | Bias reduction vs naive estimator on confounded simulation |



\---



\## 7. IMPLEMENTATION ORDER



\### Rationale for ordering

Dependencies flow left to right. Each stage must be complete before the next begins. Engine modules are built last — they depend on all scratch implementations being stable.



```

Week 1 — Label Noise (standalone, no causal dependencies)

&#x20; Day 1-2:  confident\_learning.py         + unit tests

&#x20; Day 3:    asymmetric\_noise.py           + unit tests

&#x20; Day 4:    cleanlab\_integration.py       + integration tests

&#x20; Day 5:    label\_noise\_engine.py         + test\_label\_noise.py



Week 2 — Feature Leakage + Missing Data

&#x20; Day 1-2:  target\_leakage.py             + temporal\_leakage.py

&#x20; Day 3:    leakage\_scanner.py            + leakage\_engine.py

&#x20; Day 4:    little\_mcar\_test.py           + missingness\_classifier.py

&#x20; Day 5:    multiple\_imputation.py        + missingness\_as\_signal.py

&#x20;           missing\_data\_engine.py        + all leakage + missing tests



Week 3 — Causal (builds on all previous)

&#x20; Day 1:    dag\_builder.py                + confounder\_detector.py

&#x20; Day 2:    simpsons\_paradox.py           + potential\_outcomes.py

&#x20; Day 3:    diff\_in\_diff.py

&#x20; Day 4:    dowhy\_integration.py          + causal\_engine.py

&#x20; Day 5:    test\_causal.py               + run\_phase4.py

&#x20;           Full integration verification

```



\### Hard dependency rules



\- `causal\_engine.py` depends on `leakage\_engine.py` being complete — causal analysis can confirm what leakage detection suspects

\- `missing\_data\_engine.py` must be complete before `causal\_engine.py` — missing data patterns affect causal identification

\- `cleanlab\_integration.py` must have a working fallback before it is used in engine layer — never a hard dependency



\### What NOT to implement in Phase 4



\- Neural network-based imputation (Phase 5 territory)

\- Continuous treatment effects (requires kernel methods — out of scope)

\- Time-series causal models (VAR, Granger causality — Phase 5)

\- Any Streamlit UI components (Phase 5)



\---



\*End of Phase 4 Architecture Design Document\*

