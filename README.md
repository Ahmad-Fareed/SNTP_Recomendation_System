# SNTP Assignment 2 — Recommendation and Information Diffusion on Epinions

**Student:** Ahmad Fareed
**Roll number:** 23F3008
**Course:** Social Network Theory and Programming (SNTP), NU-FAST Chiniot-Faisalabad
**Deliverable:** runnable source, tests, saved settings/seeds, measured results, searchable PDF report

This repository implements all eight assignment requirements on the Epinions
rating and trust network. The submitted results were produced by the bounded
**fast-demo** configuration (`configs/fast_demo.yaml`), which uses the complete
rating/trust files and a 2-hop ego subgraph for diffusion. The larger
`configs/default.yaml` was **not executed**; no full-scale number is claimed
anywhere. See [Limitations](#12-limitations-and-deviations).

All diffusion, intervention and simulated-satisfaction values are **simulated**.
Epinions contains no adoption timestamps or cascade histories, so no observed
diffusion outcome exists to compare against.

---

## 1. Quick start

Tested with **Python 3.12.10** on Windows 11 (PowerShell). The commands are
equivalent on Linux/macOS with the usual virtual-environment activation.

```powershell
# from the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .

# 103 tests; the suite runs in about 1.7 s
python -m pytest -q

# Inspect the dataset without fitting models
python -m sntp.cli inspect --data-dir Assignment

# Bounded end-to-end demonstration; writes results/fast_demo/
python -m sntp.cli run --config configs/fast_demo.yaml
```

The measured pipeline took **474.9 s (7.9 min) wall clock** with **473.6 s of
stage time** on the development machine. Longest stages: MF training 120.2 s,
R8 intervention/CELF/scaling 140.0 s, and R5 influence maximization 69.7 s.
Wall-clock time varies with machine load; `results/fast_demo/stage_timings.json`
is the authoritative record for the submitted run.

To regenerate the searchable PDF report and the seven standalone PNG charts
(see [Section 9](#9-report-generation)):

```powershell
python -m pip install -r requirements-report.txt
python scripts/generate_report.py --results results/fast_demo `
  --output report/SNTP_Assignment_02_Report.pdf `
  --figures-dir results/fast_demo/plots `
  --roll-number 23F3008 --student "Ahmad Fareed"
```

`configs/default.yaml` is the full-scale configuration. It disables the ego
subgraph, uses larger Monte Carlo/CELF budgets and the full ranking workload.
It is expected to take many hours and was **not executed**.

---

## 2. Dataset and splits

The Epinions files are included under `Assignment/ratings_data/epinions/`:

| File | Rows | Meaning |
|---|---:|---|
| `ratings_data.txt` | 664,824 | `user item rating` (ratings 1–5) |
| `trust_data.txt` | 487,002 | `source target 1`; "source trusts target" |

Parsing found **0** malformed, out-of-range or duplicate rating records, and
**181** self-loops in the trust file were removed (487,183 raw rows → 487,002
valid directed edges). The loader discovers files with `rglob` and uses one
copy of each; a duplicated trust file is no longer shipped.

**Trust direction:** a row `A B 1` means A trusts B. Influence propagates in
the opposite direction, B → A, so the propagation graph stores edge `B -> A`
(B can influence A).

**Split:** deterministic 70/15/15 train/validation/test split with seed 42.
Users with at least three ratings contribute at least one record to each split;
users with fewer records are kept in training but excluded from evaluation.

| Split | Ratings |
|---|---:|
| Train | 463,663 |
| Validation | 100,603 |
| Test | 100,558 |
| Total | 664,824 |

Dataset summary: 49,289 users, 139,738 items, 664,824 ratings, 487,002 trust
edges, sparsity 0.999903. Mean ratings per user 16.55 (median 6.0); mean ratings
per item 4.76 (median 1.0). 40,162 users both rate and trust; 105,966 trust
edges are reciprocal. Ratings are skewed positive (301,053 fives versus 43,228
ones).

---

## 3. Requirement-to-code map

| Requirement | Implementation | Main saved evidence |
|---|---|---|
| 1. Data preparation | `src/sntp/data.py`, `src/sntp/splits.py`, `cli.py inspect` | `data_summary.json`, `split_info.json` |
| 2. Rating prediction + top-K | `recommenders/item_based.py`, `matrix_factorization.py`, `_ranking.py` | `requirement_2_results.json`, `model_selection.json` |
| 3. Trust-aware recommendation | `recommenders/social.py`, `experiments.run_requirement_3` | `requirement_3_results.json` |
| 4. ICM, SIR, preference-aware | `diffusion/icm.py`, `diffusion/sir.py`, `selection.py` | `requirement_4_results.json` |
| 5. Influence maximization | `influence.py`, `experiments.run_requirement_5` | `requirement_5_results.json` |
| 6. Group recommendation | `groups.py`, `experiments.run_requirement_6` | `requirement_6_results.json` |
| 7. Comprehensive evaluation | `metrics.py`, `experiments.run_requirement_7` | `requirement_7_results.json` |
| 8. Intervention, CELF, scaling | `experiments.run_requirement_8`, `influence.py` | `requirement_8_results.json` |

Shared experiment-design helpers live in `src/sntp/selection.py`.

### Requirement 1 — data preparation and exploration

`EpinionsDataset` parses both files, validates ratings, deduplicates exact
duplicates, builds ID mappings, a sparse rating matrix, and the directed trust
adjacency. `data_summary.json` records parsing counters and the distributions
used in the report.

### Requirement 2 — baseline recommenders

* **Item-based CF:** cosine item similarity over the training matrix, 20
  neighbours, minimum 2 co-rated items, unobserved items fall back to the
  user mean then the global mean.
* **Matrix factorisation (SGD):** 20 factors, learning rate 0.01,
  regularisation 0.005, 20 epochs, seed 42. Untrained rows are zeroed, so
  cold-start fallbacks are the documented μ+b_i, μ+b_u, μ.
* **Ranking:** all top-N paths order by **unclipped** model scores and clip
  only the reported rating. MF uses a configurable item-bias weight
  (`ranking_item_bias_weight = 0.05`) selected on validation.

The MF model was originally dominated by the global item bias (99.91% of
within-user score variance; mean top-10 overlap 9.55/10 with a bias-only
ranking). Reducing the bias weight to 0.05 widens catalogue coverage but costs
a little test nDCG@10 — the trade-off is disclosed in the report and in
`model_selection.json`:

| Metric | Pre-correction MF (w=1.0) | Selected MF (w=0.05) |
|---|---:|---:|
| Validation nDCG@10 | 0.00266 | **0.00711** |
| Validation unique items | 15 | **278** |
| Test nDCG@10 | 0.00343 | 0.00314 |
| Test unique items | ~16 | **275** |

Model, lambda and target choices use validation/training data only; test is
read once for the final table (`test_used_for_selection: false`).

### Requirement 3 — trust-aware recommendation

The social model blends the MF score with trust-neighbour and similar-user
terms. λ is selected on **validation** RMSE only:

| λ | Validation MAE | Validation RMSE |
|---:|---:|---:|
| 0.0 | 0.80337 | 1.04929 |
| **0.3** | **0.80139** | **1.04605** |
| 0.5 | 0.80423 | 1.05229 |

A separate comparison on 165 held-out ratings from 50 sampled users found
trust neighbours (RMSE 1.0701) and similar unconnected users (1.0652) almost
tied versus the baseline (1.0664), so this sample does **not** establish that
trust beats similarity.

### Requirement 4 — diffusion models

Fast-demo design: 10 simulations, 5 seed users, 2-hop ego subgraph (9,883 users,
301,710 directed edges), maximum 25 rounds. Two targets are used, both selected
from **training-only** popularity bands with the count closest to the band
median (ascending item-ID ties):

* **Rare target:** original item **3** (index 2), 1 training rating, band of
  66,605 one-rating items.
* **Moderate target:** original item **69** (index 68), 9 training ratings, band
  of 10,807 items with 6–20 training ratings.

**Common random numbers (CRN).** A sequential RNG restarted at the same integer
seed still produces different live edges when the cascade path changes. This
implementation instead pre-generates:

* **one activation threshold per directed edge** per trial, reused across every
  seed set, every p and both ICM variants;
* **one adoption threshold per user** per trial for preference-aware ICM;
* **one transmission threshold per (edge, round)** and **one recovery threshold
  per (user, round)** per trial for SIR, reused across β, γ and seed sets.

Within a trial the finite-trial objective is therefore deterministic, monotone
and submodular, so paired differences are meaningful and the CELF lazy bound is
valid. Thresholds are **not** shared across different graphs because edge
identities differ; the observed-vs-randomized comparison therefore uses each
graph's own thresholds but the same seed sets and trial seeds.

Measured results (rare target, 10 trials):

| Model | Setting | Mean | SD | Mean rounds |
|---|---|---:|---:|---:|
| Canonical ICM | p = 0.01 | 69.9 | 111.6 | 6.9 |
| Canonical ICM | p = 0.05 | 4,695.5 | 49.3 | 11.0 |
| Preference-aware ICM | p = 0.01 | 9.3 | 6.1 | 2.6 |
| Preference-aware ICM | p = 0.05 | 694.3 | 359.3 | 11.4 |
| SIR | β = 0.01, γ = 0.0 | 2,271.7 | 982.8 | 25.0 |
| SIR | β = 0.01, γ = 0.2 | 198.7 | 304.3 | 15.6 |
| SIR | β = 0.05, γ = 0.0 | 8,277.7 | 97.9 | 25.0 |
| SIR | β = 0.05, γ = 0.2 | 6,133.4 | 64.1 | 25.0 |

Preference-aware adoption cuts the cascade because low-affinity users often do
not adopt even when exposed. SIR with γ = 0 never recovers within the 25-round
horizon (SI-like); γ = 0.2 has the largest effect when transmission is weak.

**Moderate-popularity target comparison** (same graph, seeds and models; its own
target-specific seed pool):

| Model | Rare p=0.01 | Moderate p=0.01 | Rare p=0.05 | Moderate p=0.05 |
|---|---:|---:|---:|---:|
| Canonical ICM | 69.9 | 95.9 | 4,695.5 | 4,228.2 |
| Preference-aware ICM | 9.3 | 5.4 | 694.3 | 106.5 |

Canonical spread differs little by target popularity; the preference-aware
differences are driven by the affinity function rather than raw popularity.

**Controlled network-structure comparison.** The active graph is randomized by
degree-preserving directed double-edge swaps (603,420 attempted, 521,458
accepted, 86.4%, zero self-loops/duplicates, degree sequence preserved), then
canonical ICM runs with the same seed sets and trial seeds:

| Setting | Observed mean | Randomized mean | Relative change |
|---|---:|---:|---:|
| p = 0.01 | 69.9 | 12.3 | **−82.4%** |
| p = 0.05 | 4,695.5 | 4,754.4 | +1.25% |

The observed edge arrangement spreads far more in the sparse regime and is
statistically indistinguishable at p = 0.05. This is one seeded
degree-preserving null, not full mixing.

### Requirement 5 — influence maximization

The comparison uses the same graph, p = 0.05, budgets {1, 3, 5}, 5 Monte Carlo
evaluation trials, and 20 random trials. Because a full greedy sweep over 9,883
users is infeasible, all three methods draw from the **same restricted pool of
the 25 highest out-degree users** (ties by ascending user index). The result
therefore describes that pool, not the whole graph.

| Budget | Random | Degree | Greedy | Greedy−degree | Greedy−random |
|---:|---:|---:|---:|---:|---:|
| 1 | 4,719.4 | 4,718.8 | 4,718.8 | 0.0 | −0.6 |
| 3 | 4,719.4 | 4,718.8 | **4,724.4** | +5.6 | +5.0 |
| 5 | 4,719.6 | 4,718.8 | **4,724.4** | +5.6 | +4.8 |

The graph is near saturation (one seed already reaches ~4,719 users), so the
lift is about 0.12%. Greedy is not worse than the baselines and is best at
budgets 3 and 5 inside this pool; random is marginally best at budget 1. This
is a weak-but-honest positive result, not evidence of a large greedy advantage.
Under CRN the marginal gains are non-negative by construction, removing the
earlier artefact where a fresh random stream produced a negative gain. Greedy
used 115 evaluations (45.1 s); degree and greedy evaluation on the common trials
cost 1.0 s and 0.9 s.

### Requirement 6 — group recommendation

Three conflicting-preference groups are built **only from the training split**
via `build_rating_matrix(train_ratings)`. Satisfaction is the recommender's
predicted rating for each member; no held-out rating is used for construction or
reporting. Group top-N lists are ordered by the recommender's **raw (unclipped)**
aggregate score, and the reported group score is the **clipped** aggregate
satisfaction, so items that clip to the same rating no longer tie.

| Strategy | Mean satisfaction | Group score | Raw rank score | Simulated spread |
|---|---:|---:|---:|---:|
| Average | 4.915 | 4.915 | 5.180 | 713.0 |
| Least misery | 4.915 | 4.717 | 4.717 | 713.0 |
| Most pleasure | 4.915 | 5.000 | 5.636 | 713.0 |

Least misery returns the lowest group score by definition; most pleasure returns
5.0 because one member predicts the ceiling, but its raw ranking score is
distinct. All three strategies selected the same top item in every group in this
run, so their simulated spreads are identical — a property of these
training-only groups, not evidence that the strategies are equivalent.

### Requirement 7 — comprehensive evaluation

Definitions used in the saved results:

* **Relevance:** a held-out test item is relevant when its observed rating ≥ 4.
  Unobserved items are **not** negatives.
* **Candidates:** all catalogue items except the user's training items; no
  negative-item sampling.
* **F1@10:** mean of per-user F1 values.
* **Coverage:** unique recommended items / 139,738, over the evaluated sample.
* **Diversity:** mean pairwise `1 − cosine(item training-interaction feature
  vectors)` within each recommendation list.
* **No accuracy metric:** with only held-out positives and no observed
  negatives, a confusion-matrix "accuracy" would be misleading.

Observed evaluation (500 seeded users for ranking; all test ratings for
MAE/RMSE):

| Model | MAE | RMSE | nDCG@10 | Coverage | Unique | Diversity |
|---|---:|---:|---:|---:|---:|---:|
| Item-based CF | 1.0740 | 1.4519 | 0.00163 | **0.01715** | 2,396 | 0.8741 |
| Matrix factorisation | 0.8025 | 1.0510 | 0.00314 | 0.00197 | 275 | **0.9696** |
| Social (λ = 0.3) | **0.8008** | **1.0479** | **0.00518** | 0.01046 | 1,461 | 0.9646 |

Separately, a preference-aware ICM simulation uses each model's affinity
function on the same target item and seed sets: mean adopters are ICF 336.2,
MF 547.2, social 744.2, and the adopter satisfaction proxy is 4.118, 4.158 and
4.372 respectively. The highest observed nDCG model (social) also produces the
largest simulated spread in this run, but these are **separate** observed and
simulated outcomes; the simulation is not a platform-wide forecast.

### Requirement 8 — intervention, CELF and scaling

**Intervention.** A minimum predicted-affinity threshold blocks low-affinity
adoption inside the preference-aware ICM; seeds always start the cascade.
Baseline is threshold 0.0 with 10 simulations and the same CRN thresholds.

| Threshold | Mean adopters | Spread change | Adopter rating proxy | Share ≥ 4 |
|---:|---:|---:|---:|---:|
| 0.0 | 668.7 | — | 4.308 | 0.838 |
| 0.3 | 668.7 | 0.0% | 4.308 | 0.838 |
| 0.6 | 655.5 | −2.0% | 4.317 | 0.844 |
| 0.8 | 106.9 | −84.0% | 4.494 | 0.989 |
| 0.9 | 7.1 | −98.9% | 4.392 | 0.831 |

In the baseline exposure population (6,003 users; affinity range 0.411–1.000),
0.0% are below 0.3, 1.08% below 0.6, 42.65% below 0.8 and 86.01% below 0.9.
Thresholds at or below every observed affinity (0.0, 0.3) change nothing; 0.6
barely moves spread; 0.8 removes 84% of spread while raising the proxy; 0.9
almost fully suppresses the cascade. The proxy is `1 + 4 × predicted affinity`
and is **not** an observed post-intervention satisfaction rating.

**CELF versus standard greedy.** Same graph, pool, budget and trials. Because
CRN makes the finite-trial objective deterministic and submodular, CELF's lazy
bound is valid:

| Method | Evaluations | Runtime (s) | Seeds | Marginal gains |
|---|---:|---:|---|---|
| Standard greedy | 33 | 61.30 | [494, 321, 22] | 4,750.4, 3.8, 0.0 |
| CELF greedy | 24 | 56.69 | [494, 321, 22] | 4,750.4, 3.8, 0.0 |

CELF used 27.27% fewer evaluations (24 vs 33) and was **1.081× faster** while
selecting the identical seed set. A controlled deterministic check (p = 1.0, one
simulation, 120 nodes, 20 candidates, budget 2) gave 39 evaluations for both
methods and identical seeds `[494, 158]`.

**Measured scaling** (one Monte Carlo trial to isolate cost):

| Dimension | Setting | Runtime | Evaluations | Memory (tracemalloc peak) |
|---|---|---:|---:|---:|
| Network 25% | 2,471 nodes / 78,115 edges | 3.01 s | 29 | 3.07 MB |
| Network 50% | 4,942 nodes / 227,570 edges | 9.90 s | 29 | 6.92 MB |
| Item catalogue 25% | 34,934 items | 1.057 ms/user | — | 1.07 MB |
| Item catalogue 50% | 69,869 items | 2.017 ms/user | — | 2.13 MB |
| Item catalogue 100% | 139,738 items | 6.523 ms/user | — | 4.26 MB |
| Budget 1 | 2,471-node subgraph | 1.55 s | 15 | 3.03 MB |
| Budget 3 | 2,471-node subgraph | 4.03 s | 42 | 3.07 MB |

Network and budget costs grow with graph size and evaluation count; item
scoring is linear in catalogue size. The item measurement isolates MF
scoring/ranking, not training. Absolute times are machine-dependent.

---

## 4. Leakage controls and regression tests

The pipeline enforces the following separation:

1. Models are fitted on `train_ratings`.
2. `build_rating_matrix` is called with the training triples for group
   construction and social/feature work; R6 never receives
   `dataset.rating_matrix`.
3. Diffusion targets are selected from **training-only** count bands (1-rating
   and 6–20-rating bands).
4. Seed sets are generated once per target from training-derived eligible users
   and reused across every comparable ICM/SIR setting.
5. Social λ is selected on validation RMSE; test is read once after the choice
   is frozen (`test_used_for_selection: false`).
6. R7 observed metrics use test ratings only as labels; they never enter
   recommendations, item features, group construction, candidates or selection.
7. R6 satisfaction is a model prediction and diffusion outcomes are simulated.
8. R5 methods share the candidate pool and evaluation trials; R8 intervention
   rows share seed sets and CRN thresholds with the baseline.

Regression tests guard these claims:

* `tests/test_leakage.py` — training-only matrix construction; mutated full
  matrix does not change R6 groups; training-only target bands; reproducible
  seed sets; degree-preserving randomization; pure group aggregation;
  validation-only λ selection; tie-aware bookkeeping.
* `tests/test_parts5_8.py` — feature diversity; per-user F1; greedy does not
  stop on negative gain; explicit candidate pool; paired marginal gains;
  CELF/standard greedy agreement under CRN; raw-score group ranking with
  clipped reporting; hand-checkable metric and group cases.
* `tests/test_all.py` — ICM seed-superset monotonicity and p-nesting under one
  live-edge realization; preference-aware monotonicity; `ICMRandomness` reuse
  and seed-mismatch rejection; SIR reproducibility and recovery-threshold
  sharing across γ; data loading; split guarantees; MF cold-start fallbacks.
* `tests/test_vectorized_ranking.py` — vectorized ranking equivalence.

Latest full run: **103 passed** in about 1.7 s (`python -m pytest -q`).

---

## 5. Saved settings and exact seeds

`configs/fast_demo.yaml` is the submitted configuration; its resolved copy is
`results/fast_demo/experiment_config.json`. Key seeds:

* split, MF, ranking sample and diversity sample seed **42**; group seed **42**
* diffusion seed **42**, 10 simulations, 5 seed users; rare-target seed sets
  generated once and reused for every ICM/SIR setting
* R4 network randomization seed **4242**, `2 × n_edges` swap attempts
* R5 selection seeds `[49, 1049, 2049, 3049, 4049]`, evaluation seeds
  `[500049 … 504049]`, 20 random trials
* R6 per-group simulation seeds use offsets `100000 + 1000 × group_index`
* R7 interaction seeds `[700049 … 704049]`
* R8 intervention seeds `[900049 … 904049]`; CELF selection seeds
  `[800049 … 804049]`; scaling selection seeds offset 820000

To reproduce from a clean checkout:

```powershell
python -m pytest -q
python -m sntp.cli run --config configs/fast_demo.yaml
```

The experiment writes each requirement JSON after that stage completes, so a
failed later stage does not erase earlier artifacts. Redirect the log if
desired:

```powershell
python -u -m sntp.cli run --config configs/fast_demo.yaml *> run.log
```

`configs/default.yaml` would write `results/full/` but was not run.

---

## 6. Results summary and runtime

**Measured fast-demo runtime: 473.6 s (7.9 min) of stage time; 474.9 s wall
clock**, including data loading, model fitting, all eight requirement stages and
result serialization. Major stages:

| Stage | Seconds |
|---|---:|
| R1 load dataset | 6.43 |
| R1 train/val/test split | 0.49 |
| R2 item-based CF fit | 48.80 |
| R2 MF training | 120.15 |
| R3 social evaluations (all λ + comparisons) | 28.87 |
| R4 ICM + SIR + target/network comparisons | 31.97 |
| R5 influence maximization | 69.72 |
| R6 group recommendation | 3.54 |
| R7 comprehensive evaluation | 13.58 |
| R8 intervention + CELF + scaling | 139.96 |
| **Stage total** | **473.58** |

MF training is the largest single stage, and R8 is the largest non-model block
because it runs repeated Monte Carlo greedy evaluations from scratch. The stage
sum excludes CLI startup and result-directory setup. The offline validation
selection sweep in `model_selection.json` is also not included.

---

## 7. Report generation

The searchable PDF report is at `report/SNTP_Assignment_02_Report.pdf`. It is a
compact two-column A4 document of **4 pages** (the requirement is at most 8),
generated directly from the saved JSON, so its tables stay tied to the measured
artifacts. It contains numbered sections, seven captioned figures, all result
tables, references, and explicit disclosure of the MF trade-off, the restricted
influence pool, and the unexecuted full-scale config.

```powershell
python -m pip install -r requirements-report.txt
python scripts/generate_report.py --results results/fast_demo `
  --output report/SNTP_Assignment_02_Report.pdf `
  --figures-dir results/fast_demo/plots `
  --roll-number 23F3008 --student "Ahmad Fareed"
```

`requirements-report.txt` adds `reportlab` and `matplotlib`; the core
experiments need only `requirements.txt`, and tests need
`requirements-dev.txt`. The same command writes seven standalone charts to
`results/fast_demo/plots/`:

| File | Content |
|---|---|
| `fig01_test_error.png` | Test MAE/RMSE per recommender |
| `fig02_rare_target_diffusion.png` | Rare-target ICM spread |
| `fig03_target_popularity.png` | Rare vs moderate target |
| `fig04_network_structure.png` | Observed vs randomized structure |
| `fig05_influence_spread.png` | Influence spread vs budget |
| `fig06_intervention.png` | Spread and proxy vs threshold |
| `fig07_cost_scaling.png` | Network and item-catalogue scaling |

---

## 8. Assumptions

* Trust is binary and static; the data has no timestamps, so trust and ratings
  are one observed snapshot.
* A rating is an explicit preference signal; unobserved items are missing, not
  negative.
* A user's training history is the only recommendation context; validation and
  test ratings are held out.
* Diffusion parameters (p, β, γ, base adoption probability, affinity mapping)
  are simulation assumptions, not fitted to adoption data, because no adoption
  traces exist.
* "Spread" counts adopters including the initial seeds. Marginal gains are
  paired differences under common random numbers.
* Group satisfaction is the recommender's predicted rating; groups are built
  from training ratings only and may overlap.
* Catalogue coverage uses all 139,738 items as the denominator, but its scope
  is the evaluated user sample (500 users in the fast demo).
* The ego subgraph is a bounded, deterministic 2-hop neighbourhood of user 0,
  not a random sample of the full trust graph.

---

## 9. Deliverables and file map

```
.
├── README.md
├── requirements.txt
├── requirements-dev.txt        # test dependency (pytest)
├── requirements-report.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── fast_demo.yaml          # submitted, measured configuration
│   └── default.yaml            # full-scale config, not run
├── src/sntp/                   # package
├── tests/                      # 103 tests
├── scripts/generate_report.py  # regenerates the PDF and PNGs from JSON
├── report/SNTP_Assignment_02_Report.pdf
├── results/fast_demo/          # submitted measured results
│   ├── data_summary.json
│   ├── split_info.json
│   ├── experiment_config.json
│   ├── stage_timings.json
│   ├── requirement_{2..8}_results.json
│   ├── model_selection.json
│   └── plots/                  # 7 standalone PNG figures
└── Assignment/ratings_data/epinions/
    ├── ratings_data.txt
    └── trust_data.txt
```

Submission archive (one inner folder, forward-slash paths, no `.git`, caches,
temporary scripts or old ZIPs):

```
23F3008_SNTP_Assignment.zip
└── 23F3008_SNTP_Assignment/
```

---

## 10. Report structure

1. Abstract
2. Data, split and protocol (R1)
3. Recommendation models (R1–R3), including the MF ranking trade-off
4. Information diffusion (R4), including CRN and the network-structure null
5. Influence maximization (R5)
6. Group recommendation (R6)
7. Comprehensive evaluation (R7)
8. Intervention, CELF and cost scaling (R8)
9. Reproducibility, limitations and conclusion
10. References

---

## 11. What was fixed relative to the pre-existing implementation

The original project ran end-to-end but had correctness and evidence problems
that were fixed before the submitted results:

* **Common random numbers.** Diffusion previously restarted a sequential RNG at
  the same integer seed per run, so "common seeds" did not give common live
  edges. Per-edge (and per-edge/per-round, per-user/per-round) thresholds are
  now generated once per trial and reused across seed sets, probabilities and
  variants. Tests assert seed-superset monotonicity, probability nesting, SIR
  reproducibility and recovery-threshold sharing.
* **CELF validity.** CELF's lazy bound was applied to a noisy objective and
  could select different seeds / appear slower. Under CRN the finite-trial
  objective is deterministic, monotone and submodular; CELF now reproduces
  standard greedy exactly with fewer evaluations (24 vs 33, 1.081×).
* **Group ranking.** Group top-N was ordered on clipped scores, creating
  artificial 5.0 ties. Ordering now uses raw aggregate scores while the reported
  group score is the clipped satisfaction; a test checks the distinction.
* **Scaling ranking.** R8 item-catalogue scoring clipped scores before `top_n`,
  collapsing distinct items into ties; it now ranks on the bias-adjusted raw
  score family.
* **R5 fairness and no negative gains.** Candidate-pool restriction is explicit
  (top 25 out-degree, same pool for all methods), marginal gains are paired,
  the greedy sentinel is `-inf`, and CRN makes gains non-negative.
* **R6 leakage.** Group construction previously used `dataset.rating_matrix`
  (all observed ratings). It now uses training triples only; a regression test
  mutates the full matrix and proves groups do not change.
* **R3 selection.** Social λ now uses a validation-only rule.
* **MF ranking correction.** The item-bias weight (0.05) was added and selected
  on validation; prediction error and returned ratings still use the full model.
  The coverage/nDCG trade-off is disclosed with verifiable values.
* **R7.** F1 is the mean of per-user F1; diversity is feature-based with
  sampling metadata; observed and simulated outcomes are separated.
* **R8.** Intervention findings report measured per-threshold effects
  (including the inactive 0.3 and the non-monotone 0.6) instead of an asserted
  monotone trade-off; CELF conclusions are generated from measurements.
* **MF cold starts.** Untrained rows are zeroed and documented fallbacks
  (μ+b_i, μ+b_u, μ) are used.
* **Report generator.** Rewritten as a compact two-column ≤8-page report that
  reads every table from JSON and exports seven PNG charts.

All reported results were produced after these fixes. Pre-fix artifacts were
used only to confirm the defects and are not part of the submitted results.

---

## 12. Limitations and deviations

1. **Full-scale configuration not executed.** `configs/default.yaml` uses the
   full graph and much larger budgets; its runtime is many hours. All reported
   numbers are from the bounded fast demo.
2. **Fast-demo scope.** Diffusion/influence/group/intervention use the 2-hop ego
   subgraph (9,883 users, 301,710 edges), not the full 49,289 users / 487,002
   trust edges. R5 further restricts candidates to the 25 highest out-degree
   users. The full config was not run.
3. **Monte Carlo budget.** R4 uses 10 simulations, R5 uses 5 evaluation trials
   and 20 random trials, R6 uses 5 simulations per item, and R8 uses 10
   intervention simulations and 5 CELF trials. Spread distributions are
   heavy-tailed; means and standard deviations are reported and small
   differences are not claimed as significant.
4. **R5 shows only a small greedy advantage.** Inside the restricted
   high-degree pool, greedy gains +5.6 adopters over degree at budgets 3 and 5
   (+0.12%); random is marginally best at budget 1. The graph is near
   saturation, compressing differences.
5. **R6 strategies tie in this run.** All three selected the same top item for
   every group, so their simulated spreads are identical. Candidate items are
   only the union of strategy top items.
6. **R7 is sample-based.** Ranking/diversity use 500 seeded users, not all
   25,045 eligible users. Unobserved items are not negatives, so
   precision/recall are conservative and no accuracy metric is reported.
7. **R8 threshold is a simulator intervention.** No online experiment was run.
   The rating proxy is predicted, not observed. CELF's speedup is a single
   measurement; the controlled deterministic check and the CRN objective are the
   correctness evidence.
8. **Memory.** `tracemalloc` measures Python allocations and RSS deltas are near
   zero because large arrays are already resident; NumPy native buffers are
   under-represented.
9. **Runtime variability.** Timings depend on CPU load and Python/NumPy/BLAS
   versions; the saved `stage_timings.json` is one run.
10. **Network null.** The randomization is one seeded degree-preserving
    rearrangement, not full mixing; observed and randomized graphs cannot share
    edge-level thresholds.
11. **Dataset.** The trust direction is source→target, so influence propagates
    target→source as documented.
