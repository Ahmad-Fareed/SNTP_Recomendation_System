# SNTP Assignment 2 — Recommendation and Information Diffusion on Epinions

**Student:** Ahmad Fareed  
**Roll number:** 23F3008  
**Course:** Social Network Theory and Programming (SNTP), NU-FAST Chiniot-Faisalabad  
**Deliverable:** runnable source, tests, saved settings/seeds, measured results, searchable PDF report

This repository implements all eight assignment requirements on the Epinions
rating and trust network. The submitted results were produced by the bounded
**fast-demo** configuration, which uses the complete rating/trust files and a
2-hop ego subgraph for the diffusion experiments. The larger
`configs/default.yaml` was not run; see [Limitations](#10-limitations-and-deviations).

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

# 94 tests (the suite runs in a few seconds)
python -m pytest -q

# Inspect the dataset without fitting models
python -m sntp.cli inspect --data-dir Assignment

# Bounded end-to-end demonstration; writes results/fast_demo/
python -m sntp.cli run --config configs/fast_demo.yaml
```

The demo above took **416.84 s (6.9 min)** on the development machine. Earlier
runs of the same configuration varied substantially with machine load, so treat
wall-clock times as machine-dependent. Most of the cost is item-based CF
fitting, MF training, and the R8 CELF/scaling measurements, not the diffusion
simulations.

To regenerate the PDF report after installing the optional report packages
(see [Section 7](#7-report-generation)):

```powershell
python scripts/generate_report.py --results results/fast_demo `
  --output report/SNTP_Assignment_02_Report.pdf `
  --figures-dir results/fast_demo/plots `
  --roll-number 23F3008 --student "Ahmad Fareed"
```

`configs/default.yaml` is the full-scale configuration. It disables the ego
subgraph, uses 20 Monte Carlo trials, larger candidate pools/budgets, and the
full ranking workload. It is expected to take many hours on this machine and
was **not executed** for the submitted results.

---

## 2. Dataset and splits

The Epinions files are included under
`Assignment/ratings_data/epinions/`:

| File | Rows | Meaning |
|---|---:|---|
| `ratings_data.txt` | 664,824 | `user item rating` (ratings 1–5) |
| `trust_data.txt` | 487,002 | `source target 1`; “source trusts target” |

Parsing found no malformed, out-of-range, or duplicate rating records. The
loader also accepts `Assignment/trust_data/epinions/trust_data.txt`; only one
trust file is required.

**Trust direction:** a row `A B 1` means A trusts B. Influence propagates in
the opposite direction, B → A. The propagation graph therefore stores edge
`B -> A`, so B can influence A.

**Split:** deterministic 70/15/15 train/validation/test split with seed 42.
Users with at least three ratings contribute at least one record to each split;
users with fewer records are kept in training but excluded from evaluation.
Test ratings are touched only by the final observed evaluation.

| Split | Ratings |
|---|---:|
| Train | 463,663 |
| Validation | 100,603 |
| Test | 100,558 |
| Total | 664,824 |

Dataset summary: 49,289 users, 139,738 items, 487,002 trust edges, sparsity
0.999903. Mean ratings per user 16.55 (median 6.0); mean ratings per item 4.76
(median 1.0).

**No adoption timestamps exist** in the data. Every cascade, round count,
spread, and simulated satisfaction value is a simulation, not an observed
diffusion outcome.

---

## 3. Requirement-to-code map

| Requirement | Implementation | Main saved evidence |
|---|---|---|
| 1. Data preparation | `src/sntp/data.py`, `src/sntp/splits.py`, `cli.py inspect` | `results/fast_demo/data_summary.json`, `split_info.json` |
| 2. Rating prediction + top-K | `src/sntp/recommenders/item_based.py`, `matrix_factorization.py`, `_ranking.py` | `requirement_2_results.json`, `model_selection.json` |
| 3. Trust-aware recommendation | `src/sntp/recommenders/social.py`, `experiments.run_requirement_3` | `requirement_3_results.json` |
| 4. ICM, SIR, preference-aware diffusion | `src/sntp/diffusion/icm.py`, `sir.py`, `selection.py` | `requirement_4_results.json` |
| 5. Influence maximization | `src/sntp/influence.py`, `experiments.run_requirement_5` | `requirement_5_results.json` |
| 6. Group recommendation | `src/sntp/groups.py`, `experiments.run_requirement_6` | `requirement_6_results.json` |
| 7. Comprehensive evaluation | `src/sntp/metrics.py`, `experiments.run_requirement_7` | `requirement_7_results.json` |
| 8. Intervention, CELF, scaling | `experiments.run_requirement_8`, `influence.py` | `requirement_8_results.json` |

Shared experiment-design helpers live in `src/sntp/selection.py`.

### Requirement 1 — data preparation and exploration

`EpinionsDataset` parses both files, validates ratings, deduplicates exact
rating duplicates, builds ID mappings, a sparse rating matrix, and the
propagation graph. `sntp.cli inspect` prints the summary used in
`data_summary.json`. `splits.py` implements the per-user constrained split.

### Requirement 2 — baseline recommenders

Two recommenders are trained on the training split:

* **Item-based CF:** cosine item similarity, `max_neighbors=20`,
  `min_corated=2`.
* **Matrix factorization:** 20 latent factors, learning rate 0.01,
  regularization 0.005, 20 epochs, seed 42. Top-N ordering uses the full
  prediction with the global item-bias term weighted by 0.05; MAE/RMSE and
  returned ratings always use the full model. The weight was selected on
  validation only.

Top-K recommendation excludes every item in the user's **training** history.
Item-based CF orders items by the **unclipped** prediction score. MF orders by
the configured ranking score (full prediction with the item-bias term weighted
by 0.05). In both cases only the predicted rating returned with an item is
clipped to [1, 5]. This prevents ties at 5.0 from replacing confidence ordering
with item-ID ordering. Item-based CF treats
an item with no stored neighbours as unsupported by the item-item model and
uses a documented training-only fallback: the user’s training mean, then the
global training mean. MF tracks which users/items have training observations;
untrained rows have zero bias/factors, giving μ+b_i, μ+b_u, or μ as documented
fallbacks instead of random initial factors. Ranking metrics use a fixed seeded
sample of 500 eligible users; MAE/RMSE use all 100,558 test ratings. The test
ranking is:

| Model | Test MAE | Test RMSE | P@10 | R@10 | F1@10 | HR@10 | nDCG@10 | Coverage | Train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Item-based CF | 1.0740 | 1.4519 | 0.0008 | 0.00281 | 0.00105 | 0.0080 | 0.00163 | **0.01715** | 62.05 |
| Matrix factorization | **0.8025** | **1.0510** | 0.0014 | 0.00553 | 0.00202 | 0.0140 | 0.00314 | 0.00197 | 168.19 |

MF is much better than item-based CF on rating error here. The ranking numbers
are low because the catalogue is large, test items are sparse, and a single
held-out item per user is often buried below rank 10. Behaviour/popularity
breakdowns are saved in `requirement_2_results.json`; MF is consistently better
across short/medium/long user histories and rare/moderate/popular item groups.
The largest item-popularity gap is on moderate-popularity items (ICF MAE 1.2514
vs MF 0.7751).

The old full-score MF ranking was dominated by global item bias: on validation,
99.91% of within-user score variance was item bias and 97.86% of top-10
positions matched a bias-only ranking. It returned 15 unique items across 500
users. The corrected ranking score is
`mu + b_u + 0.05*b_i + p_u.q_i`; prediction error and returned ratings still use
the full model. On test, MF now returns 275 unique items (coverage 0.00197,
about 17x the old reach). Test nDCG@10 is 0.00314 versus 0.00343 before, a small
decrease; the selection used validation only, where nDCG@10 rose from 0.00266
to 0.00711. This trade-off is reported rather than hidden. The social model
below also uses the corrected baseline ordering and reaches 1,461 unique items.

### Requirement 3 — trust-aware recommendation

A social recommender blends the MF prediction with the mean rating of trusted
neighbours:

`prediction = (1 - λ) * baseline + λ * trust_neighbour_prediction`

The social weight is selected **only on validation RMSE** from
`λ ∈ {0.0, 0.3, 0.5}`. Test is reported afterward and is never used for
selection.

| λ | Validation RMSE | Test MAE | Test RMSE | P@10 | HR@10 | nDCG@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 (baseline) | 1.04929 | 0.80255 | 1.05099 | 0.0014 | 0.014 | 0.00314 |
| **0.3 (selected)** | **1.04605** | **0.80080** | **1.04787** | **0.0024** | **0.024** | **0.00518** |
| 0.5 | 1.05229 | 0.80369 | 1.05417 | 0.0018 | 0.018 | 0.00423 |

The selected λ = 0.3 improves test MAE by 0.00175 and RMSE by 0.00312, and also
improves ranking. Sparse users (21,911 users with 1-5 ratings) improve from MAE
0.85055 to 0.84909. In the controlled trust-vs-similarity comparison on 50
sampled users / 165 test ratings, trusted neighbours gave MAE 0.83534,
unconnected similar users 0.82943, and the baseline 0.83734 - a small sample,
so this comparison is descriptive, not a significance test.

### Requirement 4 — diffusion models

All diffusion runs use the same pre-generated seed sets and the same controlled
Monte Carlo seed schedule across comparable settings. Part 4 retains the
original **rare-item** target: original item **3** (index 2), with 1 training
rating, selected from the explicit inclusive 1-rating band (66,605 eligible
items). It adds a **moderate-popularity** target: original item **69** (index
68), with 9 training ratings, selected from the explicit 6–20 training-rating
band (10,807 eligible items). Both selections use training ratings only and
choose the count closest to the band median, with ascending item-ID ties. The
moderate experiment uses the same graph, models, propagation probabilities,
simulation seeds, and seed size; its seed users are drawn from its own
target-specific eligible pool. The rare experiment is retained, not replaced.

Fast-demo design: 5 seeds, 3 simulations, 2-hop ego subgraph (9,883 users,
301,710 edges), max 25 rounds. The same three rare-target seed sets are reused
for every rare ICM/SIR setting:

```
[734, 3024, 4170, 7316, 13374]
[883, 1357, 2386, 2860, 13374]
[1413, 6945, 7018, 7985, 10792]
```

| Model | Setting | Mean adopters/infected | Std | Mean rounds |
|---|---|---:|---:|---:|
| Canonical ICM | p = 0.01 | 125.33 | 167.36 | 9.00 |
| Canonical ICM | p = 0.05 | 4,684.33 | 21.70 | 11.00 |
| Preference-aware ICM | p = 0.01 | 6.00 | 0.82 | 1.00 |
| Preference-aware ICM | p = 0.05 | 501.33 | 351.96 | 7.67 |
| SIR | β = 0.01, γ = 0.0 | 2,024.00 | 915.16 | 25.00 |
| SIR | β = 0.01, γ = 0.2 | 530.67 | 692.39 | 17.67 |
| SIR | β = 0.05, γ = 0.0 | 8,180.33 | 116.48 | 25.00 |
| SIR | β = 0.05, γ = 0.2 | 6,039.67 | 25.17 | 25.00 |

The p = 0.01 canonical ICM result is extremely heavy-tailed: only 3 simulations
produced one cascade of 362 adopters and two near-seed-only cascades. The large
standard deviations and CVs reflect that, not an implementation error. The SIR
runs with γ = 0 use all 25 rounds because infection never recovers; their
“rounds” are capped by the configured maximum. Preference-aware adoption gives
much smaller cascades at the same seed sets because predicted affinity gates
exposure/adoption. Mean rounds, min/max, CV, runtime, and per-simulation rows
are in `requirement_4_results.json`.

**Moderate-popularity target experiment.** The same canonical and
preference-aware ICM models are run on item 69 at the same propagation
probabilities and simulation seeds. The seed pool changes because it is
target-specific; the graph and Monte Carlo schedule do not. The measured means
are:

| Model | Rare target p=0.01 | Moderate target p=0.01 | Rare target p=0.05 | Moderate target p=0.05 |
|---|---:|---:|---:|---:|
| Canonical ICM | 125.33 | 5.00 | 4,684.33 | 3,111.33 |
| Preference-aware ICM | 6.00 | 5.00 | 501.33 | 10.67 |

The moderate target changes the canonical p=0.05 mean by −33.58% and the
preference-aware p=0.05 mean by −97.87%. Results are stored in
`target_popularity_comparison` and are descriptive, not an identified
popularity effect.

**Controlled network-structure null.** The active graph is randomized by
degree-preserving directed double-edge swaps: node set, edge count, and every
in/out degree are preserved. Of 603,420 attempted swaps, 521,458 were accepted
(86.4%), with zero self-loops or duplicate edges and an identical degree
sequence. Canonical ICM then uses the same rare-target seed sets, simulation
seeds, probabilities, and round cap on the randomized graph:

| Setting | Observed mean | Randomized mean | Difference | Relative |
|---|---:|---:|---:|---:|
| p=0.01 | 125.33 | 7.33 | −118.00 | −94.15% |
| p=0.05 | 4,684.33 | 4,780.00 | +95.67 | +2.04% |

This isolates edge arrangement within one seeded null model; it is not a fully
mixed null distribution. Exact swap counts and per-setting outcomes are saved
in `network_structure_comparison`.

### Requirement 5 — influence maximization

The comparison uses the same 9,883-node / 301,710-edge propagation graph, the
same propagation probability (0.05), the same budgets (1, 3, 5), the same
Monte Carlo evaluation seeds, and a **shared candidate pool**. The pool is
restricted to the **top 15 out-degree users** (ties by ascending user index)
and is identical for random, degree, and greedy. This restriction makes the
comparison computationally bounded, but it also gives degree centrality a
structural advantage; the results describe that restricted pool, not all graph
users.

Random selection is repeated for 3 seeded trials. Degree and greedy are
deterministic selectors. Every method is evaluated on the same evaluation
seeds; random’s reported mean is the mean of its trial means. Marginal gains
are paired differences between the same Monte Carlo trials at adjacent
evaluated budgets, not differences of unrelated runs.

| Budget | Random mean | Degree mean | Greedy mean | Best measured |
|---:|---:|---:|---:|---|
| 1 | 4,711.0 | 4,712.0 | **4,726.5** | greedy |
| 3 | 4,690.0 | **4,747.5** | 4,687.0 | degree |
| 5 | 4,692.5 | **4,712.5** | 4,700.5 | degree |

At budget 5, greedy − degree = −12.0 adopters (paired 95% CI [−19.84, −4.16]),
so there is no evidence in this run that greedy beats degree. Greedy exceeds
random by only 8.0 adopters (0.2%) at budget 5. The restricted top-degree pool,
dense graph, and near-saturation spreads (about 47% of the 9,883 active users)
make differences small. Degree and greedy adopter sets overlap substantially
(mean Jaccard 0.551). Greedy selected the full budget in every row; the
standard-greedy sentinel is `-inf`, so a negative Monte Carlo marginal gain does
not stop selection. Greedy selection took 14.25 s for 65 evaluations; degree
selection was effectively instant; random selection for all trials took 1.58 s.

### Requirement 6 — group recommendation

Groups are built **only from the training split** via
`build_rating_matrix(train_ratings)`; the full `dataset.rating_matrix` is never
passed to group construction. Construction forms conflicting groups from users
with low, middle, and high mean training ratings. The fast demo has 3 groups ×
4 members (9,308 eligible group users, 463,663 training nonzeros); groups may
overlap across groups. Group construction is deterministic with seed 42.

Satisfaction is the recommender’s predicted rating for each member; no held-out
rating is used to construct groups or to report satisfaction. Diffusion is a
preference-aware ICM with the group members as seeds, 2 simulations per item,
and controlled seeds.

| Strategy | Mean satisfaction | Mean min satisfaction | Mean disagreement range | Mean simulated spread | Spread std |
|---|---:|---:|---:|---:|---:|
| Average | 4.915 | 4.717 | 0.283 | 226.33 | 222.33 |
| Least Misery | 4.915 | 4.717 | 0.283 | 226.33 | 222.33 |
| Most Pleasure | 4.658 | 4.238 | 0.762 | 210.33 | 206.33 |

Average and least misery tie on mean and minimum predicted satisfaction; average also has the highest mean simulated spread, while most pleasure is lower on satisfaction and spread. The between-strategy mean difference is only 0.257, while the spread standard deviation is up to 222.3. One rare cascade dominates the average, so the small spread ordering is not a stable strategy effect. Satisfaction is a model prediction, not an observed group rating.

Across 3 groups, the item with the highest mean predicted satisfaction was also the item with the highest simulated spread in 3 groups (100.0%). This agreement counts a non-empty overlap between exact tied best sets; it does not require the two best sets to be equal. Per-group members, top-5 lists, member ratings, spreads, and candidate comparisons are saved in `requirement_6_results.json`.

### Requirement 7 — comprehensive evaluation

Definitions used in the saved results:

* **Relevance:** a held-out test item is relevant when its observed rating ≥ 4.
  Unobserved items are **not** treated as negatives.
* **Candidates:** all catalogue items except the user’s training items; no
  negative-item sampling.
* **F1@10:** mean of per-user F1 values, not the harmonic mean of aggregate
  precision and recall.
* **Coverage:** unique recommended items / 139,738 catalogue items over the
  evaluated user sample. This is a sample-scope coverage, not an estimate over
  all 49,289 users.
* **Diversity:** mean pairwise `1 - cosine(item training-interaction feature
  vectors)` within each recommendation list, using train-derived item features
  and a bounded deterministic list sample. This is feature diversity, not
  category diversity.
* **No accuracy classification metric:** with only held-out positives and no
  observed negatives, a confusion-matrix “accuracy” would be misleading.

Fast-demo observed evaluation (500 seeded users for ranking; all test ratings
for MAE/RMSE):

| Model | MAE | RMSE | P@10 | R@10 | F1@10 | HR@10 | nDCG@10 | Coverage | Diversity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Item-based CF | 1.0740 | 1.4519 | 0.0008 | 0.00281 | 0.00105 | 0.0080 | 0.00163 | **0.01715** | 0.8741 |
| Matrix factorization | 0.8025 | 1.0510 | 0.0014 | 0.00553 | 0.00202 | 0.0140 | 0.00314 | 0.00197 | **0.9696** |
| Social (λ=0.3) | **0.8008** | **1.0479** | **0.0024** | **0.00873** | **0.00343** | **0.0240** | **0.00518** | 0.01046 | 0.9646 |

The social model is best on error and all ranking metrics in this run;
item-based CF has the broadest catalogue coverage but much worse accuracy. The
separate simulated preference-aware interaction uses one training-selected
target item and common seeds; its mean adopters are ICF 753.0, MF 914.5, social
940.0, and the highest simulated spread agrees with the highest observed nDCG
model in this run. These are **separate** observed and simulated outcomes; the
simulation is not a forecast of platform adoption. The observed/simulated
separation is explicit in `requirement_7_results.json`.

Diversity metadata now reports 500 evaluated lists, `diversity_is_sampled=true`,
seed 42, and scope `ranking_user_sample`. The flag is true because the lists
come from the seeded ranking-user sample even though all 500 sampled lists were
used; the old false/null values described only the internal list cap.

### Requirement 8 — intervention, CELF, and scaling

**Intervention.** The documented intervention applies a minimum predicted
affinity threshold inside the preference-aware ICM: exposed users below the
threshold do not adopt or propagate. Seeds are platform-promoted users and
always start the cascade. Baseline is threshold 0.0 with 5 simulations and the
same seed sets/seeds as the intervention rows.

| Threshold | Mean adopters | Mean exposed | Conversion | Adopter rating proxy | Spread change | Proxy change |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 (baseline) | 705.8 | 1,745.6 | 0.4043 | 4.3078 | - | - |
| 0.3 | 705.8 | 1,745.6 | 0.4043 | 4.3078 | +0.00% | +0.0000 |
| 0.6 | 725.8 | 1,791.4 | 0.4052 | 4.3204 | +2.83% | +0.0126 |
| 0.8 | 146.8 | 506.6 | 0.2898 | 4.5052 | -79.20% | +0.1974 |
| 0.9 | 6.6 | 22.0 | 0.3000 | 4.3272 | -99.06% | +0.0194 |

The threshold effect is a cliff, not a smooth trade-off. In the baseline
exposure population (4,672 users), affinity ranged
0.4110-1.0000; 0.00% were
below 0.3, 1.05% below 0.6, 41.78% below 0.8,
and 85.10% below 0.9. Threshold 0.3 therefore cannot change the
baseline cascade. Threshold 0.6 changes only a few paths and slightly increases
spread and the proxy. Threshold 0.8 removes 79.20% of spread while raising the
simulator-side adopter rating proxy by 0.1974; threshold 0.9 removes 99.06% of
spread with a smaller but still positive proxy change (+0.0194). Shared seeds
reduce but do not remove path divergence when filtering changes which users are
exposed. The proxy is `1 + 4 * predicted affinity`; it is **not** an observed
post-intervention satisfaction rating.

**CELF vs standard greedy.** Same graph, candidate pool, budget, simulations,
and selection seeds:

| Method | Evaluations | Runtime (s) | Seeds | Marginal gains |
|---|---:|---:|---|---|
| Standard greedy | 33 | 34.62 | 22, 321, 570 | 4760.5, 0.5, 9.5 |
| CELF | 25 | 27.53 | 22, 321, 206 | 4760.5, 0.5, -13.5 |

CELF used 24.24% fewer evaluations and was
**1.257x faster in this measured setting**, but it
selected a different third seed under noisy Monte Carlo estimates. CELF's lazy
bound is exact for a deterministic monotone submodular objective, not for a
finite-sample noisy estimate. A controlled deterministic check (`p=1.0`, one
simulation, 120 nodes, 20 candidates,
budget 2) gave 39 evaluations for
both methods and identical seeds `[494, 158]`, confirming the
implementation rather than asserting an unconditional speedup. Memory:
tracemalloc peak 8.452 MB (standard) vs
8.454 MB (CELF); RSS deltas were
2.199 MB and -0.004 MB.
NumPy native allocations may be under-represented.

**Measured scaling.**

| Dimension | Setting | Runtime | Evaluations | Memory (tracemalloc peak) |
|---|---|---:|---:|---:|
| Network | 2,471 nodes / 78,115 edges | 3.72 s | 29 | 2.35 MB |
| Network | 4,942 nodes / 227,570 edges | 11.90 s | 29 | 5.36 MB |
| Budget | budget 1 | 2.63 s | 15 | 2.27 MB |
| Budget | budget 3 | 5.67 s | 42 | 2.35 MB |
| Item catalogue 25% | 34,934 items | 1.08 ms/user | - | 1.07 MB |
| Item catalogue 50% | 69,869 items | 1.83 ms/user | - | 2.13 MB |
| Item catalogue 100% | 139,738 items | 6.06 ms/user | - | 4.26 MB |

Network and budget costs grow with graph size and number of evaluations; the
item-catalogue measurement isolates MF scoring/ranking, not training. The
1.08 -> 1.83 -> 6.06
ms/user pattern is a measured cache/allocation effect, not a claimed linear law.
Absolute times are machine-dependent; compare relative slopes within this run.
RSS deltas were near zero because large arrays were already resident;
tracemalloc undercounts NumPy native buffers.

---

## 4. Leakage controls and regression tests

The experiment pipeline enforces the following separation:

1. Models are fitted on `train_ratings`.
2. `build_rating_matrix` is called with the training triples for all
   group-construction and social/feature work. The full
   `dataset.rating_matrix` is not passed into R6.
3. Diffusion targets are selected from **training-only** count bands: the
   retained rare target uses the 1-rating band, and the added moderate target
   uses the inclusive 6–20 band.
4. Seed sets are generated once from each target-specific training-derived
   eligible-user pool and reused across every comparable ICM/SIR setting.
5. Social λ is selected on validation RMSE. Test is reported only after the
   choice is frozen (`test_used_for_selection: false` in the JSON).
6. R7 observed metrics use test ratings only as labels. Test ratings never
   enter recommendations, item features, group construction, candidate
   generation, or parameter selection.
7. R6 satisfaction is a model prediction, and the diffusion outcome is
   simulated; neither is an observed post-recommendation rating.
8. R5 methods share the candidate pool and evaluation seeds; R8 intervention
   rows share seed sets/seeds with the baseline.

Regression tests specifically guard the leakage and fairness claims:

* `tests/test_leakage.py`
  * training-only matrix construction;
  * full `dataset.rating_matrix` mutation does not change R6 groups;
  * rare (1) and moderate (6–20) target bands, deterministic training-only
    target selection;
  * seed-set generation is reproducible;
  * degree-preserving directed randomization preserves nodes, edges, and
    in/out degrees and is reproducible;
  * group score aggregation is a pure function of member scores;
  * R3 selects λ on validation even when test ratings are perturbed;
  * R6 tie-aware best-set bookkeeping.
* `tests/test_parts5_8.py`
  * feature-based diversity, per-user F1;
  * greedy does not stop on negative marginal gain;
  * explicit candidate pool is respected;
  * marginal gains equal paired spread increments;
  * manual/hand-checkable metric and group cases.
* `tests/test_all.py`, `tests/test_vectorized_ranking.py`
  * data loading, split guarantees, recommender correctness;
  * unclipped ordering scores with post-selection rating clipping;
  * item-based unsupported-item fallback;
  * MF cold-start fallbacks (μ+b_i, μ+b_u, μ);
  * vectorized ranking equivalence.

Latest full test run: **94 passed** in about 1.7 s (`python -m pytest -q`); the suite is a few seconds, not a long-running experiment.

---

## 5. Saved settings and exact reproducibility

`configs/fast_demo.yaml` is the submitted configuration. The resolved copy is
also saved as `results/fast_demo/experiment_config.json`. Key seeds:

* split seed 42; ranking sample seed 42; diversity sample seed 42
* MF seed 42; group seed 42
* diffusion seed 42, 3 seed sets, 3 simulations; seed-set generator uses
  `seed=42`
* R4 network randomization seed 4242, `2 × n_edges` swap attempts
* R5 selection seeds `[49, 1049]`, evaluation seeds `[500049, 501049]`
* R6 per-group simulation seeds `[100049, 101049]`, `[101049, 102049]`,
  `[102049, 103049]`
* R8 intervention seeds `[900049, 901049, 902049, 903049, 904049]`
* R8 CELF selection seeds `[800049, 801049]`
* scaling seeds offset 820,000

To reproduce from a clean checkout:

```powershell
python -m pytest -q
python -m sntp.cli run --config configs/fast_demo.yaml
```

The output directory is `results/fast_demo/`. The experiment writes each
requirement JSON after that stage completes, so a failed later stage does not
erase earlier artifacts. The CLI logs to stderr; redirect if desired:

```powershell
python -u -m sntp.cli run --config configs/fast_demo.yaml *> run.log
```

The same command with `configs/default.yaml` writes `results/full/` but was not
run; see below.

---

## 6. Results summary and runtime

**Measured fast-demo runtime: 416.84 s (6.9 min)**, including data loading,
model fitting, all eight requirement stages, and result serialization. Major
stages:

| Stage | Seconds |
|---|---:|
| R1 load dataset | 7.92 |
| R1 split | 0.59 |
| R2 item-based CF fit | 62.05 |
| R2 MF training | 168.19 |
| R3 social evaluations (all λ + comparisons) | 31.21 |
| R4 ICM + SIR + target/network comparisons | 12.05 |
| R5 influence maximization | 17.34 |
| R6 group recommendation | 1.94 |
| R7 comprehensive evaluation | 13.42 |
| R8 intervention + CELF + scaling | 88.34 |
| Total (CLI wall clock) | **416.84** |

The stage-timing sum is 415.43 s; the CLI wall clock also includes startup and
result-directory setup. MF training is the largest single stage. R8 is the
largest non-model block because it runs repeated Monte Carlo greedy evaluations
from scratch. Run-to-run wall clock has varied substantially with machine load;
the saved `stage_timings.json` is the authoritative record for the submitted
run. The offline validation selection sweep in `model_selection.json` is not
included in this runtime.

---

## 7. Report generation

The searchable PDF report is at `report/SNTP_Assignment_02_Report.pdf`. It is
generated from the saved JSON, so the report's tables stay tied to the saved
artifacts; the README summarizes the same values, and the JSON remains the
authoritative record:

```powershell
python -m pip install -r requirements-report.txt
python scripts/generate_report.py --results results/fast_demo `
  --output report/SNTP_Assignment_02_Report.pdf `
  --figures-dir results/fast_demo/plots `
  --roll-number 23F3008 --student "Ahmad Fareed"
```

`requirements-report.txt` adds `reportlab` and `matplotlib`; the core
experiments need only `requirements.txt`, and tests need
`requirements-dev.txt`. The generated PDF contains selectable text (it is not a
scanned image). The same command writes seven standalone charts to
`results/fast_demo/plots/` (`fig01_test_error.png` through
`fig07_network_scaling.png`).

---

## 8. Assumptions

* Trust is binary and static; the data has no timestamps, so trust and ratings
  are treated as one observed snapshot.
* A rating is an explicit preference signal; unobserved items are missing, not
  negative.
* A user’s training history is the only recommendation context; test and
  validation ratings are held out.
* Diffusion parameters (`p`, `β`, `γ`, base adoption probability, affinity
  mapping) are simulation assumptions, not fitted to adoption data, because no
  adoption traces exist.
* “Spread” counts adopters including the initial seeds. Marginal gains are
  paired differences under common random numbers.
* Group satisfaction is the recommender’s predicted rating; groups are
  constructed from training ratings only and may overlap across groups.
* Catalogue coverage is computed against all 139,738 items, but its denominator
  scope is the evaluated user sample (500 users in the fast demo).
* The fast-demo ego subgraph is a bounded, deterministic 2-hop neighborhood of
  user 0; it is not a random sample of the full trust graph.

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
├── tests/                      # 94 tests
├── scripts/generate_report.py  # regenerates the PDF and PNGs from JSON
├── report/SNTP_Assignment_02_Report.pdf
├── results/fast_demo/          # submitted measured results
│   ├── data_summary.json
│   ├── split_info.json
│   ├── experiment_config.json
│   ├── stage_timings.json
│   ├── requirement_{2..8}_results.json
│   └── plots/                  # 7 standalone PNG figures
└── Assignment/ratings_data/epinions/
    ├── ratings_data.txt
    └── trust_data.txt
```

Submission archive:

```
23F3008_SNTP_Assignment.zip
└── 23F3008_SNTP_Assignment/
```

The archive uses forward-slash paths, contains one inner folder, and excludes
generated `*.egg-info/`, `__pycache__`, `.pytest_cache`, `.git`, logs, and prior
ZIPs.

---

## 10. Limitations and deviations

1. **Full-scale configuration not executed.** `configs/default.yaml` uses the
   full graph and much larger Monte Carlo/CELF budgets; its estimated runtime is
   many hours. All reported numbers are from the bounded fast demo. No
   unexecuted full-run result is claimed.
2. **Fast-demo scope.** Diffusion, influence, group, and intervention results
   use the 2-hop ego subgraph (9,883 users, 301,710 edges), not the full 49,289
   users / 487,002 trust edges. R5 further restricts candidates to the top 15
   out-degree users across all methods. The fast demo uses a smaller ICF
   neighbourhood than the full config (20 vs 50); MF uses 20 factors/20 epochs
   with the learning-rate, regularization, and bias-corrected ranking setting
   selected on validation. The full config was not run.
3. **Monte Carlo noise.** R4 has only 3 simulations; R5 uses 2 evaluation
   simulations and 3 random trials; R6 uses 2 simulations per item; R8 CELF uses
   2. Spread distributions are heavy-tailed. Means and standard deviations are
   reported; small differences are not claimed as significant.
4. **R5 does not show greedy superiority.** On the restricted top-degree pool,
   degree centrality is best at budgets 3 and 5; greedy is best only at budget
   1. The dense graph is near saturation (~47% of active users), compressing
   differences. The paired greedy−degree difference at budget 5 is negative.
5. **R6 strategy differences are small.** The mean-satisfaction spread between
   strategies is about 0.257; spread is dominated by rare large cascades.
   Candidate items are only the union of strategy top items, not all catalogue
   items.
6. **R7 is a sample-based evaluation.** Ranking uses 500 seeded users, not all
   25,045 eligible users. Coverage/diversity are list-sample properties.
   Unobserved items are not negatives, so precision/recall are conservative and
   no accuracy metric is reported. The simulation retains the rare target and
   separately adds a moderate-popularity target in Part 4; neither is a
   platform-wide adoption forecast.
7. **R8 threshold is a simulator intervention.** No online randomized
   experiment was run. The 0.8 threshold cuts simulated spread by 79.2% while
   raising the rating proxy; 0.9 cuts 99.1% with a smaller positive proxy
   change. In the baseline exposure population, 41.78% have affinity below 0.8
   and 85.10% below 0.9; 0.3 is inactive because no exposed user is below it.
   The rating proxy is predicted, not observed. CELF's 1.257x speedup is a
   single noisy-Monte-Carlo measurement; CELF selected different seeds and is
   not guaranteed to be faster or identical under finite samples. The
   deterministic controlled check is the correctness evidence.
8. **Memory measurement.** `tracemalloc` measures Python allocations, and RSS
   deltas are near zero because large arrays are already resident. NumPy native
   buffers are under-represented; do not read the MB columns as total process
   memory.
9. **Runtime variability.** Timings depend on CPU load and Python/NumPy/BLAS
   versions. The saved stage timings are from one run; prior runs of the same
   config varied by roughly 1.6×.
10. **Dataset duplication.** The repository contains a duplicate trust file at
    `Assignment/trust_data/epinions/trust_data.txt`; the loader uses the
    `ratings_data/epinions` copy and only one is needed. The submission archive
    includes the `ratings_data/epinions` copy.

---

## 11. What was fixed relative to the pre-existing implementation

The original project ran end-to-end but had several correctness and evidence
problems that were fixed before the submitted results:

* R6 group construction used `dataset.rating_matrix` (all observed ratings,
  including validation/test). It now uses a matrix built explicitly from
  training triples; a regression test mutates the full matrix and proves the
  groups do not change.
* R3 chose the social λ without a validation-only rule. It now selects on
  validation RMSE and reports `test_used_for_selection: false`.
* R4 did not guarantee identical pre-generated seed sets across ICM/SIR
  settings. Seed sets are now generated once and reused; common simulation
  seeds are recorded.
* R5 could stop greedy selection after a negative Monte Carlo marginal gain
  because the sentinel was `-1.0`. It now uses `-inf` and fills the full budget.
  Candidate-pool restriction and common evaluation seeds are explicit.
* R5’s marginal gains were not paired. They are now paired differences on the
  same Monte Carlo trials, with CIs and win/loss counts.
* Recommendation ranking used clipped scores, creating artificial 5.0 ties.
  It now orders by unclipped model scores and clips only the reported predicted
  rating; batch and single-user paths have regression coverage. Item-based CF
  unsupported items now use a documented training-only fallback.
* Full-score MF ranking was dominated by the global item-bias component
  (99.91% of within-user score variance; 97.86% top-10 overlap with a bias-only
  ranking). The ranking score now has a configurable item-bias weight; the
  submitted weight 0.05 was selected on validation. Prediction error and
  returned ratings still use the full model, and no random ranking noise was
  added.
* Diversity metadata now records whether the evaluated lists came from the
  seeded ranking-user sample, the correct sample seed, and the sampling scope.
  The earlier `is_sampled=false`/`null` values described only the internal
  list cap and were wrong for the 500-user sample.
* The submission archive excludes generated `src/sntp.egg-info/` metadata as
  well as caches, `.git`, and prior ZIPs.
* MF cold-start users/items could expose random initial factors. Training
  observations are tracked, untrained rows are zeroed, and the documented
  fallbacks are μ+b_i, μ+b_u, or μ.
* R7’s F1 was an aggregate harmonic mean. It is now the mean of per-user F1
  values, and diversity is feature-based from training item vectors, with
  sampling metadata.
* R7’s simulated diffusion was mixed with observed metrics. It is now a
  separate section with its own seeds and target item.
* R4 used a quantile popularity band without an explicit rare item. It now
  retains the one-training-rating rare target and adds a documented 6–20
  moderate target, plus a degree-preserving network-structure comparison.
* R8 contained asserted CELF/speedup/seed claims. Conclusions are now generated
  from measured evaluations, runtime, memory, seed agreement, and a controlled
  deterministic check.
* R8’s intervention findings asserted a monotone spread reduction. They now
  report the measured threshold-by-threshold effects, including the inactive
  0.3 threshold, the non-monotone 0.6 result, and the 0.8/0.9 trade-off.
* Diffusion `n_rounds` was not meaningful in the original code. ICM/SIR now
  report actual attempted rounds and direct ever-infected counts.
* Tests were added for leakage, seed reproducibility, tie-aware group
  bookkeeping, feature diversity, negative-gain greedy behavior, candidate-pool
  respect, paired marginal gains, unclipped/bias-weighted ranking, cold starts, explicit
  target bands, and degree-preserving randomization.
* The report generator now reads target/network keys dynamically, exports
  seven PNG charts under `results/fast_demo/plots/`, and no longer forces a
  page break that produced an almost-empty page.

All reported results were produced after these fixes. The pre-fix baseline
artifacts were used only to confirm the defects and are not part of the
submitted results.
