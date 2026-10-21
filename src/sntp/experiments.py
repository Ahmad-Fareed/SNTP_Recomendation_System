"""
Experiment runner for SNTP Assignment 2, Parts 1-8.

Orchestrates:
  - Requirement 1: Data loading, statistics, and split.
  - Requirement 2: Item-based CF and MF training + evaluation.
  - Requirement 3: Social extension experiments.
  - Requirement 4: ICM and SIR diffusion simulations.
  - Requirement 5: Influence maximization under a promotion budget.
  - Requirement 6: Group recommendation and simulated spread.
  - Requirement 7: Comprehensive observed evaluation and simulated interaction.
  - Requirement 8: Intervention, CELF optimization, and cost scaling.

All experiments save machine-readable results (JSON) alongside configuration.
"""

import json
import logging
import os
import time
import tracemalloc
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from sntp.data import load_dataset
from sntp.splits import split_ratings
from sntp.selection import (
    build_rating_matrix,
    generate_seed_sets,
    make_simulation_seeds,
    randomize_directed_graph_degree_preserving,
    select_target_item,
)
from sntp.metrics import (
    evaluate_predictions,
    evaluate_ranking,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    recommendation_coverage,
    intra_list_diversity,
)
from sntp.recommenders.item_based import ItemBasedCF
from sntp.recommenders.matrix_factorization import BiasedMF
from sntp.recommenders.social import SocialRecommender, SimilarUserRecommender
from sntp.diffusion.icm import ICMRandomness, ICMSimulator
from sntp.diffusion.sir import SIRRandomness, SIRSimulator
from sntp.influence import (
    estimate_spread,
    random_seed_selection,
    degree_centrality_seed_selection,
    greedy_influence_maximization,
    compute_adopter_overlap,
)
from sntp.groups import (
    generate_groups,
    recommend_for_group,
    score_group_items,
    evaluate_group_satisfaction,
)

logger = logging.getLogger(__name__)

# Stage timings collected during run_all(); saved to stage_timings.json.
_STAGE_TIMINGS: list[dict] = []


def _stage_start(name: str) -> float:
    """Log the start of a pipeline stage and return its start time."""
    logger.info(">>> START: %s", name)
    return time.time()


def _stage_end(name: str, t0: float) -> float:
    """Log the end of a pipeline stage with elapsed time and record it."""
    elapsed = time.time() - t0
    _STAGE_TIMINGS.append({"stage": name, "seconds": round(elapsed, 2)})
    logger.info("<<< DONE:  %s (%.1fs)", name, elapsed)
    return elapsed


def _save_json(data, path):
    """Save dict to JSON, converting sets and numpy types."""
    def _convert(obj):
        if isinstance(obj, set):
            return sorted(list(obj))
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_convert)
    logger.info("Saved: %s", path)


def _build_user_item_dict(triples, user_to_idx, item_to_idx):
    """Build user_idx -> {item_idx: rating} dict from (uid, iid, r) triples."""
    d = defaultdict(dict)
    for uid, iid, r in triples:
        uidx = user_to_idx.get(uid)
        iidx = item_to_idx.get(iid)
        if uidx is not None and iidx is not None:
            d[uidx][iidx] = float(r)
    return dict(d)


def _build_train_user_items_set(triples, user_to_idx, item_to_idx):
    """Build user_idx -> set(item_idx) from training triples."""
    d = defaultdict(set)
    for uid, iid, _ in triples:
        uidx = user_to_idx.get(uid)
        iidx = item_to_idx.get(iid)
        if uidx is not None and iidx is not None:
            d[uidx].add(iidx)
    return dict(d)


# ---------------------------------------------------------------------------
# Requirement 1: Data pipeline
# ---------------------------------------------------------------------------

def run_requirement_1(config: dict, results_dir: Path):
    """Load data, compute statistics, create split."""
    logger.info("=" * 60)
    logger.info("REQUIREMENT 1: Data loading and representation")
    logger.info("=" * 60)

    data_dir = config["data_dir"]
    seed = config.get("split_seed", 42)

    ts = _stage_start("R1: load dataset")
    dataset = load_dataset(data_dir)
    summary = dataset.summary()
    _save_json(summary, results_dir / "data_summary.json")
    _stage_end("R1: load dataset", ts)

    # Split
    ts = _stage_start("R1: train/val/test split")
    train, val, test, split_info = split_ratings(
        dataset.raw_ratings, seed=seed,
        train_frac=config.get("train_frac", 0.70),
        val_frac=config.get("val_frac", 0.15),
    )
    _save_json(split_info, results_dir / "split_info.json")
    _stage_end("R1: train/val/test split", ts)

    logger.info("Data summary: %d users, %d items, %d ratings, %d trust edges",
                summary["n_users"], summary["n_items"],
                summary["n_ratings"], summary["n_trust_edges"])
    logger.info("Sparsity: %s", summary["sparsity"])
    logger.info("Split: train=%d, val=%d, test=%d",
                len(train), len(val), len(test))

    return dataset, train, val, test, split_info


# ---------------------------------------------------------------------------
# Requirement 2: Two recommendation approaches
# ---------------------------------------------------------------------------

def run_requirement_2(dataset, train, val, test, config: dict, results_dir: Path):
    """Train and evaluate item-based CF and MF recommenders."""
    logger.info("=" * 60)
    logger.info("REQUIREMENT 2: Recommendation approaches")
    logger.info("=" * 60)

    from sntp.data import EpinionsDataset
    from scipy import sparse

    # Build training data structures
    train_user_ratings = defaultdict(list)
    train_item_ratings = defaultdict(list)
    rows, cols, vals = [], [], []
    for uid, iid, r in train:
        uidx = dataset.user_to_idx.get(uid)
        iidx = dataset.item_to_idx.get(iid)
        if uidx is not None and iidx is not None:
            train_user_ratings[uidx].append((iidx, r))
            train_item_ratings[iidx].append((uidx, r))
            rows.append(uidx)
            cols.append(iidx)
            vals.append(r)

    train_matrix = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(dataset.n_users, dataset.n_items),
        dtype=np.float32,
    )

    train_user_items = _build_train_user_items_set(train, dataset.user_to_idx, dataset.item_to_idx)

    results = {}

    # --- A. Item-based CF ---
    logger.info("--- Item-based Collaborative Filtering ---")
    icf_config = config.get("item_based_cf", {})
    icf = ItemBasedCF(
        max_neighbors=icf_config.get("max_neighbors", 50),
        min_corated=icf_config.get("min_corated", 2),
    )
    t0 = time.time()
    ts = _stage_start("R2: item-based CF fit")
    icf.fit(train_matrix, dict(train_user_ratings), dict(train_item_ratings))
    icf_train_time = time.time() - t0
    _stage_end("R2: item-based CF fit", ts)

    # Evaluate on test set
    ts = _stage_start("R2: item-based CF MAE/RMSE")
    icf_pred_results = evaluate_predictions(
        icf, test, dataset.user_to_idx, dataset.item_to_idx
    )
    _stage_end("R2: item-based CF MAE/RMSE", ts)
    icf_pred_results["train_time_seconds"] = round(icf_train_time, 2)
    logger.info("Item-based CF test: MAE=%.4f, RMSE=%.4f, evaluated=%d",
                icf_pred_results["mae"], icf_pred_results["rmse"],
                icf_pred_results["n_evaluated"])

    # Ranking evaluation
    k = config.get("top_k", 10)
    sample_users = config.get("ranking_sample_users", None)
    sample_seed = config.get("ranking_sample_seed", 42)
    ts = _stage_start("R2: item-based CF top-K ranking")
    icf_rank_results = evaluate_ranking(
        icf, test, train_user_items,
        dataset.user_to_idx, dataset.item_to_idx,
        k=k, relevance_threshold=config.get("relevance_threshold", 4),
        label="ICF",
        sample_users=sample_users, sample_seed=sample_seed,
    )
    _stage_end("R2: item-based CF top-K ranking", ts)
    logger.info("Item-based CF ranking: P@%d=%.4f, HR@%d=%.4f",
                k, icf_rank_results["precision_at_k"],
                k, icf_rank_results["hit_rate_at_k"])

    results["item_based_cf"] = {
        "config": icf_config,
        "prediction": icf_pred_results,
        "ranking": icf_rank_results,
    }

    # --- B. Matrix Factorization ---
    logger.info("--- Biased Matrix Factorization ---")
    mf_config = config.get("matrix_factorization", {})
    mf = BiasedMF(
        n_factors=mf_config.get("n_factors", 20),
        lr=mf_config.get("lr", 0.005),
        reg=mf_config.get("reg", 0.02),
        n_epochs=mf_config.get("n_epochs", 20),
        seed=mf_config.get("seed", 42),
        ranking_item_bias_weight=mf_config.get("ranking_item_bias_weight", 1.0),
    )
    t0 = time.time()
    ts = _stage_start("R2: MF training")
    mf.fit(train, dataset.n_users, dataset.n_items,
           dataset.user_to_idx, dataset.item_to_idx, val_triples=val)
    mf_train_time = time.time() - t0
    _stage_end("R2: MF training", ts)

    ts = _stage_start("R2: MF MAE/RMSE")
    mf_pred_results = evaluate_predictions(
        mf, test, dataset.user_to_idx, dataset.item_to_idx
    )
    _stage_end("R2: MF MAE/RMSE", ts)
    mf_pred_results["train_time_seconds"] = round(mf_train_time, 2)
    logger.info("MF test: MAE=%.4f, RMSE=%.4f, evaluated=%d",
                mf_pred_results["mae"], mf_pred_results["rmse"],
                mf_pred_results["n_evaluated"])

    ts = _stage_start("R2: MF top-K ranking")
    mf_rank_results = evaluate_ranking(
        mf, test, train_user_items,
        dataset.user_to_idx, dataset.item_to_idx,
        k=k, relevance_threshold=config.get("relevance_threshold", 4),
        label="MF",
        sample_users=sample_users, sample_seed=sample_seed,
    )
    _stage_end("R2: MF top-K ranking", ts)
    logger.info("MF ranking: P@%d=%.4f, HR@%d=%.4f",
                k, mf_rank_results["precision_at_k"],
                k, mf_rank_results["hit_rate_at_k"])

    results["matrix_factorization"] = {
        "config": {k_: v_ for k_, v_ in mf_config.items()},
        "prediction": mf_pred_results,
        "ranking": mf_rank_results,
        "epoch_history": mf.epoch_history,
    }

    # --- Behavior analysis: short vs. long history ---
    logger.info("--- Behavior analysis by user history length ---")
    ts = _stage_start("R2: behaviour/popularity group analysis")
    user_train_counts = defaultdict(int)
    for uid, _, _ in train:
        user_train_counts[uid] += 1

    # Group users: short (<=5 ratings), medium (6-20), long (>20)
    groups = {
        "short_1_5": [],
        "medium_6_20": [],
        "long_21_plus": [],
    }
    for uid, count in user_train_counts.items():
        if count <= 5:
            groups["short_1_5"].append(uid)
        elif count <= 20:
            groups["medium_6_20"].append(uid)
        else:
            groups["long_21_plus"].append(uid)

    behavior_results = {}
    for group_name, group_users in groups.items():
        group_set = set(group_users)
        group_test = [(u, i, r) for u, i, r in test if u in group_set]
        if not group_test:
            behavior_results[group_name] = {"n_users": len(group_users), "n_test_ratings": 0}
            continue

        icf_group = evaluate_predictions(icf, group_test, dataset.user_to_idx, dataset.item_to_idx)
        mf_group = evaluate_predictions(mf, group_test, dataset.user_to_idx, dataset.item_to_idx)

        behavior_results[group_name] = {
            "n_users": len(group_users),
            "n_test_ratings": len(group_test),
            "item_based_cf": icf_group,
            "matrix_factorization": mf_group,
        }
        logger.info("Group %s (%d users, %d test): ICF MAE=%.4f, MF MAE=%.4f",
                    group_name, len(group_users), len(group_test),
                    icf_group["mae"], mf_group["mae"])

    results["behavior_analysis"] = behavior_results

    # --- Item popularity analysis ---
    item_train_counts = defaultdict(int)
    for _, iid, _ in train:
        item_train_counts[iid] += 1

    item_groups = {
        "rare_1_5": [],
        "moderate_6_20": [],
        "popular_21_plus": [],
    }
    for iid, count in item_train_counts.items():
        if count <= 5:
            item_groups["rare_1_5"].append(iid)
        elif count <= 20:
            item_groups["moderate_6_20"].append(iid)
        else:
            item_groups["popular_21_plus"].append(iid)

    item_behavior = {}
    for group_name, group_items in item_groups.items():
        group_set = set(group_items)
        group_test = [(u, i, r) for u, i, r in test if i in group_set]
        if not group_test:
            item_behavior[group_name] = {"n_items": len(group_items), "n_test_ratings": 0}
            continue

        icf_group = evaluate_predictions(icf, group_test, dataset.user_to_idx, dataset.item_to_idx)
        mf_group = evaluate_predictions(mf, group_test, dataset.user_to_idx, dataset.item_to_idx)

        item_behavior[group_name] = {
            "n_items": len(group_items),
            "n_test_ratings": len(group_test),
            "item_based_cf": icf_group,
            "matrix_factorization": mf_group,
        }

    results["item_popularity_analysis"] = item_behavior
    _stage_end("R2: behaviour/popularity group analysis", ts)

    _save_json(results, results_dir / "requirement_2_results.json")
    return icf, mf, results


# ---------------------------------------------------------------------------
# Requirement 3: Social extension
# ---------------------------------------------------------------------------

def _make_social_recommender(baseline, dataset, train_user_dict, social_weight):
    """Build a SocialRecommender with training-only neighbour ratings."""
    rec = SocialRecommender(
        baseline,
        dataset.trust_graph,
        social_weight=float(social_weight),
        global_mean=getattr(baseline, "global_mean", 3.0),
    )
    rec.set_training_data(train_user_dict)
    return rec


def run_requirement_3(dataset, train, val, test, baseline, config: dict, results_dir: Path):
    """Social-enhanced recommendation experiments.

    Hyper-parameter selection (the social weight lambda) is performed on the
    **validation** split.  The test split is used only after the selected
    lambda has been fixed, and every lambda is also reported on test for
    transparency.
    """
    logger.info("=" * 60)
    logger.info("REQUIREMENT 3: Social extension")
    logger.info("=" * 60)

    train_user_dict = _build_user_item_dict(train, dataset.user_to_idx, dataset.item_to_idx)
    train_user_items = _build_train_user_items_set(train, dataset.user_to_idx, dataset.item_to_idx)

    social_config = config.get("social", {})
    lambda_values = list(social_config.get("lambda_values", [0.0, 0.1, 0.3, 0.5, 0.7]))
    k_similar = social_config.get("k_similar", 20)
    k = config.get("top_k", 10)
    sample_users = config.get("ranking_sample_users", None)
    sample_seed = config.get("ranking_sample_seed", 42)

    results = {}

    # --- Baseline (lambda=0, same as underlying recommender) ---
    ts = _stage_start("R3: baseline evaluation")
    baseline_val = evaluate_predictions(
        baseline, val, dataset.user_to_idx, dataset.item_to_idx
    )
    baseline_pred = evaluate_predictions(
        baseline, test, dataset.user_to_idx, dataset.item_to_idx
    )
    baseline_rank = evaluate_ranking(
        baseline, test, train_user_items,
        dataset.user_to_idx, dataset.item_to_idx,
        k=k, relevance_threshold=config.get("relevance_threshold", 4),
        label="R3 baseline",
        sample_users=sample_users, sample_seed=sample_seed,
    )
    _stage_end("R3: baseline evaluation", ts)
    results["baseline"] = {
        "validation": baseline_val,
        "prediction": baseline_pred,
        "ranking": baseline_rank,
    }
    logger.info("Baseline: val MAE=%.4f, test MAE=%.4f",
                baseline_val["mae"], baseline_pred["mae"])

    # --- Validation-based selection of the social weight ---
    ts = _stage_start("R3: validation lambda selection")
    validation_rows = []
    for lam in lambda_values:
        rec = _make_social_recommender(baseline, dataset, train_user_dict, lam)
        val_pred = evaluate_predictions(rec, val, dataset.user_to_idx, dataset.item_to_idx)
        validation_rows.append({"lambda": float(lam), **val_pred})
    # Lowest validation RMSE wins; MAE and then the smaller lambda break ties.
    best_row = min(validation_rows, key=lambda row: (
        float("inf") if np.isnan(row["rmse"]) else row["rmse"],
        float("inf") if np.isnan(row["mae"]) else row["mae"],
        row["lambda"],
    ))
    best_lambda = float(best_row["lambda"])
    results["selection"] = {
        "criterion": (
            "minimum validation RMSE; ties by validation MAE, then smaller lambda"
        ),
        "candidates": validation_rows,
        "best_lambda": best_lambda,
        "test_used_for_selection": False,
    }
    results["best_lambda"] = best_lambda
    _stage_end("R3: validation lambda selection", ts)
    logger.info("Selected social lambda=%.3f on validation (RMSE=%.4f)",
                best_lambda, best_row["rmse"])

    # --- Social-enhanced at every lambda (test evaluation after selection) ---
    for lam in lambda_values:
        ts = _stage_start(f"R3: social lambda={lam}")
        social_rec = _make_social_recommender(baseline, dataset, train_user_dict, lam)
        soc_pred = evaluate_predictions(
            social_rec, test, dataset.user_to_idx, dataset.item_to_idx
        )
        soc_rank = evaluate_ranking(
            social_rec, test, train_user_items,
            dataset.user_to_idx, dataset.item_to_idx,
            k=k, relevance_threshold=config.get("relevance_threshold", 4),
            label=f"Social lambda={lam}",
            sample_users=sample_users, sample_seed=sample_seed,
        )
        _stage_end(f"R3: social lambda={lam}", ts)
        results[f"social_lambda_{lam}"] = {"prediction": soc_pred, "ranking": soc_rank}
        logger.info("Social (λ=%.1f): test MAE=%.4f, RMSE=%.4f",
                    lam, soc_pred["mae"], soc_pred["rmse"])

    # --- Users with limited history: does social help? ---
    ts = _stage_start("R3: sparse-user analysis")
    user_train_counts = defaultdict(int)
    for uid, _, _ in train:
        user_train_counts[uid] += 1
    sparse_users = [uid for uid, c in user_train_counts.items() if c <= 5]
    sparse_user_set = set(sparse_users)
    sparse_test = [(u, i, r) for u, i, r in test if u in sparse_user_set]

    if sparse_test:
        social_rec = _make_social_recommender(
            baseline, dataset, train_user_dict, best_lambda
        )
        sparse_baseline = evaluate_predictions(
            baseline, sparse_test, dataset.user_to_idx, dataset.item_to_idx
        )
        sparse_social = evaluate_predictions(
            social_rec, sparse_test, dataset.user_to_idx, dataset.item_to_idx
        )
        results["sparse_users_analysis"] = {
            "n_sparse_users": len(sparse_users),
            "n_test_ratings": len(sparse_test),
            "lambda_used": best_lambda,
            "baseline": sparse_baseline,
            "social": sparse_social,
        }
        logger.info("Sparse users (%d, test=%d): baseline MAE=%.4f, social MAE=%.4f",
                    len(sparse_users), len(sparse_test),
                    sparse_baseline["mae"], sparse_social["mae"])
    _stage_end("R3: sparse-user analysis", ts)

    # --- Control: similar-but-unconnected users ---
    logger.info("Computing similar-but-unconnected user comparison (sampled)...")
    ts = _stage_start("R3: trust vs similar-user comparison")
    users_with_trust = [u for u in dataset.trust_graph if len(dataset.trust_graph[u]) > 0
                        and u in train_user_dict]
    n_sample = min(social_config.get("comparison_sample", 200), len(users_with_trust))
    rng = np.random.RandomState(config.get("split_seed", 42))
    sampled_users = (rng.choice(users_with_trust, size=n_sample, replace=False).tolist()
                     if users_with_trust else [])

    if sampled_users:
        sim_rec = SimilarUserRecommender(
            baseline, dataset.trust_graph, social_weight=best_lambda,
            k_similar=k_similar,
            global_mean=getattr(baseline, "global_mean", 3.0),
        )
        sim_rec.set_training_data(train_user_dict)
        sim_rec.compute_similar_users(sampled_users)

        sampled_set = set(dataset.idx_to_user[u] for u in sampled_users)
        sampled_test = [(u, i, r) for u, i, r in test if u in sampled_set]

        if sampled_test:
            social_rec = _make_social_recommender(
                baseline, dataset, train_user_dict, best_lambda
            )
            comp_baseline = evaluate_predictions(
                baseline, sampled_test, dataset.user_to_idx, dataset.item_to_idx
            )
            comp_social = evaluate_predictions(
                social_rec, sampled_test, dataset.user_to_idx, dataset.item_to_idx
            )
            comp_similar = evaluate_predictions(
                sim_rec, sampled_test, dataset.user_to_idx, dataset.item_to_idx
            )
            results["trust_vs_similarity_comparison"] = {
                "lambda_used": best_lambda,
                "k_similar": k_similar,
                "n_sampled_users": len(sampled_users),
                "n_test_ratings": len(sampled_test),
                "baseline": comp_baseline,
                "trust_neighbors": comp_social,
                "similar_unconnected": comp_similar,
            }
            logger.info(
                "Comparison (%d users): baseline MAE=%.4f, trust MAE=%.4f, similar MAE=%.4f",
                len(sampled_users), comp_baseline["mae"],
                comp_social["mae"], comp_similar["mae"],
            )
    _stage_end("R3: trust vs similar-user comparison", ts)

    _save_json(results, results_dir / "requirement_3_results.json")
    return results


# ---------------------------------------------------------------------------
# Shared experiment-design helpers (Parts 4-8)
# ---------------------------------------------------------------------------

def _resolve_diffusion_graph(dataset, config: dict):
    """Return the active propagation graph, its user set, and a description.

    The fast-demo configuration documents an ego-network subgraph.  Every
    diffusion/influence experiment uses this same resolver so that graph
    comparisons are made on exactly the same node and edge set.
    """
    diff_config = config.get("diffusion", {})
    subgraph_cfg = diff_config.get("subgraph", {})
    if subgraph_cfg.get("enabled", False):
        center_user = int(subgraph_cfg.get("center_user", 0))
        hops = int(subgraph_cfg.get("hops", 2))
        sub_users, _, sub_prop = dataset.get_ego_network(center_user, hops=hops)
        active_users = {int(u) for u in sub_users}
        graph = {int(u): [int(v) for v in vs] for u, vs in sub_prop.items()}
        graph_info = {
            "type": "ego_subgraph",
            "center_user": center_user,
            "hops": hops,
            "n_users": len(active_users),
            "n_edges": int(sum(len(v) for v in graph.values())),
            "definition": (
                "users reachable from the documented centre within the given "
                "number of trust/propagation hops; induced propagation edges only"
            ),
        }
        logger.info(
            "Using documented ego subgraph: center=%d, hops=%d (%d users, %d edges)",
            center_user, hops, graph_info["n_users"], graph_info["n_edges"],
        )
    else:
        active_users = set(range(dataset.n_users))
        graph = dataset.propagation_graph
        graph_info = {
            "type": "full_graph",
            "n_users": dataset.n_users,
            "n_edges": int(sum(len(v) for v in graph.values())),
            "definition": "full propagation graph over all indexed users",
        }
        logger.info(
            "Using full propagation graph: %d users, %d edges",
            graph_info["n_users"], graph_info["n_edges"],
        )
    return graph, active_users, graph_info


def _target_eligible_users(dataset, train, target_original_id, active_users):
    """Users eligible to receive a seed recommendation for ``target_original_id``.

    Only the **training** split is consulted.  A user is excluded when the
    training split already contains a rating for the target item.  Validation
    and test ratings are deliberately not read, because doing so would leak
    held-out information into the simulation design.
    """
    train_raters = set()
    for uid, iid, _ in train:
        if iid == target_original_id:
            uidx = dataset.user_to_idx.get(uid)
            if uidx is not None:
                train_raters.add(int(uidx))
    eligible = sorted(int(u) for u in active_users if int(u) not in train_raters)
    return eligible, train_raters


def _distribution(values) -> dict:
    """Mean/std/min/max/coefficient-of-variation summary for a metric."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0, "mean": None, "std": None, "min": None, "max": None,
            "coefficient_of_variation": None,
        }
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    return {
        "n": int(arr.size),
        "mean": round(mean, 4),
        "std": round(std, 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "coefficient_of_variation": (
            round(std / mean, 4) if abs(mean) > 1e-12 else None
        ),
    }


def _paired_difference(a, b) -> dict:
    """Paired difference ``a - b`` with a normal-approximation 95% interval.

    The two sequences must be aligned on the same Monte Carlo trials
    (common random numbers).
    """
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    if a_arr.shape != b_arr.shape or a_arr.size == 0:
        return {
            "n": 0, "mean_difference": None, "std_difference": None,
            "ci95_low": None, "ci95_high": None,
            "wins": 0, "ties": 0, "losses": 0,
        }
    d = a_arr - b_arr
    n = int(d.size)
    mean_d = float(np.mean(d))
    std_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    half = 1.96 * std_d / np.sqrt(n) if n > 1 else 0.0
    return {
        "n": n,
        "mean_difference": round(float(mean_d), 4),
        "std_difference": round(float(std_d), 4),
        "ci95_low": round(float(mean_d - half), 4),
        "ci95_high": round(float(mean_d + half), 4),
        "wins": int(np.sum(d > 0)),
        "ties": int(np.sum(d == 0)),
        "losses": int(np.sum(d < 0)),
    }


def _induced_subgraph(graph, nodes) -> dict:
    """Induce a propagation graph on ``nodes``."""
    keep = {int(u) for u in nodes}
    return {
        int(u): [int(v) for v in graph.get(u, []) if int(v) in keep]
        for u in keep
    }


def _reverse_graph(graph) -> dict:
    """Build the reverse adjacency list of a directed graph."""
    rev = defaultdict(list)
    for u, neighbours in graph.items():
        for v in neighbours:
            rev[int(v)].append(int(u))
    return dict(rev)


def _bfs_induced_subgraph(graph, reverse, n_nodes: int, start: int):
    """Deterministically grow a connected-ish induced subgraph with BFS.

    Both propagation and reverse edges are followed for reachability.  If the
    reachable component is smaller than ``n_nodes`` the component is returned
    as-is and the actual size is recorded by the caller.
    """
    from collections import deque

    target = max(1, int(n_nodes))
    visited = set()
    queue = deque([int(start)])
    while queue and len(visited) < target:
        u = int(queue.popleft())
        if u in visited:
            continue
        visited.add(u)
        for v in list(graph.get(u, [])) + list(reverse.get(u, [])):
            v = int(v)
            if v not in visited:
                queue.append(v)
    return _induced_subgraph(graph, visited), sorted(visited)


def _measure_call(fn):
    """Run ``fn`` and return (result, wall_seconds, memory_measurement).

    ``tracemalloc`` reports Python-level allocations; RSS deltas (when psutil
    is available) give a rough process-level view.  Both are recorded because
    NumPy allocations are not fully represented by tracemalloc.
    """
    try:
        import psutil
        process = psutil.Process()
        rss_before = process.memory_info().rss
    except Exception:  # pragma: no cover - psutil is optional
        process = None
        rss_before = None

    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    memory = {
        "tracemalloc_peak_mb": round(peak / (1024.0 * 1024.0), 3),
        "tracemalloc_current_mb": round(current / (1024.0 * 1024.0), 3),
    }
    if process is not None:
        rss_after = process.memory_info().rss
        memory["rss_before_mb"] = round(rss_before / (1024.0 * 1024.0), 3)
        memory["rss_after_mb"] = round(rss_after / (1024.0 * 1024.0), 3)
        memory["rss_delta_mb"] = round(
            (rss_after - rss_before) / (1024.0 * 1024.0), 3
        )
    else:
        memory["rss_before_mb"] = None
        memory["rss_after_mb"] = None
        memory["rss_delta_mb"] = None
    return result, elapsed, memory


# ---------------------------------------------------------------------------
# Requirement 4: Diffusion models
# ---------------------------------------------------------------------------

def _affinity_fn_factory(recommender, item_idx: int, user_to_idx: dict,
                         default_affinity: float = 0.3, n_users: int = 0):
    """Create an affinity function for a specific item.

    affinity(u) = clip((predicted_rating(u, item) - 1) / 4, 0, 1)
    Falls back to default_affinity if no prediction is available.
    Uses vectorized precomputation when predict_batch is available.
    """
    if hasattr(recommender, "predict_batch") and n_users > 0:
        all_u = np.arange(n_users, dtype=np.int64)
        all_i = np.full(n_users, item_idx, dtype=np.int64)
        preds = recommender.predict_batch(all_u, all_i)
        aff_arr = np.clip((preds - 1.0) / 4.0, 0.0, 1.0)
        aff_arr[np.isnan(aff_arr)] = default_affinity
        return lambda user_idx: float(aff_arr[user_idx]) if 0 <= user_idx < n_users else default_affinity

    def affinity(user_idx):
        pred = recommender.predict(user_idx, item_idx)
        if pred is None or np.isnan(pred):
            return default_affinity
        return float(np.clip((pred - 1.0) / 4.0, 0.0, 1.0))
    return affinity


def run_requirement_4(dataset, train, recommender, config: dict, results_dir: Path):
    """Run ICM and SIR diffusion experiments.

    Design controls
    ---------------
    * The target item is chosen from a documented **training-only** popularity
      band (see :func:`sntp.selection.select_target_item`).
    * Seed users are sampled once with :func:`generate_seed_sets` and the
      identical sets are reused for every ICM/SIR parameter setting.
    * A single controlled Monte Carlo seed schedule is reused across settings.
    * Every reported outcome is simulated; no adoption timestamp is observed
      in the Epinions data.
    """
    logger.info("=" * 60)
    logger.info("REQUIREMENT 4: Diffusion models")
    logger.info("=" * 60)

    diff_config = config.get("diffusion", {})
    n_simulations = int(diff_config.get("n_simulations", 10))
    seed_size = int(diff_config.get("seed_size", 10))
    base_seed = int(diff_config.get("seed", 42))
    max_rounds = int(diff_config.get("max_rounds", 50))
    propagation_probs = [float(p) for p in diff_config.get("propagation_probs", [0.01, 0.05, 0.1])]
    recovery_probs = [float(g) for g in diff_config.get("recovery_probs", [0.0, 0.1, 0.3, 0.5])]
    default_affinity = float(diff_config.get("default_affinity", 0.3))
    base_adopt_prob = float(diff_config.get("base_adopt_prob", 0.5))
    target_items_cfg = diff_config.get("target_items", {})
    rare_cfg = target_items_cfg.get("rare", {})
    moderate_cfg = target_items_cfg.get("moderate", {})
    rare_min = int(rare_cfg.get("min_train_ratings", 1))
    rare_max = int(rare_cfg.get("max_train_ratings", 1))
    moderate_min = int(moderate_cfg.get("min_train_ratings", 6))
    moderate_max = int(moderate_cfg.get("max_train_ratings", 20))

    if n_simulations <= 0:
        raise ValueError("diffusion.n_simulations must be positive")

    active_graph, active_users, graph_info = _resolve_diffusion_graph(dataset, config)

    # Retain the original rare-item experiment as the primary Part 4 target.
    target = select_target_item(
        train, dataset.item_to_idx,
        min_train_ratings=rare_min, max_train_ratings=rare_max,
    )
    target_orig = int(target["original_id"])
    target_idx = int(target["idx"])

    # Add a documented moderate-popularity target for a controlled comparison
    # while preserving every existing rare-target result.
    moderate_target = select_target_item(
        train, dataset.item_to_idx,
        min_train_ratings=moderate_min, max_train_ratings=moderate_max,
    )
    moderate_target_orig = int(moderate_target["original_id"])
    moderate_target_idx = int(moderate_target["idx"])

    eligible, train_raters = _target_eligible_users(
        dataset, train, target_orig, active_users
    )
    if not eligible:
        raise ValueError("No eligible seed users for the diffusion target item")
    moderate_eligible, moderate_train_raters = _target_eligible_users(
        dataset, train, moderate_target_orig, active_users
    )
    if not moderate_eligible:
        raise ValueError("No eligible seed users for the moderate target item")

    seed_sets = generate_seed_sets(
        eligible, seed_size=seed_size,
        n_simulations=n_simulations, seed=base_seed,
    )
    moderate_seed_sets = generate_seed_sets(
        moderate_eligible, seed_size=seed_size,
        n_simulations=n_simulations, seed=base_seed,
    )
    sim_seeds = make_simulation_seeds(base_seed, n_simulations, offset=0)

    # Common random outcomes: one threshold per edge per trial, generated from
    # the trial seed and reused across every seed set, propagation probability,
    # and ICM variant.  This gives genuine paired comparisons instead of
    # restarting a sequential RNG at the same integer seed.
    icm_randomness_by_seed = {
        int(s): ICMRandomness(active_graph, int(s)) for s in sim_seeds
    }

    aff_fn = _affinity_fn_factory(
        recommender, target_idx, dataset.user_to_idx,
        default_affinity=default_affinity, n_users=dataset.n_users,
    )
    moderate_aff_fn = _affinity_fn_factory(
        recommender, moderate_target_idx, dataset.user_to_idx,
        default_affinity=default_affinity, n_users=dataset.n_users,
    )

    results = {
        "design": {
            "target_item": target,
            "target_item_moderate": moderate_target,
            "target_popularity_design": (
                "The rare target is the retained original Part 4 experiment. "
                "The moderate target is an additional training-only comparison "
                "under the same graph, models, seed schedule, and seed size."
            ),
            "target_selection_uses": "training split only",
            "n_train_raters_of_target": len(train_raters),
            "n_eligible_users": len(eligible),
            "n_train_raters_of_moderate_target": len(moderate_train_raters),
            "n_eligible_users_moderate_target": len(moderate_eligible),
            "seed_size": seed_size,
            "n_simulations": n_simulations,
            "seed_sets": seed_sets,
            "simulation_seeds": sim_seeds,
            "same_seed_sets_for_all_settings": True,
            "same_simulation_seeds_for_all_settings": True,
            "common_random_numbers": {
                "icm": (
                    "one per-edge activation threshold per trial, reused across "
                    "seed sets, propagation probabilities, and both ICM variants"
                ),
                "preference_aware_icm": (
                    "shares the per-edge exposure thresholds and adds one "
                    "per-user adoption threshold per trial"
                ),
                "sir": (
                    "one transmission threshold per (edge, round) and one "
                    "recovery threshold per (user, round), reused across beta, "
                    "gamma, and seed sets for each trial"
                ),
                "not_claimed": (
                    "Thresholds are shared within a trial; separate trials are "
                    "independent. Thresholds are not shared across different "
                    "graphs because edge identities differ."
                ),
            },
            "graph_info": graph_info,
            "max_rounds": max_rounds,
            "base_adopt_prob": base_adopt_prob,
            "default_affinity": default_affinity,
            "outcome_scope": (
                "simulated adoption/exposure; observed ratings are used only "
                "to fit the recommender and to compute training popularity"
            ),
        },
        "icm_canonical": {},
        "icm_preference_aware": {},
        "sir": {},
        "comparison": {},
        "target_popularity_comparison": {},
        "network_structure_comparison": {},
    }

    # --- ICM canonical ---
    logger.info("--- ICM Canonical ---")
    ts = _stage_start("R4: ICM canonical")
    for p_prob in propagation_probs:
        key = f"p={p_prob}"
        run_results = []
        t0 = time.perf_counter()
        for sim_i, sim_seed in enumerate(sim_seeds):
            seed_users = set(seed_sets[sim_i])
            icm = ICMSimulator(
                active_graph, propagation_prob=p_prob,
                max_rounds=max_rounds, seed=sim_seed,
                common_randomness=icm_randomness_by_seed[int(sim_seed)],
            )
            result = icm.run_canonical(seed_users)
            run_results.append({
                "simulation_index": sim_i,
                "sim_seed": int(sim_seed),
                "n_adopters": int(result["n_adopters"]),
                "n_rounds": int(result["n_rounds"]),
                "rounds_attempted": int(result["rounds_attempted"]),
            })
        elapsed = time.perf_counter() - t0

        adopter_dist = _distribution(r["n_adopters"] for r in run_results)
        round_dist = _distribution(r["n_rounds"] for r in run_results)
        results["icm_canonical"][key] = {
            "propagation_prob": p_prob,
            "runs": run_results,
            "summary": {
                "adopters": adopter_dist,
                "rounds": round_dist,
                "mean_adopters": adopter_dist["mean"],
                "std_adopters": adopter_dist["std"],
                "mean_rounds": round_dist["mean"],
            },
            "runtime_seconds": round(elapsed, 4),
        }
        logger.info(
            "ICM canonical (p=%.2f): mean=%.1f ± %.1f adopters, mean rounds=%.1f",
            p_prob, adopter_dist["mean"], adopter_dist["std"],
            round_dist["mean"] or 0.0,
        )

    # --- Controlled network-structure comparison ---
    _stage_end("R4: ICM canonical", ts)
    net_cfg = diff_config.get("network_structure_comparison", {})
    if bool(net_cfg.get("enabled", True)):
        logger.info("--- Controlled network-structure comparison ---")
        ts = _stage_start("R4: network-structure comparison")
        t0 = time.perf_counter()
        randomized_graph, randomization = randomize_directed_graph_degree_preserving(
            active_graph,
            nodes=active_users,
            swap_multiplier=int(net_cfg.get("swap_multiplier", 2)),
            seed=int(net_cfg.get("seed", 4242)),
        )
        randomization_seconds = time.perf_counter() - t0
        if randomization["n_edges"] != graph_info["n_edges"]:
            raise RuntimeError(
                "randomized graph edge count differs from the observed graph"
            )
        randomized_info = {
            "type": "degree_preserving_randomization",
            "n_users": int(randomization["n_nodes"]),
            "n_edges": int(randomization["n_edges"]),
            "definition": (
                "same node set, edge count, and directed in/out-degree sequence "
                "as the observed active graph; edges rearranged by accepted "
                "double-edge swaps"
            ),
            "randomization": randomization,
            "randomization_runtime_seconds": round(randomization_seconds, 4),
            "null_model_note": (
                "This is one seeded degree-preserving rearrangement, not a "
                "claim of full mixing over all graphs with that degree sequence."
            ),
        }
        logger.info(
            "Randomized structure: %d nodes, %d edges, %d/%d swaps accepted "
            "(%.1f%%), degrees preserved",
            randomized_info["n_users"], randomized_info["n_edges"],
            randomization["swaps_accepted"], randomization["swaps_attempted"],
            100.0 * randomization["acceptance_rate"],
        )

        structure_results = {}
        randomized_icm_randomness = {
            int(s): ICMRandomness(randomized_graph, int(s)) for s in sim_seeds
        }
        for p_prob in propagation_probs:
            key = f"p={p_prob}"
            observed_summary = results["icm_canonical"][key]["summary"]
            observed_counts = [r["n_adopters"] for r in results["icm_canonical"][key]["runs"]]
            random_counts = []
            random_rounds = []
            run_rows = []
            t_start = time.perf_counter()
            for sim_i, sim_seed in enumerate(sim_seeds):
                seed_users = set(seed_sets[sim_i])
                icm = ICMSimulator(
                    randomized_graph, propagation_prob=p_prob,
                    max_rounds=max_rounds, seed=sim_seed,
                    common_randomness=randomized_icm_randomness[int(sim_seed)],
                )
                result = icm.run_canonical(seed_users)
                random_counts.append(int(result["n_adopters"]))
                random_rounds.append(int(result["n_rounds"]))
                run_rows.append({
                    "simulation_index": sim_i,
                    "sim_seed": int(sim_seed),
                    "n_adopters": int(result["n_adopters"]),
                    "n_rounds": int(result["n_rounds"]),
                })
            elapsed = time.perf_counter() - t_start
            random_dist = _distribution(random_counts)
            random_round_dist = _distribution(random_rounds)
            paired = _paired_difference(random_counts, observed_counts)
            observed_mean = observed_summary["mean_adopters"]
            random_mean = random_dist["mean"]
            relative = (
                round(100.0 * (random_mean - observed_mean) / observed_mean, 3)
                if observed_mean not in (None, 0) else None
            )
            structure_results[key] = {
                "propagation_prob": p_prob,
                "observed": observed_summary,
                "randomized": {
                    "adopters": random_dist,
                    "rounds": random_round_dist,
                    "runs": run_rows,
                    "mean_adopters": random_mean,
                    "std_adopters": random_dist["std"],
                    "mean_rounds": random_round_dist["mean"],
                },
                "paired_randomized_minus_observed": paired,
                "relative_change_pct": relative,
                "runtime_seconds": round(elapsed, 4),
            }
            logger.info(
                "Network structure (p=%.2f): observed mean=%.1f, randomized "
                "mean=%.1f (%+.1f%%), paired difference=%+.1f",
                p_prob, observed_mean or 0.0, random_mean or 0.0,
                relative or 0.0, paired["mean_difference"] or 0.0,
            )

        results["network_structure_comparison"] = {
            "design": {
                "question": (
                    "How much of the simulated cascade is explained by the "
                    "observed directed edge arrangement rather than by node "
                    "count, edge count, and degree sequence?"
                ),
                "model": "canonical ICM",
                "observed_graph": graph_info,
                "randomized_graph": randomized_info,
                "same_seed_sets": seed_sets,
                "same_simulation_seeds": sim_seeds,
                "only_structure_changes": True,
                "common_random_numbers": (
                    "Each graph has its own per-edge thresholds for a trial seed; "
                    "observed and randomized runs are therefore not paired edge "
                    "by edge, but every p within a graph reuses the same "
                    "thresholds and both graphs use the same seed sets."
                ),
                "outcome_scope": "simulated adoptions, not observed cascades",
            },
            "settings": structure_results,
            "runtime_seconds": round(time.perf_counter() - t0 - randomization_seconds, 4),
        }
        _stage_end("R4: network-structure comparison", ts)

    # --- ICM preference-aware ---
    logger.info("--- ICM Preference-Aware ---")
    ts = _stage_start("R4: ICM preference-aware")
    for p_prob in propagation_probs:
        key = f"p={p_prob}"
        run_results = []
        t0 = time.perf_counter()
        for sim_i, sim_seed in enumerate(sim_seeds):
            seed_users = set(seed_sets[sim_i])
            icm = ICMSimulator(
                active_graph, propagation_prob=p_prob,
                max_rounds=max_rounds, seed=sim_seed,
                common_randomness=icm_randomness_by_seed[int(sim_seed)],
            )
            result = icm.run_preference_aware(
                seed_users, affinity_fn=aff_fn,
                base_adopt_prob=base_adopt_prob,
            )
            run_results.append({
                "simulation_index": sim_i,
                "sim_seed": int(sim_seed),
                "n_adopters": int(result["n_adopters"]),
                "n_exposed": int(result["n_exposed"]),
                "n_rounds": int(result["n_rounds"]),
                "rounds_attempted": int(result["rounds_attempted"]),
            })
        elapsed = time.perf_counter() - t0

        adopter_dist = _distribution(r["n_adopters"] for r in run_results)
        exposed_dist = _distribution(r["n_exposed"] for r in run_results)
        round_dist = _distribution(r["n_rounds"] for r in run_results)
        results["icm_preference_aware"][key] = {
            "propagation_prob": p_prob,
            "runs": run_results,
            "summary": {
                "adopters": adopter_dist,
                "exposed": exposed_dist,
                "rounds": round_dist,
                "mean_adopters": adopter_dist["mean"],
                "std_adopters": adopter_dist["std"],
                "mean_exposed": exposed_dist["mean"],
                "std_exposed": exposed_dist["std"],
                "mean_rounds": round_dist["mean"],
            },
            "runtime_seconds": round(elapsed, 4),
        }
        logger.info(
            "ICM preference-aware (p=%.2f): mean=%.1f ± %.1f adopters, %.1f exposed",
            p_prob, adopter_dist["mean"], adopter_dist["std"],
            exposed_dist["mean"],
        )

    # --- SIR ---
    _stage_end("R4: ICM preference-aware", ts)
    logger.info("--- SIR Model ---")
    ts = _stage_start("R4: SIR")
    sir_keys = [
        (p_prob, g_prob, f"beta={p_prob}_gamma={g_prob}")
        for p_prob in propagation_probs for g_prob in recovery_probs
    ]
    sir_rows: dict[str, list[dict]] = {key: [] for _, _, key in sir_keys}
    sir_runtime: dict[str, float] = {key: 0.0 for _, _, key in sir_keys}
    sir_randomness_seconds = 0.0
    # Trial-outer loop: one SIRRandomness object per trial is reused across every
    # β/γ setting, so changing β or γ reuses identical infection/recovery draws.
    for sim_i, sim_seed in enumerate(sim_seeds):
        t_gen = time.perf_counter()
        sir_randomness = SIRRandomness(
            active_graph, max_rounds, int(sim_seed), n_users=dataset.n_users
        )
        sir_randomness_seconds += time.perf_counter() - t_gen
        seed_users = set(seed_sets[sim_i])
        for p_prob, g_prob, key in sir_keys:
            t0 = time.perf_counter()
            sir = SIRSimulator(
                active_graph, transmission_prob=p_prob,
                recovery_prob=g_prob, max_rounds=max_rounds, seed=sim_seed,
                common_randomness=sir_randomness,
            )
            result = sir.run(
                seed_users, affinity_fn=aff_fn,
                base_adopt_prob=base_adopt_prob,
                all_users=active_users,
            )
            sir_runtime[key] += time.perf_counter() - t0
            sir_rows[key].append({
                "simulation_index": sim_i,
                "sim_seed": int(sim_seed),
                "n_ever_infected": int(result["n_ever_infected"]),
                "n_final_recovered": int(result["n_final_recovered"]),
                "n_rounds": int(result["n_rounds"]),
                "rounds_attempted": int(result["rounds_attempted"]),
                "final_S": int(result["state_history"][-1]["S"]),
                "final_I": int(result["state_history"][-1]["I"]),
                "final_R": int(result["state_history"][-1]["R"]),
            })

    for p_prob, g_prob, key in sir_keys:
        run_results = sir_rows[key]
        infected_dist = _distribution(r["n_ever_infected"] for r in run_results)
        round_dist = _distribution(r["n_rounds"] for r in run_results)
        results["sir"][key] = {
            "transmission_prob": p_prob,
            "recovery_prob": g_prob,
            "runs": run_results,
            "summary": {
                "ever_infected": infected_dist,
                "rounds": round_dist,
                "mean_ever_infected": infected_dist["mean"],
                "std_ever_infected": infected_dist["std"],
                "mean_rounds": round_dist["mean"],
            },
            "runtime_seconds": round(sir_runtime[key], 4),
        }
        logger.info(
            "SIR (β=%.2f, γ=%.2f): mean=%.1f ± %.1f ever infected, mean rounds=%.1f",
            p_prob, g_prob, infected_dist["mean"], infected_dist["std"],
            round_dist["mean"] or 0.0,
        )
    _stage_end("R4: SIR", ts)
    results["design"]["common_random_numbers"]["sir_randomness_generation_seconds"] = (
        round(sir_randomness_seconds, 4)
    )

    # --- Moderate-popularity target experiment (rare target retained above) ---
    logger.info("--- Moderate-popularity target experiment ---")
    ts = _stage_start("R4: moderate-popularity target")
    t_moderate = time.perf_counter()
    moderate_canonical = {}
    moderate_preference = {}
    for p_prob in propagation_probs:
        key = f"p={p_prob}"

        canonical_runs = []
        t0 = time.perf_counter()
        for sim_i, sim_seed in enumerate(sim_seeds):
            icm = ICMSimulator(
                active_graph, propagation_prob=p_prob,
                max_rounds=max_rounds, seed=sim_seed,
                common_randomness=icm_randomness_by_seed[int(sim_seed)],
            )
            result = icm.run_canonical(set(moderate_seed_sets[sim_i]))
            canonical_runs.append({
                "simulation_index": sim_i,
                "sim_seed": int(sim_seed),
                "n_adopters": int(result["n_adopters"]),
                "n_rounds": int(result["n_rounds"]),
            })
        canonical_dist = _distribution(r["n_adopters"] for r in canonical_runs)
        canonical_rounds = _distribution(r["n_rounds"] for r in canonical_runs)
        moderate_canonical[key] = {
            "propagation_prob": p_prob,
            "runs": canonical_runs,
            "summary": {
                "adopters": canonical_dist,
                "rounds": canonical_rounds,
                "mean_adopters": canonical_dist["mean"],
                "std_adopters": canonical_dist["std"],
                "mean_rounds": canonical_rounds["mean"],
            },
            "runtime_seconds": round(time.perf_counter() - t0, 4),
        }

        preference_runs = []
        t0 = time.perf_counter()
        for sim_i, sim_seed in enumerate(sim_seeds):
            icm = ICMSimulator(
                active_graph, propagation_prob=p_prob,
                max_rounds=max_rounds, seed=sim_seed,
                common_randomness=icm_randomness_by_seed[int(sim_seed)],
            )
            result = icm.run_preference_aware(
                set(moderate_seed_sets[sim_i]), affinity_fn=moderate_aff_fn,
                base_adopt_prob=base_adopt_prob,
            )
            preference_runs.append({
                "simulation_index": sim_i,
                "sim_seed": int(sim_seed),
                "n_adopters": int(result["n_adopters"]),
                "n_exposed": int(result["n_exposed"]),
                "n_rounds": int(result["n_rounds"]),
            })
        pref_dist = _distribution(r["n_adopters"] for r in preference_runs)
        pref_exposed = _distribution(r["n_exposed"] for r in preference_runs)
        pref_rounds = _distribution(r["n_rounds"] for r in preference_runs)
        moderate_preference[key] = {
            "propagation_prob": p_prob,
            "runs": preference_runs,
            "summary": {
                "adopters": pref_dist,
                "exposed": pref_exposed,
                "rounds": pref_rounds,
                "mean_adopters": pref_dist["mean"],
                "std_adopters": pref_dist["std"],
                "mean_exposed": pref_exposed["mean"],
                "std_exposed": pref_exposed["std"],
                "mean_rounds": pref_rounds["mean"],
            },
            "runtime_seconds": round(time.perf_counter() - t0, 4),
        }
        logger.info(
            "Moderate target (p=%.2f): canonical mean=%.1f, preference mean=%.1f",
            p_prob, canonical_dist["mean"], pref_dist["mean"],
        )

    target_comparison = {}
    for p_prob in propagation_probs:
        key = f"p={p_prob}"
        rare_can = results["icm_canonical"][key]["summary"]["mean_adopters"]
        rare_pref = results["icm_preference_aware"][key]["summary"]["mean_adopters"]
        mod_can = moderate_canonical[key]["summary"]["mean_adopters"]
        mod_pref = moderate_preference[key]["summary"]["mean_adopters"]
        def _delta(new, old):
            if new is None or old is None:
                return None
            return new - old
        def _pct(new, old):
            if new is None or old in (None, 0):
                return None
            return 100.0 * (new - old) / old
        target_comparison[key] = {
            "rare_canonical_mean_adopters": rare_can,
            "moderate_canonical_mean_adopters": mod_can,
            "moderate_minus_rare_canonical": _delta(mod_can, rare_can),
            "moderate_relative_canonical_pct": _pct(mod_can, rare_can),
            "rare_preference_mean_adopters": rare_pref,
            "moderate_preference_mean_adopters": mod_pref,
            "moderate_minus_rare_preference": _delta(mod_pref, rare_pref),
            "moderate_relative_preference_pct": _pct(mod_pref, rare_pref),
        }

    results["target_popularity_comparison"] = {
        "design": {
            "question": (
                "How do the same diffusion models behave on the retained "
                "rare-item target versus a documented 6-20 training-rating "
                "target?"
            ),
            "rare_target_item": target,
            "moderate_target_item": moderate_target,
            "same_graph": graph_info,
            "same_models": ["canonical ICM", "preference-aware ICM"],
            "same_propagation_probs": propagation_probs,
            "same_simulation_seeds": sim_seeds,
            "target_specific_seed_sets": True,
            "rare_seed_sets": seed_sets,
            "moderate_seed_sets": moderate_seed_sets,
            "n_rare_eligible_users": len(eligible),
            "n_moderate_eligible_users": len(moderate_eligible),
            "outcome_scope": (
                "simulated adoptions with target-specific seed users; the "
                "rare experiment is retained, not replaced"
            ),
        },
        "rare": {
            "target_item": target,
            "seed_sets": seed_sets,
            "icm_canonical": results["icm_canonical"],
            "icm_preference_aware": results["icm_preference_aware"],
        },
        "moderate": {
            "target_item": moderate_target,
            "seed_sets": moderate_seed_sets,
            "icm_canonical": moderate_canonical,
            "icm_preference_aware": moderate_preference,
        },
        "comparison": target_comparison,
        "runtime_seconds": round(time.perf_counter() - t_moderate, 4),
    }
    _stage_end("R4: moderate-popularity target", ts)

    # --- Data-driven comparison of the two ICM variants ---
    for p_prob in propagation_probs:
        key = f"p={p_prob}"
        canon = results["icm_canonical"][key]["summary"]["mean_adopters"]
        pref = results["icm_preference_aware"][key]["summary"]["mean_adopters"]
        delta = None if (canon is None or pref is None) else pref - canon
        pct = None
        if delta is not None and canon not in (None, 0):
            pct = 100.0 * delta / canon
        results["comparison"][key] = {
            "canonical_mean_adopters": canon,
            "preference_aware_mean_adopters": pref,
            "preference_minus_canonical": None if delta is None else round(delta, 3),
            "preference_relative_change_pct": None if pct is None else round(pct, 2),
            "note": (
                "Both variants reuse the same seed sets and Monte Carlo seeds; "
                "differences are simulation outcomes, not observed adoptions."
            ),
        }

    _save_json(results, results_dir / "requirement_4_results.json")
    return results


# ---------------------------------------------------------------------------
# Requirement 5: Influence Maximization
# ---------------------------------------------------------------------------

def _top_degree_candidates(graph, users, limit: int) -> list[int]:
    """Top out-degree users, with deterministic ascending-index tie-breaking."""
    pool = sorted({int(u) for u in users})
    if limit is None or limit <= 0 or limit >= len(pool):
        return pool
    pool.sort(key=lambda u: (-len(graph.get(u, [])), u))
    return sorted(pool[:int(limit)])


def _jaccard(a, b) -> float:
    a_set, b_set = set(a), set(b)
    union = a_set | b_set
    return float(len(a_set & b_set) / len(union)) if union else 1.0


def run_requirement_5(dataset, config: dict, results_dir: Path):
    """Run Influence Maximization experiments (Part 5).

    Fair-comparison controls
    ------------------------
    * Random, degree, and greedy selection all draw from the **same**
      candidate pool on the **same** propagation graph.
    * Every method is evaluated at the **same** budgets with the **same**
      Monte Carlo seeds (common random numbers).
    * Marginal gains are paired simulation differences between consecutive
      evaluated budgets, not differences of separately seeded averages.
    * Greedy candidates are scored by their marginal contribution to the
      current seed set.  The random baseline is repeated over several seeded
      trials so its sampling variability is visible.
    """
    logger.info("=" * 60)
    logger.info("REQUIREMENT 5: Influence Maximization")
    logger.info("=" * 60)
    ts = _stage_start("R5: Influence Maximization")

    active_graph, active_users, graph_info = _resolve_diffusion_graph(dataset, config)

    infl_cfg = config.get("influence", {})
    budgets = sorted({int(b) for b in infl_cfg.get("budgets", [1, 3, 5]) if int(b) > 0})
    if not budgets:
        raise ValueError("influence.budgets must contain at least one positive budget")
    p_prob = float(infl_cfg.get("propagation_prob", 0.05))
    n_sims = int(infl_cfg.get("n_simulations", 3))
    cand_pool_size = infl_cfg.get("candidate_pool_size", 30)
    seed = int(infl_cfg.get("seed", 42))
    n_random_trials = int(infl_cfg.get("random_trials", 5))
    max_rounds = int(config.get("diffusion", {}).get("max_rounds", 100))
    max_b = max(budgets)

    eligible = sorted(int(u) for u in active_users)
    pool = _top_degree_candidates(
        active_graph, eligible,
        int(cand_pool_size) if cand_pool_size is not None else len(eligible),
    )
    pool_restricted = len(pool) < len(eligible)

    selection_seeds = make_simulation_seeds(seed, n_sims, offset=0)
    evaluation_seeds = make_simulation_seeds(seed, n_sims, offset=500_000)
    # One threshold set per evaluation trial, reused for every method and budget.
    evaluation_randomness = {
        int(s): ICMRandomness(active_graph, int(s)) for s in evaluation_seeds
    }

    results = {
        "design": {
            "graph_info": graph_info,
            "propagation_prob": p_prob,
            "budgets": budgets,
            "n_simulations": n_sims,
            "selection_seeds": selection_seeds,
            "evaluation_seeds": evaluation_seeds,
            "n_random_trials": n_random_trials,
            "max_rounds": max_rounds,
            "candidate_pool": {
                "n_eligible": len(eligible),
                "n_candidates": len(pool),
                "restricted": bool(pool_restricted),
                "rule": (
                    "top out-degree users in the propagation graph, ties by "
                    "ascending user index"
                    if pool_restricted else
                    "all eligible users on the active graph"
                ),
                "same_pool_for_all_methods": True,
            },
            "marginal_gain_definition": (
                "paired mean difference between the same Monte Carlo trials at "
                "budget b and at the previous evaluated budget; the number of "
                "added seeds is recorded per row"
            ),
            "common_random_numbers": (
                "One per-edge activation threshold per trial is reused for every "
                "seed set, budget, candidate, and both greedy variants. The "
                "finite-trial objective is therefore deterministic and monotone, "
                "so extra seeds cannot reduce measured spread and marginal gains "
                "cannot be negative from fresh Monte Carlo noise."
            ),
            "outcome_scope": "simulated canonical-ICM spread on the active graph",
        },
        "methods": {"random": {}, "degree": {}, "greedy": {}},
        "adopter_overlaps": {},
        "seed_overlaps": {},
        "runtimes_seconds": {},
        "analysis": {},
    }

    # --- Greedy selection ---
    logger.info("Running greedy influence maximization up to budget %d...", max_b)
    t0_greedy = time.perf_counter()
    greedy_res = greedy_influence_maximization(
        propagation_graph=active_graph,
        eligible_users=pool,
        budget=max_b,
        propagation_prob=p_prob,
        n_simulations=n_sims,
        candidate_pool=pool,
        selection_seeds=selection_seeds,
        seed=seed,
        use_celf=False,
    )
    greedy_selection_time = time.perf_counter() - t0_greedy

    # --- Degree centrality selection (same candidate pool) ---
    logger.info("Running degree centrality seed selection up to budget %d...", max_b)
    t0_deg = time.perf_counter()
    all_deg_seeds = degree_centrality_seed_selection(
        propagation_graph=active_graph,
        eligible_users=pool,
        budget=max_b,
    )
    deg_selection_time = time.perf_counter() - t0_deg

    # --- Random selection: repeated seeded trials from the same pool ---
    logger.info("Running %d random seed-selection trials from the same pool...", n_random_trials)
    t0_rnd = time.perf_counter()
    random_trials = []
    for trial in range(n_random_trials):
        trial_seed = seed + 1000 * (trial + 1)
        seeds_full = random_seed_selection(pool, budget=max_b, seed=trial_seed)
        trial_record = {
            "trial": trial,
            "selection_seed": trial_seed,
            "seeds": seeds_full,
            "by_budget": {},
        }
        for b in budgets:
            seeds_b = seeds_full[:b]
            mean_s, std_s, adopter_sets = estimate_spread(
                active_graph, seeds_b,
                propagation_prob=p_prob,
                n_simulations=n_sims,
                max_rounds=max_rounds,
                sim_seeds=evaluation_seeds,
                randomness_by_seed=evaluation_randomness,
            )
            counts = [len(s) for s in adopter_sets]
            trial_record["by_budget"][str(b)] = {
                "mean_spread": round(mean_s, 4),
                "std_spread": round(std_s, 4),
                "simulation_spread_counts": counts,
                "adopter_sets": adopter_sets if b == max_b else None,
            }
        random_trials.append(trial_record)
    random_selection_time = time.perf_counter() - t0_rnd

    # --- Evaluate degree and greedy at each budget with the same seeds ---
    method_raw = {
        "random": {b: [] for b in budgets},
        "degree": {},
        "greedy": {},
    }
    t0_eval_deg = time.perf_counter()
    for b in budgets:
        seeds_b = all_deg_seeds[:b]
        mean_s, std_s, adopter_sets = estimate_spread(
            active_graph, seeds_b,
            propagation_prob=p_prob,
            n_simulations=n_sims,
            max_rounds=max_rounds,
            sim_seeds=evaluation_seeds,
            randomness_by_seed=evaluation_randomness,
        )
        method_raw["degree"][b] = {
            "seeds": seeds_b,
            "mean_spread": round(mean_s, 4),
            "std_spread": round(std_s, 4),
            "simulation_spread_counts": [len(s) for s in adopter_sets],
            "adopter_sets": adopter_sets if b == max_b else None,
        }
    deg_eval_time = time.perf_counter() - t0_eval_deg

    t0_eval_grd = time.perf_counter()
    for b in budgets:
        seeds_b = greedy_res["seeds"][:b]
        mean_s, std_s, adopter_sets = estimate_spread(
            active_graph, seeds_b,
            propagation_prob=p_prob,
            n_simulations=n_sims,
            max_rounds=max_rounds,
            sim_seeds=evaluation_seeds,
            randomness_by_seed=evaluation_randomness,
        )
        method_raw["greedy"][b] = {
            "seeds": seeds_b,
            "mean_spread": round(mean_s, 4),
            "std_spread": round(std_s, 4),
            "simulation_spread_counts": [len(s) for s in adopter_sets],
            "adopter_sets": adopter_sets if b == max_b else None,
        }
    greedy_eval_time = time.perf_counter() - t0_eval_grd

    # --- Aggregate the random trials ---
    for b in budgets:
        trial_means = [tr["by_budget"][str(b)]["mean_spread"] for tr in random_trials]
        trial_stds = [tr["by_budget"][str(b)]["std_spread"] for tr in random_trials]
        representative = random_trials[0]
        results["methods"]["random"][str(b)] = {
            "seeds": representative["seeds"][:b],
            "seeds_from_trial": 0,
            "n_random_trials": len(random_trials),
            "mean_spread": round(float(np.mean(trial_means)), 4),
            "std_spread": round(float(np.std(trial_means)), 4),
            "mean_within_trial_std": round(float(np.mean(trial_stds)), 4),
            "trial_mean_spreads": [round(v, 4) for v in trial_means],
            "simulation_spread_counts": representative["by_budget"][str(b)]["simulation_spread_counts"],
        }

    for method in ("degree", "greedy"):
        for b in budgets:
            raw = method_raw[method][b]
            results["methods"][method][str(b)] = {
                "seeds": raw["seeds"],
                "mean_spread": raw["mean_spread"],
                "std_spread": raw["std_spread"],
                "simulation_spread_counts": raw["simulation_spread_counts"],
            }

    # --- Paired marginal gains between consecutive evaluated budgets ---
    for method in ("degree", "greedy"):
        prev_b = None
        for b in budgets:
            if prev_b is None:
                results["methods"][method][str(b)]["marginal_gain"] = (
                    results["methods"][method][str(b)]["mean_spread"]
                )
                results["methods"][method][str(b)]["marginal_gain_std"] = (
                    results["methods"][method][str(b)]["std_spread"]
                )
                results["methods"][method][str(b)]["seeds_added_since_previous_budget"] = b
                results["methods"][method][str(b)]["marginal_gain_note"] = (
                    "gain from the empty seed set"
                )
            else:
                diff = _paired_difference(
                    method_raw[method][b]["simulation_spread_counts"],
                    method_raw[method][prev_b]["simulation_spread_counts"],
                )
                results["methods"][method][str(b)]["marginal_gain"] = diff["mean_difference"]
                results["methods"][method][str(b)]["marginal_gain_std"] = diff["std_difference"]
                results["methods"][method][str(b)]["marginal_gain_ci95_low"] = diff["ci95_low"]
                results["methods"][method][str(b)]["marginal_gain_ci95_high"] = diff["ci95_high"]
                results["methods"][method][str(b)]["marginal_gain_wins"] = diff["wins"]
                results["methods"][method][str(b)]["marginal_gain_losses"] = diff["losses"]
                results["methods"][method][str(b)]["seeds_added_since_previous_budget"] = b - prev_b
                results["methods"][method][str(b)]["marginal_gain_note"] = (
                    "paired Monte Carlo difference versus the previous evaluated budget"
                )
            prev_b = b

    prev_b = None
    for b in budgets:
        if prev_b is None:
            trial_level_gain = [
                tr["by_budget"][str(b)]["mean_spread"] for tr in random_trials
            ]
            results["methods"]["random"][str(b)]["marginal_gain"] = round(
                float(np.mean(trial_level_gain)), 4
            )
            results["methods"]["random"][str(b)]["marginal_gain_std"] = round(
                float(np.std(trial_level_gain)), 4
            )
            results["methods"]["random"][str(b)]["seeds_added_since_previous_budget"] = b
            results["methods"]["random"][str(b)]["marginal_gain_note"] = (
                "gain from the empty seed set; mean across random trials"
            )
        else:
            trial_gains = []
            for tr in random_trials:
                cur = tr["by_budget"][str(b)]["simulation_spread_counts"]
                prev = tr["by_budget"][str(prev_b)]["simulation_spread_counts"]
                trial_gains.append(_paired_difference(cur, prev)["mean_difference"])
            results["methods"]["random"][str(b)]["marginal_gain"] = round(
                float(np.mean(trial_gains)), 4
            )
            results["methods"]["random"][str(b)]["marginal_gain_std"] = round(
                float(np.std(trial_gains)), 4
            )
            results["methods"]["random"][str(b)]["seeds_added_since_previous_budget"] = b - prev_b
            results["methods"]["random"][str(b)]["marginal_gain_note"] = (
                "mean paired difference across random trials"
            )
        prev_b = b

    # --- Overlap at the largest budget ---
    def _adopter_sets_for(method):
        if method == "random":
            return random_trials[0]["by_budget"][str(max_b)]["adopter_sets"] or []
        return method_raw[method][max_b]["adopter_sets"] or []

    random_sets = _adopter_sets_for("random")
    degree_sets = _adopter_sets_for("degree")
    greedy_sets = _adopter_sets_for("greedy")
    results["adopter_overlaps"] = {
        "random_vs_degree": round(compute_adopter_overlap(random_sets, degree_sets), 3),
        "degree_vs_greedy": round(compute_adopter_overlap(degree_sets, greedy_sets), 3),
        "random_vs_greedy": round(compute_adopter_overlap(random_sets, greedy_sets), 3),
        "note": (
            "Average Jaccard similarity of adopter sets across the same "
            "evaluation seeds at the largest budget; random uses trial 0."
        ),
    }
    results["seed_overlaps"] = {
        "random_vs_degree": round(_jaccard(random_trials[0]["seeds"][:max_b],
                                           all_deg_seeds[:max_b]), 3),
        "degree_vs_greedy": round(_jaccard(all_deg_seeds[:max_b],
                                           greedy_res["seeds"][:max_b]), 3),
        "random_vs_greedy": round(_jaccard(random_trials[0]["seeds"][:max_b],
                                           greedy_res["seeds"][:max_b]), 3),
    }

    results["runtimes_seconds"] = {
        "random_selection_all_trials": round(random_selection_time, 4),
        "degree_selection": round(deg_selection_time, 4),
        "greedy_selection": round(greedy_selection_time, 4),
        "degree_evaluation": round(deg_eval_time, 4),
        "greedy_evaluation": round(greedy_eval_time, 4),
        "greedy_n_evaluations": greedy_res["n_evaluations"],
        "greedy_internal_runtime": greedy_res["runtime_seconds"],
        "greedy_spread_progression": greedy_res["spread_progression"],
        "greedy_selection_seeds": greedy_res["selection_seeds"],
    }

    # --- Data-driven comparison ---
    comparison = {}
    observations = []
    for b in budgets:
        r = results["methods"]["random"][str(b)]["mean_spread"]
        d = results["methods"]["degree"][str(b)]["mean_spread"]
        g = results["methods"]["greedy"][str(b)]["mean_spread"]
        scores = {"random": r, "degree": d, "greedy": g}
        best = max(scores, key=lambda m: (scores[m], -["random", "degree", "greedy"].index(m)))
        row = {
            "budget": b,
            "mean_spreads": {k: round(v, 3) for k, v in scores.items()},
            "best_method": best,
            "greedy_minus_random": round(g - r, 3),
            "greedy_minus_degree": round(g - d, 3),
            "greedy_lift_vs_random_pct": (
                round(100.0 * (g - r) / r, 2) if r not in (0, None) else None
            ),
            "greedy_lift_vs_degree_pct": (
                round(100.0 * (g - d) / d, 2) if d not in (0, None) else None
            ),
            "random_trial_std": results["methods"]["random"][str(b)]["std_spread"],
        }
        if b == max_b:
            row["paired_greedy_vs_degree"] = _paired_difference(
                method_raw["greedy"][b]["simulation_spread_counts"],
                method_raw["degree"][b]["simulation_spread_counts"],
            )
            row["paired_greedy_vs_random_trial0"] = _paired_difference(
                method_raw["greedy"][b]["simulation_spread_counts"],
                random_trials[0]["by_budget"][str(b)]["simulation_spread_counts"],
            )
        comparison[str(b)] = row

        observations.append(
            f"At budget {b}, mean simulated spread was random={r:.2f}, "
            f"degree={d:.2f}, greedy={g:.2f}; the highest measured mean was {best}."
        )
        if b == max_b:
            if g > r:
                observations.append(
                    f"At the largest budget, greedy exceeded the mean random "
                    f"baseline by {g - r:.2f} adopters ({100.0 * (g - r) / max(r, 1e-9):.1f}%)."
                )
            elif g < r:
                observations.append(
                    f"At the largest budget, greedy was below the mean random "
                    f"baseline by {r - g:.2f} adopters."
                )
            else:
                observations.append("At the largest budget, greedy and the mean random baseline tied.")

        degree_marginal = results["methods"]["degree"][str(b)]["marginal_gain"]
        if b != budgets[0] and degree_marginal is not None and degree_marginal < 0:
            observations.append(
                f"Degree centrality's measured marginal gain at budget {b} was "
                f"negative ({degree_marginal:.2f}); extra high-degree seeds did not "
                "increase mean simulated spread in this trial set."
            )

    overlap = results["adopter_overlaps"]
    if overlap["degree_vs_greedy"] >= 0.5:
        observations.append(
            "Degree and greedy adopter sets overlapped substantially "
            f"(mean Jaccard {overlap['degree_vs_greedy']:.3f}), so high-degree "
            "seeds largely reached the same audience."
        )
    else:
        observations.append(
            "Degree and greedy adopter sets were substantially different "
            f"(mean Jaccard {overlap['degree_vs_greedy']:.3f}), indicating that "
            "the marginal strategy reached a different audience."
        )

    results["analysis"] = {
        "comparison": comparison,
        "observations": observations,
        "uncertainty_note": (
            "Random spread varies across seeded trials; std_spread for random "
            "is the standard deviation of trial means. Degree/greedy rows use "
            "the same common random numbers for paired comparisons."
        ),
        "candidate_pool_note": (
            "All methods use the same candidate pool. When the pool is "
            "restricted to top out-degree users, the result describes that "
            "restricted pool, not all graph users."
        ),
    }

    _stage_end("R5: Influence Maximization", ts)
    _save_json(results, results_dir / "requirement_5_results.json")
    return results


# ---------------------------------------------------------------------------
# Requirement 6: Group Recommendation
# ---------------------------------------------------------------------------

def run_requirement_6(dataset, train, recommender, config: dict, results_dir: Path):
    """Run Group Recommendation experiments (Part 6).

    Implements and compares average satisfaction, least misery, and most
    pleasure.  Group construction and all preference predictions use the
    **training split only**.  The diffusion outcomes are simulated and are
    kept separate from observed ratings.
    """
    logger.info("=" * 60)
    logger.info("REQUIREMENT 6: Group Recommendation")
    logger.info("=" * 60)
    ts = _stage_start("R6: Group Recommendation")

    active_graph, active_users, graph_info = _resolve_diffusion_graph(dataset, config)

    grp_cfg = config.get("group_recommendation", {})
    n_groups = int(grp_cfg.get("n_groups", 3))
    group_size = int(grp_cfg.get("group_size", 4))
    conflicting = bool(grp_cfg.get("conflicting", True))
    top_n = int(grp_cfg.get("top_n", 5))
    diff_sims = int(grp_cfg.get("diffusion_simulations", 3))
    p_prob = float(grp_cfg.get("propagation_prob", 0.05))
    base_adopt_prob = float(grp_cfg.get("base_adopt_prob", 0.5))
    seed = int(grp_cfg.get("seed", 42))
    max_rounds = int(grp_cfg.get("max_rounds", 25))
    default_affinity = float(grp_cfg.get("default_affinity", 0.3))

    # Training-only rating matrix.  Passing dataset.rating_matrix here would
    # leak validation/test preferences into the conflict-aware grouping.
    train_matrix = build_rating_matrix(
        dataset.n_users, dataset.n_items, train,
        dataset.user_to_idx, dataset.item_to_idx,
    )
    train_users_with_ratings = set(
        int(u) for u in np.flatnonzero(np.diff(train_matrix.indptr) > 0)
    )
    group_users = sorted(train_users_with_ratings & active_users)
    if len(group_users) < group_size:
        logger.warning(
            "Only %d active users with training ratings available for groups; "
            "falling back to all users with training ratings.",
            len(group_users),
        )
        group_users = sorted(train_users_with_ratings)

    groups = generate_groups(
        user_ids=group_users,
        n_groups=n_groups,
        group_size=group_size,
        conflicting=conflicting,
        user_item_matrix=train_matrix,
        seed=seed,
    )

    strategies = ["average", "least_misery", "most_pleasure"]
    train_user_items = _build_train_user_items_set(
        train, dataset.user_to_idx, dataset.item_to_idx
    )

    results = {
        "design": {
            "group_construction": {
                "matrix_source": "training split only (build_rating_matrix)",
                "train_matrix_nnz": int(train_matrix.nnz),
                "n_active_users": len(active_users),
                "n_group_users": len(group_users),
                "conflicting": conflicting,
                "n_groups_requested": n_groups,
                "group_size": group_size,
                "seed": seed,
            },
            "graph_info": graph_info,
            "satisfaction_definition": (
                "mean predicted rating across group members from the "
                "recommender trained on the training split; no held-out rating "
                "is used for group construction or reported satisfaction"
            ),
            "ranking_definition": (
                "Group top-N ordering uses the recommender's raw (unclipped) "
                "member scores; the reported group score is the clipped "
                "aggregate satisfaction for the selected item, so ties at the "
                "rating ceiling no longer determine the order."
            ),
            "diffusion_design": {
                "model": "preference-aware ICM",
                "propagation_prob": p_prob,
                "base_adopt_prob": base_adopt_prob,
                "n_simulations_per_item": diff_sims,
                "max_rounds": max_rounds,
                "default_affinity": default_affinity,
                "seed_users": "the group members",
                "outcome_scope": "simulated",
                "common_random_numbers": (
                    "For a group, every candidate item's preference-aware ICM "
                    "runs reuse the same per-edge exposure and per-user adoption "
                    "thresholds for each simulation seed. Differences across "
                    "items therefore come from the affinity function, not from a "
                    "fresh random stream."
                ),
            },
        },
        "n_groups": len(groups),
        "group_size": group_size,
        "conflicting": conflicting,
        "strategies": strategies,
        "groups": {},
        "summary": {},
        "analysis": {},
    }

    strat_stats = {
        s: {
            "top_item_satisfaction_mean": [],
            "top_item_satisfaction_min": [],
            "top_item_satisfaction_max": [],
            "top_item_satisfaction_range": [],
            "group_score": [],
            "group_ranking_score": [],
            "spread_mean": [],
            "spread_std": [],
            "top_item_is_best_satisfaction": [],
            "top_item_is_best_spread": [],
        }
        for s in strategies
    }

    for g_idx, group in enumerate(groups):
        g_key = f"group_{g_idx}"
        group_sim_seeds = make_simulation_seeds(
            seed, diff_sims, offset=100_000 + g_idx * 1_000
        )
        group_randomness = {
            int(s): ICMRandomness(active_graph, int(s)) for s in group_sim_seeds
        }
        results["groups"][g_key] = {
            "members": [int(u) for u in group],
            "simulation_seeds": group_sim_seeds,
            "strategies": {},
            "candidates": {},
        }

        # Strategy top-1 items.
        strategy_items = {}
        for strat in strategies:
            recs = recommend_for_group(
                recommender=recommender,
                group=group,
                n_items=dataset.n_items,
                n=top_n,
                strategy=strat,
                train_user_items=train_user_items,
                exclude_seen_by_any=True,
            )
            if not recs:
                results["groups"][g_key]["strategies"][strat] = {
                    "top_recommendations": [],
                    "error": "no recommendation produced",
                }
                continue
            top_item, group_score = recs[0]
            raw_scores = score_group_items(
                recommender, group, dataset.n_items,
                strategy=strat, raw_scores=True,
            )
            strategy_items[strat] = {
                "top_item": int(top_item),
                "group_score": float(group_score),
                "group_ranking_score": float(raw_scores[top_item]),
                "top_recommendations": recs,
            }

        if not strategy_items:
            continue

        candidate_items = sorted({v["top_item"] for v in strategy_items.values()})
        satisfaction_by_item = {}
        spread_by_item = {}

        for item in candidate_items:
            sat = evaluate_group_satisfaction(recommender, group, item)
            satisfaction_by_item[str(item)] = {
                "member_ratings": {str(k): round(v, 4) for k, v in sat["member_ratings"].items()},
                "mean": round(sat["mean_satisfaction"], 4),
                "min": round(sat["min_satisfaction"], 4),
                "max": round(sat["max_satisfaction"], 4),
                "variance": round(sat["variance"], 4),
                "range": round(sat["range"], 4),
            }

            aff_fn = _affinity_fn_factory(
                recommender, item, dataset.user_to_idx,
                default_affinity=default_affinity, n_users=dataset.n_users,
            )
            counts = []
            exposed = []
            rounds = []
            t0 = time.perf_counter()
            for sim_seed in group_sim_seeds:
                sim = ICMSimulator(
                    active_graph, propagation_prob=p_prob,
                    max_rounds=max_rounds, seed=sim_seed,
                    common_randomness=group_randomness[int(sim_seed)],
                )
                out = sim.run_preference_aware(
                    seed_users=set(group),
                    affinity_fn=aff_fn,
                    base_adopt_prob=base_adopt_prob,
                )
                counts.append(int(out["n_adopters"]))
                exposed.append(int(out["n_exposed"]))
                rounds.append(int(out["n_rounds"]))
            elapsed = time.perf_counter() - t0
            spread_by_item[str(item)] = {
                "mean_adopters": round(float(np.mean(counts)), 3),
                "std_adopters": round(float(np.std(counts)), 3),
                "mean_exposed": round(float(np.mean(exposed)), 3),
                "mean_rounds": round(float(np.mean(rounds)), 3),
                "adopter_counts": counts,
                "runtime_seconds": round(elapsed, 4),
            }

        best_sat_value = max(
            satisfaction_by_item[str(i)]["mean"] for i in candidate_items
        )
        best_spread_value = max(
            spread_by_item[str(i)]["mean_adopters"] for i in candidate_items
        )
        # Preserve a canonical representative (lowest item ID) but treat all
        # exact ties as best when flagging strategy top items.
        best_sat_items = [
            i for i in candidate_items
            if satisfaction_by_item[str(i)]["mean"] == best_sat_value
        ]
        best_spread_items = [
            i for i in candidate_items
            if spread_by_item[str(i)]["mean_adopters"] == best_spread_value
        ]
        best_sat_item = min(best_sat_items)
        best_spread_item = min(best_spread_items)
        best_sat_set = set(best_sat_items)
        best_spread_set = set(best_spread_items)

        results["groups"][g_key]["candidates"] = {
            "items": candidate_items,
            "satisfaction": satisfaction_by_item,
            "spread": spread_by_item,
            "best_satisfaction_item": int(best_sat_item),
            "best_spread_item": int(best_spread_item),
            "best_satisfaction_items": [int(i) for i in best_sat_items],
            "best_spread_items": [int(i) for i in best_spread_items],
            "best_satisfaction_equals_best_spread": bool(
                best_sat_set == best_spread_set
            ),
            "best_satisfaction_and_spread_overlap": bool(
                best_sat_set & best_spread_set
            ),
        }

        for strat, info in strategy_items.items():
            top_item = info["top_item"]
            sat = satisfaction_by_item[str(top_item)]
            spread = spread_by_item[str(top_item)]
            is_best_sat = top_item in best_sat_set
            is_best_spread = top_item in best_spread_set
            results["groups"][g_key]["strategies"][strat] = {
                "top_recommendations": info["top_recommendations"],
                "top_item": top_item,
                "group_score": round(info["group_score"], 4),
                "group_ranking_score": round(info["group_ranking_score"], 4),
                "satisfaction": sat,
                "diffusion_spread": spread,
                "top_item_is_best_satisfaction": is_best_sat,
                "top_item_is_best_spread": is_best_spread,
            }

            strat_stats[strat]["top_item_satisfaction_mean"].append(sat["mean"])
            strat_stats[strat]["top_item_satisfaction_min"].append(sat["min"])
            strat_stats[strat]["top_item_satisfaction_max"].append(sat["max"])
            strat_stats[strat]["top_item_satisfaction_range"].append(sat["range"])
            strat_stats[strat]["group_score"].append(info["group_score"])
            strat_stats[strat]["group_ranking_score"].append(info["group_ranking_score"])
            strat_stats[strat]["spread_mean"].append(spread["mean_adopters"])
            strat_stats[strat]["spread_std"].append(spread["std_adopters"])
            strat_stats[strat]["top_item_is_best_satisfaction"].append(int(is_best_sat))
            strat_stats[strat]["top_item_is_best_spread"].append(int(is_best_spread))

    for strat in strategies:
        stats = strat_stats[strat]
        if not stats["top_item_satisfaction_mean"]:
            results["summary"][strat] = {"n_groups": 0}
            continue
        results["summary"][strat] = {
            "n_groups": len(stats["top_item_satisfaction_mean"]),
            "avg_mean_satisfaction": round(float(np.mean(stats["top_item_satisfaction_mean"])), 3),
            "avg_min_satisfaction": round(float(np.mean(stats["top_item_satisfaction_min"])), 3),
            "avg_max_satisfaction": round(float(np.mean(stats["top_item_satisfaction_max"])), 3),
            "avg_disagreement_range": round(float(np.mean(stats["top_item_satisfaction_range"])), 3),
            "avg_group_score": round(float(np.mean(stats["group_score"])), 3),
            "avg_group_ranking_score": round(float(np.mean(stats["group_ranking_score"])), 3),
            "avg_diffusion_spread": round(float(np.mean(stats["spread_mean"])), 3),
            "avg_diffusion_std": round(float(np.mean(stats["spread_std"])), 3),
            "top_item_best_satisfaction_count": int(sum(stats["top_item_is_best_satisfaction"])),
            "top_item_best_spread_count": int(sum(stats["top_item_is_best_spread"])),
        }
        logger.info(
            "Strategy '%s': avg mean satisfaction=%.2f, avg min=%.2f, "
            "disagreement=%.2f, simulated spread=%.2f",
            strat,
            results["summary"][strat]["avg_mean_satisfaction"],
            results["summary"][strat]["avg_min_satisfaction"],
            results["summary"][strat]["avg_disagreement_range"],
            results["summary"][strat]["avg_diffusion_spread"],
        )

    # Data-driven group-level analysis.
    valid_groups = [
        g for g in results["groups"].values()
        if g.get("candidates") and g["candidates"]["items"]
    ]
    n_agree = sum(
        1 for g in valid_groups
        if g["candidates"]["best_satisfaction_and_spread_overlap"]
    )
    strategy_satisfaction = {
        s: results["summary"].get(s, {}).get("avg_mean_satisfaction")
        for s in strategies
    }
    strategy_min_satisfaction = {
        s: results["summary"].get(s, {}).get("avg_min_satisfaction")
        for s in strategies
    }
    strategy_spread = {
        s: results["summary"].get(s, {}).get("avg_diffusion_spread")
        for s in strategies
    }
    best_mean_strategy = max(
        strategies,
        key=lambda s: (strategy_satisfaction[s] if strategy_satisfaction[s] is not None else -np.inf),
    )
    best_min_strategy = max(
        strategies,
        key=lambda s: (strategy_min_satisfaction[s] if strategy_min_satisfaction[s] is not None else -np.inf),
    )
    best_spread_strategy = max(
        strategies,
        key=lambda s: (strategy_spread[s] if strategy_spread[s] is not None else -np.inf),
    )

    findings = []
    if valid_groups:
        findings.append(
            f"Across {len(valid_groups)} groups, the item with the highest mean "
            f"predicted satisfaction was also the item with the highest simulated "
            f"spread in {n_agree} groups ({100.0 * n_agree / len(valid_groups):.1f}%)."
        )
    findings.append(
        f"Highest average mean satisfaction: {best_mean_strategy}; highest "
        f"average least-misery (minimum) satisfaction: {best_min_strategy}; "
        f"highest average simulated spread: {best_spread_strategy}."
    )
    if best_mean_strategy != best_spread_strategy:
        findings.append(
            "The strategy with the highest average member satisfaction did not "
            "coincide with the strategy whose top items produced the highest "
            "average simulated spread. This is the measured satisfaction/spread "
            "trade-off for these groups."
        )
    else:
        findings.append(
            "The strategy with the highest average member satisfaction also "
            "produced the highest average simulated spread in this run."
        )

    results["analysis"] = {
        "n_groups_with_candidates": len(valid_groups),
        "n_best_satisfaction_and_spread_overlap": int(n_agree),
        "agreement_rate": round(n_agree / len(valid_groups), 4) if valid_groups else None,
        "best_strategy_by_mean_satisfaction": best_mean_strategy,
        "best_strategy_by_min_satisfaction": best_min_strategy,
        "best_strategy_by_simulated_spread": best_spread_strategy,
        "strategy_mean_satisfaction": strategy_satisfaction,
        "strategy_min_satisfaction": strategy_min_satisfaction,
        "strategy_simulated_spread": strategy_spread,
        "findings": findings,
        "limitations": [
            "Satisfaction is a model prediction, not an observed group rating.",
            "Diffusion outcomes are simulated with preference-aware ICM.",
            "Groups may overlap; each group is a distinct member set but the "
            "same user can appear in more than one group.",
            "Different strategies choose different target items, so spread "
            "differences combine item effects and strategy effects; the "
            "candidate-item analysis reports the item-level detail.",
            "Best-item agreement counts a non-empty overlap between the exact "
            "tied best sets; it does not require the two best sets to be equal.",
        ],
    }

    _stage_end("R6: Group Recommendation", ts)
    _save_json(results, results_dir / "requirement_6_results.json")
    return results


# ---------------------------------------------------------------------------
# Requirement 7: Comprehensive Evaluation
# ---------------------------------------------------------------------------

_METRIC_DIRECTION = {
    "mae": "lower",
    "rmse": "lower",
    "precision_at_k": "higher",
    "recall_at_k": "higher",
    "f1_at_k": "higher",
    "hit_rate_at_k": "higher",
    "ndcg_at_k": "higher",
    "coverage": "higher",
    "diversity": "higher",
}


def _best_model_for_metric(values: dict, direction: str):
    valid = {k: v for k, v in values.items() if v is not None and not np.isnan(v)}
    if not valid:
        return None
    return (min if direction == "lower" else max)(valid, key=valid.get)


def run_requirement_7(dataset, train, val, test, models_dict: dict, config: dict, results_dir: Path):
    """Run Comprehensive Evaluation on held-out test split (Part 7).

    Observed evaluation
    -------------------
    MAE, RMSE, Precision@K, Recall@K, F1@K, HitRate@K, nDCG@K, catalogue
    coverage, and feature-based intra-list diversity on held-out test ratings.

    Simulated interaction
    ---------------------
    A preference-aware ICM cascade for the documented training-only target
    item, with identical seed sets and Monte Carlo seeds for every model.
    These outcomes are explicitly separated from observed metrics.

    Definitions
    -----------
    * Relevant item: held-out test rating >= ``relevance_threshold``.
    * Candidate items: all catalogue items except the user's training items;
      no negative-item sampling is used.
    * Precision uses K as the denominator; recall uses the number of relevant
      held-out items; F1 is the mean of per-user F1 scores.
    * Coverage is the fraction of the catalogue recommended at least once in
      the evaluated lists (sample scope is recorded).
    * Diversity is the mean pairwise 1 - cosine distance between item feature
      vectors built from training interactions.
    """
    logger.info("=" * 60)
    logger.info("REQUIREMENT 7: Comprehensive Evaluation")
    logger.info("=" * 60)
    ts = _stage_start("R7: Comprehensive Evaluation")

    k = int(config.get("top_k", 10))
    thresh = int(config.get("relevance_threshold", 4))
    sample_users = config.get("ranking_sample_users", 500)
    sample_seed = int(config.get("ranking_sample_seed", 42))
    diversity_sample = config.get("diversity_sample_users", 500)
    eval_cfg = config.get("evaluation", {})

    train_matrix = build_rating_matrix(
        dataset.n_users, dataset.n_items, train,
        dataset.user_to_idx, dataset.item_to_idx,
    )
    train_user_items = _build_train_user_items_set(
        train, dataset.user_to_idx, dataset.item_to_idx
    )
    # One row per item, one column per user; training interactions only.
    item_features = train_matrix.T.tocsr()

    results = {
        "definitions": {
            "split": "train / validation / test; test is used only for final observed evaluation",
            "relevance_threshold": thresh,
            "relevance_definition": (
                f"held-out test rating >= {thresh}; unobserved items are not "
                "treated as negative examples"
            ),
            "candidate_definition": (
                "all catalogue items except the user's training items; no "
                "negative-item sampling"
            ),
            "cutoff_k": k,
            "f1_definition": "mean of per-user F1 scores",
            "coverage_definition": (
                "unique recommended items / catalogue size, over the evaluated "
                "user sample; sampling scope is reported per model"
            ),
            "diversity_definition": (
                "mean pairwise 1-cosine(item training-interaction feature "
                "vectors) within each recommendation list; the list scope and "
                "sampling stage are reported per model"
            ),
            "accuracy_note": (
                "No accuracy classification metric is reported: relevance is "
                "defined only on held-out ratings, and unobserved items are not "
                "negatives."
            ),
        },
        "observed": {
            "scope": "held-out test ratings and recommendations derived from training data",
            "models": {},
            "comparison": {},
        },
        "simulated": {
            "scope": "preference-aware ICM outcomes; not observed adoptions",
            "models": {},
            "design": {},
        },
        "tradeoff_analysis": {},
    }

    for name, model in models_dict.items():
        if model is None:
            continue
        logger.info("Comprehensive evaluation for %s...", name)
        pred_res = evaluate_predictions(
            model, test, dataset.user_to_idx, dataset.item_to_idx
        )
        rank_res = evaluate_ranking(
            model, test, train_user_items,
            dataset.user_to_idx, dataset.item_to_idx,
            k=k, relevance_threshold=thresh, label=f"R7-{name}",
            sample_users=sample_users, sample_seed=sample_seed,
            item_features=item_features,
            max_diversity_users=diversity_sample,
            diversity_seed=sample_seed,
        )

        results["observed"]["models"][name] = {
            "prediction_error": {
                "mae": round(pred_res["mae"], 4) if not np.isnan(pred_res["mae"]) else None,
                "rmse": round(pred_res["rmse"], 4) if not np.isnan(pred_res["rmse"]) else None,
                "n_evaluated": pred_res["n_evaluated"],
                "n_skipped": pred_res["n_skipped"],
            },
            "ranking": {
                "precision_at_k": round(rank_res["precision_at_k"], 5),
                "recall_at_k": round(rank_res["recall_at_k"], 5),
                "f1_at_k": round(rank_res["f1_at_k"], 5),
                "hit_rate_at_k": round(rank_res["hit_rate_at_k"], 5),
                "ndcg_at_k": round(rank_res["ndcg_at_k"], 5),
                "coverage": round(rank_res["coverage"], 5),
                "n_unique_items": rank_res["n_unique_items"],
                "diversity": (
                    round(rank_res["diversity"], 5)
                    if rank_res["diversity"] is not None else None
                ),
            },
            "evaluation_metadata": {
                "n_users_evaluated": rank_res["n_users_evaluated"],
                "total_eligible_users": rank_res["total_eligible_users"],
                "is_sampled": rank_res["is_sampled"],
                "sample_seed": rank_res["sample_seed"],
                "catalog_size": rank_res["catalog_size"],
                "diversity_n_lists": rank_res["diversity_n_lists"],
                "diversity_is_sampled": rank_res["diversity_is_sampled"],
                "diversity_sample_seed": rank_res["diversity_sample_seed"],
                "diversity_sampling_scope": rank_res["diversity_sampling_scope"],
                "eval_time_seconds": rank_res["eval_time_seconds"],
            },
        }

    # Observed metric comparison, derived from the values just measured.
    observed_values = {
        "mae": {m: v["prediction_error"]["mae"] for m, v in results["observed"]["models"].items()},
        "rmse": {m: v["prediction_error"]["rmse"] for m, v in results["observed"]["models"].items()},
        **{
            metric: {m: v["ranking"][metric] for m, v in results["observed"]["models"].items()}
            for metric in ("precision_at_k", "recall_at_k", "f1_at_k",
                           "hit_rate_at_k", "ndcg_at_k", "coverage", "diversity")
        },
    }
    for metric, values in observed_values.items():
        best = _best_model_for_metric(values, _METRIC_DIRECTION[metric])
        results["observed"]["comparison"][metric] = {
            "direction": _METRIC_DIRECTION[metric],
            "values": values,
            "best_model": best,
        }

    # --- Simulated interaction on the documented target item ---
    run_interaction = bool(eval_cfg.get("run_diffusion_interaction", True))
    if run_interaction:
        interaction_sims = int(eval_cfg.get("interaction_simulations", 3))
        seed_size = int(eval_cfg.get("interaction_seed_size", config.get("diffusion", {}).get("seed_size", 5)))
        base_seed = int(eval_cfg.get("interaction_seed", config.get("split_seed", 42)))
        p_prob = float(eval_cfg.get("interaction_propagation_prob", 0.05))
        base_adopt_prob = float(eval_cfg.get("interaction_base_adopt_prob", 0.5))
        max_rounds = int(eval_cfg.get("interaction_max_rounds", 25))
        default_affinity = float(eval_cfg.get("interaction_default_affinity", 0.3))
        target_min = int(eval_cfg.get("target_min_train_ratings", 6))
        target_max = int(eval_cfg.get("target_max_train_ratings", 20))

        active_graph, active_users, graph_info = _resolve_diffusion_graph(dataset, config)
        target = select_target_item(
            train, dataset.item_to_idx,
            min_train_ratings=target_min, max_train_ratings=target_max,
        )
        eligible, _ = _target_eligible_users(
            dataset, train, int(target["original_id"]), active_users
        )
        seed_sets = generate_seed_sets(
            eligible, seed_size=seed_size,
            n_simulations=interaction_sims, seed=base_seed,
        )
        sim_seeds = make_simulation_seeds(base_seed, interaction_sims, offset=700_000)
        interaction_randomness = {
            int(s): ICMRandomness(active_graph, int(s)) for s in sim_seeds
        }

        results["simulated"]["design"] = {
            "model": "preference-aware ICM",
            "target_item": target,
            "target_selection_uses": "training split only",
            "graph_info": graph_info,
            "n_eligible_users": len(eligible),
            "seed_size": seed_size,
            "n_simulations": interaction_sims,
            "seed_sets": seed_sets,
            "simulation_seeds": sim_seeds,
            "same_seed_sets_and_seeds_for_all_models": True,
            "common_random_numbers": (
                "Every model reuses one per-edge exposure and per-user adoption "
                "threshold set per trial, so spread differences come from the "
                "model's affinity function rather than fresh random streams."
            ),
            "propagation_prob": p_prob,
            "base_adopt_prob": base_adopt_prob,
            "max_rounds": max_rounds,
            "default_affinity": default_affinity,
            "satisfaction_proxy": (
                "predicted rating 1 + 4 * affinity for adopters; a simulator-side "
                "proxy, not an observed rating"
            ),
        }

        for name, model in models_dict.items():
            if model is None:
                continue
            aff_fn = _affinity_fn_factory(
                model, int(target["idx"]), dataset.user_to_idx,
                default_affinity=default_affinity, n_users=dataset.n_users,
            )
            counts, exposed_counts, rounds, satisfactions = [], [], [], []
            t0 = time.perf_counter()
            for sim_i, sim_seed in enumerate(sim_seeds):
                sim = ICMSimulator(
                    active_graph, propagation_prob=p_prob,
                    max_rounds=max_rounds, seed=sim_seed,
                    common_randomness=interaction_randomness[int(sim_seed)],
                )
                out = sim.run_preference_aware(
                    seed_users=set(seed_sets[sim_i]),
                    affinity_fn=aff_fn,
                    base_adopt_prob=base_adopt_prob,
                )
                adopters = out["adopters"]
                counts.append(int(out["n_adopters"]))
                exposed_counts.append(int(out["n_exposed"]))
                rounds.append(int(out["n_rounds"]))
                satisfactions.extend(
                    1.0 + 4.0 * float(aff_fn(u)) for u in adopters
                )
            elapsed = time.perf_counter() - t0
            results["simulated"]["models"][name] = {
                "adopters": _distribution(counts),
                "exposed": _distribution(exposed_counts),
                "rounds": _distribution(rounds),
                "adopter_satisfaction_proxy": _distribution(satisfactions),
                "runtime_seconds": round(elapsed, 4),
            }
            logger.info(
                "Simulated interaction %s: mean adopters=%.1f, exposed=%.1f",
                name, float(np.mean(counts)), float(np.mean(exposed_counts)),
            )

        if results["simulated"]["models"]:
            spread_vals = {
                m: v["adopters"]["mean"]
                for m, v in results["simulated"]["models"].items()
            }
            best_spread = max(spread_vals, key=spread_vals.get)
            results["tradeoff_analysis"]["simulated_spread_ranking"] = spread_vals
            results["tradeoff_analysis"]["best_simulated_spread_model"] = best_spread

    # Observed quality vs simulated spread comparison.
    tradeoff = results["tradeoff_analysis"]
    ndcg_values = observed_values["ndcg_at_k"]
    best_ndcg = _best_model_for_metric(ndcg_values, "higher")
    tradeoff["best_observed_ndcg_model"] = best_ndcg
    tradeoff["best_observed_rmse_model"] = _best_model_for_metric(
        observed_values["rmse"], "lower"
    )
    spread_vals = tradeoff.get("simulated_spread_ranking", {})
    if best_ndcg is not None and spread_vals:
        best_spread = tradeoff.get("best_simulated_spread_model")
        tradeoff["observed_ndcg_and_simulated_spread_agree"] = bool(best_ndcg == best_spread)
        tradeoff["finding"] = (
            f"The best observed nDCG@{k} model was {best_ndcg}; the largest "
            f"simulated preference-aware spread was produced by {best_spread}. "
            + (
                "The two rankings coincide in this run."
                if best_ndcg == best_spread else
                "The two rankings differ, so higher observed ranking quality did "
                "not automatically imply a larger simulated cascade in this run."
            )
        )
    else:
        tradeoff["finding"] = (
            "Observed quality metrics are reported separately; no simulated "
            "interaction was requested."
        )

    results["tradeoff_analysis"]["limitations"] = [
        "Observed and simulated outcomes answer different questions and are not "
        "interchangeable.",
        "The simulated cascade uses one target item chosen from a training "
        "popularity band; it is not a forecast of platform-wide adoption.",
        "Ranking metrics are computed on a fixed seeded user sample when "
        "ranking_sample_users is set.",
        "Unobserved items are not treated as negative examples.",
        "Coverage and diversity depend on the evaluated list sample and the "
        "training-derived item features.",
    ]

    _stage_end("R7: Comprehensive Evaluation", ts)
    _save_json(results, results_dir / "requirement_7_results.json")
    return results


# ---------------------------------------------------------------------------
# Requirement 8: Intervention & Computational Cost
# ---------------------------------------------------------------------------

def run_requirement_8(dataset, train, recommender, config: dict, results_dir: Path):
    """Run Intervention & Computational Cost experiments (Part 8).

    Contents
    --------
    1. Relevance-threshold intervention: measured spread and adopter
       satisfaction at several thresholds, using common seed sets and common
       Monte Carlo seeds.
    2. CELF versus standard greedy: measured wall-clock time, evaluation
       counts, memory, selected seeds, and a deterministic controlled check.
    3. Cost scaling: BFS-induced network sizes, item-catalogue sizes, and
       promotion budgets, with wall-clock and memory measurements.
    4. Measured stage-level bottleneck analysis.
    """
    logger.info("=" * 60)
    logger.info("REQUIREMENT 8: Intervention & Cost Analysis")
    logger.info("=" * 60)
    ts = _stage_start("R8: Intervention & Cost Analysis")

    active_graph, active_users, graph_info = _resolve_diffusion_graph(dataset, config)
    int_cfg = config.get("intervention", {})
    diff_cfg = config.get("diffusion", {})
    train_matrix = build_rating_matrix(
        dataset.n_users, dataset.n_items, train,
        dataset.user_to_idx, dataset.item_to_idx,
    )
    k = int(config.get("top_k", 10))

    thresholds = sorted({
        float(t) for t in int_cfg.get("thresholds", [0.0, 0.3, 0.5, 0.7])
    })
    if 0.0 not in thresholds:
        thresholds = [0.0] + thresholds
    p_prob = float(int_cfg.get("propagation_prob", 0.05))
    base_adopt_prob = float(int_cfg.get("base_adopt_prob", 0.5))
    n_sims = int(int_cfg.get("n_simulations", 5))
    seed_size = int(int_cfg.get("seed_size", 5))
    seed = int(int_cfg.get("seed", 42))
    max_rounds = int(int_cfg.get("max_rounds", 25))
    default_affinity = float(int_cfg.get("default_affinity", 0.3))
    target_min = int(int_cfg.get("target_min_train_ratings", 6))
    target_max = int(int_cfg.get("target_max_train_ratings", 20))

    target = select_target_item(
        train, dataset.item_to_idx,
        min_train_ratings=target_min, max_train_ratings=target_max,
    )
    eligible, _ = _target_eligible_users(
        dataset, train, int(target["original_id"]), active_users
    )
    seed_sets = generate_seed_sets(
        eligible, seed_size=seed_size, n_simulations=n_sims, seed=seed
    )
    sim_seeds = make_simulation_seeds(seed, n_sims, offset=900_000)
    # Per-edge and per-user thresholds reused across every intervention
    # threshold, so thresholds are compared on the same random outcomes.
    intervention_randomness = {
        int(s): ICMRandomness(active_graph, int(s)) for s in sim_seeds
    }
    aff_fn = _affinity_fn_factory(
        recommender, int(target["idx"]), dataset.user_to_idx,
        default_affinity=default_affinity, n_users=dataset.n_users,
    )

    # --- 1. Relevance-threshold intervention ---
    logger.info("--- Intervention: relevance threshold ---")
    intervention_results = {}
    baseline_exposed = set()
    for th in thresholds:
        run_rows = []
        counts, exposed_counts, rounds = [], [], []
        all_sat, nonseed_sat = [], []
        t0 = time.perf_counter()
        for sim_i, sim_seed in enumerate(sim_seeds):
            seed_set = set(seed_sets[sim_i])
            sim = ICMSimulator(
                active_graph, propagation_prob=p_prob,
                max_rounds=max_rounds, seed=sim_seed,
                common_randomness=intervention_randomness[int(sim_seed)],
            )
            out = sim.run_preference_aware(
                seed_users=seed_set,
                affinity_fn=aff_fn,
                base_adopt_prob=base_adopt_prob,
                min_affinity_threshold=th,
            )
            adopters = set(out["adopters"])
            if th == 0.0:
                # Record the baseline exposure population once so the report
                # can explain why a mild threshold may be inactive.
                baseline_exposed.update(out["exposed"])
            nonseed = adopters - seed_set
            adopter_affinities = [float(aff_fn(u)) for u in adopters]
            nonseed_affinities = [float(aff_fn(u)) for u in nonseed]
            adopter_ratings = [1.0 + 4.0 * a for a in adopter_affinities]
            nonseed_ratings = [1.0 + 4.0 * a for a in nonseed_affinities]
            counts.append(int(out["n_adopters"]))
            exposed_counts.append(int(out["n_exposed"]))
            rounds.append(int(out["n_rounds"]))
            all_sat.extend(adopter_ratings)
            nonseed_sat.extend(nonseed_ratings)
            run_rows.append({
                "simulation_index": sim_i,
                "sim_seed": sim_seed,
                "n_adopters": int(out["n_adopters"]),
                "n_exposed": int(out["n_exposed"]),
                "n_nonseed_adopters": len(nonseed),
                "n_rounds": int(out["n_rounds"]),
                "mean_adopter_rating": round(float(np.mean(adopter_ratings)), 4) if adopter_ratings else None,
                "mean_nonseed_adopter_rating": round(float(np.mean(nonseed_ratings)), 4) if nonseed_ratings else None,
            })
        elapsed = time.perf_counter() - t0

        count_dist = _distribution(counts)
        exposed_dist = _distribution(exposed_counts)
        round_dist = _distribution(rounds)
        sat_dist = _distribution(all_sat)
        nonseed_sat_dist = _distribution(nonseed_sat)
        mean_ad = count_dist["mean"] or 0.0
        mean_exp = exposed_dist["mean"] or 0.0
        results_key = f"threshold_{th:g}"
        intervention_results[results_key] = {
            "threshold": th,
            "adopters": count_dist,
            "exposed": exposed_dist,
            "rounds": round_dist,
            "mean_adopters": count_dist["mean"],
            "mean_exposed": exposed_dist["mean"],
            "conversion_efficiency": round(mean_ad / mean_exp, 4) if mean_exp > 0 else None,
            "mean_adopter_rating_proxy": sat_dist["mean"],
            "mean_nonseed_adopter_rating_proxy": nonseed_sat_dist["mean"],
            "adopter_rating_proxy_distribution": sat_dist,
            "share_adopters_rating_ge_4": (
                round(float(np.mean(np.asarray(all_sat) >= 4.0)), 4)
                if all_sat else None
            ),
            "runs": run_rows,
            "runtime_seconds": round(elapsed, 4),
        }
        logger.info(
            "Intervention threshold %.2f: adopters=%.1f, exposed=%.1f, "
            "conversion=%.3f, mean adopter rating=%.3f",
            th, mean_ad, mean_exp,
            intervention_results[results_key]["conversion_efficiency"] or 0.0,
            sat_dist["mean"] or 0.0,
        )

    baseline_key = "threshold_0"
    baseline_row = intervention_results.get(baseline_key)
    if baseline_row is not None:
        base_ad = baseline_row["mean_adopters"] or 0.0
        base_sat = baseline_row["mean_adopter_rating_proxy"]
        for key, row in intervention_results.items():
            if key == baseline_key:
                continue
            row["relative_to_baseline"] = {
                "spread_change": round((row["mean_adopters"] or 0.0) - base_ad, 3),
                "spread_change_pct": (
                    round(100.0 * ((row["mean_adopters"] or 0.0) - base_ad) / base_ad, 2)
                    if base_ad else None
                ),
                "adopter_satisfaction_change": (
                    round(row["mean_adopter_rating_proxy"] - base_sat, 4)
                    if row["mean_adopter_rating_proxy"] is not None and base_sat is not None
                    else None
                ),
                "exposed_change": round((row["mean_exposed"] or 0.0) - (baseline_row["mean_exposed"] or 0.0), 3),
            }

    baseline_affinity = None
    if baseline_exposed:
        values = np.asarray([float(aff_fn(u)) for u in baseline_exposed], dtype=float)
        baseline_affinity = {
            "n_exposed": int(values.size),
            "mean_affinity": round(float(np.mean(values)), 4),
            "min_affinity": round(float(np.min(values)), 4),
            "max_affinity": round(float(np.max(values)), 4),
            "share_below_threshold": {
                f"{th:.2f}": round(float(np.mean(values < th)), 4)
                for th in thresholds if th > 0.0
            },
            "note": (
                "Affinity distribution of the users exposed in the no-threshold "
                "baseline; a threshold above the maximum cannot change that "
                "baseline cascade."
            ),
        }

    intervention_findings = []
    if baseline_row is not None:
        effects = []
        for key, row in intervention_results.items():
            if key == baseline_key:
                continue
            rel = row.get("relative_to_baseline", {})
            spread_pct = rel.get("spread_change_pct")
            sat_change = rel.get("adopter_satisfaction_change")
            effects.append((row["threshold"], spread_pct, sat_change))
            intervention_findings.append(
                f"Threshold {row['threshold']:.2f}: spread changed by "
                f"{spread_pct}% and mean adopter rating proxy changed by "
                f"{sat_change} relative to the no-threshold baseline."
            )

        reductions = [
            e for e in effects if e[1] is not None and e[1] < -0.05
        ]
        if reductions:
            # Prefer a threshold that reduces spread without also lowering the
            # simulator-side satisfaction proxy; if none exists, report the
            # largest reduction and its measured proxy change.
            non_decreasing = [
                e for e in reductions if e[2] is not None and e[2] >= 0.0
            ]
            candidates = non_decreasing or reductions
            best = max(candidates, key=lambda e: -e[1])
            qualifier = (
                "among thresholds with a non-negative proxy change"
                if non_decreasing else "overall"
            )
            intervention_findings.append(
                f"The largest measured spread reduction {qualifier} was at "
                f"threshold {best[0]:.2f}: {best[1]:.2f}% versus baseline, "
                f"with an adopter-rating-proxy change of {best[2]}. This is the "
                "measured spread/satisfaction trade-off; the proxy is "
                "simulator-side, not an observed satisfaction rating."
            )
        else:
            intervention_findings.append(
                "No tested threshold reduced mean simulated spread relative to "
                "the no-threshold baseline in this run. Shared Monte Carlo seeds "
                "do not remove path divergence when filtering changes which "
                "users are exposed and which cascades form, so this run does "
                "not establish a spread-reduction benefit."
            )
        intervention_findings.append(
            "A threshold can reduce low-affinity adoption, but the measured "
            "table, not an assumed monotone effect, is the evidence. The "
            "rating proxy is simulator-side and is not observed satisfaction."
        )

    # --- 2. CELF versus standard greedy ---
    logger.info("--- CELF versus standard greedy ---")
    celf_cfg = int_cfg.get("celf", {})
    celf_budget = int(celf_cfg.get("budget", 3))
    celf_pool_size = int(celf_cfg.get("candidate_pool_size", 15))
    celf_sims = int(celf_cfg.get("n_simulations", n_sims))
    celf_pool = _top_degree_candidates(
        active_graph, sorted(active_users), celf_pool_size
    )
    celf_seeds = make_simulation_seeds(seed, celf_sims, offset=800_000)

    def _run_greedy(use_celf: bool):
        return greedy_influence_maximization(
            active_graph, celf_pool,
            budget=celf_budget,
            propagation_prob=p_prob,
            n_simulations=celf_sims,
            candidate_pool=celf_pool,
            selection_seeds=celf_seeds,
            seed=seed,
            use_celf=use_celf,
        )

    greedy_std, t_std, mem_std = _measure_call(lambda: _run_greedy(False))
    greedy_celf, t_celf, mem_celf = _measure_call(lambda: _run_greedy(True))
    speedup = t_std / t_celf if t_celf > 0 else None
    eval_reduction = (
        100.0 * (greedy_std["n_evaluations"] - greedy_celf["n_evaluations"])
        / greedy_std["n_evaluations"]
        if greedy_std["n_evaluations"] else None
    )
    identical_seeds = greedy_std["seeds"] == greedy_celf["seeds"]
    celf_comparison = {
        "test_budget": celf_budget,
        "candidate_pool_size": len(celf_pool),
        "n_simulations": celf_sims,
        "selection_seeds": celf_seeds,
        "standard_greedy": {
            "n_evaluations": greedy_std["n_evaluations"],
            "runtime_seconds": round(t_std, 4),
            "memory": mem_std,
            "seeds": greedy_std["seeds"],
            "marginal_gains": greedy_std["marginal_gains"],
        },
        "celf_greedy": {
            "n_evaluations": greedy_celf["n_evaluations"],
            "runtime_seconds": round(t_celf, 4),
            "memory": mem_celf,
            "seeds": greedy_celf["seeds"],
            "marginal_gains": greedy_celf["marginal_gains"],
        },
        "speedup_factor": round(speedup, 3) if speedup is not None else None,
        "evaluation_reduction_pct": round(eval_reduction, 2) if eval_reduction is not None else None,
        "identical_seeds": bool(identical_seeds),
        "seed_jaccard": round(_jaccard(greedy_std["seeds"], greedy_celf["seeds"]), 3),
        "note": (
            "With fixed common random outcomes the finite-trial ICM objective is "
            "deterministic, monotone, and submodular, so the CELF lazy bound is "
            "valid for this measured objective: CELF should select the same seed "
            "set with no more evaluations than standard greedy. The measurement "
            "checks that claim rather than assuming it."
        ),
        "memory_method": (
            "tracemalloc Python allocations plus RSS delta when psutil is "
            "available; NumPy native allocations may be under-represented"
        ),
    }
    if speedup is None:
        celf_finding = "CELF runtime could not be compared."
    elif speedup > 1.0:
        celf_finding = (
            f"CELF used {greedy_celf['n_evaluations']} evaluations versus "
            f"{greedy_std['n_evaluations']} for standard greedy and ran "
            f"{speedup:.2f}x faster in this measured setting."
        )
    else:
        celf_finding = (
            f"CELF did not produce a wall-clock speedup in this measured "
            f"setting ({t_celf:.3f}s vs {t_std:.3f}s; "
            f"{greedy_celf['n_evaluations']} vs {greedy_std['n_evaluations']} "
            "evaluations). Lazy re-evaluation overhead can outweigh the saved "
            "evaluations at this pool size even when the objective is "
            "deterministic."
        )
    if not identical_seeds:
        celf_finding += (
            " The selected seed sets still differed even though the measured "
            "objective is deterministic under common random outcomes; this "
            "flags a tie-breaking or implementation discrepancy."
        )

    # Controlled deterministic check on a small induced graph.
    controlled_check = None
    if bool(celf_cfg.get("controlled_check", True)):
        reverse_graph = _reverse_graph(active_graph)
        start = max(
            active_users,
            key=lambda u: (len(active_graph.get(u, [])), -int(u)),
        )
        small_graph, small_nodes = _bfs_induced_subgraph(
            active_graph, reverse_graph, min(120, len(active_users)), start
        )
        small_pool = _top_degree_candidates(small_graph, small_nodes, 20)
        small_seeds = make_simulation_seeds(seed, 1, offset=810_000)
        kwargs = dict(
            propagation_graph=small_graph,
            eligible_users=small_pool,
            budget=min(2, len(small_pool)),
            propagation_prob=1.0,
            n_simulations=1,
            candidate_pool=small_pool,
            selection_seeds=small_seeds,
            seed=seed,
        )
        small_std = greedy_influence_maximization(use_celf=False, **kwargs)
        small_celf = greedy_influence_maximization(use_celf=True, **kwargs)
        controlled_check = {
            "graph_nodes": len(small_nodes),
            "candidate_pool_size": len(small_pool),
            "budget": kwargs["budget"],
            "propagation_prob": 1.0,
            "n_simulations": 1,
            "standard_n_evaluations": small_std["n_evaluations"],
            "celf_n_evaluations": small_celf["n_evaluations"],
            "identical_seeds": small_std["seeds"] == small_celf["seeds"],
            "standard_seeds": small_std["seeds"],
            "celf_seeds": small_celf["seeds"],
            "note": (
                "Deterministic ICM (p=1.0, one Monte Carlo trial) gives a "
                "monotone submodular objective, so CELF should match standard "
                "greedy with no more evaluations."
            ),
        }

    # --- 3. Cost scaling ---
    logger.info("--- Cost scaling ---")
    scaling_cfg = config.get("scaling", {})
    network_fractions = [float(f) for f in scaling_cfg.get(
        "network_size_fractions", [0.15, 0.35, 0.60]
    )]
    item_fractions = [float(f) for f in scaling_cfg.get(
        "item_size_fractions", [0.1, 0.25, 0.5, 1.0]
    )]
    budget_values = sorted({int(b) for b in scaling_cfg.get(
        "budgets", [1, 2, 3, 5]
    ) if int(b) > 0})
    scaling_sims = int(scaling_cfg.get("n_simulations", 2))
    scaling_candidate_pool = int(scaling_cfg.get("candidate_pool_size", 20))
    scaling_budget = int(scaling_cfg.get("network_budget", 2))
    scaling_seeds = make_simulation_seeds(seed, scaling_sims, offset=820_000)
    reverse_graph = _reverse_graph(active_graph)
    start = max(
        active_users,
        key=lambda u: (len(active_graph.get(u, [])), -int(u)),
    )

    network_scaling = []
    for frac in network_fractions:
        target_n = max(2, int(round(len(active_users) * frac)))
        sub_graph, sub_nodes = _bfs_induced_subgraph(
            active_graph, reverse_graph, target_n, start
        )
        sub_pool = _top_degree_candidates(
            sub_graph, sub_nodes, min(scaling_candidate_pool, len(sub_nodes))
        )
        res, seconds, memory = _measure_call(lambda: greedy_influence_maximization(
            propagation_graph=sub_graph,
            eligible_users=sub_pool,
            budget=min(scaling_budget, len(sub_pool)),
            propagation_prob=p_prob,
            n_simulations=scaling_sims,
            candidate_pool=sub_pool,
            selection_seeds=scaling_seeds,
            seed=seed,
            use_celf=False,
        ))
        network_scaling.append({
            "target_fraction": frac,
            "requested_nodes": target_n,
            "n_nodes": len(sub_nodes),
            "n_edges": int(sum(len(v) for v in sub_graph.values())),
            "n_candidates": len(sub_pool),
            "budget": min(scaling_budget, len(sub_pool)),
            "runtime_seconds": round(seconds, 4),
            "memory": memory,
            "n_evaluations": res["n_evaluations"],
            "seeds": res["seeds"],
        })
        logger.info(
            "Network scaling %.0f%%: %d nodes, %d edges, %.2fs",
            frac * 100, len(sub_nodes), sum(len(v) for v in sub_graph.values()), seconds,
        )

    # Item-catalogue scaling for the trained MF scorer.
    item_scaling = []
    item_scaling_note = None
    if hasattr(recommender, "item_factors") and recommender.item_factors is not None:
        users_with_train = [
            int(u) for u in np.flatnonzero(np.diff(train_matrix.indptr) > 0)
            if int(u) in active_users and int(u) < recommender.n_users
        ]
        sample_n = int(scaling_cfg.get("item_user_sample", 30))
        if len(users_with_train) > sample_n:
            rng = np.random.RandomState(seed)
            users_with_train = sorted(
                int(u) for u in rng.choice(users_with_train, size=sample_n, replace=False)
            )
        from sntp.recommenders._ranking import top_n_from_scores

        for frac in item_fractions:
            n_sub = max(1, int(round(recommender.n_items * frac)))
            sub_factors = recommender.item_factors[:n_sub]
            sub_bias = recommender.item_bias[:n_sub]

            def _score_subset():
                weight = float(getattr(
                    recommender, "ranking_item_bias_weight", 1.0
                ))
                for u in users_with_train:
                    scores = (
                        recommender.global_mean
                        + recommender.user_bias[u]
                        + sub_bias
                        + sub_factors.dot(recommender.user_factors[u])
                    )
                    # Rank on the same bias-adjusted raw score family used by
                    # the recommender's own top-N path; clipping before ranking
                    # would collapse distinct items into ties.
                    scores = scores - (1.0 - weight) * sub_bias
                    top_n_from_scores(
                        scores, k,
                        recommender._train_user_items.get(u, set()),
                    )

            _, seconds, memory = _measure_call(_score_subset)
            n_users_measured = max(len(users_with_train), 1)
            item_scaling.append({
                "fraction": frac,
                "n_items": n_sub,
                "n_users_measured": len(users_with_train),
                "runtime_seconds": round(seconds, 4),
                "ms_per_user": round(1000.0 * seconds / n_users_measured, 4),
                "memory": memory,
                "item_factor_bytes": int(n_sub * recommender.n_factors * 8),
            })
            logger.info(
                "Item scaling %.0f%%: %d items, %.3f ms/user",
                frac * 100, n_sub, 1000.0 * seconds / n_users_measured,
            )
        item_scaling_note = (
            "Isolates MF scoring/ranking over item prefixes; it does not include "
            "model training, which scales with observed ratings rather than the "
            "whole catalogue."
        )
    else:
        item_scaling_note = "Recommender has no item factors; item scaling was skipped."

    # Promotion-budget scaling on a bounded induced graph.
    budget_fraction = float(scaling_cfg.get("budget_network_fraction", 0.25))
    budget_nodes = max(2, int(round(len(active_users) * budget_fraction)))
    budget_graph, budget_nodes_list = _bfs_induced_subgraph(
        active_graph, reverse_graph, budget_nodes, start
    )
    budget_pool = _top_degree_candidates(
        budget_graph, budget_nodes_list,
        min(scaling_candidate_pool, len(budget_nodes_list)),
    )
    budget_scaling = []
    for b in budget_values:
        if b > len(budget_pool):
            continue
        res, seconds, memory = _measure_call(lambda: greedy_influence_maximization(
            propagation_graph=budget_graph,
            eligible_users=budget_pool,
            budget=b,
            propagation_prob=p_prob,
            n_simulations=scaling_sims,
            candidate_pool=budget_pool,
            selection_seeds=scaling_seeds,
            seed=seed,
            use_celf=False,
        ))
        budget_scaling.append({
            "budget": b,
            "runtime_seconds": round(seconds, 4),
            "memory": memory,
            "n_evaluations": res["n_evaluations"],
            "n_seeds": len(res["seeds"]),
        })
        logger.info("Budget scaling budget=%d: %.3fs, %d evaluations",
                    b, seconds, res["n_evaluations"])

    # Stage-level bottleneck snapshot.  R8 itself is still running, so its
    # timing is taken from the measured sub-experiments below.
    stage_rows = [dict(s) for s in _STAGE_TIMINGS]
    measured_r8 = {
        "intervention": round(sum(
            r["runtime_seconds"] for r in intervention_results.values()
        ), 4),
        "celf": round(t_std + t_celf, 4),
        "network_scaling": round(sum(r["runtime_seconds"] for r in network_scaling), 4),
        "item_scaling": round(sum(r["runtime_seconds"] for r in item_scaling), 4),
        "budget_scaling": round(sum(r["runtime_seconds"] for r in budget_scaling), 4),
    }
    stage_total = sum(float(s["seconds"]) for s in stage_rows)
    top_stages = sorted(stage_rows, key=lambda s: s["seconds"], reverse=True)[:5]
    for row in top_stages:
        row["share_of_completed_stages_pct"] = (
            round(100.0 * row["seconds"] / stage_total, 2) if stage_total > 0 else None
        )

    results = {
        "design": {
            "target_item": target,
            "target_selection_uses": "training split only",
            "graph_info": graph_info,
            "n_eligible_users": len(eligible),
            "seed_size": seed_size,
            "seed_sets": seed_sets,
            "simulation_seeds": sim_seeds,
            "same_seed_sets_and_seeds_across_thresholds": True,
            "common_random_numbers": (
                "Each trial has one per-edge exposure threshold and one per-user "
                "adoption threshold; every intervention threshold reuses them."
            ),
            "outcome_scope": "simulated exposures/adoptions and simulator-side rating proxy",
        },
        "intervention": {
            "rule": (
                "Exposed users whose predicted affinity for the target item is "
                "below the threshold neither adopt nor propagate the item."
            ),
            "baseline_threshold": 0.0,
            "thresholds": thresholds,
            "baseline_exposed_affinity": baseline_affinity,
            "results": intervention_results,
            "findings": intervention_findings,
            "limitations": [
                "The adopter rating proxy (1 + 4 * affinity) is predicted by the "
                "recommender, not observed after the intervention.",
                "The intervention is applied inside the simulator; no online "
                "randomized experiment was run.",
                "Shared per-edge exposure and per-user adoption thresholds mean "
                "threshold changes are compared on the same random outcomes; "
                "filtering still changes which paths form, so downstream "
                "divergence is a real cascade effect, not fresh noise.",
            ],
        },
        "celf_optimization": {
            "comparison": celf_comparison,
            "controlled_check": controlled_check,
            "finding": celf_finding,
        },
        "scaling": {
            "network": {
                "measurement": (
                    "BFS-induced connected-ish subgraph grown from the "
                    "highest-out-degree active user; induced propagation edges; "
                    "greedy selection timed from scratch at each size"
                ),
                "rows": network_scaling,
            },
            "item_catalogue": {
                "measurement": item_scaling_note,
                "rows": item_scaling,
            },
            "promotion_budget": {
                "measurement": (
                    f"greedy selection from scratch on a BFS-induced subgraph "
                    f"with {len(budget_nodes_list)} nodes and "
                    f"{len(budget_pool)} candidates"
                ),
                "rows": budget_scaling,
            },
            "memory_method": (
                "tracemalloc Python-level allocations plus RSS delta where "
                "psutil is available; NumPy native buffers may be undercounted"
            ),
            "limitations": [
                "The absolute times depend on the machine and on Python/NumPy "
                "versions; compare relative slopes within this run.",
                "BFS-induced subgraphs are deterministic but not uniform random "
                "samples of the full graph.",
                "The item-catalogue measurement isolates scoring/ranking, not "
                "recommender training.",
            ],
        },
        "bottleneck_analysis": {
            "stage_timings_snapshot": stage_rows,
            "completed_stage_total_seconds": round(stage_total, 2),
            "largest_stages": top_stages,
            "r8_sub_experiment_seconds": measured_r8,
            "interpretation": (
                "The largest completed pipeline stages are reported above. "
                "Within influence maximization, Monte Carlo spread estimates "
                "dominate because each candidate evaluation runs repeated ICM "
                "simulations. The CELF comparison measures whether lazy "
                "evaluation reduces that cost."
            ),
        },
        "summary": {
            "intervention_finding": (
                "See intervention.findings; spread and adopter-rating changes "
                "are reported together, not asserted in advance."
            ),
            "celf_finding": celf_finding,
            "cost_finding": (
                "Measured runtime and memory are reported for network size, "
                "item-catalogue size, and promotion budget; the tables, not a "
                "pre-written speedup claim, are the evidence."
            ),
        },
    }

    _stage_end("R8: Intervention & Cost Analysis", ts)
    _save_json(results, results_dir / "requirement_8_results.json")
    return results


# ---------------------------------------------------------------------------
# Full experiment pipeline (Parts 1-8)
# ---------------------------------------------------------------------------

def run_all(config: dict):
    """Run all experiments for Parts 1-8."""
    t_total = time.time()
    _STAGE_TIMINGS.clear()
    results_dir = Path(config.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    _save_json(config, results_dir / "experiment_config.json")

    def _save_timings():
        _save_json({"stages": _STAGE_TIMINGS,
                    "total_seconds": round(time.time() - t_total, 2)},
                   results_dir / "stage_timings.json")

    # Requirement 1
    dataset, train, val, test, split_info = run_requirement_1(config, results_dir)
    _save_timings()

    # Requirement 2
    icf, mf, req2_results = run_requirement_2(dataset, train, val, test, config, results_dir)
    _save_timings()

    # Requirement 3
    req3_results = run_requirement_3(dataset, train, val, test, mf, config, results_dir)
    _save_timings()

    # Requirement 4
    req4_results = run_requirement_4(dataset, train, mf, config, results_dir)
    _save_timings()

    # Requirement 5 (Influence Maximization)
    req5_results = run_requirement_5(dataset, config, results_dir)
    _save_timings()

    # Requirement 6 (Group Recommendation)
    req6_results = run_requirement_6(dataset, train, mf, config, results_dir)
    _save_timings()

    # Requirement 7 (Comprehensive Evaluation across models)
    best_lambda = req3_results.get("best_lambda", 0.3)
    train_user_dict = _build_user_item_dict(train, dataset.user_to_idx, dataset.item_to_idx)
    social_model = SocialRecommender(
        baseline=mf,
        trust_graph=dataset.trust_graph,
        social_weight=best_lambda,
        global_mean=mf.global_mean if hasattr(mf, "global_mean") else 3.0,
    )
    social_model.set_training_data(train_user_dict)

    models_dict = {
        "item_based_cf": icf,
        "matrix_factorization": mf,
        "social_recommender": social_model,
    }
    req7_results = run_requirement_7(dataset, train, val, test, models_dict, config, results_dir)
    _save_timings()

    # Requirement 8 (Intervention & Cost Analysis)
    req8_results = run_requirement_8(dataset, train, mf, config, results_dir)
    _save_timings()

    total = time.time() - t_total
    logger.info("=" * 60)
    logger.info("ALL EXPERIMENTS (PARTS 1-8) COMPLETE in %.1fs (%.1f min). Results saved to: %s",
                total, total / 60, results_dir)
    for entry in _STAGE_TIMINGS:
        logger.info("  %-45s %8.1fs", entry["stage"], entry["seconds"])
    logger.info("=" * 60)

    return {
        "requirement_1": {"split_info": split_info},
        "requirement_2": req2_results,
        "requirement_3": req3_results,
        "requirement_4": req4_results,
        "requirement_5": req5_results,
        "requirement_6": req6_results,
        "requirement_7": req7_results,
        "requirement_8": req8_results,
    }
