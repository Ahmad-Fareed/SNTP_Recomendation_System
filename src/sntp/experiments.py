"""
Experiment runner for SNTP Assignment 2, Parts 1-4.

Orchestrates:
  - Requirement 1: Data loading, statistics, and split.
  - Requirement 2: Item-based CF and MF training + evaluation.
  - Requirement 3: Social extension experiments.
  - Requirement 4: ICM and SIR diffusion simulations.

All experiments save machine-readable results (JSON) alongside configuration.
"""

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from sntp.data import load_dataset
from sntp.splits import split_ratings
from sntp.metrics import evaluate_predictions, evaluate_ranking
from sntp.recommenders.item_based import ItemBasedCF
from sntp.recommenders.matrix_factorization import BiasedMF
from sntp.recommenders.social import SocialRecommender, SimilarUserRecommender
from sntp.diffusion.icm import ICMSimulator
from sntp.diffusion.sir import SIRSimulator

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
    ts = _stage_start("R2: item-based CF top-K ranking")
    icf_rank_results = evaluate_ranking(
        icf, test, train_user_items,
        dataset.user_to_idx, dataset.item_to_idx,
        k=k, relevance_threshold=config.get("relevance_threshold", 4),
        label="ICF",
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

def run_requirement_3(dataset, train, val, test, baseline, config: dict, results_dir: Path):
    """Social-enhanced recommendation experiments."""
    logger.info("=" * 60)
    logger.info("REQUIREMENT 3: Social extension")
    logger.info("=" * 60)

    train_user_dict = _build_user_item_dict(train, dataset.user_to_idx, dataset.item_to_idx)
    train_user_items = _build_train_user_items_set(train, dataset.user_to_idx, dataset.item_to_idx)

    social_config = config.get("social", {})
    lambda_values = social_config.get("lambda_values", [0.0, 0.1, 0.3, 0.5, 0.7])
    k_similar = social_config.get("k_similar", 20)
    k = config.get("top_k", 10)

    results = {}

    # --- Baseline (lambda=0, same as underlying recommender) ---
    ts = _stage_start("R3: baseline evaluation")
    baseline_pred = evaluate_predictions(
        baseline, test, dataset.user_to_idx, dataset.item_to_idx
    )
    baseline_rank = evaluate_ranking(
        baseline, test, train_user_items,
        dataset.user_to_idx, dataset.item_to_idx,
        k=k, relevance_threshold=config.get("relevance_threshold", 4),
        label="R3 baseline",
    )
    _stage_end("R3: baseline evaluation", ts)
    results["baseline"] = {"prediction": baseline_pred, "ranking": baseline_rank}
    logger.info("Baseline: MAE=%.4f, RMSE=%.4f", baseline_pred["mae"], baseline_pred["rmse"])

    # --- Social-enhanced at different lambda ---
    for lam in lambda_values:
        ts = _stage_start(f"R3: social lambda={lam}")
        social_rec = SocialRecommender(
            baseline, dataset.trust_graph, social_weight=lam,
            global_mean=baseline.global_mean if hasattr(baseline, 'global_mean') else 3.0,
        )
        social_rec.set_training_data(train_user_dict)

        soc_pred = evaluate_predictions(
            social_rec, test, dataset.user_to_idx, dataset.item_to_idx
        )
        soc_rank = evaluate_ranking(
            social_rec, test, train_user_items,
            dataset.user_to_idx, dataset.item_to_idx,
            k=k, relevance_threshold=config.get("relevance_threshold", 4),
            label=f"Social lambda={lam}",
        )
        _stage_end(f"R3: social lambda={lam}", ts)
        results[f"social_lambda_{lam}"] = {"prediction": soc_pred, "ranking": soc_rank}
        logger.info("Social (λ=%.1f): MAE=%.4f, RMSE=%.4f", lam, soc_pred["mae"], soc_pred["rmse"])

    # --- Users with limited history: does social help? ---
    ts = _stage_start("R3: sparse-user analysis")
    user_train_counts = defaultdict(int)
    for uid, _, _ in train:
        user_train_counts[uid] += 1
    sparse_users = [uid for uid, c in user_train_counts.items() if c <= 5]
    sparse_user_set = set(sparse_users)  # build once (was rebuilt per test rating)
    sparse_test = [(u, i, r) for u, i, r in test if u in sparse_user_set]

    if sparse_test:
        best_lambda = lambda_values[len(lambda_values) // 2]  # middle value
        social_rec = SocialRecommender(
            baseline, dataset.trust_graph, social_weight=best_lambda,
            global_mean=baseline.global_mean if hasattr(baseline, 'global_mean') else 3.0,
        )
        social_rec.set_training_data(train_user_dict)

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
    # Sample users who have trust connections for the comparison
    users_with_trust = [u for u in dataset.trust_graph if len(dataset.trust_graph[u]) > 0
                        and u in train_user_dict]
    n_sample = min(social_config.get("comparison_sample", 200), len(users_with_trust))
    rng = np.random.RandomState(config.get("split_seed", 42))
    sampled_users = rng.choice(users_with_trust, size=n_sample, replace=False).tolist() if users_with_trust else []

    if sampled_users:
        best_lambda = lambda_values[len(lambda_values) // 2]
        sim_rec = SimilarUserRecommender(
            baseline, dataset.trust_graph, social_weight=best_lambda,
            k_similar=k_similar,
            global_mean=baseline.global_mean if hasattr(baseline, 'global_mean') else 3.0,
        )
        sim_rec.set_training_data(train_user_dict)
        sim_rec.compute_similar_users(sampled_users)

        # Evaluate only on sampled users' test ratings
        sampled_set = set(dataset.idx_to_user[u] for u in sampled_users)
        sampled_test = [(u, i, r) for u, i, r in test if u in sampled_set]

        if sampled_test:
            social_rec = SocialRecommender(
                baseline, dataset.trust_graph, social_weight=best_lambda,
                global_mean=baseline.global_mean if hasattr(baseline, 'global_mean') else 3.0,
            )
            social_rec.set_training_data(train_user_dict)

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
# Requirement 4: Diffusion models
# ---------------------------------------------------------------------------

def _affinity_fn_factory(recommender, item_idx: int, user_to_idx: dict,
                         default_affinity: float = 0.3):
    """Create an affinity function for a specific item.

    affinity(u) = clip((predicted_rating(u, item) - 1) / 4, 0, 1)
    Falls back to default_affinity if no prediction is available.
    """
    def affinity(user_idx):
        pred = recommender.predict(user_idx, item_idx)
        if pred is None or np.isnan(pred):
            return default_affinity
        return float(np.clip((pred - 1.0) / 4.0, 0.0, 1.0))
    return affinity


def run_requirement_4(dataset, train, recommender, config: dict, results_dir: Path):
    """Run ICM and SIR diffusion experiments."""
    logger.info("=" * 60)
    logger.info("REQUIREMENT 4: Diffusion models")
    logger.info("=" * 60)

    diff_config = config.get("diffusion", {})
    n_simulations = diff_config.get("n_simulations", 10)
    seed_size = diff_config.get("seed_size", 10)
    base_seed = diff_config.get("seed", 42)
    max_rounds = diff_config.get("max_rounds", 50)
    propagation_probs = diff_config.get("propagation_probs", [0.01, 0.05, 0.1])
    recovery_probs = diff_config.get("recovery_probs", [0.0, 0.1, 0.3, 0.5])
    default_affinity = diff_config.get("default_affinity", 0.3)

    # Select a target item for diffusion (moderately popular)
    from collections import Counter
    item_counts = Counter()
    for _, iid, _ in train:
        item_counts[iid] += 1
    # Pick an item near the median popularity
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1])
    target_item_orig = sorted_items[len(sorted_items) // 2][0]
    target_item_idx = dataset.item_to_idx[target_item_orig]

    logger.info("Target item for diffusion: original_id=%d, idx=%d, train_ratings=%d",
                target_item_orig, target_item_idx, item_counts[target_item_orig])

    # Users who have NOT rated the target item (eligible for adoption)
    rated_users = set()
    for uid, iid, _ in train:
        if iid == target_item_orig:
            rated_users.add(dataset.user_to_idx[uid])

    # Select seed users from those who haven't rated it
    all_user_indices = set(range(dataset.n_users))
    eligible = list(all_user_indices - rated_users)

    # Affinity function
    aff_fn = _affinity_fn_factory(recommender, target_item_idx, dataset.user_to_idx,
                                  default_affinity=default_affinity)

    rng = np.random.RandomState(base_seed)
    results = {
        "target_item": {"original_id": target_item_orig, "idx": target_item_idx},
        "seed_size": seed_size,
        "n_simulations": n_simulations,
        "icm_canonical": {},
        "icm_preference_aware": {},
        "sir": {},
    }

    # --- ICM Canonical ---
    logger.info("--- ICM Canonical ---")
    ts = _stage_start("R4: ICM canonical")
    for p_prob in propagation_probs:
        key = f"p={p_prob}"
        run_results = []
        for sim_i in range(n_simulations):
            sim_seed = base_seed + sim_i
            seed_users = set(rng.choice(eligible, size=min(seed_size, len(eligible)), replace=False))

            icm = ICMSimulator(
                dataset.propagation_graph, propagation_prob=p_prob,
                max_rounds=max_rounds, seed=sim_seed,
            )
            result = icm.run_canonical(seed_users)
            run_results.append({
                "sim_seed": sim_seed,
                "n_adopters": result["n_adopters"],
                "n_rounds": result["n_rounds"],
                "seed_users": list(seed_users),
            })

        adopter_counts = [r["n_adopters"] for r in run_results]
        results["icm_canonical"][key] = {
            "propagation_prob": p_prob,
            "runs": run_results,
            "summary": {
                "mean_adopters": float(np.mean(adopter_counts)),
                "std_adopters": float(np.std(adopter_counts)),
                "min_adopters": int(np.min(adopter_counts)),
                "max_adopters": int(np.max(adopter_counts)),
            },
        }
        logger.info("ICM canonical (p=%.2f): mean=%.1f ± %.1f adopters",
                    p_prob, np.mean(adopter_counts), np.std(adopter_counts))

    # --- ICM Preference-Aware ---
    _stage_end("R4: ICM canonical", ts)
    logger.info("--- ICM Preference-Aware ---")
    ts = _stage_start("R4: ICM preference-aware")
    for p_prob in propagation_probs:
        key = f"p={p_prob}"
        run_results = []
        for sim_i in range(n_simulations):
            sim_seed = base_seed + sim_i
            seed_users = set(rng.choice(eligible, size=min(seed_size, len(eligible)), replace=False))

            icm = ICMSimulator(
                dataset.propagation_graph, propagation_prob=p_prob,
                max_rounds=max_rounds, seed=sim_seed,
            )
            result = icm.run_preference_aware(
                seed_users, affinity_fn=aff_fn,
                base_adopt_prob=diff_config.get("base_adopt_prob", 0.5),
            )
            run_results.append({
                "sim_seed": sim_seed,
                "n_adopters": result["n_adopters"],
                "n_exposed": result["n_exposed"],
                "n_rounds": result["n_rounds"],
            })

        adopter_counts = [r["n_adopters"] for r in run_results]
        exposed_counts = [r["n_exposed"] for r in run_results]
        results["icm_preference_aware"][key] = {
            "propagation_prob": p_prob,
            "runs": run_results,
            "summary": {
                "mean_adopters": float(np.mean(adopter_counts)),
                "std_adopters": float(np.std(adopter_counts)),
                "mean_exposed": float(np.mean(exposed_counts)),
                "std_exposed": float(np.std(exposed_counts)),
            },
        }
        logger.info("ICM preference-aware (p=%.2f): mean=%.1f adopters, %.1f exposed",
                    p_prob, np.mean(adopter_counts), np.mean(exposed_counts))

    # --- SIR ---
    _stage_end("R4: ICM preference-aware", ts)
    logger.info("--- SIR Model ---")
    ts = _stage_start("R4: SIR")
    for p_prob in propagation_probs:
        for g_prob in recovery_probs:
            key = f"beta={p_prob}_gamma={g_prob}"
            run_results = []
            for sim_i in range(n_simulations):
                sim_seed = base_seed + sim_i
                seed_users = set(rng.choice(eligible, size=min(seed_size, len(eligible)), replace=False))

                sir = SIRSimulator(
                    dataset.propagation_graph, transmission_prob=p_prob,
                    recovery_prob=g_prob, max_rounds=max_rounds, seed=sim_seed,
                )
                result = sir.run(seed_users, affinity_fn=aff_fn,
                                base_adopt_prob=diff_config.get("base_adopt_prob", 0.5),
                                all_users=all_user_indices)
                run_results.append({
                    "sim_seed": sim_seed,
                    "n_ever_infected": result["n_ever_infected"],
                    "n_final_recovered": result["n_final_recovered"],
                    "n_rounds": result["n_rounds"],
                    "final_S": result["state_history"][-1]["S"],
                    "final_I": result["state_history"][-1]["I"],
                    "final_R": result["state_history"][-1]["R"],
                })

            ever_infected = [r["n_ever_infected"] for r in run_results]
            results["sir"][key] = {
                "transmission_prob": p_prob,
                "recovery_prob": g_prob,
                "runs": run_results,
                "summary": {
                    "mean_ever_infected": float(np.mean(ever_infected)),
                    "std_ever_infected": float(np.std(ever_infected)),
                },
            }
            logger.info("SIR (β=%.2f, γ=%.2f): mean=%.1f ± %.1f ever infected",
                        p_prob, g_prob, np.mean(ever_infected), np.std(ever_infected))
    _stage_end("R4: SIR", ts)

    _save_json(results, results_dir / "requirement_4_results.json")
    return results


# ---------------------------------------------------------------------------
# Full experiment pipeline
# ---------------------------------------------------------------------------

def run_all(config: dict):
    """Run all experiments for Parts 1-4."""
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

    # Requirement 3 (using MF as baseline — generally better for social extension)
    req3_results = run_requirement_3(dataset, train, val, test, mf, config, results_dir)
    _save_timings()

    # Requirement 4 (using MF for affinity predictions)
    req4_results = run_requirement_4(dataset, train, mf, config, results_dir)
    _save_timings()

    total = time.time() - t_total
    logger.info("=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE in %.1fs (%.1f min). Results saved to: %s",
                total, total / 60, results_dir)
    for entry in _STAGE_TIMINGS:
        logger.info("  %-45s %8.1fs", entry["stage"], entry["seconds"])
    logger.info("=" * 60)

    return {
        "requirement_1": {"split_info": split_info},
        "requirement_2": req2_results,
        "requirement_3": req3_results,
        "requirement_4": req4_results,
    }
