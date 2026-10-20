"""
Evaluation metrics for recommendation quality.

Accuracy metrics (on held-out observed ratings):
  - MAE  (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

Ranking metrics (on top-K recommendation lists):
  - Precision@K
  - HitRate@K

A held-out item is considered "relevant" if its observed rating >= relevance_threshold
(default: 4).
"""

import logging
import time

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)


def mae(predictions: list[tuple[float, float]]) -> float:
    """Mean Absolute Error over (predicted, actual) pairs."""
    if not predictions:
        return float("nan")
    errors = [abs(pred - actual) for pred, actual in predictions]
    return float(np.mean(errors))


def rmse(predictions: list[tuple[float, float]]) -> float:
    """Root Mean Squared Error over (predicted, actual) pairs."""
    if not predictions:
        return float("nan")
    sq_errors = [(pred - actual) ** 2 for pred, actual in predictions]
    return float(np.sqrt(np.mean(sq_errors)))


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Precision@K: fraction of recommended[:k] items that are relevant.

    Parameters
    ----------
    recommended : list[int]
        Ranked list of item indices (most recommended first).
    relevant : set[int]
        Set of item indices considered relevant for this user.
    k : int
        Cutoff.
    """
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def hit_rate_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """HitRate@K: 1.0 if at least one relevant item appears in recommended[:k], else 0.0."""
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    return 1.0 if any(item in relevant for item in top_k) else 0.0


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Recall@K: fraction of relevant items that appear in recommended[:k]."""
    if k <= 0 or not relevant:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return float(hits / len(relevant))


def f1_at_k(precision: float, recall: float) -> float:
    """F1@K: harmonic mean of precision and recall."""
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) <= 0.0:
        return 0.0
    return float(2.0 * (precision * recall) / (precision + recall))


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance)."""
    if k <= 0 or not relevant:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0

    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in relevant:
            dcg += 1.0 / np.log2(idx + 2)

    n_ideal = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(n_ideal))
    if idcg <= 0.0:
        return 0.0
    return float(dcg / idcg)


def recommendation_coverage(all_recs: list[list[int]], n_total_items: int) -> float:
    """Catalog coverage: fraction of unique items recommended at least once."""
    if n_total_items <= 0 or not all_recs:
        return 0.0
    unique_items = set()
    for rec_list in all_recs:
        unique_items.update(rec_list)
    return float(len(unique_items) / n_total_items)


def intra_list_diversity(
    recommended: list[int],
    item_similarities: np.ndarray | dict | None = None,
) -> float:
    """Intra-list diversity: average pairwise distance (1 - similarity) in recommended list.

    If item_similarities is None, distinct items are assumed to have distance 1.0.
    """
    n = len(recommended)
    if n <= 1:
        return 0.0

    n_pairs = n * (n - 1) // 2
    total_dist = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            it_i, it_j = recommended[i], recommended[j]
            if item_similarities is None:
                sim = 1.0 if it_i == it_j else 0.0
            elif callable(item_similarities):
                sim = float(item_similarities(it_i, it_j))
            elif isinstance(item_similarities, np.ndarray):
                sim = float(item_similarities[it_i, it_j])
            elif isinstance(item_similarities, dict):
                sim = float(item_similarities.get((it_i, it_j), item_similarities.get((it_j, it_i), 0.0)))
            else:
                sim = 0.0
            total_dist += (1.0 - sim)

    return float(total_dist / n_pairs)


def _normalised_item_features(item_features) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return row-oriented item vectors and their Euclidean norms.

    ``item_features`` must have **one row per item** and one column per
    feature (for this project: users).  Binary training-interaction vectors or
    centred rating vectors are both valid.
    """
    X = sparse.csr_matrix(item_features, dtype=np.float64)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms = np.where(norms > 1e-12, norms, 0.0)
    return X, norms


def _diversity_from_prepared(
    recommended: list[int],
    X: sparse.csr_matrix,
    norms: np.ndarray,
) -> float:
    """Mean pairwise ``1 - cosine`` distance for one recommendation list."""
    items = [int(i) for i in recommended if 0 <= int(i) < X.shape[0]]
    n = len(items)
    if n <= 1:
        return 0.0

    sub = X[items, :]
    sub_norms = norms[items]
    dots = np.asarray((sub @ sub.T).toarray(), dtype=np.float64)
    denom = np.outer(sub_norms, sub_norms)
    sim = np.zeros_like(dots)
    valid = denom > 1e-12
    sim[valid] = dots[valid] / denom[valid]
    # Clip numerical noise; cosine similarity is defined on [-1, 1].
    sim = np.clip(sim, -1.0, 1.0)

    total = 0.0
    for i in range(n):
        total += float(np.sum(1.0 - sim[i, i + 1:]))
    return total / (n * (n - 1) / 2.0)


def intra_list_diversity_from_features(
    recommended: list[int],
    item_features,
) -> float:
    """Intra-list diversity using cosine distance between item feature vectors.

    This is the documented diversity measure used by the ranking evaluation:
    for a list of ``n`` distinct items, it is the mean pairwise
    ``1 - cosine(item_i, item_j)`` over the ``n(n-1)/2`` pairs.  It returns 0
    for lists with fewer than two items and treats zero-norm item vectors as
    having zero similarity to every other item.
    """
    X, norms = _normalised_item_features(item_features)
    return _diversity_from_prepared(recommended, X, norms)


def mean_intra_list_diversity(
    all_recs: list[list[int]],
    item_features,
    max_users: int | None = None,
    seed: int = 42,
) -> dict:
    """Average feature-based intra-list diversity over recommendation lists.

    ``max_users`` optionally caps the number of lists evaluated (deterministic
    seeded sample) so that full-catalogue evaluation remains bounded.
    """
    if not all_recs or item_features is None:
        return {"mean_diversity": 0.0, "n_lists": 0, "is_sampled": False,
                "sample_seed": None}

    recs = all_recs
    is_sampled = False
    if max_users is not None and 0 < max_users < len(recs):
        rng = np.random.RandomState(seed)
        idx = np.sort(rng.choice(len(recs), size=max_users, replace=False))
        recs = [recs[i] for i in idx]
        is_sampled = True

    X, norms = _normalised_item_features(item_features)
    values = [_diversity_from_prepared(rec, X, norms) for rec in recs]
    return {
        "mean_diversity": float(np.mean(values)) if values else 0.0,
        "n_lists": int(len(values)),
        "is_sampled": bool(is_sampled),
        "sample_seed": int(seed) if is_sampled else None,
    }


def evaluate_predictions(recommender, test_ratings, user_to_idx, item_to_idx,
                         rating_min=1, rating_max=5):
    """Evaluate a recommender on held-out ratings using MAE and RMSE.

    Parameters
    ----------
    recommender : object
        Must have a .predict(user_idx, item_idx) method or .predict_batch(u, i).
    test_ratings : list[tuple[int, int, int]]
        (user_id, item_id, rating) triples using original IDs.
    user_to_idx, item_to_idx : dict
        ID mappings.

    Returns
    -------
    results : dict with 'mae', 'rmse', 'n_evaluated', 'n_skipped'.
    """
    u_list = []
    i_list = []
    actual_list = []
    n_skipped = 0
    for uid, iid, actual in test_ratings:
        uidx = user_to_idx.get(uid)
        iidx = item_to_idx.get(iid)
        if uidx is None or iidx is None:
            n_skipped += 1
            continue
        u_list.append(uidx)
        i_list.append(iidx)
        actual_list.append(float(actual))

    if not u_list:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "n_evaluated": 0,
            "n_skipped": n_skipped,
        }

    # Vectorized fast path if supported
    if hasattr(recommender, "predict_batch"):
        preds = recommender.predict_batch(
            np.asarray(u_list, dtype=np.int64),
            np.asarray(i_list, dtype=np.int64),
            rating_min=float(rating_min),
            rating_max=float(rating_max),
        )
        preds = np.asarray(preds, dtype=np.float64)
        nan_mask = np.isnan(preds)
        if nan_mask.any():
            n_skipped += int(nan_mask.sum())
            valid = ~nan_mask
            preds = preds[valid]
            actuals = np.asarray(actual_list, dtype=np.float64)[valid]
        else:
            actuals = np.asarray(actual_list, dtype=np.float64)

        preds = np.clip(preds, rating_min, rating_max)
        diff = np.abs(preds - actuals)
        return {
            "mae": float(np.mean(diff)) if diff.size > 0 else float("nan"),
            "rmse": float(np.sqrt(np.mean(diff ** 2))) if diff.size > 0 else float("nan"),
            "n_evaluated": int(diff.size),
            "n_skipped": n_skipped,
        }

    # Fallback to per-pair prediction loop
    predictions = []
    for uidx, iidx, actual in zip(u_list, i_list, actual_list):
        pred = recommender.predict(uidx, iidx)
        if pred is None or np.isnan(pred):
            n_skipped += 1
            continue
        pred = float(np.clip(pred, rating_min, rating_max))
        predictions.append((pred, actual))

    return {
        "mae": mae(predictions),
        "rmse": rmse(predictions),
        "n_evaluated": len(predictions),
        "n_skipped": n_skipped,
    }


def evaluate_ranking(recommender, test_ratings, train_user_items,
                     user_to_idx, item_to_idx, k=10, relevance_threshold=4,
                     label: str = "ranking", log_every_seconds: float = 30.0,
                     sample_users: int | None = None, sample_seed: int = 42,
                     batch_size: int = 256, item_features=None,
                     max_diversity_users: int | None = 500,
                     diversity_seed: int = 42):
    """Evaluate top-K ranking quality.

    Parameters
    ----------
    recommender : object
        Must have a .recommend(user_idx, n, exclude) method or .recommend_batch(...).
    test_ratings : list[tuple[int, int, int]]
        Held-out (user_id, item_id, rating) using original IDs.
    train_user_items : dict[int, set[int]]
        user_idx -> set of item indices seen during training (to exclude).
    k : int
        Cutoff for Precision@K and HitRate@K.
    relevance_threshold : int
        Minimum rating to consider a held-out item relevant.
    label : str
        Name used in progress log messages.
    log_every_seconds : float
        Minimum interval between progress log lines.
    sample_users : int or None
        If provided, evaluate on a fixed, seeded sample of evaluation users.
    sample_seed : int
        Random seed for evaluation user sampling.
    batch_size : int
        Batch size when using recommender.recommend_batch.
    item_features : sparse matrix or None
        One row per item, one column per feature (training interactions are
        used in this project).  When provided, feature-based intra-list
        diversity is computed.
    max_diversity_users : int or None
        Cap on the number of recommendation lists used for the diversity
        estimate.  ``None`` evaluates every list.  The sample is deterministic
        and seeded by ``diversity_seed``.
    diversity_seed : int
        Seed for the diversity-list sample.

    Returns
    -------
    results : dict with ranking metrics and evaluation metadata.

    Notes
    -----
    * Relevance is binary: a held-out item is relevant when its observed test
      rating is ``>= relevance_threshold``.  Unobserved items are *not*
      treated as negatives; they are simply not in the relevant set.
    * Precision@K uses ``K`` as the denominator, even when fewer than ``K``
      eligible items exist.  Recall@K uses the number of relevant held-out
      items as the denominator.
    * F1@K is the mean of the per-user F1 scores, not the harmonic mean of the
      mean precision and mean recall.
    * Coverage is the fraction of the full item catalogue that appears at
      least once in the evaluated recommendation lists.
    * Diversity is the mean pairwise cosine distance between item feature
      vectors, computed on a bounded deterministic sample of lists.
    """
    from collections import defaultdict

    # Build per-user relevant items from test set
    user_relevant = defaultdict(set)
    for uid, iid, r in test_ratings:
        uidx = user_to_idx.get(uid)
        iidx = item_to_idx.get(iid)
        if uidx is None or iidx is None:
            continue
        if r >= relevance_threshold:
            user_relevant[uidx].add(iidx)

    eval_users = [u for u, items in user_relevant.items() if items]
    n_total = len(eval_users)
    eval_users.sort()  # deterministic sort before sampling

    is_sampled = False
    if sample_users is not None and 0 < sample_users < n_total:
        rng = np.random.RandomState(sample_seed)
        sampled_indices = rng.choice(n_total, size=sample_users, replace=False)
        sampled_indices.sort()
        eval_users = [eval_users[idx] for idx in sampled_indices]
        is_sampled = True
        logger.info(
            "[%s] Ranking evaluation over %d sampled users (seed=%d, out of %d eligible, K=%d)...",
            label, len(eval_users), sample_seed, n_total, k,
        )
    else:
        logger.info(
            "[%s] Ranking evaluation over %d users (full catalogue, K=%d)...",
            label, n_total, k,
        )

    t_start = time.time()
    t_last = t_start
    precisions = []
    hit_rates = []
    recalls = []
    f1_scores = []
    ndcgs = []
    all_recs = []

    # Use batched recommendation if available
    if hasattr(recommender, "recommend_batch"):
        n_eval = len(eval_users)
        for i in range(0, n_eval, batch_size):
            batch_u = eval_users[i:i + batch_size]
            batch_excludes = [train_user_items.get(u, set()) for u in batch_u]
            batch_recs = recommender.recommend_batch(batch_u, n=k, excludes=batch_excludes)
            for uidx, recs in zip(batch_u, batch_recs):
                rel_items = user_relevant[uidx]
                rec_list = [item_idx for item_idx, _ in recs]
                all_recs.append(rec_list)
                p = precision_at_k(rec_list, rel_items, k)
                r = recall_at_k(rec_list, rel_items, k)
                precisions.append(p)
                hit_rates.append(hit_rate_at_k(rec_list, rel_items, k))
                recalls.append(r)
                f1_scores.append(f1_at_k(p, r))
                ndcgs.append(ndcg_at_k(rec_list, rel_items, k))

            now = time.time()
            if now - t_last >= log_every_seconds:
                elapsed = now - t_start
                rate = len(precisions) / elapsed if elapsed > 0 else 0.0
                eta = (n_eval - len(precisions)) / rate if rate > 0 else 0.0
                logger.info("[%s] %d/%d users (%.1f%%), %.1f users/s, elapsed %.0fs, ETA %.0fs",
                            label, len(precisions), n_eval, 100.0 * len(precisions) / n_eval,
                            rate, elapsed, eta)
                t_last = now
    else:
        # Fallback to single-user loop
        n_eval = len(eval_users)
        for n_done, uidx in enumerate(eval_users, start=1):
            rel_items = user_relevant[uidx]
            exclude = train_user_items.get(uidx, set())
            recs = recommender.recommend(uidx, k, exclude=exclude)
            rec_list = [item_idx for item_idx, _ in recs]
            all_recs.append(rec_list)
            p = precision_at_k(rec_list, rel_items, k)
            r = recall_at_k(rec_list, rel_items, k)
            precisions.append(p)
            hit_rates.append(hit_rate_at_k(rec_list, rel_items, k))
            recalls.append(r)
            f1_scores.append(f1_at_k(p, r))
            ndcgs.append(ndcg_at_k(rec_list, rel_items, k))

            now = time.time()
            if now - t_last >= log_every_seconds:
                elapsed = now - t_start
                rate = n_done / elapsed if elapsed > 0 else 0.0
                eta = (n_eval - n_done) / rate if rate > 0 else 0.0
                logger.info("[%s] %d/%d users (%.1f%%), %.1f users/s, elapsed %.0fs, ETA %.0fs",
                            label, n_done, n_eval, 100.0 * n_done / n_eval, rate, elapsed, eta)
                t_last = now

    elapsed = time.time() - t_start
    logger.info("[%s] Ranking evaluation done: %d users in %.1fs (%.2f ms/user)",
                label, len(precisions), elapsed,
                1000.0 * elapsed / max(len(precisions), 1))

    mean_p = float(np.mean(precisions)) if precisions else float("nan")
    mean_r = float(np.mean(recalls)) if recalls else float("nan")
    mean_f1 = float(np.mean(f1_scores)) if f1_scores else float("nan")
    n_catalog = len(item_to_idx) if item_to_idx is not None else 0
    cov = recommendation_coverage(all_recs, n_catalog)

    if item_features is not None:
        diversity = mean_intra_list_diversity(
            all_recs, item_features, max_users=max_diversity_users,
            seed=diversity_seed,
        )
    else:
        diversity = {"mean_diversity": None, "n_lists": 0,
                     "is_sampled": False, "sample_seed": None}

    return {
        "precision_at_k": mean_p,
        "recall_at_k": mean_r,
        "f1_at_k": mean_f1,
        "hit_rate_at_k": float(np.mean(hit_rates)) if hit_rates else float("nan"),
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else float("nan"),
        "coverage": cov,
        "diversity": diversity["mean_diversity"],
        "diversity_method": (
            "mean pairwise 1-cosine(item feature vectors) over recommendation lists"
            if item_features is not None else None
        ),
        "diversity_n_lists": diversity["n_lists"],
        "diversity_is_sampled": diversity["is_sampled"],
        "diversity_sample_seed": diversity["sample_seed"],
        "catalog_size": int(n_catalog),
        "k": k,
        "relevance_threshold": relevance_threshold,
        "relevance_definition": (
            f"held-out test rating >= {relevance_threshold}; unobserved items "
            "are not treated as negatives"
        ),
        "candidate_definition": (
            "all catalogue items except the user's training items; no negative "
            "item sampling"
        ),
        "n_users_evaluated": len(precisions),
        "total_eligible_users": n_total,
        "is_sampled": is_sampled,
        "sample_seed": sample_seed if is_sampled else None,
        "eval_time_seconds": round(elapsed, 2),
    }
