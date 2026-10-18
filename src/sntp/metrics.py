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


def evaluate_predictions(recommender, test_ratings, user_to_idx, item_to_idx,
                         rating_min=1, rating_max=5):
    """Evaluate a recommender on held-out ratings using MAE and RMSE.

    Parameters
    ----------
    recommender : object
        Must have a .predict(user_idx, item_idx) method.
    test_ratings : list[tuple[int, int, int]]
        (user_id, item_id, rating) triples using original IDs.
    user_to_idx, item_to_idx : dict
        ID mappings.

    Returns
    -------
    results : dict with 'mae', 'rmse', 'n_evaluated', 'n_skipped'.
    """
    predictions = []
    n_skipped = 0
    for uid, iid, actual in test_ratings:
        uidx = user_to_idx.get(uid)
        iidx = item_to_idx.get(iid)
        if uidx is None or iidx is None:
            n_skipped += 1
            continue
        pred = recommender.predict(uidx, iidx)
        if pred is None or np.isnan(pred):
            n_skipped += 1
            continue
        # Clip to valid range
        pred = float(np.clip(pred, rating_min, rating_max))
        predictions.append((pred, float(actual)))

    return {
        "mae": mae(predictions),
        "rmse": rmse(predictions),
        "n_evaluated": len(predictions),
        "n_skipped": n_skipped,
    }


def evaluate_ranking(recommender, test_ratings, train_user_items,
                     user_to_idx, item_to_idx, k=10, relevance_threshold=4,
                     label: str = "ranking", log_every_seconds: float = 30.0):
    """Evaluate top-K ranking quality.

    Parameters
    ----------
    recommender : object
        Must have a .recommend(user_idx, n, exclude) method returning [(item_idx, score), ...].
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

    Returns
    -------
    results : dict with 'precision_at_k', 'hit_rate_at_k', 'n_users_evaluated'.
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

    n_total = len(user_relevant)
    logger.info("[%s] Ranking evaluation over %d users (full catalogue, K=%d)...",
                label, n_total, k)
    t_start = time.time()
    t_last = t_start

    precisions = []
    hit_rates = []
    for n_done, (uidx, rel_items) in enumerate(user_relevant.items(), start=1):
        if not rel_items:
            continue
        exclude = train_user_items.get(uidx, set())
        recs = recommender.recommend(uidx, k, exclude=exclude)
        rec_list = [item_idx for item_idx, _ in recs]
        precisions.append(precision_at_k(rec_list, rel_items, k))
        hit_rates.append(hit_rate_at_k(rec_list, rel_items, k))

        now = time.time()
        if now - t_last >= log_every_seconds:
            elapsed = now - t_start
            rate = n_done / elapsed
            eta = (n_total - n_done) / rate if rate > 0 else float("inf")
            logger.info("[%s] %d/%d users (%.1f%%), %.1f users/s, elapsed %.0fs, ETA %.0fs",
                        label, n_done, n_total, 100.0 * n_done / n_total, rate, elapsed, eta)
            t_last = now

    elapsed = time.time() - t_start
    logger.info("[%s] Ranking evaluation done: %d users in %.1fs (%.2f ms/user)",
                label, len(precisions), elapsed,
                1000.0 * elapsed / max(len(precisions), 1))

    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else float("nan"),
        "hit_rate_at_k": float(np.mean(hit_rates)) if hit_rates else float("nan"),
        "k": k,
        "relevance_threshold": relevance_threshold,
        "n_users_evaluated": len(precisions),
        "eval_time_seconds": round(elapsed, 2),
    }
