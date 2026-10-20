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
                     batch_size: int = 256):
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

    Returns
    -------
    results : dict with ranking metrics and evaluation metadata.
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
                precisions.append(precision_at_k(rec_list, rel_items, k))
                hit_rates.append(hit_rate_at_k(rec_list, rel_items, k))

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
            precisions.append(precision_at_k(rec_list, rel_items, k))
            hit_rates.append(hit_rate_at_k(rec_list, rel_items, k))

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

    return {
        "precision_at_k": float(np.mean(precisions)) if precisions else float("nan"),
        "hit_rate_at_k": float(np.mean(hit_rates)) if hit_rates else float("nan"),
        "k": k,
        "relevance_threshold": relevance_threshold,
        "n_users_evaluated": len(precisions),
        "total_eligible_users": n_total,
        "is_sampled": is_sampled,
        "sample_seed": sample_seed if is_sampled else None,
        "eval_time_seconds": round(elapsed, 2),
    }
