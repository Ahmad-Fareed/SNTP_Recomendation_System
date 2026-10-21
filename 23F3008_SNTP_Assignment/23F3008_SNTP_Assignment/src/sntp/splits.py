"""
Deterministic train / validation / test split for ratings.

Strategy (70 / 15 / 15):
  - For users with >= 3 ratings: at least 1 rating guaranteed in each split.
    Remaining ratings are assigned randomly according to the 70/15/15 ratio.
  - For users with < 3 ratings: all ratings go to training.
    These users are excluded from per-user held-out evaluation and this is logged.

The split is fully deterministic given the seed.
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


def split_ratings(ratings, seed: int = 42, train_frac: float = 0.70,
                  val_frac: float = 0.15):
    """Split ratings into train / validation / test sets.

    Parameters
    ----------
    ratings : list[tuple[int, int, int]]
        (user_id, item_id, rating) triples (original IDs).
    seed : int
        Random seed for reproducibility.
    train_frac, val_frac : float
        Fractions for train and validation; test gets the remainder.

    Returns
    -------
    train, val, test : list[tuple[int, int, int]]
    split_info : dict
        Statistics about the split.
    """
    rng = np.random.RandomState(seed)
    test_frac = 1.0 - train_frac - val_frac

    # Group ratings by user
    user_ratings = defaultdict(list)
    for uid, iid, r in ratings:
        user_ratings[uid].append((uid, iid, r))

    train, val, test = [], [], []
    excluded_users = []
    included_users = []

    for uid in sorted(user_ratings.keys()):
        user_recs = user_ratings[uid]
        n = len(user_recs)

        if n < 3:
            # All go to training; user excluded from held-out evaluation
            train.extend(user_recs)
            excluded_users.append(uid)
            continue

        included_users.append(uid)

        # Shuffle deterministically
        indices = np.arange(n)
        rng.shuffle(indices)

        # Guarantee at least 1 in each split
        guaranteed_train = indices[0]
        guaranteed_val = indices[1]
        guaranteed_test = indices[2]

        remaining = indices[3:]

        # Split remaining by target fractions
        if len(remaining) > 0:
            n_remaining = len(remaining)
            # Adjust fractions to account for the 3 guaranteed records
            n_train_extra = max(0, round(train_frac * n) - 1)
            n_val_extra = max(0, round(val_frac * n) - 1)
            # Clamp to available
            n_train_extra = min(n_train_extra, n_remaining)
            n_val_extra = min(n_val_extra, n_remaining - n_train_extra)

            train_indices = list(remaining[:n_train_extra])
            val_indices = list(remaining[n_train_extra:n_train_extra + n_val_extra])
            test_indices = list(remaining[n_train_extra + n_val_extra:])
        else:
            train_indices, val_indices, test_indices = [], [], []

        # Add guaranteed records
        train_indices.append(guaranteed_train)
        val_indices.append(guaranteed_val)
        test_indices.append(guaranteed_test)

        for idx in train_indices:
            train.append(user_recs[idx])
        for idx in val_indices:
            val.append(user_recs[idx])
        for idx in test_indices:
            test.append(user_recs[idx])

    split_info = {
        "seed": seed,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": round(test_frac, 4),
        "total_ratings": len(ratings),
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "users_included_in_evaluation": len(included_users),
        "users_excluded_lt3_ratings": len(excluded_users),
    }
    logger.info("Split: %s", split_info)
    return train, val, test, split_info
