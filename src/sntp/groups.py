"""
Part 6: Group Recommendation strategies and evaluation.

Implements group recommendation for groups with potentially conflicting preferences:
1. Average satisfaction: Mean predicted rating across group members.
2. Least misery: Minimum predicted rating across group members.
3. Most pleasure: Maximum predicted rating across group members.

Also evaluates member satisfaction, disagreement/variance, and connects
group recommendations with diffusion spread (preference-aware ICM).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

AGGREGATION_STRATEGIES = ("average", "least_misery", "most_pleasure")


def generate_groups(
    user_ids: list[int],
    n_groups: int = 5,
    group_size: int = 4,
    conflicting: bool = True,
    user_item_matrix=None,
    seed: int = 42,
) -> list[list[int]]:
    """Generate groups of users, optionally with conflicting preferences.

    Parameters
    ----------
    user_ids : list[int]
        Pool of eligible user indices.
    n_groups : int
        Number of groups to generate.
    group_size : int
        Number of users per group.
    conflicting : bool
        If True and user_item_matrix is available, selects users with low or
        negative rating correlation / contrasting mean ratings to simulate
        conflicting preferences.
    user_item_matrix : scipy.sparse matrix or None
        Sparse user-item ratings matrix.
    seed : int
        Random seed.

    Returns
    -------
    groups : list[list[int]]
    """
    rng = np.random.RandomState(seed)
    users = list(user_ids)
    if len(users) < group_size:
        return [users] if users else []

    groups = []

    if conflicting and user_item_matrix is not None:
        # Compute user activity / mean ratings to mix contrasting profiles
        user_means = {}
        for u in users:
            row = user_item_matrix.getrow(u)
            if row.nnz > 0:
                user_means[u] = float(row.data.mean())
            else:
                user_means[u] = 3.0

        sorted_users = sorted(users, key=lambda u: user_means[u])
        low_pool = sorted_users[: len(sorted_users) // 3]
        high_pool = sorted_users[2 * len(sorted_users) // 3 :]
        mid_pool = sorted_users[len(sorted_users) // 3 : 2 * len(sorted_users) // 3]

        for g_idx in range(n_groups):
            # Mix users from contrasting spectrums to ensure preference conflicts
            half = group_size // 2
            rest = group_size - half
            u_low = (
                rng.choice(low_pool, size=min(half, len(low_pool)), replace=False).tolist()
                if low_pool
                else []
            )
            u_high = (
                rng.choice(high_pool, size=min(rest, len(high_pool)), replace=False).tolist()
                if high_pool
                else []
            )
            group = u_low + u_high
            if len(group) < group_size and mid_pool:
                needed = group_size - len(group)
                group.extend(rng.choice(mid_pool, size=min(needed, len(mid_pool)), replace=False).tolist())
            groups.append([int(u) for u in group])
    else:
        # Uniform random group generation without replacement per group
        for _ in range(n_groups):
            chosen = rng.choice(users, size=group_size, replace=False).tolist()
            groups.append([int(u) for u in chosen])

    return groups


def score_group_items(
    recommender,
    group: list[int],
    n_items: int,
    strategy: str = "average",
) -> np.ndarray:
    """Compute aggregate scores for all items for a given group.

    Parameters
    ----------
    recommender : object
        Recommender model with score_all or predict.
    group : list[int]
        List of user indices in the group.
    n_items : int
        Total number of items in catalog.
    strategy : str
        'average', 'least_misery', or 'most_pleasure'.

    Returns
    -------
    group_scores : np.ndarray of shape (n_items,)
    """
    if strategy not in AGGREGATION_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of {AGGREGATION_STRATEGIES}"
        )

    # Gather predicted ratings for each group member across all items
    member_scores = []
    for u in group:
        if hasattr(recommender, "score_all"):
            scores = recommender.score_all(u)
        else:
            scores = np.array([recommender.predict(u, i) for i in range(n_items)], dtype=np.float64)
        member_scores.append(scores)

    matrix = np.vstack(member_scores)  # shape (len(group), n_items)

    if strategy == "average":
        return np.mean(matrix, axis=0)
    elif strategy == "least_misery":
        return np.min(matrix, axis=0)
    elif strategy == "most_pleasure":
        return np.max(matrix, axis=0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def recommend_for_group(
    recommender,
    group: list[int],
    n_items: int,
    n: int = 5,
    strategy: str = "average",
    train_user_items: dict[int, set[int]] | None = None,
    exclude_seen_by_any: bool = True,
) -> list[tuple[int, float]]:
    """Recommend top-N items for a group using the specified aggregation strategy.

    Parameters
    ----------
    recommender : object
        Trained recommendation model.
    group : list[int]
        User indices in the group.
    n_items : int
        Catalog size.
    n : int
        Number of items to recommend.
    strategy : str
        One of 'average', 'least_misery', 'most_pleasure'.
    train_user_items : dict[int, set[int]] or None
        Training items per user.
    exclude_seen_by_any : bool
        If True, exclude items rated by any member in training.

    Returns
    -------
    recommendations : list of (item_idx, group_score)
    """
    scores = score_group_items(recommender, group, n_items, strategy=strategy)

    exclude_set = set()
    if train_user_items is not None:
        if exclude_seen_by_any:
            for u in group:
                exclude_set.update(train_user_items.get(u, set()))
        else:
            # Exclude only items seen by ALL members
            sets = [train_user_items.get(u, set()) for u in group]
            if sets:
                exclude_set = set.intersection(*sets)

    for item_idx in exclude_set:
        if 0 <= item_idx < len(scores):
            scores[item_idx] = -np.inf

    # Top-N selection with deterministic tie-breaking
    top_indices = np.argsort(-scores, kind="stable")[:n]
    results = []
    for idx in top_indices:
        if np.isneginf(scores[idx]):
            break
        results.append((int(idx), float(scores[idx])))

    return results


def evaluate_group_satisfaction(
    recommender,
    group: list[int],
    item_idx: int,
) -> dict[str, Any]:
    """Evaluate individual and group satisfaction metrics for an item.

    Parameters
    ----------
    recommender : object
        Recommender with predict or score_all.
    group : list[int]
        User indices in the group.
    item_idx : int
        Target item index.

    Returns
    -------
    metrics : dict with keys:
        'member_ratings': dict[int, float]
        'mean_satisfaction': float
        'min_satisfaction': float
        'max_satisfaction': float
        'variance': float
        'std': float
        'range': float (max - min)
    """
    member_ratings = {}
    for u in group:
        if hasattr(recommender, "predict"):
            pred = recommender.predict(u, item_idx)
        elif hasattr(recommender, "score_all"):
            pred = float(recommender.score_all(u)[item_idx])
        else:
            pred = 3.0
        member_ratings[int(u)] = float(pred)

    ratings = list(member_ratings.values())
    return {
        "member_ratings": member_ratings,
        "mean_satisfaction": float(np.mean(ratings)),
        "min_satisfaction": float(np.min(ratings)),
        "max_satisfaction": float(np.max(ratings)),
        "variance": float(np.var(ratings)),
        "std": float(np.std(ratings)),
        "range": float(np.max(ratings) - np.min(ratings)),
    }
