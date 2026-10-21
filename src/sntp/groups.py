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
        Sparse user-item ratings matrix.  **Pass the training matrix only.**
        Using the full observed matrix would leak validation/test preferences
        into group construction.
    seed : int
        Random seed.

    Returns
    -------
    groups : list[list[int]]
        Each group contains distinct user indices.  Groups may overlap each
        other; each group itself is a set.
    """
    rng = np.random.RandomState(seed)
    users = list(dict.fromkeys(int(u) for u in user_ids))
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
            if len(group) < group_size:
                remaining = [u for u in users if u not in group]
                needed = group_size - len(group)
                if remaining:
                    group.extend(rng.choice(remaining, size=min(needed, len(remaining)),
                                            replace=False).tolist())
            groups.append([int(u) for u in dict.fromkeys(group)])
    else:
        # Uniform random group generation without replacement per group
        for _ in range(n_groups):
            chosen = rng.choice(users, size=group_size, replace=False).tolist()
            groups.append([int(u) for u in chosen])

    return groups


def aggregate_member_scores(member_scores: np.ndarray, strategy: str) -> np.ndarray:
    """Aggregate a ``(n_members, n_items)`` score matrix into group scores.

    This pure function is the single implementation used by
    :func:`score_group_items`; exposing it makes the three aggregation rules
    directly testable on hand-checkable matrices.

    * ``average``       -> arithmetic mean over members
    * ``least_misery``  -> element-wise minimum (maximise the worst member)
    * ``most_pleasure`` -> element-wise maximum (maximise the happiest member)
    """
    if strategy not in AGGREGATION_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Expected one of {AGGREGATION_STRATEGIES}"
        )
    matrix = np.asarray(member_scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("member_scores must be a 2-D (members x items) array")
    if matrix.shape[0] == 0:
        raise ValueError("group must contain at least one member")

    if strategy == "average":
        return np.mean(matrix, axis=0)
    if strategy == "least_misery":
        return np.min(matrix, axis=0)
    return np.max(matrix, axis=0)


def _member_scores(recommender, user_idx: int, n_items: int,
                   raw_scores: bool) -> np.ndarray:
    """Return one member's scores, preserving raw values when requested."""
    if hasattr(recommender, "score_all"):
        if raw_scores:
            try:
                return np.asarray(
                    recommender.score_all(user_idx, raw_scores=True),
                    dtype=np.float64,
                )
            except TypeError:
                # Recommenders without the raw-scores option (test doubles).
                pass
        return np.asarray(recommender.score_all(user_idx), dtype=np.float64)
    return np.array(
        [recommender.predict(user_idx, i) for i in range(n_items)],
        dtype=np.float64,
    )


def score_group_items(
    recommender,
    group: list[int],
    n_items: int,
    strategy: str = "average",
    raw_scores: bool = False,
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
    raw_scores : bool
        If True, ask the recommender for unclipped member scores.  Aggregates
        are then ordering scores, not ratings.  Calling code that reports
        satisfaction must request clipped scores separately.

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
        scores = _member_scores(recommender, u, n_items, raw_scores)
        if scores.shape != (n_items,):
            raise ValueError(
                f"recommender returned shape {scores.shape} for user {u}; "
                f"expected ({n_items},)"
            )
        member_scores.append(scores)

    matrix = np.vstack(member_scores)  # shape (len(group), n_items)
    return aggregate_member_scores(matrix, strategy)


def recommend_for_group(
    recommender,
    group: list[int],
    n_items: int,
    n: int = 5,
    strategy: str = "average",
    train_user_items: dict[int, set[int]] | None = None,
    exclude_seen_by_any: bool = True,
    rank_with_raw_scores: bool = True,
) -> list[tuple[int, float]]:
    """Recommend top-N items for a group using the specified aggregation strategy.

    Ordering uses the recommender's raw (unclipped) member scores when
    available; the returned score is the clipped group satisfaction for the
    selected item.  This avoids artificial ties at the rating ceiling while
    keeping the reported number interpretable as a group rating.

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
    rank_with_raw_scores : bool
        If True, aggregate raw member scores for ordering and separately
        aggregate clipped scores for the reported satisfaction value.

    Returns
    -------
    recommendations : list of (item_idx, group_satisfaction)
    """
    ranking_scores = score_group_items(
        recommender, group, n_items, strategy=strategy, raw_scores=rank_with_raw_scores
    )
    if rank_with_raw_scores:
        display_scores = score_group_items(
            recommender, group, n_items, strategy=strategy, raw_scores=False
        )
    else:
        display_scores = ranking_scores

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

    ranking_scores = np.array(ranking_scores, dtype=np.float64, copy=True)
    for item_idx in exclude_set:
        if 0 <= item_idx < len(ranking_scores):
            ranking_scores[item_idx] = -np.inf

    # Top-N selection with deterministic tie-breaking
    top_indices = np.argsort(-ranking_scores, kind="stable")[:n]
    results = []
    for idx in top_indices:
        if np.isneginf(ranking_scores[idx]):
            break
        results.append((int(idx), float(display_scores[idx])))

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
