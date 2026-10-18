"""
Shared top-N selection over a full-catalogue score vector.

All recommenders score *every* catalogue item (full-catalogue ranking; no
candidate sampling) and then call :func:`top_n_from_scores`.  Selection is
exact and deterministic:

  * excluded items (e.g. training items) are never returned;
  * items are ordered by score descending;
  * ties are broken by ascending item index — identical to a stable sort of
    ``[(item_idx, score) for item_idx in range(n_items)]`` by ``-score``,
    which is what the original per-item Python loops did.

Uses ``np.partition`` (O(n_items)) instead of a full sort (O(n log n)).
"""

import numpy as np


def top_n_from_scores(scores: np.ndarray, n: int,
                      exclude=None) -> list[tuple[int, float]]:
    """Return the top-``n`` ``(item_idx, score)`` pairs, excluding ``exclude``.

    Parameters
    ----------
    scores : np.ndarray, shape (n_items,)
        Score for every catalogue item.  **Modified in place** (excluded
        entries are set to ``-inf``); pass a copy if you need the original.
    n : int
        Number of items to return.
    exclude : iterable[int] or None
        Item indices that must not be recommended.
    """
    n_items = scores.shape[0]
    if n <= 0 or n_items == 0:
        return []

    if exclude:
        ex = np.fromiter(exclude, dtype=np.int64, count=len(exclude))
        ex = ex[(ex >= 0) & (ex < n_items)]
        scores[ex] = -np.inf
    # Never return NaN-scored items (treated like "no prediction").
    nan_mask = np.isnan(scores)
    if nan_mask.any():
        scores[nan_mask] = -np.inf

    n_eff = min(n, n_items)
    # Value of the n_eff-th largest score.
    kth = np.partition(scores, n_items - n_eff)[n_items - n_eff]

    if kth == -np.inf:
        # Fewer than n valid candidates: return all valid ones.
        valid = np.flatnonzero(scores > -np.inf)
        order = valid[np.argsort(-scores[valid], kind="stable")]
    else:
        above = np.flatnonzero(scores > kth)          # ascending index order
        above = above[np.argsort(-scores[above], kind="stable")]
        need = n_eff - above.shape[0]
        ties = np.flatnonzero(scores == kth)[:need]   # lowest indices first
        order = np.concatenate([above, ties])

    return [(int(i), float(scores[i])) for i in order]
