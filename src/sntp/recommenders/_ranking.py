"""
Shared top-N selection utilities.

Single-user path
----------------
:func:`top_n_from_scores` ranks one full-catalogue score vector (used by each
recommender's ``recommend``).

Batched path (used by ranking evaluation)
-----------------------------------------
:func:`top_n_batch_from_candidates` and :func:`top_n_batch_dense` rank many
users at once without per-item Python loops.

Selection semantics are identical everywhere:

  * excluded items (e.g. training items, ineligible items) are never returned;
  * NaN / -inf scores are never returned;
  * items are ordered by score descending;
  * ties are broken by ascending item index — identical to a stable sort of
    ``[(item_idx, score) for item_idx in range(n_items)]`` by ``-score``.
"""

import numpy as np
from scipy import sparse


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


# ---------------------------------------------------------------------------
# Batched selection
# ---------------------------------------------------------------------------

def top_n_batch_from_candidates(rows: np.ndarray, cols: np.ndarray,
                                vals: np.ndarray, n_rows: int,
                                n: int) -> list[list[tuple[int, float]]]:
    """Top-``n`` per row from explicit ``(row, item, score)`` candidate triples.

    Each (row, item) pair must appear at most once.  Candidates with NaN or
    ``-inf`` scores are dropped.  Returns one ranked list per row.
    """
    out: list[list[tuple[int, float]]] = [[] for _ in range(n_rows)]
    if n <= 0 or n_rows == 0:
        return out
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    ok = vals > -np.inf          # also False for NaN
    if not ok.all():
        rows, cols, vals = rows[ok], cols[ok], vals[ok]
    if rows.size == 0:
        return out

    # Sort by row, then score descending, then item index ascending.
    order = np.lexsort((cols, -vals, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    starts = np.searchsorted(rows, np.arange(n_rows), side="left")
    rank = np.arange(rows.size) - starts[rows]
    keep = rank < n
    rows, cols, vals = rows[keep], cols[keep], vals[keep]

    bounds = np.searchsorted(rows, np.arange(n_rows + 1), side="left")
    cols_l = cols.tolist()
    vals_l = vals.tolist()
    for r in range(n_rows):
        a, b = bounds[r], bounds[r + 1]
        if b > a:
            out[r] = list(zip(cols_l[a:b], vals_l[a:b]))
    return out


def top_n_batch_dense(scores: np.ndarray, n: int) -> list[list[tuple[int, float]]]:
    """Top-``n`` per row of a dense ``(n_users, n_items)`` score matrix.

    Entries that must not be returned should already be ``-inf`` (NaN is
    also treated as ``-inf``).  **Modifies** ``scores`` in place for NaNs.
    """
    n_rows, n_items = scores.shape
    if n <= 0 or n_items == 0 or n_rows == 0:
        return [[] for _ in range(n_rows)]
    nan_mask = np.isnan(scores)
    if nan_mask.any():
        scores[nan_mask] = -np.inf
    n_eff = min(n, n_items)
    # n_eff-th largest value per row; everything >= it is a candidate
    # (this keeps *all* ties so the index tie-break stays exact).
    kth = np.partition(scores, n_items - n_eff, axis=1)[:, n_items - n_eff]
    mask = scores >= kth[:, None]
    mask &= scores > -np.inf
    r, c = np.nonzero(mask)
    return top_n_batch_from_candidates(r, c, scores[r, c], n_rows, n_eff)


def build_exclusion_mask(user_indices, n_items: int, excludes=None,
                         internal_lookup=None, candidate_mask=None) -> np.ndarray:
    """Dense boolean ``(n_users, n_items)`` mask; True = must not be recommended.

    Combines (a) items outside ``candidate_mask`` (ineligible catalogue items),
    (b) per-user ``excludes[r]`` and (c) ``internal_lookup[user]`` (the
    recommender's own record of the user's training items).
    """
    b = len(user_indices)
    if candidate_mask is None:
        ex = np.zeros((b, n_items), dtype=bool)
    else:
        cm = np.asarray(candidate_mask, dtype=bool)
        if cm.shape[0] != n_items:
            raise ValueError(f"candidate_mask has length {cm.shape[0]}, expected {n_items}")
        ex = np.broadcast_to(~cm, (b, n_items)).copy()
    rr, cc = exclusion_pairs(user_indices, n_items, excludes, internal_lookup)
    ex[rr, cc] = True
    return ex


def exclusion_pairs(user_indices, n_items: int, excludes=None, internal_lookup=None):
    """(row, item) index arrays of per-user exclusions (training items etc.)."""
    rows_parts, cols_parts = [], []
    for r, u in enumerate(user_indices):
        for src in ((excludes[r] if excludes is not None else None),
                    (internal_lookup.get(int(u)) if internal_lookup is not None else None)):
            if src:
                idx = np.fromiter(src, dtype=np.int64, count=len(src))
                idx = idx[(idx >= 0) & (idx < n_items)]
                rows_parts.append(np.full(idx.shape[0], r, dtype=np.int64))
                cols_parts.append(idx)
    if not rows_parts:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    return np.concatenate(rows_parts), np.concatenate(cols_parts)


def recommend_batch_dense(score_batch_fn, user_indices, n: int, n_items: int,
                          excludes=None, internal_lookup=None,
                          candidate_mask=None) -> list[list[tuple[int, float]]]:
    """Generic batched recommend for models that score every item (e.g. MF)."""
    scores = score_batch_fn(user_indices)
    ex = build_exclusion_mask(user_indices, n_items, excludes, internal_lookup,
                              candidate_mask)
    scores[ex] = -np.inf
    return top_n_batch_dense(scores, n)


def rows_or_empty(mat: sparse.csr_matrix, idx: np.ndarray) -> sparse.csr_matrix:
    """``mat[idx]`` where out-of-range row indices yield empty rows."""
    idx = np.asarray(idx, dtype=np.int64)
    inr = (idx >= 0) & (idx < mat.shape[0])
    if inr.all():
        return mat[idx]
    safe = np.where(inr, idx, 0)
    sub = sparse.csr_matrix(mat[safe])
    keep = np.repeat(inr, np.diff(sub.indptr))
    sub.data = np.where(keep, sub.data, 0)
    sub.eliminate_zeros()
    return sub
