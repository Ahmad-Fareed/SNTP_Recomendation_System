"""
Item-based collaborative filtering.

Algorithm:
  1. Compute item-item similarity using Adjusted Cosine Similarity on co-rated users
     from training data only.
  2. For each item, store only the top-K most similar neighbors to keep memory manageable.
  3. Predict rating for (user u, item i):
       pred(u, i) = sum_{j in N(i) ∩ rated_by(u)} sim(i,j) * r(u,j)
                    / sum_{j in N(i) ∩ rated_by(u)} |sim(i,j)|
     where N(i) are the stored neighbors of item i.
  4. If no usable neighbors exist, fall back to the user's mean rating,
     then to the global mean.

Similarity formula — Adjusted Cosine:
  sim(i, j) = sum_{u in co-rated} (r(u,i) - mean(u)) * (r(u,j) - mean(u))
              / (||r(·,i) - mean(·)||_2 * ||r(·,j) - mean(·)||_2)

  This removes user-specific rating bias before computing cosine similarity.

Implementation notes:
  - Similarities are computed with blocked sparse matrix products
    (C^T C on the user-mean-centred training matrix C, and B^T B on its binary
    pattern for co-rated counts) instead of a Python loop over user item pairs.
  - Neighbour selection keeps the ``max_neighbors`` largest |sim| per item;
    ties in |sim| are broken by ascending neighbour index (deterministic).
    Pairs whose dot product is exactly 0 (sim = 0) are not stored: they
    contribute 0 to both numerator and denominator, so predictions are
    unaffected.

Ranking candidates (``recommend_batch``):
  - Items reachable through the neighbour graph from the user's rated items
    are scored with the weighted average; every other eligible item has the
    fallback score.  Only neighbour-reachable items plus the ``n`` lowest-index
    fallback items are materialised, which yields exactly the same top-n as
    scoring the whole catalogue.
  - An item with no stored neighbours is **unsupported by the item-item
    model**.  Its documented fallback uses training data only: the user's mean
    training rating, then the global training mean for a user with no training
    ratings.  Held-out ratings are never consulted.
  - Items are ordered by raw model scores.  The predicted rating returned with
    a recommendation is clipped to [1, 5] only after selection.

Limitations:
  - Requires co-rated users; items with very few ratings have unreliable similarities.
  - Cold-start items (no training ratings) cannot be recommended.
  - K-neighbor limit may miss some relevant similarities.
"""

import logging
import time

import numpy as np
from scipy import sparse

from sntp.recommenders._ranking import (
    clip_ranked, exclusion_pairs, rows_or_empty, top_n_batch_from_candidates,
    top_n_from_scores,
)

logger = logging.getLogger(__name__)


class ItemBasedCF:
    """Item-based collaborative filtering with adjusted cosine similarity."""

    def __init__(self, max_neighbors: int = 50, min_corated: int = 2,
                 block_size: int = 8192):
        """
        Parameters
        ----------
        max_neighbors : int
            Maximum number of similar items stored per item.
        min_corated : int
            Minimum co-rated users required to compute a similarity.
        block_size : int
            Number of target items per sparse-product block in ``fit``
            (bounds peak memory; does not affect results).
        """
        self.max_neighbors = max_neighbors
        self.min_corated = min_corated
        self.block_size = block_size
        self.similarities = {}   # item_idx -> [(neighbor_idx, sim), ...]
        self.user_means = None
        self.global_mean = 0.0
        self.train_user_ratings = {}  # user_idx -> {item_idx: rating}
        self.n_items = 0
        # Sparse (target_item x neighbor_item) matrices of the stored top-K
        # similarities, in CSC so a user's rated-item columns slice cheaply.
        self._sim_csc = None
        self._abs_sim_csc = None
        # CSR copies (row = target item) and transposes for batched scoring.
        self._sim_csr = None
        self._abs_sim_csr = None
        self._simT_csr = None
        self._abs_simT_csr = None
        # Training rating matrix (float64) and its binary pattern.
        self._R_csr = None
        self._B_csr = None
        self._user_counts = None
        # Per-user arrays for vectorized scoring.
        self._user_items_arr = {}   # user_idx -> np.ndarray[int]
        self._user_ratings_arr = {}  # user_idx -> np.ndarray[float]

    def fit(self, rating_matrix: sparse.csr_matrix, user_ratings: dict, item_ratings: dict):
        """Compute item-item similarities from training data.

        Parameters
        ----------
        rating_matrix : sparse.csr_matrix
            (n_users x n_items) training rating matrix.
        user_ratings : dict[int, list[tuple[int, int]]]
            user_idx -> [(item_idx, rating), ...]
        item_ratings : dict[int, list[tuple[int, int]]]
            item_idx -> [(user_idx, rating), ...]
            (kept for API compatibility; the same data is read from
            ``rating_matrix``).
        """
        t0 = time.time()
        R = sparse.csr_matrix(rating_matrix, dtype=np.float64, copy=True)
        R.sum_duplicates()
        R.sort_indices()
        R.eliminate_zeros()
        n_users, self.n_items = R.shape
        n_items = self.n_items

        # User means over observed ratings only.
        counts = np.diff(R.indptr)
        sums = np.asarray(R.sum(axis=1)).ravel()
        self.user_means = np.zeros(n_users, dtype=np.float64)
        has = counts > 0
        self.user_means[has] = sums[has] / counts[has]
        self._user_counts = counts

        self.global_mean = float(rating_matrix.data.mean()) if rating_matrix.nnz > 0 else 3.0

        # Build user -> {item: rating} lookup for fast prediction
        self.train_user_ratings = {}
        for uidx, items in user_ratings.items():
            self.train_user_ratings[uidx] = {iidx: r for iidx, r in items}

        # Mean-centred matrix C and binary co-rating pattern B.
        row_of = np.repeat(np.arange(n_users), counts)
        adj = R.data - self.user_means[row_of]
        C = sparse.csr_matrix((adj, R.indices.copy(), R.indptr.copy()), shape=R.shape)
        B32 = sparse.csr_matrix((np.ones(R.nnz, dtype=np.float32), R.indices.copy(),
                                 R.indptr.copy()), shape=R.shape)
        norms = np.sqrt(np.bincount(R.indices, weights=adj * adj, minlength=n_items))
        item_pop = np.bincount(R.indices, minlength=n_items)
        n_rated_items = int((item_pop > 0).sum())
        logger.info("Computing item-item similarities for %d items (blocked sparse products)...",
                    n_rated_items)

        Ct = C.T.tocsr()      # items x users
        Bt = B32.T.tocsr()
        K = self.max_neighbors
        rows_parts, cols_parts, sims_parts = [], [], []
        t_last = time.time()
        for a in range(0, n_items, self.block_size):
            b = min(a + self.block_size, n_items)
            if item_pop[a:b].sum() == 0:
                continue
            dots = (Ct[a:b] @ C).tocoo()         # sum of adjusted products
            cnt = (Bt[a:b] @ B32).tocsr()         # co-rated user counts
            r_loc, c = dots.row.astype(np.int64), dots.col.astype(np.int64)
            d = dots.data
            r = r_loc + a
            # Co-rated counts at the same (row, col) positions.
            cnt_vals = np.asarray(cnt[r_loc, c]).ravel()
            ok = (cnt_vals >= self.min_corated) & (r != c)
            ok &= (norms[r] >= 1e-10) & (norms[c] >= 1e-10)
            r, c, d = r[ok], c[ok], d[ok]
            s = d / (norms[r] * norms[c])
            if r.size:
                # Top-K by |sim| desc, ties -> ascending neighbour index.
                order = np.lexsort((c, -np.abs(s), r))
                r, c, s = r[order], c[order], s[order]
                starts = np.searchsorted(r, r, side="left")
                keep = (np.arange(r.size) - starts) < K
                rows_parts.append(r[keep])
                cols_parts.append(c[keep])
                sims_parts.append(s[keep])
            if time.time() - t_last >= 15:
                logger.info("  item similarity blocks: %d/%d items done (%.0fs)",
                            b, n_items, time.time() - t0)
                t_last = time.time()

        if rows_parts:
            rows = np.concatenate(rows_parts)
            cols = np.concatenate(cols_parts)
            sims = np.concatenate(sims_parts)
        else:
            rows = cols = np.zeros(0, dtype=np.int64)
            sims = np.zeros(0, dtype=np.float64)

        # Neighbour lists (already ordered by |sim| desc, index asc).
        self.similarities = {}
        bounds = np.searchsorted(rows, np.arange(n_items + 1), side="left")
        cols_l, sims_l = cols.tolist(), sims.tolist()
        for iidx in np.flatnonzero(np.diff(bounds)).tolist():
            lo, hi = bounds[iidx], bounds[iidx + 1]
            self.similarities[iidx] = list(zip(cols_l[lo:hi], sims_l[lo:hi]))

        self._R_csr = R
        self._B_csr = sparse.csr_matrix((np.ones(R.nnz), R.indices, R.indptr), shape=R.shape)
        self._build_scoring_structures(rows, cols, sims)

        elapsed = time.time() - t0
        logger.info(
            "Item-based CF fitted in %.1fs. %d / %d items have neighbors (%d stored pairs).",
            elapsed, len(self.similarities), n_rated_items, sims.size,
        )

    def predict(self, user_idx: int, item_idx: int,
                rating_min: float = 1.0, rating_max: float = 5.0) -> float:
        """Predict rating for a single (user, item) pair.

        Fallback chain:
          1. Weighted average of similar items rated by the user.
          2. User's mean training rating.
          3. Global mean.
        """
        user_items = self.train_user_ratings.get(user_idx, {})
        neighbors = self.similarities.get(item_idx, [])

        num = 0.0
        den = 0.0
        for neighbor_idx, sim in neighbors:
            if neighbor_idx in user_items:
                num += sim * user_items[neighbor_idx]
                den += abs(sim)

        if den > 1e-10:
            pred = num / den
        elif user_items:
            pred = float(np.mean(list(user_items.values())))
        else:
            pred = self.global_mean

        return float(np.clip(pred, rating_min, rating_max))

    def _build_scoring_structures(self, rows, cols, sims):
        """Build sparse similarity matrices and per-user arrays for scoring.

        S[i, j] = sim(i, j) for every stored neighbor j of target item i, so
        for a user u with rated items J_u and ratings r_u:
            num = S[:, J_u] @ r_u      (= sum_j sim(i,j) * r(u,j)  for all i)
            den = |S|[:, J_u] @ 1      (= sum_j |sim(i,j)|         for all i)
        which is exactly the per-item computation in predict().
        """
        shape = (self.n_items, self.n_items)
        sim_csr = sparse.csr_matrix(
            (np.asarray(sims, dtype=np.float64),
             (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
            shape=shape,
        )
        self._sim_csr = sim_csr
        self._abs_sim_csr = abs(sim_csr)
        self._sim_csc = sim_csr.tocsc()
        self._abs_sim_csc = abs(self._sim_csc)
        self._simT_csr = sim_csr.T.tocsr()
        self._abs_simT_csr = self._abs_sim_csr.T.tocsr()

        self._user_items_arr = {}
        self._user_ratings_arr = {}
        for uidx, items in self.train_user_ratings.items():
            if items:
                self._user_items_arr[uidx] = np.fromiter(items.keys(), dtype=np.int64,
                                                         count=len(items))
                self._user_ratings_arr[uidx] = np.fromiter(items.values(), dtype=np.float64,
                                                           count=len(items))

    def score_all(self, user_idx: int, rating_min: float = 1.0,
                  rating_max: float = 5.0,
                  raw_scores: bool = False) -> np.ndarray:
        """Predicted rating for *every* catalogue item for one user (vectorized).

        Applies the same formula and fallback chain as :meth:`predict`:
        neighbor-weighted average -> training-only user mean -> training-only
        global mean.  ``raw_scores=False`` clips the returned ratings to the
        rating scale; ``raw_scores=True`` returns the unclipped scores for
        ranking.
        """
        items = self._user_items_arr.get(user_idx)
        if items is None:
            value = float(self.global_mean)
            if not raw_scores:
                value = float(np.clip(value, rating_min, rating_max))
            return np.full(self.n_items, value, dtype=np.float64)

        ratings = self._user_ratings_arr[user_idx]
        num = self._sim_csc[:, items] @ ratings
        den = np.asarray(self._abs_sim_csc[:, items].sum(axis=1)).ravel()

        user_mean = float(np.mean(ratings))
        scores = np.full(self.n_items, user_mean, dtype=np.float64)
        has_nb = den > 1e-10
        scores[has_nb] = num[has_nb] / den[has_nb]
        if not raw_scores:
            np.clip(scores, rating_min, rating_max, out=scores)
        return scores

    def recommend(self, user_idx: int, n: int = 10,
                  exclude: set | None = None) -> list[tuple[int, float]]:
        """Recommend top-n items not in exclude set.

        Every catalogue item is scored (full-catalogue ranking).  Items are
        ordered by raw model scores; the reported predicted rating is clipped
        to [1, 5] after selection.  Ties are broken by ascending item index.
        """
        if exclude is None:
            exclude = set()
        # Add training items to exclusion
        train_items = set(self.train_user_ratings.get(user_idx, {}).keys())
        exclude = exclude | train_items

        raw_scores = self.score_all(user_idx, raw_scores=True)
        ranked = top_n_from_scores(raw_scores, n, exclude)
        return clip_ranked(ranked)

    # ------------------------------------------------------------------
    # Batched / vectorized paths
    # ------------------------------------------------------------------

    def _fallback_values(self, users: np.ndarray, rating_min: float,
                         rating_max: float) -> np.ndarray:
        """Per-user training-only fallback (user mean, else global mean).

        Clipped to the supplied bounds.  Pass ``(-inf, inf)`` when the values
        are needed for raw-score ordering.
        """
        users = np.asarray(users, dtype=np.int64)
        n_rows = self.user_means.shape[0]
        inr = (users >= 0) & (users < n_rows)
        fb = np.full(users.shape[0], self.global_mean, dtype=np.float64)
        safe = np.where(inr, users, 0)
        has = inr & (self._user_counts[safe] > 0)
        fb[has] = self.user_means[users[has]]
        return np.clip(fb, rating_min, rating_max)

    def predict_batch(self, user_idx: np.ndarray, item_idx: np.ndarray,
                      rating_min: float = 1.0, rating_max: float = 5.0) -> np.ndarray:
        """Vectorized :meth:`predict` for aligned arrays of (user, item) pairs."""
        u = np.asarray(user_idx, dtype=np.int64)
        i = np.asarray(item_idx, dtype=np.int64)
        if u.size == 0:
            return np.zeros(0, dtype=np.float64)
        R_rows = rows_or_empty(self._R_csr, u)
        B_rows = rows_or_empty(self._B_csr, u)
        num = np.asarray(rows_or_empty(self._sim_csr, i).multiply(R_rows).sum(axis=1)).ravel()
        den = np.asarray(rows_or_empty(self._abs_sim_csr, i).multiply(B_rows).sum(axis=1)).ravel()
        # Unclipped user-mean / global-mean fallback (clipped below, as predict()).
        fb = self._fallback_values(u, -np.inf, np.inf)
        has_nb = den > 1e-10
        pred = fb
        pred[has_nb] = num[has_nb] / den[has_nb]
        return np.clip(pred, rating_min, rating_max)

    def score_batch(self, user_indices, rating_min: float = 1.0,
                    rating_max: float = 5.0, raw_scores: bool = False) -> np.ndarray:
        """Dense ``(n_users, n_items)`` version of :meth:`score_all`.

        ``raw_scores=True`` returns unclipped scores for ranking.
        """
        users = np.asarray(user_indices, dtype=np.int64)
        num = (rows_or_empty(self._R_csr, users) @ self._simT_csr).toarray()
        den = (rows_or_empty(self._B_csr, users) @ self._abs_simT_csr).toarray()
        fb = self._fallback_values(users, -np.inf, np.inf)
        scores = np.broadcast_to(fb[:, None], num.shape).copy()
        has_nb = den > 1e-10
        scores[has_nb] = num[has_nb] / den[has_nb]
        if raw_scores:
            return scores
        np.clip(scores, rating_min, rating_max, out=scores)
        return scores

    def recommend_batch(self, user_indices, n: int = 10, excludes=None,
                        candidate_mask: np.ndarray | None = None,
                        rating_min: float = 1.0, rating_max: float = 5.0):
        """Top-n for many users using sparse candidate generation.

        Candidates are (a) items reachable through the neighbour graph from
        the user's rated items (scored by the weighted average) and (b) the
        ``n`` lowest-index remaining eligible items, which all carry the
        fallback score.  Because every non-reachable item has the identical
        fallback score and ties are broken by ascending index, this returns
        exactly the same lists as full-catalogue scoring restricted to
        ``candidate_mask`` (all items when None).

        Items are ordered by raw scores, then each reported predicted rating
        is clipped to ``[rating_min, rating_max]``.  An item with no stored
        neighbours uses the documented training-only fallback: user mean, then
        global training mean.

        Excluded: ``excludes[r]`` for each user plus the user's training items.
        """
        users = np.asarray(user_indices, dtype=np.int64)
        b = users.shape[0]
        n_items = self.n_items
        if b == 0 or n <= 0:
            return [[] for _ in range(b)]
        cm = (np.ones(n_items, dtype=bool) if candidate_mask is None
              else np.asarray(candidate_mask, dtype=bool))

        # (row, item) pairs that must never be returned.
        ex_r, ex_c = exclusion_pairs(users, n_items, excludes, self.train_user_ratings)
        ex_keys = np.unique(ex_r * n_items + ex_c)

        # (a) neighbour-reachable candidates, scored on the sparse structure.
        den = (rows_or_empty(self._B_csr, users) @ self._abs_simT_csr).tocsr()
        num = (rows_or_empty(self._R_csr, users) @ self._simT_csr).tocsr()
        den.sort_indices()
        num.sort_indices()
        den_r = np.repeat(np.arange(b, dtype=np.int64), np.diff(den.indptr))
        den_keys = den_r * n_items + den.indices
        num_r = np.repeat(np.arange(b, dtype=np.int64), np.diff(num.indptr))
        num_keys = num_r * n_items + num.indices
        num_on_den = np.zeros(den.nnz, dtype=np.float64)
        if num.nnz:
            # num's sparsity pattern is a subset of den's (sim != 0 => |sim| > 0).
            num_on_den[np.searchsorted(den_keys, num_keys)] = num.data
        reach = den.data > 1e-10
        cand_ok = reach & cm[den.indices] & ~np.isin(den_keys, ex_keys)
        c_rows = den_r[cand_ok]
        c_cols = den.indices[cand_ok].astype(np.int64)
        # Raw scores: clipping happens only after top-n selection.
        c_vals = num_on_den[cand_ok] / den.data[cand_ok]

        # (b) fallback items: n lowest-index eligible items that are neither
        # excluded nor neighbour-reachable (they all share the fallback score).
        # The fallback is training-only: user mean, else global training mean.
        fb = self._fallback_values(users, -np.inf, np.inf)
        eligible = np.flatnonzero(cm)
        reach_keys = den_keys[reach]
        blocked_keys = np.union1d(ex_keys, reach_keys)
        blk_r = blocked_keys // n_items
        blk_bounds = np.searchsorted(blk_r, np.arange(b + 1))
        f_rows, f_cols = [], []
        for r in range(b):
            blocked = blocked_keys[blk_bounds[r]:blk_bounds[r + 1]] - r * n_items
            prefix = eligible[:n + blocked.shape[0]]
            picked = prefix[~np.isin(prefix, blocked, assume_unique=True)][:n]
            f_rows.append(np.full(picked.shape[0], r, dtype=np.int64))
            f_cols.append(picked)
        f_rows = np.concatenate(f_rows) if f_rows else np.zeros(0, dtype=np.int64)
        f_cols = np.concatenate(f_cols) if f_cols else np.zeros(0, dtype=np.int64)

        rows = np.concatenate([c_rows, f_rows])
        cols = np.concatenate([c_cols, f_cols])
        vals = np.concatenate([c_vals, fb[f_rows]])
        ranked = top_n_batch_from_candidates(rows, cols, vals, b, n)
        return [clip_ranked(rows_i, rating_min, rating_max) for rows_i in ranked]
