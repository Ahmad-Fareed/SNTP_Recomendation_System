"""
Social-enhanced recommendation: trust-neighbor extension for a baseline recommender.

Model:
  social_prediction(u, i) = (1 - λ) * baseline_prediction(u, i)
                           + λ * trust_neighbor_prediction(u, i)

  trust_neighbor_prediction(u, i) =
      mean of ratings that u's trusted users gave to item i
      (using only training ratings).

  λ (social_weight) controls the contribution of social information.
  At λ = 0, this reduces exactly to the baseline recommender.

Fallback behavior:
  - If no trusted neighbor has rated item i: use baseline prediction only (ignore λ).
  - If baseline prediction is unavailable: use trust neighbor prediction only.
  - If neither is available: user mean → global mean (same as baseline fallback).

Limitations:
  - Trust does NOT imply identical taste. Two users connected by trust may
    have very different preferences for specific items.
  - Trust edges in Epinions are binary (present/absent) — we do not have
    measured trust strengths. The social weight λ is a tuning parameter,
    not a measured quantity.
  - Correlation between trust and rating agreement should be empirically
    verified, not assumed.
"""

import logging
import time
from collections import defaultdict

import numpy as np
from scipy import sparse

from sntp.recommenders._ranking import (
    clip_ranked, recommend_batch_dense, rows_or_empty, top_n_from_scores,
)

logger = logging.getLogger(__name__)


def _baseline_scores(baseline, user_idx: int, n_items: int,
                     raw_scores: bool = False) -> np.ndarray:
    """Baseline prediction for every item; NaN where the baseline returns None.

    Uses the baseline's vectorized ``score_all`` when available, otherwise
    falls back to calling ``predict`` per item.  ``raw_scores=True`` requests
    the baseline's unclipped ranking score; baselines that do not expose that
    option are used through their normal (clipped) public score.
    """
    if raw_scores and hasattr(baseline, "ranking_score_all"):
        values = baseline.ranking_score_all(user_idx)
        return np.asarray(values, dtype=np.float64).copy()
    if hasattr(baseline, "score_all"):
        try:
            values = baseline.score_all(user_idx, raw_scores=raw_scores)
        except TypeError:
            values = baseline.score_all(user_idx)
        return np.asarray(values, dtype=np.float64).copy()
    out = np.empty(n_items, dtype=np.float64)
    for iidx in range(n_items):
        p = baseline.predict(user_idx, iidx)
        out[iidx] = np.nan if p is None else p
    return out


def _baseline_scores_batch(baseline, users: np.ndarray, n_items: int,
                           raw_scores: bool = False) -> np.ndarray:
    """Dense baseline scores for many users (NaN where unavailable)."""
    if raw_scores and hasattr(baseline, "ranking_score_batch"):
        values = baseline.ranking_score_batch(users)
        return np.asarray(values, dtype=np.float64)
    if hasattr(baseline, "score_batch"):
        try:
            values = baseline.score_batch(users, raw_scores=raw_scores)
        except TypeError:
            values = baseline.score_batch(users)
        return np.asarray(values, dtype=np.float64)
    return np.vstack([
        _baseline_scores(baseline, int(u), n_items, raw_scores=raw_scores)
        for u in users
    ])


def _baseline_predict_pairs(baseline, u: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Baseline predictions for aligned (user, item) arrays (NaN if None)."""
    if hasattr(baseline, "predict_batch"):
        return np.asarray(baseline.predict_batch(u, i), dtype=np.float64)
    out = np.empty(u.shape[0], dtype=np.float64)
    for k, (uu, ii) in enumerate(zip(u.tolist(), i.tolist())):
        p = baseline.predict(uu, ii)
        out[k] = np.nan if p is None else p
    return out


def _build_rating_csr(user_ratings: dict, n_items: int) -> sparse.csr_matrix:
    """user_idx -> {item_idx: rating} dict  ->  CSR (n_rows x n_items)."""
    rows, cols, vals = [], [], []
    for u, items in user_ratings.items():
        for i, r in items.items():
            rows.append(u)
            cols.append(i)
            vals.append(float(r))
    n_rows = (max(user_ratings.keys()) + 1) if user_ratings else 0
    if cols:
        n_items = max(n_items, max(cols) + 1)
    return sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n_rows, n_items),
    )


def _build_adjacency(neighbor_lists: dict, n_cols: int, min_rows: int = 0) -> sparse.csr_matrix:
    """user -> [neighbour, ...]  ->  CSR adjacency (n_rows x n_cols).

    Neighbours outside ``[0, n_cols)`` (e.g. trust-only users with no rating
    row) are dropped, exactly like ``_neighbor_mean_scores``.  Repeated
    neighbours are summed, which matches the per-item loop counting them twice.
    """
    rows, cols = [], []
    for u, nbs in neighbor_lists.items():
        for v in nbs:
            if 0 <= v < n_cols:
                rows.append(u)
                cols.append(v)
    n_rows = max([min_rows] + [u + 1 for u in neighbor_lists.keys()])
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n_rows, n_cols),
    )


def _neighbor_mean_scores(rating_csr, neighbors, n_items: int):
    """Mean training rating of ``neighbors`` for every item.

    Returns (sums, counts) arrays of length n_items; items no neighbor rated
    have count 0.  Neighbors are processed in the given order, so the sum
    matches the per-item Python loop in ``_trust_neighbor_prediction``.
    """
    if rating_csr is None or not len(neighbors):
        return None, None
    nb = np.asarray(neighbors, dtype=np.int64)
    nb = nb[(nb >= 0) & (nb < rating_csr.shape[0])]
    if nb.size == 0:
        return None, None
    sub = rating_csr[nb]
    sums = np.bincount(sub.indices, weights=sub.data, minlength=n_items)[:n_items]
    counts = np.bincount(sub.indices, minlength=n_items)[:n_items]
    return sums, counts


def _neighbor_stats_batch(adj, rating_csr, binary_csr, users, n_items: int):
    """Dense (sums, counts) of neighbours' training ratings for many users."""
    A = rows_or_empty(adj, users)
    sums = (A @ rating_csr).toarray()[:, :n_items]
    counts = (A @ binary_csr).toarray()[:, :n_items]
    return sums, counts


def _neighbor_stats_pairs(adj, rating_csc, binary_csc, u: np.ndarray, i: np.ndarray):
    """(sums, counts) of neighbours' ratings for aligned (user, item) pairs."""
    A = rows_or_empty(adj, u)                              # pairs x n_rating_rows
    i_ok = (i >= 0) & (i < rating_csc.shape[1])
    i_safe = np.where(i_ok, i, 0)
    Rt = sparse.csr_matrix(rating_csc[:, i_safe].T)        # pairs x n_rating_rows
    Bt = sparse.csr_matrix(binary_csc[:, i_safe].T)
    sums = np.asarray(A.multiply(Rt).sum(axis=1)).ravel()
    counts = np.asarray(A.multiply(Bt).sum(axis=1)).ravel()
    sums[~i_ok] = 0.0
    counts[~i_ok] = 0.0
    return sums, counts


def _blend_scores(base: np.ndarray, sums, counts, social_weight: float,
                  global_mean: float, clip_scores: bool = True) -> np.ndarray:
    """Vectorized version of the predict() blending / fallback rules.

    ``clip_scores=False`` preserves the raw blended scores for ranking.  The
    caller clips only the reported ratings after top-n selection.
    """
    scores = base
    base_nan = np.isnan(base)
    if sums is not None:
        has_nb = counts > 0
        nb_mean = np.zeros_like(base)
        nb_mean[has_nb] = sums[has_nb] / counts[has_nb]
        both = has_nb & ~base_nan
        combined = (1 - social_weight) * base[both] + social_weight * nb_mean[both]
        only_nb = has_nb & base_nan
        if clip_scores:
            combined = np.clip(combined, 1.0, 5.0)
            nb_mean = np.clip(nb_mean, 1.0, 5.0)
        scores[both] = combined
        scores[only_nb] = nb_mean[only_nb]
        base_nan = base_nan & ~has_nb
    scores[base_nan] = global_mean
    return scores


class _NeighborBlendMixin:
    """Shared batched paths for recommenders that blend a baseline with the
    mean training rating of a per-user neighbour set (``self._adj``)."""

    def _prepare_matrices(self, user_ratings: dict):
        self._rating_csr = _build_rating_csr(user_ratings, getattr(self.baseline, "n_items", 0))
        R = self._rating_csr
        self._binary_csr = sparse.csr_matrix(
            (np.ones(R.nnz, dtype=np.float64), R.indices, R.indptr), shape=R.shape)
        self._rating_csc = R.tocsc()
        self._binary_csc = self._binary_csr.tocsc()

    def score_batch(self, user_indices, raw_scores: bool = False) -> np.ndarray:
        """Dense ``(n_users, n_items)`` version of ``score_all``.

        ``raw_scores=True`` requests unclipped baseline and blended scores for
        ranking.  The public prediction path keeps ``raw_scores=False``.
        """
        users = np.asarray(user_indices, dtype=np.int64)
        n_items = self.baseline.n_items
        base = _baseline_scores_batch(
            self.baseline, users, n_items, raw_scores=raw_scores
        ).copy()
        sums, counts = _neighbor_stats_batch(self._adj, self._rating_csr,
                                             self._binary_csr, users, n_items)
        return _blend_scores(
            base, sums, counts, self.social_weight, self.global_mean,
            clip_scores=not raw_scores,
        )

    def predict_batch(self, user_idx: np.ndarray, item_idx: np.ndarray,
                      rating_min: float = 1.0, rating_max: float = 5.0) -> np.ndarray:
        """Vectorized ``predict`` for aligned arrays of (user, item) pairs."""
        u = np.asarray(user_idx, dtype=np.int64)
        i = np.asarray(item_idx, dtype=np.int64)
        base = _baseline_predict_pairs(self.baseline, u, i).copy()
        sums, counts = _neighbor_stats_pairs(self._adj, self._rating_csc,
                                             self._binary_csc, u, i)
        scores = _blend_scores(base, sums, counts, self.social_weight, self.global_mean)
        return np.clip(scores, rating_min, rating_max)

    def recommend_batch(self, user_indices, n: int = 10, excludes=None,
                        candidate_mask: np.ndarray | None = None):
        """Top-n for many users (vectorized; same rules as ``recommend``).

        Items are ordered by the configured raw blended scores; reported
        ratings are the full social predictions clipped to [1, 5] after
        selection.
        """
        users = np.asarray(user_indices, dtype=np.int64)
        ranked = recommend_batch_dense(
            lambda us: self.score_batch(us, raw_scores=True),
            users, n, self.baseline.n_items,
            excludes, self._train_user_items, candidate_mask,
        )
        # Ordering may use a bias-corrected baseline score; the returned value
        # is the full social prediction, clipped to the rating scale.
        full_scores = self.score_batch(users, raw_scores=False)
        return [
            clip_ranked([(item, full_scores[row_idx, item])
                         for item, _ in rows])
            for row_idx, rows in enumerate(ranked)
        ]


class SocialRecommender(_NeighborBlendMixin):
    """Wraps a baseline recommender with a trust-neighbor component."""

    def __init__(self, baseline, trust_graph: dict, social_weight: float = 0.3,
                 global_mean: float = 3.0):
        """
        Parameters
        ----------
        baseline : object
            Must have .predict(user_idx, item_idx) and .recommend(user_idx, n, exclude).
        trust_graph : dict[int, list[int]]
            trust_graph[u] = list of user indices that u trusts.
        social_weight : float
            λ in [0, 1]. At 0, purely baseline; at 1, purely social.
        global_mean : float
            Fallback value.
        """
        self.baseline = baseline
        self.trust_graph = trust_graph
        self.social_weight = social_weight
        self.global_mean = global_mean
        self.train_user_ratings = {}  # user_idx -> {item_idx: rating}
        self._train_user_items = {}
        self._rating_csr = None
        self._adj = None

    def set_training_data(self, user_ratings: dict):
        """Set training data for trust-neighbor lookup.

        Parameters
        ----------
        user_ratings : dict[int, dict[int, float]]
            user_idx -> {item_idx: rating} from training data only.
        """
        self.train_user_ratings = user_ratings
        self._train_user_items = {u: set(items.keys()) for u, items in user_ratings.items()}
        self._prepare_matrices(user_ratings)
        self._adj = _build_adjacency(self.trust_graph, self._rating_csr.shape[0],
                                     min_rows=getattr(self.baseline, "n_users", 0))

    def _trust_neighbor_prediction(self, user_idx: int, item_idx: int):
        """Predict using ratings of trusted users.

        Returns (prediction, count) or (None, 0) if no trusted neighbor rated the item.
        """
        trusted = self.trust_graph.get(user_idx, [])
        if not trusted:
            return None, 0

        ratings = []
        for t_idx in trusted:
            t_ratings = self.train_user_ratings.get(t_idx, {})
            if item_idx in t_ratings:
                ratings.append(t_ratings[item_idx])

        if not ratings:
            return None, 0

        return float(np.mean(ratings)), len(ratings)

    def predict(self, user_idx: int, item_idx: int) -> float | None:
        """Social-enhanced prediction.

        Combines baseline and trust-neighbor predictions using social_weight (λ).
        """
        base_pred = self.baseline.predict(user_idx, item_idx)
        trust_pred, n_trust = self._trust_neighbor_prediction(user_idx, item_idx)

        if trust_pred is not None and base_pred is not None:
            combined = (1 - self.social_weight) * base_pred + self.social_weight * trust_pred
            return float(np.clip(combined, 1.0, 5.0))
        elif trust_pred is not None:
            return float(np.clip(trust_pred, 1.0, 5.0))
        elif base_pred is not None:
            return base_pred
        else:
            return self.global_mean

    def score_all(self, user_idx: int, raw_scores: bool = False) -> np.ndarray:
        """Social-enhanced prediction for every catalogue item (vectorized).

        Same blending and fallback rules as :meth:`predict`.
        ``raw_scores=True`` preserves the unclipped values for ranking.
        """
        n_items = self.baseline.n_items
        base = _baseline_scores(
            self.baseline, user_idx, n_items, raw_scores=raw_scores
        )
        sums, counts = _neighbor_mean_scores(
            self._rating_csr, self.trust_graph.get(user_idx, []), n_items)
        return _blend_scores(
            base, sums, counts, self.social_weight, self.global_mean,
            clip_scores=not raw_scores,
        )

    def recommend(self, user_idx: int, n: int = 10,
                  exclude: set | None = None) -> list[tuple[int, float]]:
        """Recommend top-n items using social-enhanced predictions.

        Every catalogue item is scored; items are ordered by the configured
        raw blended scores.  The reported value is the full social prediction,
        clipped to [1, 5].  Ties are broken by ascending item index.
        """
        if exclude is None:
            exclude = set()
        train_items = self._train_user_items.get(user_idx, set())
        exclude = exclude | train_items

        ranked = top_n_from_scores(
            self.score_all(user_idx, raw_scores=True), n, exclude)
        full_scores = self.score_all(user_idx, raw_scores=False)
        return clip_ranked([(item, full_scores[item]) for item, _ in ranked])


class SimilarUserRecommender(_NeighborBlendMixin):
    """Control condition: uses top-K rating-similar (but NOT trusted) users.

    This serves as a comparison baseline to test whether trust neighbors
    provide more useful information than preference-similar unconnected users.
    Uses the same prediction formula as SocialRecommender but with similarity
    neighbors instead of trust neighbors.
    """

    def __init__(self, baseline, trust_graph: dict, social_weight: float = 0.3,
                 k_similar: int = 20, global_mean: float = 3.0):
        self.baseline = baseline
        self.trust_graph = trust_graph
        self.social_weight = social_weight
        self.k_similar = k_similar
        self.global_mean = global_mean
        self.train_user_ratings = {}
        self._train_user_items = {}
        self._rating_csr = None
        self._adj = None
        self._similar_users = {}  # user_idx -> [(similar_user_idx, similarity), ...]

    def set_training_data(self, user_ratings: dict):
        """Set training data and compute similar-but-unconnected users."""
        self.train_user_ratings = user_ratings
        self._train_user_items = {u: set(items.keys()) for u, items in user_ratings.items()}
        self._prepare_matrices(user_ratings)
        self._rebuild_adjacency()

    def _rebuild_adjacency(self):
        nb = {u: [s for s, _ in sims] for u, sims in self._similar_users.items()}
        self._adj = _build_adjacency(nb, self._rating_csr.shape[0],
                                     min_rows=getattr(self.baseline, "n_users", 0))

    def compute_similar_users(self, target_users: list[int]):
        """For each target user, find top-K similar users NOT in their trust network.

        Uses cosine similarity on co-rated items from training data:
        ratings are centred by each user's overall training mean, and the
        numerator / both norms are summed over co-rated items only; at least
        2 co-rated items are required.  Computed with sparse products against
        all users at once; ties are broken by ascending user index.
        """
        t0 = time.time()
        logger.info("Computing similar-but-unconnected users for %d targets...", len(target_users))

        R = self._rating_csr
        n_rows = R.shape[0]
        counts = np.diff(R.indptr)
        means = np.zeros(n_rows, dtype=np.float64)
        has = counts > 0
        means[has] = np.asarray(R.sum(axis=1)).ravel()[has] / counts[has]
        row_of = np.repeat(np.arange(n_rows), counts)
        C = sparse.csr_matrix((R.data - means[row_of], R.indices, R.indptr), shape=R.shape)
        C2 = C.multiply(C).tocsr()
        Bm = self._binary_csr
        Ct, C2t, Bt = C.T.tocsr(), C2.T.tocsr(), Bm.T.tocsr()

        for uidx in target_users:
            if not (0 <= uidx < n_rows) or counts[uidx] == 0:
                self._similar_users[uidx] = []
                continue
            cu, c2u, bu = Ct[:, [uidx]], C2t[:, [uidx]], Bt[:, [uidx]]
            num = (C @ cu).toarray().ravel()           # sum_c cu_c * cv_c
            den_u2 = (Bm @ c2u).toarray().ravel()       # sum_{c common} cu_c^2
            den_v2 = (C2 @ bu).toarray().ravel()        # sum_{c common} cv_c^2
            n_common = (Bm @ bu).toarray().ravel()
            den_u, den_v = np.sqrt(den_u2), np.sqrt(den_v2)

            valid = (n_common >= 2) & has & (den_u >= 1e-10) & (den_v >= 1e-10)
            valid[uidx] = False
            trusted = np.asarray(self.trust_graph.get(uidx, []), dtype=np.int64)
            trusted = trusted[(trusted >= 0) & (trusted < n_rows)]
            valid[trusted] = False
            cand = np.flatnonzero(valid)
            sims = num[cand] / (den_u[cand] * den_v[cand])
            order = np.lexsort((cand, -sims))[:self.k_similar]
            self._similar_users[uidx] = list(zip(cand[order].tolist(), sims[order].tolist()))

        self._rebuild_adjacency()
        logger.info("Similar-user search done in %.1fs.", time.time() - t0)

    def predict(self, user_idx: int, item_idx: int) -> float | None:
        """Predict using baseline + similar-unconnected-user component."""
        base_pred = self.baseline.predict(user_idx, item_idx)

        # Similar-user prediction
        similar = self._similar_users.get(user_idx, [])
        ratings = []
        for s_idx, sim in similar:
            s_ratings = self.train_user_ratings.get(s_idx, {})
            if item_idx in s_ratings:
                ratings.append(s_ratings[item_idx])

        if ratings and base_pred is not None:
            sim_pred = float(np.mean(ratings))
            combined = (1 - self.social_weight) * base_pred + self.social_weight * sim_pred
            return float(np.clip(combined, 1.0, 5.0))
        elif ratings:
            return float(np.clip(np.mean(ratings), 1.0, 5.0))
        elif base_pred is not None:
            return base_pred
        else:
            return self.global_mean

    def score_all(self, user_idx: int, raw_scores: bool = False) -> np.ndarray:
        """Similar-user-enhanced prediction for every item (vectorized predict).

        ``raw_scores=True`` preserves unclipped values for ranking.
        """
        n_items = self.baseline.n_items
        base = _baseline_scores(
            self.baseline, user_idx, n_items, raw_scores=raw_scores
        )
        similar = [s for s, _ in self._similar_users.get(user_idx, [])]
        sums, counts = _neighbor_mean_scores(self._rating_csr, similar, n_items)
        return _blend_scores(
            base, sums, counts, self.social_weight, self.global_mean,
            clip_scores=not raw_scores,
        )

    def recommend(self, user_idx: int, n: int = 10,
                  exclude: set | None = None) -> list[tuple[int, float]]:
        if exclude is None:
            exclude = set()
        train_items = self._train_user_items.get(user_idx, set())
        exclude = exclude | train_items

        ranked = top_n_from_scores(
            self.score_all(user_idx, raw_scores=True), n, exclude)
        full_scores = self.score_all(user_idx, raw_scores=False)
        return clip_ranked([(item, full_scores[item]) for item, _ in ranked])
