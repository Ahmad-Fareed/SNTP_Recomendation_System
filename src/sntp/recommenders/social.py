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
from collections import defaultdict

import numpy as np
from scipy import sparse

from sntp.recommenders._ranking import top_n_from_scores

logger = logging.getLogger(__name__)


def _baseline_scores(baseline, user_idx: int, n_items: int) -> np.ndarray:
    """Baseline prediction for every item; NaN where the baseline returns None.

    Uses the baseline's vectorized ``score_all`` when available, otherwise
    falls back to calling ``predict`` per item.
    """
    if hasattr(baseline, "score_all"):
        return np.asarray(baseline.score_all(user_idx), dtype=np.float64).copy()
    out = np.empty(n_items, dtype=np.float64)
    for iidx in range(n_items):
        p = baseline.predict(user_idx, iidx)
        out[iidx] = np.nan if p is None else p
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


def _blend_scores(base: np.ndarray, sums, counts, social_weight: float,
                  global_mean: float) -> np.ndarray:
    """Vectorized version of the predict() blending / fallback rules."""
    scores = base
    base_nan = np.isnan(base)
    if sums is not None:
        has_nb = counts > 0
        nb_mean = np.zeros_like(base)
        nb_mean[has_nb] = sums[has_nb] / counts[has_nb]
        both = has_nb & ~base_nan
        scores[both] = np.clip(
            (1 - social_weight) * base[both] + social_weight * nb_mean[both], 1.0, 5.0)
        only_nb = has_nb & base_nan
        scores[only_nb] = np.clip(nb_mean[only_nb], 1.0, 5.0)
        base_nan = base_nan & ~has_nb
    scores[base_nan] = global_mean
    return scores


class SocialRecommender:
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

    def set_training_data(self, user_ratings: dict):
        """Set training data for trust-neighbor lookup.

        Parameters
        ----------
        user_ratings : dict[int, dict[int, float]]
            user_idx -> {item_idx: rating} from training data only.
        """
        self.train_user_ratings = user_ratings
        self._train_user_items = {u: set(items.keys()) for u, items in user_ratings.items()}
        self._rating_csr = _build_rating_csr(user_ratings, getattr(self.baseline, "n_items", 0))

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

    def score_all(self, user_idx: int) -> np.ndarray:
        """Social-enhanced prediction for every catalogue item (vectorized).

        Same blending and fallback rules as :meth:`predict`.
        """
        n_items = self.baseline.n_items
        base = _baseline_scores(self.baseline, user_idx, n_items)
        sums, counts = _neighbor_mean_scores(
            self._rating_csr, self.trust_graph.get(user_idx, []), n_items)
        return _blend_scores(base, sums, counts, self.social_weight, self.global_mean)

    def recommend(self, user_idx: int, n: int = 10,
                  exclude: set | None = None) -> list[tuple[int, float]]:
        """Recommend top-n items using social-enhanced predictions.

        Every catalogue item is scored (full-catalogue ranking); ties are
        broken by ascending item index.
        """
        if exclude is None:
            exclude = set()
        train_items = self._train_user_items.get(user_idx, set())
        exclude = exclude | train_items

        return top_n_from_scores(self.score_all(user_idx), n, exclude)


class SimilarUserRecommender:
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
        self._similar_users = {}  # user_idx -> [(similar_user_idx, similarity), ...]

    def set_training_data(self, user_ratings: dict):
        """Set training data and compute similar-but-unconnected users."""
        self.train_user_ratings = user_ratings
        self._train_user_items = {u: set(items.keys()) for u, items in user_ratings.items()}
        self._rating_csr = _build_rating_csr(user_ratings, getattr(self.baseline, "n_items", 0))

    def compute_similar_users(self, target_users: list[int]):
        """For each target user, find top-K similar users NOT in their trust network.

        Uses cosine similarity on co-rated items from training data.
        """
        logger.info("Computing similar-but-unconnected users for %d targets...", len(target_users))

        all_users = sorted(self.train_user_ratings.keys())

        for uidx in target_users:
            u_ratings = self.train_user_ratings.get(uidx, {})
            if not u_ratings:
                self._similar_users[uidx] = []
                continue

            trusted_set = set(self.trust_graph.get(uidx, []))
            u_items = set(u_ratings.keys())
            u_mean = np.mean(list(u_ratings.values()))

            similarities = []
            for vidx in all_users:
                if vidx == uidx or vidx in trusted_set:
                    continue
                v_ratings = self.train_user_ratings.get(vidx, {})
                if not v_ratings:
                    continue

                # Co-rated items
                common = u_items & set(v_ratings.keys())
                if len(common) < 2:
                    continue

                v_mean = np.mean(list(v_ratings.values()))

                # Cosine similarity on co-rated items (mean-centered)
                num = sum((u_ratings[c] - u_mean) * (v_ratings[c] - v_mean) for c in common)
                den_u = np.sqrt(sum((u_ratings[c] - u_mean) ** 2 for c in common))
                den_v = np.sqrt(sum((v_ratings[c] - v_mean) ** 2 for c in common))

                if den_u < 1e-10 or den_v < 1e-10:
                    continue

                sim = num / (den_u * den_v)
                similarities.append((vidx, sim))

            similarities.sort(key=lambda x: -x[1])
            self._similar_users[uidx] = similarities[:self.k_similar]

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

    def score_all(self, user_idx: int) -> np.ndarray:
        """Similar-user-enhanced prediction for every item (vectorized predict)."""
        n_items = self.baseline.n_items
        base = _baseline_scores(self.baseline, user_idx, n_items)
        similar = [s for s, _ in self._similar_users.get(user_idx, [])]
        sums, counts = _neighbor_mean_scores(self._rating_csr, similar, n_items)
        return _blend_scores(base, sums, counts, self.social_weight, self.global_mean)

    def recommend(self, user_idx: int, n: int = 10,
                  exclude: set | None = None) -> list[tuple[int, float]]:
        if exclude is None:
            exclude = set()
        train_items = self._train_user_items.get(user_idx, set())
        exclude = exclude | train_items

        return top_n_from_scores(self.score_all(user_idx), n, exclude)
