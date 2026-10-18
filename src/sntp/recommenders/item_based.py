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

Limitations:
  - Requires co-rated users; items with very few ratings have unreliable similarities.
  - Cold-start items (no training ratings) cannot be recommended.
  - K-neighbor limit may miss some relevant similarities.
"""

import logging
import time
from collections import defaultdict

import numpy as np
from scipy import sparse

from sntp.recommenders._ranking import top_n_from_scores

logger = logging.getLogger(__name__)


class ItemBasedCF:
    """Item-based collaborative filtering with adjusted cosine similarity."""

    def __init__(self, max_neighbors: int = 50, min_corated: int = 2):
        """
        Parameters
        ----------
        max_neighbors : int
            Maximum number of similar items stored per item.
        min_corated : int
            Minimum co-rated users required to compute a similarity.
        """
        self.max_neighbors = max_neighbors
        self.min_corated = min_corated
        self.similarities = {}   # item_idx -> [(neighbor_idx, sim), ...]
        self.user_means = None
        self.global_mean = 0.0
        self.train_user_ratings = {}  # user_idx -> {item_idx: rating}
        self.n_items = 0
        # Sparse (target_item x neighbor_item) matrices of the stored top-K
        # similarities, in CSC so a user's rated-item columns slice cheaply.
        self._sim_csc = None
        self._abs_sim_csc = None
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
        """
        t0 = time.time()
        self.n_items = rating_matrix.shape[1]
        n_users = rating_matrix.shape[0]

        # Compute user means (over observed ratings only)
        self.user_means = np.zeros(n_users, dtype=np.float64)
        user_counts = np.zeros(n_users, dtype=np.float64)
        for uidx, items in user_ratings.items():
            ratings_arr = [r for _, r in items]
            if ratings_arr:
                self.user_means[uidx] = np.mean(ratings_arr)
                user_counts[uidx] = len(ratings_arr)

        self.global_mean = float(rating_matrix.data.mean()) if rating_matrix.nnz > 0 else 3.0

        # Build user -> {item: rating} lookup for fast prediction
        self.train_user_ratings = {}
        for uidx, items in user_ratings.items():
            self.train_user_ratings[uidx] = {iidx: r for iidx, r in items}

        # Build item -> set of (user_idx, adjusted_rating) for similarity computation
        item_user_adj = {}  # item_idx -> dict{user_idx: adjusted_rating}
        for iidx, users in item_ratings.items():
            item_user_adj[iidx] = {}
            for uidx, r in users:
                item_user_adj[iidx][uidx] = r - self.user_means[uidx]

        # Compute similarities
        item_indices = sorted(item_ratings.keys())
        logger.info("Computing item-item similarities for %d items...", len(item_indices))

        # For efficiency, build inverted index: user -> items they rated
        user_items_set = defaultdict(set)
        for iidx, users in item_ratings.items():
            for uidx, _ in users:
                user_items_set[uidx].add(iidx)

        # Compute pairwise similarities using co-rated users
        # For scalability, iterate through users and accumulate dot products
        # between pairs of items that they co-rated.
        n_items_total = self.n_items
        # Use dictionaries to accumulate numerator and denominator components
        dot_products = defaultdict(float)    # (i, j) -> sum of adjusted products
        norms_sq = defaultdict(float)        # item_idx -> sum of squared adjusted ratings
        corated_counts = defaultdict(int)    # (i, j) -> count of co-rated users

        # Pre-compute item norms
        for iidx in item_indices:
            for uidx, adj_r in item_user_adj[iidx].items():
                norms_sq[iidx] += adj_r ** 2

        # Accumulate dot products through user iteration
        for uidx, items in user_items_set.items():
            items_list = sorted(items)
            for idx_a in range(len(items_list)):
                i = items_list[idx_a]
                adj_i = item_user_adj[i].get(uidx, 0.0)
                for idx_b in range(idx_a + 1, len(items_list)):
                    j = items_list[idx_b]
                    adj_j = item_user_adj[j].get(uidx, 0.0)
                    pair = (i, j) if i < j else (j, i)
                    dot_products[pair] += adj_i * adj_j
                    corated_counts[pair] += 1

        # Build neighbor lists
        item_neighbors = defaultdict(list)
        for (i, j), dot in dot_products.items():
            if corated_counts[(i, j)] < self.min_corated:
                continue
            norm_i = np.sqrt(norms_sq.get(i, 0))
            norm_j = np.sqrt(norms_sq.get(j, 0))
            if norm_i < 1e-10 or norm_j < 1e-10:
                continue
            sim = dot / (norm_i * norm_j)
            item_neighbors[i].append((j, sim))
            item_neighbors[j].append((i, sim))

        # Keep top-K neighbors per item
        for iidx in item_neighbors:
            neighbors = item_neighbors[iidx]
            neighbors.sort(key=lambda x: -abs(x[1]))
            self.similarities[iidx] = neighbors[:self.max_neighbors]

        self._build_scoring_structures()

        elapsed = time.time() - t0
        n_with_neighbors = sum(1 for v in self.similarities.values() if v)
        logger.info(
            "Item-based CF fitted in %.1fs. %d / %d items have neighbors.",
            elapsed, n_with_neighbors, len(item_indices),
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

    def _build_scoring_structures(self):
        """Build sparse similarity matrices and per-user arrays for score_all().

        S[i, j] = sim(i, j) for every stored neighbor j of target item i, so
        for a user u with rated items J_u and ratings r_u:
            num = S[:, J_u] @ r_u      (= sum_j sim(i,j) * r(u,j)  for all i)
            den = |S|[:, J_u] @ 1      (= sum_j |sim(i,j)|         for all i)
        which is exactly the per-item computation in predict().
        """
        rows, cols, data = [], [], []
        for iidx, neighbors in self.similarities.items():
            for j, sim in neighbors:
                rows.append(iidx)
                cols.append(j)
                data.append(sim)
        shape = (self.n_items, self.n_items)
        sim_mat = sparse.csc_matrix(
            (np.asarray(data, dtype=np.float64),
             (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
            shape=shape,
        )
        self._sim_csc = sim_mat
        self._abs_sim_csc = abs(sim_mat)

        self._user_items_arr = {}
        self._user_ratings_arr = {}
        for uidx, items in self.train_user_ratings.items():
            if items:
                self._user_items_arr[uidx] = np.fromiter(items.keys(), dtype=np.int64,
                                                         count=len(items))
                self._user_ratings_arr[uidx] = np.fromiter(items.values(), dtype=np.float64,
                                                           count=len(items))

    def score_all(self, user_idx: int, rating_min: float = 1.0,
                  rating_max: float = 5.0) -> np.ndarray:
        """Predicted rating for *every* catalogue item for one user (vectorized).

        Applies the same formula and fallback chain as :meth:`predict`:
        neighbor-weighted average -> user mean -> global mean, then clipping.
        """
        items = self._user_items_arr.get(user_idx)
        if items is None:
            # No training ratings: predict() returns the global mean for all items.
            return np.full(self.n_items, float(np.clip(self.global_mean, rating_min,
                                                       rating_max)))

        ratings = self._user_ratings_arr[user_idx]
        num = self._sim_csc[:, items] @ ratings
        den = np.asarray(self._abs_sim_csc[:, items].sum(axis=1)).ravel()

        user_mean = float(np.mean(ratings))
        scores = np.full(self.n_items, user_mean, dtype=np.float64)
        has_nb = den > 1e-10
        scores[has_nb] = num[has_nb] / den[has_nb]
        np.clip(scores, rating_min, rating_max, out=scores)
        return scores

    def recommend(self, user_idx: int, n: int = 10,
                  exclude: set | None = None) -> list[tuple[int, float]]:
        """Recommend top-n items not in exclude set.

        Every catalogue item is scored (full-catalogue ranking).
        Returns list of (item_idx, predicted_rating) sorted by predicted rating
        descending; ties are broken by ascending item index.
        """
        if exclude is None:
            exclude = set()
        # Add training items to exclusion
        train_items = set(self.train_user_ratings.get(user_idx, {}).keys())
        exclude = exclude | train_items

        scores = self.score_all(user_idx)
        return top_n_from_scores(scores, n, exclude)
