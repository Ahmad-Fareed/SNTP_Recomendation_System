"""
Biased Matrix Factorization with Stochastic Gradient Descent (SGD).

Model:
  predicted_rating(u, i) = μ + b_u + b_i + p_u · q_i

where:
  μ      = global mean rating (from training data)
  b_u    = user bias term
  b_i    = item bias term
  p_u    = latent factor vector for user u  (dimension d)
  q_i    = latent factor vector for item i  (dimension d)

Training:
  Minimize regularized squared error on observed training ratings:
    L = Σ_{(u,i) ∈ train} (r_ui - pred(u,i))^2
        + λ * (||p_u||^2 + ||q_i||^2 + b_u^2 + b_i^2)

  SGD updates per observed (u, i, r):
    e = r - pred(u, i)
    b_u ← b_u + α * (e - λ * b_u)
    b_i ← b_i + α * (e - λ * b_i)
    p_u ← p_u + α * (e * q_i - λ * p_u)
    q_i ← q_i + α * (e * p_u - λ * q_i)

  Missing ratings are never treated as zero — only observed ratings contribute to training.

Limitations:
  - Cold-start: new users/items not in training get only bias + global mean.
  - Latent factors have no direct interpretability.
  - Requires tuning of d, α, λ, and number of epochs.
"""

import logging
import time

import numpy as np

from sntp.recommenders._ranking import top_n_from_scores

logger = logging.getLogger(__name__)


class BiasedMF:
    """Biased latent-factor matrix factorization trained with SGD."""

    def __init__(self, n_factors: int = 20, lr: float = 0.005,
                 reg: float = 0.02, n_epochs: int = 20, seed: int = 42):
        """
        Parameters
        ----------
        n_factors : int
            Dimensionality of latent factor vectors.
        lr : float
            Learning rate (α).
        reg : float
            Regularization strength (λ).
        n_epochs : int
            Number of SGD training epochs.
        seed : int
            Random seed for reproducible initialization.
        """
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed

        self.global_mean = 0.0
        self.user_bias = None
        self.item_bias = None
        self.user_factors = None
        self.item_factors = None
        self.n_users = 0
        self.n_items = 0
        self._train_user_items = {}

    def fit(self, train_triples, n_users: int, n_items: int,
            user_to_idx: dict, item_to_idx: dict,
            val_triples=None):
        """Train the model on observed ratings.

        Parameters
        ----------
        train_triples : list[tuple[int, int, int]]
            (user_id, item_id, rating) using original IDs.
        n_users, n_items : int
            Total number of users/items in the index space.
        user_to_idx, item_to_idx : dict
            Original ID -> integer index mappings.
        val_triples : list or None
            Optional validation set for epoch-level monitoring.
        """
        t0 = time.time()
        self.n_users = n_users
        self.n_items = n_items

        rng = np.random.RandomState(self.seed)

        # Convert to index space
        indexed = []
        for uid, iid, r in train_triples:
            uidx = user_to_idx.get(uid)
            iidx = item_to_idx.get(iid)
            if uidx is not None and iidx is not None:
                indexed.append((uidx, iidx, float(r)))

        if not indexed:
            logger.warning("No valid training ratings after indexing.")
            return

        ratings_arr = np.array([r for _, _, r in indexed], dtype=np.float64)
        self.global_mean = float(ratings_arr.mean())

        # Initialize parameters
        scale = 0.1 / np.sqrt(self.n_factors)
        self.user_bias = np.zeros(n_users, dtype=np.float64)
        self.item_bias = np.zeros(n_items, dtype=np.float64)
        self.user_factors = rng.normal(0, scale, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, scale, (n_items, self.n_factors))

        # Build training user items for exclusion during recommendation
        from collections import defaultdict
        self._train_user_items = defaultdict(set)
        for uidx, iidx, _ in indexed:
            self._train_user_items[uidx].add(iidx)

        # Prepare validation if provided (as index arrays for vectorized RMSE)
        val_u = val_i = val_r = None
        if val_triples:
            val_indexed = []
            for uid, iid, r in val_triples:
                uidx = user_to_idx.get(uid)
                iidx = item_to_idx.get(iid)
                if uidx is not None and iidx is not None:
                    val_indexed.append((uidx, iidx, float(r)))
            if val_indexed:
                val_u = np.array([v[0] for v in val_indexed], dtype=np.int64)
                val_i = np.array([v[1] for v in val_indexed], dtype=np.int64)
                val_r = np.array([v[2] for v in val_indexed], dtype=np.float64)

        # SGD training.
        # Performance notes (the update rule is unchanged):
        #  * Each epoch shuffles an index permutation in place.  This consumes
        #    the RNG identically to shuffling the (n, 3) rating array itself, so
        #    the visiting order is the same as the original implementation.
        #  * User/item ids are pre-converted to Python ints and biases are kept
        #    in Python float lists (IEEE-754 doubles, same arithmetic as numpy
        #    float64 scalars) to avoid per-rating numpy scalar overhead.
        #  * Factor rows are updated in place through views.
        n_train = len(indexed)
        u_list = [u for u, _, _ in indexed]
        i_list = [i for _, i, _ in indexed]
        r_list = [r for _, _, r in indexed]
        order = np.arange(n_train)

        P = self.user_factors
        Q = self.item_factors
        bu = self.user_bias.tolist()
        bi = self.item_bias.tolist()
        mu = self.global_mean
        lr = self.lr
        reg = self.reg

        epoch_history = []

        for epoch in range(self.n_epochs):
            t_epoch = time.time()
            rng.shuffle(order)

            train_loss = 0.0
            for idx in order.tolist():
                u = u_list[idx]
                i = i_list[idx]
                r = r_list[idx]
                pu = P[u]
                qi = Q[i]

                pred = mu + bu[u] + bi[i] + float(pu.dot(qi))
                err = r - pred

                # Update biases
                bu[u] += lr * (err - reg * bu[u])
                bi[i] += lr * (err - reg * bi[i])

                # Update latent factors (save old p_u before updating)
                old_pu = pu.copy()
                pu += lr * (err * qi - reg * pu)
                qi += lr * (err * old_pu - reg * qi)

                train_loss += err ** 2

            train_rmse = float(np.sqrt(train_loss / n_train))

            # Validation RMSE (vectorized; same formula as _raw_predict)
            val_rmse = None
            if val_u is not None:
                bu_arr = np.asarray(bu)
                bi_arr = np.asarray(bi)
                val_pred = (mu + bu_arr[val_u] + bi_arr[val_i]
                            + np.einsum("ij,ij->i", P[val_u], Q[val_i]))
                val_rmse = float(np.sqrt(np.mean((val_r - val_pred) ** 2)))

            epoch_history.append({
                "epoch": epoch + 1,
                "train_rmse": round(train_rmse, 5),
                "val_rmse": round(val_rmse, 5) if val_rmse is not None else None,
            })

            val_str = f", val_rmse={val_rmse:.5f}" if val_rmse is not None else ""
            logger.info(
                "Epoch %d/%d: train_rmse=%.5f%s (%.1fs)",
                epoch + 1, self.n_epochs, train_rmse, val_str, time.time() - t_epoch,
            )

        self.user_bias = np.asarray(bu, dtype=np.float64)
        self.item_bias = np.asarray(bi, dtype=np.float64)

        elapsed = time.time() - t0
        logger.info("MF trained in %.1fs over %d epochs.", elapsed, self.n_epochs)
        self.epoch_history = epoch_history

    def _raw_predict(self, user_idx: int, item_idx: int) -> float:
        """Raw prediction without clipping (used during training)."""
        pred = self.global_mean
        if self.user_bias is not None and user_idx < self.n_users:
            pred += self.user_bias[user_idx]
        if self.item_bias is not None and item_idx < self.n_items:
            pred += self.item_bias[item_idx]
        if (self.user_factors is not None and self.item_factors is not None
                and user_idx < self.n_users and item_idx < self.n_items):
            pred += self.user_factors[user_idx].dot(self.item_factors[item_idx])
        return pred

    def predict(self, user_idx: int, item_idx: int,
                rating_min: float = 1.0, rating_max: float = 5.0) -> float:
        """Predict and clip rating for (user, item).

        Returns clipped prediction. Never returns None — cold-start users/items
        get global_mean + available biases.
        """
        pred = self._raw_predict(user_idx, item_idx)
        return float(np.clip(pred, rating_min, rating_max))

    def score_all(self, user_idx: int, rating_min: float = 1.0,
                  rating_max: float = 5.0) -> np.ndarray:
        """Clipped predicted rating for every catalogue item (vectorized predict)."""
        if user_idx < self.n_users:
            preds = (self.global_mean
                     + self.user_bias[user_idx]
                     + self.item_bias
                     + self.item_factors.dot(self.user_factors[user_idx]))
        else:
            preds = self.global_mean + self.item_bias
        return np.clip(preds, rating_min, rating_max)

    def recommend(self, user_idx: int, n: int = 10,
                  exclude: set | None = None) -> list[tuple[int, float]]:
        """Recommend top-n unseen items.

        Every catalogue item is scored (full-catalogue ranking).
        Returns list of (item_idx, predicted_rating) sorted descending; ties
        are broken by ascending item index.
        """
        if exclude is None:
            exclude = set()
        train_items = self._train_user_items.get(user_idx, set())
        exclude = exclude | train_items

        scores = self.score_all(user_idx)
        return top_n_from_scores(scores, n, exclude)
