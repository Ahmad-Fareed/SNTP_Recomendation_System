"""Reproducible experiment-design helpers.

These helpers exist to make the eight requirements comparable and to keep
every learned/selected quantity away from held-out data:

* :func:`build_rating_matrix` builds a matrix from an explicit list of rating
  triples, so callers must pass the training split (or the validation split)
  rather than the full dataset.
* :func:`select_target_item` chooses the item used by the diffusion
  experiments from a documented *training-only* popularity band.
* :func:`generate_seed_sets` pre-generates the seed users used by diffusion
  simulations so that every comparable parameter setting reuses exactly the
  same seed sets.
* :func:`make_simulation_seeds` creates the controlled random-seed schedule
  used for Monte Carlo trials.

The module contains no access to validation or test ratings.  Callers are
responsible for passing the correct split.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np
from scipy import sparse


def build_rating_matrix(
    n_users: int,
    n_items: int,
    triples: Iterable[tuple[int, int, int]],
    user_to_idx: dict[int, int],
    item_to_idx: dict[int, int],
    dtype=np.float32,
) -> sparse.csr_matrix:
    """Build a sparse ``(n_users, n_items)`` matrix from rating triples.

    Ratings whose user or item is not in the index mappings are skipped.
    Duplicate ``(user, item)`` entries are summed by CSR construction; the
    project's loader removes duplicates before splitting, so this is only a
    defensive behaviour.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for uid, iid, rating in triples:
        uidx = user_to_idx.get(uid)
        iidx = item_to_idx.get(iid)
        if uidx is None or iidx is None:
            continue
        rows.append(int(uidx))
        cols.append(int(iidx))
        vals.append(float(rating))
    return sparse.csr_matrix(
        (np.asarray(vals, dtype=dtype), (np.asarray(rows, dtype=np.int64),
                                         np.asarray(cols, dtype=np.int64))),
        shape=(int(n_users), int(n_items)),
        dtype=dtype,
    )


def select_target_item(
    train_triples: Sequence[tuple[int, int, int]],
    item_to_idx: dict[int, int],
    lower_quantile: float = 0.40,
    upper_quantile: float = 0.60,
) -> dict:
    """Select the diffusion target item from a training-popularity band.

    Selection rule (documented and deterministic):

    1. Count how many **training** ratings each item has.
    2. Compute the ``lower_quantile`` and ``upper_quantile`` of the item
       training-count distribution (default: the interquartile middle 20%,
       i.e. the 40th-60th percentile).
    3. Keep items whose training count lies inside that inclusive band.
    4. Pick the item whose training count is closest to the median count;
       ties are broken by the smallest original item ID.

    No validation or test rating is consulted.  If the band is empty (for
    example with a tiny synthetic dataset), the item closest to the median is
    used and the fallback is recorded in ``selection_rule``.
    """
    if not 0.0 <= lower_quantile <= upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower <= upper <= 1")

    counts = Counter(int(iid) for _, iid, _ in train_triples
                     if int(iid) in item_to_idx)
    if not counts:
        raise ValueError("cannot select a target item from empty training data")

    values = np.asarray(sorted(counts.values()), dtype=np.float64)
    lower_count = float(np.quantile(values, lower_quantile))
    upper_count = float(np.quantile(values, upper_quantile))
    median_count = float(np.median(values))

    band = [iid for iid, count in counts.items()
            if lower_count <= count <= upper_count]
    used_fallback = False
    if not band:
        used_fallback = True
        band = list(counts.keys())

    target_id = min(band, key=lambda iid: (abs(counts[iid] - median_count), iid))
    rule = (
        "Choose the item whose TRAINING rating count is closest to the median "
        f"count among items in the inclusive [{lower_quantile:.2f}, "
        f"{upper_quantile:.2f}] training-count quantile band; ties by "
        "ascending original item ID."
    )
    if used_fallback:
        rule += " The quantile band was empty, so the median-closest item was used."

    return {
        "original_id": int(target_id),
        "idx": int(item_to_idx[target_id]),
        "train_rating_count": int(counts[target_id]),
        "median_train_rating_count": median_count,
        "band_lower_quantile": float(lower_quantile),
        "band_upper_quantile": float(upper_quantile),
        "band_lower_count": lower_count,
        "band_upper_count": upper_count,
        "band_size_items": int(len(band)),
        "n_items_with_train_ratings": int(len(counts)),
        "selection_rule": rule,
        "used_median_fallback": bool(used_fallback),
    }


def generate_seed_sets(
    eligible_users: Sequence[int],
    seed_size: int,
    n_simulations: int,
    seed: int = 42,
) -> list[list[int]]:
    """Pre-generate ``n_simulations`` seed sets without replacement.

    Every comparable diffusion parameter setting must reuse these exact sets.
    Users are sampled uniformly from ``eligible_users``; each set is sorted so
    that the JSON output and simulation iteration order are deterministic.
    """
    users = sorted({int(u) for u in eligible_users})
    if n_simulations < 0:
        raise ValueError("n_simulations must be non-negative")
    if not users or seed_size <= 0:
        return [[] for _ in range(n_simulations)]

    k = min(int(seed_size), len(users))
    rng = np.random.RandomState(int(seed))
    seed_sets: list[list[int]] = []
    for _ in range(int(n_simulations)):
        chosen = rng.choice(users, size=k, replace=False).tolist()
        seed_sets.append(sorted(int(u) for u in chosen))
    return seed_sets


def make_simulation_seeds(base_seed: int, n_simulations: int,
                          offset: int = 0) -> list[int]:
    """Controlled Monte Carlo seed schedule.

    A fixed offset lets selection and final evaluation use separate random
    streams while remaining reproducible (for example ``offset=0`` for greedy
    candidate evaluation and ``offset=500_000`` for final spread estimates).
    """
    return [int(base_seed) + int(offset) + i * 1000 + 7
            for i in range(int(n_simulations))]
