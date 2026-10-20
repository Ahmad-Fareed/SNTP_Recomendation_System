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

import random
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
    min_train_ratings: int = 6,
    max_train_ratings: int = 20,
) -> dict:
    """Select the diffusion target item from an explicit training-count band.

    Selection rule (documented and deterministic):

    1. Count how many **training** ratings each item has.
    2. Keep items whose count is in the inclusive
       ``[min_train_ratings, max_train_ratings]`` band (default 6–20).
       This avoids an ultra-sparse target with only one or two observations,
       while still keeping the item away from the head of the popularity
       distribution.
    3. Pick the item whose count is closest to the median count *within the
       band*; ties are broken by the smallest original item ID.

    No validation or test rating is consulted.  If the band is empty (for
    example on a tiny synthetic dataset), the item whose count is closest to
    the midpoint of the requested band is used and the fallback is recorded in
    ``selection_rule``.
    """
    if min_train_ratings < 1 or max_train_ratings < min_train_ratings:
        raise ValueError(
            "need 1 <= min_train_ratings <= max_train_ratings"
        )

    counts = Counter(int(iid) for _, iid, _ in train_triples
                     if int(iid) in item_to_idx)
    if not counts:
        raise ValueError("cannot select a target item from empty training data")

    band = [iid for iid, count in counts.items()
            if min_train_ratings <= count <= max_train_ratings]
    used_fallback = False
    if band:
        band_counts = np.asarray([counts[iid] for iid in band], dtype=np.float64)
        band_median = float(np.median(band_counts))
        target_id = min(
            band, key=lambda iid: (abs(counts[iid] - band_median), iid)
        )
    else:
        used_fallback = True
        midpoint = 0.5 * (min_train_ratings + max_train_ratings)
        band_median = midpoint
        target_id = min(
            counts, key=lambda iid: (abs(counts[iid] - midpoint), iid)
        )

    rule = (
        "Choose the item whose TRAINING rating count is closest to the median "
        f"count among items with {min_train_ratings}–{max_train_ratings} "
        "training ratings (inclusive); ties by ascending original item ID."
    )
    if used_fallback:
        rule += (
            " The requested count band was empty, so the item closest to the "
            "band midpoint was used."
        )

    return {
        "original_id": int(target_id),
        "idx": int(item_to_idx[target_id]),
        "train_rating_count": int(counts[target_id]),
        "band_min_train_ratings": int(min_train_ratings),
        "band_max_train_ratings": int(max_train_ratings),
        "band_median_train_rating_count": band_median,
        "band_size_items": int(len(band)),
        "n_items_with_train_ratings": int(len(counts)),
        "selection_rule": rule,
        "used_count_band_fallback": bool(used_fallback),
    }


def randomize_directed_graph_degree_preserving(
    graph: dict[int, Sequence[int]],
    *,
    nodes: Iterable[int] | None = None,
    swap_multiplier: int = 2,
    seed: int = 42,
) -> tuple[dict[int, list[int]], dict]:
    """Degree-preserving randomized directed graph for a controlled comparison.

    The observed graph and the returned graph have exactly the same node set,
    edge count, and directed in/out-degree sequence.  Edges are rearranged by
    double-edge swaps: two edges ``a->b`` and ``c->d`` become ``a->d`` and
    ``c->b``.  Swaps that would create a self-loop, a duplicate edge, or a
    non-simple graph are rejected.  The number of attempted and accepted swaps
    is recorded.

    This is a null-model comparison, not a claim that the randomized graph is
    the unique or fully mixed graph with that degree sequence.  It is
    deterministic for a fixed ``seed`` and ``swap_multiplier``.

    Parameters
    ----------
    graph : dict[int, sequence[int]]
        Observed directed adjacency list.
    nodes : iterable[int] or None
        Complete node set to preserve.  If omitted, graph keys and neighbours
        are used.  Isolated nodes are retained in the returned graph.
    swap_multiplier : int
        Number of swap attempts is ``swap_multiplier * n_edges``.
    seed : int
        Deterministic random seed.
    """
    if swap_multiplier < 0:
        raise ValueError("swap_multiplier must be non-negative")

    if nodes is None:
        node_set = {int(u) for u in graph}
        node_set.update(int(v) for neighbours in graph.values() for v in neighbours)
    else:
        # Preserve exactly the requested node set; edges to nodes outside it
        # are not part of the controlled graph.
        node_set = {int(u) for u in nodes}
    node_list = sorted(node_set)
    if not node_list:
        return {}, {
            "n_nodes": 0, "n_edges": 0, "swaps_attempted": 0,
            "swaps_accepted": 0, "acceptance_rate": 0.0,
            "swap_multiplier": int(swap_multiplier), "seed": int(seed),
            "degree_sequence_preserved": True,
        }

    max_node = max(node_list)
    encode = lambda u, v: int(u) * (max_node + 1) + int(v)
    edge_keys = set()
    edges: list[tuple[int, int]] = []
    duplicate_edges_removed = 0
    self_loops_removed = 0
    for u in node_list:
        for v in graph.get(u, []):
            v = int(v)
            if v not in node_set:
                continue
            if u == v:
                self_loops_removed += 1
                continue
            key = encode(u, v)
            if key in edge_keys:
                duplicate_edges_removed += 1
                continue
            edge_keys.add(key)
            edges.append((u, v))

    n_edges = len(edges)
    degree_out_before = Counter(u for u, _ in edges)
    degree_in_before = Counter(v for _, v in edges)

    rng = random.Random(int(seed))
    attempts = int(swap_multiplier) * n_edges
    accepted = 0
    for _ in range(attempts):
        if n_edges < 2:
            break
        i = rng.randrange(n_edges)
        j = rng.randrange(n_edges - 1)
        if j >= i:
            j += 1
        a, b = edges[i]
        c, d = edges[j]
        if a == d or c == b:
            continue
        new_i = encode(a, d)
        new_j = encode(c, b)
        if new_i == new_j:
            continue
        old_i, old_j = encode(a, b), encode(c, d)
        if new_i in edge_keys or new_j in edge_keys:
            continue
        edge_keys.remove(old_i)
        edge_keys.remove(old_j)
        edge_keys.add(new_i)
        edge_keys.add(new_j)
        edges[i] = (a, d)
        edges[j] = (c, b)
        accepted += 1

    degree_out_after = Counter(u for u, _ in edges)
    degree_in_after = Counter(v for _, v in edges)
    degrees_preserved = (
        degree_out_before == degree_out_after
        and degree_in_before == degree_in_after
    )
    if not degrees_preserved:
        raise RuntimeError("degree-preserving randomization changed the degree sequence")

    randomized = {u: [] for u in node_list}
    for u, v in edges:
        randomized[u].append(v)
    for u in node_list:
        randomized[u].sort()

    info = {
        "n_nodes": len(node_list),
        "n_edges": n_edges,
        "swaps_attempted": int(attempts),
        "swaps_accepted": int(accepted),
        "acceptance_rate": round(accepted / attempts, 6) if attempts else 0.0,
        "swap_multiplier": int(swap_multiplier),
        "seed": int(seed),
        "duplicate_edges_removed": int(duplicate_edges_removed),
        "self_loops_removed": int(self_loops_removed),
        "degree_sequence_preserved": True,
        "max_out_degree": max((len(v) for v in randomized.values()), default=0),
    }
    return randomized, info


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
