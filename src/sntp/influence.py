"""
Part 5: Influence Maximization algorithms on social propagation networks.

Implements and compares three seed selection strategies:
1. Random seed selection.
2. Degree-centrality seed selection (out-degree in propagation graph).
3. Greedy selection based on marginal expected spread (via Monte Carlo ICM).

Also implements CELF (Cost-Effective Lazy Forward) optimization for Part 8,
leveraging submodularity of information cascade spread to reduce evaluations.
"""

from __future__ import annotations

import heapq
import logging
import time
from typing import Any

import numpy as np

from sntp.diffusion.icm import ICMSimulator

logger = logging.getLogger(__name__)


def estimate_spread(
    propagation_graph: dict[int, list[int]],
    seed_users: set[int] | list[int],
    propagation_prob: float = 0.1,
    n_simulations: int = 10,
    max_rounds: int = 100,
    seed: int = 42,
) -> tuple[float, float, list[set[int]]]:
    """Estimate expected spread (mean, std) of a seed set via Monte Carlo ICM.

    Parameters
    ----------
    propagation_graph : dict[int, list[int]]
        Adjacency list (sender -> receivers).
    seed_users : set[int] or list[int]
        Initial active nodes.
    propagation_prob : float
        Edge activation probability.
    n_simulations : int
        Number of independent Monte Carlo iterations.
    max_rounds : int
        Max rounds per simulation.
    seed : int
        Base random seed.

    Returns
    -------
    mean_spread : float
        Average number of total adopters (including seeds).
    std_spread : float
        Standard deviation of adopters count across simulations.
    sim_adopter_sets : list[set[int]]
        Set of adopters for each simulation run.
    """
    seeds = set(seed_users)
    if not seeds:
        return 0.0, 0.0, [set() for _ in range(n_simulations)]

    adopter_counts = []
    adopter_sets = []

    for i in range(n_simulations):
        sim = ICMSimulator(
            propagation_graph=propagation_graph,
            propagation_prob=propagation_prob,
            max_rounds=max_rounds,
            seed=seed + i * 1000 + 7,
        )
        res = sim.run_canonical(seeds)
        adopters = res["adopters"]
        adopter_counts.append(len(adopters))
        adopter_sets.append(adopters)

    mean_s = float(np.mean(adopter_counts))
    std_s = float(np.std(adopter_counts)) if len(adopter_counts) > 1 else 0.0
    return mean_s, std_s, adopter_sets


def random_seed_selection(
    eligible_users: list[int],
    budget: int,
    seed: int = 42,
) -> list[int]:
    """Select budget seeds uniformly at random without replacement.

    Parameters
    ----------
    eligible_users : list[int]
        Pool of eligible candidate user indices.
    budget : int
        Number of seeds to select.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    selected_seeds : list[int]
    """
    if budget <= 0 or not eligible_users:
        return []
    k = min(budget, len(eligible_users))
    rng = np.random.RandomState(seed)
    selected = rng.choice(eligible_users, size=k, replace=False).tolist()
    return [int(u) for u in selected]


def degree_centrality_seed_selection(
    propagation_graph: dict[int, list[int]],
    eligible_users: list[int] | None = None,
    budget: int = 5,
) -> list[int]:
    """Select top-budget seeds by out-degree in the propagation graph.

    In the propagation graph, out-degree represents the number of users that
    node u can directly influence (i.e., users who trust u).
    Ties are broken by lowest user index.

    Parameters
    ----------
    propagation_graph : dict[int, list[int]]
        Adjacency list.
    eligible_users : list[int] or None
        Candidate users. If None, considers all keys in propagation_graph.
    budget : int
        Number of seeds to select.

    Returns
    -------
    selected_seeds : list[int]
    """
    if budget <= 0:
        return []

    candidates = (
        list(eligible_users)
        if eligible_users is not None
        else list(propagation_graph.keys())
    )
    if not candidates:
        return []

    # Sort key: (-out_degree, user_idx)
    candidates.sort(key=lambda u: (-len(propagation_graph.get(u, [])), u))
    return candidates[:budget]


def greedy_influence_maximization(
    propagation_graph: dict[int, list[int]],
    eligible_users: list[int],
    budget: int = 5,
    propagation_prob: float = 0.1,
    n_simulations: int = 10,
    candidate_pool_size: int | None = None,
    seed: int = 42,
    use_celf: bool = False,
) -> dict[str, Any]:
    """Greedy influence maximization based on marginal expected spread.

    Supports both standard greedy search and CELF (Cost-Effective Lazy Forward)
    lazy evaluation.

    Parameters
    ----------
    propagation_graph : dict[int, list[int]]
        Propagation graph (influencer -> influenced).
    eligible_users : list[int]
        Candidate user indices.
    budget : int
        Promotion budget (number of seeds).
    propagation_prob : float
        Propagation probability per edge.
    n_simulations : int
        Number of Monte Carlo simulations per spread estimate.
    candidate_pool_size : int or None
        If provided and smaller than len(eligible_users), restricts candidate
        pool to top candidates ranked by out-degree for tractability.
    seed : int
        Base random seed.
    use_celf : bool
        If True, applies CELF lazy forward optimization.

    Returns
    -------
    result : dict with keys:
        'seeds': list[int]
        'marginal_gains': list[float]
        'spread_progression': list[float]
        'spread_stds': list[float]
        'n_evaluations': int
        'runtime_seconds': float
        'strategy': str
    """
    t_start = time.time()
    n_evals = 0

    if budget <= 0 or not eligible_users:
        return {
            "seeds": [],
            "marginal_gains": [],
            "spread_progression": [],
            "spread_stds": [],
            "n_evaluations": 0,
            "runtime_seconds": 0.0,
            "strategy": "celf" if use_celf else "greedy",
        }

    # Restrict candidate pool if requested
    candidates = list(set(eligible_users))
    if candidate_pool_size is not None and candidate_pool_size < len(candidates):
        candidates.sort(key=lambda u: (-len(propagation_graph.get(u, [])), u))
        candidates = candidates[:candidate_pool_size]

    selected_seeds: list[int] = []
    marginal_gains: list[float] = []
    spread_progression: list[float] = []
    spread_stds: list[float] = []

    current_spread = 0.0

    if not use_celf:
        # Standard greedy
        remaining_candidates = set(candidates)
        for step in range(budget):
            if not remaining_candidates:
                break

            best_candidate = None
            best_gain = -1.0
            best_spread = current_spread
            best_std = 0.0

            # Sort candidate loop deterministically
            for cand in sorted(remaining_candidates):
                test_set = set(selected_seeds) | {cand}
                sp_mean, sp_std, _ = estimate_spread(
                    propagation_graph=propagation_graph,
                    seed_users=test_set,
                    propagation_prob=propagation_prob,
                    n_simulations=n_simulations,
                    seed=seed,
                )
                n_evals += 1
                gain = sp_mean - current_spread

                if gain > best_gain or (
                    gain == best_gain and (best_candidate is None or cand < best_candidate)
                ):
                    best_gain = gain
                    best_candidate = cand
                    best_spread = sp_mean
                    best_std = sp_std

            if best_candidate is None:
                break

            selected_seeds.append(best_candidate)
            remaining_candidates.remove(best_candidate)
            marginal_gains.append(float(best_gain))
            spread_progression.append(float(best_spread))
            spread_stds.append(float(best_std))
            current_spread = best_spread

    else:
        # CELF optimization (Leskovec et al. 2007)
        # Max-heap elements: (-marginal_gain, user_id, iteration_tag)
        heap: list[tuple[float, int, int]] = []

        # Step 0: Initial single-node evaluations
        for cand in sorted(candidates):
            sp_mean, sp_std, _ = estimate_spread(
                propagation_graph=propagation_graph,
                seed_users={cand},
                propagation_prob=propagation_prob,
                n_simulations=n_simulations,
                seed=seed,
            )
            n_evals += 1
            gain = sp_mean
            heapq.heappush(heap, (-gain, cand, 0))

        # Iterations 1..budget
        for step in range(budget):
            if not heap:
                break

            while True:
                neg_gain, cand, tag = heapq.heappop(heap)
                # If tag equals current step, this candidate's gain was computed with current seed set
                if tag == step:
                    selected_seeds.append(cand)
                    gain = -neg_gain
                    marginal_gains.append(float(gain))
                    current_spread += gain
                    spread_progression.append(float(current_spread))
                    spread_stds.append(0.0)  # CELF lazy track
                    break

                # Otherwise, re-evaluate with current seed set
                test_set = set(selected_seeds) | {cand}
                sp_mean, sp_std, _ = estimate_spread(
                    propagation_graph=propagation_graph,
                    seed_users=test_set,
                    propagation_prob=propagation_prob,
                    n_simulations=n_simulations,
                    seed=seed,
                )
                n_evals += 1
                new_gain = sp_mean - current_spread
                heapq.heappush(heap, (-new_gain, cand, step))

    elapsed = time.time() - t_start

    return {
        "seeds": selected_seeds,
        "marginal_gains": marginal_gains,
        "spread_progression": spread_progression,
        "spread_stds": spread_stds,
        "n_evaluations": n_evals,
        "runtime_seconds": round(elapsed, 3),
        "strategy": "celf" if use_celf else "greedy",
    }


def compute_adopter_overlap(
    adopter_sets_a: list[set[int]],
    adopter_sets_b: list[set[int]],
) -> float:
    """Compute average Jaccard similarity between two sets of simulation outcomes.

    Parameters
    ----------
    adopter_sets_a : list[set[int]]
        Adopter sets from method A across simulations.
    adopter_sets_b : list[set[int]]
        Adopter sets from method B across simulations.

    Returns
    -------
    mean_jaccard : float in [0, 1]
    """
    if not adopter_sets_a or not adopter_sets_b:
        return 0.0

    jaccards = []
    n = min(len(adopter_sets_a), len(adopter_sets_b))
    for i in range(n):
        s_a = adopter_sets_a[i]
        s_b = adopter_sets_b[i]
        union = s_a | s_b
        if not union:
            jaccards.append(1.0)
        else:
            jaccards.append(len(s_a & s_b) / len(union))

    return float(np.mean(jaccards))
