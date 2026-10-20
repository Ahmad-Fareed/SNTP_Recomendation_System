"""
Independent Cascade Model (ICM) for information diffusion.

Two variants are implemented:
  1. Canonical ICM:  each newly activated node tries ONCE to activate each
     inactive neighbor with a fixed propagation probability p.
  2. Preference-aware ICM:  exposure and adoption are separate stages.
     An exposed user adopts with probability based on predicted preference affinity.

Graph direction convention:
  The propagation graph has edges from influencer -> influenced.
  If "A trusts B", B can influence A, so propagation edge is B -> A.

Timing:
  - Synchronous rounds: newly adopting users spread in the NEXT round, not the current one.
  - Each active user gets exactly one attempt per outgoing edge.
  - A failed attempt is never retried by the same sender.

Events tracked:
  - 'recommend': user received a direct platform recommendation.
  - 'expose': user became exposed through a neighbor.
  - 'adopt': user adopted the item.

The dataset has NO observed adoption timestamps or diffusion histories.
All propagation results are simulated and must be labeled as such.
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class ICMSimulator:
    """Independent Cascade Model simulator."""

    def __init__(self, propagation_graph: dict, propagation_prob: float = 0.1,
                 max_rounds: int = 100, seed: int = 42):
        """
        Parameters
        ----------
        propagation_graph : dict[int, list[int]]
            Adjacency list where propagation_graph[v] = users that v can influence.
        propagation_prob : float
            Base probability of successful propagation per edge (canonical ICM).
        max_rounds : int
            Safety limit on simulation rounds.
        seed : int
            Random seed for reproducibility.
        """
        self.propagation_graph = propagation_graph
        self.propagation_prob = propagation_prob
        self.max_rounds = max_rounds
        self.seed = seed

    def run_canonical(self, seed_users: set[int]) -> dict:
        """Run canonical ICM from a set of initial adopters.

        Parameters
        ----------
        seed_users : set[int]
            Initially activated user indices (received direct recommendation).

        Returns
        -------
        result : dict with keys:
            'adopters': set of all adopters
            'log': list of event dicts per round
            'n_rounds': number of simulation rounds
            'seed_users': the initial seed set
        """
        rng = np.random.RandomState(self.seed)
        adopters = set(seed_users)
        newly_active = set(seed_users)
        attempted = defaultdict(set)  # sender -> set of receivers already attempted
        event_log = []

        # Log initial recommendations
        for u in seed_users:
            event_log.append({
                "round": 0, "event": "recommend", "user": u, "source": "platform"
            })
            event_log.append({
                "round": 0, "event": "adopt", "user": u, "source": "platform"
            })

        for round_num in range(1, self.max_rounds + 1):
            next_active = set()

            for sender in newly_active:
                neighbors = self.propagation_graph.get(sender, [])
                for receiver in neighbors:
                    if receiver in adopters or receiver in next_active:
                        continue
                    if receiver in attempted[sender]:
                        continue  # already tried

                    attempted[sender].add(receiver)

                    if rng.random() < self.propagation_prob:
                        next_active.add(receiver)
                        event_log.append({
                            "round": round_num, "event": "adopt",
                            "user": receiver, "source": sender,
                        })

            if not next_active:
                break

            adopters.update(next_active)
            newly_active = next_active

        return {
            "adopters": adopters,
            "log": event_log,
            "n_rounds": round_num if newly_active else round_num,
            "seed_users": seed_users,
            "n_adopters": len(adopters),
        }

    def run_preference_aware(self, seed_users: set[int],
                             affinity_fn, base_adopt_prob: float = 0.5,
                             min_affinity_threshold: float = 0.0) -> dict:
        """Run preference-aware ICM with separate exposure and adoption stages.

        Parameters
        ----------
        seed_users : set[int]
            Initially activated user indices.
        affinity_fn : callable(user_idx) -> float in [0, 1]
            Preference affinity for the target item. Returns the documented
            default (e.g. 0.3) if no prediction is available — never silently
            returns 0.
        base_adopt_prob : float
            Base adoption probability, multiplied by affinity.
            adoption_prob = base_adopt_prob * affinity(user)
        min_affinity_threshold : float
            Relevance threshold (intervention). Exposed users with affinity
            below this threshold do not adopt or propagate the item.

        Returns
        -------
        result : dict with exposure and adoption tracking.
        """
        rng = np.random.RandomState(self.seed)
        adopters = set(seed_users)
        exposed = set(seed_users)
        newly_active = set(seed_users)
        attempted = defaultdict(set)
        event_log = []

        for u in seed_users:
            event_log.append({
                "round": 0, "event": "recommend", "user": u, "source": "platform"
            })
            event_log.append({
                "round": 0, "event": "adopt", "user": u, "source": "platform"
            })

        for round_num in range(1, self.max_rounds + 1):
            newly_exposed = set()
            next_active = set()

            # Stage 1: propagation attempts (exposure)
            for sender in newly_active:
                neighbors = self.propagation_graph.get(sender, [])
                for receiver in neighbors:
                    if receiver in adopters:
                        continue
                    if receiver in attempted[sender]:
                        continue

                    attempted[sender].add(receiver)

                    if rng.random() < self.propagation_prob:
                        if receiver not in exposed:
                            newly_exposed.add(receiver)
                            exposed.add(receiver)
                            event_log.append({
                                "round": round_num, "event": "expose",
                                "user": receiver, "source": sender,
                            })

            # Stage 2: adoption decision for newly exposed users
            for user in newly_exposed:
                aff = affinity_fn(user)
                if min_affinity_threshold > 0.0 and aff < min_affinity_threshold:
                    # Relevance intervention: low affinity blocks adoption and downstream propagation
                    continue

                adopt_prob = base_adopt_prob * aff
                adopt_prob = min(max(adopt_prob, 0.0), 1.0)

                if rng.random() < adopt_prob:
                    next_active.add(user)
                    adopters.add(user)
                    event_log.append({
                        "round": round_num, "event": "adopt",
                        "user": user, "source": "preference",
                    })

            if not next_active:
                break

            newly_active = next_active

        return {
            "adopters": adopters,
            "exposed": exposed,
            "log": event_log,
            "n_rounds": round_num,
            "seed_users": seed_users,
            "n_adopters": len(adopters),
            "n_exposed": len(exposed),
        }
