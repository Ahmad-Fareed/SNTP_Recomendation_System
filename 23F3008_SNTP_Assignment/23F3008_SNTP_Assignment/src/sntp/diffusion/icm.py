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

Common random numbers (CRN):
  A simulator draws one activation threshold per directed edge and one adoption
  threshold per user when it is constructed.  An edge activates when its
  threshold is below p.  The thresholds depend only on the trial seed and the
  graph, not on the seed set, the propagation probability, or the cascade path.
  Consequently the same ``ICMRandomness`` object (or a fresh simulator with the
  same seed and graph) gives coupled outcomes when seed sets or p are varied.
  This is what makes paired comparisons and the CELF lazy bound meaningful.
  Resetting a sequential RNG to the same integer seed inside every run would
  NOT provide this property because the draw positions depend on the cascade.

Events tracked:
  - 'recommend': user received a direct platform recommendation.
  - 'expose': user became exposed through a neighbor.
  - 'adopt': user adopted the item.

The dataset has NO observed adoption timestamps or diffusion histories.
All propagation results are simulated and must be labeled as such.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class ICMRandomness:
    """Reusable common random outcomes for one ICM trial.

    Attributes
    ----------
    seed : int
        Trial seed used to generate the thresholds.
    edge_thresholds : dict[int, np.ndarray]
        For each sender, one float32 threshold per outgoing edge, aligned with
        the sender's neighbor list in the propagation graph.
    adoption_thresholds : np.ndarray
        One float32 adoption threshold per user index.
    """

    def __init__(self, propagation_graph: dict, seed: int):
        self.seed = int(seed)
        rng = np.random.RandomState(self.seed)
        self.edge_thresholds: dict[int, np.ndarray] = {}
        max_node = -1
        for sender in sorted(propagation_graph):
            sender = int(sender)
            neighbors = propagation_graph[sender]
            max_node = max(max_node, sender)
            if neighbors:
                max_node = max(max_node, max(int(v) for v in neighbors))
            self.edge_thresholds[sender] = rng.random_sample(
                len(neighbors)
            ).astype(np.float32)
        if max_node < 0:
            max_node = 0
        self.adoption_thresholds = rng.random_sample(max_node + 1).astype(np.float32)


class ICMSimulator:
    """Independent Cascade Model simulator."""

    def __init__(self, propagation_graph: dict, propagation_prob: float = 0.1,
                 max_rounds: int = 100, seed: int = 42,
                 common_randomness: ICMRandomness | None = None):
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
        common_randomness : ICMRandomness, optional
            Pre-generated outcomes to reuse across comparable runs.  When
            supplied, ``seed`` is only retained for provenance and must match
            the randomness object's seed.
        """
        self.propagation_graph = propagation_graph
        self.propagation_prob = propagation_prob
        self.max_rounds = max_rounds
        self.seed = int(seed)
        if common_randomness is None:
            common_randomness = ICMRandomness(propagation_graph, self.seed)
        elif common_randomness.seed != self.seed:
            raise ValueError(
                "common_randomness seed does not match simulator seed: "
                f"{common_randomness.seed} != {self.seed}"
            )
        self.randomness = common_randomness

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
        seed_users = set(seed_users)
        p = float(self.propagation_prob)
        adopters = set(seed_users)
        newly_active = set(seed_users)
        attempted = defaultdict(set)  # sender -> set of receivers already attempted
        event_log = []
        last_adoption_round = 0
        rounds_attempted = 0

        # Log initial recommendations
        for u in seed_users:
            event_log.append({
                "round": 0, "event": "recommend", "user": u, "source": "platform"
            })
            event_log.append({
                "round": 0, "event": "adopt", "user": u, "source": "platform"
            })

        for round_num in range(1, self.max_rounds + 1):
            rounds_attempted = round_num
            next_active = set()

            for sender in newly_active:
                neighbors = self.propagation_graph.get(sender, [])
                thresholds = self.randomness.edge_thresholds.get(sender)
                if thresholds is None:
                    continue
                for pos, receiver in enumerate(neighbors):
                    if receiver in adopters or receiver in next_active:
                        continue
                    if receiver in attempted[sender]:
                        continue  # already tried

                    attempted[sender].add(receiver)

                    if float(thresholds[pos]) < p:
                        next_active.add(receiver)
                        event_log.append({
                            "round": round_num, "event": "adopt",
                            "user": receiver, "source": sender,
                        })

            if not next_active:
                break

            adopters.update(next_active)
            newly_active = next_active
            last_adoption_round = round_num

        return {
            "adopters": adopters,
            "log": event_log,
            # Cascade adoption depth: the last round in which at least one
            # non-seed user adopted.  The terminal no-adoption round is not
            # counted.
            "n_rounds": last_adoption_round,
            "rounds_attempted": rounds_attempted,
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

        Modelling note
        --------------
        A non-seed user receives at most one adoption decision after first
        exposure.  This keeps the preference-aware variant a user-level
        decision model rather than an edge-level independent-cascade model.
        ``n_rounds`` reports adoption depth (the last round with an adoption).

        Returns
        -------
        result : dict with exposure and adoption tracking.
        """
        seed_users = set(seed_users)
        p = float(self.propagation_prob)
        adopters = set(seed_users)
        exposed = set(seed_users)
        newly_active = set(seed_users)
        attempted = defaultdict(set)
        event_log = []
        last_adoption_round = 0
        rounds_attempted = 0

        for u in seed_users:
            event_log.append({
                "round": 0, "event": "recommend", "user": u, "source": "platform"
            })
            event_log.append({
                "round": 0, "event": "adopt", "user": u, "source": "platform"
            })

        for round_num in range(1, self.max_rounds + 1):
            rounds_attempted = round_num
            newly_exposed = set()
            next_active = set()

            # Stage 1: propagation attempts (exposure)
            for sender in newly_active:
                neighbors = self.propagation_graph.get(sender, [])
                thresholds = self.randomness.edge_thresholds.get(sender)
                if thresholds is None:
                    continue
                for pos, receiver in enumerate(neighbors):
                    if receiver in adopters:
                        continue
                    if receiver in attempted[sender]:
                        continue

                    attempted[sender].add(receiver)

                    if float(thresholds[pos]) < p:
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
                threshold = self.randomness.adoption_thresholds
                if user < threshold.size and float(threshold[user]) < adopt_prob:
                    next_active.add(user)
                    adopters.add(user)
                    event_log.append({
                        "round": round_num, "event": "adopt",
                        "user": user, "source": "preference",
                    })

            if not next_active:
                break

            newly_active = next_active
            last_adoption_round = round_num

        return {
            "adopters": adopters,
            "exposed": exposed,
            "log": event_log,
            "n_rounds": last_adoption_round,
            "rounds_attempted": rounds_attempted,
            "seed_users": seed_users,
            "n_adopters": len(adopters),
            "n_exposed": len(exposed),
        }
