"""
SIR (Susceptible-Infected-Recovered) epidemic model for information diffusion.

States:
  - Susceptible (S): has not adopted the item and is eligible to adopt.
  - Infected (I): has adopted and is currently spreading the item.
  - Recovered (R): has stopped spreading; cannot become susceptible again.

Transitions (per synchronous round):
  1. Transmission: each Infected user attempts to infect each Susceptible neighbor
     with probability β (transmission_prob). Optionally scaled by preference affinity.
  2. Recovery: each Infected user recovers with probability γ (recovery_prob).
     Recovery means the user stops actively propagating the item.
     At γ = 0: no recovery → SI-like behavior (infected forever, never stop spreading).
  3. Newly infected users can spread starting in the NEXT round.

Timing convention:
  - At the start of each round, transmit from users who were Infected at round start.
  - Then process recovery for all Infected users (including those from prior rounds).
  - Newly infected users join the Infected pool for the next round.

The dataset has NO observed adoption timestamps. All results are simulated.
Recovery represents "user stops actively promoting the item" — a modeling assumption.
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class SIRSimulator:
    """SIR epidemic model for item adoption diffusion."""

    def __init__(self, propagation_graph: dict, transmission_prob: float = 0.1,
                 recovery_prob: float = 0.2, max_rounds: int = 100, seed: int = 42):
        """
        Parameters
        ----------
        propagation_graph : dict[int, list[int]]
            Adjacency: propagation_graph[v] = users that v can influence.
        transmission_prob : float
            β — probability of infection per edge per round.
        recovery_prob : float
            γ — probability of an Infected user recovering each round.
            At γ = 0, behavior is SI-like (no recovery).
        max_rounds : int
            Safety limit.
        seed : int
            Random seed.
        """
        self.propagation_graph = propagation_graph
        self.transmission_prob = transmission_prob
        self.recovery_prob = recovery_prob
        self.max_rounds = max_rounds
        self.seed = seed

    def run(self, seed_users: set[int], affinity_fn=None,
            base_adopt_prob: float = 1.0,
            all_users: set[int] | None = None) -> dict:
        """Run SIR simulation.

        Parameters
        ----------
        seed_users : set[int]
            Initially infected user indices.
        affinity_fn : callable(user_idx) -> float in [0, 1], optional
            If provided, transmission probability is β * base_adopt_prob * affinity(user).
            If None, uses flat β.
        base_adopt_prob : float
            Scaling factor for preference-aware transmission.
        all_users : set[int] or None
            Full set of user indices. If None, inferred from propagation graph.

        Returns
        -------
        result : dict with state history, event log, and summary statistics.
        """
        seed_users = set(seed_users)
        rng = np.random.RandomState(self.seed)

        # Initialize states
        if all_users is None:
            all_users_set = set()
            for u in self.propagation_graph:
                all_users_set.add(u)
                for v in self.propagation_graph[u]:
                    all_users_set.add(v)
        else:
            all_users_set = set(all_users)

        susceptible = all_users_set - seed_users
        infected = set(seed_users)
        recovered = set()
        ever_infected = set(seed_users)

        event_log = []
        state_history = []
        last_infection_round = 0
        rounds_attempted = 0

        # Log initial infections
        for u in seed_users:
            event_log.append({
                "round": 0, "event": "infect", "user": u, "source": "platform"
            })

        state_history.append({
            "round": 0,
            "S": len(susceptible),
            "I": len(infected),
            "R": len(recovered),
        })

        for round_num in range(1, self.max_rounds + 1):
            rounds_attempted = round_num
            newly_infected = set()

            # --- Transmission phase ---
            # Only users infected at the START of this round attempt transmission
            current_infected = list(infected)

            for sender in current_infected:
                neighbors = self.propagation_graph.get(sender, [])
                for receiver in neighbors:
                    if receiver not in susceptible:
                        continue

                    # Compute transmission probability
                    t_prob = self.transmission_prob
                    if affinity_fn is not None:
                        aff = affinity_fn(receiver)
                        t_prob = self.transmission_prob * base_adopt_prob * aff
                        t_prob = min(max(t_prob, 0.0), 1.0)

                    if rng.random() < t_prob:
                        newly_infected.add(receiver)
                        ever_infected.add(receiver)
                        event_log.append({
                            "round": round_num, "event": "infect",
                            "user": receiver, "source": sender,
                        })

            # --- Recovery phase ---
            newly_recovered = set()
            for u in current_infected:
                if self.recovery_prob > 0 and rng.random() < self.recovery_prob:
                    newly_recovered.add(u)
                    event_log.append({
                        "round": round_num, "event": "recover", "user": u,
                    })

            # --- Update states ---
            susceptible -= newly_infected
            infected = (infected - newly_recovered) | newly_infected
            recovered |= newly_recovered
            if newly_infected:
                last_infection_round = round_num

            state_history.append({
                "round": round_num,
                "S": len(susceptible),
                "I": len(infected),
                "R": len(recovered),
            })

            # Stop if no infected users remain
            if not infected:
                break

        return {
            "ever_infected": ever_infected,
            "final_infected": infected,
            "final_recovered": recovered,
            "final_susceptible": susceptible,
            "state_history": state_history,
            "log": event_log,
            # Infection depth: last round with a new infection; the terminal
            # no-transmission round is not counted.
            "n_rounds": last_infection_round,
            "rounds_attempted": rounds_attempted,
            "seed_users": seed_users,
            "n_ever_infected": len(ever_infected),
            "n_final_recovered": len(recovered),
        }
