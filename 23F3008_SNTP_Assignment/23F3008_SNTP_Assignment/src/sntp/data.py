"""
Data loading, validation, and representation for the Epinions dataset.

Record formats:
  - Ratings: user_id  item_id  rating  (space-separated integers)
  - Trust:   user_id  trusted_user_id  trust_value  (space-separated, may have leading whitespace)

Edge semantics:
  A row "A  B  1" in the trust file means "user A trusts user B."
  For information propagation, B can influence A, so the propagation
  edge direction is B -> A.

Data categories:
  - Observed:  recorded ratings and trust edges loaded from files.
  - Inferred:  predicted ratings, estimated similarities (produced later).
  - Simulated: exposures, adoptions, propagation outcomes (produced later).
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_line(line: str, expected_fields: int, line_number: int, source: str):
    """Parse a whitespace-separated line into integer tokens.

    Returns a tuple of ints on success or None on failure (with a log message).
    Handles leading/trailing whitespace and blank lines.
    """
    stripped = line.strip()
    if not stripped:
        return None  # blank line — silently skip
    parts = stripped.split()
    if len(parts) != expected_fields:
        logger.warning(
            "%s line %d: expected %d fields, got %d — skipping: %r",
            source, line_number, expected_fields, len(parts), stripped,
        )
        return None
    try:
        return tuple(int(p) for p in parts)
    except ValueError:
        logger.warning(
            "%s line %d: non-integer value — skipping: %r",
            source, line_number, stripped,
        )
        return None


def load_ratings(path: str | Path, rating_min: int = 1, rating_max: int = 5):
    """Load ratings from a text file.

    Returns
    -------
    records : list[tuple[int, int, int]]
        Valid (user_id, item_id, rating) triples.
    stats : dict
        Parsing statistics including malformed, out-of-range, and duplicate counts.
    """
    path = Path(path)
    records = []
    seen = {}  # (user, item) -> line_number of first occurrence
    stats = {
        "total_lines": 0,
        "blank_lines": 0,
        "malformed_lines": 0,
        "out_of_range": 0,
        "duplicate_ratings": 0,
        "valid_records": 0,
    }
    duplicate_details = []

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            stats["total_lines"] += 1
            parsed = _parse_line(raw_line, 3, lineno, "ratings")
            if parsed is None:
                if raw_line.strip() == "":
                    stats["blank_lines"] += 1
                else:
                    stats["malformed_lines"] += 1
                continue

            uid, iid, rating = parsed

            # Rating-scale validation
            if not (rating_min <= rating <= rating_max):
                stats["out_of_range"] += 1
                logger.warning(
                    "ratings line %d: rating %d outside [%d, %d] — skipping",
                    lineno, rating, rating_min, rating_max,
                )
                continue

            # Duplicate policy: keep the FIRST occurrence, log the duplicate
            key = (uid, iid)
            if key in seen:
                stats["duplicate_ratings"] += 1
                duplicate_details.append(
                    (lineno, uid, iid, rating, seen[key])
                )
                continue

            seen[key] = lineno
            records.append((uid, iid, rating))
            stats["valid_records"] += 1

    if duplicate_details:
        logger.info(
            "Duplicate rating policy: kept first occurrence, discarded %d duplicates.",
            len(duplicate_details),
        )
    logger.info("Ratings loaded: %s", stats)
    return records, stats


def load_trust(path: str | Path):
    """Load trust edges from a text file.

    Returns
    -------
    edges : list[tuple[int, int, int]]
        Valid (user_id, trusted_user_id, trust_value) triples.
    stats : dict
        Parsing statistics.
    """
    path = Path(path)
    edges = []
    seen = set()
    stats = {
        "total_lines": 0,
        "blank_lines": 0,
        "malformed_lines": 0,
        "self_loops": 0,
        "duplicate_edges": 0,
        "valid_edges": 0,
        "unique_trust_values": set(),
    }

    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            stats["total_lines"] += 1
            parsed = _parse_line(raw_line, 3, lineno, "trust")
            if parsed is None:
                if raw_line.strip() == "":
                    stats["blank_lines"] += 1
                else:
                    stats["malformed_lines"] += 1
                continue

            uid, tid, tval = parsed
            stats["unique_trust_values"].add(tval)

            # Self-loop check
            if uid == tid:
                stats["self_loops"] += 1
                logger.debug("trust line %d: self-loop user %d — skipping", lineno, uid)
                continue

            # Duplicate edge check
            key = (uid, tid)
            if key in seen:
                stats["duplicate_edges"] += 1
                continue

            seen.add(key)
            edges.append((uid, tid, tval))
            stats["valid_edges"] += 1

    # Convert set to sorted list for serialization
    stats["unique_trust_values"] = sorted(stats["unique_trust_values"])
    logger.info("Trust loaded: %s", stats)
    return edges, stats


# ---------------------------------------------------------------------------
# Dataset — unified representation
# ---------------------------------------------------------------------------

class EpinionsDataset:
    """Container for the Epinions social-recommendation dataset.

    Attributes
    ----------
    ratings : list[tuple[int, int, int]]
        (user_id, item_id, rating) triples.
    trust_edges : list[tuple[int, int, int]]
        (user_id, trusted_user_id, trust_value) triples.
    user_to_idx / idx_to_user : dict
        Bidirectional user-ID ↔ integer-index mappings.
    item_to_idx / idx_to_item : dict
        Bidirectional item-ID ↔ integer-index mappings.
    rating_matrix : scipy.sparse.csr_matrix
        Sparse (n_users × n_items) rating matrix (observed entries only).
    user_ratings : dict[int, list[tuple[int, int]]]
        user_idx -> [(item_idx, rating), ...] sorted by item_idx.
    item_ratings : dict[int, list[tuple[int, int]]]
        item_idx -> [(user_idx, rating), ...] sorted by user_idx.
    trust_graph : dict[int, list[int]]
        Adjacency list: trust_graph[u_idx] = list of user indices that u trusts.
    propagation_graph : dict[int, list[int]]
        Adjacency for information flow: propagation_graph[u_idx] = list of user
        indices that u can *influence* (i.e., who trust u).
    """

    def __init__(self, ratings, trust_edges, rating_stats, trust_stats):
        self.raw_ratings = ratings
        self.raw_trust = trust_edges
        self.rating_stats = rating_stats
        self.trust_stats = trust_stats

        # --- Build ID mappings ---
        all_users = set()
        all_items = set()
        for uid, iid, _ in ratings:
            all_users.add(uid)
            all_items.add(iid)
        for uid, tid, _ in trust_edges:
            all_users.add(uid)
            all_users.add(tid)

        self.user_to_idx = {u: i for i, u in enumerate(sorted(all_users))}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}
        self.item_to_idx = {it: i for i, it in enumerate(sorted(all_items))}
        self.idx_to_item = {i: it for it, i in self.item_to_idx.items()}

        self.n_users = len(self.user_to_idx)
        self.n_items = len(self.item_to_idx)

        # --- Sparse rating matrix ---
        rows, cols, vals = [], [], []
        for uid, iid, r in ratings:
            rows.append(self.user_to_idx[uid])
            cols.append(self.item_to_idx[iid])
            vals.append(r)

        self.rating_matrix = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

        # --- User and item rating histories ---
        self.user_ratings = defaultdict(list)
        self.item_ratings = defaultdict(list)
        for uid, iid, r in ratings:
            uidx = self.user_to_idx[uid]
            iidx = self.item_to_idx[iid]
            self.user_ratings[uidx].append((iidx, r))
            self.item_ratings[iidx].append((uidx, r))

        # Sort for deterministic access
        for k in self.user_ratings:
            self.user_ratings[k].sort()
        for k in self.item_ratings:
            self.item_ratings[k].sort()

        # --- Trust graph (directed) ---
        # trust_graph[u] = [v, ...] means u trusts v
        self.trust_graph = defaultdict(list)
        # propagation_graph[v] = [u, ...] means v can influence u (because u trusts v)
        self.propagation_graph = defaultdict(list)

        for uid, tid, _ in trust_edges:
            uidx = self.user_to_idx[uid]
            tidx = self.user_to_idx[tid]
            self.trust_graph[uidx].append(tidx)
            self.propagation_graph[tidx].append(uidx)

        # Compute user sets for overlap analysis
        self._rating_users = {self.user_to_idx[uid] for uid, _, _ in ratings}
        self._trust_users = set()
        for uid, tid, _ in trust_edges:
            self._trust_users.add(self.user_to_idx[uid])
            self._trust_users.add(self.user_to_idx[tid])

    def summary(self) -> dict:
        """Return a dictionary of dataset statistics."""
        ratings_per_user = [len(self.user_ratings[u]) for u in range(self.n_users)
                           if u in self.user_ratings]
        ratings_per_item = [len(self.item_ratings[i]) for i in range(self.n_items)
                           if i in self.item_ratings]
        trust_per_user = [len(self.trust_graph[u]) for u in range(self.n_users)
                         if u in self.trust_graph]

        # Rating distribution
        all_ratings = [r for _, _, r in self.raw_ratings]
        rating_counts = {}
        for r in sorted(set(all_ratings)):
            rating_counts[r] = all_ratings.count(r)

        # Sparsity
        total_possible = self.n_users * self.n_items
        sparsity = 1.0 - (len(self.raw_ratings) / total_possible) if total_possible > 0 else 0.0

        # Overlap
        overlap = self._rating_users & self._trust_users
        rating_only = self._rating_users - self._trust_users
        trust_only = self._trust_users - self._rating_users

        # Reciprocal trust edges
        trust_set = set()
        for u in self.trust_graph:
            for v in self.trust_graph[u]:
                trust_set.add((u, v))
        reciprocal = sum(1 for (u, v) in trust_set if (v, u) in trust_set) // 2

        return {
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_ratings": len(self.raw_ratings),
            "n_trust_edges": len(self.raw_trust),
            "sparsity": round(sparsity, 6),
            "rating_distribution": rating_counts,
            "ratings_per_user": {
                "min": int(np.min(ratings_per_user)) if ratings_per_user else 0,
                "max": int(np.max(ratings_per_user)) if ratings_per_user else 0,
                "mean": round(float(np.mean(ratings_per_user)), 2) if ratings_per_user else 0,
                "median": round(float(np.median(ratings_per_user)), 2) if ratings_per_user else 0,
            },
            "ratings_per_item": {
                "min": int(np.min(ratings_per_item)) if ratings_per_item else 0,
                "max": int(np.max(ratings_per_item)) if ratings_per_item else 0,
                "mean": round(float(np.mean(ratings_per_item)), 2) if ratings_per_item else 0,
                "median": round(float(np.median(ratings_per_item)), 2) if ratings_per_item else 0,
            },
            "trust_per_user": {
                "min": int(np.min(trust_per_user)) if trust_per_user else 0,
                "max": int(np.max(trust_per_user)) if trust_per_user else 0,
                "mean": round(float(np.mean(trust_per_user)), 2) if trust_per_user else 0,
                "median": round(float(np.median(trust_per_user)), 2) if trust_per_user else 0,
            },
            "user_overlap": {
                "both_rating_and_trust": len(overlap),
                "rating_only": len(rating_only),
                "trust_only": len(trust_only),
            },
            "reciprocal_trust_edges": reciprocal,
            "trust_values_observed": self.trust_stats.get("unique_trust_values", []),
            "parsing": {
                "ratings": self.rating_stats,
                "trust": self.trust_stats,
            },
        }

    def get_reciprocal_trust_graph(self):
        """Return adjacency list containing only reciprocal trust edges."""
        trust_set = set()
        for u in self.trust_graph:
            for v in self.trust_graph[u]:
                trust_set.add((u, v))

        reciprocal = defaultdict(list)
        for (u, v) in trust_set:
            if (v, u) in trust_set:
                reciprocal[u].append(v)
        return dict(reciprocal)

    def get_ego_network(self, center_idx: int, hops: int = 2):
        """Return the subgraph of users within `hops` of center_idx in the trust graph."""
        visited = {center_idx}
        frontier = {center_idx}
        for _ in range(hops):
            next_frontier = set()
            for u in frontier:
                for v in self.trust_graph.get(u, []):
                    if v not in visited:
                        visited.add(v)
                        next_frontier.add(v)
                # Also follow reverse edges for reachability
                for v in self.propagation_graph.get(u, []):
                    if v not in visited:
                        visited.add(v)
                        next_frontier.add(v)
            frontier = next_frontier
            if not frontier:
                break

        # Build sub-adjacency
        sub_trust = defaultdict(list)
        sub_prop = defaultdict(list)
        for u in visited:
            for v in self.trust_graph.get(u, []):
                if v in visited:
                    sub_trust[u].append(v)
            for v in self.propagation_graph.get(u, []):
                if v in visited:
                    sub_prop[u].append(v)

        return visited, dict(sub_trust), dict(sub_prop)


def load_dataset(data_dir: str | Path) -> EpinionsDataset:
    """Auto-discover and load the Epinions dataset from a directory tree.

    Searches for ratings_data.txt and trust_data.txt under data_dir,
    using only one copy of each file even if duplicates exist.
    """
    data_dir = Path(data_dir)
    # Discover files
    rating_files = list(data_dir.rglob("ratings_data.txt"))
    trust_files = list(data_dir.rglob("trust_data.txt"))

    if not rating_files:
        raise FileNotFoundError(f"No ratings_data.txt found under {data_dir}")
    if not trust_files:
        raise FileNotFoundError(f"No trust_data.txt found under {data_dir}")

    # Use the first occurrence of each (duplicates confirmed identical)
    rating_path = rating_files[0]
    trust_path = trust_files[0]
    logger.info("Using ratings file: %s", rating_path)
    logger.info("Using trust file: %s", trust_path)

    if len(trust_files) > 1:
        logger.info(
            "Found %d trust files; using only %s (duplicates not merged).",
            len(trust_files), trust_path,
        )

    ratings, r_stats = load_ratings(rating_path)
    trust_edges, t_stats = load_trust(trust_path)

    return EpinionsDataset(ratings, trust_edges, r_stats, t_stats)
