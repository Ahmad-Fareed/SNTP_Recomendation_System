"""
Unit tests for SNTP Assignment 2, Parts 1-4.

Tests cover:
  - Data parsing (malformed inputs, blank lines, duplicates, self-loops, leading whitespace)
  - ID mappings and sparse matrix construction
  - Train/val/test split reproducibility and guarantees
  - Item-based CF predictions on small examples
  - MF training convergence
  - Social recommender fallbacks and lambda=0 behavior
  - ICM: zero propagation, seed-only, direction, determinism
  - SIR: zero recovery (SI-like), zero transmission, state transitions
  - Metrics: MAE, RMSE, Precision@K, HitRate@K
"""

import os
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sntp.data import _parse_line, load_ratings, load_trust, EpinionsDataset
from sntp.splits import split_ratings
from sntp.metrics import mae, rmse, precision_at_k, hit_rate_at_k
from sntp.recommenders.item_based import ItemBasedCF
from sntp.recommenders.matrix_factorization import BiasedMF
from sntp.recommenders.social import SocialRecommender
from sntp.diffusion.icm import ICMSimulator
from sntp.diffusion.sir import SIRSimulator


class TestParsing(unittest.TestCase):
    """Test data parsing edge cases."""

    def test_parse_line_normal(self):
        result = _parse_line("1 100 4", 3, 1, "test")
        self.assertEqual(result, (1, 100, 4))

    def test_parse_line_leading_whitespace(self):
        """Trust file has leading spaces."""
        result = _parse_line(" 22605 42915 1", 3, 1, "test")
        self.assertEqual(result, (22605, 42915, 1))

    def test_parse_line_blank(self):
        self.assertIsNone(_parse_line("", 3, 1, "test"))
        self.assertIsNone(_parse_line("   ", 3, 1, "test"))
        self.assertIsNone(_parse_line("\n", 3, 1, "test"))

    def test_parse_line_malformed(self):
        self.assertIsNone(_parse_line("1 100", 3, 1, "test"))  # too few
        self.assertIsNone(_parse_line("1 100 4 5", 3, 1, "test"))  # too many
        self.assertIsNone(_parse_line("abc def ghi", 3, 1, "test"))  # non-integer

    def test_load_ratings_duplicates(self):
        """Duplicate policy: keep first, discard later."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1 100 4\n1 100 5\n1 101 3\n")
            f.flush()
            records, stats = load_ratings(f.name)
        os.unlink(f.name)
        self.assertEqual(len(records), 2)
        self.assertEqual(stats["duplicate_ratings"], 1)
        # First occurrence kept
        self.assertEqual(records[0], (1, 100, 4))

    def test_load_ratings_out_of_range(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1 100 0\n1 101 6\n1 102 3\n")
            f.flush()
            records, stats = load_ratings(f.name)
        os.unlink(f.name)
        self.assertEqual(len(records), 1)
        self.assertEqual(stats["out_of_range"], 2)

    def test_load_trust_self_loops(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1 2 1\n2 2 1\n3 1 1\n")
            f.flush()
            edges, stats = load_trust(f.name)
        os.unlink(f.name)
        self.assertEqual(len(edges), 2)
        self.assertEqual(stats["self_loops"], 1)

    def test_load_trust_duplicates(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1 2 1\n1 2 1\n1 3 1\n")
            f.flush()
            edges, stats = load_trust(f.name)
        os.unlink(f.name)
        self.assertEqual(len(edges), 2)
        self.assertEqual(stats["duplicate_edges"], 1)


class TestDataset(unittest.TestCase):
    """Test dataset construction and graph semantics."""

    def setUp(self):
        self.ratings = [(1, 10, 5), (1, 11, 3), (2, 10, 4), (2, 12, 2), (3, 11, 5)]
        self.trust = [(1, 2, 1), (2, 3, 1)]  # 1 trusts 2, 2 trusts 3
        self.ds = EpinionsDataset(self.ratings, self.trust, {}, {})

    def test_id_mappings(self):
        self.assertEqual(len(self.ds.user_to_idx), 3)
        self.assertEqual(len(self.ds.item_to_idx), 3)
        # Roundtrip
        for uid in [1, 2, 3]:
            idx = self.ds.user_to_idx[uid]
            self.assertEqual(self.ds.idx_to_user[idx], uid)

    def test_sparse_matrix_shape(self):
        self.assertEqual(self.ds.rating_matrix.shape, (3, 3))
        self.assertEqual(self.ds.rating_matrix.nnz, 5)

    def test_trust_graph_direction(self):
        """1 trusts 2 → trust_graph[idx(1)] contains idx(2)."""
        idx1 = self.ds.user_to_idx[1]
        idx2 = self.ds.user_to_idx[2]
        idx3 = self.ds.user_to_idx[3]
        self.assertIn(idx2, self.ds.trust_graph[idx1])
        self.assertIn(idx3, self.ds.trust_graph[idx2])

    def test_propagation_graph_direction(self):
        """1 trusts 2 → 2 can influence 1 → propagation_graph[idx(2)] contains idx(1)."""
        idx1 = self.ds.user_to_idx[1]
        idx2 = self.ds.user_to_idx[2]
        self.assertIn(idx1, self.ds.propagation_graph[idx2])

    def test_summary_keys(self):
        summary = self.ds.summary()
        self.assertIn("n_users", summary)
        self.assertIn("sparsity", summary)
        self.assertIn("rating_distribution", summary)


class TestSplits(unittest.TestCase):
    """Test reproducible splitting."""

    def setUp(self):
        # User 1: 10 ratings, User 2: 5 ratings, User 3: 2 ratings (excluded)
        self.ratings = []
        for i in range(10):
            self.ratings.append((1, 100 + i, 3))
        for i in range(5):
            self.ratings.append((2, 200 + i, 4))
        self.ratings.append((3, 300, 5))
        self.ratings.append((3, 301, 2))

    def test_reproducibility(self):
        t1, v1, te1, _ = split_ratings(self.ratings, seed=42)
        t2, v2, te2, _ = split_ratings(self.ratings, seed=42)
        self.assertEqual(t1, t2)
        self.assertEqual(v1, v2)
        self.assertEqual(te1, te2)

    def test_different_seeds(self):
        t1, _, _, _ = split_ratings(self.ratings, seed=42)
        t2, _, _, _ = split_ratings(self.ratings, seed=99)
        self.assertNotEqual(t1, t2)

    def test_guarantee_per_split(self):
        """Users with >= 3 ratings get at least 1 in each split."""
        train, val, test, info = split_ratings(self.ratings, seed=42)

        # User 3 (< 3 ratings) → all in train
        u3_train = [r for r in train if r[0] == 3]
        u3_val = [r for r in val if r[0] == 3]
        u3_test = [r for r in test if r[0] == 3]
        self.assertEqual(len(u3_train), 2)
        self.assertEqual(len(u3_val), 0)
        self.assertEqual(len(u3_test), 0)

        # User 1 (10 ratings) → at least 1 in each
        u1_train = [r for r in train if r[0] == 1]
        u1_val = [r for r in val if r[0] == 1]
        u1_test = [r for r in test if r[0] == 1]
        self.assertGreaterEqual(len(u1_train), 1)
        self.assertGreaterEqual(len(u1_val), 1)
        self.assertGreaterEqual(len(u1_test), 1)

    def test_no_leak(self):
        """No rating appears in more than one split."""
        train, val, test, _ = split_ratings(self.ratings, seed=42)
        all_records = train + val + test
        self.assertEqual(len(all_records), len(self.ratings))
        # No duplicate (user, item) across splits
        train_set = {(u, i) for u, i, _ in train}
        val_set = {(u, i) for u, i, _ in val}
        test_set = {(u, i) for u, i, _ in test}
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)

    def test_excluded_users_reported(self):
        _, _, _, info = split_ratings(self.ratings, seed=42)
        self.assertEqual(info["users_excluded_lt3_ratings"], 1)  # User 3


class TestMetrics(unittest.TestCase):
    """Test evaluation metrics."""

    def test_mae(self):
        preds = [(3.0, 4.0), (2.0, 5.0), (4.0, 4.0)]
        self.assertAlmostEqual(mae(preds), (1 + 3 + 0) / 3)

    def test_rmse(self):
        preds = [(3.0, 4.0), (2.0, 5.0), (4.0, 4.0)]
        expected = np.sqrt((1 + 9 + 0) / 3)
        self.assertAlmostEqual(rmse(preds), expected)

    def test_empty_predictions(self):
        import math
        self.assertTrue(math.isnan(mae([])))
        self.assertTrue(math.isnan(rmse([])))

    def test_precision_at_k(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 4, 6}
        self.assertAlmostEqual(precision_at_k(recommended, relevant, 5), 2 / 5)
        self.assertAlmostEqual(precision_at_k(recommended, relevant, 2), 1 / 2)

    def test_hit_rate_at_k(self):
        self.assertEqual(hit_rate_at_k([1, 2, 3], {2}, 3), 1.0)
        self.assertEqual(hit_rate_at_k([1, 2, 3], {5}, 3), 0.0)
        self.assertEqual(hit_rate_at_k([1, 2, 3], {2}, 1), 0.0)  # 2 not in top-1


class TestItemBasedCF(unittest.TestCase):
    """Test item-based CF on a small hand-checkable example."""

    def setUp(self):
        # 3 users, 4 items
        # User 0: items 0(5), 1(3), 2(4)
        # User 1: items 0(4), 1(2), 3(5)
        # User 2: items 1(5), 2(3), 3(4)
        data = [
            (0, 0, 5.0), (0, 1, 3.0), (0, 2, 4.0),
            (1, 0, 4.0), (1, 1, 2.0), (1, 3, 5.0),
            (2, 1, 5.0), (2, 2, 3.0), (2, 3, 4.0),
        ]
        rows = [d[0] for d in data]
        cols = [d[1] for d in data]
        vals = [d[2] for d in data]
        self.matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(3, 4), dtype=np.float32)

        self.user_ratings = defaultdict(list)
        self.item_ratings = defaultdict(list)
        for u, i, r in data:
            self.user_ratings[u].append((i, r))
            self.item_ratings[i].append((u, r))

        self.icf = ItemBasedCF(max_neighbors=10, min_corated=2)
        self.icf.fit(self.matrix, dict(self.user_ratings), dict(self.item_ratings))

    def test_prediction_returns_float(self):
        pred = self.icf.predict(0, 3)  # User 0 hasn't rated item 3
        self.assertIsInstance(pred, float)

    def test_prediction_range(self):
        pred = self.icf.predict(0, 3)
        self.assertGreaterEqual(pred, 1.0)
        self.assertLessEqual(pred, 5.0)

    def test_recommend_excludes_seen(self):
        recs = self.icf.recommend(0, n=5)
        seen_items = {0, 1, 2}
        for item, score in recs:
            self.assertNotIn(item, seen_items)

    def test_user_mean_fallback(self):
        """Item with no neighbors should fall back to user mean."""
        # Remove all similarities to force fallback
        icf2 = ItemBasedCF(max_neighbors=10, min_corated=100)
        icf2.fit(self.matrix, dict(self.user_ratings), dict(self.item_ratings))
        pred = icf2.predict(0, 3)
        user_mean = (5 + 3 + 4) / 3
        self.assertAlmostEqual(pred, user_mean, places=2)


class TestBiasedMF(unittest.TestCase):
    """Test MF training and prediction."""

    def setUp(self):
        self.train = [(1, 10, 5), (1, 11, 3), (2, 10, 4), (2, 12, 2), (3, 11, 5), (3, 12, 3)]
        self.u2i = {1: 0, 2: 1, 3: 2}
        self.i2i = {10: 0, 11: 1, 12: 2}
        self.mf = BiasedMF(n_factors=5, lr=0.01, reg=0.01, n_epochs=50, seed=42)
        self.mf.fit(self.train, 3, 3, self.u2i, self.i2i)

    def test_predictions_in_range(self):
        for u in range(3):
            for i in range(3):
                pred = self.mf.predict(u, i)
                self.assertGreaterEqual(pred, 1.0)
                self.assertLessEqual(pred, 5.0)

    def test_training_reduces_error(self):
        """Training RMSE should decrease over epochs."""
        first = self.mf.epoch_history[0]["train_rmse"]
        last = self.mf.epoch_history[-1]["train_rmse"]
        self.assertLess(last, first)

    def test_recommend_excludes_seen(self):
        recs = self.mf.recommend(0, n=5)
        seen = {0, 1}  # User 0 trained on items 10->0, 11->1
        for item, _ in recs:
            self.assertNotIn(item, seen)


class TestSocialRecommender(unittest.TestCase):
    """Test social extension behavior."""

    def setUp(self):
        # Simple baseline that returns fixed predictions
        class DummyBaseline:
            global_mean = 3.0
            n_items = 3
            def predict(self, u, i):
                return 3.5
            def recommend(self, u, n, exclude=None):
                return [(i, 3.5) for i in range(self.n_items) if i not in (exclude or set())][:n]

        self.baseline = DummyBaseline()
        # User 0 trusts users 1 and 2
        self.trust = {0: [1, 2]}

    def test_lambda_zero_equals_baseline(self):
        """At lambda=0, social prediction should equal baseline."""
        sr = SocialRecommender(self.baseline, self.trust, social_weight=0.0)
        sr.set_training_data({1: {0: 5.0}, 2: {0: 4.0}})
        pred = sr.predict(0, 0)
        self.assertAlmostEqual(pred, 3.5)

    def test_social_weight_effect(self):
        """Higher lambda should shift prediction toward trust neighbors."""
        sr_low = SocialRecommender(self.baseline, self.trust, social_weight=0.1)
        sr_high = SocialRecommender(self.baseline, self.trust, social_weight=0.9)
        train_data = {1: {0: 5.0}, 2: {0: 5.0}}
        sr_low.set_training_data(train_data)
        sr_high.set_training_data(train_data)

        pred_low = sr_low.predict(0, 0)
        pred_high = sr_high.predict(0, 0)
        # Both trusted users rate item 0 at 5.0, baseline at 3.5
        # Higher lambda → closer to 5.0
        self.assertGreater(pred_high, pred_low)

    def test_missing_neighbor_ratings(self):
        """If no trusted neighbor rated the item, use baseline only."""
        sr = SocialRecommender(self.baseline, self.trust, social_weight=0.5)
        sr.set_training_data({1: {}, 2: {}})  # No ratings
        pred = sr.predict(0, 0)
        self.assertAlmostEqual(pred, 3.5)  # Falls back to baseline


class TestICM(unittest.TestCase):
    """Test Independent Cascade Model on hand-checkable graphs."""

    def _linear_graph(self):
        """A -> B -> C (A can influence B, B can influence C)."""
        # A=0, B=1, C=2
        # propagation_graph[0] = [1], propagation_graph[1] = [2]
        return {0: [1], 1: [2]}

    def test_zero_propagation(self):
        """With p=0, only seed users should adopt."""
        icm = ICMSimulator(self._linear_graph(), propagation_prob=0.0, seed=42)
        result = icm.run_canonical({0})
        self.assertEqual(result["adopters"], {0})

    def test_full_propagation(self):
        """With p=1, cascade should reach all reachable nodes."""
        icm = ICMSimulator(self._linear_graph(), propagation_prob=1.0, seed=42)
        result = icm.run_canonical({0})
        self.assertEqual(result["adopters"], {0, 1, 2})

    def test_direction_respected(self):
        """Propagation only follows graph direction."""
        # Only 0->1, 1->2; starting from 2 should NOT reach 0 or 1
        icm = ICMSimulator(self._linear_graph(), propagation_prob=1.0, seed=42)
        result = icm.run_canonical({2})
        self.assertEqual(result["adopters"], {2})

    def test_deterministic_with_same_seed(self):
        graph = {0: [1, 2, 3], 1: [4, 5]}
        icm1 = ICMSimulator(graph, propagation_prob=0.5, seed=123)
        icm2 = ICMSimulator(graph, propagation_prob=0.5, seed=123)
        r1 = icm1.run_canonical({0})
        r2 = icm2.run_canonical({0})
        self.assertEqual(r1["adopters"], r2["adopters"])

    def test_preference_aware_zero_affinity(self):
        """Zero affinity → no adoptions beyond seeds (exposure but no adoption)."""
        icm = ICMSimulator(self._linear_graph(), propagation_prob=1.0, seed=42)
        result = icm.run_preference_aware(
            {0}, affinity_fn=lambda u: 0.0, base_adopt_prob=1.0,
        )
        # Seed adopts, B gets exposed but affinity=0 → 0 adoption probability
        self.assertEqual(result["adopters"], {0})
        # B should be exposed though
        self.assertIn(1, result["exposed"])


class TestSIR(unittest.TestCase):
    """Test SIR model on hand-checkable graphs."""

    def _linear_graph(self):
        return {0: [1], 1: [2]}

    def test_zero_transmission(self):
        """β=0 → no spread beyond seeds."""
        sir = SIRSimulator(self._linear_graph(), transmission_prob=0.0,
                          recovery_prob=0.5, seed=42)
        result = sir.run({0})
        self.assertEqual(result["ever_infected"], {0})

    def test_full_transmission_no_recovery(self):
        """β=1, γ=0 → SI-like, reaches all reachable nodes."""
        sir = SIRSimulator(self._linear_graph(), transmission_prob=1.0,
                          recovery_prob=0.0, seed=42)
        result = sir.run({0})
        self.assertEqual(result["ever_infected"], {0, 1, 2})
        # γ=0 → no one recovers
        self.assertEqual(len(result["final_recovered"]), 0)

    def test_recovery_works(self):
        """With γ=1 (immediate recovery), infected users recover each round."""
        sir = SIRSimulator(self._linear_graph(), transmission_prob=1.0,
                          recovery_prob=1.0, max_rounds=10, seed=42)
        result = sir.run({0})
        # Seed infects node 1 in round 1, but seed recovers in round 1
        # Node 1 can infect node 2 in round 2 but recovers in round 2
        # All eventually recover
        self.assertGreater(len(result["final_recovered"]), 0)

    def test_direction_respected(self):
        """Starting from leaf node, no spread in reverse direction."""
        sir = SIRSimulator(self._linear_graph(), transmission_prob=1.0,
                          recovery_prob=0.0, seed=42)
        result = sir.run({2})
        self.assertEqual(result["ever_infected"], {2})

    def test_state_history_consistent(self):
        """S + I + R should remain constant across rounds."""
        sir = SIRSimulator(self._linear_graph(), transmission_prob=0.5,
                          recovery_prob=0.3, seed=42)
        result = sir.run({0}, all_users={0, 1, 2})
        total = 3
        for state in result["state_history"]:
            self.assertEqual(state["S"] + state["I"] + state["R"], total)


if __name__ == "__main__":
    unittest.main()
