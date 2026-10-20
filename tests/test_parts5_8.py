"""
Tests for SNTP Assignment 2: Parts 5-8.

Covers:
- Part 5: Influence Maximization (Random, Degree Centrality, Greedy, Overlap)
- Part 6: Group Recommendation (Average, Least Misery, Most Pleasure, Satisfaction)
- Part 7: Comprehensive Evaluation Metrics (Recall, F1, nDCG, Coverage, Diversity)
- Part 8: Intervention (Min Affinity Threshold) & CELF Optimization
"""

import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse

from sntp.metrics import (
    precision_at_k,
    hit_rate_at_k,
    recall_at_k,
    f1_at_k,
    ndcg_at_k,
    recommendation_coverage,
    intra_list_diversity,
    intra_list_diversity_from_features,
    mean_intra_list_diversity,
)
from sntp.influence import (
    estimate_spread,
    random_seed_selection,
    degree_centrality_seed_selection,
    greedy_influence_maximization,
    compute_adopter_overlap,
)
from sntp.groups import (
    generate_groups,
    score_group_items,
    recommend_for_group,
    evaluate_group_satisfaction,
)
from sntp.diffusion.icm import ICMSimulator


class TestNewMetrics(unittest.TestCase):
    """Test new evaluation metrics for Part 7."""

    def test_recall_at_k(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 4, 6, 8}  # 4 relevant items in total
        # top 5 has items 2 and 4 (2 hits out of 4 relevant)
        self.assertAlmostEqual(recall_at_k(recommended, relevant, 5), 2 / 4)
        # top 2 has item 2 (1 hit out of 4 relevant)
        self.assertAlmostEqual(recall_at_k(recommended, relevant, 2), 1 / 4)
        # empty or k=0
        self.assertEqual(recall_at_k([], relevant, 5), 0.0)
        self.assertEqual(recall_at_k(recommended, set(), 5), 0.0)
        self.assertEqual(recall_at_k(recommended, relevant, 0), 0.0)

    def test_f1_at_k(self):
        p = 0.5
        r = 0.5
        self.assertAlmostEqual(f1_at_k(p, r), 0.5)
        self.assertEqual(f1_at_k(0.0, 0.5), 0.0)
        self.assertEqual(f1_at_k(0.5, 0.0), 0.0)
        self.assertEqual(f1_at_k(0.0, 0.0), 0.0)

    def test_ndcg_at_k(self):
        # Perfect ranking: all top slots are relevant
        recommended = [1, 2, 3, 4]
        relevant = {1, 2}
        self.assertAlmostEqual(ndcg_at_k(recommended, relevant, 2), 1.0)

        # Reversed ranking: slots 2 and 3 are relevant instead of 0 and 1
        # DCG@2: 0 (since items 1, 2 not in relevant)
        self.assertEqual(ndcg_at_k([5, 6, 1, 2], relevant, 2), 0.0)

        # Partial ranking: item 1 at rank 0, non-relevant at rank 1, item 2 at rank 2
        # DCG@3 = 1/log2(2) + 0 + 1/log2(4) = 1.0 + 0.5 = 1.5
        # IDCG@3 = 1/log2(2) + 1/log2(3) = 1.0 + 0.63092975 = 1.63092975
        expected_ndcg = (1.0 + 1.0 / np.log2(4)) / (1.0 + 1.0 / np.log2(3))
        self.assertAlmostEqual(ndcg_at_k([1, 5, 2], relevant, 3), expected_ndcg, places=5)

    def test_recommendation_coverage(self):
        all_recs = [[0, 1, 2], [1, 2, 3], [0, 4]]
        n_catalog = 10
        # Unique items recommended: {0, 1, 2, 3, 4} -> 5 / 10 = 0.5
        self.assertAlmostEqual(recommendation_coverage(all_recs, n_catalog), 0.5)
        self.assertEqual(recommendation_coverage([], 10), 0.0)

    def test_intra_list_diversity(self):
        # All distinct items with None similarity matrix -> diversity = 1.0
        self.assertAlmostEqual(intra_list_diversity([0, 1, 2, 3]), 1.0)
        # Single item -> diversity = 0.0
        self.assertEqual(intra_list_diversity([0]), 0.0)

        # With explicit similarity matrix
        sim_matrix = np.array([
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.5],
            [0.2, 0.5, 1.0],
        ])
        # Pairs: (0, 1): dist = 0.2; (0, 2): dist = 0.8; (1, 2): dist = 0.5
        # mean dist = (0.2 + 0.8 + 0.5) / 3 = 0.5
        self.assertAlmostEqual(intra_list_diversity([0, 1, 2], sim_matrix), 0.5)

    def test_feature_based_intra_list_diversity(self):
        # Rows are item feature vectors (here: two binary user features).
        features = sparse.csr_matrix([
            [1.0, 0.0],   # item 0
            [0.0, 1.0],   # item 1
            [1.0, 1.0],   # item 2
            [0.0, 0.0],   # item 3 (zero vector)
        ])
        # Orthogonal item vectors -> cosine 0 -> distance 1.
        self.assertAlmostEqual(
            intra_list_diversity_from_features([0, 1], features), 1.0
        )
        # 45-degree vectors -> cosine 1/sqrt(2) -> distance 1 - 1/sqrt(2).
        self.assertAlmostEqual(
            intra_list_diversity_from_features([0, 2], features),
            1.0 - 1.0 / np.sqrt(2.0),
            places=7,
        )
        # A zero vector is documented as having zero similarity to every item.
        self.assertAlmostEqual(
            intra_list_diversity_from_features([0, 3], features), 1.0
        )
        summary = mean_intra_list_diversity(
            [[0, 1], [0, 2]], features, max_users=None
        )
        self.assertEqual(summary["n_lists"], 2)
        self.assertAlmostEqual(
            summary["mean_diversity"],
            (1.0 + (1.0 - 1.0 / np.sqrt(2.0))) / 2.0,
            places=7,
        )


class TestInfluenceMaximization(unittest.TestCase):
    """Test Part 5 influence maximization algorithms."""

    def setUp(self):
        # Small directed propagation graph:
        # 0 -> 1, 2, 3 (out-degree 3)
        # 1 -> 4 (out-degree 1)
        # 2 -> 5 (out-degree 1)
        # 3 -> 6 (out-degree 1)
        # 7 -> 8, 9, 10 (out-degree 3, separate component)
        self.graph = {
            0: [1, 2, 3],
            1: [4],
            2: [5],
            3: [6],
            7: [8, 9, 10],
            4: [],
            5: [],
            6: [],
            8: [],
            9: [],
            10: [],
        }
        self.users = list(range(11))

    def test_random_seed_selection(self):
        seeds = random_seed_selection(self.users, budget=4, seed=42)
        self.assertEqual(len(seeds), 4)
        self.assertEqual(len(set(seeds)), 4)
        # Deterministic
        seeds2 = random_seed_selection(self.users, budget=4, seed=42)
        self.assertEqual(seeds, seeds2)

    def test_degree_centrality_selection(self):
        # Nodes 0 and 7 have out-degree 3.
        # Nodes 1, 2, 3 have out-degree 1.
        seeds = degree_centrality_seed_selection(self.graph, self.users, budget=2)
        self.assertEqual(seeds, [0, 7])

    def test_greedy_and_celf_equivalence(self):
        # Run standard greedy
        res_grd = greedy_influence_maximization(
            self.graph,
            self.users,
            budget=2,
            propagation_prob=1.0,  # deterministic cascade
            n_simulations=1,
            candidate_pool_size=11,
            seed=42,
            use_celf=False,
        )
        # Run CELF greedy
        res_celf = greedy_influence_maximization(
            self.graph,
            self.users,
            budget=2,
            propagation_prob=1.0,
            n_simulations=1,
            candidate_pool_size=11,
            seed=42,
            use_celf=True,
        )

        # Both should select the exact same optimal seeds
        self.assertEqual(res_grd["seeds"], res_celf["seeds"])
        # With full propagation, picking 0 (activates 1,2,3,4,5,6 -> 7 nodes)
        # and 7 (activates 8,9,10 -> 4 nodes) gives complete coverage of 11 nodes.
        self.assertEqual(set(res_grd["seeds"]), {0, 7})
        # CELF should perform fewer evaluations than or equal to standard greedy
        self.assertLessEqual(res_celf["n_evaluations"], res_grd["n_evaluations"])

    def test_compute_adopter_overlap(self):
        sets_a = [{0, 1, 2, 3}, {0, 1, 4}]
        sets_b = [{0, 1, 2, 3}, {0, 1, 5}]
        # Run 0: identical -> 1.0
        # Run 1: intersection {0, 1} = 2, union {0, 1, 4, 5} = 4 -> 0.5
        # Average = 0.75
        self.assertAlmostEqual(compute_adopter_overlap(sets_a, sets_b), 0.75)

    def test_greedy_does_not_stop_on_negative_monte_carlo_gains(self):
        # Regression for the old ``best_gain = -1.0`` sentinel.  Every
        # measured spread here is negative, so every marginal gain is below
        # -1.  A correct implementation must still fill the budget.
        def fake_estimate_spread(propagation_graph, seed_users, **kwargs):
            n = len(set(seed_users))
            return -5.0 - n, 0.0, [set(seed_users)]

        with patch("sntp.influence.estimate_spread", fake_estimate_spread):
            result = greedy_influence_maximization(
                self.graph,
                self.users,
                budget=3,
                propagation_prob=0.05,
                n_simulations=1,
                candidate_pool=self.users,
                seed=42,
                use_celf=False,
            )
        self.assertEqual(len(result["seeds"]), 3)
        self.assertEqual(len(result["marginal_gains"]), 3)

    def test_explicit_candidate_pool_is_reported(self):
        pool = [0, 1, 2, 7]
        result = greedy_influence_maximization(
            self.graph,
            self.users,
            budget=2,
            propagation_prob=1.0,
            n_simulations=1,
            candidate_pool=pool,
            seed=42,
            use_celf=False,
        )
        self.assertEqual(result["candidate_pool"], sorted(pool))
        self.assertEqual(result["n_candidates"], len(pool))
        self.assertTrue(set(result["seeds"]).issubset(set(pool)))

    def test_greedy_marginal_gains_match_spread_increments(self):
        # With p=1.0 and one Monte Carlo trial the cascade is deterministic,
        # so the reported marginal gains must equal the increments of the
        # spread progression exactly.
        result = greedy_influence_maximization(
            self.graph,
            self.users,
            budget=2,
            propagation_prob=1.0,
            n_simulations=1,
            candidate_pool=self.users,
            seed=42,
            use_celf=False,
        )
        progression = result["spread_progression"]
        gains = result["marginal_gains"]
        self.assertAlmostEqual(gains[0], progression[0])
        self.assertAlmostEqual(gains[1], progression[1] - progression[0])
        # Seeds 0 and 7 together reach all 11 nodes in this graph.
        self.assertEqual(set(result["seeds"]), {0, 7})
        self.assertAlmostEqual(progression[-1], 11.0)


class DummyRecommender:
    """Mock recommender for group and intervention tests."""

    def __init__(self, score_matrix: np.ndarray):
        self.score_matrix = score_matrix
        self.n_users, self.n_items = score_matrix.shape

    def predict(self, u: int, i: int) -> float:
        return float(self.score_matrix[u, i])

    def score_all(self, u: int) -> np.ndarray:
        return self.score_matrix[u].copy()


class TestGroupRecommendation(unittest.TestCase):
    """Test Part 6 group recommendation strategies."""

    def setUp(self):
        # 3 users, 4 items
        # User 0: [5.0, 4.0, 1.0, 2.0]
        # User 1: [1.0, 4.0, 5.0, 2.0]
        # User 2: [2.0, 4.0, 2.0, 5.0]
        self.scores = np.array([
            [5.0, 4.0, 1.0, 2.0],
            [1.0, 4.0, 5.0, 2.0],
            [2.0, 4.0, 2.0, 5.0],
        ])
        self.recommender = DummyRecommender(self.scores)
        self.group = [0, 1, 2]

    def test_score_group_items_average(self):
        # Item 0: (5+1+2)/3 = 2.67
        # Item 1: (4+4+4)/3 = 4.0
        # Item 2: (1+5+2)/3 = 2.67
        # Item 3: (2+2+5)/3 = 3.0
        avg_scores = score_group_items(self.recommender, self.group, n_items=4, strategy="average")
        self.assertAlmostEqual(avg_scores[1], 4.0)
        self.assertAlmostEqual(avg_scores[3], 3.0)

    def test_score_group_items_least_misery(self):
        # Item 0 min: 1.0
        # Item 1 min: 4.0 (consensus choice!)
        # Item 2 min: 1.0
        # Item 3 min: 2.0
        lm_scores = score_group_items(self.recommender, self.group, n_items=4, strategy="least_misery")
        self.assertAlmostEqual(lm_scores[1], 4.0)
        self.assertAlmostEqual(lm_scores[0], 1.0)
        self.assertAlmostEqual(lm_scores[3], 2.0)
        # Top recommended item under least misery is item 1
        recs = recommend_for_group(self.recommender, self.group, n_items=4, n=1, strategy="least_misery")
        self.assertEqual(recs[0][0], 1)

    def test_score_group_items_most_pleasure(self):
        # Item 0 max: 5.0
        # Item 1 max: 4.0
        # Item 2 max: 5.0
        # Item 3 max: 5.0
        mp_scores = score_group_items(self.recommender, self.group, n_items=4, strategy="most_pleasure")
        self.assertAlmostEqual(mp_scores[0], 5.0)
        self.assertAlmostEqual(mp_scores[1], 4.0)

    def test_evaluate_group_satisfaction(self):
        sat = evaluate_group_satisfaction(self.recommender, self.group, item_idx=1)
        self.assertAlmostEqual(sat["mean_satisfaction"], 4.0)
        self.assertAlmostEqual(sat["min_satisfaction"], 4.0)
        self.assertAlmostEqual(sat["max_satisfaction"], 4.0)
        self.assertAlmostEqual(sat["range"], 0.0)

        sat0 = evaluate_group_satisfaction(self.recommender, self.group, item_idx=0)
        self.assertAlmostEqual(sat0["min_satisfaction"], 1.0)
        self.assertAlmostEqual(sat0["max_satisfaction"], 5.0)
        self.assertAlmostEqual(sat0["range"], 4.0)


class TestIntervention(unittest.TestCase):
    """Test Part 8 intervention (relevance-aware propagation threshold)."""

    def test_relevance_threshold_blocks_low_affinity(self):
        # 0 -> 1 -> 2
        prop_graph = {0: [1], 1: [2], 2: []}
        sim = ICMSimulator(prop_graph, propagation_prob=1.0, seed=42)

        # User 1 has affinity 0.2 (below 0.5 threshold)
        affinity_map = {0: 1.0, 1: 0.2, 2: 0.9}

        # Without threshold (threshold = 0.0), user 1 adopts with prob 1.0 * 0.2 = 0.2
        # With threshold = 0.5, user 1 NEVER adopts, blocking cascade to user 2
        res = sim.run_preference_aware(
            seed_users={0},
            affinity_fn=lambda u: affinity_map[u],
            base_adopt_prob=1.0,
            min_affinity_threshold=0.5,
        )
        self.assertNotIn(1, res["adopters"])
        self.assertNotIn(2, res["adopters"])
        self.assertEqual(res["adopters"], {0})


if __name__ == "__main__":
    unittest.main()
