"""
Equivalence tests for the vectorized full-catalogue ranking paths.

Each optimized ``score_all`` / ``recommend`` is checked against a brute-force
reference that calls ``predict`` for every item and stable-sorts the result
(the behaviour of the original per-item Python loops).
"""

import sys
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sntp.recommenders._ranking import top_n_from_scores
from sntp.recommenders.item_based import ItemBasedCF
from sntp.recommenders.matrix_factorization import BiasedMF
from sntp.recommenders.social import SocialRecommender, SimilarUserRecommender


def _brute_force_recommend(rec, user_idx, n, exclude, n_items):
    """Reference: predict every item, stable sort by -score (original loop)."""
    candidates = []
    for iidx in range(n_items):
        if iidx in exclude:
            continue
        pred = rec.predict(user_idx, iidx)
        if pred is not None:
            candidates.append((iidx, pred))
    candidates.sort(key=lambda x: -x[1])
    return candidates[:n]


def _random_ratings(n_users=60, n_items=80, density=0.12, seed=0):
    rng = np.random.RandomState(seed)
    triples = []
    for u in range(n_users):
        for i in range(n_items):
            if rng.rand() < density:
                triples.append((u, i, int(rng.randint(1, 6))))
    return triples


def _assert_same_ranking(tc, got, ref, places=9):
    tc.assertEqual([i for i, _ in got], [i for i, _ in ref])
    for (_, s1), (_, s2) in zip(got, ref):
        tc.assertAlmostEqual(s1, s2, places=places)


class TestTopN(unittest.TestCase):

    def test_ties_broken_by_lowest_index(self):
        scores = np.array([3.0, 5.0, 5.0, 4.0, 5.0, 1.0])
        got = top_n_from_scores(scores.copy(), 2)
        self.assertEqual([i for i, _ in got], [1, 2])

    def test_matches_stable_sort(self):
        rng = np.random.RandomState(1)
        for _ in range(50):
            scores = rng.randint(1, 6, size=40).astype(float)  # many ties
            exclude = set(rng.choice(40, size=8, replace=False).tolist())
            ref = sorted([(i, s) for i, s in enumerate(scores) if i not in exclude],
                         key=lambda x: -x[1])[:7]
            got = top_n_from_scores(scores.copy(), 7, exclude)
            self.assertEqual(got, [(i, float(s)) for i, s in ref])

    def test_fewer_candidates_than_n(self):
        scores = np.array([2.0, 4.0, 3.0])
        got = top_n_from_scores(scores, 10, exclude={0})
        self.assertEqual([i for i, _ in got], [1, 2])

    def test_all_excluded(self):
        self.assertEqual(top_n_from_scores(np.array([1.0, 2.0]), 5, {0, 1}), [])


class TestItemBasedVectorized(unittest.TestCase):

    def setUp(self):
        self.n_users, self.n_items = 60, 80
        triples = _random_ratings(self.n_users, self.n_items)
        ur, ir = defaultdict(list), defaultdict(list)
        for u, i, r in triples:
            ur[u].append((i, r))
            ir[i].append((u, r))
        rows, cols, vals = zip(*triples)
        m = sparse.csr_matrix((vals, (rows, cols)), shape=(self.n_users, self.n_items),
                              dtype=np.float32)
        self.icf = ItemBasedCF(max_neighbors=5, min_corated=2)
        self.icf.fit(m, dict(ur), dict(ir))

    def test_score_all_matches_predict(self):
        for u in range(self.n_users):
            scores = self.icf.score_all(u)
            ref = np.array([self.icf.predict(u, i) for i in range(self.n_items)])
            np.testing.assert_allclose(scores, ref, rtol=0, atol=1e-12)

    def test_score_all_unknown_user_uses_global_mean(self):
        scores = self.icf.score_all(10_000)
        ref = self.icf.predict(10_000, 0)
        np.testing.assert_allclose(scores, ref)

    def test_recommend_matches_brute_force(self):
        for u in range(self.n_users):
            exclude = {u % self.n_items}
            got = self.icf.recommend(u, 10, exclude=exclude)
            full_ex = exclude | set(self.icf.train_user_ratings.get(u, {}).keys())
            ref = _brute_force_recommend(self.icf, u, 10, full_ex, self.n_items)
            _assert_same_ranking(self, got, ref)


class TestMFVectorized(unittest.TestCase):

    def setUp(self):
        self.n_users, self.n_items = 40, 50
        triples = _random_ratings(self.n_users, self.n_items, seed=3)
        ident_u = {u: u for u in range(self.n_users)}
        ident_i = {i: i for i in range(self.n_items)}
        self.mf = BiasedMF(n_factors=4, lr=0.01, reg=0.02, n_epochs=5, seed=7)
        self.mf.fit(triples, self.n_users, self.n_items, ident_u, ident_i,
                    val_triples=triples[:20])

    def test_score_all_matches_predict(self):
        for u in range(self.n_users):
            ref = np.array([self.mf.predict(u, i) for i in range(self.n_items)])
            np.testing.assert_allclose(self.mf.score_all(u), ref, rtol=0, atol=1e-12)

    def test_recommend_matches_brute_force(self):
        for u in range(self.n_users):
            got = self.mf.recommend(u, 10)
            ex = self.mf._train_user_items.get(u, set())
            ref = _brute_force_recommend(self.mf, u, 10, ex, self.n_items)
            _assert_same_ranking(self, got, ref)

    def test_val_rmse_matches_loop(self):
        """Vectorized validation RMSE equals the per-rating _raw_predict loop."""
        val = [(u, i, float(r)) for u, i, r in _random_ratings(self.n_users, self.n_items,
                                                                seed=3)[:20]]
        loss = sum((r - self.mf._raw_predict(u, i)) ** 2 for u, i, r in val)
        self.assertAlmostEqual(self.mf.epoch_history[-1]["val_rmse"],
                               round(np.sqrt(loss / len(val)), 5), places=5)

    def test_training_deterministic(self):
        triples = _random_ratings(self.n_users, self.n_items, seed=3)
        ident_u = {u: u for u in range(self.n_users)}
        ident_i = {i: i for i in range(self.n_items)}
        mf2 = BiasedMF(n_factors=4, lr=0.01, reg=0.02, n_epochs=5, seed=7)
        mf2.fit(triples, self.n_users, self.n_items, ident_u, ident_i,
                val_triples=triples[:20])
        np.testing.assert_array_equal(self.mf.user_factors, mf2.user_factors)
        np.testing.assert_array_equal(self.mf.item_bias, mf2.item_bias)


class TestSocialVectorized(unittest.TestCase):

    def setUp(self):
        self.n_users, self.n_items = 40, 50
        triples = _random_ratings(self.n_users, self.n_items, seed=5)
        ident_u = {u: u for u in range(self.n_users)}
        ident_i = {i: i for i in range(self.n_items)}
        self.mf = BiasedMF(n_factors=4, n_epochs=3, seed=1)
        self.mf.fit(triples, self.n_users, self.n_items, ident_u, ident_i)
        self.train = defaultdict(dict)
        for u, i, r in triples:
            self.train[u][i] = float(r)
        self.train = dict(self.train)
        rng = np.random.RandomState(9)
        # Include a trusted user index outside the rating matrix (trust-only user).
        self.trust = {u: rng.choice(self.n_users, size=5, replace=False).tolist() + [999]
                      for u in range(self.n_users)}

    def test_social_score_all_matches_predict(self):
        for lam in (0.0, 0.3, 1.0):
            sr = SocialRecommender(self.mf, self.trust, social_weight=lam,
                                   global_mean=self.mf.global_mean)
            sr.set_training_data(self.train)
            for u in range(self.n_users):
                ref = np.array([sr.predict(u, i) for i in range(self.n_items)])
                np.testing.assert_allclose(sr.score_all(u), ref, rtol=0, atol=1e-12)

    def test_social_recommend_matches_brute_force(self):
        sr = SocialRecommender(self.mf, self.trust, social_weight=0.5,
                               global_mean=self.mf.global_mean)
        sr.set_training_data(self.train)
        for u in range(self.n_users):
            got = sr.recommend(u, 10)
            ex = set(self.train.get(u, {}).keys())
            ref = _brute_force_recommend(sr, u, 10, ex, self.n_items)
            _assert_same_ranking(self, got, ref)

    def test_social_with_icf_baseline(self):
        ur, ir = defaultdict(list), defaultdict(list)
        rows, cols, vals = [], [], []
        for u, items in self.train.items():
            for i, r in items.items():
                ur[u].append((i, r)); ir[i].append((u, r))
                rows.append(u); cols.append(i); vals.append(r)
        m = sparse.csr_matrix((vals, (rows, cols)), shape=(self.n_users, self.n_items))
        icf = ItemBasedCF(max_neighbors=5, min_corated=2)
        icf.fit(m, dict(ur), dict(ir))
        sr = SocialRecommender(icf, self.trust, social_weight=0.3, global_mean=icf.global_mean)
        sr.set_training_data(self.train)
        for u in range(self.n_users):
            ref = np.array([sr.predict(u, i) for i in range(self.n_items)])
            np.testing.assert_allclose(sr.score_all(u), ref, rtol=0, atol=1e-12)

    def test_similar_user_score_all_matches_predict(self):
        sim = SimilarUserRecommender(self.mf, self.trust, social_weight=0.4, k_similar=5,
                                     global_mean=self.mf.global_mean)
        sim.set_training_data(self.train)
        sim.compute_similar_users(list(range(10)))
        for u in range(10):
            ref = np.array([sim.predict(u, i) for i in range(self.n_items)])
            np.testing.assert_allclose(sim.score_all(u), ref, rtol=0, atol=1e-12)

    def test_baseline_without_score_all_falls_back(self):
        class Dummy:
            global_mean = 3.0
            n_items = 4

            def predict(self, u, i):
                return None if i == 3 else 3.5

        sr = SocialRecommender(Dummy(), {0: [1]}, social_weight=0.5)
        sr.set_training_data({1: {0: 5.0, 3: 2.0}})
        ref = [sr.predict(0, i) for i in range(4)]
        np.testing.assert_allclose(sr.score_all(0), ref)


if __name__ == "__main__":
    unittest.main()
