"""Regression tests for leakage and reproducible experiment design.

These tests are deliberately small and hand-checkable.  They exercise the
experiment-level behaviour that matters for the assignment:

* only training triples enter the matrix used for group construction;
* changing ``dataset.rating_matrix`` (which contains all observed ratings)
  cannot change the groups produced by Requirement 6;
* the social weight is selected on validation, not test;
* target-item and seed-set selection helpers are deterministic and
  training-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from sntp.data import EpinionsDataset
from sntp.experiments import run_requirement_3, run_requirement_6
from sntp.groups import aggregate_member_scores
from sntp.selection import (
    build_rating_matrix,
    generate_seed_sets,
    make_simulation_seeds,
    randomize_directed_graph_degree_preserving,
    select_target_item,
)


class DummyRecommender:
    """Deterministic recommender with the methods used by Parts 3 and 6."""

    def __init__(self, n_users: int, n_items: int, seed: int = 7):
        rng = np.random.RandomState(seed)
        self.n_users = n_users
        self.n_items = n_items
        self.global_mean = 3.0
        self.scores = rng.uniform(1.0, 5.0, size=(n_users, n_items))
        self.user_bias = rng.normal(0, 0.1, n_users)
        self.item_bias = rng.normal(0, 0.1, n_items)

    def predict(self, u, i, rating_min=1.0, rating_max=5.0):
        return float(np.clip(self.scores[u, i], rating_min, rating_max))

    def predict_batch(self, u, i, rating_min=1.0, rating_max=5.0):
        return np.clip(self.scores[np.asarray(u), np.asarray(i)],
                       rating_min, rating_max)

    def score_all(self, u, rating_min=1.0, rating_max=5.0):
        return np.clip(self.scores[u], rating_min, rating_max)

    def recommend(self, u, n=10, exclude=None):
        exclude = set(exclude or set())
        order = np.argsort(-self.scores[u], kind="stable")
        out = []
        for idx in order:
            if int(idx) in exclude:
                continue
            out.append((int(idx), float(self.scores[u, idx])))
            if len(out) >= n:
                break
        return out

    def recommend_batch(self, users, n=10, excludes=None):
        excludes = excludes or [set() for _ in users]
        return [self.recommend(u, n=n, exclude=ex)
                for u, ex in zip(users, excludes)]


def _make_dataset() -> EpinionsDataset:
    ratings = []
    for u in range(10):
        for i in range(5):
            if (u + i) % 2 == 0:
                ratings.append((u, i, 1 + (u + i) % 5))
    trust = [(u, (u + 1) % 10, 1) for u in range(10)]
    return EpinionsDataset(ratings, trust, {}, {})


def _split(dataset: EpinionsDataset):
    """Small deterministic train/validation/test split on original IDs."""
    train, val, test = [], [], []
    for idx, triple in enumerate(dataset.raw_ratings):
        if idx % 10 < 6:
            train.append(triple)
        elif idx % 10 < 8:
            val.append(triple)
        else:
            test.append(triple)
    return train, val, test


def _tiny_diffusion_config() -> dict:
    return {
        "diffusion": {
            "subgraph": {"enabled": True, "center_user": 0, "hops": 2},
            "n_simulations": 1,
            "seed_size": 2,
            "seed": 42,
            "max_rounds": 5,
            "propagation_probs": [0.2],
            "recovery_probs": [0.0],
            "default_affinity": 0.3,
        },
        "group_recommendation": {
            "n_groups": 2,
            "group_size": 3,
            "conflicting": True,
            "top_n": 3,
            "diffusion_simulations": 1,
            "propagation_prob": 0.2,
            "base_adopt_prob": 0.5,
            "max_rounds": 5,
            "seed": 42,
        },
    }


def test_build_rating_matrix_uses_only_supplied_triples():
    user_to_idx = {10: 0, 11: 1}
    item_to_idx = {100: 0, 101: 1, 102: 2}
    train = [(10, 100, 5), (11, 101, 1)]

    train_matrix = build_rating_matrix(2, 3, train, user_to_idx, item_to_idx)
    assert train_matrix.shape == (2, 3)
    assert train_matrix.nnz == 2
    assert train_matrix[0, 0] == 5.0
    assert train_matrix[1, 1] == 1.0
    assert train_matrix[0, 2] == 0.0

    # Adding held-out ratings to the argument list would change the matrix;
    # the point of the helper is that callers pass only the training split.
    full_matrix = build_rating_matrix(
        2, 3, train + [(10, 102, 2)], user_to_idx, item_to_idx
    )
    assert full_matrix.nnz == 3


def test_select_target_item_is_training_only_and_documented():
    # Ten items: training counts 1..10.  The explicit 6-20 band contains
    # counts 6..10; the band median is 8, selecting original item 7 (count 8).
    train = []
    item_to_idx = {}
    for item in range(10):
        item_to_idx[item] = item
        for u in range(item + 1):
            train.append((u, item, 4))
    selected = select_target_item(train, item_to_idx, 6, 20)
    assert selected["original_id"] == 7
    assert selected["train_rating_count"] == 8
    assert selected["band_min_train_ratings"] == 6
    assert selected["band_max_train_ratings"] == 20
    assert selected["selection_rule"].startswith("Choose the item")
    assert selected["used_count_band_fallback"] is False

    # The selected count is inside the documented band and the selection is
    # independent of validation/test triples (none are passed to the helper).
    assert 6 <= selected["train_rating_count"] <= 20


def test_select_target_item_records_empty_band_fallback():
    train = [(u, item, 4) for item in range(5) for u in range(item + 1)]
    item_to_idx = {item: item for item in range(5)}
    selected = select_target_item(train, item_to_idx, 6, 20)
    assert selected["used_count_band_fallback"] is True
    assert selected["band_size_items"] == 0
    assert selected["train_rating_count"] == 5
    assert "band midpoint" in selected["selection_rule"]


def test_seed_sets_and_simulation_seeds_are_reproducible():
    users = list(range(50))
    first = generate_seed_sets(users, seed_size=5, n_simulations=4, seed=42)
    second = generate_seed_sets(users, seed_size=5, n_simulations=4, seed=42)
    assert first == second
    assert len(first) == 4
    assert all(len(s) == 5 and len(set(s)) == 5 for s in first)
    assert make_simulation_seeds(42, 4) == make_simulation_seeds(42, 4)


def test_degree_preserving_randomization_is_a_controlled_null_model():
    graph = {
        0: [1, 2, 3],
        1: [0, 3],
        2: [0, 1],
        3: [1, 2],
        4: [],
    }
    first, info = randomize_directed_graph_degree_preserving(
        graph, nodes=range(5), swap_multiplier=20, seed=7
    )
    second, info2 = randomize_directed_graph_degree_preserving(
        graph, nodes=range(5), swap_multiplier=20, seed=7
    )
    assert first == second
    assert info == info2
    assert info["degree_sequence_preserved"] is True
    assert info["n_nodes"] == 5
    assert info["n_edges"] == sum(len(v) for v in graph.values())
    assert set(first) == set(graph)
    assert sum(len(v) for v in first.values()) == info["n_edges"]
    out_before = {u: len(v) for u, v in graph.items()}
    out_after = {u: len(v) for u, v in first.items()}
    assert out_before == out_after
    in_before = {u: sum(u in neighbours for neighbours in graph.values())
                 for u in graph}
    in_after = {u: sum(u in neighbours for neighbours in first.values())
                for u in graph}
    assert in_before == in_after
    for u, neighbours in first.items():
        assert u not in neighbours
        assert len(neighbours) == len(set(neighbours))
    # A 20x swap multiplier on this small graph should actually rearrange it.
    assert first != graph


def test_group_aggregation_is_hand_checkable():
    scores = np.array([
        [5.0, 1.0, 3.0],
        [1.0, 4.0, 3.0],
        [3.0, 2.0, 3.0],
    ])
    np.testing.assert_allclose(
        aggregate_member_scores(scores, "average"), [3.0, 7.0 / 3.0, 3.0]
    )
    np.testing.assert_allclose(
        aggregate_member_scores(scores, "least_misery"), [1.0, 1.0, 3.0]
    )
    np.testing.assert_allclose(
        aggregate_member_scores(scores, "most_pleasure"), [5.0, 4.0, 3.0]
    )


def test_requirement_6_groups_do_not_read_full_rating_matrix(tmp_path: Path):
    dataset = _make_dataset()
    train, _, _ = _split(dataset)
    recommender = DummyRecommender(dataset.n_users, dataset.n_items)
    config = _tiny_diffusion_config()

    # First run with the full observed matrix present on the dataset.
    dataset.rating_matrix = sparse.csr_matrix(
        (dataset.n_users, dataset.n_items), dtype=np.float32
    )
    result_empty = run_requirement_6(
        dataset, train, recommender, config, tmp_path / "empty"
    )

    # Replace it with a contradictory matrix.  Requirement 6 must still build
    # its grouping matrix from ``train`` and therefore produce identical groups.
    dense = np.full((dataset.n_users, dataset.n_items), 5.0, dtype=np.float32)
    dataset.rating_matrix = sparse.csr_matrix(dense)
    result_dense = run_requirement_6(
        dataset, train, recommender, config, tmp_path / "dense"
    )

    groups_empty = {
        key: value["members"] for key, value in result_empty["groups"].items()
    }
    groups_dense = {
        key: value["members"] for key, value in result_dense["groups"].items()
    }
    assert groups_empty == groups_dense
    assert result_empty["design"]["group_construction"]["matrix_source"].startswith(
        "training split only"
    )
    assert (
        result_empty["design"]["group_construction"]["train_matrix_nnz"]
        == len(train)
    )

    # Candidate best-set bookkeeping is internally consistent, including ties.
    for group in result_empty["groups"].values():
        candidates = group["candidates"]
        sat_best = set(candidates["best_satisfaction_items"])
        spread_best = set(candidates["best_spread_items"])
        assert candidates["best_satisfaction_item"] == min(sat_best)
        assert candidates["best_spread_item"] == min(spread_best)
        assert candidates["best_satisfaction_and_spread_overlap"] == bool(
            sat_best & spread_best
        )
        for item in sat_best | spread_best:
            assert item in candidates["items"]


class _DummyBaseline:
    """Minimal baseline for Requirement 3's social wrapper."""

    def __init__(self, n_users: int, n_items: int):
        self.n_users = n_users
        self.n_items = n_items
        self.global_mean = 3.0


def _run_requirement_3_with_fake_metrics(dataset, train, val, test, tmp_path):
    """Run R3 with metrics that deliberately reward different lambdas on val/test.

    Validation prefers lambda=0.3; test prefers lambda=0.0.  If the
    implementation selected on test, ``best_lambda`` would be 0.0.
    """
    from unittest.mock import patch

    baseline = _DummyBaseline(dataset.n_users, dataset.n_items)

    def fake_evaluate_predictions(recommender, triples, *args, **kwargs):
        is_val = triples is val
        is_test = triples is test
        lam = getattr(recommender, "social_weight", None)
        if is_val:
            rmse = {0.0: 1.0, 0.3: 0.1}.get(lam, 0.5)
        elif is_test:
            rmse = {0.0: 0.1, 0.3: 1.0}.get(lam, 0.5)
        else:
            rmse = 0.5
        return {
            "mae": rmse,
            "rmse": rmse,
            "n_evaluated": len(triples),
            "n_skipped": 0,
        }

    def fake_evaluate_ranking(*args, **kwargs):
        return {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "f1_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "coverage": 0.0,
            "diversity": 0.0,
            "n_users_evaluated": 1,
            "total_eligible_users": 1,
            "is_sampled": False,
            "sample_seed": None,
            "catalog_size": dataset.n_items,
            "diversity_n_lists": 0,
            "diversity_is_sampled": False,
            "diversity_sample_seed": None,
            "eval_time_seconds": 0.0,
        }

    config = {
        "top_k": 2,
        "relevance_threshold": 4,
        "ranking_sample_users": None,
        "ranking_sample_seed": 42,
        "split_seed": 42,
        "social": {
            "lambda_values": [0.0, 0.3],
            "k_similar": 2,
            "comparison_sample": 2,
        },
    }
    with patch("sntp.experiments.evaluate_predictions", fake_evaluate_predictions), \
         patch("sntp.experiments.evaluate_ranking", fake_evaluate_ranking):
        return run_requirement_3(
            dataset, train, val, test, baseline, config, tmp_path
        )


def test_social_lambda_selected_on_validation_not_test(tmp_path: Path):
    dataset = _make_dataset()
    train, val, test = _split(dataset)

    first = _run_requirement_3_with_fake_metrics(
        dataset, train, val, test, tmp_path / "first"
    )
    assert first["best_lambda"] == 0.3
    assert first["selection"]["test_used_for_selection"] is False

    # Perturb the test split.  Validation evidence is unchanged, so the
    # selected lambda must remain 0.3 even though test now favours lambda=0.0.
    perturbed_test = [
        (u, i, 1 if r >= 4 else 5) for u, i, r in test
    ]
    second = _run_requirement_3_with_fake_metrics(
        dataset, train, val, perturbed_test, tmp_path / "second"
    )
    assert second["best_lambda"] == 0.3
