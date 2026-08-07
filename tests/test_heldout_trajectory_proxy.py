"""held-out 多时长位移误差代理的核心口径测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
sys.path.insert(0, str(MODELS_DIR))

from analyze_heldout_trajectory_proxy import (  # noqa: E402
    Episode,
    endpoint_errors_for_episodes,
    paired_id_bootstrap,
    split_continuous_bounds,
    summarize_horizon,
    window_group_indices,
)


class ContinuousEpisodeTest(unittest.TestCase):
    def test_hourly_gap_is_split_and_never_bridged(self) -> None:
        times = np.asarray(
            [
                "2020-01-01T00:00:00",
                "2020-01-01T01:00:00",
                "2020-01-01T03:00:00",
                "2020-01-01T04:00:00",
            ],
            dtype="datetime64[s]",
        )
        self.assertEqual(
            split_continuous_bounds(times),
            [(0, 2), (2, 4)],
        )

    def test_duplicate_or_backward_time_is_split(self) -> None:
        times = np.asarray(
            [
                "2020-01-01T01:00:00",
                "2020-01-01T01:00:00",
                "2020-01-01T00:00:00",
            ],
            dtype="datetime64[s]",
        )
        self.assertEqual(
            split_continuous_bounds(times),
            [(0, 1), (1, 2), (2, 3)],
        )


class EndpointProxyTest(unittest.TestCase):
    def test_known_constant_velocity_error_accumulates_to_km(self) -> None:
        episode = Episode("a", 0, 0, 0, 24)
        linear_error = np.column_stack(
            (np.ones(24), np.zeros(24))
        )
        mlp_error = linear_error * 0.5
        linear = endpoint_errors_for_episodes(
            linear_error,
            [episode],
            horizons=(24,),
        )[24]
        mlp = endpoint_errors_for_episodes(
            mlp_error,
            [episode],
            horizons=(24,),
        )[24]
        self.assertEqual(len(linear), 1)
        self.assertAlmostEqual(linear[0], 86.4)
        self.assertAlmostEqual(mlp[0], 43.2)

    def test_incomplete_tail_is_discarded(self) -> None:
        episode = Episode("a", 0, 0, 0, 30)
        error = np.ones((30, 2))
        values = endpoint_errors_for_episodes(
            error,
            [episode],
            horizons=(24,),
        )[24]
        self.assertEqual(len(values), 1)

    def test_source_episode_boundaries_are_not_crossed(self) -> None:
        episodes = [
            Episode("a", 0, 0, 0, 4),
            Episode("a", 0, 1, 4, 8),
        ]
        error = np.ones((8, 2))
        values = endpoint_errors_for_episodes(
            error,
            episodes,
            horizons=(6,),
        )[6]
        self.assertEqual(len(values), 0)


class EqualIdAggregationTest(unittest.TestCase):
    def test_equal_id_mean_is_not_pooled_window_median(self) -> None:
        groups = np.asarray([0, 0, 0, 1], dtype=np.int32)
        endpoint = {
            "linear": np.asarray([1.0, 1.0, 1.0, 100.0]),
            "frozen_core6_mlp": np.asarray([0.5, 0.5, 0.5, 90.0]),
        }
        summary = summarize_horizon(
            6,
            groups,
            endpoint,
            ["long", "short"],
            bootstrap_replicates=200,
            seed=48,
        )
        linear = summary["models"]["linear"]
        self.assertAlmostEqual(linear["pooled_window_median_km"], 1.0)
        self.assertAlmostEqual(
            linear["equal_id_mean_of_window_medians_km"],
            50.5,
        )
        comparison = summary["comparisons_vs_linear"][
            "frozen_core6_mlp"
        ]["median"]
        self.assertEqual(comparison["id_wins"], 2)
        self.assertAlmostEqual(comparison["id_win_rate"], 1.0)

    def test_window_group_order_matches_episode_order(self) -> None:
        episodes = [
            Episode("b", 0, 0, 0, 12),
            Episode("a", 1, 0, 12, 18),
        ]
        groups = window_group_indices(
            episodes,
            ["a", "b"],
            horizons=(6,),
        )[6]
        np.testing.assert_array_equal(groups, [1, 1, 0])

    def test_paired_bootstrap_is_deterministic(self) -> None:
        baseline = np.asarray([1.0, 2.0, 3.0])
        candidate = np.asarray([0.9, 1.8, 2.7])
        first = paired_id_bootstrap(
            baseline,
            candidate,
            seed=42,
            replicates=500,
        )
        second = paired_id_bootstrap(
            baseline,
            candidate,
            seed=42,
            replicates=500,
        )
        self.assertEqual(first, second)
        self.assertGreater(
            first["relative_improvement_percent_ci95"][0],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
