"""扩展区域 original_ID 数量搜索测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
sys.path.insert(0, str(MODELS_DIR))

from search_regional_extent import summarize_extent_search  # noqa: E402


def _frame(
    original_id: str,
    longitudes: list[float],
    latitudes: list[float] | None = None,
) -> pd.DataFrame:
    if latitudes is None:
        latitudes = [30.0] * len(longitudes)
    return pd.DataFrame(
        {
            "original_ID": [original_id] * len(longitudes),
            "latitude": np.asarray(latitudes),
            "longitude": np.asarray(longitudes),
        }
    )


class RegionalExtentSearchTest(unittest.TestCase):
    def test_counts_only_in_rectangle_and_accumulates_segments(self) -> None:
        trajectories = [
            _frame("train-a", [130.0, 145.0, 160.0]),
            _frame("train-a", [135.0, 145.0]),
            _frame("val-a", [139.0, 141.0]),
            _frame(
                "test-a",
                [106.0, 139.0, 138.0],
                [10.0, 30.0, 30.0],
            ),
        ]
        manifest = {
            "splits": {
                "train": {"original_ids": ["train-a"]},
                "val": {"original_ids": ["val-a"]},
                "test": {"original_ids": ["test-a"]},
            }
        }
        result = summarize_extent_search(
            trajectories,
            manifest,
            min_regional_points=2,
            minimum_total_ids=3,
            minimum_lineage_ids={"train": 1, "val": 1, "test": 1},
        )
        first, second = result["candidates"]
        self.assertEqual(first["longitude"], [105.0, 140.0])
        self.assertEqual(first["n_original_ids"], 2)
        self.assertEqual(
            first["lineage_counts"],
            {"train": 1, "val": 0, "test": 1},
        )
        self.assertFalse(first["passed"])
        self.assertEqual(second["n_original_ids"], 3)
        self.assertEqual(
            second["lineage_counts"],
            {"train": 1, "val": 1, "test": 1},
        )
        self.assertTrue(second["passed"])
        self.assertEqual(
            result["selected_range"]["longitude"],
            [105.0, 150.0],
        )

    def test_threshold_is_per_id_across_segments(self) -> None:
        trajectories = [
            _frame("train-a", [130.0]),
            _frame("train-a", [131.0]),
            _frame("val-a", [132.0, 133.0]),
            _frame("test-a", [134.0, 135.0]),
        ]
        manifest = {
            "splits": {
                "train": {"original_ids": ["train-a"]},
                "val": {"original_ids": ["val-a"]},
                "test": {"original_ids": ["test-a"]},
            }
        }
        result = summarize_extent_search(
            trajectories,
            manifest,
            east_longitudes=(140.0,),
            min_regional_points=2,
            minimum_total_ids=3,
            minimum_lineage_ids={"train": 1, "val": 1, "test": 1},
        )
        self.assertEqual(result["candidates"][0]["n_original_ids"], 3)
        self.assertTrue(result["candidates"][0]["passed"])


if __name__ == "__main__":
    unittest.main()
