"""Global MLP 空间信息汇总计算测试。"""

import unittest

import numpy as np

from src.models.training.summarize_global_mlp_spatial import (
    longitude_band_comparison,
    metric_deltas,
)


class GlobalMlpSpatialSummaryTest(unittest.TestCase):
    def test_metric_deltas(self):
        self.assertEqual(
            metric_deltas(
                {"r2": 0.3, "rmse": 0.2},
                {"r2": 0.1, "rmse": 0.25},
            ),
            {"r2": 0.19999999999999998, "rmse": -0.04999999999999999},
        )

    def test_longitude_bands_are_non_overlapping(self):
        longitude = np.asarray(
            [-180.0, -90.0, 0.0, 90.0, 179.0],
            dtype=np.float64,
        )
        radians = np.deg2rad(longitude)
        target = np.asarray(
            [[0.0, 0.0]] * len(longitude),
            dtype=np.float32,
        )
        predictions = {
            "B": np.asarray(
                [[0.1, 0.1]] * len(longitude),
                dtype=np.float32,
            ),
            "E": np.asarray(
                [[0.2, 0.2]] * len(longitude),
                dtype=np.float32,
            ),
        }
        records = longitude_band_comparison(
            target,
            predictions,
            np.sin(radians),
            np.cos(radians),
            edges=np.asarray([-180.0, 0.0, 180.0]),
        )
        self.assertEqual(
            [record["n_samples"] for record in records],
            [2, 3],
        )
        self.assertEqual(
            sum(record["n_samples"] for record in records),
            len(longitude),
        )


if __name__ == "__main__":
    unittest.main()
