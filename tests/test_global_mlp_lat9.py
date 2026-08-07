"""Global 九维空间 MLP 数据标准化协议测试。"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.models.training.run_global_mlp_lat9 import (
    FEATURE_COLUMNS,
    standardize_cached_splits,
)


class GlobalMlpLat9Test(unittest.TestCase):
    def test_scaler_only_fits_train_and_preserves_feature_order(self):
        core_train = np.asarray(
            [
                [1, 2, 3, 4, 0, 1],
                [3, 4, 5, 6, 1, 0],
                [5, 6, 7, 8, -1, 0],
            ],
            dtype=np.float32,
        )
        raw_splits = {
            "train": {
                "core6": core_train,
                "sin_latitude": np.asarray(
                    [-0.5, 0.0, 0.5],
                    np.float32,
                ),
                "sin_longitude": np.asarray(
                    [-1.0, 0.0, 1.0],
                    np.float32,
                ),
                "cos_longitude": np.asarray(
                    [0.0, 1.0, 0.0],
                    np.float32,
                ),
                "target": np.zeros((3, 2), np.float32),
            },
            "val": {
                "core6": np.asarray(
                    [[9, 10, 11, 12, 0, -1]],
                    np.float32,
                ),
                "sin_latitude": np.asarray([1.0], np.float32),
                "sin_longitude": np.asarray([0.5], np.float32),
                "cos_longitude": np.asarray([-0.5], np.float32),
                "target": np.zeros((1, 2), np.float32),
            },
            "test": {
                "core6": np.asarray(
                    [[-3, -2, -1, 0, 0.5, 0.5]],
                    np.float32,
                ),
                "sin_latitude": np.asarray([-1.0], np.float32),
                "sin_longitude": np.asarray([-0.5], np.float32),
                "cos_longitude": np.asarray([-0.5], np.float32),
                "target": np.zeros((1, 2), np.float32),
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            scaler_path = Path(temporary) / "scaler.pkl"
            splits, scaler = standardize_cached_splits(
                raw_splits,
                scaler_path,
            )
            self.assertTrue(scaler_path.is_file())
            self.assertEqual(int(scaler.n_samples_seen_), 3)
            np.testing.assert_allclose(
                splits["X_train"].mean(axis=0),
                0.0,
                atol=1e-6,
            )
            self.assertEqual(splits["feature_cols"], FEATURE_COLUMNS)
            self.assertEqual(
                FEATURE_COLUMNS[-3:],
                [
                    "sin_latitude",
                    "sin_longitude",
                    "cos_longitude",
                ],
            )


if __name__ == "__main__":
    unittest.main()
