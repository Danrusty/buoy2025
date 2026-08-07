"""全局纬度信息 × 模型类型析因实验共用协议测试。"""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.training.global_factorial import (
    CORE6_FEATURES,
    LATITUDE_FEATURE,
    _extract_valid_arrays,
    assemble_features,
    latitude_band_metrics,
    load_cached_split,
    macro_original_id_metrics,
    original_id_win_rate,
    prepare_cache,
    sin_latitude,
    validate_frozen_split_manifest,
)


class GlobalFactorialTest(unittest.TestCase):
    @staticmethod
    def _frame(original_id, latitudes):
        n_rows = len(latitudes)
        sequence = np.arange(n_rows, dtype=np.float32)
        return pd.DataFrame(
            {
                "original_ID": [original_id] * n_rows,
                "latitude": latitudes,
                "era5_u10": sequence + 1.0,
                "era5_v10": sequence + 2.0,
                "era5_swh": sequence + 3.0,
                "era5_mwp": sequence + 4.0,
                "era5_wave_dir_sin": sequence * 0.0,
                "era5_wave_dir_cos": sequence * 0.0 + 1.0,
                "ve": sequence + 0.3,
                "vn": sequence + 0.2,
                "cfsv2_u": sequence * 0.0 + 0.1,
                "cfsv2_v": sequence * 0.0 + 0.1,
            }
        )

    def test_sin_latitude_definition_and_feature_order(self):
        encoded = sin_latitude(np.asarray([-90.0, 0.0, 30.0, 90.0]))
        np.testing.assert_allclose(
            encoded,
            [-1.0, 0.0, 0.5, 1.0],
            atol=1e-7,
        )
        core6 = np.arange(12, dtype=np.float32).reshape(2, 6)
        lat7 = assemble_features(core6, np.asarray([0.25, -0.5]))
        self.assertEqual(lat7.shape, (2, 7))
        np.testing.assert_array_equal(lat7[:, :6], core6)
        np.testing.assert_array_equal(lat7[:, 6], [0.25, -0.5])
        self.assertEqual(
            list(CORE6_FEATURES) + [LATITUDE_FEATURE],
            [
                "era5_u10",
                "era5_v10",
                "era5_swh",
                "era5_mwp",
                "era5_wave_dir_sin",
                "era5_wave_dir_cos",
                "sin_latitude",
            ],
        )

    def test_added_latitude_does_not_change_frozen_row_filter(self):
        frame = pd.DataFrame(
            {
                "original_ID": ["A", "A", "A"],
                "latitude": [30.0, np.nan, 40.0],
                "era5_u10": [1.0, 2.0, np.nan],
                "era5_v10": [2.0, 3.0, 4.0],
                "era5_swh": [1.0, 1.0, 1.0],
                "era5_mwp": [8.0, 8.0, 8.0],
                "era5_wave_dir_sin": [0.0, 0.0, 0.0],
                "era5_wave_dir_cos": [1.0, 1.0, 1.0],
                "ve": [0.3, 0.4, 0.5],
                "vn": [0.2, 0.3, 0.4],
                "cfsv2_u": [0.1, 0.1, 0.1],
                "cfsv2_v": [0.1, 0.1, 0.1],
            }
        )
        with self.assertRaisesRegex(ValueError, "latitude"):
            _extract_valid_arrays(frame)

        frame.loc[1, "latitude"] = 0.0
        extracted = _extract_valid_arrays(frame)
        self.assertEqual(extracted["core6"].shape, (2, 6))
        np.testing.assert_allclose(
            extracted["target"],
            [[0.2, 0.1], [0.3, 0.2]],
        )

    def test_split_manifest_disjointness(self):
        manifest = {
            "group_column": "original_ID",
            "random_seed": 42,
            "feature_columns": list(CORE6_FEATURES),
            "splits": {
                "train": {
                    "n_original_ids": 2,
                    "original_ids": ["A", "B"],
                },
                "val": {
                    "n_original_ids": 1,
                    "original_ids": ["C"],
                },
                "test": {
                    "n_original_ids": 1,
                    "original_ids": ["D"],
                },
            },
        }
        lookup = validate_frozen_split_manifest(manifest)
        self.assertEqual(lookup["A"], {"split": "train", "group_index": 0})
        self.assertEqual(lookup["D"], {"split": "test", "group_index": 0})

        manifest["splits"]["test"]["original_ids"] = ["A"]
        with self.assertRaisesRegex(ValueError, "leak"):
            validate_frozen_split_manifest(manifest)

    def test_cache_inherits_exact_synthetic_split_and_row_order(self):
        trajectories = [
            self._frame("A", [10.0, 20.0]),
            self._frame("B", [30.0]),
            self._frame("C", [-10.0]),
            self._frame("D", [-20.0, -30.0]),
        ]
        manifest = {
            "group_column": "original_ID",
            "random_seed": 42,
            "feature_columns": list(CORE6_FEATURES),
            "splits": {
                "train": {
                    "n_original_ids": 2,
                    "n_segments": 2,
                    "n_samples": 3,
                    "original_ids": ["B", "A"],
                },
                "val": {
                    "n_original_ids": 1,
                    "n_segments": 1,
                    "n_samples": 1,
                    "original_ids": ["C"],
                },
                "test": {
                    "n_original_ids": 1,
                    "n_segments": 1,
                    "n_samples": 2,
                    "original_ids": ["D"],
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.pkl"
            split_path = root / "split.json"
            cache_dir = root / "cache"
            artifact_path = root / "data_manifest.json"
            with source_path.open("wb") as file:
                pickle.dump(trajectories, file)
            split_path.write_text(json.dumps(manifest), encoding="utf-8")

            payload = prepare_cache(
                source_path=source_path,
                split_manifest_path=split_path,
                cache_dir=cache_dir,
                artifact_manifest_path=artifact_path,
                expected_source_sha256=None,
            )
            self.assertEqual(payload["split_counts"]["train"]["n_samples"], 3)
            train = load_cached_split(
                "train",
                artifact_manifest_path=artifact_path,
            )
            np.testing.assert_array_equal(
                train["group_index"],
                [1, 1, 0],
            )
            np.testing.assert_allclose(
                train["latitude"],
                [10.0, 20.0, 30.0],
            )
            np.testing.assert_allclose(
                train["target"],
                [[0.2, 0.1], [1.2, 1.1], [0.2, 0.1]],
                atol=1e-6,
            )

    def test_macro_id_weighting_and_win_rate(self):
        y_true = np.zeros((6, 2), dtype=np.float32)
        groups = np.asarray([0, 1, 1, 1, 1, 1], dtype=np.int32)
        candidate = np.asarray(
            [[1.0, 1.0]] + [[0.2, 0.2]] * 5,
            dtype=np.float32,
        )
        reference = np.asarray(
            [[2.0, 2.0]] + [[0.1, 0.1]] * 5,
            dtype=np.float32,
        )
        metrics = macro_original_id_metrics(y_true, candidate, groups)
        self.assertAlmostEqual(metrics["rmse"], 0.6, places=6)
        comparison = original_id_win_rate(
            y_true,
            candidate,
            reference,
            groups,
        )
        self.assertEqual(comparison["wins"], 1)
        self.assertEqual(comparison["losses"], 1)
        self.assertEqual(comparison["win_rate"], 0.5)

    def test_latitude_band_edges_are_non_overlapping(self):
        y_true = np.asarray(
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=np.float32,
        )
        y_pred = np.asarray(
            [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
            dtype=np.float32,
        )
        records = latitude_band_metrics(
            y_true,
            y_pred,
            np.asarray([-30.0, 0.0, 90.0]),
        )
        self.assertEqual(sum(record["n_samples"] for record in records), 3)
        self.assertEqual(
            [record["n_samples"] for record in records],
            [1, 1, 1],
        )


if __name__ == "__main__":
    unittest.main()
