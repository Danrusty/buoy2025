"""冻结 global 经度循环编码补充缓存测试。"""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.training.global_factorial import (
    CORE6_FEATURES,
    prepare_cache,
)
from src.models.training.global_longitude import (
    COS_LONGITUDE_FEATURE,
    SIN_LONGITUDE_FEATURE,
    encode_longitude,
    load_longitude_split,
    prepare_longitude_cache,
)


class GlobalLongitudeTest(unittest.TestCase):
    @staticmethod
    def _frame(original_id, latitudes, longitudes):
        n_rows = len(latitudes)
        sequence = np.arange(n_rows, dtype=np.float32)
        return pd.DataFrame(
            {
                "original_ID": [original_id] * n_rows,
                "latitude": latitudes,
                "longitude": longitudes,
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

    def test_longitude_encoding_is_circular_at_dateline(self):
        sin_value, cos_value = encode_longitude(
            [-180.0, -90.0, 0.0, 90.0, 180.0]
        )
        np.testing.assert_allclose(
            sin_value,
            [0.0, -1.0, 0.0, 1.0, 0.0],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            cos_value,
            [-1.0, 0.0, 1.0, 0.0, -1.0],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            [sin_value[0], cos_value[0]],
            [sin_value[-1], cos_value[-1]],
            atol=1e-7,
        )

    def test_supplement_inherits_exact_base_row_order(self):
        trajectories = [
            self._frame("A", [10.0, 20.0], [170.0, -170.0]),
            self._frame("B", [30.0], [90.0]),
            self._frame("C", [-10.0], [0.0]),
            self._frame("D", [-20.0, -30.0], [-90.0, 180.0]),
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
            base_manifest_path = root / "base_manifest.json"
            longitude_manifest_path = root / "longitude_manifest.json"
            with source_path.open("wb") as file:
                pickle.dump(trajectories, file)
            split_path.write_text(json.dumps(manifest), encoding="utf-8")

            prepare_cache(
                source_path=source_path,
                split_manifest_path=split_path,
                cache_dir=cache_dir,
                artifact_manifest_path=base_manifest_path,
                expected_source_sha256=None,
            )
            payload = prepare_longitude_cache(
                source_path=source_path,
                split_manifest_path=split_path,
                base_data_manifest_path=base_manifest_path,
                cache_dir=cache_dir,
                artifact_manifest_path=longitude_manifest_path,
                expected_source_sha256=None,
            )
            train = load_longitude_split(
                "train",
                artifact_manifest_path=longitude_manifest_path,
            )
            expected_sin, expected_cos = encode_longitude(
                [170.0, -170.0, 90.0]
            )
            np.testing.assert_allclose(
                train[SIN_LONGITUDE_FEATURE],
                expected_sin,
            )
            np.testing.assert_allclose(
                train[COS_LONGITUDE_FEATURE],
                expected_cos,
            )
            self.assertEqual(
                payload["split_counts"]["train"]["n_samples"],
                3,
            )
            self.assertIn(
                "逐轨迹 latitude 与基础缓存完全相等",
                payload["row_contract"]["alignment_checks"],
            )


if __name__ == "__main__":
    unittest.main()
