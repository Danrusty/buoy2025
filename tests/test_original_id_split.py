"""original_ID 分组切分的最小回归测试。"""

from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
sys.path.insert(0, str(MODELS_DIR))

from data_loader import (  # noqa: E402
    FEATURE_COLS,
    load_and_split_data,
    validate_predefined_id_splits,
)


CORE_FEATURES = [
    "era5_u10",
    "era5_v10",
    "era5_swh",
    "era5_mwp",
    "era5_wave_dir_sin",
    "era5_wave_dir_cos",
]


def _trajectory(original_id: str, offset: float) -> pd.DataFrame:
    rows = 4
    data = {
        column: np.full(rows, offset + index, dtype=np.float32)
        for index, column in enumerate(FEATURE_COLS)
    }
    data.update(
        {
            "original_ID": [original_id] * rows,
            "ve": np.full(rows, offset + 0.2, dtype=np.float32),
            "vn": np.full(rows, offset - 0.1, dtype=np.float32),
            "cfsv2_u": np.full(rows, offset, dtype=np.float32),
            "cfsv2_v": np.full(rows, offset, dtype=np.float32),
        }
    )
    return pd.DataFrame(data)


class OriginalIdSplitTest(unittest.TestCase):
    def test_all_segments_of_one_buoy_stay_in_one_split(self) -> None:
        trajectories = []
        for index in range(20):
            original_id = f"buoy-{index:02d}"
            trajectories.append(_trajectory(original_id, float(index)))
            trajectories.append(_trajectory(original_id, float(index) + 0.5))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = temp_path / "trajectories.pkl"
            with data_path.open("wb") as file:
                pickle.dump(trajectories, file)

            splits = load_and_split_data(
                filepath=data_path,
                artifact_dir=temp_path / "artifacts",
                sample_mode=False,
            )

            id_sets = {
                name: set(splits["id_splits"][name])
                for name in ("train", "val", "test")
            }
            self.assertFalse(id_sets["train"] & id_sets["val"])
            self.assertFalse(id_sets["train"] & id_sets["test"])
            self.assertFalse(id_sets["val"] & id_sets["test"])
            self.assertEqual(set.union(*id_sets.values()), {
                f"buoy-{index:02d}" for index in range(20)
            })

            for name, ids in id_sets.items():
                self.assertEqual(
                    splits["split_stats"][name]["n_segments"],
                    2 * len(ids),
                )

            np.testing.assert_allclose(
                splits["X_train"].mean(axis=0),
                np.zeros(len(FEATURE_COLS)),
                atol=1e-6,
            )
            self.assertTrue((temp_path / "artifacts" / "x_scaler.pkl").is_file())
            self.assertTrue(
                (temp_path / "artifacts" / "split_manifest.json").is_file()
            )

            core_splits = load_and_split_data(
                filepath=data_path,
                artifact_dir=temp_path / "core_artifacts",
                sample_mode=False,
                feature_cols=CORE_FEATURES,
            )
            self.assertEqual(core_splits["X_train"].shape[1], 6)
            self.assertEqual(core_splits["feature_cols"], CORE_FEATURES)
            self.assertEqual(core_splits["id_splits"], splits["id_splits"])

    def test_predefined_split_is_used_and_validated(self) -> None:
        trajectories = [
            _trajectory(f"buoy-{index:02d}", float(index))
            for index in range(10)
        ]
        predefined = {
            "train": [f"buoy-{index:02d}" for index in range(6)],
            "val": ["buoy-06", "buoy-07"],
            "test": ["buoy-08", "buoy-09"],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_path = temp_path / "trajectories.pkl"
            with data_path.open("wb") as file:
                pickle.dump(trajectories, file)

            splits = load_and_split_data(
                filepath=data_path,
                artifact_dir=temp_path / "artifacts",
                sample_mode=False,
                feature_cols=CORE_FEATURES,
                predefined_id_splits=predefined,
                split_provenance={"source": "unit-test"},
            )
            self.assertEqual(splits["id_splits"], predefined)

            manifest = (
                temp_path / "artifacts" / "split_manifest.json"
            ).read_text(encoding="utf-8")
            self.assertIn('"split_strategy": "predefined_original_id_split"', manifest)
            self.assertIn('"source": "unit-test"', manifest)

        with self.assertRaisesRegex(ValueError, "完整覆盖"):
            validate_predefined_id_splits(
                {
                    "train": ["buoy-00"],
                    "val": ["buoy-01"],
                    "test": ["buoy-02"],
                },
                [f"buoy-{index:02d}" for index in range(4)],
            )


if __name__ == "__main__":
    unittest.main()
