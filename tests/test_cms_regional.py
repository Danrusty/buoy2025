"""CMS 逐行筛选、门槛与防泄漏切分回归测试。"""

from __future__ import annotations

import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
sys.path.insert(0, str(MODELS_DIR))

from cms_regional import (  # noqa: E402
    choose_regional_id_splits,
    grouped_id_split,
    prepare_cms_dataset,
    region_memberships,
)


def _trajectory(
    original_id: str,
    *,
    eligible: bool,
    circular: bool,
) -> pd.DataFrame:
    rows = 4
    latitude = np.asarray([25.0, 25.0, 0.0, 25.0], dtype=np.float32)
    longitude = np.asarray([120.0, 120.0, 0.0, 120.0], dtype=np.float32)
    if not eligible:
        latitude[1:] = 0.0
        longitude[1:] = 0.0
    wave_sin = np.full(rows, 0.8 if circular else 0.2)
    wave_cos = np.full(rows, 0.6 if circular else 0.9)
    offset = float(int(original_id.split("-")[-1]))
    return pd.DataFrame(
        {
            "ID": [original_id] * rows,
            "time": pd.date_range("2020-01-01", periods=rows, freq="h"),
            "latitude": latitude,
            "longitude": longitude,
            "ve": np.full(rows, offset + 0.2, dtype=np.float32),
            "vn": np.full(rows, offset - 0.1, dtype=np.float32),
            "original_ID": [original_id] * rows,
            "segment_index": np.zeros(rows, dtype=np.int64),
            "cfsv2_u": np.full(rows, offset, dtype=np.float64),
            "cfsv2_v": np.full(rows, offset, dtype=np.float64),
            "era5_u10": np.full(rows, 3.0, dtype=np.float64),
            "era5_v10": np.full(rows, -2.0, dtype=np.float64),
            "era5_wind_speed": np.full(rows, np.sqrt(13.0)),
            "era5_wind_dir_sin": np.full(rows, -2.0 / np.sqrt(13.0)),
            "era5_wind_dir_cos": np.full(rows, 3.0 / np.sqrt(13.0)),
            "era5_swh": np.full(rows, 1.5, dtype=np.float64),
            "era5_mwp": np.full(rows, 7.0, dtype=np.float64),
            "era5_wave_dir_sin": wave_sin,
            "era5_wave_dir_cos": wave_cos,
        }
    )


class CmsRegionalTest(unittest.TestCase):
    def test_region_memberships_follow_inclusive_rectangles(self) -> None:
        memberships = region_memberships(
            np.asarray([32.0, 23.0, 20.0, 0.0]),
            np.asarray([120.0, 120.0, 110.0, 0.0]),
        )
        np.testing.assert_array_equal(
            memberships["BYS"],
            [True, False, False, False],
        )
        np.testing.assert_array_equal(
            memberships["ECS"],
            [True, True, False, False],
        )
        np.testing.assert_array_equal(
            memberships["NSCS"],
            [False, True, True, False],
        )
        np.testing.assert_array_equal(
            memberships["CMS"],
            [True, True, True, False],
        )

    def test_grouped_split_is_deterministic_and_disjoint(self) -> None:
        ids = [f"buoy-{index:02d}" for index in range(23)]
        first = grouped_id_split(ids, random_seed=42)
        second = grouped_id_split(list(reversed(ids)), random_seed=42)
        self.assertEqual(first, second)
        self.assertEqual(
            [len(first[name]) for name in ("train", "val", "test")],
            [15, 4, 4],
        )
        sets = {name: set(values) for name, values in first.items()}
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])

    def test_global_split_is_rejected_when_eval_ids_are_too_few(self) -> None:
        ids = {f"buoy-{index:02d}" for index in range(10)}
        manifest = {
            "splits": {
                "train": {"original_ids": sorted(ids)[:8]},
                "val": {"original_ids": [sorted(ids)[8]]},
                "test": {"original_ids": [sorted(ids)[9]]},
            }
        }
        selected, provenance = choose_regional_id_splits(
            ids,
            manifest,
            min_inherited_eval_ids=3,
        )
        self.assertEqual(
            provenance["strategy"],
            "regenerated_group_shuffle_split",
        )
        self.assertTrue(provenance["inheritance_rejection_reasons"])
        self.assertEqual(
            set(selected["train"] + selected["val"] + selected["test"]),
            ids,
        )

    def test_prepare_filters_rows_not_whole_global_trajectories(self) -> None:
        v1 = [
            _trajectory(f"buoy-{index:02d}", eligible=True, circular=False)
            for index in range(8)
        ]
        v2 = [
            _trajectory(f"buoy-{index:02d}", eligible=True, circular=True)
            for index in range(8)
        ]
        v1.append(_trajectory("buoy-99", eligible=False, circular=False))
        v2.append(_trajectory("buoy-99", eligible=False, circular=True))

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            v1_path = root / "v1.pkl"
            v2_path = root / "v2.pkl"
            filtered_path = root / "cms.pkl"
            diagnostics_path = root / "cms_diagnostics.json"
            manifest_path = root / "global_split.json"
            artifact_dir = root / "artifacts"
            with v1_path.open("wb") as file:
                pickle.dump(v1, file)
            with v2_path.open("wb") as file:
                pickle.dump(v2, file)
            manifest_path.write_text(
                json.dumps(
                    {
                        "splits": {
                            "train": {
                                "original_ids": [
                                    f"buoy-{index:02d}" for index in range(6)
                                ]
                            },
                            "val": {"original_ids": ["buoy-06"]},
                            "test": {"original_ids": ["buoy-07"]},
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = prepare_cms_dataset(
                mask_source_path=v1_path,
                circular_source_path=v2_path,
                filtered_data_path=filtered_path,
                diagnostics_path=diagnostics_path,
                artifact_dir=artifact_dir,
                global_split_manifest_path=manifest_path,
                min_regional_points=2,
                min_inherited_eval_ids=2,
                code_commit="unit-test",
            )

            with filtered_path.open("rb") as file:
                filtered = pickle.load(file)
            self.assertEqual(len(filtered), 16)
            self.assertEqual(sum(map(len, filtered)), 24)
            self.assertNotIn(
                "buoy-99",
                {
                    str(frame["original_ID"].iloc[0])
                    for frame in filtered
                },
            )
            self.assertTrue(
                all(
                    region_memberships(
                        frame["latitude"],
                        frame["longitude"],
                    )["CMS"].all()
                    for frame in filtered
                )
            )
            self.assertTrue(
                all(
                    np.allclose(frame["era5_wave_dir_sin"], 0.8)
                    for frame in filtered
                )
            )
            self.assertEqual(
                result["statistics"]["cms_dataset"]["n_original_ids"],
                8,
            )
            self.assertEqual(
                result["statistics"]["excluded_by_minimum_threshold"][
                    "n_original_ids"
                ],
                1,
            )
            row_index = np.load(result["row_index_path"])
            self.assertEqual(len(row_index["source_row_index"]), 24)
            self.assertEqual(
                result["statistics"]["split"][
                    "pairwise_original_id_intersections"
                ],
                {"train_val": 0, "train_test": 0, "val_test": 0},
            )


if __name__ == "__main__":
    unittest.main()
