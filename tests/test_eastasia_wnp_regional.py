"""扩展区域行级筛选、lineage 与 adapter membership 测试。"""

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

from cms_global_adapter import build_adapter_data  # noqa: E402
from eastasia_wnp_regional import (  # noqa: E402
    eawnp_memberships,
    prepare_eawnp_dataset,
)


def _trajectory(
    original_id: str,
    *,
    eligible: bool,
    circular: bool,
) -> pd.DataFrame:
    rows = 5
    longitude = np.asarray(
        [130.0, 160.0, 171.0, 160.0, 104.0],
        dtype=np.float64,
    )
    latitude = np.asarray(
        [30.0, 30.0, 30.0, 46.0, 30.0],
        dtype=np.float64,
    )
    if not eligible:
        longitude[1:] = 171.0
    offset = float(sum(map(ord, original_id)) % 10)
    return pd.DataFrame(
        {
            "ID": [original_id] * rows,
            "time": pd.date_range("2020-01-01", periods=rows, freq="h"),
            "latitude": latitude,
            "longitude": longitude,
            "ve": np.full(rows, offset + 0.2),
            "vn": np.full(rows, offset - 0.1),
            "original_ID": [original_id] * rows,
            "segment_index": np.zeros(rows, dtype=np.int64),
            "cfsv2_u": np.full(rows, offset),
            "cfsv2_v": np.full(rows, offset),
            "era5_u10": np.full(rows, 3.0),
            "era5_v10": np.full(rows, -2.0),
            "era5_swh": np.full(rows, 1.5),
            "era5_mwp": np.full(rows, 7.0),
            "era5_wave_dir_sin": np.full(
                rows,
                0.8 if circular else 0.2,
            ),
            "era5_wave_dir_cos": np.full(
                rows,
                0.6 if circular else 0.9,
            ),
        }
    )


class _ZeroSession:
    def run(
        self,
        output_names: list[str],
        inputs: dict[str, np.ndarray],
    ) -> list[np.ndarray]:
        del output_names
        return [np.zeros((len(inputs["input"]), 2), dtype=np.float32)]


class EastAsiaWnpRegionalTest(unittest.TestCase):
    def test_memberships_keep_expanded_rows_outside_original_cms(self) -> None:
        memberships = eawnp_memberships(
            np.asarray([30.0, 30.0, 45.0, 46.0]),
            np.asarray([120.0, 160.0, 170.0, 160.0]),
        )
        np.testing.assert_array_equal(
            memberships["EAWNP"],
            [True, True, True, False],
        )
        np.testing.assert_array_equal(
            memberships["CMS"],
            [True, False, False, False],
        )
        np.testing.assert_array_equal(
            memberships["WEST_105_140"],
            [True, False, False, False],
        )
        np.testing.assert_array_equal(
            memberships["EAST_140_170"],
            [False, True, True, False],
        )

    def test_prepare_filters_rows_and_inherits_lineage(self) -> None:
        ids = ["train-a", "train-b", "val-a", "test-a"]
        mask = [
            _trajectory(value, eligible=True, circular=False)
            for value in ids
        ]
        circular = [
            _trajectory(value, eligible=True, circular=True)
            for value in ids
        ]
        mask.append(
            _trajectory("excluded-a", eligible=False, circular=False)
        )
        circular.append(
            _trajectory("excluded-a", eligible=False, circular=True)
        )
        manifest = {
            "random_seed": 42,
            "splits": {
                "train": {"original_ids": ["train-a", "train-b"]},
                "val": {"original_ids": ["val-a"]},
                "test": {"original_ids": ["test-a", "excluded-a"]},
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mask_path = root / "mask.pkl"
            circular_path = root / "circular.pkl"
            manifest_path = root / "manifest.json"
            filtered_path = root / "filtered.pkl"
            diagnostics_path = root / "diagnostics.json"
            artifact_dir = root / "artifacts"
            with mask_path.open("wb") as file:
                pickle.dump(mask, file)
            with circular_path.open("wb") as file:
                pickle.dump(circular, file)
            manifest_path.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            result = prepare_eawnp_dataset(
                mask_source_path=mask_path,
                circular_source_path=circular_path,
                global_split_manifest_path=manifest_path,
                filtered_data_path=filtered_path,
                diagnostics_path=diagnostics_path,
                artifact_dir=artifact_dir,
                min_regional_points=2,
                code_commit="unit-test",
                expected_mask_sha256=None,
                expected_circular_sha256=None,
                expected_population=None,
            )
            self.assertEqual(
                result["population"],
                {
                    "total": 4,
                    "train": 2,
                    "val": 1,
                    "test": 1,
                    "samples": 8,
                },
            )
            with filtered_path.open("rb") as file:
                frames = pickle.load(file)
            self.assertEqual(sum(map(len, frames)), 8)
            self.assertTrue(
                all(
                    eawnp_memberships(
                        frame["latitude"],
                        frame["longitude"],
                    )["EAWNP"].all()
                    for frame in frames
                )
            )
            self.assertTrue(
                all(
                    np.allclose(frame["era5_wave_dir_sin"], 0.8)
                    for frame in frames
                )
            )
            data = build_adapter_data(
                frames,
                ids,
                _ZeroSession(),
                membership_function=eawnp_memberships,
                required_membership="EAWNP",
            )
            self.assertTrue(data.memberships["EAWNP"].all())
            self.assertFalse(data.memberships["CMS"].all())
            self.assertEqual(len(np.unique(data.groups)), 4)


if __name__ == "__main__":
    unittest.main()
