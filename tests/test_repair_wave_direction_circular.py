"""方向专修复生产流程的单元测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PROCESS_DIR = Path(__file__).resolve().parents[1] / "src" / "data_process"
sys.path.insert(0, str(DATA_PROCESS_DIR))

from repair_wave_direction_circular import (  # noqa: E402
    MonthGrid,
    _minimal_longitude_arc,
    _unwrap_longitudes,
    bilinear_interpolate_exact_times,
    build_offsets,
    collect_month_queries,
    initialize_or_resume_work,
    interpolate_trajectory_month,
)


class RepairWaveDirectionTest(unittest.TestCase):
    def test_exact_time_bilinear_ignores_adjacent_nan_slice(self) -> None:
        times = np.asarray(
            ["2020-01-01T00:00", "2020-01-01T01:00"],
            dtype="datetime64[ns]",
        )
        values = np.asarray(
            [
                [[0.0, 2.0], [4.0, 6.0]],
                [[np.nan, np.nan], [np.nan, np.nan]],
            ],
            dtype=np.float32,
        )
        result = bilinear_interpolate_exact_times(
            values,
            grid_times=times,
            grid_latitudes=np.asarray([0.0, 1.0]),
            grid_longitudes=np.asarray([10.0, 11.0]),
            query_times=times[:1],
            query_latitudes=np.asarray([0.5]),
            query_longitudes=np.asarray([10.5]),
        )
        self.assertAlmostEqual(float(result[0]), 3.0)

    def test_longitude_window_unwraps_dateline_continuously(self) -> None:
        start, end = _minimal_longitude_arc(
            np.asarray([359.5, 0.5]),
            padding_degrees=1.0,
        )
        self.assertAlmostEqual(start, 358.5)
        self.assertAlmostEqual(end, 361.5)
        np.testing.assert_allclose(
            _unwrap_longitudes(np.asarray([359.75, 0.25]), start, end),
            [359.75, 360.25],
        )

    def test_local_circular_interpolation_crosses_dateline(self) -> None:
        times = np.asarray(
            ["2020-01-01T00:00", "2020-01-01T01:00"],
            dtype="datetime64[ns]",
        )
        latitudes = np.asarray([0.0, 1.0])
        longitudes = np.asarray([0.0, 1.0, 359.0])
        mwd = np.empty((2, 2, 3), dtype=np.float32)
        mwd[:, :, 0] = 1.0
        mwd[:, :, 1] = 2.0
        mwd[:, :, 2] = 359.0
        grid = MonthGrid(
            times=times,
            latitudes=latitudes,
            longitudes=longitudes,
            mwd=mwd,
        )
        trajectory = pd.DataFrame(
            {
                "latitude": [0.5, 0.5],
                "longitude": [359.75, 0.25],
            }
        )

        result = interpolate_trajectory_month(
            grid=grid,
            trajectory=trajectory,
            query_times=np.asarray(
                ["2020-01-01T00:00", "2020-01-01T00:00"],
                dtype="datetime64[ns]",
            ),
            query_latitudes=np.asarray([0.5, 0.5]),
            query_longitudes=np.asarray([359.75, 0.25]),
        )

        self.assertTrue(np.all(np.abs(result.sin) < 0.03))
        self.assertTrue(np.all(result.cos > 0.99))
        self.assertEqual(result.padding_degrees, 1.0)

    def test_all_nan_local_window_expands_to_valid_mwd(self) -> None:
        times = np.asarray(
            ["2020-01-01T00:00", "2020-01-01T01:00"],
            dtype="datetime64[ns]",
        )
        mwd = np.full((2, 5, 5), np.nan, dtype=np.float32)
        mwd[:, 4, 2] = 90.0
        grid = MonthGrid(
            times=times,
            latitudes=np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]),
            longitudes=np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
            mwd=mwd,
        )
        trajectory = pd.DataFrame(
            {"latitude": [0.0], "longitude": [2.0]}
        )

        result = interpolate_trajectory_month(
            grid=grid,
            trajectory=trajectory,
            query_times=times[:1],
            query_latitudes=np.asarray([0.0]),
            query_longitudes=np.asarray([2.0]),
        )

        self.assertEqual(result.padding_degrees, 2.0)
        self.assertEqual(result.initial_all_nan_time_count, 1)
        self.assertAlmostEqual(float(result.sin[0]), 1.0, places=6)
        self.assertAlmostEqual(float(result.cos[0]), 0.0, places=6)

    def test_collect_month_queries_preserves_flat_positions(self) -> None:
        columns = {
            "latitude": [1.0, 2.0, 3.0],
            "longitude": [10.0, 20.0, 30.0],
            "era5_wave_dir_sin": [0.0, 0.1, 0.2],
            "era5_wave_dir_cos": [1.0, 0.9, 0.8],
        }
        first = pd.DataFrame(
            {
                "time": pd.to_datetime(
                    [
                        "2020-01-31T23:00",
                        "2020-02-01T00:00",
                        "2020-02-01T01:00",
                    ]
                ),
                **columns,
            }
        )
        second = first.iloc[:2].copy()
        trajectories = [first, second]
        offsets, row_counts = build_offsets(trajectories, [0, 1])
        self.assertEqual(row_counts, [3, 2])

        batch = collect_month_queries(
            trajectories,
            [0, 1],
            offsets,
            "2020-02",
        )

        np.testing.assert_array_equal(batch.flat_indices, [1, 2, 4])
        np.testing.assert_array_equal(batch.source_indices, [0, 0, 1])
        np.testing.assert_array_equal(batch.row_positions, [1, 2, 1])

    def test_memmap_work_directory_resumes_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "source.pkl"
            input_path.write_bytes(b"source identity")
            wave_dir = root / "wave"
            wave_dir.mkdir()
            work_dir = root / "work"
            arguments = {
                "work_dir": work_dir,
                "input_path": input_path,
                "wave_dir": wave_dir,
                "selected_indices": [2, 5],
                "row_counts": [3, 2],
                "months": ["2020-01", "2020-02"],
            }

            manifest, sin_output, cos_output = initialize_or_resume_work(
                **arguments
            )
            sin_output[0] = 0.25
            cos_output[0] = 0.75
            sin_output.flush()
            cos_output.flush()
            manifest["completed_months"] = ["2020-01"]
            (work_dir / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            del sin_output, cos_output

            resumed, resumed_sin, resumed_cos = (
                initialize_or_resume_work(**arguments)
            )
            self.assertEqual(resumed["completed_months"], ["2020-01"])
            self.assertAlmostEqual(float(resumed_sin[0]), 0.25)
            self.assertAlmostEqual(float(resumed_cos[0]), 0.75)


if __name__ == "__main__":
    unittest.main()
