"""波向圆周插值的最小回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import xarray as xr


DATA_PROCESS_DIR = Path(__file__).resolve().parents[1] / "src" / "data_process"
sys.path.insert(0, str(DATA_PROCESS_DIR))

from wave_direction import (  # noqa: E402
    encode_mwd_degrees,
    interpolate_mwd_circular,
)


def _data_array(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=np.float32),
        dims=("time", "lat", "lon"),
        coords={
            "time": np.asarray(
                ["2020-01-01T00:00", "2020-01-01T01:00"],
                dtype="datetime64[ns]",
            ),
            "lat": [0.0, 1.0],
            "lon": [10.0, 11.0],
        },
        name="mwd",
    )


class WaveDirectionTest(unittest.TestCase):
    def test_coming_from_contract_does_not_add_180_degrees(self) -> None:
        sin_value, cos_value = encode_mwd_degrees(np.asarray([90.0]))
        np.testing.assert_allclose(sin_value, [1.0], atol=1e-7)
        np.testing.assert_allclose(cos_value, [0.0], atol=1e-7)

    def test_spatial_and_temporal_wrap_interpolate_to_north(self) -> None:
        spatial = _data_array(
            [
                [[359.0, 1.0], [359.0, 1.0]],
                [[359.0, 1.0], [359.0, 1.0]],
            ]
        )
        spatial_result = interpolate_mwd_circular(
            spatial,
            lat=0.5,
            lon=10.5,
            time=np.datetime64("2020-01-01T00:00"),
        )
        self.assertAlmostEqual(float(spatial_result.sin), 0.0, places=6)
        self.assertAlmostEqual(float(spatial_result.cos), 1.0, places=6)

        temporal = _data_array(
            [
                [[359.0, 359.0], [359.0, 359.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        )
        temporal_result = interpolate_mwd_circular(
            temporal,
            lat=0.5,
            lon=10.5,
            time=np.datetime64("2020-01-01T00:30"),
        )
        self.assertAlmostEqual(float(temporal_result.sin), 0.0, places=6)
        self.assertAlmostEqual(float(temporal_result.cos), 1.0, places=6)

    def test_constant_direction_and_coast_fill_remain_unit_length(self) -> None:
        values = np.full((2, 2, 2), 45.0, dtype=np.float32)
        values[:, 0, 1] = np.nan
        result = interpolate_mwd_circular(
            _data_array(values),
            lat=0.25,
            lon=10.75,
            time=np.datetime64("2020-01-01T00:30"),
        )
        expected = np.sqrt(0.5)
        self.assertAlmostEqual(float(result.sin), expected, places=6)
        self.assertAlmostEqual(float(result.cos), expected, places=6)
        self.assertAlmostEqual(
            float(np.hypot(result.sin, result.cos)),
            1.0,
            places=6,
        )

    def test_opposite_directions_are_reported_as_near_zero(self) -> None:
        opposing = _data_array(
            [
                [[90.0, 270.0], [90.0, 270.0]],
                [[90.0, 270.0], [90.0, 270.0]],
            ]
        )
        result = interpolate_mwd_circular(
            opposing,
            lat=0.5,
            lon=10.5,
            time=np.datetime64("2020-01-01T00:00"),
        )
        self.assertTrue(bool(result.near_zero))
        self.assertLess(float(result.resultant_length), 1.0e-6)
        self.assertTrue(np.isnan(result.sin))
        self.assertTrue(np.isnan(result.cos))


if __name__ == "__main__":
    unittest.main()
