"""ERA5 mean wave direction 的圆周编码、填补与插值。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.interpolate import NearestNDInterpolator


# 该阈值只防止数值除零，不代表对波向不确定性的科学筛选阈值。
NEAR_ZERO_THRESHOLD = 1.0e-6


@dataclass(frozen=True)
class CircularDirectionResult:
    """圆周插值及归一化结果。"""

    sin: np.ndarray
    cos: np.ndarray
    resultant_length: np.ndarray
    near_zero: np.ndarray


def encode_mwd_degrees(
    mwd_degrees: np.ndarray | xr.DataArray,
) -> tuple[np.ndarray | xr.DataArray, np.ndarray | xr.DataArray]:
    """
    按 ERA5 coming-from 约定编码波向。

    角度保持北向 0 度、顺时针增加，不做 180 度反向转换。
    """
    radians = np.deg2rad(mwd_degrees)
    return np.sin(radians), np.cos(radians)


def normalize_direction_components(
    sin_values: np.ndarray,
    cos_values: np.ndarray,
    near_zero_threshold: float = NEAR_ZERO_THRESHOLD,
) -> CircularDirectionResult:
    """归一化插值后的方向向量，并显式标记无可靠方向的近零向量。"""
    if near_zero_threshold <= 0:
        raise ValueError("near_zero_threshold 必须大于 0。")

    sin_array = np.asarray(sin_values)
    cos_array = np.asarray(cos_values)
    if sin_array.shape != cos_array.shape:
        raise ValueError(
            f"sin/cos shape 不一致: {sin_array.shape} != {cos_array.shape}"
        )

    result_dtype = np.result_type(sin_array.dtype, cos_array.dtype, np.float32)
    sin_array = sin_array.astype(result_dtype, copy=False)
    cos_array = cos_array.astype(result_dtype, copy=False)
    resultant_length = np.hypot(sin_array, cos_array)
    finite = (
        np.isfinite(sin_array)
        & np.isfinite(cos_array)
        & np.isfinite(resultant_length)
    )
    near_zero = finite & (resultant_length < near_zero_threshold)
    safe = finite & ~near_zero

    normalized_sin = np.full(sin_array.shape, np.nan, dtype=result_dtype)
    normalized_cos = np.full(cos_array.shape, np.nan, dtype=result_dtype)
    np.divide(
        sin_array,
        resultant_length,
        out=normalized_sin,
        where=safe,
    )
    np.divide(
        cos_array,
        resultant_length,
        out=normalized_cos,
        where=safe,
    )
    return CircularDirectionResult(
        sin=normalized_sin,
        cos=normalized_cos,
        resultant_length=resultant_length,
        near_zero=near_zero,
    )


def coast_fill_mwd_components(
    mwd: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:
    """
    用同一 mwd 有效掩膜对 sin/cos 分量执行最近海洋格点填补。

    最近邻只选择一个原始方向，不对角度本身做算术平均，因此不会产生
    0/360 度跨界错误。
    """
    required_dims = {"time", "lat", "lon"}
    if not required_dims.issubset(mwd.dims):
        raise ValueError(
            f"mwd 必须包含维度 {sorted(required_dims)}，实际为 {mwd.dims}"
        )

    ordered = mwd.transpose("time", "lat", "lon")
    mwd_values = np.asarray(ordered.values)
    sin_values, cos_values = encode_mwd_degrees(mwd_values)
    sin_filled = np.asarray(sin_values).copy()
    cos_filled = np.asarray(cos_values).copy()

    lat_2d, lon_2d = np.meshgrid(
        ordered.lat.values,
        ordered.lon.values,
        indexing="ij",
    )
    coordinates = np.column_stack([lat_2d.ravel(), lon_2d.ravel()])

    for time_index in range(mwd_values.shape[0]):
        valid_mask = np.isfinite(mwd_values[time_index])
        missing_mask = ~valid_mask
        if not missing_mask.any() or not valid_mask.any():
            continue

        valid_points = coordinates[valid_mask.ravel()]
        missing_points = coordinates[missing_mask.ravel()]
        sin_interp = NearestNDInterpolator(
            valid_points,
            sin_filled[time_index][valid_mask],
        )
        cos_interp = NearestNDInterpolator(
            valid_points,
            cos_filled[time_index][valid_mask],
        )
        sin_filled[time_index][missing_mask] = sin_interp(missing_points)
        cos_filled[time_index][missing_mask] = cos_interp(missing_points)

    coordinates_map = {
        "time": ordered.time,
        "lat": ordered.lat,
        "lon": ordered.lon,
    }
    return (
        xr.DataArray(
            sin_filled,
            dims=("time", "lat", "lon"),
            coords=coordinates_map,
            name="mwd_sin",
        ),
        xr.DataArray(
            cos_filled,
            dims=("time", "lat", "lon"),
            coords=coordinates_map,
            name="mwd_cos",
        ),
    )


def interpolate_mwd_circular(
    mwd: xr.DataArray,
    *,
    lat: xr.DataArray | float,
    lon: xr.DataArray | float,
    time: xr.DataArray | np.datetime64,
    near_zero_threshold: float = NEAR_ZERO_THRESHOLD,
) -> CircularDirectionResult:
    """先插值 mwd 的单位向量分量，再归一化为最终训练特征。"""
    mwd_sin, mwd_cos = coast_fill_mwd_components(mwd)
    interpolated_sin = mwd_sin.interp(
        lat=lat,
        lon=lon,
        time=time,
        method="linear",
    ).values
    interpolated_cos = mwd_cos.interp(
        lat=lat,
        lon=lon,
        time=time,
        method="linear",
    ).values
    return normalize_direction_components(
        interpolated_sin,
        interpolated_cos,
        near_zero_threshold=near_zero_threshold,
    )
