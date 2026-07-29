"""
仅重算 ERA5 mean wave direction 的圆周插值特征。

该脚本以现有 v1 全特征数据集为底稿，只替换
``era5_wave_dir_sin`` 和 ``era5_wave_dir_cos``。处理按月份集中进行：

1. 每个月只读取一次 ERA5 ``mwd`` 全球网格；
2. 按整条轨迹的固定空间窗口提取当月内存切片；
3. 先将原始角度编码为 sin/cos，再按 v1 的同一有效掩膜和局部窗口
   填补海岸缺测，随后插值并归一化；
4. 将结果写入磁盘 memmap，逐月更新 manifest，支持中断续跑；
5. 全部月份完成后，原位替换 v1 数据中的两个波向列并保存独立 v2。

ERA5 文件当前按 ``(time=1, lat=361, lon=720)`` 分块。逐轨迹裁剪仍会
读取完整的全球小时块，因此按月份集中读取能显著减少重复 I/O。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import pandas as pd
import xarray as xr

from wave_direction import (
    NEAR_ZERO_THRESHOLD,
    coast_fill_mwd_components,
    normalize_direction_components,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "processed_data" / "trajectories_with_all_features.pkl"
)
DEFAULT_WAVE_DIR = PROJECT_ROOT / "reanalysis" / "wave"
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "trajectories_with_all_features_circular_mwd_v2.pkl"
)
DEFAULT_WORK_DIR = (
    PROJECT_ROOT / "processed_data" / "wave_direction_circular_v2_work"
)

WORK_SCHEMA_VERSION = 3
DIAGNOSTIC_SCHEMA_VERSION = 1
ALGORITHM_VERSION = "mwd_circular_exact_hour_local_month_v3"
PATCH_DTYPE = np.dtype("float64")
WINDOW_PADDING_SEQUENCE = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
WAVE_DIRECTION_COLUMNS = (
    "era5_wave_dir_sin",
    "era5_wave_dir_cos",
)
REQUIRED_COLUMNS = (
    "time",
    "latitude",
    "longitude",
    *WAVE_DIRECTION_COLUMNS,
)

logger = logging.getLogger("repair_wave_direction_circular")


@dataclass(frozen=True)
class QueryBatch:
    """一个月份内所有待插值轨迹点的扁平批次。"""

    flat_indices: np.ndarray
    source_indices: np.ndarray
    row_positions: np.ndarray
    times: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    old_sin: np.ndarray
    old_cos: np.ndarray


@dataclass(frozen=True)
class MonthGrid:
    """一个 ERA5 月份的原始 mwd 网格。"""

    times: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray
    mwd: np.ndarray


@dataclass(frozen=True)
class TrajectoryMonthResult:
    """单条轨迹单月插值结果及缺测扩窗诊断。"""

    sin: np.ndarray
    cos: np.ndarray
    padding_degrees: float
    initial_all_nan_time_count: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法序列化为 JSON: {type(value)!r}")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """先写同目录临时文件，再原子替换，避免中断留下半份 manifest。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_ready,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def parse_selected_indices(
    value: str | None,
    trajectory_count: int,
) -> list[int]:
    """解析逗号分隔的源轨迹索引；空值表示选择全部轨迹。"""
    if value is None:
        return list(range(trajectory_count))

    selected = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        index = int(token)
        if not 0 <= index < trajectory_count:
            raise ValueError(
                f"轨迹索引越界: {index}，有效范围 0~{trajectory_count - 1}"
            )
        if index not in selected:
            selected.append(index)
    if not selected:
        raise ValueError("--indices 未包含有效轨迹索引。")
    return selected


def build_wave_catalog(wave_dir: Path) -> dict[str, Path]:
    """建立 YYYY-MM 到 ERA5 月文件的唯一映射。"""
    catalog: dict[str, Path] = {}
    for path in sorted(wave_dir.glob("wave_*.nc")):
        parts = path.stem.split("_")
        if len(parts) < 2 or len(parts[1]) < 6:
            continue
        token = parts[1][:6]
        try:
            month = pd.Period(token, freq="M").strftime("%Y-%m")
        except ValueError:
            continue
        if month in catalog:
            raise ValueError(
                f"月份 {month} 存在多个 ERA5 文件: "
                f"{catalog[month]} 和 {path}"
            )
        catalog[month] = path.resolve()
    if not catalog:
        raise FileNotFoundError(f"{wave_dir} 中未找到 wave_YYYYMM*.nc")
    return catalog


def _decode_era5_times(values: np.ndarray) -> np.ndarray:
    """将 ERA5 浮点 YYYYMMDD.fraction 或 datetime 坐标统一为 ns。"""
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.datetime64):
        return array.astype("datetime64[ns]")
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"不支持的 ERA5 时间 dtype: {array.dtype}")

    time_float = array.astype(np.float64)
    date_ints = time_float.astype(np.int64)
    fractions = time_float - date_ints
    hours = np.rint(fractions * 24.0).astype(np.int64)
    dates = pd.to_datetime(date_ints, format="%Y%m%d")
    decoded = dates + pd.to_timedelta(hours, unit="h")
    return decoded.to_numpy(dtype="datetime64[ns]")


def _read_raw_mwd(
    path: Path,
    *,
    time_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """只读取 mwd，避免把同文件中的 swh/mwp 一并载入。"""
    with xr.open_dataset(path) as dataset:
        if "mwd" not in dataset:
            raise KeyError(f"{path} 缺少 mwd 变量。")
        mwd = dataset["mwd"]
        rename = {}
        if "latitude" in mwd.dims:
            rename["latitude"] = "lat"
        if "longitude" in mwd.dims:
            rename["longitude"] = "lon"
        if "valid_time" in mwd.dims and "time" not in mwd.dims:
            rename["valid_time"] = "time"
        if rename:
            mwd = mwd.rename(rename)
        required_dims = {"time", "lat", "lon"}
        if not required_dims.issubset(mwd.dims):
            raise ValueError(
                f"{path} 的 mwd 维度不符合预期: {mwd.dims}"
            )

        mwd = mwd.transpose("time", "lat", "lon")
        raw_times = _decode_era5_times(mwd.time.values)
        latitudes = np.asarray(mwd.lat.values, dtype=np.float64)
        longitudes = np.mod(
            np.asarray(mwd.lon.values, dtype=np.float64),
            360.0,
        )

        if time_index is None:
            values = np.asarray(mwd.load().values, dtype=np.float32)
            times = raw_times
        else:
            values = np.asarray(
                mwd.isel(time=[time_index]).load().values,
                dtype=np.float32,
            )
            times = raw_times[[time_index]]

    if latitudes[0] > latitudes[-1]:
        latitudes = latitudes[::-1]
        values = values[:, ::-1, :]

    longitude_order = np.argsort(longitudes)
    if not np.array_equal(longitude_order, np.arange(len(longitudes))):
        longitudes = longitudes[longitude_order]
        values = values[:, :, longitude_order]

    time_order = np.argsort(times)
    times = times[time_order]
    values = values[time_order]
    unique_times, unique_positions = np.unique(times, return_index=True)
    values = values[unique_positions]

    if len(latitudes) < 2 or len(longitudes) < 2 or len(unique_times) < 1:
        raise ValueError(f"{path} 的网格坐标长度不足。")
    if np.any(np.diff(latitudes) <= 0):
        raise ValueError(f"{path} 的纬度坐标不是严格递增。")
    if np.any(np.diff(longitudes) <= 0):
        raise ValueError(f"{path} 的经度坐标不是严格递增。")

    return unique_times, latitudes, longitudes, values


def load_month_grid(path: Path) -> MonthGrid:
    """每个月只从磁盘读取一次原始 mwd 全球网格。"""
    times, latitudes, longitudes, mwd = _read_raw_mwd(path)
    return MonthGrid(
        times=times,
        latitudes=latitudes,
        longitudes=longitudes,
        mwd=mwd,
    )


def _minimal_longitude_arc(
    longitudes: np.ndarray,
    padding_degrees: float = 1.0,
) -> tuple[float, float]:
    """返回覆盖所有轨迹经度的最短连续圆周区间，可超出 0~360。"""
    values = np.unique(
        np.mod(np.asarray(longitudes, dtype=np.float64), 360.0)
    )
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("轨迹经度为空或包含非有限值。")
    if padding_degrees < 0:
        raise ValueError("经度 padding 不能为负数。")
    if values.size == 1:
        return (
            float(values[0] - padding_degrees),
            float(values[0] + padding_degrees),
        )

    gaps = np.diff(np.concatenate([values, values[:1] + 360.0]))
    largest_gap = int(np.argmax(gaps))
    start = float(values[(largest_gap + 1) % len(values)])
    end = float(values[largest_gap])
    if end < start:
        end += 360.0
    return start - padding_degrees, end + padding_degrees


def _unwrap_longitudes(
    longitudes: np.ndarray,
    interval_start: float,
    interval_end: float,
) -> np.ndarray:
    """将查询经度移动到给定的连续圆周区间附近。"""
    values = np.mod(np.asarray(longitudes, dtype=np.float64), 360.0)
    center = 0.5 * (interval_start + interval_end)
    return values + 360.0 * np.rint((center - values) / 360.0)


def _local_mwd_window(
    grid: MonthGrid,
    trajectory: pd.DataFrame,
    query_times: np.ndarray,
    query_latitudes: np.ndarray,
    query_longitudes: np.ndarray,
    padding_degrees: float = 1.0,
) -> tuple[xr.DataArray, np.ndarray]:
    """
    按整条轨迹的空间包围盒裁剪当月内存网格，保持 v1 的填补作用域。
    """
    trajectory_latitudes = trajectory["latitude"].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(trajectory_latitudes).all():
        raise ValueError("轨迹纬度包含非有限值。")
    lat_min = max(
        float(grid.latitudes[0]),
        float(trajectory_latitudes.min() - padding_degrees),
    )
    lat_max = min(
        float(grid.latitudes[-1]),
        float(trajectory_latitudes.max() + padding_degrees),
    )
    lat_start = max(
        0,
        int(np.searchsorted(grid.latitudes, lat_min, side="left")),
    )
    lat_stop = min(
        len(grid.latitudes),
        int(np.searchsorted(grid.latitudes, lat_max, side="right")),
    )

    lon_start, lon_end = _minimal_longitude_arc(
        trajectory["longitude"].to_numpy(dtype=np.float64),
        padding_degrees=padding_degrees,
    )
    candidate_longitudes = np.concatenate(
        [
            grid.longitudes - 360.0,
            grid.longitudes,
            grid.longitudes + 360.0,
        ]
    )
    candidate_indices = np.tile(
        np.arange(len(grid.longitudes), dtype=np.int64),
        3,
    )
    longitude_mask = (
        (candidate_longitudes >= lon_start)
        & (candidate_longitudes <= lon_end)
    )
    local_longitudes = candidate_longitudes[longitude_mask]
    local_lon_indices = candidate_indices[longitude_mask]
    order = np.argsort(local_longitudes)
    local_longitudes = local_longitudes[order]
    local_lon_indices = local_lon_indices[order]

    query_times = np.asarray(query_times, dtype="datetime64[ns]")
    if (
        query_times.min() < grid.times[0]
        or query_times.max() > grid.times[-1]
    ):
        raise ValueError(
            "轨迹时间超出所属月份 ERA5 时间范围；"
            "当前数据应为整点小时，不能跨月外插。"
        )
    time_start = max(
        0,
        int(np.searchsorted(grid.times, query_times.min(), side="left")) - 1,
    )
    time_stop = min(
        len(grid.times),
        int(np.searchsorted(grid.times, query_times.max(), side="right")) + 1,
    )

    if (
        lat_stop - lat_start < 2
        or len(local_longitudes) < 2
        or time_stop - time_start < 2
    ):
        raise ValueError(
            "局部 mwd 窗口至少需要 2 个 time/lat/lon 网格点。"
        )

    local_values = grid.mwd[
        time_start:time_stop,
        lat_start:lat_stop,
        :,
    ][:, :, local_lon_indices]
    local = xr.DataArray(
        local_values,
        dims=("time", "lat", "lon"),
        coords={
            "time": grid.times[time_start:time_stop],
            "lat": grid.latitudes[lat_start:lat_stop],
            "lon": local_longitudes,
        },
        name="mwd",
    )
    adjusted_query_longitudes = _unwrap_longitudes(
        query_longitudes,
        lon_start,
        lon_end,
    )
    return local, adjusted_query_longitudes


def bilinear_interpolate_exact_times(
    values: np.ndarray,
    *,
    grid_times: np.ndarray,
    grid_latitudes: np.ndarray,
    grid_longitudes: np.ndarray,
    query_times: np.ndarray,
    query_latitudes: np.ndarray,
    query_longitudes: np.ndarray,
) -> np.ndarray:
    """
    精确选择 ERA5 整点时间层，仅执行二维双线性空间插值。

    轨迹时间均为整点。显式选择时间层可避免相邻全 NaN 海冰时次通过
    三维线性插值污染本来有效的当前时次。
    """
    query_times = np.asarray(query_times, dtype="datetime64[ns]")
    grid_times = np.asarray(grid_times, dtype="datetime64[ns]")
    query_latitudes = np.asarray(query_latitudes, dtype=np.float64)
    query_longitudes = np.asarray(query_longitudes, dtype=np.float64)
    grid_latitudes = np.asarray(grid_latitudes, dtype=np.float64)
    grid_longitudes = np.asarray(grid_longitudes, dtype=np.float64)

    if values.shape != (
        len(grid_times),
        len(grid_latitudes),
        len(grid_longitudes),
    ):
        raise ValueError("方向分量 shape 与局部网格坐标不一致。")
    if not (
        len(query_times)
        == len(query_latitudes)
        == len(query_longitudes)
    ):
        raise ValueError("查询 time/latitude/longitude 长度不一致。")

    time_indices = np.searchsorted(grid_times, query_times)
    valid_time_index = time_indices < len(grid_times)
    matched_time = np.zeros(len(query_times), dtype=bool)
    matched_time[valid_time_index] = (
        grid_times[time_indices[valid_time_index]]
        == query_times[valid_time_index]
    )
    if not matched_time.all():
        examples = query_times[~matched_time][:10].astype(str).tolist()
        raise ValueError(
            "轨迹时间未精确匹配 ERA5 整点时次，不能执行双线性插值。"
            f"示例: {examples}"
        )

    def brackets(
        grid: np.ndarray,
        query: np.ndarray,
        axis_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.any(query < grid[0]) or np.any(query > grid[-1]):
            raise ValueError(
                f"{axis_name} 查询超出局部网格范围: "
                f"query=[{query.min()}, {query.max()}], "
                f"grid=[{grid[0]}, {grid[-1]}]"
            )
        upper = np.searchsorted(grid, query, side="right")
        upper = np.clip(upper, 1, len(grid) - 1)
        lower = upper - 1
        weight = (query - grid[lower]) / (grid[upper] - grid[lower])
        return lower, upper, weight

    y0, y1, wy = brackets(
        grid_latitudes,
        query_latitudes,
        "latitude",
    )
    x0, x1, wx = brackets(
        grid_longitudes,
        query_longitudes,
        "longitude",
    )
    value_00 = values[time_indices, y0, x0]
    value_01 = values[time_indices, y0, x1]
    value_10 = values[time_indices, y1, x0]
    value_11 = values[time_indices, y1, x1]
    lower_lat = value_00 * (1.0 - wx) + value_01 * wx
    upper_lat = value_10 * (1.0 - wx) + value_11 * wx
    return lower_lat * (1.0 - wy) + upper_lat * wy


def interpolate_trajectory_month(
    *,
    grid: MonthGrid,
    trajectory: pd.DataFrame,
    query_times: np.ndarray,
    query_latitudes: np.ndarray,
    query_longitudes: np.ndarray,
) -> TrajectoryMonthResult:
    """按 v1 的局部海岸填补规则，圆周插值一条轨迹的一个月份。"""
    local_mwd = None
    adjusted_longitudes = None
    initial_all_nan_time_count = 0
    selected_padding = WINDOW_PADDING_SEQUENCE[-1]
    for padding_degrees in WINDOW_PADDING_SEQUENCE:
        candidate, candidate_longitudes = _local_mwd_window(
            grid,
            trajectory,
            query_times,
            query_latitudes,
            query_longitudes,
            padding_degrees=padding_degrees,
        )
        time_indices = np.searchsorted(
            candidate.time.values,
            np.asarray(query_times, dtype="datetime64[ns]"),
        )
        unique_time_indices = np.unique(time_indices)
        all_nan_time_count = sum(
            not np.isfinite(candidate.values[index]).any()
            for index in unique_time_indices
        )
        if padding_degrees == WINDOW_PADDING_SEQUENCE[0]:
            initial_all_nan_time_count = int(all_nan_time_count)
        if all_nan_time_count == 0:
            local_mwd = candidate
            adjusted_longitudes = candidate_longitudes
            selected_padding = padding_degrees
            break

    if local_mwd is None or adjusted_longitudes is None:
        raise RuntimeError(
            "扩大到 64 度后，查询时次的 mwd 局部网格仍全部缺测。"
        )

    mwd_sin, mwd_cos = coast_fill_mwd_components(local_mwd)
    interpolation_args = {
        "grid_times": local_mwd.time.values,
        "grid_latitudes": local_mwd.lat.values,
        "grid_longitudes": local_mwd.lon.values,
        "query_times": query_times,
        "query_latitudes": query_latitudes,
        "query_longitudes": adjusted_longitudes,
    }
    interpolated_sin = bilinear_interpolate_exact_times(
        mwd_sin.values,
        **interpolation_args,
    )
    interpolated_cos = bilinear_interpolate_exact_times(
        mwd_cos.values,
        **interpolation_args,
    )
    return TrajectoryMonthResult(
        sin=interpolated_sin,
        cos=interpolated_cos,
        padding_degrees=selected_padding,
        initial_all_nan_time_count=initial_all_nan_time_count,
    )


def _month_token(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype="datetime64[ns]").astype("datetime64[M]")


def selected_months(
    trajectories: list[pd.DataFrame],
    selected_indices: Iterable[int],
) -> list[str]:
    months: set[str] = set()
    for source_index in selected_indices:
        times = pd.to_datetime(
            trajectories[source_index]["time"],
            errors="raise",
        ).to_numpy(dtype="datetime64[ns]")
        month_values = np.unique(_month_token(times))
        months.update(
            pd.Period(str(value), freq="M").strftime("%Y-%m")
            for value in month_values
        )
    return sorted(months)


def build_offsets(
    trajectories: list[pd.DataFrame],
    selected_indices: list[int],
) -> tuple[np.ndarray, list[int]]:
    row_counts = [len(trajectories[index]) for index in selected_indices]
    offsets = np.zeros(len(row_counts) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(row_counts, dtype=np.int64)
    return offsets, row_counts


def collect_month_queries(
    trajectories: list[pd.DataFrame],
    selected_indices: list[int],
    offsets: np.ndarray,
    month: str,
) -> QueryBatch:
    """收集一个月份的全部选中轨迹点，并保留回写位置。"""
    target_month = np.datetime64(month, "M")
    fields: dict[str, list[np.ndarray]] = {
        "flat_indices": [],
        "source_indices": [],
        "row_positions": [],
        "times": [],
        "latitudes": [],
        "longitudes": [],
        "old_sin": [],
        "old_cos": [],
    }

    for output_index, source_index in enumerate(selected_indices):
        trajectory = trajectories[source_index]
        missing = set(REQUIRED_COLUMNS) - set(trajectory.columns)
        if missing:
            raise ValueError(
                f"第 {source_index} 条轨迹缺少列: {sorted(missing)}"
            )
        times = pd.to_datetime(
            trajectory["time"],
            errors="raise",
        ).to_numpy(dtype="datetime64[ns]")
        positions = np.flatnonzero(_month_token(times) == target_month)
        if positions.size == 0:
            continue

        fields["flat_indices"].append(offsets[output_index] + positions)
        fields["source_indices"].append(
            np.full(positions.size, source_index, dtype=np.int32)
        )
        fields["row_positions"].append(positions.astype(np.int64))
        fields["times"].append(times[positions])
        fields["latitudes"].append(
            trajectory["latitude"].to_numpy(dtype=np.float64)[positions]
        )
        fields["longitudes"].append(
            trajectory["longitude"].to_numpy(dtype=np.float64)[positions]
        )
        fields["old_sin"].append(
            trajectory[WAVE_DIRECTION_COLUMNS[0]].to_numpy(
                dtype=np.float64
            )[positions]
        )
        fields["old_cos"].append(
            trajectory[WAVE_DIRECTION_COLUMNS[1]].to_numpy(
                dtype=np.float64
            )[positions]
        )

    if not fields["times"]:
        empty_float = np.empty(0, dtype=np.float64)
        empty_int = np.empty(0, dtype=np.int64)
        return QueryBatch(
            flat_indices=empty_int,
            source_indices=np.empty(0, dtype=np.int32),
            row_positions=empty_int,
            times=np.empty(0, dtype="datetime64[ns]"),
            latitudes=empty_float,
            longitudes=empty_float,
            old_sin=empty_float,
            old_cos=empty_float,
        )

    return QueryBatch(
        flat_indices=np.concatenate(fields["flat_indices"]),
        source_indices=np.concatenate(fields["source_indices"]),
        row_positions=np.concatenate(fields["row_positions"]),
        times=np.concatenate(fields["times"]),
        latitudes=np.concatenate(fields["latitudes"]),
        longitudes=np.concatenate(fields["longitudes"]),
        old_sin=np.concatenate(fields["old_sin"]),
        old_cos=np.concatenate(fields["old_cos"]),
    )


def _circular_angle_difference_degrees(
    old_sin: np.ndarray,
    old_cos: np.ndarray,
    new_sin: np.ndarray,
    new_cos: np.ndarray,
) -> np.ndarray:
    cross = old_cos * new_sin - old_sin * new_cos
    dot = old_sin * new_sin + old_cos * new_cos
    return np.abs(np.rad2deg(np.arctan2(cross, dot)))


def process_month(
    *,
    month: str,
    batch: QueryBatch,
    catalog: dict[str, Path],
    trajectories: list[pd.DataFrame],
    sin_output: np.memmap,
    cos_output: np.memmap,
) -> dict[str, Any]:
    started = perf_counter()
    grid = load_month_grid(catalog[month])
    load_seconds = perf_counter() - started

    interpolation_started = perf_counter()
    interpolated_sin = np.empty(len(batch.times), dtype=np.float64)
    interpolated_cos = np.empty(len(batch.times), dtype=np.float64)
    source_indices = np.unique(batch.source_indices)
    expanded_window_chunks = 0
    expanded_window_points = 0
    maximum_padding_degrees = WINDOW_PADDING_SEQUENCE[0]
    expanded_window_examples = []
    for source_index in source_indices:
        positions = np.flatnonzero(batch.source_indices == source_index)
        local_result = interpolate_trajectory_month(
            grid=grid,
            trajectory=trajectories[int(source_index)],
            query_times=batch.times[positions],
            query_latitudes=batch.latitudes[positions],
            query_longitudes=batch.longitudes[positions],
        )
        interpolated_sin[positions] = local_result.sin
        interpolated_cos[positions] = local_result.cos
        maximum_padding_degrees = max(
            maximum_padding_degrees,
            local_result.padding_degrees,
        )
        if local_result.padding_degrees > WINDOW_PADDING_SEQUENCE[0]:
            expanded_window_chunks += 1
            expanded_window_points += int(len(positions))
            if len(expanded_window_examples) < 20:
                expanded_window_examples.append(
                    {
                        "source_trajectory_index": int(source_index),
                        "month": month,
                        "points": int(len(positions)),
                        "initial_all_nan_time_count": (
                            local_result.initial_all_nan_time_count
                        ),
                        "padding_degrees": (
                            local_result.padding_degrees
                        ),
                    }
                )

    direction = normalize_direction_components(
        interpolated_sin,
        interpolated_cos,
        near_zero_threshold=NEAR_ZERO_THRESHOLD,
    )
    interpolation_seconds = perf_counter() - interpolation_started

    nonfinite = (
        ~np.isfinite(direction.resultant_length)
        | (
            (~np.isfinite(direction.sin) | ~np.isfinite(direction.cos))
            & ~direction.near_zero
        )
    )
    if nonfinite.any():
        positions = np.flatnonzero(nonfinite)[:20]
        examples = [
            {
                "source_trajectory_index": int(
                    batch.source_indices[position]
                ),
                "row_position": int(batch.row_positions[position]),
                "time": str(batch.times[position]),
                "latitude": float(batch.latitudes[position]),
                "longitude": float(batch.longitudes[position]),
            }
            for position in positions
        ]
        raise RuntimeError(
            f"{month} 出现 {np.count_nonzero(nonfinite)} 个非近零缺测结果，"
            f"示例: {examples}"
        )

    sin_output[batch.flat_indices] = direction.sin.astype(
        PATCH_DTYPE,
        copy=False,
    )
    cos_output[batch.flat_indices] = direction.cos.astype(
        PATCH_DTYPE,
        copy=False,
    )
    sin_output.flush()
    cos_output.flush()

    finite_resultant = direction.resultant_length[
        np.isfinite(direction.resultant_length)
    ]
    near_zero_positions = np.flatnonzero(direction.near_zero)
    examples = []
    for position in near_zero_positions[:20]:
        examples.append(
            {
                "source_trajectory_index": int(
                    batch.source_indices[position]
                ),
                "row_position": int(batch.row_positions[position]),
                "time": str(batch.times[position]),
                "latitude": float(batch.latitudes[position]),
                "longitude": float(batch.longitudes[position]),
                "resultant_length": float(
                    direction.resultant_length[position]
                ),
            }
        )

    angle_difference = _circular_angle_difference_degrees(
        batch.old_sin,
        batch.old_cos,
        direction.sin,
        direction.cos,
    )
    finite_difference = angle_difference[np.isfinite(angle_difference)]
    return {
        "points": int(len(batch.times)),
        "trajectory_month_chunks": int(len(source_indices)),
        "expanded_window_chunks": expanded_window_chunks,
        "expanded_window_points": expanded_window_points,
        "maximum_padding_degrees": maximum_padding_degrees,
        "expanded_window_examples": expanded_window_examples,
        "load_seconds": load_seconds,
        "interpolation_seconds": interpolation_seconds,
        "total_seconds": perf_counter() - started,
        "finite_resultant_count": int(finite_resultant.size),
        "minimum_resultant_length": (
            float(finite_resultant.min())
            if finite_resultant.size
            else None
        ),
        "below_0_1_count": int(
            np.count_nonzero(finite_resultant < 0.1)
        ),
        "near_zero_count": int(near_zero_positions.size),
        "near_zero_examples": examples,
        "angle_difference_v1_vs_v2": {
            "finite_count": int(finite_difference.size),
            "changed_over_1_degree": int(
                np.count_nonzero(finite_difference > 1.0)
            ),
            "changed_over_10_degrees": int(
                np.count_nonzero(finite_difference > 10.0)
            ),
            "maximum_degrees": (
                float(finite_difference.max())
                if finite_difference.size
                else None
            ),
        },
    }


def _dataset_signature(
    selected_indices: list[int],
    row_counts: list[int],
) -> str:
    payload = json.dumps(
        {
            "selected_indices": selected_indices,
            "row_counts": row_counts,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def initialize_or_resume_work(
    *,
    work_dir: Path,
    input_path: Path,
    wave_dir: Path,
    selected_indices: list[int],
    row_counts: list[int],
    months: list[str],
) -> tuple[dict[str, Any], np.memmap, np.memmap]:
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = work_dir / "manifest.json"
    sin_path = work_dir / "wave_dir_sin.npy"
    cos_path = work_dir / "wave_dir_cos.npy"
    total_rows = int(sum(row_counts))
    source_stat = input_path.stat()

    expected = {
        "source_path": str(input_path.resolve()),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "wave_dir": str(wave_dir.resolve()),
        "selected_indices": selected_indices,
        "row_counts": row_counts,
        "dataset_signature": _dataset_signature(
            selected_indices,
            row_counts,
        ),
        "months": months,
        "total_rows": total_rows,
        "patch_dtype": PATCH_DTYPE.name,
        "near_zero_threshold": NEAR_ZERO_THRESHOLD,
        "algorithm_version": ALGORITHM_VERSION,
        "code_git_commit": _git_commit(),
    }

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, expected_value in expected.items():
            if manifest.get(key) != expected_value:
                raise ValueError(
                    f"工作目录与本次输入不兼容: {key}，"
                    f"现有={manifest.get(key)!r}，预期={expected_value!r}"
                )
        if not sin_path.exists() or not cos_path.exists():
            raise FileNotFoundError("manifest 存在但 memmap 文件缺失。")
        sin_output = np.load(sin_path, mmap_mode="r+")
        cos_output = np.load(cos_path, mmap_mode="r+")
        if sin_output.shape != (total_rows,) or cos_output.shape != (
            total_rows,
        ):
            raise ValueError("现有 memmap shape 与 manifest 不一致。")
        logger.info(
            "恢复工作目录，已完成月份: %s",
            manifest.get("completed_months", []),
        )
        return manifest, sin_output, cos_output

    if sin_path.exists() or cos_path.exists():
        raise FileExistsError(
            f"{work_dir} 中存在 memmap 但没有 manifest，拒绝覆盖。"
        )
    sin_output = np.lib.format.open_memmap(
        sin_path,
        mode="w+",
        dtype=PATCH_DTYPE,
        shape=(total_rows,),
    )
    cos_output = np.lib.format.open_memmap(
        cos_path,
        mode="w+",
        dtype=PATCH_DTYPE,
        shape=(total_rows,),
    )
    sin_output[:] = np.nan
    cos_output[:] = np.nan
    sin_output.flush()
    cos_output.flush()

    manifest = {
        "schema_version": WORK_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        **expected,
        "completed_months": [],
        "month_statistics": {},
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest, sin_output, cos_output


def _aggregate_month_statistics(
    month_statistics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    stats = list(month_statistics.values())
    minimum_values = [
        item["minimum_resultant_length"]
        for item in stats
        if item.get("minimum_resultant_length") is not None
    ]
    examples = []
    for item in stats:
        examples.extend(item.get("near_zero_examples", []))
        if len(examples) >= 20:
            break
    expanded_examples = []
    for item in stats:
        expanded_examples.extend(item.get("expanded_window_examples", []))
        if len(expanded_examples) >= 20:
            break
    return {
        "finite_resultant_count": sum(
            item.get("finite_resultant_count", 0) for item in stats
        ),
        "minimum_resultant_length": (
            min(minimum_values) if minimum_values else None
        ),
        "below_0_1_count": sum(
            item.get("below_0_1_count", 0) for item in stats
        ),
        "near_zero_count": sum(
            item.get("near_zero_count", 0) for item in stats
        ),
        "near_zero_examples": examples[:20],
        "near_zero_threshold": NEAR_ZERO_THRESHOLD,
        "expanded_window_chunks": sum(
            item.get("expanded_window_chunks", 0) for item in stats
        ),
        "expanded_window_points": sum(
            item.get("expanded_window_points", 0) for item in stats
        ),
        "maximum_padding_degrees": max(
            (
                item.get(
                    "maximum_padding_degrees",
                    WINDOW_PADDING_SEQUENCE[0],
                )
                for item in stats
            ),
            default=WINDOW_PADDING_SEQUENCE[0],
        ),
        "expanded_window_examples": expanded_examples[:20],
    }


def assemble_output(
    *,
    trajectories: list[pd.DataFrame],
    selected_indices: list[int],
    offsets: np.ndarray,
    sin_output: np.memmap,
    cos_output: np.memmap,
    output_path: Path,
    input_path: Path,
    work_manifest: dict[str, Any],
    compute_hashes: bool,
) -> dict[str, Any]:
    """验证补丁完整性，原位替换两列，并保存独立 v2 Pickle。"""
    expected_months = set(work_manifest["months"])
    completed_months = set(work_manifest["completed_months"])
    missing_months = sorted(expected_months - completed_months)
    if missing_months:
        raise RuntimeError(f"仍有月份未完成: {missing_months}")

    sin_values = np.asarray(sin_output)
    cos_values = np.asarray(cos_output)
    finite = np.isfinite(sin_values) & np.isfinite(cos_values)
    if not finite.all():
        positions = np.flatnonzero(~finite)[:20].tolist()
        raise RuntimeError(
            f"波向补丁存在 {np.count_nonzero(~finite)} 个非有限值，"
            f"示例扁平位置: {positions}"
        )
    norm = np.hypot(sin_values, cos_values)
    unit_error = np.abs(norm - 1.0)

    angle_differences = np.empty(len(sin_values), dtype=np.float32)
    output_trajectories = []
    for output_index, source_index in enumerate(selected_indices):
        start = int(offsets[output_index])
        stop = int(offsets[output_index + 1])
        trajectory = trajectories[source_index]
        old_sin = trajectory[WAVE_DIRECTION_COLUMNS[0]].to_numpy(
            dtype=np.float64
        )
        old_cos = trajectory[WAVE_DIRECTION_COLUMNS[1]].to_numpy(
            dtype=np.float64
        )
        angle_differences[start:stop] = (
            _circular_angle_difference_degrees(
                old_sin,
                old_cos,
                sin_values[start:stop],
                cos_values[start:stop],
            ).astype(np.float32)
        )
        trajectory[WAVE_DIRECTION_COLUMNS[0]] = sin_values[start:stop]
        trajectory[WAVE_DIRECTION_COLUMNS[1]] = cos_values[start:stop]
        output_trajectories.append(trajectory)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_name(f"{output_path.name}.tmp")
    with temporary_output.open("wb") as file:
        pickle.dump(
            output_trajectories,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(temporary_output, output_path)

    finite_difference = angle_differences[
        np.isfinite(angle_differences)
    ]
    diagnostics = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "direction_convention": {
            "source": "ERA5 mean_wave_direction",
            "meaning": "coming-from",
            "zero_direction": "north",
            "positive_rotation": "clockwise",
            "add_180_degrees": False,
            "sin_definition": "sin(deg2rad(mwd))",
            "cos_definition": "cos(deg2rad(mwd))",
        },
        "processing": {
            "mode": "mwd_only_month_centric",
            "algorithm_version": ALGORITHM_VERSION,
            "code_git_commit": work_manifest.get("code_git_commit"),
            "source_file": str(input_path.resolve()),
            "output_file": str(output_path.resolve()),
            "selected_trajectory_count": len(selected_indices),
            "selected_source_indices": selected_indices,
            "total_rows": int(len(sin_values)),
            "months": work_manifest["months"],
            "non_direction_columns_modified": False,
            "patch_dtype": PATCH_DTYPE.name,
        },
        "circular_statistics": {
            **_aggregate_month_statistics(
                work_manifest["month_statistics"]
            ),
            "unit_norm_max_abs_error": float(unit_error.max()),
            "angle_difference_v1_vs_v2": {
                "finite_count": int(finite_difference.size),
                "changed_over_1_degree": int(
                    np.count_nonzero(finite_difference > 1.0)
                ),
                "changed_over_10_degrees": int(
                    np.count_nonzero(finite_difference > 10.0)
                ),
                "p50_degrees": float(
                    np.percentile(finite_difference, 50)
                ),
                "p90_degrees": float(
                    np.percentile(finite_difference, 90)
                ),
                "p99_degrees": float(
                    np.percentile(finite_difference, 99)
                ),
                "maximum_degrees": float(finite_difference.max()),
            },
        },
        "file_integrity": {
            "source_size_bytes": input_path.stat().st_size,
            "output_size_bytes": output_path.stat().st_size,
            "source_sha256": (
                _sha256_file(input_path) if compute_hashes else None
            ),
            "output_sha256": (
                _sha256_file(output_path) if compute_hashes else None
            ),
        },
        "month_statistics": work_manifest["month_statistics"],
    }
    diagnostics_path = output_path.with_name(
        f"{output_path.stem}_diagnostics.json"
    )
    _write_json_atomic(diagnostics_path, diagnostics)
    return diagnostics


def repair_wave_direction(
    *,
    input_path: Path,
    wave_dir: Path,
    output_path: Path,
    work_dir: Path,
    selected_indices_text: str | None = None,
    max_months: int | None = None,
    compute_hashes: bool = True,
) -> dict[str, Any] | None:
    """执行可恢复的方向专修复；未完成全部月份时返回 None。"""
    input_path = input_path.resolve()
    wave_dir = wave_dir.resolve()
    output_path = output_path.resolve()
    work_dir = work_dir.resolve()

    logger.info("加载 v1 数据集: %s", input_path)
    with input_path.open("rb") as file:
        trajectories = pickle.load(file)
    logger.info("已加载 %d 条轨迹", len(trajectories))

    selected_indices = parse_selected_indices(
        selected_indices_text,
        len(trajectories),
    )
    offsets, row_counts = build_offsets(
        trajectories,
        selected_indices,
    )
    months = selected_months(trajectories, selected_indices)
    catalog = build_wave_catalog(wave_dir)
    missing_wave_months = sorted(set(months) - set(catalog))
    if missing_wave_months:
        raise FileNotFoundError(
            f"缺少以下月份的 ERA5 波浪文件: {missing_wave_months}"
        )
    logger.info(
        "选择 %d 条轨迹、%d 个点、%d 个月份",
        len(selected_indices),
        int(offsets[-1]),
        len(months),
    )

    manifest, sin_output, cos_output = initialize_or_resume_work(
        work_dir=work_dir,
        input_path=input_path,
        wave_dir=wave_dir,
        selected_indices=selected_indices,
        row_counts=row_counts,
        months=months,
    )
    completed = set(manifest["completed_months"])
    pending = [month for month in months if month not in completed]
    if max_months is not None:
        if max_months < 0:
            raise ValueError("--max-months 不能为负数。")
        pending = pending[:max_months]

    manifest_path = work_dir / "manifest.json"
    for month in pending:
        batch = collect_month_queries(
            trajectories,
            selected_indices,
            offsets,
            month,
        )
        logger.info(
            "处理 %s: %d 个轨迹点，ERA5=%s",
            month,
            len(batch.times),
            catalog[month].name,
        )
        stats = process_month(
            month=month,
            batch=batch,
            catalog=catalog,
            trajectories=trajectories,
            sin_output=sin_output,
            cos_output=cos_output,
        )
        manifest["month_statistics"][month] = stats
        manifest["completed_months"] = sorted(
            set(manifest["completed_months"]) | {month}
        )
        manifest["updated_at_utc"] = _utc_now()
        _write_json_atomic(manifest_path, manifest)
        logger.info(
            "%s 完成: %.1fs（读取 %.1fs，局部填补/插值 %.1fs），"
            "min(r)=%s，近零=%d",
            month,
            stats["total_seconds"],
            stats["load_seconds"],
            stats["interpolation_seconds"],
            stats["minimum_resultant_length"],
            stats["near_zero_count"],
        )

    incomplete = sorted(set(months) - set(manifest["completed_months"]))
    if incomplete:
        logger.info(
            "按 --max-months 暂停，剩余 %d 个月；再次运行相同命令即可恢复。",
            len(incomplete),
        )
        return None

    aggregate = _aggregate_month_statistics(
        manifest["month_statistics"]
    )
    if aggregate["near_zero_count"] > 0:
        raise RuntimeError(
            "全量处理发现近零方向向量，已保留工作目录和示例，"
            "未生成最终 v2。请先检查 manifest 中的 near_zero_examples。"
        )

    logger.info("全部月份完成，开始组装独立 v2 数据集。")
    diagnostics = assemble_output(
        trajectories=trajectories,
        selected_indices=selected_indices,
        offsets=offsets,
        sin_output=sin_output,
        cos_output=cos_output,
        output_path=output_path,
        input_path=input_path,
        work_manifest=manifest,
        compute_hashes=compute_hashes,
    )
    logger.info("v2 数据集已保存: %s", output_path)
    logger.info(
        "诊断报告: %s",
        output_path.with_name(f"{output_path.stem}_diagnostics.json"),
    )
    return diagnostics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="仅重算 ERA5 mwd 圆周插值并生成可恢复的 v2 数据集。"
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )
    parser.add_argument(
        "--wave-dir",
        type=Path,
        default=DEFAULT_WAVE_DIR,
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
    )
    parser.add_argument(
        "--indices",
        help="逗号分隔的源轨迹索引；省略时处理全部轨迹。",
    )
    parser.add_argument(
        "--max-months",
        type=int,
        help="本次最多处理多少个尚未完成的月份，用于恢复机制测试。",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="跳过输入/输出 SHA256，仅用于小样本性能测试。",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        help="单一运行日志路径；默认写入工作目录 repair.log。",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    log_path = args.log_path or args.work_dir / "repair.log"
    _configure_logging(log_path.resolve())
    try:
        repair_wave_direction(
            input_path=args.input_path,
            wave_dir=args.wave_dir,
            output_path=args.output_path,
            work_dir=args.work_dir,
            selected_indices_text=args.indices,
            max_months=args.max_months,
            compute_hashes=not args.skip_hashes,
        )
    except Exception:
        logger.exception("波向圆周修复失败。")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
