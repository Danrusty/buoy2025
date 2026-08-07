"""训练 global core6 + 纬度正弦 + 经度循环编码九维 MLP。

本实验从已冻结 lat7 定义出发，只追加 ``sin(longitude)`` 和
``cos(longitude)``。网络结构、目标、loss、优化器、early stopping、
随机种子与数据切分均保持不变。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from .global_factorial import CORE6_FEATURES, LATITUDE_FEATURE
from .global_longitude import (
    COS_LONGITUDE_FEATURE,
    LONGITUDE_CACHE_MANIFEST_PATH,
    LONGITUDE_FEATURES,
    SIN_LONGITUDE_FEATURE,
)
from .global_mlp_spatial import (
    SpatialMlpSpec,
    run_spatial_mlp,
    standardize_cached_splits as _standardize_cached_splits,
)


RUN_NAME = "global_mlp_lat9_v1"
FEATURE_COLUMNS = (
    list(CORE6_FEATURES)
    + [LATITUDE_FEATURE]
    + list(LONGITUDE_FEATURES)
)
SPEC = SpatialMlpSpec(
    run_name=RUN_NAME,
    feature_columns=tuple(FEATURE_COLUMNS),
    spatial_cache_keys=(
        "sin_latitude",
        SIN_LONGITUDE_FEATURE,
        COS_LONGITUDE_FEATURE,
    ),
    coordinate_formulas=(
        ("sin_latitude", "sin(deg2rad(latitude))"),
        ("sin_longitude", "sin(deg2rad(longitude))"),
        ("cos_longitude", "cos(deg2rad(longitude))"),
    ),
    unique_variable=(
        "在 frozen lat7 后追加 sin(longitude)、cos(longitude) 为第 8、9 维"
    ),
    prediction_filename="mlp_lat9_test_pred.npy",
    longitude_cache_manifest_path=LONGITUDE_CACHE_MANIFEST_PATH,
)


def standardize_cached_splits(
    raw_splits: dict[str, dict[str, np.ndarray]],
    scaler_path: Path,
) -> tuple[dict[str, Any], StandardScaler]:
    """保留 lat9 公开入口，并委托给共用标准化协议。"""
    return _standardize_cached_splits(
        raw_splits,
        scaler_path,
        SPEC,
    )


def run(require_cuda: bool = True) -> dict[str, Any]:
    """运行 lat9 受控训练与一次 test 评价。"""
    return run_spatial_mlp(SPEC, require_cuda=require_cuda)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "训练 global core6 + sin(latitude) + "
            "sin/cos(longitude) 九维 MLP"
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="仅用于调试；允许在无 CUDA 时继续运行",
    )
    args = parser.parse_args()
    run(require_cuda=not args.allow_cpu)


if __name__ == "__main__":
    main()
