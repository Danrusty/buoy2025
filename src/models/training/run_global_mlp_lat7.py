"""训练 global core6 + ``sin(latitude)`` 七维输入 MLP。

本入口保留 lat7 已冻结的实验定义；实际训练由空间 MLP 共用协议执行，确保
后续 lat9 只改变输入列，不复制或悄悄改变网络与训练逻辑。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from .global_factorial import CORE6_FEATURES, LATITUDE_FEATURE
from .global_mlp_spatial import (
    SpatialMlpSpec,
    run_spatial_mlp,
    standardize_cached_splits as _standardize_cached_splits,
)


RUN_NAME = "global_mlp_lat7_v1"
FEATURE_COLUMNS = list(CORE6_FEATURES) + [LATITUDE_FEATURE]
SPEC = SpatialMlpSpec(
    run_name=RUN_NAME,
    feature_columns=tuple(FEATURE_COLUMNS),
    spatial_cache_keys=("sin_latitude",),
    coordinate_formulas=(
        ("sin_latitude", "sin(deg2rad(latitude))"),
    ),
    unique_variable="在 frozen core6 后追加 sin(latitude) 为第 7 维",
    prediction_filename="mlp_lat7_test_pred.npy",
)


def standardize_cached_splits(
    raw_splits: dict[str, dict[str, np.ndarray]],
    scaler_path: Path,
) -> tuple[dict[str, Any], StandardScaler]:
    """保留 lat7 公开入口，并委托给共用标准化协议。"""
    return _standardize_cached_splits(
        raw_splits,
        scaler_path,
        SPEC,
    )


def run(require_cuda: bool = True) -> dict[str, Any]:
    """运行 lat7 受控实验；已有冻结产物时拒绝覆盖。"""
    return run_spatial_mlp(SPEC, require_cuda=require_cuda)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练 global core6 + sin(latitude) 七维 MLP",
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
