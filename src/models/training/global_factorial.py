"""全局纬度信息 × 模型类型析因实验的共用数据与评价协议。

三个实验分支均通过本模块继承冻结 global 模型的 ``original_ID`` 切分，
生成逐行完全一致的共享缓存，并对 MLP 和 XGBoost 预测使用同一套指标定义。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..data_loader import PROJECT_ROOT
from ..evaluation import regression_metrics


EXPERIMENT_NAME = "global_factorial_v1"
FROZEN_GLOBAL_COMMIT = "f2a0170"
SOURCE_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "trajectories_with_all_features_circular_mwd_v2.pkl"
)
SOURCE_SHA256 = (
    "22ab0a32ff9472a6f8b8f57af5fd96b93cdeb76d45b4ef6b0a798fa1befb937e"
)
FROZEN_RUN_DIR = (
    PROJECT_ROOT
    / "trained_models"
    / "ablation_circular_mwd_v2_final"
    / "core_6"
)
FROZEN_SPLIT_MANIFEST_PATH = FROZEN_RUN_DIR / "split_manifest.json"
FROZEN_METRICS_PATH = FROZEN_RUN_DIR / "mlp_metrics.json"
FROZEN_ONNX_PATH = (
    PROJECT_ROOT
    / "deployment"
    / "releases"
    / "wdf_core6_circular_mwd_v2"
    / "wdf_drifter.onnx"
)
CACHE_DIR = PROJECT_ROOT / "processed_data" / EXPERIMENT_NAME
ARTIFACT_DIR = PROJECT_ROOT / "trained_models" / EXPERIMENT_NAME
DATA_MANIFEST_PATH = ARTIFACT_DIR / "data_manifest.json"
FROZEN_REPLAY_PATH = ARTIFACT_DIR / "frozen_global_replay.json"

CORE6_FEATURES = (
    "era5_u10",
    "era5_v10",
    "era5_swh",
    "era5_mwp",
    "era5_wave_dir_sin",
    "era5_wave_dir_cos",
)
LATITUDE_COLUMN = "latitude"
LATITUDE_FEATURE = "sin_latitude"
OBSERVATION_COLUMNS = ("ve", "vn")
CURRENT_COLUMNS = ("cfsv2_u", "cfsv2_v")
TARGET_COLUMNS = ("residual_u", "residual_v")
GROUP_COLUMN = "original_ID"
SPLIT_NAMES = ("train", "val", "test")
LATITUDE_BAND_EDGES = np.asarray(
    [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0],
    dtype=np.float64,
)

logger = logging.getLogger(__name__)


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """以分块方式计算文件 SHA256，避免把完整产物载入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sin_latitude(latitude_degrees: np.ndarray | Iterable[float]) -> np.ndarray:
    """将纬度编码为 float32 的 ``sin(deg2rad(latitude))``。"""
    latitude = np.asarray(latitude_degrees, dtype=np.float64)
    if not np.all(np.isfinite(latitude)):
        raise ValueError("latitude contains non-finite values")
    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise ValueError("latitude must be within [-90, 90] degrees")
    return np.sin(np.deg2rad(latitude)).astype(np.float32)


def _canonical_original_id(frame: Any, trajectory_index: int) -> str:
    if GROUP_COLUMN not in frame.columns:
        raise ValueError(
            f"trajectory {trajectory_index} is missing {GROUP_COLUMN!r}"
        )
    values = (
        frame[GROUP_COLUMN].dropna().astype(str).str.strip().unique()
    )
    if len(values) != 1 or not values[0]:
        raise ValueError(
            f"trajectory {trajectory_index} has {len(values)} canonical "
            f"{GROUP_COLUMN} values: {values[:5].tolist()}"
        )
    return str(values[0])


def _extract_valid_arrays(frame: Any) -> dict[str, np.ndarray]:
    """严格复用冻结 core6 行筛选，并派生纬度输入。"""
    frozen_required = list(
        dict.fromkeys(
            CORE6_FEATURES + OBSERVATION_COLUMNS + CURRENT_COLUMNS
        )
    )
    required = set(frozen_required + [LATITUDE_COLUMN, GROUP_COLUMN])
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"trajectory is missing columns: {sorted(missing)}")

    # 与冻结 loader 的 frame[frozen_required].dropna() 完全等价。
    # 纬度只在该 mask 之后校验，绝不改变任一集合的样本成员关系。
    valid = frame[frozen_required].notna().all(axis=1)
    if not bool(valid.any()):
        return {
            "core6": np.empty((0, len(CORE6_FEATURES)), dtype=np.float32),
            "sin_latitude": np.empty((0,), dtype=np.float32),
            "latitude": np.empty((0,), dtype=np.float32),
            "target": np.empty((0, 2), dtype=np.float32),
        }

    core6 = frame.loc[valid, list(CORE6_FEATURES)].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    latitude = frame.loc[valid, LATITUDE_COLUMN].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    observation = frame.loc[valid, list(OBSERVATION_COLUMNS)].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    current = frame.loc[valid, list(CURRENT_COLUMNS)].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    target = observation - current

    for name, values in (
        ("core6", core6),
        ("latitude", latitude),
        ("target", target),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite retained values")

    return {
        "core6": core6,
        "sin_latitude": sin_latitude(latitude),
        "latitude": latitude,
        "target": target.astype(np.float32, copy=False),
    }


def validate_frozen_split_manifest(
    manifest: dict[str, Any],
) -> dict[str, dict[str, int | str]]:
    """验证三集合 ID 两两不交，并返回严格的 ID 分配索引。"""
    if manifest.get("group_column") != GROUP_COLUMN:
        raise ValueError(
            f"frozen group column is {manifest.get('group_column')!r}, "
            f"expected {GROUP_COLUMN!r}"
        )
    if int(manifest.get("random_seed", -1)) != 42:
        raise ValueError("frozen split random seed must be 42")
    if tuple(manifest.get("feature_columns", ())) != CORE6_FEATURES:
        raise ValueError("frozen split feature order does not match core6")

    lookup: dict[str, dict[str, int | str]] = {}
    seen: set[str] = set()
    for split_name in SPLIT_NAMES:
        split = manifest["splits"][split_name]
        original_ids = [str(value) for value in split["original_ids"]]
        if len(original_ids) != int(split["n_original_ids"]):
            raise ValueError(
                f"{split_name} ID count disagrees with n_original_ids"
            )
        if len(original_ids) != len(set(original_ids)):
            raise ValueError(f"{split_name} contains duplicate original_ID")
        overlap = seen.intersection(original_ids)
        if overlap:
            raise ValueError(
                f"frozen splits leak original_ID values: {sorted(overlap)[:5]}"
            )
        seen.update(original_ids)
        for group_index, original_id in enumerate(original_ids):
            lookup[original_id] = {
                "split": split_name,
                "group_index": group_index,
            }
    return lookup


def _cache_filename(kind: str, split_name: str) -> str:
    return f"{kind}_{split_name}.npy"


def _open_cache_arrays(
    cache_dir: Path,
    split_manifest: dict[str, Any],
) -> dict[str, dict[str, np.memmap]]:
    arrays: dict[str, dict[str, np.memmap]] = {}
    for split_name in SPLIT_NAMES:
        n_samples = int(
            split_manifest["splits"][split_name]["n_samples"]
        )
        arrays[split_name] = {
            "core6": np.lib.format.open_memmap(
                cache_dir / _cache_filename("X_core6", split_name),
                mode="w+",
                dtype=np.float32,
                shape=(n_samples, len(CORE6_FEATURES)),
            ),
            "sin_latitude": np.lib.format.open_memmap(
                cache_dir / _cache_filename("sin_latitude", split_name),
                mode="w+",
                dtype=np.float32,
                shape=(n_samples,),
            ),
            "latitude": np.lib.format.open_memmap(
                cache_dir / _cache_filename("latitude", split_name),
                mode="w+",
                dtype=np.float32,
                shape=(n_samples,),
            ),
            "target": np.lib.format.open_memmap(
                cache_dir / _cache_filename("y", split_name),
                mode="w+",
                dtype=np.float32,
                shape=(n_samples, 2),
            ),
            "group_index": np.lib.format.open_memmap(
                cache_dir / _cache_filename("group_index", split_name),
                mode="w+",
                dtype=np.int32,
                shape=(n_samples,),
            ),
        }
    return arrays


def _flush_cache_arrays(
    arrays: dict[str, dict[str, np.memmap]],
) -> None:
    for split_arrays in arrays.values():
        for values in split_arrays.values():
            values.flush()


def _cache_file_records(
    cache_dir: Path,
    split_manifest: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, dict[str, Any]]] = {}
    shapes = {
        "core6": lambda n: [n, len(CORE6_FEATURES)],
        "sin_latitude": lambda n: [n],
        "latitude": lambda n: [n],
        "target": lambda n: [n, 2],
        "group_index": lambda n: [n],
    }
    dtypes = {
        "core6": "float32",
        "sin_latitude": "float32",
        "latitude": "float32",
        "target": "float32",
        "group_index": "int32",
    }
    prefixes = {
        "core6": "X_core6",
        "sin_latitude": "sin_latitude",
        "latitude": "latitude",
        "target": "y",
        "group_index": "group_index",
    }
    for split_name in SPLIT_NAMES:
        n_samples = int(
            split_manifest["splits"][split_name]["n_samples"]
        )
        records[split_name] = {}
        for kind in shapes:
            path = cache_dir / _cache_filename(
                prefixes[kind],
                split_name,
            )
            records[split_name][kind] = {
                "path": _relative_path(path),
                "shape": shapes[kind](n_samples),
                "dtype": dtypes[kind],
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return records


def prepare_cache(
    source_path: Path = SOURCE_PATH,
    split_manifest_path: Path = FROZEN_SPLIT_MANIFEST_PATH,
    cache_dir: Path = CACHE_DIR,
    artifact_manifest_path: Path = DATA_MANIFEST_PATH,
    expected_source_sha256: str | None = SOURCE_SHA256,
) -> dict[str, Any]:
    """从冻结 global 数据集生成逐行一致的共用缓存。"""
    source_path = source_path.resolve()
    split_manifest_path = split_manifest_path.resolve()
    cache_dir = cache_dir.resolve()
    artifact_manifest_path = artifact_manifest_path.resolve()

    if cache_dir.exists() or artifact_manifest_path.exists():
        if cache_dir.is_dir() and artifact_manifest_path.is_file():
            logger.info("Existing cache found; validating instead of rewriting")
            return validate_cache(
                artifact_manifest_path=artifact_manifest_path,
                verify_checksums=True,
            )
        raise FileExistsError(
            "partial factorial cache exists; inspect before rebuilding: "
            f"{cache_dir}, {artifact_manifest_path}"
        )

    source_size = source_path.stat().st_size
    source_sha256 = sha256_file(source_path)
    if (
        expected_source_sha256 is not None
        and source_sha256 != expected_source_sha256
    ):
        raise ValueError(
            f"source SHA256 mismatch: {source_sha256} != "
            f"{expected_source_sha256}"
        )

    split_manifest = json.loads(
        split_manifest_path.read_text(encoding="utf-8")
    )
    id_lookup = validate_frozen_split_manifest(split_manifest)

    temporary_dir = cache_dir.with_name(
        f".{cache_dir.name}.tmp-{os.getpid()}"
    )
    if temporary_dir.exists():
        raise FileExistsError(temporary_dir)
    temporary_dir.mkdir(parents=True)

    logger.info("Loading source pickle: %s", source_path)
    try:
        with source_path.open("rb") as file:
            trajectories = pickle.load(file)
        logger.info("Loaded %d trajectory segments", len(trajectories))

        arrays = _open_cache_arrays(temporary_dir, split_manifest)
        offsets = {name: 0 for name in SPLIT_NAMES}
        segment_counts = {name: 0 for name in SPLIT_NAMES}
        seen_groups = {name: set() for name in SPLIT_NAMES}

        for trajectory_index, frame in enumerate(trajectories):
            original_id = _canonical_original_id(frame, trajectory_index)
            extracted = _extract_valid_arrays(frame)
            n_rows = len(extracted["latitude"])
            if n_rows == 0:
                continue

            assignment = id_lookup.get(original_id)
            if assignment is None:
                raise ValueError(
                    f"valid original_ID {original_id!r} absent from frozen split"
                )
            split_name = str(assignment["split"])
            group_index = int(assignment["group_index"])
            start = offsets[split_name]
            stop = start + n_rows
            capacity = len(arrays[split_name]["latitude"])
            if stop > capacity:
                raise ValueError(
                    f"{split_name} overflow at trajectory {trajectory_index}: "
                    f"{stop} > {capacity}"
                )

            arrays[split_name]["core6"][start:stop] = extracted["core6"]
            arrays[split_name]["sin_latitude"][start:stop] = extracted[
                "sin_latitude"
            ]
            arrays[split_name]["latitude"][start:stop] = extracted["latitude"]
            arrays[split_name]["target"][start:stop] = extracted["target"]
            arrays[split_name]["group_index"][start:stop] = group_index
            offsets[split_name] = stop
            segment_counts[split_name] += 1
            seen_groups[split_name].add(group_index)

            if (trajectory_index + 1) % 250 == 0:
                logger.info(
                    "Processed %d/%d trajectory segments",
                    trajectory_index + 1,
                    len(trajectories),
                )

        for split_name in SPLIT_NAMES:
            frozen = split_manifest["splits"][split_name]
            checks = {
                "sample": (offsets[split_name], int(frozen["n_samples"])),
                "segment": (
                    segment_counts[split_name],
                    int(frozen["n_segments"]),
                ),
                "group": (
                    len(seen_groups[split_name]),
                    int(frozen["n_original_ids"]),
                ),
            }
            for label, (actual, expected) in checks.items():
                if actual != expected:
                    raise ValueError(
                        f"{split_name} {label} count mismatch: "
                        f"{actual} != {expected}"
                    )

        _flush_cache_arrays(arrays)
        del arrays, trajectories
        gc.collect()
        temporary_dir.replace(cache_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lineage": {
            "frozen_global_branch": "master",
            "frozen_global_commit": FROZEN_GLOBAL_COMMIT,
            "frozen_training_run": _relative_path(FROZEN_RUN_DIR),
            "row_filter": (
                "drop rows missing any frozen core6 feature, ve, vn, "
                "cfsv2_u, or cfsv2_v; latitude is validated but does not "
                "change row membership"
            ),
        },
        "source": {
            "path": _relative_path(source_path),
            "size_bytes": source_size,
            "sha256": source_sha256,
        },
        "frozen_split_manifest": {
            "path": _relative_path(split_manifest_path),
            "sha256": sha256_file(split_manifest_path),
            "random_seed": int(split_manifest["random_seed"]),
        },
        "features": {
            "core6": list(CORE6_FEATURES),
            "latitude_feature": LATITUDE_FEATURE,
            "latitude_formula": "sin(deg2rad(latitude))",
            "lat7_order": list(CORE6_FEATURES) + [LATITUDE_FEATURE],
        },
        "targets": {
            "columns": list(TARGET_COLUMNS),
            "residual_u": "ve - cfsv2_u",
            "residual_v": "vn - cfsv2_v",
        },
        "split_counts": {
            split_name: {
                key: int(split_manifest["splits"][split_name][key])
                for key in ("n_original_ids", "n_segments", "n_samples")
            }
            for split_name in SPLIT_NAMES
        },
        "cache_files": _cache_file_records(cache_dir, split_manifest),
    }
    _json_dump(artifact_manifest_path, payload)
    logger.info("Factorial cache frozen: %s", artifact_manifest_path)
    return payload


def validate_cache(
    artifact_manifest_path: Path = DATA_MANIFEST_PATH,
    verify_checksums: bool = False,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, Any]:
    """校验指定集合的缓存形状、类型、血缘，并可选重算 SHA256。"""
    artifact_manifest_path = artifact_manifest_path.resolve()
    payload = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )
    if payload.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("unexpected factorial cache experiment name")
    if payload["lineage"]["frozen_global_commit"] != FROZEN_GLOBAL_COMMIT:
        raise ValueError("cache does not inherit the frozen global commit")
    if tuple(payload["features"]["core6"]) != CORE6_FEATURES:
        raise ValueError("cached core6 feature order is invalid")
    if payload["features"]["lat7_order"] != (
        list(CORE6_FEATURES) + [LATITUDE_FEATURE]
    ):
        raise ValueError("cached lat7 feature order is invalid")

    unknown_splits = set(split_names) - set(SPLIT_NAMES)
    if unknown_splits:
        raise ValueError(f"未知 split：{sorted(unknown_splits)}")
    if not split_names:
        raise ValueError("split_names 不能为空")

    for split_name in split_names:
        for kind, record in payload["cache_files"][split_name].items():
            path = PROJECT_ROOT / record["path"]
            if not path.is_file():
                raise FileNotFoundError(path)
            values = np.load(path, mmap_mode="r")
            if list(values.shape) != record["shape"]:
                raise ValueError(
                    f"{split_name}.{kind} shape mismatch: "
                    f"{values.shape} != {record['shape']}"
                )
            if str(values.dtype) != record["dtype"]:
                raise ValueError(
                    f"{split_name}.{kind} dtype mismatch: "
                    f"{values.dtype} != {record['dtype']}"
                )
            if path.stat().st_size != int(record["size_bytes"]):
                raise ValueError(f"{split_name}.{kind} size mismatch")
            if verify_checksums and sha256_file(path) != record["sha256"]:
                raise ValueError(f"{split_name}.{kind} SHA256 mismatch")

        group_index = np.load(
            PROJECT_ROOT
            / payload["cache_files"][split_name]["group_index"]["path"],
            mmap_mode="r",
        )
        n_groups = int(
            payload["split_counts"][split_name]["n_original_ids"]
        )
        if len(group_index):
            if int(group_index.min()) != 0:
                raise ValueError(f"{split_name} group_index does not start at 0")
            if int(group_index.max()) != n_groups - 1:
                raise ValueError(
                    f"{split_name} group_index does not cover all IDs"
                )
            if len(np.unique(group_index)) != n_groups:
                raise ValueError(
                    f"{split_name} group_index has missing IDs"
                )
    return payload


def load_cached_split(
    split_name: str,
    artifact_manifest_path: Path = DATA_MANIFEST_PATH,
) -> dict[str, np.ndarray]:
    """把一个冻结集合加载为只读 NumPy memmap。"""
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"unknown split: {split_name}")
    payload = validate_cache(
        artifact_manifest_path=artifact_manifest_path,
        verify_checksums=False,
        split_names=(split_name,),
    )
    return {
        kind: np.load(PROJECT_ROOT / record["path"], mmap_mode="r")
        for kind, record in payload["cache_files"][split_name].items()
    }


def assemble_features(
    core6: np.ndarray,
    latitude_feature: np.ndarray | None = None,
) -> np.ndarray:
    """返回 core6，或把 ``sin_latitude`` 追加为第 7 列。"""
    core = np.asarray(core6)
    if core.ndim != 2 or core.shape[1] != len(CORE6_FEATURES):
        raise ValueError(f"expected core6 shape (N, 6), got {core.shape}")
    if latitude_feature is None:
        return core
    latitude_feature = np.asarray(latitude_feature)
    if latitude_feature.shape != (len(core),):
        raise ValueError(
            f"sin_latitude must have shape ({len(core)},), "
            f"got {latitude_feature.shape}"
        )
    return np.column_stack((core, latitude_feature)).astype(
        np.float32,
        copy=False,
    )


def _per_group_components(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_index: np.ndarray,
) -> dict[str, np.ndarray]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    groups = np.asarray(group_index, dtype=np.int64)
    if true.shape != pred.shape or true.ndim != 2 or true.shape[1] != 2:
        raise ValueError("y_true and y_pred must both have shape (N, 2)")
    if groups.shape != (len(true),):
        raise ValueError("group_index must have shape (N,)")
    if len(groups) == 0 or groups.min() < 0:
        raise ValueError("group_index must be non-empty and non-negative")

    n_groups = int(groups.max()) + 1
    counts = np.bincount(groups, minlength=n_groups).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError("group_index must be contiguous without empty groups")
    error = pred - true

    def by_component(values: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [
                np.bincount(
                    groups,
                    weights=values[:, component],
                    minlength=n_groups,
                )
                for component in range(2)
            ]
        )

    true_sum = by_component(true)
    return {
        "counts": counts,
        "sse": by_component(error**2),
        "sae": by_component(np.abs(error)),
        "error_sum": by_component(error),
        "sst": by_component(true**2) - (true_sum**2 / counts[:, None]),
    }


def macro_original_id_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_index: np.ndarray,
) -> dict[str, float | int]:
    """计算每个物理浮标等权的宏平均指标。"""
    components = _per_group_components(y_true, y_pred, group_index)
    counts = components["counts"]
    sse = components["sse"]
    sst = components["sst"]
    valid_r2 = sst > 0
    r2 = np.full_like(sst, np.nan, dtype=np.float64)
    r2[valid_r2] = 1.0 - (sse[valid_r2] / sst[valid_r2])
    def finite_mean(values: np.ndarray) -> float:
        finite = np.isfinite(values)
        return float(values[finite].mean()) if np.any(finite) else float("nan")

    r2_u = finite_mean(r2[:, 0])
    r2_v = finite_mean(r2[:, 1])
    per_group_rmse = np.sqrt(sse.sum(axis=1) / (2.0 * counts))
    per_group_mae = components["sae"].sum(axis=1) / (2.0 * counts)
    per_group_bias = components["error_sum"] / counts[:, None]
    return {
        "n_original_ids": int(len(counts)),
        "r2_u": r2_u,
        "r2_v": r2_v,
        "r2_joint": float((r2_u + r2_v) / 2.0),
        "rmse": float(per_group_rmse.mean()),
        "mae": float(per_group_mae.mean()),
        "bias_u": float(per_group_bias[:, 0].mean()),
        "bias_v": float(per_group_bias[:, 1].mean()),
    }


def original_id_win_rate(
    y_true: np.ndarray,
    candidate_pred: np.ndarray,
    reference_pred: np.ndarray,
    group_index: np.ndarray,
    tolerance: float = 1e-12,
) -> dict[str, float | int]:
    """按 original_ID 的联合 RMSE 与冻结基准比较。"""
    candidate = _per_group_components(y_true, candidate_pred, group_index)
    reference = _per_group_components(y_true, reference_pred, group_index)
    counts = candidate["counts"]
    candidate_rmse = np.sqrt(
        candidate["sse"].sum(axis=1) / (2.0 * counts)
    )
    reference_rmse = np.sqrt(
        reference["sse"].sum(axis=1) / (2.0 * counts)
    )
    delta = candidate_rmse - reference_rmse
    wins = int(np.sum(delta < -tolerance))
    ties = int(np.sum(np.abs(delta) <= tolerance))
    losses = int(np.sum(delta > tolerance))
    return {
        "n_original_ids": int(len(counts)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": float(wins / len(counts)),
        "median_rmse_difference": float(np.median(delta)),
        "mean_rmse_difference": float(np.mean(delta)),
    }


def latitude_band_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    latitude: np.ndarray,
    edges: np.ndarray = LATITUDE_BAND_EDGES,
) -> list[dict[str, Any]]:
    """报告固定且互不重叠纬度带内的逐行加权指标。"""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    lat = np.asarray(latitude, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if lat.shape != (len(true),):
        raise ValueError("latitude must have shape (N,)")
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("latitude edges must be strictly increasing")

    records: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            mask = (lat >= lower) & (lat <= upper)
            interval = "closed"
        else:
            mask = (lat >= lower) & (lat < upper)
            interval = "left_closed"
        if not np.any(mask):
            continue
        error = np.asarray(pred[mask], dtype=np.float64) - np.asarray(
            true[mask],
            dtype=np.float64,
        )
        records.append(
            {
                "lower_degrees": float(lower),
                "upper_degrees": float(upper),
                "interval": interval,
                "n_samples": int(mask.sum()),
                **regression_metrics(true[mask], pred[mask]),
                "bias_u": float(error[:, 0].mean()),
                "bias_v": float(error[:, 1].mean()),
            }
        )
    return records


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_index: np.ndarray,
    latitude: np.ndarray,
    reference_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    """执行完整且共用的析因实验评价协议。"""
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(
        y_true,
        dtype=np.float64,
    )
    result: dict[str, Any] = {
        "row_weighted": {
            **regression_metrics(y_true, y_pred),
            "bias_u": float(error[:, 0].mean()),
            "bias_v": float(error[:, 1].mean()),
        },
        "macro_original_id": macro_original_id_metrics(
            y_true,
            y_pred,
            group_index,
        ),
        "latitude_bands": latitude_band_metrics(
            y_true,
            y_pred,
            latitude,
        ),
    }
    if reference_pred is not None:
        result["vs_frozen_global_by_original_id"] = original_id_win_rate(
            y_true,
            y_pred,
            reference_pred,
            group_index,
        )
    return result


def predict_onnx_batched(
    onnx_path: Path,
    features: np.ndarray,
    batch_size: int = 262_144,
) -> np.ndarray:
    """以有界 CPU batch 将原始物理量输入 ONNX。"""
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path.resolve()),
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.name != "input" or output_meta.name != "output":
        raise ValueError("unexpected frozen ONNX input/output names")
    if input_meta.type != "tensor(float)":
        raise ValueError(f"unexpected ONNX input dtype: {input_meta.type}")

    predictions = np.empty((len(features), 2), dtype=np.float32)
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        batch = np.asarray(features[start:stop], dtype=np.float32)
        predictions[start:stop] = session.run(
            [output_meta.name],
            {input_meta.name: batch},
        )[0]
    return predictions


def replay_frozen_global(
    output_path: Path = FROZEN_REPLAY_PATH,
    prediction_path: Path | None = None,
) -> dict[str, Any]:
    """在共用测试行回放冻结 global ONNX，并核验历史指标。"""
    split = load_cached_split("test")
    predictions = predict_onnx_batched(
        FROZEN_ONNX_PATH,
        split["core6"],
    )
    prediction_path = prediction_path or (
        CACHE_DIR / "frozen_core6_test_pred.npy"
    )
    np.save(prediction_path, predictions)

    evaluation = evaluate_predictions(
        split["target"],
        predictions,
        split["group_index"],
        split["latitude"],
    )
    reported = json.loads(FROZEN_METRICS_PATH.read_text(encoding="utf-8"))[
        "test"
    ]
    differences = {
        key: float(evaluation["row_weighted"][key] - reported[key])
        for key in ("r2_u", "r2_v", "r2_joint", "rmse", "mae")
    }
    maximum_difference = max(abs(value) for value in differences.values())
    tolerance = 2e-6
    if maximum_difference > tolerance:
        raise ValueError(
            "frozen ONNX replay does not reproduce reported metrics: "
            f"{maximum_difference:.3e} > {tolerance:.3e}"
        )

    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": _relative_path(FROZEN_ONNX_PATH),
            "sha256": sha256_file(FROZEN_ONNX_PATH),
            "input_features": list(CORE6_FEATURES),
            "scaler_inside_graph": True,
        },
        "prediction_cache": {
            "path": _relative_path(prediction_path),
            "shape": list(predictions.shape),
            "dtype": str(predictions.dtype),
            "sha256": sha256_file(prediction_path),
        },
        "reported_test_metrics": reported,
        "replayed_evaluation": evaluation,
        "replay_metric_differences": differences,
        "maximum_absolute_metric_difference": maximum_difference,
        "acceptance_tolerance": tolerance,
        "status": "passed",
    }
    _json_dump(output_path, payload)
    logger.info("Frozen global ONNX replay passed: %s", output_path)
    return payload


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and validate shared global factorial data",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="materialize the shared cache")
    validate_parser = subparsers.add_parser(
        "validate",
        help="validate an existing cache",
    )
    validate_parser.add_argument("--checksums", action="store_true")
    subparsers.add_parser(
        "replay-frozen",
        help="replay the frozen global ONNX on test rows",
    )
    args = parser.parse_args()
    _setup_logging()

    if args.command == "prepare":
        payload = prepare_cache()
        print(json.dumps(payload["split_counts"], indent=2))
    elif args.command == "validate":
        payload = validate_cache(verify_checksums=args.checksums)
        print(json.dumps(payload["split_counts"], indent=2))
    else:
        payload = replay_frozen_global()
        print(
            json.dumps(
                payload["replayed_evaluation"]["row_weighted"],
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
