"""为冻结 global 数据追加循环经度特征缓存。

本模块不改写 ``global_factorial_v1`` 已冻结的数据清单，也不重新定义有效行。
它逐段复用冻结 core6 掩码，并用现有 latitude 与 group_index 缓存核对行顺序，
只追加 ``sin(longitude)`` 和 ``cos(longitude)`` 两列供 MLP lat9 实验使用。
"""

from __future__ import annotations

import argparse
import gc
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
from .global_factorial import (
    ARTIFACT_DIR as FACTORIAL_ARTIFACT_DIR,
    CACHE_DIR,
    CORE6_FEATURES,
    DATA_MANIFEST_PATH,
    FROZEN_SPLIT_MANIFEST_PATH,
    LATITUDE_FEATURE,
    SOURCE_PATH,
    SOURCE_SHA256,
    SPLIT_NAMES,
    _canonical_original_id,
    _extract_valid_arrays,
    frozen_valid_mask,
    load_cached_split,
    sha256_file,
    validate_cache,
    validate_frozen_split_manifest,
)


EXPERIMENT_NAME = "global_longitude_supplement_v1"
LONGITUDE_COLUMN = "longitude"
SIN_LONGITUDE_FEATURE = "sin_longitude"
COS_LONGITUDE_FEATURE = "cos_longitude"
LONGITUDE_FEATURES = (
    SIN_LONGITUDE_FEATURE,
    COS_LONGITUDE_FEATURE,
)
LONGITUDE_CACHE_MANIFEST_PATH = (
    FACTORIAL_ARTIFACT_DIR / "longitude_cache_manifest.json"
)

logger = logging.getLogger(__name__)


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def encode_longitude(
    longitude_degrees: np.ndarray | Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    """将 [-180, 180] 经度编码为 float32 的正弦、余弦分量。"""
    longitude = np.asarray(longitude_degrees, dtype=np.float64)
    if not np.all(np.isfinite(longitude)):
        raise ValueError("longitude 含非有限值")
    if np.any((longitude < -180.0) | (longitude > 180.0)):
        raise ValueError("longitude 必须位于 [-180, 180] 度")
    radians = np.deg2rad(longitude)
    return (
        np.sin(radians).astype(np.float32),
        np.cos(radians).astype(np.float32),
    )


def _cache_filename(feature_name: str, split_name: str) -> str:
    return f"{feature_name}_{split_name}.npy"


def _target_paths(cache_dir: Path) -> dict[str, dict[str, Path]]:
    return {
        split_name: {
            feature: cache_dir / _cache_filename(feature, split_name)
            for feature in LONGITUDE_FEATURES
        }
        for split_name in SPLIT_NAMES
    }


def _open_temporary_arrays(
    temporary_dir: Path,
    split_counts: dict[str, dict[str, int]],
) -> dict[str, dict[str, np.memmap]]:
    arrays: dict[str, dict[str, np.memmap]] = {}
    for split_name in SPLIT_NAMES:
        n_samples = int(split_counts[split_name]["n_samples"])
        arrays[split_name] = {
            feature: np.lib.format.open_memmap(
                temporary_dir / _cache_filename(feature, split_name),
                mode="w+",
                dtype=np.float32,
                shape=(n_samples,),
            )
            for feature in LONGITUDE_FEATURES
        }
    return arrays


def _flush_arrays(
    arrays: dict[str, dict[str, np.memmap]],
) -> None:
    for split_arrays in arrays.values():
        for values in split_arrays.values():
            values.flush()


def _cache_records(
    target_paths: dict[str, dict[str, Path]],
) -> dict[str, dict[str, dict[str, Any]]]:
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for split_name, split_paths in target_paths.items():
        records[split_name] = {}
        for feature, path in split_paths.items():
            values = np.load(path, mmap_mode="r")
            records[split_name][feature] = {
                "path": _relative_path(path),
                "shape": list(values.shape),
                "dtype": str(values.dtype),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    return records


def prepare_longitude_cache(
    source_path: Path = SOURCE_PATH,
    split_manifest_path: Path = FROZEN_SPLIT_MANIFEST_PATH,
    base_data_manifest_path: Path = DATA_MANIFEST_PATH,
    cache_dir: Path = CACHE_DIR,
    artifact_manifest_path: Path = LONGITUDE_CACHE_MANIFEST_PATH,
    expected_source_sha256: str | None = SOURCE_SHA256,
) -> dict[str, Any]:
    """按冻结行顺序生成经度循环编码，并把补充清单最后原子写入。"""
    source_path = source_path.resolve()
    split_manifest_path = split_manifest_path.resolve()
    base_data_manifest_path = base_data_manifest_path.resolve()
    cache_dir = cache_dir.resolve()
    artifact_manifest_path = artifact_manifest_path.resolve()
    target_paths = _target_paths(cache_dir)
    flat_targets = [
        path
        for split_paths in target_paths.values()
        for path in split_paths.values()
    ]

    if artifact_manifest_path.is_file() and all(
        path.is_file() for path in flat_targets
    ):
        logger.info("经度补充缓存已存在，执行完整校验")
        return validate_longitude_cache(
            artifact_manifest_path=artifact_manifest_path,
            verify_checksums=True,
        )
    if artifact_manifest_path.exists() or any(
        path.exists() for path in flat_targets
    ):
        raise FileExistsError(
            "检测到不完整的经度补充缓存，请先人工核查："
            f"{artifact_manifest_path}"
        )

    base_manifest = validate_cache(
        artifact_manifest_path=base_data_manifest_path,
        verify_checksums=True,
    )
    source_sha256 = sha256_file(source_path)
    if (
        expected_source_sha256 is not None
        and source_sha256 != expected_source_sha256
    ):
        raise ValueError(
            f"源数据 SHA256 不一致：{source_sha256} != "
            f"{expected_source_sha256}"
        )
    if base_manifest["source"]["sha256"] != source_sha256:
        raise ValueError("经度源数据与冻结基础清单不是同一文件")

    split_manifest = json.loads(
        split_manifest_path.read_text(encoding="utf-8")
    )
    id_lookup = validate_frozen_split_manifest(split_manifest)
    base_splits = {
        split_name: load_cached_split(
            split_name,
            artifact_manifest_path=base_data_manifest_path,
        )
        for split_name in SPLIT_NAMES
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = cache_dir / (
        f".longitude-supplement.tmp-{os.getpid()}"
    )
    if temporary_dir.exists():
        raise FileExistsError(temporary_dir)
    temporary_dir.mkdir()

    offsets = {name: 0 for name in SPLIT_NAMES}
    segment_counts = {name: 0 for name in SPLIT_NAMES}
    seen_groups = {name: set() for name in SPLIT_NAMES}
    longitude_min = float("inf")
    longitude_max = float("-inf")
    max_unit_circle_deviation = 0.0

    logger.info("载入源轨迹：%s", source_path)
    try:
        with source_path.open("rb") as file:
            trajectories = pickle.load(file)
        logger.info("共载入 %d 个轨迹片段", len(trajectories))
        arrays = _open_temporary_arrays(
            temporary_dir,
            base_manifest["split_counts"],
        )

        for trajectory_index, frame in enumerate(trajectories):
            original_id = _canonical_original_id(frame, trajectory_index)
            base = _extract_valid_arrays(frame)
            n_rows = len(base["latitude"])
            if n_rows == 0:
                continue
            if LONGITUDE_COLUMN not in frame.columns:
                raise ValueError(
                    f"轨迹 {trajectory_index} 缺少 longitude 字段"
                )

            valid = frozen_valid_mask(frame)
            longitude = frame.loc[valid, LONGITUDE_COLUMN].to_numpy(
                dtype=np.float32,
                copy=True,
            )
            if len(longitude) != n_rows:
                raise ValueError(
                    f"轨迹 {trajectory_index} 的经度行数与冻结掩码不一致"
                )
            sin_longitude, cos_longitude = encode_longitude(longitude)

            assignment = id_lookup.get(original_id)
            if assignment is None:
                raise ValueError(
                    f"有效 original_ID {original_id!r} 不在冻结切分中"
                )
            split_name = str(assignment["split"])
            group_index = int(assignment["group_index"])
            start = offsets[split_name]
            stop = start + n_rows
            capacity = len(base_splits[split_name]["latitude"])
            if stop > capacity:
                raise ValueError(
                    f"{split_name} 在轨迹 {trajectory_index} 处越界："
                    f"{stop} > {capacity}"
                )

            cached_latitude = np.asarray(
                base_splits[split_name]["latitude"][start:stop]
            )
            cached_groups = np.asarray(
                base_splits[split_name]["group_index"][start:stop]
            )
            if not np.array_equal(base["latitude"], cached_latitude):
                raise ValueError(
                    f"{split_name} 在轨迹 {trajectory_index} 处纬度行序错位"
                )
            if not np.all(cached_groups == group_index):
                raise ValueError(
                    f"{split_name} 在轨迹 {trajectory_index} 处 ID 行序错位"
                )

            arrays[split_name][SIN_LONGITUDE_FEATURE][
                start:stop
            ] = sin_longitude
            arrays[split_name][COS_LONGITUDE_FEATURE][
                start:stop
            ] = cos_longitude
            offsets[split_name] = stop
            segment_counts[split_name] += 1
            seen_groups[split_name].add(group_index)

            longitude_min = min(longitude_min, float(longitude.min()))
            longitude_max = max(longitude_max, float(longitude.max()))
            deviation = np.max(
                np.abs(
                    sin_longitude.astype(np.float64) ** 2
                    + cos_longitude.astype(np.float64) ** 2
                    - 1.0
                )
            )
            max_unit_circle_deviation = max(
                max_unit_circle_deviation,
                float(deviation),
            )

            if (trajectory_index + 1) % 250 == 0:
                logger.info(
                    "已处理 %d/%d 个轨迹片段",
                    trajectory_index + 1,
                    len(trajectories),
                )

        for split_name in SPLIT_NAMES:
            expected = base_manifest["split_counts"][split_name]
            checks = {
                "样本": (offsets[split_name], int(expected["n_samples"])),
                "片段": (
                    segment_counts[split_name],
                    int(expected["n_segments"]),
                ),
                "original_ID": (
                    len(seen_groups[split_name]),
                    int(expected["n_original_ids"]),
                ),
            }
            for label, (actual, required) in checks.items():
                if actual != required:
                    raise ValueError(
                        f"{split_name} {label}数量不一致："
                        f"{actual} != {required}"
                    )

        _flush_arrays(arrays)
        del arrays, trajectories
        gc.collect()

        for split_name in SPLIT_NAMES:
            for feature in LONGITUDE_FEATURES:
                temporary_path = (
                    temporary_dir
                    / _cache_filename(feature, split_name)
                )
                temporary_path.replace(
                    target_paths[split_name][feature]
                )
        temporary_dir.rmdir()
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "lineage": {
            "base_data_manifest": _relative_path(
                base_data_manifest_path
            ),
            "base_data_manifest_sha256": sha256_file(
                base_data_manifest_path
            ),
            "source": _relative_path(source_path),
            "source_sha256": source_sha256,
            "frozen_split_manifest": _relative_path(
                split_manifest_path
            ),
            "frozen_split_manifest_sha256": sha256_file(
                split_manifest_path
            ),
        },
        "row_contract": {
            "mask": (
                "与 frozen core6 完全相同；longitude 不参与行筛选"
            ),
            "alignment_checks": [
                "逐轨迹 latitude 与基础缓存完全相等",
                "逐轨迹 group_index 与基础缓存完全相等",
                "三集合样本、片段和 original_ID 数量完全相等",
            ],
        },
        "features": {
            "longitude_source_column": LONGITUDE_COLUMN,
            "longitude_features": list(LONGITUDE_FEATURES),
            "sin_formula": "sin(deg2rad(longitude))",
            "cos_formula": "cos(deg2rad(longitude))",
            "lat9_order": (
                list(CORE6_FEATURES)
                + [LATITUDE_FEATURE]
                + list(LONGITUDE_FEATURES)
            ),
        },
        "observed_longitude_range_degrees": [
            longitude_min,
            longitude_max,
        ],
        "max_unit_circle_deviation": max_unit_circle_deviation,
        "split_counts": base_manifest["split_counts"],
        "cache_files": _cache_records(target_paths),
    }
    _json_dump(artifact_manifest_path, payload)
    logger.info("经度补充缓存已冻结：%s", artifact_manifest_path)
    return payload


def validate_longitude_cache(
    artifact_manifest_path: Path = LONGITUDE_CACHE_MANIFEST_PATH,
    verify_checksums: bool = False,
) -> dict[str, Any]:
    """校验经度缓存血缘、形状、类型及单位圆约束。"""
    artifact_manifest_path = artifact_manifest_path.resolve()
    payload = json.loads(
        artifact_manifest_path.read_text(encoding="utf-8")
    )
    if payload.get("experiment") != EXPERIMENT_NAME:
        raise ValueError("经度补充缓存实验名不正确")

    base_manifest_path = _resolve_path(
        payload["lineage"]["base_data_manifest"]
    )
    if (
        sha256_file(base_manifest_path)
        != payload["lineage"]["base_data_manifest_sha256"]
    ):
        raise ValueError("冻结基础数据清单 SHA256 不一致")
    expected_order = (
        list(CORE6_FEATURES)
        + [LATITUDE_FEATURE]
        + list(LONGITUDE_FEATURES)
    )
    if payload["features"]["lat9_order"] != expected_order:
        raise ValueError("lat9 特征顺序不正确")

    for split_name in SPLIT_NAMES:
        loaded: dict[str, np.ndarray] = {}
        for feature in LONGITUDE_FEATURES:
            record = payload["cache_files"][split_name][feature]
            path = _resolve_path(record["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            values = np.load(path, mmap_mode="r")
            if list(values.shape) != record["shape"]:
                raise ValueError(
                    f"{split_name}.{feature} shape 不一致"
                )
            if str(values.dtype) != record["dtype"]:
                raise ValueError(
                    f"{split_name}.{feature} dtype 不一致"
                )
            if path.stat().st_size != int(record["size_bytes"]):
                raise ValueError(
                    f"{split_name}.{feature} 文件大小不一致"
                )
            if verify_checksums and sha256_file(path) != record["sha256"]:
                raise ValueError(
                    f"{split_name}.{feature} SHA256 不一致"
                )
            loaded[feature] = values

        sin_values = loaded[SIN_LONGITUDE_FEATURE]
        cos_values = loaded[COS_LONGITUDE_FEATURE]
        for start in range(0, len(sin_values), 1_000_000):
            stop = min(start + 1_000_000, len(sin_values))
            unit_norm = (
                np.asarray(sin_values[start:stop], dtype=np.float64) ** 2
                + np.asarray(cos_values[start:stop], dtype=np.float64) ** 2
            )
            if not np.all(np.isfinite(unit_norm)):
                raise ValueError(
                    f"{split_name} 经度编码含非有限值"
                )
            if float(np.max(np.abs(unit_norm - 1.0))) > 2e-6:
                raise ValueError(
                    f"{split_name} 经度编码偏离单位圆"
                )
    return payload


def load_longitude_split(
    split_name: str,
    artifact_manifest_path: Path = LONGITUDE_CACHE_MANIFEST_PATH,
) -> dict[str, np.ndarray]:
    """加载一个冻结集合的经度正弦、余弦只读 memmap。"""
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"未知 split：{split_name}")
    payload = validate_longitude_cache(
        artifact_manifest_path=artifact_manifest_path,
        verify_checksums=False,
    )
    return {
        feature: np.load(
            _resolve_path(
                payload["cache_files"][split_name][feature]["path"]
            ),
            mmap_mode="r",
        )
        for feature in LONGITUDE_FEATURES
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成或校验 frozen global 经度循环编码补充缓存",
    )
    parser.add_argument(
        "command",
        choices=("prepare", "validate"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.command == "prepare":
        payload = prepare_longitude_cache()
    else:
        payload = validate_longitude_cache(verify_checksums=True)
    print(
        json.dumps(
            {
                "experiment": payload["experiment"],
                "split_counts": payload["split_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
