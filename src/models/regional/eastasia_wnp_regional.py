"""15–45 N、105–170 E 扩展区域的严格行级数据准备。"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data_loader import PROJECT_ROOT
from .cms_regional import (
    CIRCULAR_SOURCE_PATH,
    GLOBAL_SPLIT_MANIFEST_PATH,
    IDENTITY_COLUMNS,
    MASK_SOURCE_PATH,
    MIN_REGIONAL_POINTS,
    MODEL_REQUIRED_COLUMNS,
    _canonical_original_id,
    _contiguous_hourly_runs,
    region_memberships,
)


MODEL_VERSION = "wdf_eastasia_wnp_global_adapter_v1"
LATITUDE_RANGE = (15.0, 45.0)
LONGITUDE_RANGE = (105.0, 170.0)
EXPECTED_MASK_SOURCE_SHA256 = (
    "7f516210e0198a40584f19519cea5ac6e524dd27208925d902619d7234734608"
)
EXPECTED_CIRCULAR_SOURCE_SHA256 = (
    "22ab0a32ff9472a6f8b8f57af5fd96b93cdeb76d45b4ef6b0a798fa1befb937e"
)
EXPECTED_POPULATION = {
    "total": 96,
    "train": 75,
    "val": 12,
    "test": 9,
    "samples": 454892,
}
FILTERED_DATA_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "trajectories_eastasia_wnp_105_170_circular_mwd_v2.pkl"
)
FILTERED_DIAGNOSTICS_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "trajectories_eastasia_wnp_105_170_circular_mwd_v2_diagnostics.json"
)
ARTIFACT_DIR = PROJECT_ROOT / "trained_models" / MODEL_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def eawnp_rectangle_mask(
    latitude: np.ndarray | pd.Series,
    longitude: np.ndarray | pd.Series,
) -> np.ndarray:
    lat = np.asarray(latitude)
    lon = np.asarray(longitude)
    if lat.shape != lon.shape:
        raise ValueError(f"经纬度 shape 不一致: {lat.shape} vs {lon.shape}")
    return (
        np.isfinite(lat)
        & np.isfinite(lon)
        & (lat >= LATITUDE_RANGE[0])
        & (lat <= LATITUDE_RANGE[1])
        & (lon >= LONGITUDE_RANGE[0])
        & (lon <= LONGITUDE_RANGE[1])
    )


def eawnp_memberships(
    latitude: np.ndarray | pd.Series,
    longitude: np.ndarray | pd.Series,
) -> dict[str, np.ndarray]:
    """返回 expanded overall、原 CMS 子集和两个经度支持带。"""
    lat = np.asarray(latitude)
    lon = np.asarray(longitude)
    overall = eawnp_rectangle_mask(lat, lon)
    cms = region_memberships(lat, lon)
    return {
        "EAWNP": overall,
        "CMS": overall & cms["CMS"],
        "BYS": overall & cms["BYS"],
        "ECS": overall & cms["ECS"],
        "NSCS": overall & cms["NSCS"],
        "WEST_105_140": overall & (lon <= 140.0),
        "EAST_140_170": overall & (lon > 140.0),
    }


def _validate_lineage(
    eligible_ids: set[str],
    manifest: dict[str, Any],
) -> dict[str, list[str]]:
    splits = {
        name: sorted(
            eligible_ids
            & set(map(str, manifest["splits"][name]["original_ids"]))
        )
        for name in ("train", "val", "test")
    }
    split_sets = {name: set(values) for name, values in splits.items()}
    if (
        split_sets["train"] & split_sets["val"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["val"] & split_sets["test"]
    ):
        raise RuntimeError("expanded frozen-global lineage 存在 ID 交集。")
    covered = set().union(*split_sets.values())
    if covered != eligible_ids:
        raise RuntimeError(
            "frozen-global manifest 未覆盖 expanded IDs: "
            f"{sorted(eligible_ids - covered)[:10]}"
        )
    return splits


def _membership_counts(
    memberships: dict[str, np.ndarray],
) -> dict[str, int]:
    return {
        name: int(mask.sum()) for name, mask in memberships.items()
    }


def prepare_eawnp_dataset(
    *,
    mask_source_path: Path = MASK_SOURCE_PATH,
    circular_source_path: Path = CIRCULAR_SOURCE_PATH,
    global_split_manifest_path: Path = GLOBAL_SPLIT_MANIFEST_PATH,
    filtered_data_path: Path = FILTERED_DATA_PATH,
    diagnostics_path: Path = FILTERED_DIAGNOSTICS_PATH,
    artifact_dir: Path = ARTIFACT_DIR,
    min_regional_points: int = MIN_REGIONAL_POINTS,
    code_commit: str,
    expected_mask_sha256: str | None = EXPECTED_MASK_SOURCE_SHA256,
    expected_circular_sha256: str | None = EXPECTED_CIRCULAR_SOURCE_SHA256,
    expected_population: dict[str, int] | None = EXPECTED_POPULATION,
) -> dict[str, Any]:
    """按行筛选矩形、应用 ID 门槛并继承 frozen-global lineage。"""
    mask_source_path = Path(mask_source_path).resolve()
    circular_source_path = Path(circular_source_path).resolve()
    global_split_manifest_path = Path(
        global_split_manifest_path
    ).resolve()
    filtered_data_path = Path(filtered_data_path).resolve()
    diagnostics_path = Path(diagnostics_path).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    targets = (
        filtered_data_path,
        diagnostics_path,
        artifact_dir / "data_statistics.json",
        artifact_dir / "region_mask.json",
        artifact_dir / "split_manifest.json",
        artifact_dir / "region_row_index.npz",
    )
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            f"拒绝覆盖 expanded 数据 artifact: {existing}"
        )
    for path in (
        mask_source_path,
        circular_source_path,
        global_split_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    mask_sha256 = _sha256(mask_source_path)
    circular_sha256 = _sha256(circular_source_path)
    if (
        expected_mask_sha256 is not None
        and mask_sha256 != expected_mask_sha256
    ):
        raise RuntimeError("mask source SHA256 不匹配。")
    if (
        expected_circular_sha256 is not None
        and circular_sha256 != expected_circular_sha256
    ):
        raise RuntimeError("circular source SHA256 不匹配。")

    with mask_source_path.open("rb") as file:
        mask_trajectories = pickle.load(file)
    with circular_source_path.open("rb") as file:
        circular_trajectories = pickle.load(file)
    if len(mask_trajectories) != len(circular_trajectories):
        raise ValueError(
            "mask/circular 源片段数不一致: "
            f"{len(mask_trajectories)} vs {len(circular_trajectories)}"
        )

    raw_id_counts: Counter[str] = Counter()
    for index, frame in enumerate(mask_trajectories):
        missing = {
            "original_ID",
            "latitude",
            "longitude",
            "time",
        } - set(frame.columns)
        if missing:
            raise ValueError(f"mask 源片段 {index} 缺列: {sorted(missing)}")
        selected = eawnp_rectangle_mask(
            frame["latitude"],
            frame["longitude"],
        )
        count = int(selected.sum())
        if count:
            raw_id_counts[_canonical_original_id(frame)] += count
    eligible_ids = {
        original_id
        for original_id, count in raw_id_counts.items()
        if count >= min_regional_points
    }
    manifest = json.loads(
        global_split_manifest_path.read_text(encoding="utf-8")
    )
    splits = _validate_lineage(eligible_ids, manifest)
    population = {
        "total": len(eligible_ids),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "samples": sum(raw_id_counts[value] for value in eligible_ids),
    }
    if (
        expected_population is not None
        and population != expected_population
    ):
        raise RuntimeError(
            f"expanded population 漂移: {population} != "
            f"{expected_population}"
        )

    id_to_split = {
        original_id: split
        for split, values in splits.items()
        for original_id in values
    }
    selected_frames: list[pd.DataFrame] = []
    row_index_parts: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    membership_counts: Counter[str] = Counter()
    month_counts: Counter[int] = Counter()
    split_rows: Counter[str] = Counter()
    split_episodes: Counter[str] = Counter()
    split_month_counts = {
        name: Counter() for name in ("train", "val", "test")
    }
    split_memberships = {
        name: Counter() for name in ("train", "val", "test")
    }
    missing_required: Counter[str] = Counter()
    source_segments_with_selected_rows = 0
    wave_changed_rows = 0

    for trajectory_index, (mask_frame, circular_frame) in enumerate(
        zip(mask_trajectories, circular_trajectories)
    ):
        if len(mask_frame) != len(circular_frame):
            raise ValueError(
                f"片段 {trajectory_index} mask/circular 行数不一致。"
            )
        original_id = _canonical_original_id(mask_frame)
        if original_id not in eligible_ids:
            continue
        if _canonical_original_id(circular_frame) != original_id:
            raise ValueError(
                f"片段 {trajectory_index} original_ID 不一致。"
            )
        selected_indices = np.flatnonzero(
            eawnp_rectangle_mask(
                mask_frame["latitude"],
                mask_frame["longitude"],
            )
        )
        if not len(selected_indices):
            continue
        source_segments_with_selected_rows += 1
        for column in IDENTITY_COLUMNS:
            if column not in mask_frame or column not in circular_frame:
                raise ValueError(
                    f"片段 {trajectory_index} 缺对齐列 {column!r}。"
                )
            left = (
                mask_frame.iloc[selected_indices][column]
                .reset_index(drop=True)
            )
            right = (
                circular_frame.iloc[selected_indices][column]
                .reset_index(drop=True)
            )
            if not left.equals(right):
                raise ValueError(
                    f"片段 {trajectory_index} 的 {column!r} 未对齐。"
                )
        changed = (
            mask_frame.iloc[selected_indices][
                "era5_wave_dir_sin"
            ].to_numpy()
            != circular_frame.iloc[selected_indices][
                "era5_wave_dir_sin"
            ].to_numpy()
        ) | (
            mask_frame.iloc[selected_indices][
                "era5_wave_dir_cos"
            ].to_numpy()
            != circular_frame.iloc[selected_indices][
                "era5_wave_dir_cos"
            ].to_numpy()
        )
        wave_changed_rows += int(changed.sum())
        for column in MODEL_REQUIRED_COLUMNS:
            missing_required[column] += int(
                circular_frame.iloc[selected_indices][column].isna().sum()
            )

        selected_memberships = eawnp_memberships(
            mask_frame.iloc[selected_indices]["latitude"],
            mask_frame.iloc[selected_indices]["longitude"],
        )
        counts = _membership_counts(selected_memberships)
        membership_counts.update(counts)
        split_name = id_to_split[original_id]
        split_rows[split_name] += len(selected_indices)
        split_memberships[split_name].update(counts)
        selected_months = pd.DatetimeIndex(
            circular_frame.iloc[selected_indices]["time"]
        ).month
        month_counts.update(map(int, selected_months))
        split_month_counts[split_name].update(map(int, selected_months))

        runs = _contiguous_hourly_runs(
            selected_indices,
            mask_frame["time"].to_numpy(),
        )
        split_episodes[split_name] += len(runs)
        for run_indices in runs:
            episode_index = len(selected_frames)
            episode = (
                circular_frame.iloc[run_indices]
                .copy()
                .reset_index(drop=True)
            )
            selected_frames.append(episode)
            count = len(run_indices)
            row_index_parts["source_trajectory_index"].append(
                np.full(count, trajectory_index, dtype=np.int32)
            )
            row_index_parts["source_row_index"].append(
                run_indices.astype(np.int32, copy=False)
            )
            row_index_parts["episode_index"].append(
                np.full(count, episode_index, dtype=np.int32)
            )
            row_index_parts["original_id"].append(
                np.full(count, original_id, dtype=f"<U{len(original_id)}")
            )

    del mask_trajectories, circular_trajectories
    selected_rows = sum(map(len, selected_frames))
    if selected_rows != population["samples"]:
        raise RuntimeError(
            f"expanded 行数守恒失败: {selected_rows} != "
            f"{population['samples']}"
        )
    if any(missing_required.values()):
        raise RuntimeError(
            f"expanded 必需列存在缺测: {dict(missing_required)}"
        )

    filtered_data_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with filtered_data_path.open("wb") as file:
        pickle.dump(selected_frames, file, protocol=pickle.HIGHEST_PROTOCOL)
    row_index_path = artifact_dir / "region_row_index.npz"
    row_index = {
        name: np.concatenate(parts)
        for name, parts in row_index_parts.items()
    }
    if any(len(values) != selected_rows for values in row_index.values()):
        raise RuntimeError("expanded row index 长度异常。")
    np.savez_compressed(row_index_path, **row_index)

    split_manifest = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "strategy": "inherit_frozen_global_original_id_lineage",
        "random_seed": manifest.get("random_seed", 42),
        "source_manifest": _portable_path(global_split_manifest_path),
        "source_manifest_sha256": _sha256(global_split_manifest_path),
        "splits": {
            name: {
                "n_original_ids": len(splits[name]),
                "n_samples": split_rows[name],
                "n_episodes": split_episodes[name],
                "original_ids": splits[name],
            }
            for name in ("train", "val", "test")
        },
        "pairwise_original_id_intersections": {
            "train_val": 0,
            "train_test": 0,
            "val_test": 0,
        },
    }
    split_manifest_path = artifact_dir / "split_manifest.json"
    _write_json(split_manifest_path, split_manifest)

    statistics = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "minimum_regional_hourly_points_per_original_id": (
            min_regional_points
        ),
        "before_minimum_threshold": {
            "n_original_ids": len(raw_id_counts),
            "n_samples": int(sum(raw_id_counts.values())),
        },
        "excluded_by_minimum_threshold": {
            "n_original_ids": len(set(raw_id_counts) - eligible_ids),
            "original_ids": sorted(set(raw_id_counts) - eligible_ids),
        },
        "dataset": {
            "n_original_ids": len(eligible_ids),
            "n_samples": selected_rows,
            "n_source_segments": source_segments_with_selected_rows,
            "n_hourly_episodes": len(selected_frames),
        },
        "membership_counts": dict(membership_counts),
        "month_counts": {
            str(month): month_counts[month] for month in range(1, 13)
        },
        "split_statistics": {
            name: {
                "n_original_ids": len(splits[name]),
                "n_samples": split_rows[name],
                "n_episodes": split_episodes[name],
                "membership_counts": dict(split_memberships[name]),
                "month_counts": {
                    str(month): split_month_counts[name][month]
                    for month in range(1, 13)
                },
            }
            for name in ("train", "val", "test")
        },
        "data_quality": {
            "missing_required_values": dict(missing_required),
            "v1_v2_identity_columns_verified": IDENTITY_COLUMNS,
            "wave_direction_rows_changed_v1_to_circular_v2": (
                wave_changed_rows
            ),
            "wave_direction_rows_total": selected_rows,
        },
    }
    statistics_path = artifact_dir / "data_statistics.json"
    _write_json(statistics_path, statistics)

    region_mask = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "coordinate_system": "WGS84 latitude/longitude degrees",
        "longitude_convention": "-180_to_180",
        "boundary_policy": "inclusive",
        "latitude": list(LATITUDE_RANGE),
        "longitude": list(LONGITUDE_RANGE),
        "row_selection": (
            "Only rows inside the rectangle are retained; entering the "
            "rectangle does not admit outside rows from the same original_ID."
        ),
        "minimum_regional_hourly_points_per_original_id": (
            min_regional_points
        ),
        "count_threshold_override": {
            "source_branch": "wdf_cms_range_search_v1",
            "source_commit": (
                "2c7111d451411bb2df96b6e75e111d9ef2451d7a"
            ),
            "accepted_counts": {
                "total": 96,
                "train": 75,
                "val": 12,
                "test": 9,
            },
        },
        "mask_source": _portable_path(mask_source_path),
        "training_feature_source": _portable_path(circular_source_path),
        "filtered_training_data": _portable_path(filtered_data_path),
        "memberships": {
            "EAWNP": "selected rectangle overall",
            "CMS": "original three-rectangle CMS union subset",
            "BYS": "original Bohai + Yellow Sea rectangle subset",
            "ECS": "original East China Sea rectangle subset",
            "NSCS": "original Northern South China Sea rectangle subset",
            "WEST_105_140": "selected rows with longitude <= 140 E",
            "EAST_140_170": "selected rows with longitude > 140 E",
        },
    }
    region_mask_path = artifact_dir / "region_mask.json"
    _write_json(region_mask_path, region_mask)

    diagnostics = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "processing": {
            "mode": (
                "rectangle_point_filter_from_v1_mask_with_"
                "circular_v2_features"
            ),
            "algorithm_version": (
                "eastasia_wnp_105_170_min24_contiguous_episode_v1"
            ),
            "code_git_commit": code_commit,
            "selected_original_id_count": len(eligible_ids),
            "selected_row_count": selected_rows,
            "selected_episode_count": len(selected_frames),
        },
        "source_integrity": {
            "mask_source_path": _portable_path(mask_source_path),
            "mask_source_size_bytes": mask_source_path.stat().st_size,
            "mask_source_sha256": mask_sha256,
            "circular_source_path": _portable_path(circular_source_path),
            "circular_source_size_bytes": (
                circular_source_path.stat().st_size
            ),
            "circular_source_sha256": circular_sha256,
            "global_split_manifest_sha256": _sha256(
                global_split_manifest_path
            ),
        },
        "output_integrity": {
            "filtered_data_path": _portable_path(filtered_data_path),
            "filtered_data_size_bytes": filtered_data_path.stat().st_size,
            "filtered_data_sha256": _sha256(filtered_data_path),
            "row_index_path": _portable_path(row_index_path),
            "row_index_size_bytes": row_index_path.stat().st_size,
            "row_index_sha256": _sha256(row_index_path),
        },
    }
    _write_json(diagnostics_path, diagnostics)
    return {
        "filtered_data_path": filtered_data_path,
        "diagnostics_path": diagnostics_path,
        "statistics_path": statistics_path,
        "region_mask_path": region_mask_path,
        "split_manifest_path": split_manifest_path,
        "row_index_path": row_index_path,
        "population": population,
        "splits": splits,
    }
