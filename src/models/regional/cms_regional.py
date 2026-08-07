"""China Marginal Seas (CMS) regional 数据准备与评价工具。"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from ..data_loader import PROJECT_ROOT, TARGET_COLS
from ..evaluation import regression_metrics


MODEL_VERSION = "wdf_cms_orig_core6_v1"
RANDOM_SEED = 42
MIN_REGIONAL_POINTS = 24
MIN_INHERITED_EVAL_IDS = 3

MASK_SOURCE_PATH = (
    PROJECT_ROOT / "processed_data" / "trajectories_with_all_features.pkl"
)
CIRCULAR_SOURCE_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "trajectories_with_all_features_circular_mwd_v2.pkl"
)
FILTERED_DATA_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "trajectories_cms_circular_mwd_v2.pkl"
)
FILTERED_DIAGNOSTICS_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "trajectories_cms_circular_mwd_v2_diagnostics.json"
)
GLOBAL_SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "trained_models"
    / "ablation_circular_mwd_v2_final"
    / "core_6"
    / "split_manifest.json"
)
GLOBAL_BASELINE_METRICS_PATH = (
    PROJECT_ROOT
    / "trained_models"
    / "ablation_circular_mwd_v2_final"
    / "core_6"
    / "linear_baseline_metrics.json"
)
GLOBAL_ONNX_PATH = (
    PROJECT_ROOT
    / "deployment"
    / "releases"
    / "wdf_core6_circular_mwd_v2"
    / "wdf_drifter.onnx"
)

CORE_FEATURES = [
    "era5_u10",
    "era5_v10",
    "era5_swh",
    "era5_mwp",
    "era5_wave_dir_sin",
    "era5_wave_dir_cos",
]
MODEL_REQUIRED_COLUMNS = [
    *CORE_FEATURES,
    "ve",
    "vn",
    "cfsv2_u",
    "cfsv2_v",
]
IDENTITY_COLUMNS = [
    "ID",
    "time",
    "latitude",
    "longitude",
    "ve",
    "vn",
    "original_ID",
    "segment_index",
    "cfsv2_u",
    "cfsv2_v",
    "era5_u10",
    "era5_v10",
    "era5_swh",
    "era5_mwp",
]
REGIONS = {
    "BYS": {
        "name": "Bohai + Yellow Sea",
        "latitude": [31.0, 41.0],
        "longitude": [117.0, 127.0],
    },
    "ECS": {
        "name": "East China Sea",
        "latitude": [23.0, 33.0],
        "longitude": [117.0, 131.0],
    },
    "NSCS": {
        "name": "Northern South China Sea",
        "latitude": [15.0, 23.0],
        "longitude": [105.0, 122.0],
    },
}
WAVE_DIRECTION_CONVENTION = {
    "source": "ERA5 mean_wave_direction",
    "meaning": "coming-from",
    "zero_direction": "north",
    "positive_rotation": "clockwise",
    "add_180_degrees": False,
    "sin_definition": "sin(deg2rad(mwd))",
    "cos_definition": "cos(deg2rad(mwd))",
}


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _canonical_original_id(frame: pd.DataFrame) -> str:
    ids = frame["original_ID"].dropna().astype(str).str.strip().unique()
    if len(ids) != 1 or not ids[0]:
        raise ValueError(
            "每条源子轨迹必须只含一个非空 original_ID，"
            f"实际为 {ids[:5].tolist()}。"
        )
    return str(ids[0])


def region_memberships(
    latitude: np.ndarray | pd.Series,
    longitude: np.ndarray | pd.Series,
) -> dict[str, np.ndarray]:
    """按用户给定的闭区间矩形返回三个区域及 CMS 并集布尔掩码。"""
    lat = np.asarray(latitude)
    lon = np.asarray(longitude)
    if lat.shape != lon.shape:
        raise ValueError(f"经纬度 shape 不一致: {lat.shape} vs {lon.shape}")

    memberships: dict[str, np.ndarray] = {}
    for key, definition in REGIONS.items():
        lat_min, lat_max = definition["latitude"]
        lon_min, lon_max = definition["longitude"]
        memberships[key] = (
            (lat >= lat_min)
            & (lat <= lat_max)
            & (lon >= lon_min)
            & (lon <= lon_max)
        )
    memberships["CMS"] = (
        memberships["BYS"] | memberships["ECS"] | memberships["NSCS"]
    )
    return memberships


def _contiguous_hourly_runs(
    selected_indices: np.ndarray,
    time_values: np.ndarray,
) -> list[np.ndarray]:
    """把一次源子轨迹中的入区行切成连续 hourly episode。"""
    if len(selected_indices) == 0:
        return []
    selected_times = np.asarray(time_values)[selected_indices]
    selected_hours = selected_times.astype("datetime64[h]").astype(np.int64)
    starts = np.flatnonzero(
        np.r_[
            True,
            (np.diff(selected_indices) != 1)
            | (np.diff(selected_hours) != 1),
        ]
    )
    stops = np.r_[starts[1:], len(selected_indices)]
    return [selected_indices[start:stop] for start, stop in zip(starts, stops)]


def grouped_id_split(
    original_ids: list[str] | np.ndarray,
    random_seed: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    """用两阶段 GroupShuffleSplit 生成可复现的 70/15/15 ID 切分。"""
    ids = np.asarray(sorted(set(map(str, original_ids))), dtype=object)
    if len(ids) < 7:
        raise ValueError("至少需要 7 个 original_ID 才能执行分组切分。")

    test_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=random_seed,
    )
    train_val_index, test_index = next(
        test_splitter.split(np.zeros((len(ids), 1)), groups=ids)
    )
    train_val_ids = ids[train_val_index]
    test_ids = ids[test_index]

    val_fraction_within_train_val = 0.15 / (0.70 + 0.15)
    val_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction_within_train_val,
        random_state=random_seed,
    )
    train_index, val_index = next(
        val_splitter.split(
            np.zeros((len(train_val_ids), 1)),
            groups=train_val_ids,
        )
    )
    result = {
        "train": sorted(map(str, train_val_ids[train_index])),
        "val": sorted(map(str, train_val_ids[val_index])),
        "test": sorted(map(str, test_ids)),
    }
    sets = {name: set(values) for name, values in result.items()}
    if (
        sets["train"] & sets["val"]
        or sets["train"] & sets["test"]
        or sets["val"] & sets["test"]
    ):
        raise RuntimeError("GroupShuffleSplit 产生了 original_ID 交集。")
    if set.union(*sets.values()) != set(map(str, ids)):
        raise RuntimeError("GroupShuffleSplit 未完整覆盖 regional original_ID。")
    return result


def choose_regional_id_splits(
    regional_ids: list[str] | set[str],
    global_manifest: dict[str, Any],
    *,
    random_seed: int = RANDOM_SEED,
    min_inherited_eval_ids: int = MIN_INHERITED_EVAL_IDS,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """优先继承 global manifest；评价集 ID 太少时改用 GroupShuffleSplit。"""
    ids = set(map(str, regional_ids))
    inherited = {
        name: sorted(
            ids
            & set(map(str, global_manifest["splits"][name]["original_ids"]))
        )
        for name in ("train", "val", "test")
    }
    inherited_sets = {name: set(values) for name, values in inherited.items()}
    inherited_union = set.union(*inherited_sets.values())
    inherited_counts = {
        name: len(values) for name, values in inherited.items()
    }
    rejection_reasons: list[str] = []
    if inherited_union != ids:
        rejection_reasons.append(
            f"global manifest 未覆盖 {len(ids - inherited_union)} 个 regional ID"
        )
    if inherited_counts["train"] < 7:
        rejection_reasons.append(
            f"继承 train 仅 {inherited_counts['train']} 个 ID（至少需 7）"
        )
    for name in ("val", "test"):
        if inherited_counts[name] < min_inherited_eval_ids:
            rejection_reasons.append(
                f"继承 {name} 仅 {inherited_counts[name]} 个 ID"
                f"（最低 {min_inherited_eval_ids}）"
            )

    if rejection_reasons:
        selected = grouped_id_split(ids, random_seed=random_seed)
        strategy = "regenerated_group_shuffle_split"
    else:
        selected = inherited
        strategy = "inherited_frozen_global_split"

    provenance = {
        "strategy": strategy,
        "random_seed": random_seed,
        "target_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "global_manifest": _portable_path(GLOBAL_SPLIT_MANIFEST_PATH),
        "inherited_candidate_counts": inherited_counts,
        "minimum_inherited_eval_ids": min_inherited_eval_ids,
        "inheritance_rejection_reasons": rejection_reasons,
        "selected_counts": {
            name: len(values) for name, values in selected.items()
        },
        "implementation": "sklearn.model_selection.GroupShuffleSplit",
    }
    return selected, provenance


def _membership_counts(
    memberships: dict[str, np.ndarray],
) -> dict[str, int]:
    return {
        "CMS_union": int(memberships["CMS"].sum()),
        "BYS": int(memberships["BYS"].sum()),
        "ECS": int(memberships["ECS"].sum()),
        "NSCS": int(memberships["NSCS"].sum()),
        "BYS_ECS_overlap": int(
            (memberships["BYS"] & memberships["ECS"]).sum()
        ),
        "BYS_NSCS_overlap": int(
            (memberships["BYS"] & memberships["NSCS"]).sum()
        ),
        "ECS_NSCS_overlap": int(
            (memberships["ECS"] & memberships["NSCS"]).sum()
        ),
        "all_three_overlap": int(
            (
                memberships["BYS"]
                & memberships["ECS"]
                & memberships["NSCS"]
            ).sum()
        ),
    }


def _add_counts(target: Counter[str], values: dict[str, int]) -> None:
    for key, value in values.items():
        target[key] += int(value)


def _split_statistics(
    trajectories: list[pd.DataFrame],
    id_splits: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    lookup = {
        original_id: split_name
        for split_name, ids in id_splits.items()
        for original_id in ids
    }
    counters: dict[str, Counter[str]] = {
        name: Counter() for name in ("train", "val", "test")
    }
    month_counts: dict[str, Counter[int]] = {
        name: Counter() for name in ("train", "val", "test")
    }
    for frame in trajectories:
        original_id = _canonical_original_id(frame)
        split_name = lookup[original_id]
        memberships = region_memberships(
            frame["latitude"],
            frame["longitude"],
        )
        counters[split_name]["n_episodes"] += 1
        _add_counts(counters[split_name], _membership_counts(memberships))
        month_counts[split_name].update(
            map(int, pd.DatetimeIndex(frame["time"]).month)
        )

    return {
        name: {
            "n_original_ids": len(id_splits[name]),
            "n_episodes": counters[name]["n_episodes"],
            "n_samples": counters[name]["CMS_union"],
            "region_membership_counts": {
                key: counters[name][key]
                for key in (
                    "BYS",
                    "ECS",
                    "NSCS",
                    "BYS_ECS_overlap",
                    "BYS_NSCS_overlap",
                    "ECS_NSCS_overlap",
                    "all_three_overlap",
                )
            },
            "month_counts": {
                str(month): month_counts[name][month]
                for month in range(1, 13)
            },
        }
        for name in ("train", "val", "test")
    }


def prepare_cms_dataset(
    *,
    mask_source_path: Path = MASK_SOURCE_PATH,
    circular_source_path: Path = CIRCULAR_SOURCE_PATH,
    filtered_data_path: Path = FILTERED_DATA_PATH,
    diagnostics_path: Path = FILTERED_DIAGNOSTICS_PATH,
    artifact_dir: Path,
    global_split_manifest_path: Path = GLOBAL_SPLIT_MANIFEST_PATH,
    min_regional_points: int = MIN_REGIONAL_POINTS,
    min_inherited_eval_ids: int = MIN_INHERITED_EVAL_IDS,
    random_seed: int = RANDOM_SEED,
    code_commit: str = "unknown",
) -> dict[str, Any]:
    """生成 CMS 逐行筛选数据、region mask、统计和 split 方案。"""
    mask_source_path = Path(mask_source_path).resolve()
    circular_source_path = Path(circular_source_path).resolve()
    filtered_data_path = Path(filtered_data_path).resolve()
    diagnostics_path = Path(diagnostics_path).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    global_split_manifest_path = Path(global_split_manifest_path).resolve()

    for required_path in (
        mask_source_path,
        circular_source_path,
        global_split_manifest_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    filtered_data_path.parent.mkdir(parents=True, exist_ok=True)

    with mask_source_path.open("rb") as file:
        mask_trajectories = pickle.load(file)
    with circular_source_path.open("rb") as file:
        circular_trajectories = pickle.load(file)
    if len(mask_trajectories) != len(circular_trajectories):
        raise ValueError(
            "v1 mask source 与 circular-v2 子轨迹数量不一致: "
            f"{len(mask_trajectories)} vs {len(circular_trajectories)}"
        )

    raw_id_counts: Counter[str] = Counter()
    before_counts: Counter[str] = Counter()
    for trajectory_index, frame in enumerate(mask_trajectories):
        missing = {"latitude", "longitude", "time", "original_ID"} - set(
            frame.columns
        )
        if missing:
            raise ValueError(
                f"第 {trajectory_index} 条 mask 源轨迹缺列: {sorted(missing)}"
            )
        memberships = region_memberships(
            frame["latitude"],
            frame["longitude"],
        )
        _add_counts(before_counts, _membership_counts(memberships))
        n_cms = int(memberships["CMS"].sum())
        if n_cms:
            raw_id_counts[_canonical_original_id(frame)] += n_cms

    eligible_ids = {
        original_id
        for original_id, count in raw_id_counts.items()
        if count >= min_regional_points
    }
    if len(eligible_ids) < 7:
        raise ValueError(
            f"满足 {min_regional_points} 点门槛的 CMS original_ID "
            f"仅 {len(eligible_ids)} 个，无法执行分组切分。"
        )

    selected_trajectories: list[pd.DataFrame] = []
    row_index_parts: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    after_counts: Counter[str] = Counter()
    month_counts: Counter[int] = Counter()
    source_segment_count = 0
    wave_changed_rows = 0
    missing_required: Counter[str] = Counter()

    for trajectory_index, (mask_frame, circular_frame) in enumerate(
        zip(mask_trajectories, circular_trajectories)
    ):
        if len(mask_frame) != len(circular_frame):
            raise ValueError(
                f"第 {trajectory_index} 条 v1/v2 行数不一致: "
                f"{len(mask_frame)} vs {len(circular_frame)}"
            )
        original_id = _canonical_original_id(mask_frame)
        if original_id not in eligible_ids:
            continue
        if _canonical_original_id(circular_frame) != original_id:
            raise ValueError(f"第 {trajectory_index} 条 v1/v2 original_ID 不一致。")

        memberships = region_memberships(
            mask_frame["latitude"],
            mask_frame["longitude"],
        )
        selected_indices = np.flatnonzero(memberships["CMS"])
        if not len(selected_indices):
            continue
        source_segment_count += 1

        for column in IDENTITY_COLUMNS:
            if column not in mask_frame or column not in circular_frame:
                raise ValueError(
                    f"第 {trajectory_index} 条 v1/v2 缺少对齐列 {column!r}。"
                )
            left = mask_frame.iloc[selected_indices][column].reset_index(drop=True)
            right = (
                circular_frame.iloc[selected_indices][column]
                .reset_index(drop=True)
            )
            if not left.equals(right):
                raise ValueError(
                    f"第 {trajectory_index} 条 CMS 行在 {column!r} 上未对齐。"
                )

        wave_sin_changed = (
            mask_frame.iloc[selected_indices]["era5_wave_dir_sin"].to_numpy()
            != circular_frame.iloc[selected_indices][
                "era5_wave_dir_sin"
            ].to_numpy()
        )
        wave_cos_changed = (
            mask_frame.iloc[selected_indices]["era5_wave_dir_cos"].to_numpy()
            != circular_frame.iloc[selected_indices][
                "era5_wave_dir_cos"
            ].to_numpy()
        )
        wave_changed_rows += int(
            np.count_nonzero(wave_sin_changed | wave_cos_changed)
        )
        for column in MODEL_REQUIRED_COLUMNS:
            missing_required[column] += int(
                circular_frame.iloc[selected_indices][column].isna().sum()
            )

        selected_memberships = {
            name: mask[selected_indices]
            for name, mask in memberships.items()
        }
        _add_counts(after_counts, _membership_counts(selected_memberships))
        month_counts.update(
            map(
                int,
                pd.DatetimeIndex(
                    circular_frame.iloc[selected_indices]["time"]
                ).month,
            )
        )

        runs = _contiguous_hourly_runs(
            selected_indices,
            mask_frame["time"].to_numpy(),
        )
        for run_indices in runs:
            episode_index = len(selected_trajectories)
            episode = (
                circular_frame.iloc[run_indices]
                .copy()
                .reset_index(drop=True)
            )
            selected_trajectories.append(episode)
            n_rows = len(run_indices)
            row_index_parts["source_trajectory_index"].append(
                np.full(n_rows, trajectory_index, dtype=np.int32)
            )
            row_index_parts["source_row_index"].append(
                run_indices.astype(np.int32, copy=False)
            )
            row_index_parts["episode_index"].append(
                np.full(n_rows, episode_index, dtype=np.int32)
            )
            row_index_parts["original_id"].append(
                np.full(n_rows, original_id, dtype=f"<U{max(1, len(original_id))}")
            )
            for name in ("BYS", "ECS", "NSCS"):
                row_index_parts[f"in_{name.lower()}"].append(
                    memberships[name][run_indices].astype(np.bool_)
                )

    del mask_trajectories, circular_trajectories

    if not selected_trajectories:
        raise ValueError("CMS 筛选后没有有效 episode。")
    selected_rows = sum(map(len, selected_trajectories))
    expected_rows = sum(
        count for original_id, count in raw_id_counts.items()
        if original_id in eligible_ids
    )
    if selected_rows != expected_rows:
        raise RuntimeError(
            f"CMS 行数守恒失败: {selected_rows} != {expected_rows}"
        )

    global_manifest = json.loads(
        global_split_manifest_path.read_text(encoding="utf-8")
    )
    id_splits, split_provenance = choose_regional_id_splits(
        eligible_ids,
        global_manifest,
        random_seed=random_seed,
        min_inherited_eval_ids=min_inherited_eval_ids,
    )
    split_provenance["global_manifest"] = _portable_path(
        global_split_manifest_path
    )
    split_stats = _split_statistics(selected_trajectories, id_splits)

    with filtered_data_path.open("wb") as file:
        pickle.dump(
            selected_trajectories,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    row_index_path = artifact_dir / "cms_region_row_index.npz"
    row_index_arrays = {
        name: np.concatenate(parts)
        for name, parts in row_index_parts.items()
    }
    if any(len(values) != selected_rows for values in row_index_arrays.values()):
        raise RuntimeError("region row index 长度与筛选数据行数不一致。")
    np.savez_compressed(row_index_path, **row_index_arrays)

    excluded_ids = sorted(set(raw_id_counts) - eligible_ids)
    statistics = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "minimum_regional_hourly_points_per_original_id": min_regional_points,
        "before_minimum_threshold": {
            "n_original_ids": len(raw_id_counts),
            "n_samples": before_counts["CMS_union"],
        },
        "excluded_by_minimum_threshold": {
            "n_original_ids": len(excluded_ids),
            "n_samples": sum(raw_id_counts[value] for value in excluded_ids),
            "original_ids": excluded_ids,
        },
        "cms_dataset": {
            "n_original_ids": len(eligible_ids),
            "n_samples": selected_rows,
            "n_source_segments": source_segment_count,
            "n_hourly_episodes": len(selected_trajectories),
        },
        "region_membership_counts": {
            key: after_counts[key]
            for key in (
                "BYS",
                "ECS",
                "NSCS",
                "BYS_ECS_overlap",
                "BYS_NSCS_overlap",
                "ECS_NSCS_overlap",
                "all_three_overlap",
            )
        },
        "month_counts": {
            str(month): month_counts[month] for month in range(1, 13)
        },
        "split": {
            "provenance": split_provenance,
            "statistics": split_stats,
            "pairwise_original_id_intersections": {
                "train_val": len(
                    set(id_splits["train"]) & set(id_splits["val"])
                ),
                "train_test": len(
                    set(id_splits["train"]) & set(id_splits["test"])
                ),
                "val_test": len(
                    set(id_splits["val"]) & set(id_splits["test"])
                ),
            },
        },
        "data_quality": {
            "missing_required_values": dict(missing_required),
            "v1_v2_identity_columns_verified": IDENTITY_COLUMNS,
            "wave_direction_rows_changed_v1_to_circular_v2": wave_changed_rows,
            "wave_direction_rows_total": selected_rows,
        },
    }
    statistics_path = artifact_dir / "cms_data_statistics.json"
    _write_json(statistics_path, statistics)

    region_mask = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "coordinate_system": "WGS84 latitude/longitude degrees",
        "longitude_convention": "-180_to_180",
        "boundary_policy": "inclusive",
        "regions": REGIONS,
        "cms_expression": "BYS OR ECS OR NSCS",
        "row_selection": (
            "Only rows inside the CMS union are retained; entering the region "
            "does not admit out-of-region rows from the same original_ID."
        ),
        "subset_policy": (
            "BYS/ECS/NSCS membership is evaluated independently from the "
            "stated rectangles. CMS union rows are never duplicated."
        ),
        "minimum_regional_hourly_points_per_original_id": min_regional_points,
        "complete_72_hour_residence_required": False,
        "mask_source": _portable_path(mask_source_path),
        "training_feature_source": _portable_path(circular_source_path),
        "filtered_training_data": _portable_path(filtered_data_path),
        "row_index_file": _portable_path(row_index_path),
    }
    region_mask_path = artifact_dir / "cms_region_mask.json"
    _write_json(region_mask_path, region_mask)

    diagnostics = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "direction_convention": WAVE_DIRECTION_CONVENTION,
        "processing": {
            "mode": "cms_point_filter_from_v1_mask_with_circular_v2_features",
            "algorithm_version": (
                "cms_union_point_filter_min24_contiguous_episode_v1"
            ),
            "code_git_commit": code_commit,
            "source_file": str(mask_source_path),
            "feature_source_file": str(circular_source_path),
            "output_file": str(filtered_data_path),
            "selected_trajectory_count": len(selected_trajectories),
            "selected_original_id_count": len(eligible_ids),
            "selected_row_count": selected_rows,
        },
        "region_mask": region_mask,
        "split_provenance": split_provenance,
        "source_integrity": {
            "mask_source_size_bytes": mask_source_path.stat().st_size,
            "mask_source_sha256": _sha256(mask_source_path),
            "circular_source_size_bytes": circular_source_path.stat().st_size,
            "circular_source_sha256": _sha256(circular_source_path),
            "global_split_manifest_sha256": _sha256(
                global_split_manifest_path
            ),
        },
        "file_integrity": {
            "output_size_bytes": filtered_data_path.stat().st_size,
            "output_sha256": _sha256(filtered_data_path),
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
        "row_index_path": row_index_path,
        "id_splits": id_splits,
        "split_provenance": split_provenance,
        "statistics": statistics,
    }


def write_regional_linear_analysis(
    splits: dict[str, Any],
    baseline_result: dict[str, Any],
    output_path: Path,
    global_baseline_path: Path = GLOBAL_BASELINE_METRICS_PATH,
) -> dict[str, Any]:
    """保存 A 矩阵、沿风/横风系数和 residual 分布。"""
    matrix = np.asarray(baseline_result["coef_matrix"], dtype=np.float64)
    intercept = np.asarray(baseline_result["intercepts"], dtype=np.float64)
    effective_wdf = float(np.trace(matrix) / 2.0)
    # A ≈ alpha I + beta J, J=[[0,-1],[1,0]]；beta>0 为逆时针横风。
    cross_wind = float((matrix[1, 0] - matrix[0, 1]) / 2.0)

    residual_summary: dict[str, Any] = {}
    for name in ("train", "val", "test"):
        values = np.asarray(splits[f"y_{name}"], dtype=np.float64)
        residual_summary[name] = {
            "n_samples": len(values),
            "mean": {
                "residual_u": float(values[:, 0].mean()),
                "residual_v": float(values[:, 1].mean()),
            },
            "std_population": {
                "residual_u": float(values[:, 0].std(ddof=0)),
                "residual_v": float(values[:, 1].std(ddof=0)),
            },
        }
    all_values = np.concatenate(
        [splits[f"y_{name}"] for name in ("train", "val", "test")]
    ).astype(np.float64)
    residual_summary["cms_overall"] = {
        "n_samples": len(all_values),
        "mean": {
            "residual_u": float(all_values[:, 0].mean()),
            "residual_v": float(all_values[:, 1].mean()),
        },
        "std_population": {
            "residual_u": float(all_values[:, 0].std(ddof=0)),
            "residual_v": float(all_values[:, 1].std(ddof=0)),
        },
    }

    global_reference = None
    if global_baseline_path.is_file():
        global_metrics = json.loads(
            global_baseline_path.read_text(encoding="utf-8")
        )
        global_matrix = np.asarray(
            global_metrics["coef_matrix"],
            dtype=np.float64,
        )
        global_reference = {
            "path": _portable_path(global_baseline_path),
            "A_matrix": global_matrix.tolist(),
            "intercept": global_metrics["intercepts"],
            "effective_wdf": float(np.trace(global_matrix) / 2.0),
            "cross_wind_coefficient": float(
                (global_matrix[1, 0] - global_matrix[0, 1]) / 2.0
            ),
        }

    analysis = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "fit_scope": "CMS training split",
        "equation": "residual = A @ [era5_u10, era5_v10] + intercept",
        "target_definition": {
            "residual_u": "ve - cfsv2_u",
            "residual_v": "vn - cfsv2_v",
        },
        "A_matrix": matrix.tolist(),
        "intercept": intercept.tolist(),
        "effective_wdf": effective_wdf,
        "effective_wdf_definition": "trace(A) / 2",
        "cross_wind_coefficient": cross_wind,
        "cross_wind_definition": "(A[1,0] - A[0,1]) / 2",
        "cross_wind_sign_convention": (
            "positive rotates the wind vector counter-clockwise"
        ),
        "symmetric_off_diagonal_mean": float(
            (matrix[0, 1] + matrix[1, 0]) / 2.0
        ),
        "residual_summary": residual_summary,
        "validation_metrics": baseline_result["validation"],
        "test_metrics": baseline_result["test"],
        "frozen_global_reference": global_reference,
    }
    _write_json(Path(output_path), analysis)
    return analysis


def _torch_predict(
    model: torch.nn.Module,
    values: np.ndarray,
    batch_size: int = 8192,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(
                device=device,
                dtype=torch.float32,
            )
            predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions)


def _finite_or_none(value: float) -> float | None:
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _subset_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(
        y_true,
        dtype=np.float64,
    )
    if len(y_true) == 0:
        return {
            "status": "no_samples",
            "r2_u": None,
            "r2_v": None,
            "r2_joint": None,
            "rmse": None,
            "bias_u": None,
            "bias_v": None,
        }
    metrics: dict[str, Any]
    if len(y_true) >= 2:
        metrics = {
            key: _finite_or_none(value)
            for key, value in regression_metrics(y_true, y_pred).items()
        }
    else:
        metrics = {
            "r2_u": None,
            "r2_v": None,
            "r2_joint": None,
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
        }
    metrics.update(
        {
            "status": "ok",
            "bias_u": float(error[:, 0].mean()),
            "bias_v": float(error[:, 1].mean()),
            "bias_vector_magnitude": float(
                np.linalg.norm(error.mean(axis=0))
            ),
        }
    )
    return metrics


def evaluate_cms_models(
    *,
    filtered_data_path: Path,
    id_splits: dict[str, list[str]],
    regional_model: torch.nn.Module,
    regional_scaler: Any,
    output_path: Path,
    global_onnx_path: Path = GLOBAL_ONNX_PATH,
) -> dict[str, Any]:
    """在同一 CMS test 行比较 regional MLP 与冻结 global MLP。"""
    filtered_data_path = Path(filtered_data_path).resolve()
    global_onnx_path = Path(global_onnx_path).resolve()
    with filtered_data_path.open("rb") as file:
        trajectories = pickle.load(file)

    test_ids = set(id_splits["test"])
    test_frames = [
        frame
        for frame in trajectories
        if _canonical_original_id(frame) in test_ids
    ]
    if not test_frames:
        raise ValueError("CMS test split 没有数据。")
    test = pd.concat(test_frames, ignore_index=True)
    required = [*MODEL_REQUIRED_COLUMNS, "latitude", "longitude", "original_ID"]
    missing_columns = set(required) - set(test.columns)
    if missing_columns:
        raise ValueError(f"CMS test 数据缺列: {sorted(missing_columns)}")
    clean = test.dropna(subset=required).copy()
    if len(clean) != len(test):
        raise ValueError(
            f"CMS test 中有 {len(test) - len(clean)} 行必需值缺失。"
        )

    x_raw = clean[CORE_FEATURES].to_numpy(dtype=np.float32, copy=True)
    y_true = np.column_stack(
        [
            clean["ve"].to_numpy(dtype=np.float32)
            - clean["cfsv2_u"].to_numpy(dtype=np.float32),
            clean["vn"].to_numpy(dtype=np.float32)
            - clean["cfsv2_v"].to_numpy(dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    x_scaled = regional_scaler.transform(x_raw).astype(
        np.float32,
        copy=False,
    )
    regional_prediction = _torch_predict(regional_model, x_scaled)

    session = ort.InferenceSession(
        str(global_onnx_path),
        providers=["CPUExecutionProvider"],
    )
    global_prediction = session.run(
        ["output"],
        {"input": x_raw},
    )[0]
    memberships = region_memberships(clean["latitude"], clean["longitude"])
    subset_masks = {
        "CMS_overall": memberships["CMS"],
        "Bohai_Yellow_Sea": memberships["BYS"],
        "East_China_Sea": memberships["ECS"],
        "Northern_South_China_Sea": memberships["NSCS"],
    }

    subsets: dict[str, Any] = {}
    original_ids = clean["original_ID"].astype(str).to_numpy()
    for name, mask in subset_masks.items():
        regional_metrics = _subset_metrics(
            y_true[mask],
            regional_prediction[mask],
        )
        global_metrics = _subset_metrics(
            y_true[mask],
            global_prediction[mask],
        )
        delta = {
            "r2_joint": (
                None
                if regional_metrics["r2_joint"] is None
                or global_metrics["r2_joint"] is None
                else regional_metrics["r2_joint"]
                - global_metrics["r2_joint"]
            ),
            "rmse": (
                None
                if regional_metrics["rmse"] is None
                or global_metrics["rmse"] is None
                else regional_metrics["rmse"] - global_metrics["rmse"]
            ),
        }
        subsets[name] = {
            "n_samples": int(mask.sum()),
            "n_original_ids": int(len(set(original_ids[mask]))),
            "regional_mlp": regional_metrics,
            "frozen_global_mlp": global_metrics,
            "regional_minus_global": delta,
        }

    report = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "evaluation_split": "test",
        "target_definition": {
            "residual_u": "ve - cfsv2_u",
            "residual_v": "vn - cfsv2_v",
        },
        "bias_definition": "mean(predicted residual - observed residual)",
        "subset_policy": (
            "Each stated rectangle is evaluated independently; CMS overall "
            "uses the deduplicated union."
        ),
        "regional_model": {
            "feature_columns": CORE_FEATURES,
            "scaler": "regional train-only StandardScaler",
        },
        "frozen_global_model": {
            "onnx_path": _portable_path(global_onnx_path),
            "onnx_sha256": _sha256(global_onnx_path),
        },
        "subsets": subsets,
    }
    _write_json(Path(output_path), report)
    return report
