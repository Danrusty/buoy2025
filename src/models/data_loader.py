"""
数据加载、物理浮标 ID 分组切分与特征标准化。

关键约束：
  1. 使用 original_ID 切分，属于同一物理浮标的全部子轨迹只能进入同一集合。
  2. StandardScaler 只在训练集上拟合。
  3. 保存 split manifest 和 scaler，保证训练、评估与部署可追溯。
"""

from __future__ import annotations

import gc
import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "processed_data" / "trajectories_with_all_features.pkl"
DEFAULT_RUN_NAME = "original_id_split"
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"

FEATURE_COLS = [
    "era5_u10",
    "era5_v10",
    "era5_wind_speed",
    "era5_wind_dir_sin",
    "era5_wind_dir_cos",
    "era5_swh",
    "era5_mwp",
    "era5_wave_dir_sin",
    "era5_wave_dir_cos",
]
WIND_COLS = ["era5_u10", "era5_v10"]
TARGET_COLS = ["residual_u", "residual_v"]
CURRENT_COLS = ["cfsv2_u", "cfsv2_v"]
OBS_COLS = ["ve", "vn"]
GROUP_COL = "original_ID"

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

logger = logging.getLogger(__name__)


def _portable_path(path: Path) -> str:
    """仓库内路径写为相对路径，外部测试路径保留绝对路径。"""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _mem_mb() -> float:
    """返回当前进程 RSS 内存（MB）。"""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024**2
    except ImportError:
        return float("nan")


def _canonical_original_id(df: pd.DataFrame, trajectory_index: int) -> str:
    """提取单条子轨迹唯一的物理浮标 ID，并对异常输入立即报错。"""
    if GROUP_COL not in df.columns:
        raise ValueError(
            f"第 {trajectory_index} 条子轨迹缺少必需列 {GROUP_COL!r}，"
            "不能执行防泄漏切分。"
        )

    ids = df[GROUP_COL].dropna().astype(str).str.strip().unique()
    if len(ids) != 1 or not ids[0]:
        raise ValueError(
            f"第 {trajectory_index} 条子轨迹包含 {len(ids)} 个有效 "
            f"{GROUP_COL}: {ids[:5].tolist()}"
        )
    return str(ids[0])


def split_original_ids(
    original_ids: list[str] | np.ndarray,
    random_seed: int = RANDOM_SEED,
) -> dict[str, list[str]]:
    """
    将唯一物理浮标 ID 可复现地切分为 70%/15%/15%。

    切分前排序，确保结果不依赖 Pickle 内部子轨迹的排列顺序。
    """
    unique_ids = np.asarray(sorted(set(map(str, original_ids))), dtype=object)
    if len(unique_ids) < 7:
        raise ValueError("至少需要 7 个唯一 original_ID 才能执行 70/15/15 切分。")

    train_val_ids, test_ids = train_test_split(
        unique_ids,
        test_size=TEST_RATIO,
        random_state=random_seed,
        shuffle=True,
    )
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO),
        random_state=random_seed,
        shuffle=True,
    )

    result = {
        "train": sorted(map(str, train_ids)),
        "val": sorted(map(str, val_ids)),
        "test": sorted(map(str, test_ids)),
    }
    sets = {name: set(ids) for name, ids in result.items()}
    if (
        sets["train"] & sets["val"]
        or sets["train"] & sets["test"]
        or sets["val"] & sets["test"]
    ):
        raise RuntimeError("original_ID 切分出现交集，防泄漏检查失败。")
    return result


def _json_ready(value: Any) -> Any:
    """将 NumPy 标量等对象转换为 JSON 可序列化类型。"""
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_manifest(
    path: Path,
    filepath: Path,
    random_seed: int,
    feature_cols: list[str],
    id_splits: dict[str, list[str]],
    split_stats: dict[str, dict[str, int]],
    sample_mode: bool,
    split_strategy: str,
    split_provenance: dict[str, Any] | None,
) -> None:
    source_stat = filepath.stat()
    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": _portable_path(filepath),
        "source_size_bytes": source_stat.st_size,
        "source_mtime_utc": datetime.fromtimestamp(
            source_stat.st_mtime, tz=timezone.utc
        ).isoformat(),
        "group_column": GROUP_COL,
        "random_seed": random_seed,
        "target_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "sample_mode": sample_mode,
        "split_strategy": split_strategy,
        "split_provenance": split_provenance,
        "feature_columns": feature_cols,
        "target_columns": TARGET_COLS,
        "splits": {
            name: {
                **split_stats[name],
                "original_ids": ids,
            }
            for name, ids in id_splits.items()
        },
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_ready),
        encoding="utf-8",
    )


def validate_predefined_id_splits(
    predefined_id_splits: dict[str, list[str] | np.ndarray],
    valid_original_ids: list[str] | np.ndarray,
) -> dict[str, list[str]]:
    """验证并规范化外部给定的 train/val/test ``original_ID`` 切分。"""
    expected_names = {"train", "val", "test"}
    actual_names = set(predefined_id_splits)
    if actual_names != expected_names:
        raise ValueError(
            "预定义切分必须且只能包含 train/val/test，"
            f"实际为: {sorted(actual_names)}"
        )

    normalized: dict[str, list[str]] = {}
    for name in ("train", "val", "test"):
        raw_ids = [str(value).strip() for value in predefined_id_splits[name]]
        if any(not value for value in raw_ids):
            raise ValueError(f"{name} 切分包含空 original_ID。")
        if len(raw_ids) != len(set(raw_ids)):
            raise ValueError(f"{name} 切分包含重复 original_ID。")
        if not raw_ids:
            raise ValueError(f"{name} 切分不能为空。")
        normalized[name] = sorted(raw_ids)

    id_sets = {name: set(ids) for name, ids in normalized.items()}
    if (
        id_sets["train"] & id_sets["val"]
        or id_sets["train"] & id_sets["test"]
        or id_sets["val"] & id_sets["test"]
    ):
        raise ValueError("预定义切分的 original_ID 存在交集。")

    expected_ids = set(map(str, valid_original_ids))
    assigned_ids = set.union(*id_sets.values())
    missing_ids = sorted(expected_ids - assigned_ids)
    unexpected_ids = sorted(assigned_ids - expected_ids)
    if missing_ids or unexpected_ids:
        raise ValueError(
            "预定义切分必须完整覆盖有效 original_ID；"
            f"缺少 {missing_ids[:5]}，多出 {unexpected_ids[:5]}。"
        )
    return normalized


def load_and_split_data(
    filepath: str | Path = DATA_PATH,
    random_seed: int = RANDOM_SEED,
    sample_mode: bool = False,
    sample_size: int = 200,
    artifact_dir: str | Path | None = None,
    save_artifacts: bool = True,
    feature_cols: list[str] | tuple[str, ...] | None = None,
    predefined_id_splits: dict[str, list[str] | np.ndarray] | None = None,
    split_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    加载数据，按 original_ID 切分，计算目标并标准化特征。

    sample_mode 下的 sample_size 表示抽取的物理浮标 ID 数，而不是子轨迹数。
    """
    selected_features = list(feature_cols or FEATURE_COLS)
    if not selected_features:
        raise ValueError("feature_cols 不能为空。")
    if len(selected_features) != len(set(selected_features)):
        raise ValueError(f"feature_cols 包含重复项: {selected_features}")
    unknown_features = set(selected_features) - set(FEATURE_COLS)
    if unknown_features:
        raise ValueError(f"未知特征: {sorted(unknown_features)}")

    filepath = Path(filepath).resolve()
    artifact_dir = Path(
        artifact_dir or TRAINED_MODELS_DIR / DEFAULT_RUN_NAME
    ).resolve()
    if save_artifacts:
        artifact_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "【采样模式】" if sample_mode else "【完整模式】"
    logger.info("=== 开始数据加载 %s ===", mode_tag)
    logger.info("步骤 1/4: 从 '%s' 加载轨迹数据...", filepath)

    try:
        with filepath.open("rb") as file:
            all_trajs = pickle.load(file)
    except FileNotFoundError:
        logger.error("文件未找到: %s", filepath)
        raise

    logger.info("原始子轨迹总数: %d", len(all_trajs))
    logger.info("当前内存（加载后）: %.0f MB", _mem_mb())

    # 先确认每个子轨迹的物理 ID；后续所有切分都只基于这些 ID。
    trajectory_ids = [
        _canonical_original_id(df, index) for index, df in enumerate(all_trajs)
    ]
    unique_original_ids = sorted(set(trajectory_ids))
    logger.info(
        "唯一物理浮标 ID: %d（子轨迹/ID = %.2f）",
        len(unique_original_ids),
        len(all_trajs) / len(unique_original_ids),
    )

    if sample_mode:
        rng = np.random.default_rng(random_seed)
        selected = rng.choice(
            unique_original_ids,
            size=min(sample_size, len(unique_original_ids)),
            replace=False,
        )
        selected_ids = set(map(str, selected))
        selected_pairs = [
            (df, original_id)
            for df, original_id in zip(all_trajs, trajectory_ids)
            if original_id in selected_ids
        ]
        all_trajs = [pair[0] for pair in selected_pairs]
        trajectory_ids = [pair[1] for pair in selected_pairs]
        del selected_pairs
        unique_original_ids = sorted(selected_ids)
        logger.info(
            "采样后: %d 个 original_ID / %d 条子轨迹",
            len(unique_original_ids),
            len(all_trajs),
        )

    logger.info("步骤 2/4: 过滤无效子轨迹并计算漂移残差...")
    model_input_cols = list(
        dict.fromkeys(selected_features + WIND_COLS + OBS_COLS + CURRENT_COLS)
    )
    required_cols = set(model_input_cols + [GROUP_COL])
    valid_trajs: list[tuple[str, pd.DataFrame]] = []
    skipped_empty = 0

    for index, (df, original_id) in enumerate(zip(all_trajs, trajectory_ids)):
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"第 {index} 条子轨迹缺少必需列: {sorted(missing)}"
            )

        clean = df[model_input_cols].dropna().copy()
        if clean.empty:
            skipped_empty += 1
            continue
        clean["residual_u"] = clean["ve"] - clean["cfsv2_u"]
        clean["residual_v"] = clean["vn"] - clean["cfsv2_v"]
        valid_trajs.append((original_id, clean))

    del all_trajs, trajectory_ids
    gc.collect()
    logger.info(
        "有效子轨迹: %d（空轨迹跳过: %d）  内存: %.0f MB",
        len(valid_trajs),
        skipped_empty,
        _mem_mb(),
    )
    if not valid_trajs:
        raise ValueError("没有有效轨迹，请检查数据列名和缺测值。")

    valid_original_ids = sorted({original_id for original_id, _ in valid_trajs})
    if predefined_id_splits is None:
        id_splits = split_original_ids(
            valid_original_ids,
            random_seed=random_seed,
        )
        split_strategy = "generated_original_id_split"
    else:
        id_splits = validate_predefined_id_splits(
            predefined_id_splits,
            valid_original_ids,
        )
        split_strategy = "predefined_original_id_split"
    id_sets = {name: set(ids) for name, ids in id_splits.items()}

    logger.info(
        "步骤 3/4: 按 original_ID 防泄漏切分（%d/%d/%d 个 ID）...",
        len(id_splits["train"]),
        len(id_splits["val"]),
        len(id_splits["test"]),
    )

    split_frames: dict[str, list[pd.DataFrame]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    split_stats = {
        name: {"n_original_ids": len(ids), "n_segments": 0, "n_samples": 0}
        for name, ids in id_splits.items()
    }
    for original_id, frame in valid_trajs:
        split_name = next(
            name for name, ids in id_sets.items() if original_id in ids
        )
        split_frames[split_name].append(frame)
        split_stats[split_name]["n_segments"] += 1
        split_stats[split_name]["n_samples"] += len(frame)

    dataframes = {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in split_frames.items()
    }
    del valid_trajs, split_frames
    gc.collect()

    total_samples = sum(stats["n_samples"] for stats in split_stats.values())
    for name in ("train", "val", "test"):
        stats = split_stats[name]
        logger.info(
            "%s: %d 个 original_ID / %d 条子轨迹 / %d 点（%.2f%%）",
            name,
            stats["n_original_ids"],
            stats["n_segments"],
            stats["n_samples"],
            100 * stats["n_samples"] / total_samples,
        )
    logger.info("original_ID 交集检查: train/val/test 两两为空")

    logger.info("步骤 4/4: 标准化输入特征（StandardScaler 仅 fit train）...")

    arrays: dict[str, np.ndarray] = {}
    for name in ("train", "val", "test"):
        frame = dataframes.pop(name)
        arrays[f"X_{name}_raw"] = frame[selected_features].to_numpy(
            dtype=np.float32, copy=True
        )
        arrays[f"X_{name}_wind"] = frame[WIND_COLS].to_numpy(
            dtype=np.float32, copy=True
        )
        arrays[f"y_{name}"] = frame[TARGET_COLS].to_numpy(
            dtype=np.float32, copy=True
        )
        del frame
        gc.collect()

    x_scaler = StandardScaler()
    arrays["X_train"] = x_scaler.fit_transform(
        arrays.pop("X_train_raw")
    ).astype(np.float32, copy=False)
    arrays["X_val"] = x_scaler.transform(
        arrays.pop("X_val_raw")
    ).astype(np.float32, copy=False)
    arrays["X_test"] = x_scaler.transform(
        arrays.pop("X_test_raw")
    ).astype(np.float32, copy=False)

    manifest_path = artifact_dir / "split_manifest.json"
    scaler_path = artifact_dir / "x_scaler.pkl"
    if save_artifacts:
        joblib.dump(x_scaler, scaler_path)
        _write_manifest(
            manifest_path,
            filepath,
            random_seed,
            selected_features,
            id_splits,
            split_stats,
            sample_mode,
            split_strategy,
            split_provenance,
        )
        logger.info("StandardScaler 已保存: %s", scaler_path)
        logger.info("切分清单已保存: %s", manifest_path)

    logger.info(
        "X shape | train=%s, val=%s, test=%s",
        arrays["X_train"].shape,
        arrays["X_val"].shape,
        arrays["X_test"].shape,
    )
    logger.info(
        "训练目标均值 | residual_u=%.4f, residual_v=%.4f m/s",
        arrays["y_train"][:, 0].mean(),
        arrays["y_train"][:, 1].mean(),
    )
    logger.info("=== 数据加载完毕 ===")

    return {
        **arrays,
        "x_scaler": x_scaler,
        "feature_cols": selected_features,
        "target_cols": TARGET_COLS.copy(),
        "id_splits": id_splits,
        "split_stats": split_stats,
        "artifact_dir": artifact_dir,
        "scaler_path": scaler_path if save_artifacts else None,
        "manifest_path": manifest_path if save_artifacts else None,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="按 original_ID 构建数据集")
    parser.add_argument("--full", action="store_true", help="使用完整数据集")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=200,
        help="采样模式使用的 original_ID 数",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    splits = load_and_split_data(
        sample_mode=not args.full,
        sample_size=args.sample_size,
        artifact_dir=TRAINED_MODELS_DIR / args.run_name,
    )

    print("\n===== 数据集摘要 =====")
    for key in ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"):
        array = splits[key]
        print(f"  {key:12s}: shape={array.shape}, dtype={array.dtype}")
    print(f"\n特征列: {splits['feature_cols']}")
    print(f"Scaler 均值 (前3): {splits['x_scaler'].mean_[:3]}")
    print("===== original_ID 防泄漏验证通过 =====")
