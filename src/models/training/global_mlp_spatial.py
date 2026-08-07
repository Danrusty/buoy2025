"""Global MLP 空间特征实验的共用训练与冻结协议。

lat7 与 lat9 入口都通过本模块复用同一网络、优化器、early stopping、
随机种子和评价逻辑。每个入口只声明新增空间列及其公式，避免两份训练代码
逐渐产生不可见差异。
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler

from ..data_loader import PROJECT_ROOT
from .global_factorial import (
    CACHE_DIR,
    CORE6_FEATURES,
    DATA_MANIFEST_PATH,
    EXPERIMENT_NAME,
    FROZEN_REPLAY_PATH,
    TARGET_COLUMNS,
    evaluate_predictions,
    load_cached_split,
    sha256_file,
    validate_cache,
)
from .global_longitude import (
    load_longitude_split,
    validate_longitude_cache,
)
from .train_mlp import plot_history, train


LOG_DIR = PROJECT_ROOT / "logs"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpatialMlpSpec:
    """定义一次只改变输入空间列的 MLP 受控实验。"""

    run_name: str
    feature_columns: tuple[str, ...]
    spatial_cache_keys: tuple[str, ...]
    coordinate_formulas: tuple[tuple[str, str], ...]
    unique_variable: str
    prediction_filename: str
    longitude_cache_manifest_path: Path | None = None

    @property
    def artifact_dir(self) -> Path:
        return PROJECT_ROOT / "trained_models" / self.run_name

    @property
    def result_dir(self) -> Path:
        return PROJECT_ROOT / "results" / self.run_name

    @property
    def prediction_path(self) -> Path:
        return CACHE_DIR / self.prediction_filename


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _setup_logging(spec: SpatialMlpSpec) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        f"{spec.run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    return log_path


def assemble_spatial_features(
    split: dict[str, np.ndarray],
    spec: SpatialMlpSpec,
) -> np.ndarray:
    """按 spec 的固定顺序拼接 core6 与一维空间特征。"""
    core6 = np.asarray(split["core6"])
    if core6.ndim != 2 or core6.shape[1] != len(CORE6_FEATURES):
        raise ValueError(
            f"core6 应为 (N, 6)，实际为 {core6.shape}"
        )
    columns: list[np.ndarray] = [core6]
    for key in spec.spatial_cache_keys:
        values = np.asarray(split[key])
        if values.shape != (len(core6),):
            raise ValueError(
                f"{key} 应为 ({len(core6)},)，实际为 {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key} 含非有限值")
        columns.append(values[:, None])
    features = np.column_stack(columns).astype(
        np.float32,
        copy=False,
    )
    if features.shape[1] != len(spec.feature_columns):
        raise ValueError(
            "拼接列数与 feature_columns 不一致："
            f"{features.shape[1]} != {len(spec.feature_columns)}"
        )
    return features


def standardize_cached_splits(
    raw_splits: dict[str, dict[str, np.ndarray]],
    scaler_path: Path,
    spec: SpatialMlpSpec,
) -> tuple[dict[str, Any], StandardScaler]:
    """仅在 train 集拟合 scaler，再变换 validation 和 test。"""
    unscaled = {
        name: assemble_spatial_features(split, spec)
        for name, split in raw_splits.items()
    }
    scaler = StandardScaler(copy=False)
    scaled = {
        "train": scaler.fit_transform(unscaled["train"]).astype(
            np.float32,
            copy=False,
        )
    }
    for name in ("val", "test"):
        scaled[name] = scaler.transform(unscaled[name]).astype(
            np.float32,
            copy=False,
        )

    if not np.allclose(
        scaled["train"].mean(axis=0, dtype=np.float64),
        0.0,
        atol=2e-5,
    ):
        raise ValueError("train 标准化均值校验失败")
    if not np.all(np.isfinite(np.asarray(scaler.scale_))):
        raise ValueError("scaler scale 含非有限值")
    if np.any(np.asarray(scaler.scale_) <= 0):
        raise ValueError("scaler scale 必须全部大于 0")

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    splits = {
        "X_train": scaled["train"],
        "X_val": scaled["val"],
        "X_test": scaled["test"],
        "y_train": raw_splits["train"]["target"],
        "y_val": raw_splits["val"]["target"],
        "y_test": raw_splits["test"]["target"],
        "feature_cols": list(spec.feature_columns),
        "target_cols": list(TARGET_COLUMNS),
        "artifact_dir": spec.artifact_dir,
    }
    return splits, scaler


def _load_raw_splits(
    spec: SpatialMlpSpec,
) -> dict[str, dict[str, np.ndarray]]:
    validate_cache(verify_checksums=True)
    raw_splits = {
        name: load_cached_split(name)
        for name in ("train", "val", "test")
    }
    if spec.longitude_cache_manifest_path is not None:
        validate_longitude_cache(
            artifact_manifest_path=spec.longitude_cache_manifest_path,
            verify_checksums=True,
        )
        for name in ("train", "val", "test"):
            longitude = load_longitude_split(
                name,
                artifact_manifest_path=(
                    spec.longitude_cache_manifest_path
                ),
            )
            for key, values in longitude.items():
                if len(values) != len(raw_splits[name]["target"]):
                    raise ValueError(
                        f"{name}.{key} 与基础缓存行数不一致"
                    )
                raw_splits[name][key] = values
    return raw_splits


@torch.no_grad()
def _predict_test(train_result: dict[str, Any]) -> np.ndarray:
    model = train_result["model"]
    model.eval()
    predictions = [
        model(features).cpu().numpy()
        for features, _ in train_result["test_ds"]
    ]
    return np.concatenate(predictions).astype(np.float32, copy=False)


def _load_frozen_prediction() -> tuple[np.ndarray, dict[str, Any]]:
    replay = json.loads(FROZEN_REPLAY_PATH.read_text(encoding="utf-8"))
    record = replay["prediction_cache"]
    path = PROJECT_ROOT / record["path"]
    if sha256_file(path) != record["sha256"]:
        raise ValueError("冻结 global test prediction SHA256 不匹配")
    prediction = np.load(path, mmap_mode="r")
    if list(prediction.shape) != record["shape"]:
        raise ValueError("冻结 global test prediction shape 不匹配")
    return prediction, replay


def _comparison_with_frozen(
    evaluation: dict[str, Any],
    frozen_evaluation: dict[str, Any],
) -> dict[str, Any]:
    candidate_row = evaluation["row_weighted"]
    frozen_row = frozen_evaluation["row_weighted"]
    candidate_macro = evaluation["macro_original_id"]
    frozen_macro = frozen_evaluation["macro_original_id"]
    return {
        "row_weighted_delta_candidate_minus_frozen": {
            key: float(candidate_row[key] - frozen_row[key])
            for key in (
                "r2_u",
                "r2_v",
                "r2_joint",
                "rmse",
                "mae",
                "bias_u",
                "bias_v",
            )
        },
        "rmse_improvement_percent": float(
            (frozen_row["rmse"] - candidate_row["rmse"])
            / frozen_row["rmse"]
            * 100.0
        ),
        "macro_original_id_delta_candidate_minus_frozen": {
            key: float(candidate_macro[key] - frozen_macro[key])
            for key in (
                "r2_u",
                "r2_v",
                "r2_joint",
                "rmse",
                "mae",
                "bias_u",
                "bias_v",
            )
        },
    }


def run_spatial_mlp(
    spec: SpatialMlpSpec,
    require_cuda: bool = True,
) -> dict[str, Any]:
    """训练、锁定最佳 validation checkpoint，再进行一次 test 评价。"""
    log_path = _setup_logging(spec)
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "完整训练要求 CUDA；如仅做调试，请显式传入 --allow-cpu"
        )
    existing = [
        path
        for path in (
            spec.artifact_dir,
            spec.result_dir,
            spec.prediction_path,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "实验产物已存在，拒绝覆盖：" + ", ".join(map(str, existing))
        )

    logger.info("=" * 72)
    logger.info("开始 %s", spec.run_name)
    logger.info("唯一实验变量：%s", spec.unique_variable)
    logger.info("特征顺序：%s", list(spec.feature_columns))
    logger.info("代码提交：%s", _git_commit())
    logger.info("=" * 72)

    raw_splits = _load_raw_splits(spec)
    spec.artifact_dir.mkdir(parents=True)
    spec.result_dir.mkdir(parents=True)
    scaler_path = spec.artifact_dir / "x_scaler.pkl"
    splits, scaler = standardize_cached_splits(
        raw_splits,
        scaler_path,
        spec,
    )
    logger.info(
        "数据 shape：train=%s, val=%s, test=%s",
        splits["X_train"].shape,
        splits["X_val"].shape,
        splits["X_test"].shape,
    )
    logger.info("scaler 已保存：%s", scaler_path)

    # train() 只依据 validation loss 保存 checkpoint，不读取 test 指标。
    train_result = train(splits)
    plot_history(
        train_result["history"],
        spec.artifact_dir,
        result_dir=spec.result_dir,
    )

    # checkpoint 到此已锁定；以下是本实验唯一一次 test 评价。
    prediction = _predict_test(train_result)
    np.save(spec.prediction_path, prediction)
    frozen_prediction, frozen_replay = _load_frozen_prediction()
    test_raw = raw_splits["test"]
    evaluation = evaluate_predictions(
        test_raw["target"],
        prediction,
        test_raw["group_index"],
        test_raw["latitude"],
        reference_pred=frozen_prediction,
    )
    frozen_evaluation = frozen_replay["replayed_evaluation"]
    comparison = _comparison_with_frozen(
        evaluation,
        frozen_evaluation,
    )

    metrics = {
        "schema_version": 1,
        "run_name": spec.run_name,
        "selection": {
            "test_evaluated_after_checkpoint_lock": True,
            "checkpoint_monitor": "validation_loss",
            "best_epoch": int(train_result["best_epoch"]),
            "validation_loss": float(train_result["best_val_loss"]),
            "validation_r2_joint": float(
                train_result["best_val_r2"]
            ),
            "parameter_count": int(train_result["parameter_count"]),
        },
        "test": evaluation,
        "frozen_global_reference": frozen_evaluation,
        "comparison": comparison,
    }
    metrics_path = spec.artifact_dir / "metrics.json"
    _json_dump(metrics_path, metrics)

    reference_count = min(8, len(prediction))
    fixed_io = {
        "feature_columns": list(spec.feature_columns),
        "target_columns": list(TARGET_COLUMNS),
        "raw_input": assemble_spatial_features(
            {
                key: values[:reference_count]
                for key, values in test_raw.items()
            },
            spec,
        ).tolist(),
        "output": prediction[:reference_count].tolist(),
    }
    fixed_io_path = spec.artifact_dir / "fixed_test_io.json"
    _json_dump(fixed_io_path, fixed_io)

    checkpoint_path = Path(train_result["best_model_path"])
    lineage: dict[str, Any] = {
        "shared_experiment": EXPERIMENT_NAME,
        "data_manifest": str(
            DATA_MANIFEST_PATH.relative_to(PROJECT_ROOT)
        ),
        "data_manifest_sha256": sha256_file(DATA_MANIFEST_PATH),
        "frozen_reference_replay": str(
            FROZEN_REPLAY_PATH.relative_to(PROJECT_ROOT)
        ),
        "frozen_reference_replay_sha256": sha256_file(
            FROZEN_REPLAY_PATH
        ),
    }
    if spec.longitude_cache_manifest_path is not None:
        lineage["longitude_cache_manifest"] = str(
            spec.longitude_cache_manifest_path.relative_to(PROJECT_ROOT)
        )
        lineage["longitude_cache_manifest_sha256"] = sha256_file(
            spec.longitude_cache_manifest_path
        )

    manifest = {
        "schema_version": 1,
        "run_name": spec.run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_git_commit": _git_commit(),
        "lineage": lineage,
        "features": list(spec.feature_columns),
        "coordinate_formulas": dict(spec.coordinate_formulas),
        "unique_experiment_variable": spec.unique_variable,
        "targets": {
            "residual_u": "ve - cfsv2_u",
            "residual_v": "vn - cfsv2_v",
        },
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(checkpoint_path),
            },
            "scaler": {
                "path": str(scaler_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(scaler_path),
                "mean": np.asarray(scaler.mean_).tolist(),
                "scale": np.asarray(scaler.scale_).tolist(),
                "n_samples_seen": int(scaler.n_samples_seen_),
            },
            "metrics": {
                "path": str(metrics_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(metrics_path),
            },
            "fixed_test_io": {
                "path": str(fixed_io_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(fixed_io_path),
            },
            "prediction_cache": {
                "path": str(
                    spec.prediction_path.relative_to(PROJECT_ROOT)
                ),
                "shape": list(prediction.shape),
                "dtype": str(prediction.dtype),
                "sha256": sha256_file(spec.prediction_path),
            },
        },
        "software": {
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(train_result["device"]),
        },
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
    }
    manifest_path = spec.artifact_dir / "run_manifest.json"
    _json_dump(manifest_path, manifest)

    candidate_row = evaluation["row_weighted"]
    logger.info("test joint R2：%.6f", candidate_row["r2_joint"])
    logger.info("test RMSE：%.6f m/s", candidate_row["rmse"])
    logger.info(
        "相对冻结 global：joint R2 %+.6f，RMSE 改善 %+.3f%%",
        comparison[
            "row_weighted_delta_candidate_minus_frozen"
        ]["r2_joint"],
        comparison["rmse_improvement_percent"],
    )
    logger.info("训练与评价完成：%s", manifest_path)
    return manifest
