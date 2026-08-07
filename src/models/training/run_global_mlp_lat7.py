"""训练 global core6 + ``sin(latitude)`` 七维输入 MLP。

本入口只改变冻结 global MLP 的输入维度。网络隐藏层、损失函数、优化器、
学习率策略、early stopping 和随机种子均复用冻结实验配置；数据逐行继承
``global_factorial_v1`` 共用缓存。最佳 validation checkpoint 锁定后才评价
test 集。
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
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
    LATITUDE_FEATURE,
    TARGET_COLUMNS,
    assemble_features,
    evaluate_predictions,
    load_cached_split,
    sha256_file,
    validate_cache,
)
from .train_mlp import plot_history, train


RUN_NAME = "global_mlp_lat7_v1"
ARTIFACT_DIR = PROJECT_ROOT / "trained_models" / RUN_NAME
RESULT_DIR = PROJECT_ROOT / "results" / RUN_NAME
LOG_DIR = PROJECT_ROOT / "logs"
PREDICTION_PATH = CACHE_DIR / "mlp_lat7_test_pred.npy"
FEATURE_COLUMNS = list(CORE6_FEATURES) + [LATITUDE_FEATURE]

logger = logging.getLogger(__name__)


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


def _setup_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        f"{RUN_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def standardize_cached_splits(
    raw_splits: dict[str, dict[str, np.ndarray]],
    scaler_path: Path,
) -> tuple[dict[str, Any], StandardScaler]:
    """仅在 train 集拟合 scaler，并构造 train/val/test 七维数组。"""
    unscaled = {
        name: assemble_features(
            split["core6"],
            split["sin_latitude"],
        )
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
        "feature_cols": FEATURE_COLUMNS.copy(),
        "target_cols": list(TARGET_COLUMNS),
        "artifact_dir": ARTIFACT_DIR,
    }
    return splits, scaler


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


def run(require_cuda: bool = True) -> dict[str, Any]:
    """运行完整七维 MLP 训练、一次 test 评价及产物冻结。"""
    log_path = _setup_logging()
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "完整训练要求 CUDA；如仅做调试，请显式传入 --allow-cpu"
        )

    logger.info("=" * 72)
    logger.info("开始 %s", RUN_NAME)
    logger.info("唯一实验变量：追加 sin(deg2rad(latitude)) 为第 7 维")
    logger.info("特征顺序：%s", FEATURE_COLUMNS)
    logger.info("代码提交：%s", _git_commit())
    logger.info("=" * 72)

    validate_cache(verify_checksums=True)
    raw_splits = {
        name: load_cached_split(name)
        for name in ("train", "val", "test")
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = ARTIFACT_DIR / "x_scaler.pkl"
    splits, scaler = standardize_cached_splits(raw_splits, scaler_path)
    logger.info(
        "数据 shape：train=%s, val=%s, test=%s",
        splits["X_train"].shape,
        splits["X_val"].shape,
        splits["X_test"].shape,
    )
    logger.info("scaler 已保存：%s", scaler_path)

    # train() 内部只用 validation loss 选择 checkpoint，不读取 test 指标。
    train_result = train(splits)
    plot_history(
        train_result["history"],
        ARTIFACT_DIR,
        result_dir=RESULT_DIR,
    )

    # 到此 checkpoint 已锁定；以下是本实验唯一一次 test 评价。
    prediction = _predict_test(train_result)
    np.save(PREDICTION_PATH, prediction)
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
    candidate_row = evaluation["row_weighted"]
    frozen_row = frozen_evaluation["row_weighted"]
    candidate_macro = evaluation["macro_original_id"]
    frozen_macro = frozen_evaluation["macro_original_id"]
    comparison = {
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

    metrics = {
        "schema_version": 1,
        "run_name": RUN_NAME,
        "selection": {
            "test_evaluated_after_checkpoint_lock": True,
            "checkpoint_monitor": "validation_loss",
            "best_epoch": int(train_result["best_epoch"]),
            "validation_loss": float(train_result["best_val_loss"]),
            "validation_r2_joint": float(train_result["best_val_r2"]),
            "parameter_count": int(train_result["parameter_count"]),
        },
        "test": evaluation,
        "frozen_global_reference": frozen_evaluation,
        "comparison": comparison,
    }
    metrics_path = ARTIFACT_DIR / "metrics.json"
    _json_dump(metrics_path, metrics)

    reference_count = min(8, len(prediction))
    fixed_io = {
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": list(TARGET_COLUMNS),
        "raw_input": assemble_features(
            test_raw["core6"][:reference_count],
            test_raw["sin_latitude"][:reference_count],
        ).tolist(),
        "output": prediction[:reference_count].tolist(),
    }
    fixed_io_path = ARTIFACT_DIR / "fixed_test_io.json"
    _json_dump(fixed_io_path, fixed_io)

    checkpoint_path = Path(train_result["best_model_path"])
    manifest = {
        "schema_version": 1,
        "run_name": RUN_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_git_commit": _git_commit(),
        "lineage": {
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
        },
        "features": FEATURE_COLUMNS,
        "latitude_formula": "sin(deg2rad(latitude))",
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
                "path": str(PREDICTION_PATH.relative_to(PROJECT_ROOT)),
                "shape": list(prediction.shape),
                "dtype": str(prediction.dtype),
                "sha256": sha256_file(PREDICTION_PATH),
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
    manifest_path = ARTIFACT_DIR / "run_manifest.json"
    _json_dump(manifest_path, manifest)

    logger.info("test joint R2：%.6f", candidate_row["r2_joint"])
    logger.info("test RMSE：%.6f m/s", candidate_row["rmse"])
    logger.info(
        "相对冻结 global：joint R2 %+.6f，RMSE %+.3f%%",
        comparison["row_weighted_delta_candidate_minus_frozen"]["r2_joint"],
        comparison["rmse_improvement_percent"],
    )
    logger.info("训练与评价完成：%s", manifest_path)
    return manifest


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
