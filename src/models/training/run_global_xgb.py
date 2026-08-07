"""运行 global XGBoost core6/lat7 两个受控实验。

两个分支共享完全相同的候选配置、validation-only 选择规则与评价流程。每个
候选分别训练 residual_u、residual_v 两个 booster，以联合 validation RMSE
选择唯一配置。程序先冻结 ``selection_lock.json``，随后才加载并评价 test，
从流程上隔离调参与最终测试。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from ..data_loader import PROJECT_ROOT
from ..evaluation import regression_metrics
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


RANDOM_SEED = 42
MAX_BIN = 256
MAX_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 50
NTHREAD = 8
FEATURE_SET_CHOICES = ("core6", "lat7")

# 候选必须在访问 test 前冻结；core6 与 lat7 不允许使用不同搜索空间。
CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "name": "depth6_eta005",
        "grow_policy": "depthwise",
        "max_depth": 6,
        "max_leaves": 0,
        "eta": 0.05,
        "min_child_weight": 128.0,
        "reg_lambda": 10.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "depth8_eta003",
        "grow_policy": "depthwise",
        "max_depth": 8,
        "max_leaves": 0,
        "eta": 0.03,
        "min_child_weight": 128.0,
        "reg_lambda": 10.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "lossguide64_eta005",
        "grow_policy": "lossguide",
        "max_depth": 0,
        "max_leaves": 64,
        "eta": 0.05,
        "min_child_weight": 128.0,
        "reg_lambda": 10.0,
        "reg_alpha": 0.0,
    },
    {
        "name": "lossguide128_eta003",
        "grow_policy": "lossguide",
        "max_depth": 0,
        "max_leaves": 128,
        "eta": 0.03,
        "min_child_weight": 256.0,
        "reg_lambda": 20.0,
        "reg_alpha": 0.0,
    },
)

LOG_DIR = PROJECT_ROOT / "logs"
logger = logging.getLogger(__name__)


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run_paths(feature_set: str) -> dict[str, Path | str]:
    if feature_set not in FEATURE_SET_CHOICES:
        raise ValueError(f"未知 feature_set：{feature_set}")
    run_name = f"global_xgb_{feature_set}_v1"
    return {
        "run_name": run_name,
        "branch": f"wdf_global_xgb_{feature_set}_v1",
        "artifact_dir": PROJECT_ROOT / "trained_models" / run_name,
        "result_dir": PROJECT_ROOT / "results" / run_name,
        "prediction_path": CACHE_DIR / f"xgb_{feature_set}_test_pred.npy",
    }


def _feature_columns(feature_set: str) -> list[str]:
    columns = list(CORE6_FEATURES)
    if feature_set == "lat7":
        columns.append(LATITUDE_FEATURE)
    return columns


def build_features(
    cached_split: dict[str, np.ndarray],
    feature_set: str,
) -> np.ndarray:
    """按固定顺序从共用缓存构建 core6 或 lat7 输入。"""
    if feature_set == "core6":
        return assemble_features(cached_split["core6"])
    if feature_set == "lat7":
        return assemble_features(
            cached_split["core6"],
            cached_split["sin_latitude"],
        )
    raise ValueError(f"未知 feature_set：{feature_set}")


def _setup_logging(run_name: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def _candidate_parameters(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "tree_method": "hist",
        "device": "cuda",
        "max_bin": MAX_BIN,
        "sampling_method": "uniform",
        "subsample": 0.8,
        "colsample_bytree": 1.0,
        "seed": RANDOM_SEED,
        "nthread": NTHREAD,
        **{key: value for key, value in candidate.items() if key != "name"},
    }


def choose_candidate(
    candidate_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """只根据 validation RMSE 选型，并以树数和预声明顺序破除平局。"""
    if not candidate_results:
        raise ValueError("candidate_results 不能为空")
    order = {
        candidate["name"]: index
        for index, candidate in enumerate(CANDIDATES)
    }
    unknown = {
        result["name"]
        for result in candidate_results
        if result["name"] not in order
    }
    if unknown:
        raise ValueError(f"出现未声明候选：{sorted(unknown)}")
    return min(
        candidate_results,
        key=lambda result: (
            float(result["validation"]["rmse"]),
            int(result["total_boosted_rounds"]),
            order[result["name"]],
        ),
    )


def accuracy_gate(
    candidate_evaluation: dict[str, Any],
    frozen_evaluation: dict[str, Any],
) -> dict[str, Any]:
    """计算是否值得进入目标环境 XGBoost 推理效率评价。"""
    candidate_row = candidate_evaluation["row_weighted"]
    frozen_row = frozen_evaluation["row_weighted"]
    candidate_macro = candidate_evaluation["macro_original_id"]
    frozen_macro = frozen_evaluation["macro_original_id"]
    r2_gain = float(
        candidate_row["r2_joint"] - frozen_row["r2_joint"]
    )
    rmse_improvement_percent = float(
        (frozen_row["rmse"] - candidate_row["rmse"])
        / frozen_row["rmse"]
        * 100.0
    )
    macro_r2_gain = float(
        candidate_macro["r2_joint"] - frozen_macro["r2_joint"]
    )
    macro_rmse_change = float(
        candidate_macro["rmse"] - frozen_macro["rmse"]
    )
    checks = {
        "row_joint_r2_gain_at_least_0_01": r2_gain >= 0.01,
        "row_rmse_improvement_at_least_1_percent": (
            rmse_improvement_percent >= 1.0
        ),
        "macro_joint_r2_improves": macro_r2_gain > 0.0,
        "macro_rmse_improves": macro_rmse_change < 0.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "row_joint_r2_gain": r2_gain,
        "row_rmse_improvement_percent": rmse_improvement_percent,
        "macro_joint_r2_gain": macro_r2_gain,
        "macro_rmse_change": macro_rmse_change,
    }


def _train_one_booster(
    candidate: dict[str, Any],
    target_name: str,
    target_index: int,
    dtrain: xgb.QuantileDMatrix,
    dval: xgb.QuantileDMatrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
) -> tuple[xgb.Booster, np.ndarray, dict[str, Any]]:
    dtrain.set_label(y_train[:, target_index])
    dval.set_label(y_val[:, target_index])
    evaluation_history: dict[str, dict[str, list[float]]] = {}
    callback = xgb.callback.EarlyStopping(
        rounds=EARLY_STOPPING_ROUNDS,
        metric_name="rmse",
        data_name="validation",
        maximize=False,
        save_best=True,
        min_delta=0.0,
    )
    started = time.perf_counter()
    booster = xgb.train(
        params=_candidate_parameters(candidate),
        dtrain=dtrain,
        num_boost_round=MAX_BOOST_ROUNDS,
        evals=[(dtrain, "train"), (dval, "validation")],
        evals_result=evaluation_history,
        callbacks=[callback],
        verbose_eval=50,
    )
    training_seconds = time.perf_counter() - started
    prediction = booster.predict(dval).astype(np.float32, copy=False)
    validation_rmse = float(
        np.sqrt(
            np.mean(
                (
                    prediction.astype(np.float64)
                    - np.asarray(y_val[:, target_index], dtype=np.float64)
                )
                ** 2
            )
        )
    )
    record = {
        "target": target_name,
        "best_iteration_zero_based": int(booster.best_iteration),
        "boosted_rounds_saved": int(booster.num_boosted_rounds()),
        "best_validation_rmse_reported": float(booster.best_score),
        "validation_rmse_recomputed": validation_rmse,
        "training_seconds": float(training_seconds),
        "history": evaluation_history,
    }
    return booster, prediction, record


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


def _comparison(
    candidate: dict[str, Any],
    frozen: dict[str, Any],
) -> dict[str, Any]:
    candidate_row = candidate["row_weighted"]
    frozen_row = frozen["row_weighted"]
    candidate_macro = candidate["macro_original_id"]
    frozen_macro = frozen["macro_original_id"]
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


def _write_search_csv(
    path: Path,
    candidate_results: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "validation_r2_u",
                "validation_r2_v",
                "validation_r2_joint",
                "validation_rmse",
                "validation_mae",
                "u_rounds",
                "v_rounds",
                "total_boosted_rounds",
                "training_seconds",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for result in candidate_results:
            validation = result["validation"]
            writer.writerow(
                {
                    "name": result["name"],
                    **{
                        f"validation_{key}": validation[key]
                        for key in (
                            "r2_u",
                            "r2_v",
                            "r2_joint",
                            "rmse",
                            "mae",
                        )
                    },
                    "u_rounds": result["targets"][0][
                        "boosted_rounds_saved"
                    ],
                    "v_rounds": result["targets"][1][
                        "boosted_rounds_saved"
                    ],
                    "total_boosted_rounds": result[
                        "total_boosted_rounds"
                    ],
                    "training_seconds": result["training_seconds"],
                }
            )


def run(
    feature_set: str,
    allow_branch_mismatch: bool = False,
) -> dict[str, Any]:
    """执行 validation 搜索、selection lock 和唯一一次 test 评价。"""
    paths = _run_paths(feature_set)
    run_name = str(paths["run_name"])
    expected_branch = str(paths["branch"])
    artifact_dir = Path(paths["artifact_dir"])
    result_dir = Path(paths["result_dir"])
    prediction_path = Path(paths["prediction_path"])
    current_branch = _git_value("branch", "--show-current")
    if current_branch != expected_branch and not allow_branch_mismatch:
        raise RuntimeError(
            f"当前分支为 {current_branch!r}，预期 {expected_branch!r}"
        )
    occupied = [
        path
        for path in (artifact_dir, result_dir, prediction_path)
        if path.exists()
    ]
    if occupied:
        raise FileExistsError(
            "检测到已有实验产物；为保护 selection lock，拒绝覆盖："
            f"{occupied}"
        )

    log_path = _setup_logging(run_name)
    feature_columns = _feature_columns(feature_set)
    logger.info("=" * 76)
    logger.info("开始 %s | branch=%s", run_name, current_branch)
    logger.info("特征顺序：%s", feature_columns)
    logger.info("候选顺序：%s", [item["name"] for item in CANDIDATES])
    logger.info("代码提交：%s", _git_value("rev-parse", "HEAD"))
    logger.info("=" * 76)

    build_info = xgb.build_info()
    if not bool(build_info.get("USE_CUDA")):
        raise RuntimeError("当前 XGBoost 构建不支持 CUDA")
    validate_cache(
        verify_checksums=True,
        split_names=("train", "val"),
    )

    # 搜索阶段只加载 train/validation，不加载 test 特征或 target。
    train_cached = load_cached_split("train")
    val_cached = load_cached_split("val")
    X_train = build_features(train_cached, feature_set)
    X_val = build_features(val_cached, feature_set)
    y_train = train_cached["target"]
    y_val = val_cached["target"]
    logger.info(
        "validation-only 数据 shape：train=%s, val=%s",
        X_train.shape,
        X_val.shape,
    )

    logger.info("构建共享 QuantileDMatrix（max_bin=%d）...", MAX_BIN)
    matrix_started = time.perf_counter()
    dtrain = xgb.QuantileDMatrix(
        X_train,
        label=y_train[:, 0],
        feature_names=feature_columns,
        max_bin=MAX_BIN,
    )
    dval = xgb.QuantileDMatrix(
        X_val,
        label=y_val[:, 0],
        feature_names=feature_columns,
        max_bin=MAX_BIN,
        ref=dtrain,
    )
    matrix_seconds = time.perf_counter() - matrix_started
    logger.info("QuantileDMatrix 构建完成：%.1f 秒", matrix_seconds)

    candidate_results: list[dict[str, Any]] = []
    selected_boosters: list[xgb.Booster] | None = None
    selected_name: str | None = None
    for candidate in CANDIDATES:
        logger.info("-" * 76)
        logger.info("候选：%s | %s", candidate["name"], candidate)
        boosters: list[xgb.Booster] = []
        predictions: list[np.ndarray] = []
        target_records: list[dict[str, Any]] = []
        for target_index, target_name in enumerate(TARGET_COLUMNS):
            logger.info("训练 %s / %s", candidate["name"], target_name)
            booster, prediction, record = _train_one_booster(
                candidate,
                target_name,
                target_index,
                dtrain,
                dval,
                y_train,
                y_val,
            )
            boosters.append(booster)
            predictions.append(prediction)
            target_records.append(record)

        joint_prediction = np.column_stack(predictions)
        validation = regression_metrics(y_val, joint_prediction)
        result = {
            "name": candidate["name"],
            "parameters": _candidate_parameters(candidate),
            "targets": target_records,
            "validation": validation,
            "total_boosted_rounds": int(
                sum(
                    record["boosted_rounds_saved"]
                    for record in target_records
                )
            ),
            "training_seconds": float(
                sum(record["training_seconds"] for record in target_records)
            ),
        }
        candidate_results.append(result)
        current_selected = choose_candidate(candidate_results)
        if current_selected["name"] == candidate["name"]:
            selected_boosters = boosters
            selected_name = candidate["name"]
        logger.info(
            "%s validation：joint R2=%.6f, RMSE=%.6f, 当前最优=%s",
            candidate["name"],
            validation["r2_joint"],
            validation["rmse"],
            current_selected["name"],
        )

    chosen = choose_candidate(candidate_results)
    if selected_boosters is None or selected_name != chosen["name"]:
        raise RuntimeError("内存中的 selected booster 与选择记录不一致")

    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    search_path = artifact_dir / "validation_search.json"
    search_payload = {
        "schema_version": 1,
        "run_name": run_name,
        "feature_set": feature_set,
        "feature_columns": feature_columns,
        "selection_metric": "validation joint RMSE, lower is better",
        "tie_break": "fewer total trees, then predeclared candidate order",
        "max_boost_rounds": MAX_BOOST_ROUNDS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "quantile_dmatrix_seconds": matrix_seconds,
        "candidates": candidate_results,
        "selected_candidate": chosen["name"],
    }
    _json_dump(search_path, search_payload)
    _write_search_csv(
        result_dir / "validation_search.csv",
        candidate_results,
    )

    model_paths = [
        artifact_dir / "model_residual_u.ubj",
        artifact_dir / "model_residual_v.ubj",
    ]
    for booster, model_path in zip(selected_boosters, model_paths):
        booster.save_model(model_path)

    # 先冻结 validation 选择与模型哈希，随后才允许加载 test。
    selection_lock_path = artifact_dir / "selection_lock.json"
    selection_lock = {
        "schema_version": 1,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_name": run_name,
        "training_code_git_commit": _git_value("rev-parse", "HEAD"),
        "test_features_or_targets_loaded_before_lock": False,
        "selection_metric": "validation joint RMSE",
        "selected_candidate": chosen["name"],
        "selected_validation_metrics": chosen["validation"],
        "validation_search": {
            "path": str(search_path.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(search_path),
        },
        "models": [
            {
                "target": target,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
                "boosted_rounds": int(booster.num_boosted_rounds()),
            }
            for target, path, booster in zip(
                TARGET_COLUMNS,
                model_paths,
                selected_boosters,
            )
        ],
    }
    _json_dump(selection_lock_path, selection_lock)
    selection_lock_sha256 = sha256_file(selection_lock_path)
    logger.info(
        "selection 已锁定：%s | SHA256=%s",
        selection_lock_path,
        selection_lock_sha256,
    )

    # selection lock 已落盘；从这里开始才首次加载 test。
    test_cached = load_cached_split("test")
    X_test = build_features(test_cached, feature_set)
    dtest = xgb.DMatrix(X_test, feature_names=feature_columns)
    test_prediction = np.column_stack(
        [booster.predict(dtest) for booster in selected_boosters]
    ).astype(np.float32, copy=False)
    np.save(prediction_path, test_prediction)
    frozen_prediction, frozen_replay = _load_frozen_prediction()
    evaluation = evaluate_predictions(
        test_cached["target"],
        test_prediction,
        test_cached["group_index"],
        test_cached["latitude"],
        reference_pred=frozen_prediction,
    )
    frozen_evaluation = frozen_replay["replayed_evaluation"]
    comparison = _comparison(evaluation, frozen_evaluation)
    gate = accuracy_gate(evaluation, frozen_evaluation)

    metrics = {
        "schema_version": 1,
        "run_name": run_name,
        "selection_lock": {
            "path": str(selection_lock_path.relative_to(PROJECT_ROOT)),
            "sha256": selection_lock_sha256,
        },
        "test_evaluated_once_after_selection_lock": True,
        "test": evaluation,
        "frozen_global_reference": frozen_evaluation,
        "comparison": comparison,
        "xgboost_efficiency_gate": gate,
    }
    metrics_path = artifact_dir / "metrics.json"
    _json_dump(metrics_path, metrics)

    reference_count = min(8, len(test_prediction))
    fixed_io_path = artifact_dir / "fixed_test_io.json"
    _json_dump(
        fixed_io_path,
        {
            "feature_columns": feature_columns,
            "target_columns": list(TARGET_COLUMNS),
            "raw_input": np.asarray(X_test[:reference_count]).tolist(),
            "output": test_prediction[:reference_count].tolist(),
        },
    )

    manifest = {
        "schema_version": 1,
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": current_branch,
        "training_code_git_commit": _git_value("rev-parse", "HEAD"),
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
        "features": feature_columns,
        "targets": {
            "residual_u": "ve - cfsv2_u",
            "residual_v": "vn - cfsv2_v",
        },
        "selected_candidate": chosen["name"],
        "artifacts": {
            "selection_lock": {
                "path": str(selection_lock_path.relative_to(PROJECT_ROOT)),
                "sha256": selection_lock_sha256,
            },
            "models": selection_lock["models"],
            "metrics": {
                "path": str(metrics_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(metrics_path),
            },
            "fixed_test_io": {
                "path": str(fixed_io_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(fixed_io_path),
            },
            "prediction_cache": {
                "path": str(prediction_path.relative_to(PROJECT_ROOT)),
                "shape": list(test_prediction.shape),
                "dtype": str(test_prediction.dtype),
                "sha256": sha256_file(prediction_path),
            },
        },
        "software": {
            "numpy": np.__version__,
            "xgboost": xgb.__version__,
            "xgboost_build_info": build_info,
        },
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
    }
    manifest_path = artifact_dir / "run_manifest.json"
    _json_dump(manifest_path, manifest)

    row = evaluation["row_weighted"]
    logger.info("test joint R2：%.6f", row["r2_joint"])
    logger.info("test RMSE：%.6f m/s", row["rmse"])
    logger.info(
        "相对冻结 global：joint R2 %+.6f，RMSE %+.3f%%",
        comparison["row_weighted_delta_candidate_minus_frozen"]["r2_joint"],
        comparison["rmse_improvement_percent"],
    )
    logger.info("XGBoost 效率门槛：%s", gate["passed"])
    logger.info("训练与评价完成：%s", manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="运行 global XGBoost core6/lat7 受控实验",
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_CHOICES,
        required=True,
    )
    parser.add_argument(
        "--allow-branch-mismatch",
        action="store_true",
        help="仅用于调试；允许在非对应实验分支运行",
    )
    args = parser.parse_args()
    run(
        feature_set=args.feature_set,
        allow_branch_mismatch=args.allow_branch_mismatch,
    )


if __name__ == "__main__":
    main()
