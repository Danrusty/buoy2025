"""严格 wind-only 消融：在冻结 core6 样本与协议上只移除波浪输入。

正式运行同时生成两类证据：
  1. 冻结 original_ID test 上的逐时残差速度指标；
  2. 6/12/24/48/72 h 的 open-loop 水平位移误差代理。

wind-only MLP 只输入 ERA5 10 m 风速分量 u10/v10，但仍继承冻结
core6 的六特征有效行掩码。因此两模型的 original_ID、子轨迹、
逐行样本、target 和随机种子一致，唯一的实验因素是波浪输入是否
进入 MLP。
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_heldout_trajectory_proxy import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    EXPECTED_SOURCE_SHA256,
    HORIZONS_HOURS,
    RANDOM_SEED,
    SOURCE_PATH,
    dataset_summary,
    endpoint_errors_for_episodes,
    load_heldout_data,
    predict_frozen_core6,
    relative_path,
    sha256_file,
    summarize_horizon,
    validate_point_metrics,
    window_group_indices,
)
from baseline import run_linear_baseline  # noqa: E402
from data_loader import PROJECT_ROOT, load_and_split_data  # noqa: E402
from evaluation import regression_metrics  # noqa: E402
from train_mlp import (  # noqa: E402
    ResidualMLP,
    evaluate_and_compare,
    plot_history,
    train,
)


STUDY_NAME = "wind_only_ablation_circular_mwd_v2"
WIND_ONLY_NAME = "wind_only_mlp"
CORE6_NAME = "frozen_core6_mlp"
WIND_ONLY_FEATURES = ["era5_u10", "era5_v10"]
CORE6_FEATURES = [
    "era5_u10",
    "era5_v10",
    "era5_swh",
    "era5_mwp",
    "era5_wave_dir_sin",
    "era5_wave_dir_cos",
]
FROZEN_CORE6_DIR = (
    PROJECT_ROOT
    / "trained_models"
    / "ablation_circular_mwd_v2_final"
    / "core_6"
)
FROZEN_SPLIT_MANIFEST = FROZEN_CORE6_DIR / "split_manifest.json"
FROZEN_METRICS = FROZEN_CORE6_DIR / "mlp_metrics.json"
ARTIFACT_DIR = PROJECT_ROOT / "trained_models" / STUDY_NAME / "wind_2"
RESULT_DIR = PROJECT_ROOT / "results" / STUDY_NAME
TRAINING_RESULT_DIR = RESULT_DIR / "wind_2"
LOG_DIR = PROJECT_ROOT / "logs"

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_output(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_clean_committed_code() -> str:
    status = _git_output("status", "--porcelain")
    if status:
        raise RuntimeError(
            "正式训练要求已提交且干净的代码树；当前发现: "
            f"{status.splitlines()[:8]}"
        )
    return _git_output("rev-parse", "HEAD")


def _setup_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / (
        f"{STUDY_NAME}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    return path


def _validate_formal_output_is_new() -> None:
    protected = [
        ARTIFACT_DIR / "best_mlp.pth",
        RESULT_DIR / "comparison.json",
        RESULT_DIR / "trajectory_proxy.json",
    ]
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(
            "正式 wind-only 产物已存在，拒绝隐式覆写: "
            f"{existing}"
        )


def validate_split_equivalence(
    wind_manifest: Mapping[str, Any],
    frozen_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """验证 wind-only 与冻结 core6 共享完全一致的样本支持。"""

    if list(wind_manifest["feature_columns"]) != WIND_ONLY_FEATURES:
        raise ValueError("wind-only feature_columns 异常。")
    if (
        list(wind_manifest.get("row_validity_feature_columns", []))
        != CORE6_FEATURES
    ):
        raise ValueError("wind-only 未继承 core6 有效行掩码。")
    if list(frozen_manifest["feature_columns"]) != CORE6_FEATURES:
        raise ValueError("冻结 core6 feature_columns 异常。")

    shared_fields = (
        "source_file",
        "source_size_bytes",
        "group_column",
        "random_seed",
        "target_ratios",
        "sample_mode",
        "target_columns",
    )
    for field in shared_fields:
        if wind_manifest.get(field) != frozen_manifest.get(field):
            raise ValueError(
                f"wind-only/core6 manifest 的 {field} 不一致。"
            )
    if wind_manifest["sample_mode"]:
        raise ValueError("正式 wind-only 实验不允许 sample_mode。")

    split_summary: dict[str, Any] = {}
    for split_name in ("train", "val", "test"):
        wind = wind_manifest["splits"][split_name]
        frozen = frozen_manifest["splits"][split_name]
        for field in (
            "n_original_ids",
            "n_segments",
            "n_samples",
            "original_ids",
        ):
            if wind[field] != frozen[field]:
                raise ValueError(
                    f"{split_name}.{field} 在 wind-only/core6 间不一致。"
                )
        split_summary[split_name] = {
            key: int(wind[key])
            for key in ("n_original_ids", "n_segments", "n_samples")
        }
    return {
        "status": "passed",
        "identical_original_ids": True,
        "identical_segments": True,
        "identical_rows": True,
        "identical_target_definition": True,
        "splits": split_summary,
    }


def _maximum_metric_difference(
    first: Mapping[str, float],
    second: Mapping[str, float],
) -> float:
    return max(
        abs(float(first[key]) - float(second[key]))
        for key in ("r2_u", "r2_v", "r2_joint", "rmse", "mae")
    )


def build_offline_comparison(
    wind_metrics: Mapping[str, Any],
    frozen_metrics: Mapping[str, Any],
    split_validation: Mapping[str, Any],
    training_code_git_commit: str,
) -> dict[str, Any]:
    """汇总逐时 test 指标，core6 相对 wind-only 的改善为正。"""

    linear_difference = _maximum_metric_difference(
        wind_metrics["linear_baseline_test"],
        frozen_metrics["linear_baseline_test"],
    )
    if linear_difference > 1e-10:
        raise ValueError(
            "wind-only 与 core6 的线性基准未完全重现，"
            f"max diff={linear_difference:.3e}"
        )

    wind = wind_metrics["test"]
    core = frozen_metrics["test"]
    delta = {
        "core6_minus_wind_only_r2_u": float(core["r2_u"] - wind["r2_u"]),
        "core6_minus_wind_only_r2_v": float(core["r2_v"] - wind["r2_v"]),
        "core6_minus_wind_only_r2_joint": float(
            core["r2_joint"] - wind["r2_joint"]
        ),
        "wind_only_minus_core6_rmse_m_s": float(
            wind["rmse"] - core["rmse"]
        ),
        "wind_only_minus_core6_mae_m_s": float(
            wind["mae"] - core["mae"]
        ),
        "core6_rmse_reduction_vs_wind_only_percent": float(
            (wind["rmse"] - core["rmse"]) / wind["rmse"] * 100.0
        ),
        "core6_mae_reduction_vs_wind_only_percent": float(
            (wind["mae"] - core["mae"]) / wind["mae"] * 100.0
        ),
    }
    return {
        "schema_version": 1,
        "study": STUDY_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_git_commit": training_code_git_commit,
        "experimental_factor": (
            "wave inputs included (core6) versus removed (wind-only); "
            "all sample, target, split, seed, architecture body, optimizer, "
            "scheduler and early-stopping settings held fixed"
        ),
        "feature_sets": {
            WIND_ONLY_NAME: WIND_ONLY_FEATURES,
            CORE6_NAME: CORE6_FEATURES,
        },
        "split_validation": dict(split_validation),
        "linear_baseline_replay_max_absolute_difference": linear_difference,
        "models": {
            WIND_ONLY_NAME: {
                "parameter_count": wind_metrics["checkpoint"][
                    "parameter_count"
                ],
                "best_epoch": wind_metrics["checkpoint"]["best_epoch"],
                "validation_loss": wind_metrics["checkpoint"][
                    "validation_loss"
                ],
                "validation_r2_joint": wind_metrics["checkpoint"][
                    "validation_r2_joint"
                ],
                "test": dict(wind),
            },
            CORE6_NAME: {
                "parameter_count": frozen_metrics["checkpoint"][
                    "parameter_count"
                ],
                "best_epoch": frozen_metrics["checkpoint"]["best_epoch"],
                "validation_loss": frozen_metrics["checkpoint"][
                    "validation_loss"
                ],
                "validation_r2_joint": frozen_metrics["checkpoint"][
                    "validation_r2_joint"
                ],
                "test": dict(core),
            },
        },
        "wave_input_gain": delta,
    }


def write_offline_outputs(comparison: Mapping[str, Any]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fieldnames = [
        "model",
        "n_features",
        "parameter_count",
        "best_epoch",
        "validation_r2_joint",
        "test_r2_u",
        "test_r2_v",
        "test_r2_joint",
        "test_rmse_m_s",
        "test_mae_m_s",
    ]
    with (RESULT_DIR / "comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for name in (WIND_ONLY_NAME, CORE6_NAME):
            model = comparison["models"][name]
            writer.writerow(
                {
                    "model": name,
                    "n_features": len(comparison["feature_sets"][name]),
                    "parameter_count": model["parameter_count"],
                    "best_epoch": model["best_epoch"],
                    "validation_r2_joint": model["validation_r2_joint"],
                    "test_r2_u": model["test"]["r2_u"],
                    "test_r2_v": model["test"]["r2_v"],
                    "test_r2_joint": model["test"]["r2_joint"],
                    "test_rmse_m_s": model["test"]["rmse"],
                    "test_mae_m_s": model["test"]["mae"],
                }
            )


def run_training(
    *,
    training_code_git_commit: str,
    source_sha256: str,
    log_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """训练 wind-only MLP，并与冻结 core6 做严格逐时比较。"""

    logger.info("读取完整 circular-MWD v2 数据并构建严格消融...")
    splits = load_and_split_data(
        filepath=SOURCE_PATH,
        sample_mode=False,
        artifact_dir=ARTIFACT_DIR,
        feature_cols=WIND_ONLY_FEATURES,
        row_validity_feature_cols=CORE6_FEATURES,
    )
    wind_manifest = _load_json(ARTIFACT_DIR / "split_manifest.json")
    frozen_manifest = _load_json(FROZEN_SPLIT_MANIFEST)
    split_validation = validate_split_equivalence(
        wind_manifest,
        frozen_manifest,
    )
    logger.info(
        "样本等价性验证通过: train/val/test rows=%s/%s/%s",
        split_validation["splits"]["train"]["n_samples"],
        split_validation["splits"]["val"]["n_samples"],
        split_validation["splits"]["test"]["n_samples"],
    )

    logger.info("重放线性风飘系数基准...")
    baseline = run_linear_baseline(splits)
    logger.info("训练 wind-only MLP...")
    train_result = train(splits)
    wind_metrics = evaluate_and_compare(train_result, baseline)
    plot_history(
        train_result["history"],
        ARTIFACT_DIR,
        result_dir=TRAINING_RESULT_DIR,
    )

    frozen_metrics = _load_json(FROZEN_METRICS)
    comparison = build_offline_comparison(
        wind_metrics,
        frozen_metrics,
        split_validation,
        training_code_git_commit,
    )
    write_offline_outputs(comparison)
    experiment = {
        "schema_version": 1,
        "study": STUDY_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_git_commit": training_code_git_commit,
        "source": {
            "path": relative_path(SOURCE_PATH),
            "size_bytes": SOURCE_PATH.stat().st_size,
            "sha256": source_sha256,
        },
        "feature_set": WIND_ONLY_NAME,
        "features": WIND_ONLY_FEATURES,
        "row_validity_feature_columns": CORE6_FEATURES,
        "random_seed": RANDOM_SEED,
        "sample_mode": False,
        "log_path": relative_path(log_path),
        "artifact_dir": relative_path(ARTIFACT_DIR),
        "result_dir": relative_path(TRAINING_RESULT_DIR),
        "split_validation": split_validation,
        "metrics": wind_metrics,
    }
    TRAINING_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (TRAINING_RESULT_DIR / "experiment.json").write_text(
        json.dumps(experiment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 在进入 held-out 轨迹代理分析前释放完整训练张量。
    del train_result, baseline, splits
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return comparison, experiment


def predict_wind_only(
    core6: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """从已保存 scaler/checkpoint 独立重放 wind-only test 预测。"""

    config_path = ARTIFACT_DIR / "model_config.json"
    checkpoint_path = ARTIFACT_DIR / "best_mlp.pth"
    scaler_path = ARTIFACT_DIR / "x_scaler.pkl"
    config = _load_json(config_path)
    if config["feature_columns"] != WIND_ONLY_FEATURES:
        raise ValueError("wind-only model_config 特征顺序异常。")
    if config["architecture"][0] != len(WIND_ONLY_FEATURES):
        raise ValueError("wind-only model_config 输入尺寸异常。")

    scaler = joblib.load(scaler_path)
    if int(scaler.n_features_in_) != len(WIND_ONLY_FEATURES):
        raise ValueError("wind-only scaler 输入尺寸异常。")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualMLP(input_size=len(WIND_ONLY_FEATURES)).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    prediction = np.empty((len(core6), 2), dtype=np.float32)
    raw_wind = np.asarray(core6[:, :2], dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(raw_wind), batch_size):
            stop = min(start + batch_size, len(raw_wind))
            standardized = np.ascontiguousarray(
                scaler.transform(raw_wind[start:stop]).astype(
                    np.float32,
                    copy=False,
                )
            )
            tensor = torch.from_numpy(standardized).to(device)
            prediction[start:stop] = model(tensor).cpu().numpy()
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, {
        "checkpoint": {
            "path": relative_path(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
        },
        "scaler": {
            "path": relative_path(scaler_path),
            "sha256": sha256_file(scaler_path),
        },
        "model_config": {
            "path": relative_path(config_path),
            "sha256": sha256_file(config_path),
        },
        "features": WIND_ONLY_FEATURES,
        "inference_device": str(device),
    }


def _metric_with_bias(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    metrics = regression_metrics(target, prediction)
    bias = (
        np.asarray(prediction, dtype=np.float64)
        - np.asarray(target, dtype=np.float64)
    ).mean(axis=0)
    return {
        **metrics,
        "bias_u": float(bias[0]),
        "bias_v": float(bias[1]),
    }


def validate_wind_point_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """核对独立重放与训练阶段冻结的 test 指标。"""

    values = _metric_with_bias(target, prediction)
    reference = _load_json(ARTIFACT_DIR / "mlp_metrics.json")["test"]
    maximum = _maximum_metric_difference(values, reference)
    if maximum > 2e-6:
        raise ValueError(
            "wind-only 独立重放未通过: "
            f"max metric diff={maximum:.3e}"
        )
    return {
        **values,
        "maximum_absolute_difference_from_training_record": maximum,
    }


def write_proxy_csv(
    path: Path,
    horizons: Mapping[str, Mapping[str, Any]],
) -> None:
    fields = [
        "horizon_hours",
        "model",
        "n_windows",
        "n_original_ids",
        "pooled_window_median_km",
        "pooled_window_p90_km",
        "equal_id_mean_of_window_medians_km",
        "equal_id_mean_of_window_p90_km",
        "median_improvement_percent_vs_wind_only",
        "median_improvement_ci95_low",
        "median_improvement_ci95_high",
        "median_id_wins",
        "median_id_ties",
        "median_id_losses",
        "median_id_win_rate",
        "p90_improvement_percent_vs_wind_only",
        "p90_improvement_ci95_low",
        "p90_improvement_ci95_high",
        "p90_id_win_rate",
        "window_win_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for horizon in sorted(horizons, key=int):
            summary = horizons[horizon]
            comparison = summary["comparisons_vs_wind_only_mlp"].get(
                CORE6_NAME
            )
            for name in (WIND_ONLY_NAME, CORE6_NAME):
                row = {
                    "horizon_hours": int(horizon),
                    "model": name,
                    "n_windows": summary["n_windows"],
                    "n_original_ids": summary["n_original_ids"],
                    **summary["models"][name],
                }
                if name == CORE6_NAME and comparison is not None:
                    median = comparison["median"]
                    p90 = comparison["p90"]
                    median_ci = median["paired_bootstrap"][
                        "relative_improvement_percent_ci95"
                    ]
                    p90_ci = p90["paired_bootstrap"][
                        "relative_improvement_percent_ci95"
                    ]
                    row.update(
                        {
                            "median_improvement_percent_vs_wind_only": median[
                                "equal_id_relative_improvement_percent"
                            ],
                            "median_improvement_ci95_low": median_ci[0],
                            "median_improvement_ci95_high": median_ci[1],
                            "median_id_wins": median["id_wins"],
                            "median_id_ties": median["id_ties"],
                            "median_id_losses": median["id_losses"],
                            "median_id_win_rate": median["id_win_rate"],
                            "p90_improvement_percent_vs_wind_only": p90[
                                "equal_id_relative_improvement_percent"
                            ],
                            "p90_improvement_ci95_low": p90_ci[0],
                            "p90_improvement_ci95_high": p90_ci[1],
                            "p90_id_win_rate": p90["id_win_rate"],
                            "window_win_rate": comparison["window_level"][
                                "win_rate"
                            ],
                        }
                    )
                writer.writerow(row)


def run_trajectory_proxy(
    *,
    training_code_git_commit: str,
    source_sha256: str,
    inference_batch_size: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """比较 core6 和 wind-only 的多时长水平位移误差代理。"""

    data = load_heldout_data(verify_source_sha256=False)
    logger.info("独立重放 wind-only MLP...")
    wind_prediction, wind_provenance = predict_wind_only(
        data.core6,
        batch_size=inference_batch_size,
    )
    logger.info("重放冻结 core6 ONNX...")
    core_prediction, core_provenance = predict_frozen_core6(
        data.core6,
        batch_size=inference_batch_size,
    )
    point_metrics = {
        WIND_ONLY_NAME: validate_wind_point_metrics(
            data.target,
            wind_prediction,
        ),
        CORE6_NAME: validate_point_metrics(
            data.target,
            {CORE6_NAME: core_prediction},
        )[CORE6_NAME],
    }

    groups = window_group_indices(data.episodes, data.test_original_ids)
    endpoints: dict[str, dict[int, np.ndarray]] = {}
    for name, prediction in (
        (WIND_ONLY_NAME, wind_prediction),
        (CORE6_NAME, core_prediction),
    ):
        logger.info("累计 %s 的多时长位移误差...", name)
        endpoints[name] = endpoint_errors_for_episodes(
            np.asarray(prediction, dtype=np.float64)
            - np.asarray(data.target, dtype=np.float64),
            data.episodes,
        )

    horizon_summaries: dict[str, Any] = {}
    for horizon in HORIZONS_HOURS:
        horizon_summaries[str(horizon)] = summarize_horizon(
            horizon,
            groups[horizon],
            {
                WIND_ONLY_NAME: endpoints[WIND_ONLY_NAME][horizon],
                CORE6_NAME: endpoints[CORE6_NAME][horizon],
            },
            data.test_original_ids,
            bootstrap_replicates=bootstrap_replicates,
            seed=RANDOM_SEED + horizon,
            baseline_name=WIND_ONLY_NAME,
        )

    payload = {
        "schema_version": 1,
        "study": STUDY_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_git_commit": training_code_git_commit,
        "scientific_scope": {
            "primary_comparison": (
                "wind_only_mlp versus frozen_core6_mlp on identical "
                "held-out rows"
            ),
            "open_loop_displacement_error_proxy": True,
            "recursive_fortran_oilspill_trajectory": False,
        },
        "protocol": {
            "split": "frozen global original_ID held-out test",
            "continuity": (
                "split within source trajectory wherever adjacent retained "
                "timestamps differ from exactly 3600 s"
            ),
            "windows": (
                "non-overlapping windows anchored at continuous episode "
                "start; incomplete tail discarded"
            ),
            "displacement_proxy": (
                "Euclidean norm of sum((predicted residual - observed "
                "residual) * 3600 seconds) / 1000, in km"
            ),
            "primary_aggregation": (
                "median endpoint error within original_ID, followed by "
                "equal-ID mean"
            ),
            "tail_aggregation": (
                "P90 endpoint error within original_ID, followed by "
                "equal-ID mean"
            ),
            "uncertainty": "paired bootstrap over original_ID",
            "bootstrap_replicates": bootstrap_replicates,
            "horizons_hours": list(HORIZONS_HOURS),
        },
        "lineage": {
            "source": {
                "path": relative_path(SOURCE_PATH),
                "size_bytes": SOURCE_PATH.stat().st_size,
                "sha256": source_sha256,
            },
            "frozen_split_manifest": {
                "path": relative_path(FROZEN_SPLIT_MANIFEST),
                "sha256": sha256_file(FROZEN_SPLIT_MANIFEST),
            },
            "models": {
                WIND_ONLY_NAME: wind_provenance,
                CORE6_NAME: core_provenance,
            },
        },
        "dataset": dataset_summary(data, groups),
        "point_metrics": point_metrics,
        "trajectory_proxy": {"horizons": horizon_summaries},
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "trajectory_proxy.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_proxy_csv(
        RESULT_DIR / "trajectory_proxy_summary.csv",
        horizon_summaries,
    )
    logger.info("多时长位移代理分析已保存。")
    return payload


def plot_combined_result(
    comparison: Mapping[str, Any],
    proxy: Mapping[str, Any],
) -> None:
    """绘制逐时与多时长结果的紧凑四面板摘要。"""

    colors = {WIND_ONLY_NAME: "#7A7F87", CORE6_NAME: "#2166AC"}
    labels = {WIND_ONLY_NAME: "Wind-only MLP", CORE6_NAME: "Wind+wave MLP"}
    names = [WIND_ONLY_NAME, CORE6_NAME]
    horizons = list(HORIZONS_HOURS)
    summaries = proxy["trajectory_proxy"]["horizons"]

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4))
    axes[0, 0].bar(
        [labels[name] for name in names],
        [comparison["models"][name]["test"]["r2_joint"] for name in names],
        color=[colors[name] for name in names],
    )
    axes[0, 0].set_ylabel("Test joint R²")
    axes[0, 0].set_title("(a) Hourly residual prediction")
    axes[0, 0].grid(axis="y", alpha=0.25)

    axes[0, 1].bar(
        [labels[name] for name in names],
        [comparison["models"][name]["test"]["rmse"] for name in names],
        color=[colors[name] for name in names],
    )
    axes[0, 1].set_ylabel("Test RMSE (m s⁻¹)")
    axes[0, 1].set_title("(b) Hourly residual error")
    axes[0, 1].grid(axis="y", alpha=0.25)

    for name in names:
        axes[1, 0].plot(
            horizons,
            [
                summaries[str(horizon)]["models"][name][
                    "equal_id_mean_of_window_medians_km"
                ]
                for horizon in horizons
            ],
            marker="o",
            linewidth=2,
            label=labels[name],
            color=colors[name],
        )
    axes[1, 0].set(
        xlabel="Integration horizon (h)",
        ylabel="Equal-ID endpoint error (km)",
        title="(c) Median horizontal-displacement proxy",
        xticks=horizons,
    )
    axes[1, 0].grid(alpha=0.25)
    axes[1, 0].legend(frameon=False)

    comparisons = [
        summaries[str(horizon)]["comparisons_vs_wind_only_mlp"][CORE6_NAME][
            "median"
        ]
        for horizon in horizons
    ]
    values = [
        item["equal_id_relative_improvement_percent"]
        for item in comparisons
    ]
    lows = [
        item["paired_bootstrap"]["relative_improvement_percent_ci95"][0]
        for item in comparisons
    ]
    highs = [
        item["paired_bootstrap"]["relative_improvement_percent_ci95"][1]
        for item in comparisons
    ]
    axes[1, 1].plot(
        horizons,
        values,
        marker="o",
        linewidth=2,
        color=colors[CORE6_NAME],
    )
    axes[1, 1].fill_between(
        horizons,
        lows,
        highs,
        color=colors[CORE6_NAME],
        alpha=0.18,
    )
    axes[1, 1].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1, 1].set(
        xlabel="Integration horizon (h)",
        ylabel="Improvement vs wind-only (%)",
        title="(d) Wave-input gain (paired 95% CI)",
        xticks=horizons,
    )
    axes[1, 1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        RESULT_DIR / "wind_wave_ablation.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def write_readme(
    comparison: Mapping[str, Any],
    proxy: Mapping[str, Any],
) -> None:
    wind = comparison["models"][WIND_ONLY_NAME]
    core = comparison["models"][CORE6_NAME]
    gain = comparison["wave_input_gain"]
    summaries = proxy["trajectory_proxy"]["horizons"]
    rows = []
    for horizon in HORIZONS_HOURS:
        summary = summaries[str(horizon)]
        comparison_row = summary["comparisons_vs_wind_only_mlp"][CORE6_NAME][
            "median"
        ]
        ci = comparison_row["paired_bootstrap"][
            "relative_improvement_percent_ci95"
        ]
        rows.append(
            "| {h} | {wind:.3f} | {core:.3f} | {gain:.3f}% | "
            "[{low:.3f}%, {high:.3f}%] | {wins}/{losses} |".format(
                h=horizon,
                wind=summary["models"][WIND_ONLY_NAME][
                    "equal_id_mean_of_window_medians_km"
                ],
                core=summary["models"][CORE6_NAME][
                    "equal_id_mean_of_window_medians_km"
                ],
                gain=comparison_row[
                    "equal_id_relative_improvement_percent"
                ],
                low=ci[0],
                high=ci[1],
                wins=comparison_row["id_wins"],
                losses=comparison_row["id_losses"],
            )
        )

    all_positive = all(
        summaries[str(horizon)]["comparisons_vs_wind_only_mlp"][CORE6_NAME][
            "median"
        ]["equal_id_relative_improvement_percent"]
        > 0.0
        for horizon in HORIZONS_HOURS
    )
    significant_horizons = [
        horizon
        for horizon in HORIZONS_HOURS
        if summaries[str(horizon)]["comparisons_vs_wind_only_mlp"][CORE6_NAME][
            "median"
        ]["paired_bootstrap"]["relative_improvement_percent_ci95"][0]
        > 0.0
    ]
    direction_text = (
        "各积分时长的点估计均支持加入波浪输入"
        if all_positive
        else "不同积分时长的改善方向并不完全一致"
    )
    significance_text = (
        "95% 配对区间下界高于零的时长为 "
        + ", ".join(f"{value} h" for value in significant_horizons)
        if significant_horizons
        else "所有时长的 95% 配对区间均覆盖零"
    )

    text = f"""# Wind-only 严格消融（circular-MWD v2）

## 设计

本实验以冻结 global core6 MLP 为对照。wind-only MLP 只输入
`era5_u10` 和 `era5_v10`；core6 输入这两项风速及 `era5_swh`、
`era5_mwp`、`era5_wave_dir_sin`和 `era5_wave_dir_cos`。两组共享
core6 有效行掩码、`original_ID` 70/15/15 切分、seed=42、target、
网络主体、优化器、学习率调度及早停规则。清单核对确认
train/validation/test 的 ID、子轨迹和逐行样本完全一致。

## 逐时残差速度结果

| 模型 | 输入数 | 参数量 | Best epoch | Test joint R² | RMSE (m/s) | MAE (m/s) |
|---|---:|---:|---:|---:|---:|---:|
| Wind-only MLP | 2 | {wind['parameter_count']:,} | {wind['best_epoch']} | {wind['test']['r2_joint']:.6f} | {wind['test']['rmse']:.6f} | {wind['test']['mae']:.6f} |
| Wind+wave core6 MLP | 6 | {core['parameter_count']:,} | {core['best_epoch']} | {core['test']['r2_joint']:.6f} | {core['test']['rmse']:.6f} | {core['test']['mae']:.6f} |

加入波浪输入后，test joint R² 变化
`{gain['core6_minus_wind_only_r2_joint']:+.6f}`，RMSE 变化
`{-gain['wind_only_minus_core6_rmse_m_s']:+.6f} m/s`（相对 wind-only 的降低幅度
`{gain['core6_rmse_reduction_vs_wind_only_percent']:+.3f}%`）。

## 多时长水平位移误差代理

主统计量为每个 `original_ID` 内的窗口端点误差中位数，随后对 ID
等权平均；区间由 {proxy['protocol']['bootstrap_replicates']:,} 次 ID 单元配对
bootstrap 给出。

| 时长 (h) | Wind-only (km) | Wind+wave (km) | 改善率 | 95% CI | ID 胜/负 |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

{direction_text}；{significance_text}。该结果与逐时指标共同用于判断
波浪输入是否增加了可识别的水平残差速度信息。

## 解释边界

位移指标沿观测位置与真实逐时 forcing 做 open-loop 累积，可表征残差
速度误差的水平时间积分。它不是递归更新位置的 Fortran 溢油轨迹，
也不包含空间偏离后的 forcing 变化、随机扩散、岸线和风化过程。
因此，该实验直接支持的是波浪变量对水平残差速度预测的边际信息，
完整溢油水平输运的改善仍由案例模拟结果界定。
"""
    (RESULT_DIR / "README.md").write_text(text, encoding="utf-8")


def write_run_manifest(
    *,
    training_code_git_commit: str,
    source_sha256: str,
) -> None:
    files = [
        "best_mlp.pth",
        "x_scaler.pkl",
        "split_manifest.json",
        "model_config.json",
        "mlp_metrics.json",
        "linear_baseline.joblib",
        "linear_baseline_metrics.json",
        "training_history.json",
    ]
    artifacts = {
        name: {
            "path": relative_path(ARTIFACT_DIR / name),
            "sha256": sha256_file(ARTIFACT_DIR / name),
        }
        for name in files
    }
    payload = {
        "schema_version": 1,
        "study": STUDY_NAME,
        "status": (
            "frozen_diagnostic"
            if source_sha256 == EXPECTED_SOURCE_SHA256
            else "debug_unverified"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_git_commit": training_code_git_commit,
        "source": {
            "path": relative_path(SOURCE_PATH),
            "size_bytes": SOURCE_PATH.stat().st_size,
            "sha256": source_sha256,
        },
        "features": WIND_ONLY_FEATURES,
        "row_validity_feature_columns": CORE6_FEATURES,
        "split_group": "original_ID",
        "random_seed": RANDOM_SEED,
        "active_deployment_changed": False,
        "artifacts": artifacts,
    }
    (ARTIFACT_DIR / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_all(
    *,
    inference_batch_size: int,
    bootstrap_replicates: int,
    skip_source_sha256: bool,
) -> None:
    _validate_formal_output_is_new()
    training_code_git_commit = _require_clean_committed_code()
    log_path = _setup_logging()
    logger.info("=" * 72)
    logger.info("正式 wind-only 严格消融 | code=%s", training_code_git_commit)
    logger.info("输入: %s", WIND_ONLY_FEATURES)
    logger.info("有效行掩码: %s", CORE6_FEATURES)
    logger.info("=" * 72)

    if skip_source_sha256:
        source_sha256 = "not_recomputed"
        logger.warning("跳过了源数据 SHA256；不得将本次结果标记为正式。")
    else:
        logger.info("验证源数据 SHA256...")
        source_sha256 = sha256_file(SOURCE_PATH)
        if source_sha256 != EXPECTED_SOURCE_SHA256:
            raise ValueError(
                f"源数据 SHA256 异常: {source_sha256}"
            )

    comparison, _ = run_training(
        training_code_git_commit=training_code_git_commit,
        source_sha256=source_sha256,
        log_path=log_path,
    )
    # 先冻结训练产物，使下游分析若中断仍可独立恢复。
    write_run_manifest(
        training_code_git_commit=training_code_git_commit,
        source_sha256=source_sha256,
    )
    proxy = run_trajectory_proxy(
        training_code_git_commit=training_code_git_commit,
        source_sha256=source_sha256,
        inference_batch_size=inference_batch_size,
        bootstrap_replicates=bootstrap_replicates,
    )
    plot_combined_result(comparison, proxy)
    write_readme(comparison, proxy)
    logger.info("正式 wind-only 消融已完成: %s", RESULT_DIR)


def resume_analysis(
    *,
    inference_batch_size: int,
    bootstrap_replicates: int,
) -> None:
    """训练已成功后，允许单独恢复下游分析。"""

    run_manifest = _load_json(ARTIFACT_DIR / "run_manifest.json")
    comparison = _load_json(RESULT_DIR / "comparison.json")
    proxy = run_trajectory_proxy(
        training_code_git_commit=run_manifest["training_code_git_commit"],
        source_sha256=run_manifest["source"]["sha256"],
        inference_batch_size=inference_batch_size,
        bootstrap_replicates=bootstrap_replicates,
    )
    plot_combined_result(comparison, proxy)
    write_readme(comparison, proxy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行 circular-MWD v2 wind-only 严格消融"
    )
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="只恢复已训练模型的位移代理分析",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=131_072,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--skip-source-sha256",
        action="store_true",
        help="仅用于调试；正式运行不得使用",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size 必须大于 0。")
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates 必须大于 0。")
    if args.analysis_only:
        _setup_logging()
        resume_analysis(
            inference_batch_size=args.inference_batch_size,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        return
    run_all(
        inference_batch_size=args.inference_batch_size,
        bootstrap_replicates=args.bootstrap_replicates,
        skip_source_sha256=args.skip_source_sha256,
    )


if __name__ == "__main__":
    main()
