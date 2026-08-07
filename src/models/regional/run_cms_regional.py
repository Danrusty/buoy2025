"""准备、训练并评价统一 China Marginal Seas core6 MLP。"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ..data_loader import PROJECT_ROOT, load_and_split_data
from ..training.baseline import run_linear_baseline
from ..training.train_mlp import (
    evaluate_and_compare,
    plot_history,
    train,
)
from .cms_regional import (
    CIRCULAR_SOURCE_PATH,
    CORE_FEATURES,
    FILTERED_DATA_PATH,
    FILTERED_DIAGNOSTICS_PATH,
    GLOBAL_ONNX_PATH,
    GLOBAL_SPLIT_MANIFEST_PATH,
    MASK_SOURCE_PATH,
    MODEL_VERSION,
    evaluate_cms_models,
    prepare_cms_dataset,
    write_regional_linear_analysis,
)


ARTIFACT_DIR = PROJECT_ROOT / "trained_models" / MODEL_VERSION
RESULT_DIR = PROJECT_ROOT / "results" / MODEL_VERSION
LOG_DIR = PROJECT_ROOT / "logs"
GLOBAL_MODEL_CONFIG_PATH = (
    PROJECT_ROOT
    / "trained_models"
    / "ablation_circular_mwd_v2_final"
    / "core_6"
    / "model_config.json"
)

logger = logging.getLogger(__name__)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _setup_logging() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        f"{MODEL_VERSION}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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


def _verify_frozen_training_contract(
    regional_config_path: Path,
    output_path: Path,
) -> dict:
    global_config = json.loads(
        GLOBAL_MODEL_CONFIG_PATH.read_text(encoding="utf-8")
    )
    regional_config = json.loads(
        regional_config_path.read_text(encoding="utf-8")
    )
    frozen_keys = [
        "architecture",
        "batch_norm",
        "dropout",
        "feature_columns",
        "target_columns",
        "batch_size",
        "max_epochs",
        "early_stopping_patience",
        "checkpoint_monitor",
        "learning_rate",
        "minimum_learning_rate",
        "optimizer",
        "weight_decay",
        "scheduler",
        "random_seed",
    ]
    comparisons = {
        key: {
            "global": global_config.get(key),
            "regional": regional_config.get(key),
            "identical": global_config.get(key) == regional_config.get(key),
        }
        for key in frozen_keys
    }
    all_identical = all(value["identical"] for value in comparisons.values())
    result = {
        "all_frozen_fields_identical": all_identical,
        "global_config": _relative(GLOBAL_MODEL_CONFIG_PATH),
        "regional_config": _relative(regional_config_path),
        "comparisons": comparisons,
        "explicitly_excluded_changes": [
            "fixed 0.03 correction",
            "regional target correction",
            "LSTM or sequence model",
            "new input features",
            "hyperparameter search",
            "separate subregion models",
        ],
    }
    _write_json(output_path, result)
    if not all_identical:
        changed = [
            key for key, value in comparisons.items()
            if not value["identical"]
        ]
        raise RuntimeError(f"冻结训练契约发生变化: {changed}")
    return result


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def _write_result_report(
    *,
    statistics: dict,
    linear_analysis: dict,
    mlp_metrics: dict,
    regional_evaluation: dict,
    split_provenance: dict,
    contract_check: dict,
) -> Path:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    subset_rows = []
    for name, values in regional_evaluation["subsets"].items():
        regional = values["regional_mlp"]
        global_model = values["frozen_global_mlp"]
        subset_rows.append(
            "| {name} | {n_samples} | {n_ids} | {r_r2} | {r_rmse} | "
            "{r_bu} | {r_bv} | {g_r2} | {g_rmse} |".format(
                name=name,
                n_samples=values["n_samples"],
                n_ids=values["n_original_ids"],
                r_r2=_fmt(regional["r2_joint"]),
                r_rmse=_fmt(regional["rmse"]),
                r_bu=_fmt(regional["bias_u"]),
                r_bv=_fmt(regional["bias_v"]),
                g_r2=_fmt(global_model["r2_joint"]),
                g_rmse=_fmt(global_model["rmse"]),
            )
        )

    split_stats = statistics["split"]["statistics"]
    overall = regional_evaluation["subsets"]["CMS_overall"]
    report_path = RESULT_DIR / "README.md"
    report_path.write_text(
        f"""# {MODEL_VERSION}

## Scope

One unified China Marginal Seas MLP was trained with the frozen global core6
target, features, network, loss, optimizer, scheduler, early stopping and
random-seed strategy. Only the row-level geographic training-data selection
and the leakage-free `original_ID` split population changed.

## Dataset

- CMS original IDs: {statistics['cms_dataset']['n_original_ids']}
- CMS hourly samples: {statistics['cms_dataset']['n_samples']}
- Continuous in-region episodes: {statistics['cms_dataset']['n_hourly_episodes']}
- BYS / ECS / NSCS membership rows:
  {statistics['region_membership_counts']['BYS']} /
  {statistics['region_membership_counts']['ECS']} /
  {statistics['region_membership_counts']['NSCS']}
- Split strategy: `{split_provenance['strategy']}`
- Train / val / test IDs:
  {split_stats['train']['n_original_ids']} /
  {split_stats['val']['n_original_ids']} /
  {split_stats['test']['n_original_ids']}
- Train / val / test rows:
  {split_stats['train']['n_samples']} /
  {split_stats['val']['n_samples']} /
  {split_stats['test']['n_samples']}
- Pairwise split ID intersections: 0

The supplied source contains no qualifying BYS rows. BYS metrics are therefore
reported as `N/A`; this model is a CMS-mask model whose observed support in this
dataset is ECS + NSCS.

## Regional linear baseline

- A matrix: `{json.dumps(linear_analysis['A_matrix'])}`
- Intercept: `{json.dumps(linear_analysis['intercept'])}`
- Effective WDF (`trace(A)/2`): {linear_analysis['effective_wdf']:.8f}
- Cross-wind coefficient: {linear_analysis['cross_wind_coefficient']:.8f}
- Test joint R2: {linear_analysis['test_metrics']['r2_joint']:.6f}
- Test RMSE: {linear_analysis['test_metrics']['rmse']:.6f} m/s

## MLP test metrics

- R2 u: {mlp_metrics['test']['r2_u']:.6f}
- R2 v: {mlp_metrics['test']['r2_v']:.6f}
- Joint R2: {mlp_metrics['test']['r2_joint']:.6f}
- RMSE: {mlp_metrics['test']['rmse']:.6f} m/s
- MAE: {mlp_metrics['test']['mae']:.6f} m/s
- Frozen global MLP on the identical CMS test rows:
  joint R2 {_fmt(overall['frozen_global_mlp']['r2_joint'])},
  RMSE {_fmt(overall['frozen_global_mlp']['rmse'])} m/s

## Test subsets

Bias is `mean(predicted residual - observed residual)`.

| Subset | Rows | IDs | Regional joint R2 | Regional RMSE | Bias u | Bias v | Global joint R2 | Global RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(subset_rows)}

## Frozen contract

All {len(contract_check['comparisons'])} checked training/interface fields are
identical to the frozen global core6 configuration.
""",
        encoding="utf-8",
    )
    return report_path


def run_full_pipeline(code_commit: str) -> dict:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("步骤 1/7: 逐行构建 CMS 数据并决定 original_ID 切分")
    prepared = prepare_cms_dataset(
        mask_source_path=MASK_SOURCE_PATH,
        circular_source_path=CIRCULAR_SOURCE_PATH,
        filtered_data_path=FILTERED_DATA_PATH,
        diagnostics_path=FILTERED_DIAGNOSTICS_PATH,
        artifact_dir=ARTIFACT_DIR,
        global_split_manifest_path=GLOBAL_SPLIT_MANIFEST_PATH,
        code_commit=code_commit,
    )
    statistics = prepared["statistics"]
    logger.info(
        "CMS: %d IDs / %d 点 / BYS=%d ECS=%d NSCS=%d",
        statistics["cms_dataset"]["n_original_ids"],
        statistics["cms_dataset"]["n_samples"],
        statistics["region_membership_counts"]["BYS"],
        statistics["region_membership_counts"]["ECS"],
        statistics["region_membership_counts"]["NSCS"],
    )
    logger.info("切分策略: %s", prepared["split_provenance"]["strategy"])

    logger.info("步骤 2/7: 使用冻结 core6 特征加载 train/val/test")
    splits = load_and_split_data(
        filepath=prepared["filtered_data_path"],
        random_seed=42,
        sample_mode=False,
        artifact_dir=ARTIFACT_DIR,
        feature_cols=CORE_FEATURES,
        predefined_id_splits=prepared["id_splits"],
        split_provenance=prepared["split_provenance"],
    )

    logger.info("步骤 3/7: 拟合 regional 线性基准")
    baseline_result = run_linear_baseline(splits)
    linear_analysis_path = ARTIFACT_DIR / "regional_linear_analysis.json"
    linear_analysis = write_regional_linear_analysis(
        splits,
        baseline_result,
        linear_analysis_path,
    )

    logger.info("步骤 4/7: 按冻结超参数训练单一 CMS MLP")
    train_result = train(splits)

    logger.info("步骤 5/7: 评价 CMS MLP 与线性基准")
    mlp_metrics = evaluate_and_compare(train_result, baseline_result)
    curve_path, history_path = plot_history(
        train_result["history"],
        ARTIFACT_DIR,
        result_dir=RESULT_DIR,
    )

    logger.info("步骤 6/7: 在相同 CMS test 行对比冻结 global MLP")
    regional_evaluation_path = ARTIFACT_DIR / "regional_evaluation.json"
    regional_evaluation = evaluate_cms_models(
        filtered_data_path=prepared["filtered_data_path"],
        id_splits=prepared["id_splits"],
        regional_model=train_result["model"],
        regional_scaler=splits["x_scaler"],
        output_path=regional_evaluation_path,
        global_onnx_path=GLOBAL_ONNX_PATH,
    )

    logger.info("步骤 7/7: 核验冻结训练契约并写入实验报告")
    contract_path = ARTIFACT_DIR / "frozen_contract_check.json"
    contract_check = _verify_frozen_training_contract(
        ARTIFACT_DIR / "model_config.json",
        contract_path,
    )
    report_path = _write_result_report(
        statistics=statistics,
        linear_analysis=linear_analysis,
        mlp_metrics=mlp_metrics,
        regional_evaluation=regional_evaluation,
        split_provenance=prepared["split_provenance"],
        contract_check=contract_check,
    )

    experiment = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_code_git_commit": code_commit,
        "python_environment": {
            "distribution": "Miniforge3",
            "conda_environment": "buoy-drifter",
            "torch": torch.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
        },
        "data": {
            "mask_source": _relative(MASK_SOURCE_PATH),
            "circular_feature_source": _relative(CIRCULAR_SOURCE_PATH),
            "filtered_dataset": _relative(FILTERED_DATA_PATH),
            "diagnostics": _relative(FILTERED_DIAGNOSTICS_PATH),
            "statistics": _relative(prepared["statistics_path"]),
            "region_mask": _relative(prepared["region_mask_path"]),
            "region_row_index": _relative(prepared["row_index_path"]),
        },
        "split_manifest": _relative(ARTIFACT_DIR / "split_manifest.json"),
        "linear_analysis": _relative(linear_analysis_path),
        "mlp_metrics": _relative(ARTIFACT_DIR / "mlp_metrics.json"),
        "regional_evaluation": _relative(regional_evaluation_path),
        "frozen_contract_check": _relative(contract_path),
        "checkpoint": _relative(train_result["best_model_path"]),
        "scaler": _relative(ARTIFACT_DIR / "x_scaler.pkl"),
        "training_curve": _relative(curve_path),
        "training_history": _relative(history_path),
        "report": _relative(report_path),
    }
    experiment_path = RESULT_DIR / "experiment.json"
    _write_json(experiment_path, experiment)
    logger.info("CMS regional 全流程完成: %s", report_path)
    return experiment


def run_prepare_only(code_commit: str) -> dict:
    prepared = prepare_cms_dataset(
        mask_source_path=MASK_SOURCE_PATH,
        circular_source_path=CIRCULAR_SOURCE_PATH,
        filtered_data_path=FILTERED_DATA_PATH,
        diagnostics_path=FILTERED_DIAGNOSTICS_PATH,
        artifact_dir=ARTIFACT_DIR,
        global_split_manifest_path=GLOBAL_SPLIT_MANIFEST_PATH,
        code_commit=code_commit,
    )
    logger.info(
        "CMS 数据准备完成: %s",
        prepared["filtered_data_path"],
    )
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in prepared.items()
        if key != "statistics"
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练统一 China Marginal Seas regional core6 MLP"
    )
    parser.add_argument(
        "--phase",
        choices=["prepare", "all"],
        default="all",
    )
    parser.add_argument(
        "--code-commit",
        default="unknown",
        help="本次数据准备与训练所用的 Git commit。",
    )
    args = parser.parse_args()

    log_path = _setup_logging()
    logger.info("=" * 72)
    logger.info("CMS regional 模型: %s", MODEL_VERSION)
    logger.info("Miniforge 环境: buoy-drifter")
    logger.info("日志: %s", log_path)
    logger.info("阶段: %s", args.phase)
    logger.info("=" * 72)

    if args.phase == "prepare":
        run_prepare_only(args.code_commit)
    else:
        run_full_pipeline(args.code_commit)


if __name__ == "__main__":
    main()
