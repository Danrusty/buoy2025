"""在同一 original_ID 切分上运行 core_6 与 full_9 最小消融实验。"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from baseline import run_linear_baseline
from data_loader import FEATURE_COLS, PROJECT_ROOT, load_and_split_data
from train_mlp import evaluate_and_compare, plot_history, train


ABLATION_NAME = "ablation_study"
ARTIFACT_ROOT = PROJECT_ROOT / "trained_models" / ABLATION_NAME
RESULT_ROOT = PROJECT_ROOT / "results" / ABLATION_NAME
LOG_ROOT = PROJECT_ROOT / "logs"

FEATURE_SETS = {
    "core_6": [
        "era5_u10",
        "era5_v10",
        "era5_swh",
        "era5_mwp",
        "era5_wave_dir_sin",
        "era5_wave_dir_cos",
    ],
    "full_9": FEATURE_COLS.copy(),
}

logger = logging.getLogger(__name__)


def _relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _setup_logging(feature_set: str) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / (
        f"ablation_{feature_set}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    root = logging.getLogger()
    root.handlers.clear()
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


def run_experiment(
    feature_set_name: str,
    sample_mode: bool,
    sample_size: int,
) -> dict:
    features = FEATURE_SETS[feature_set_name]
    artifact_dir = ARTIFACT_ROOT / feature_set_name
    result_dir = RESULT_ROOT / feature_set_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    log_path = _setup_logging(feature_set_name)
    logger.info("=" * 68)
    logger.info("Ablation %s | %d features", feature_set_name, len(features))
    logger.info("特征顺序: %s", features)
    logger.info("模式: %s", "sample" if sample_mode else "full")
    logger.info("=" * 68)

    splits = load_and_split_data(
        sample_mode=sample_mode,
        sample_size=sample_size,
        artifact_dir=artifact_dir,
        feature_cols=features,
    )
    baseline_result = run_linear_baseline(splits)
    train_result = train(splits)
    mlp_metrics = evaluate_and_compare(train_result, baseline_result)
    curve_path, history_path = plot_history(
        train_result["history"],
        artifact_dir,
        result_dir=result_dir,
    )

    metadata = {
        "ablation_name": ABLATION_NAME,
        "feature_set": feature_set_name,
        "features": features,
        "n_features": len(features),
        "sample_mode": sample_mode,
        "split_group": "original_ID",
        "random_seed": 42,
        "log_path": _relative_path(log_path),
        "artifact_dir": _relative_path(artifact_dir),
        "result_dir": _relative_path(result_dir),
        "curve_path": _relative_path(curve_path),
        "history_path": _relative_path(history_path),
        "metrics": mlp_metrics,
    }
    (result_dir / "experiment.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("实验元数据已保存: %s", result_dir / "experiment.json")
    return metadata


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_identical_splits(manifests: dict[str, dict]) -> None:
    reference = manifests["core_6"]
    for split_name in ("train", "val", "test"):
        reference_ids = reference["splits"][split_name]["original_ids"]
        candidate_ids = manifests["full_9"]["splits"][split_name]["original_ids"]
        if reference_ids != candidate_ids:
            raise RuntimeError(
                f"{split_name} original_ID 在 core_6/full_9 间不一致。"
            )
        for key in ("n_original_ids", "n_segments", "n_samples"):
            if (
                reference["splits"][split_name][key]
                != manifests["full_9"]["splits"][split_name][key]
            ):
                raise RuntimeError(
                    f"{split_name}.{key} 在 core_6/full_9 间不一致。"
                )


def summarize_if_complete() -> Path | None:
    required = {
        name: {
            "mlp": ARTIFACT_ROOT / name / "mlp_metrics.json",
            "baseline": ARTIFACT_ROOT / name / "linear_baseline_metrics.json",
            "manifest": ARTIFACT_ROOT / name / "split_manifest.json",
            "config": ARTIFACT_ROOT / name / "model_config.json",
            "history": ARTIFACT_ROOT / name / "training_history.json",
        }
        for name in FEATURE_SETS
    }
    if not all(
        path.is_file()
        for files in required.values()
        for path in files.values()
    ):
        logger.info("两组完整产物尚未齐备，暂不生成总表。")
        return None

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    loaded = {
        name: {kind: _load_json(path) for kind, path in files.items()}
        for name, files in required.items()
    }
    manifests = {name: values["manifest"] for name, values in loaded.items()}
    _assert_identical_splits(manifests)
    if any(manifest["sample_mode"] for manifest in manifests.values()):
        raise RuntimeError("检测到 sample_mode 产物，不能生成论文消融总表。")

    rows = []
    for name, features in FEATURE_SETS.items():
        mlp = loaded[name]["mlp"]
        baseline = loaded[name]["baseline"]
        rows.append(
            {
                "feature_set": name,
                "n_features": len(features),
                "parameter_count": mlp["checkpoint"]["parameter_count"],
                "best_epoch": mlp["checkpoint"]["best_epoch"],
                "val_loss": mlp["checkpoint"]["validation_loss"],
                "val_r2_joint": mlp["checkpoint"]["validation_r2_joint"],
                "test_r2_u": mlp["test"]["r2_u"],
                "test_r2_v": mlp["test"]["r2_v"],
                "test_r2_joint": mlp["test"]["r2_joint"],
                "test_rmse": mlp["test"]["rmse"],
                "test_mae": mlp["test"]["mae"],
                "linear_test_r2_joint": baseline["test"]["r2_joint"],
                "linear_test_rmse": baseline["test"]["rmse"],
            }
        )

    by_name = {row["feature_set"]: row for row in rows}
    core = by_name["core_6"]
    full = by_name["full_9"]
    delta = {
        "full_9_minus_core_6_test_r2_joint": (
            full["test_r2_joint"] - core["test_r2_joint"]
        ),
        "full_9_minus_core_6_test_rmse": full["test_rmse"] - core["test_rmse"],
        "full_9_minus_core_6_test_mae": full["test_mae"] - core["test_mae"],
        "full_9_minus_core_6_val_r2_joint": (
            full["val_r2_joint"] - core["val_r2_joint"]
        ),
    }

    comparison = {
        "study": ABLATION_NAME,
        "split_method": "original_ID",
        "random_seed": manifests["core_6"]["random_seed"],
        "joint_r2_definition": "mean(R2_residual_u, R2_residual_v), float64",
        "feature_sets": FEATURE_SETS,
        "split_stats": {
            name: {
                key: manifests["core_6"]["splits"][name][key]
                for key in ("n_original_ids", "n_segments", "n_samples")
            }
            for name in ("train", "val", "test")
        },
        "results": rows,
        "delta": delta,
    }
    comparison_path = RESULT_ROOT / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    csv_path = RESULT_ROOT / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    report_path = RESULT_ROOT / "README.md"
    better = "full_9" if full["test_rmse"] < core["test_rmse"] else "core_6"
    report_path.write_text(
        f"""# Minimal Feature Ablation: core_6 vs full_9

Both models use the same physical-buoy `original_ID` split, random seed,
network body, optimizer, scheduler, early stopping rule, and test set. Only
the input feature set changes.

## Feature Sets

- `core_6`: `{", ".join(FEATURE_SETS["core_6"])}`
- `full_9`: `{", ".join(FEATURE_SETS["full_9"])}`

## Results

| Feature set | Params | Best epoch | Val joint R2 | Test R2 u | Test R2 v | Test joint R2 | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core_6 | {core["parameter_count"]:,} | {core["best_epoch"]} | {core["val_r2_joint"]:.6f} | {core["test_r2_u"]:.6f} | {core["test_r2_v"]:.6f} | {core["test_r2_joint"]:.6f} | {core["test_rmse"]:.6f} | {core["test_mae"]:.6f} |
| full_9 | {full["parameter_count"]:,} | {full["best_epoch"]} | {full["val_r2_joint"]:.6f} | {full["test_r2_u"]:.6f} | {full["test_r2_v"]:.6f} | {full["test_r2_joint"]:.6f} | {full["test_rmse"]:.6f} | {full["test_mae"]:.6f} |

`full_9 - core_6`:

- Test joint R2: {delta["full_9_minus_core_6_test_r2_joint"]:+.6f}
- Test RMSE: {delta["full_9_minus_core_6_test_rmse"]:+.6f} m/s
- Test MAE: {delta["full_9_minus_core_6_test_mae"]:+.6f} m/s
- Validation joint R2: {delta["full_9_minus_core_6_val_r2_joint"]:+.6f}

Lower test RMSE in this controlled run: **{better}**.

Joint R2 is calculated in float64 as the arithmetic mean of the separate
`residual_u` and `residual_v` R2 values.
""",
        encoding="utf-8",
    )
    _plot_comparison(loaded)
    logger.info("消融总表已保存: %s", report_path)
    return report_path


def _plot_comparison(loaded: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {"core_6": "tab:blue", "full_9": "tab:orange"}
    for name in FEATURE_SETS:
        history = loaded[name]["history"]
        epochs = range(1, len(history["val_loss"]) + 1)
        axes[0].plot(
            epochs,
            history["val_loss"],
            label=name,
            color=colors[name],
        )
        axes[1].plot(
            epochs,
            history["val_r2"],
            label=name,
            color=colors[name],
        )

    axes[0].set_title("Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[1].set_title("Validation Joint R2")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("R2")
    for axis in axes:
        axis.grid(True)
        axis.legend()
    fig.tight_layout()
    fig.savefig(
        RESULT_ROOT / "ablation_comparison.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 WDF 最小特征消融")
    parser.add_argument(
        "--feature-set",
        choices=[*FEATURE_SETS, "summarize"],
        required=True,
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    if args.feature_set == "summarize":
        logging.basicConfig(level=logging.INFO)
        summarize_if_complete()
        return

    run_experiment(
        args.feature_set,
        sample_mode=not args.full,
        sample_size=args.sample_size,
    )
    summarize_if_complete()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
