"""汇总 global MLP/XGBoost × core6/lat7 四格析因实验。

本脚本只读取三个实验分支中已提交的冻结指标和当前工作区的预测缓存，先验证
数据与冻结基准血缘一致，再计算主效应、交互项、逐 original_ID 直接胜率和
纬度带差异。它不训练模型，也不改变任何 selection lock。
"""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..data_loader import PROJECT_ROOT
from .global_factorial import (
    DATA_MANIFEST_PATH,
    FROZEN_REPLAY_PATH,
    load_cached_split,
    original_id_win_rate,
    sha256_file,
)


RESULT_DIR = PROJECT_ROOT / "results" / "global_factorial_v1"
CELL_SPECS = {
    "A": {
        "label": "Frozen MLP core6",
        "model": "MLP",
        "features": "core6",
        "branch": "master",
        "metrics_path": None,
        "manifest_path": None,
    },
    "B": {
        "label": "MLP lat7",
        "model": "MLP",
        "features": "core6 + sin(latitude)",
        "branch": "wdf_global_mlp_lat7_v1",
        "metrics_path": "trained_models/global_mlp_lat7_v1/metrics.json",
        "manifest_path": (
            "trained_models/global_mlp_lat7_v1/run_manifest.json"
        ),
    },
    "C": {
        "label": "XGBoost core6",
        "model": "XGBoost",
        "features": "core6",
        "branch": "wdf_global_xgb_core6_v1",
        "metrics_path": "trained_models/global_xgb_core6_v1/metrics.json",
        "manifest_path": (
            "trained_models/global_xgb_core6_v1/run_manifest.json"
        ),
    },
    "D": {
        "label": "XGBoost lat7",
        "model": "XGBoost",
        "features": "core6 + sin(latitude)",
        "branch": "wdf_global_xgb_lat7_v1",
        "metrics_path": "trained_models/global_xgb_lat7_v1/metrics.json",
        "manifest_path": (
            "trained_models/global_xgb_lat7_v1/run_manifest.json"
        ),
    },
}


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _git_json(branch: str, path: str) -> dict[str, Any]:
    return json.loads(_git("show", f"{branch}:{path}"))


def _git_object_size(branch: str, path: str) -> int:
    return int(_git("cat-file", "-s", f"{branch}:{path}").strip())


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def factorial_effects(
    cells: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """按标准 2×2 定义计算纬度、模型类型与交互效应。"""
    required = {"A", "B", "C", "D"}
    if set(cells) != required:
        raise ValueError(f"实验格必须恰好为 {sorted(required)}")
    metric_names = set(cells["A"])
    if any(set(cells[name]) != metric_names for name in required):
        raise ValueError("四格指标集合不一致")

    effects: dict[str, dict[str, float]] = {}
    for metric in sorted(metric_names):
        a = float(cells["A"][metric])
        b = float(cells["B"][metric])
        c = float(cells["C"][metric])
        d = float(cells["D"][metric])
        effects[metric] = {
            "latitude_under_mlp_B_minus_A": b - a,
            "model_without_latitude_C_minus_A": c - a,
            "latitude_under_xgboost_D_minus_C": d - c,
            "model_with_latitude_D_minus_B": d - b,
            "interaction_D_minus_C_minus_B_plus_A": d - c - b + a,
        }
    return effects


def _load_cells() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    frozen_replay = json.loads(
        FROZEN_REPLAY_PATH.read_text(encoding="utf-8")
    )
    cells: dict[str, dict[str, Any]] = {
        "A": frozen_replay["replayed_evaluation"]
    }
    manifests: dict[str, dict[str, Any]] = {}
    for name in ("B", "C", "D"):
        spec = CELL_SPECS[name]
        branch = str(spec["branch"])
        metrics = _git_json(branch, str(spec["metrics_path"]))
        manifest = _git_json(branch, str(spec["manifest_path"]))
        cells[name] = metrics["test"]
        manifests[name] = manifest
    return cells, manifests


def _validate_lineage(
    cells: dict[str, dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
) -> dict[str, str]:
    data_sha256 = sha256_file(DATA_MANIFEST_PATH)
    frozen_sha256 = sha256_file(FROZEN_REPLAY_PATH)
    for name, manifest in manifests.items():
        lineage = manifest["lineage"]
        if lineage["data_manifest_sha256"] != data_sha256:
            raise ValueError(f"{name} data manifest SHA256 不一致")
        if lineage["frozen_reference_replay_sha256"] != frozen_sha256:
            raise ValueError(f"{name} frozen replay SHA256 不一致")

        branch = str(CELL_SPECS[name]["branch"])
        metrics = _git_json(
            branch,
            str(CELL_SPECS[name]["metrics_path"]),
        )
        reference = metrics["frozen_global_reference"]["row_weighted"]
        actual = cells["A"]["row_weighted"]
        for key in ("r2_u", "r2_v", "r2_joint", "rmse", "mae"):
            if abs(float(reference[key]) - float(actual[key])) > 1e-12:
                raise ValueError(
                    f"{name} 的冻结基准 {key} 与 A 不一致"
                )

    branch_commits = {
        name: _git("rev-parse", str(spec["branch"])).strip()
        for name, spec in CELL_SPECS.items()
    }
    return {
        "data_manifest_sha256": data_sha256,
        "frozen_replay_sha256": frozen_sha256,
        **{
            f"cell_{name}_branch_commit": commit
            for name, commit in branch_commits.items()
        },
    }


def _model_size_bytes(
    manifests: dict[str, dict[str, Any]],
) -> dict[str, int]:
    frozen_onnx = (
        PROJECT_ROOT
        / "deployment"
        / "releases"
        / "wdf_core6_circular_mwd_v2"
        / "wdf_drifter.onnx"
    )
    sizes = {"A": frozen_onnx.stat().st_size}
    b_manifest = manifests["B"]
    sizes["B"] = _git_object_size(
        str(CELL_SPECS["B"]["branch"]),
        b_manifest["artifacts"]["checkpoint"]["path"],
    )
    for name in ("C", "D"):
        branch = str(CELL_SPECS[name]["branch"])
        sizes[name] = sum(
            _git_object_size(branch, record["path"])
            for record in manifests[name]["artifacts"]["models"]
        )
    return sizes


def _prediction_paths(
    manifests: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    frozen = json.loads(
        FROZEN_REPLAY_PATH.read_text(encoding="utf-8")
    )["prediction_cache"]
    records = {"A": frozen}
    for name, manifest in manifests.items():
        records[name] = manifest["artifacts"]["prediction_cache"]
    return records


def _load_predictions(
    manifests: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for name, record in _prediction_paths(manifests).items():
        path = PROJECT_ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(
                f"{name} prediction cache 不存在：{path}"
            )
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"{name} prediction cache SHA256 不一致")
        values = np.load(path, mmap_mode="r")
        if list(values.shape) != record["shape"]:
            raise ValueError(f"{name} prediction cache shape 不一致")
        predictions[name] = values
    return predictions


def _head_to_head(
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    test = load_cached_split("test")
    comparisons = {
        "B_vs_A": ("B", "A"),
        "C_vs_A": ("C", "A"),
        "D_vs_A": ("D", "A"),
        "D_vs_B": ("D", "B"),
        "D_vs_C": ("D", "C"),
    }
    return {
        label: original_id_win_rate(
            test["target"],
            predictions[candidate],
            predictions[reference],
            test["group_index"],
        )
        for label, (candidate, reference) in comparisons.items()
    }


def _latitude_effects(
    cells: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    by_cell = {
        name: {
            (
                float(record["lower_degrees"]),
                float(record["upper_degrees"]),
            ): record
            for record in evaluation["latitude_bands"]
        }
        for name, evaluation in cells.items()
    }
    bands = sorted(by_cell["A"])
    output = []
    for lower, upper in bands:
        output.append(
            {
                "lower_degrees": lower,
                "upper_degrees": upper,
                "n_samples": int(
                    by_cell["A"][(lower, upper)]["n_samples"]
                ),
                **{
                    f"{name}_r2_joint": float(
                        by_cell[name][(lower, upper)]["r2_joint"]
                    )
                    for name in ("A", "B", "C", "D")
                },
                "B_minus_A_r2_joint": float(
                    by_cell["B"][(lower, upper)]["r2_joint"]
                    - by_cell["A"][(lower, upper)]["r2_joint"]
                ),
                "D_minus_C_r2_joint": float(
                    by_cell["D"][(lower, upper)]["r2_joint"]
                    - by_cell["C"][(lower, upper)]["r2_joint"]
                ),
                "D_minus_A_r2_joint": float(
                    by_cell["D"][(lower, upper)]["r2_joint"]
                    - by_cell["A"][(lower, upper)]["r2_joint"]
                ),
            }
        )
    return output


def _rows(
    cells: dict[str, dict[str, Any]],
    model_sizes: dict[str, int],
) -> list[dict[str, Any]]:
    rows = []
    for name in ("A", "B", "C", "D"):
        row = cells[name]["row_weighted"]
        macro = cells[name]["macro_original_id"]
        wins = cells[name].get("vs_frozen_global_by_original_id")
        spec = CELL_SPECS[name]
        rows.append(
            {
                "cell": name,
                "label": spec["label"],
                "model": spec["model"],
                "features": spec["features"],
                "r2_u": row["r2_u"],
                "r2_v": row["r2_v"],
                "r2_joint": row["r2_joint"],
                "rmse": row["rmse"],
                "mae": row["mae"],
                "macro_id_r2_joint": macro["r2_joint"],
                "macro_id_rmse": macro["rmse"],
                "win_rate_vs_A": (
                    None if wins is None else wins["win_rate"]
                ),
                "model_size_bytes": model_sizes[name],
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(
    path: Path,
    rows: list[dict[str, Any]],
    effects: dict[str, dict[str, float]],
    head_to_head: dict[str, Any],
) -> None:
    by_name = {row["cell"]: row for row in rows}
    table_lines = [
        "| 格 | 模型/输入 | joint R² | RMSE | MAE | macro-ID R² | "
        "macro-ID RMSE | vs A 胜率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("A", "B", "C", "D"):
        row = by_name[name]
        win_rate = (
            "—"
            if row["win_rate_vs_A"] is None
            else f"{100 * row['win_rate_vs_A']:.1f}%"
        )
        table_lines.append(
            f"| {name} | {row['label']} | {row['r2_joint']:.6f} | "
            f"{row['rmse']:.6f} | {row['mae']:.6f} | "
            f"{row['macro_id_r2_joint']:.6f} | "
            f"{row['macro_id_rmse']:.6f} | {win_rate} |"
        )

    r2 = effects["r2_joint"]
    rmse = effects["rmse"]
    content = f"""# Global 纬度信息 × 模型类型析因实验结果

## 结论

四格实验均使用同一数据、同一 `original_ID` 切分和同一 test 行。D
（XGBoost + `sin(latitude)`）是数值最优，但相对冻结 A 的 joint R² 只提高
{by_name['D']['r2_joint'] - by_name['A']['r2_joint']:+.6f}，RMSE 只降低
{100 * (by_name['A']['rmse'] - by_name['D']['rmse']) / by_name['A']['rmse']:.3f}%。
它没有达到预先规定的 +0.01 joint R² / 1% RMSE 门槛，因此不进入 Windows
推理效率评价，也不替换冻结 global MLP。

## 四格结果

{chr(10).join(table_lines)}

## 析因效应

- MLP 下加入纬度（B − A）：joint R² {r2['latitude_under_mlp_B_minus_A']:+.6f}，
  RMSE {rmse['latitude_under_mlp_B_minus_A']:+.6f} m/s。
- 不含纬度时改用 XGBoost（C − A）：joint R²
  {r2['model_without_latitude_C_minus_A']:+.6f}，RMSE
  {rmse['model_without_latitude_C_minus_A']:+.6f} m/s。
- XGBoost 下加入纬度（D − C）：joint R²
  {r2['latitude_under_xgboost_D_minus_C']:+.6f}，RMSE
  {rmse['latitude_under_xgboost_D_minus_C']:+.6f} m/s。
- 含纬度时 XGBoost 相对 MLP（D − B）：joint R²
  {r2['model_with_latitude_D_minus_B']:+.6f}，RMSE
  {rmse['model_with_latitude_D_minus_B']:+.6f} m/s。
- 交互项：joint R²
  {r2['interaction_D_minus_C_minus_B_plus_A']:+.6f}。

## original_ID 直接比较

- D 相对 B：{head_to_head['D_vs_B']['wins']} 胜 /
  {head_to_head['D_vs_B']['ties']} 平 /
  {head_to_head['D_vs_B']['losses']} 负，胜率
  {100 * head_to_head['D_vs_B']['win_rate']:.1f}%。
- D 相对 C：{head_to_head['D_vs_C']['wins']} 胜 /
  {head_to_head['D_vs_C']['ties']} 平 /
  {head_to_head['D_vs_C']['losses']} 负，胜率
  {100 * head_to_head['D_vs_C']['win_rate']:.1f}%。

## 科学解释

纬度在 MLP 和 XGBoost 下都带来小幅、方向一致的改善；单独替换 XGBoost
没有收益。模型类型与纬度存在很弱的正交互，但幅度不足以解释当前 global
baseline 的主要误差。因此现有证据更支持“缺少粗粒度空间信息只解释少量误差，
模型类型不是主要瓶颈”，而不是立即替换冻结模型或进入部署。
"""
    path.write_text(content, encoding="utf-8")


def run() -> dict[str, Any]:
    """生成经过血缘与预测缓存校验的统一比较产物。"""
    cells, manifests = _load_cells()
    lineage = _validate_lineage(cells, manifests)
    model_sizes = _model_size_bytes(manifests)
    rows = _rows(cells, model_sizes)
    numeric_cells = {
        name: {
            "r2_joint": float(cells[name]["row_weighted"]["r2_joint"]),
            "rmse": float(cells[name]["row_weighted"]["rmse"]),
            "mae": float(cells[name]["row_weighted"]["mae"]),
            "macro_id_r2_joint": float(
                cells[name]["macro_original_id"]["r2_joint"]
            ),
            "macro_id_rmse": float(
                cells[name]["macro_original_id"]["rmse"]
            ),
        }
        for name in ("A", "B", "C", "D")
    }
    effects = factorial_effects(numeric_cells)
    predictions = _load_predictions(manifests)
    head_to_head = _head_to_head(predictions)
    latitude_effects = _latitude_effects(cells)

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "global_factorial_v1",
        "lineage_validation": {
            "status": "passed",
            **lineage,
        },
        "cells": rows,
        "factorial_effects": effects,
        "head_to_head_original_id_rmse": head_to_head,
        "latitude_bands": latitude_effects,
        "decision": {
            "best_cell": "D",
            "xgboost_efficiency_gate_passed": False,
            "run_windows_efficiency_benchmark": False,
            "replace_frozen_global_model": False,
        },
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _json_dump(RESULT_DIR / "comparison.json", payload)
    _write_csv(RESULT_DIR / "comparison.csv", rows)
    _write_readme(
        RESULT_DIR / "README.md",
        rows,
        effects,
        head_to_head,
    )
    return payload


if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                "best_cell": result["decision"]["best_cell"],
                "cells": {
                    row["cell"]: {
                        "r2_joint": row["r2_joint"],
                        "rmse": row["rmse"],
                    }
                    for row in result["cells"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
