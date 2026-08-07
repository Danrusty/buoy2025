"""汇总 frozen core6、MLP lat7、MLP lat9 与既有 XGBoost 结果。

脚本先验证所有模型继承同一冻结数据与 test 预测缓存，再计算 lat9 相对
lat7 的逐行指标、逐 original_ID 胜率、纬度带和经度带差异。XGBoost
core6/lat7 只作为已完成实验的上下文读取，本轮不训练 XGBoost lat9。
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
from ..evaluation import regression_metrics
from .global_factorial import (
    DATA_MANIFEST_PATH,
    FROZEN_REPLAY_PATH,
    load_cached_split,
    original_id_win_rate,
    sha256_file,
)
from .global_longitude import (
    COS_LONGITUDE_FEATURE,
    LONGITUDE_CACHE_MANIFEST_PATH,
    SIN_LONGITUDE_FEATURE,
    load_longitude_split,
)


RESULT_DIR = PROJECT_ROOT / "results" / "global_mlp_spatial_v1"
CELL_ORDER = ("A", "B", "E", "C", "D")
CELL_SPECS = {
    "A": {
        "label": "Frozen MLP core6",
        "model": "MLP",
        "features": "core6",
        "n_features": 6,
        "branch": "master",
        "metrics_path": None,
        "manifest_path": None,
    },
    "B": {
        "label": "MLP lat7",
        "model": "MLP",
        "features": "core6 + sin(lat)",
        "n_features": 7,
        "branch": None,
        "metrics_path": "trained_models/global_mlp_lat7_v1/metrics.json",
        "manifest_path": (
            "trained_models/global_mlp_lat7_v1/run_manifest.json"
        ),
    },
    "E": {
        "label": "MLP lat9",
        "model": "MLP",
        "features": "core6 + sin(lat) + sin/cos(lon)",
        "n_features": 9,
        "branch": None,
        "metrics_path": "trained_models/global_mlp_lat9_v1/metrics.json",
        "manifest_path": (
            "trained_models/global_mlp_lat9_v1/run_manifest.json"
        ),
    },
    "C": {
        "label": "XGBoost core6",
        "model": "XGBoost",
        "features": "core6",
        "n_features": 6,
        "branch": "wdf_global_xgb_core6_v1",
        "metrics_path": "trained_models/global_xgb_core6_v1/metrics.json",
        "manifest_path": (
            "trained_models/global_xgb_core6_v1/run_manifest.json"
        ),
    },
    "D": {
        "label": "XGBoost lat7",
        "model": "XGBoost",
        "features": "core6 + sin(lat)",
        "n_features": 7,
        "branch": "wdf_global_xgb_lat7_v1",
        "metrics_path": "trained_models/global_xgb_lat7_v1/metrics.json",
        "manifest_path": (
            "trained_models/global_xgb_lat7_v1/run_manifest.json"
        ),
    },
}
ROW_METRICS = (
    "r2_u",
    "r2_v",
    "r2_joint",
    "rmse",
    "mae",
)
LONGITUDE_EDGES = np.asarray(
    [-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0],
    dtype=np.float64,
)


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _read_json(cell: str, path: str) -> dict[str, Any]:
    branch = CELL_SPECS[cell]["branch"]
    if branch is None:
        return json.loads(
            (PROJECT_ROOT / path).read_text(encoding="utf-8")
        )
    return json.loads(_git("show", f"{branch}:{path}"))


def _artifact_commit(cell: str, path: str | None) -> str:
    branch = CELL_SPECS[cell]["branch"]
    if path is None:
        return _git("rev-parse", str(branch)).strip()
    reference = "HEAD" if branch is None else str(branch)
    return _git(
        "log",
        "-1",
        "--format=%H",
        reference,
        "--",
        path,
    ).strip()


def _json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def metric_deltas(
    candidate: dict[str, float],
    reference: dict[str, float],
) -> dict[str, float]:
    """返回候选减参考的同名指标差。"""
    if set(candidate) != set(reference):
        raise ValueError("候选与参考指标集合不一致")
    return {
        key: float(candidate[key] - reference[key])
        for key in candidate
    }


def longitude_band_comparison(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    sin_longitude: np.ndarray,
    cos_longitude: np.ndarray,
    edges: np.ndarray = LONGITUDE_EDGES,
) -> list[dict[str, Any]]:
    """按循环编码还原经度，并报告互不重叠经度带指标。"""
    true = np.asarray(y_true)
    sin_value = np.asarray(sin_longitude, dtype=np.float64)
    cos_value = np.asarray(cos_longitude, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if sin_value.shape != (len(true),):
        raise ValueError("sin_longitude shape 不正确")
    if cos_value.shape != (len(true),):
        raise ValueError("cos_longitude shape 不正确")
    if len(edges) < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("经度带边界必须严格递增")

    longitude = np.rad2deg(np.arctan2(sin_value, cos_value))
    output: list[dict[str, Any]] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        if index == len(edges) - 2:
            mask = (longitude >= lower) & (longitude <= upper)
            interval = "closed"
        else:
            mask = (longitude >= lower) & (longitude < upper)
            interval = "left_closed"
        if not np.any(mask):
            continue
        row: dict[str, Any] = {
            "lower_degrees": float(lower),
            "upper_degrees": float(upper),
            "interval": interval,
            "n_samples": int(mask.sum()),
        }
        for cell, prediction in predictions.items():
            metrics = regression_metrics(true[mask], prediction[mask])
            row[f"{cell}_r2_joint"] = metrics["r2_joint"]
            row[f"{cell}_rmse"] = metrics["rmse"]
        if {"B", "E"}.issubset(predictions):
            row["E_minus_B_r2_joint"] = float(
                row["E_r2_joint"] - row["B_r2_joint"]
            )
            row["E_minus_B_rmse"] = float(
                row["E_rmse"] - row["B_rmse"]
            )
        output.append(row)
    return output


def _load_cells() -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    frozen = json.loads(
        FROZEN_REPLAY_PATH.read_text(encoding="utf-8")
    )
    cells = {"A": frozen["replayed_evaluation"]}
    manifests: dict[str, dict[str, Any]] = {}
    for cell in ("B", "E", "C", "D"):
        spec = CELL_SPECS[cell]
        metrics = _read_json(cell, str(spec["metrics_path"]))
        manifest = _read_json(cell, str(spec["manifest_path"]))
        cells[cell] = metrics["test"]
        manifests[cell] = manifest
    return cells, manifests


def _validate_lineage(
    cells: dict[str, dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    data_sha256 = sha256_file(DATA_MANIFEST_PATH)
    frozen_sha256 = sha256_file(FROZEN_REPLAY_PATH)
    longitude_sha256 = sha256_file(LONGITUDE_CACHE_MANIFEST_PATH)
    for cell, manifest in manifests.items():
        lineage = manifest["lineage"]
        if lineage["data_manifest_sha256"] != data_sha256:
            raise ValueError(f"{cell} 基础数据清单 SHA256 不一致")
        if lineage["frozen_reference_replay_sha256"] != frozen_sha256:
            raise ValueError(f"{cell} frozen replay SHA256 不一致")
        if cell == "E":
            if (
                lineage["longitude_cache_manifest_sha256"]
                != longitude_sha256
            ):
                raise ValueError("E 经度补充清单 SHA256 不一致")

        metrics = _read_json(
            cell,
            str(CELL_SPECS[cell]["metrics_path"]),
        )
        reference = metrics["frozen_global_reference"]["row_weighted"]
        actual = cells["A"]["row_weighted"]
        for key in ROW_METRICS:
            if abs(float(reference[key]) - float(actual[key])) > 1e-12:
                raise ValueError(
                    f"{cell} 的冻结参考 {key} 与 A 不一致"
                )

    return {
        "status": "passed",
        "data_manifest_sha256": data_sha256,
        "frozen_replay_sha256": frozen_sha256,
        "longitude_cache_manifest_sha256": longitude_sha256,
        "artifact_commits": {
            cell: _artifact_commit(
                cell,
                CELL_SPECS[cell]["metrics_path"],
            )
            for cell in CELL_ORDER
        },
    }


def _prediction_records(
    manifests: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    frozen = json.loads(
        FROZEN_REPLAY_PATH.read_text(encoding="utf-8")
    )
    records = {"A": frozen["prediction_cache"]}
    records.update(
        {
            cell: manifest["artifacts"]["prediction_cache"]
            for cell, manifest in manifests.items()
        }
    )
    return records


def _load_predictions(
    manifests: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {}
    for cell, record in _prediction_records(manifests).items():
        path = PROJECT_ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(
                f"{cell} prediction cache 不存在：{path}"
            )
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"{cell} prediction cache SHA256 不一致")
        prediction = np.load(path, mmap_mode="r")
        if list(prediction.shape) != record["shape"]:
            raise ValueError(f"{cell} prediction cache shape 不一致")
        predictions[cell] = prediction
    return predictions


def _head_to_head(
    predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    test = load_cached_split("test")
    pairs = {
        "B_vs_A": ("B", "A"),
        "E_vs_A": ("E", "A"),
        "E_vs_B": ("E", "B"),
        "C_vs_A": ("C", "A"),
        "D_vs_A": ("D", "A"),
        "D_vs_B": ("D", "B"),
        "D_vs_E": ("D", "E"),
    }
    return {
        label: original_id_win_rate(
            test["target"],
            predictions[candidate],
            predictions[reference],
            test["group_index"],
        )
        for label, (candidate, reference) in pairs.items()
    }


def _latitude_bands(
    cells: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = ("A", "B", "E")
    indexed = {
        cell: {
            (
                float(record["lower_degrees"]),
                float(record["upper_degrees"]),
            ): record
            for record in cells[cell]["latitude_bands"]
        }
        for cell in selected
    }
    output = []
    for lower, upper in sorted(indexed["A"]):
        row = {
            "lower_degrees": lower,
            "upper_degrees": upper,
            "n_samples": int(
                indexed["A"][(lower, upper)]["n_samples"]
            ),
        }
        for cell in selected:
            record = indexed[cell][(lower, upper)]
            row[f"{cell}_r2_joint"] = float(record["r2_joint"])
            row[f"{cell}_rmse"] = float(record["rmse"])
        row["E_minus_B_r2_joint"] = (
            row["E_r2_joint"] - row["B_r2_joint"]
        )
        row["E_minus_B_rmse"] = row["E_rmse"] - row["B_rmse"]
        output.append(row)
    return output


def _rows(
    cells: dict[str, dict[str, Any]],
    head_to_head: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for cell in CELL_ORDER:
        spec = CELL_SPECS[cell]
        row = cells[cell]["row_weighted"]
        macro = cells[cell]["macro_original_id"]
        validation_r2 = None
        best_epoch = None
        if cell in ("B", "E"):
            metrics = _read_json(
                cell,
                str(spec["metrics_path"]),
            )
            validation_r2 = metrics["selection"][
                "validation_r2_joint"
            ]
            best_epoch = metrics["selection"]["best_epoch"]
        versus_a = (
            None
            if cell == "A"
            else head_to_head[f"{cell}_vs_A"]
        )
        rows.append(
            {
                "cell": cell,
                "label": spec["label"],
                "model": spec["model"],
                "features": spec["features"],
                "n_features": spec["n_features"],
                "validation_r2_joint": validation_r2,
                "best_epoch": best_epoch,
                **{key: float(row[key]) for key in ROW_METRICS},
                "macro_id_r2_joint": float(macro["r2_joint"]),
                "macro_id_rmse": float(macro["rmse"]),
                "win_rate_vs_A": (
                    None
                    if versus_a is None
                    else float(versus_a["win_rate"])
                ),
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
    deltas: dict[str, dict[str, float]],
    head_to_head: dict[str, Any],
    longitude_bands: list[dict[str, Any]],
) -> None:
    by_cell = {row["cell"]: row for row in rows}
    table = [
        "| 格 | 模型/输入 | val joint R² | test R²_u | test R²_v | "
        "test joint R² | RMSE | MAE | vs A ID 胜率 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in CELL_ORDER:
        row = by_cell[cell]
        validation = (
            "—"
            if row["validation_r2_joint"] is None
            else f"{row['validation_r2_joint']:.6f}"
        )
        win_rate = (
            "—"
            if row["win_rate_vs_A"] is None
            else f"{100 * row['win_rate_vs_A']:.1f}%"
        )
        table.append(
            f"| {cell} | {row['label']} | {validation} | "
            f"{row['r2_u']:.6f} | {row['r2_v']:.6f} | "
            f"{row['r2_joint']:.6f} | {row['rmse']:.6f} | "
            f"{row['mae']:.6f} | {win_rate} |"
        )

    e_vs_b = deltas["E_minus_B"]
    direct = head_to_head["E_vs_B"]
    band_lines = []
    for band in longitude_bands:
        band_lines.append(
            f"| [{band['lower_degrees']:.0f}, "
            f"{band['upper_degrees']:.0f}"
            f"{']' if band['interval'] == 'closed' else ')'} | "
            f"{band['n_samples']:,} | "
            f"{band['B_r2_joint']:.6f} | "
            f"{band['E_r2_joint']:.6f} | "
            f"{band['E_minus_B_r2_joint']:+.6f} |"
        )

    content = f"""# Global MLP 空间信息递增实验

## 结论

在 lat7 上继续加入经度循环编码没有改善独立 test。lat9 的 validation joint
R² 从 {by_cell['B']['validation_r2_joint']:.6f} 提高到
{by_cell['E']['validation_r2_joint']:.6f}，但相对 lat7 的 test joint R²
{e_vs_b['r2_joint']:+.6f}，RMSE {e_vs_b['rmse']:+.6f} m/s。
因此当前 global MLP 仍以 lat7 为较好的空间输入版本，不继续训练 XGBoost
lat9，也不改变 frozen global 部署模型。

## 全部可比结果

{chr(10).join(table)}

A/B/E 是本次 MLP 空间特征递增链；C/D 是此前已冻结的 XGBoost 上下文，
本轮没有重新训练它们。

## lat9 相对 lat7

- test R²_u：{e_vs_b['r2_u']:+.6f}；
- test R²_v：{e_vs_b['r2_v']:+.6f}；
- test joint R²：{e_vs_b['r2_joint']:+.6f}；
- test RMSE：{e_vs_b['rmse']:+.6f} m/s；
- test MAE：{e_vs_b['mae']:+.6f} m/s；
- 逐 original_ID：{direct['wins']} 胜 / {direct['ties']} 平 /
  {direct['losses']} 负，胜率 {100 * direct['win_rate']:.1f}%，
  平均 ID-RMSE 差 {direct['mean_rmse_difference']:+.6f} m/s。

## 经度带差异

| 经度带 | 样本数 | lat7 joint R² | lat9 joint R² | lat9 − lat7 |
|---|---:|---:|---:|---:|
{chr(10).join(band_lines)}

## 解释

经度特征的收益具有明显空间异质性：部分经度带改善，另一些经度带退化。
validation 提高、test 降低的组合提示，这组固定绝对经度编码可能吸收了
不同 split 的空间分布差异，但没有形成稳定的 global 泛化增益。这个结果
不能证明经度在物理上无关，只能说明当前数据切分、目标和固定 MLP 下，
直接追加经度正余弦不是有效的整体改进路径。
"""
    path.write_text(content, encoding="utf-8")


def run() -> dict[str, Any]:
    """生成经过血缘和预测缓存校验的五模型比较。"""
    cells, manifests = _load_cells()
    lineage = _validate_lineage(cells, manifests)
    predictions = _load_predictions(manifests)
    head_to_head = _head_to_head(predictions)
    rows = _rows(cells, head_to_head)
    numeric = {
        cell: {
            key: float(cells[cell]["row_weighted"][key])
            for key in ROW_METRICS
        }
        for cell in CELL_ORDER
    }
    deltas = {
        "B_minus_A": metric_deltas(numeric["B"], numeric["A"]),
        "E_minus_A": metric_deltas(numeric["E"], numeric["A"]),
        "E_minus_B": metric_deltas(numeric["E"], numeric["B"]),
        "C_minus_A": metric_deltas(numeric["C"], numeric["A"]),
        "D_minus_A": metric_deltas(numeric["D"], numeric["A"]),
        "D_minus_B": metric_deltas(numeric["D"], numeric["B"]),
        "D_minus_E": metric_deltas(numeric["D"], numeric["E"]),
    }

    test = load_cached_split("test")
    longitude = load_longitude_split("test")
    longitude_bands = longitude_band_comparison(
        test["target"],
        {
            cell: predictions[cell]
            for cell in ("A", "B", "E")
        },
        longitude[SIN_LONGITUDE_FEATURE],
        longitude[COS_LONGITUDE_FEATURE],
    )
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "global_mlp_spatial_v1",
        "lineage_validation": lineage,
        "cells": rows,
        "deltas": deltas,
        "head_to_head_original_id_rmse": head_to_head,
        "latitude_bands": _latitude_bands(cells),
        "longitude_bands": longitude_bands,
        "decision": {
            "best_mlp_spatial_cell": "B",
            "best_available_numeric_cell": "D",
            "longitude_improves_independent_test": False,
            "train_xgboost_lat9": False,
            "replace_frozen_global_model": False,
        },
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    _json_dump(RESULT_DIR / "comparison.json", payload)
    _write_csv(RESULT_DIR / "comparison.csv", rows)
    _write_readme(
        RESULT_DIR / "README.md",
        rows,
        deltas,
        head_to_head,
        longitude_bands,
    )
    return payload


if __name__ == "__main__":
    result = run()
    print(
        json.dumps(
            {
                "best_mlp_spatial_cell": result["decision"][
                    "best_mlp_spatial_cell"
                ],
                "E_minus_B": result["deltas"]["E_minus_B"],
                "E_vs_B_original_id": result[
                    "head_to_head_original_id_rmse"
                ]["E_vs_B"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
