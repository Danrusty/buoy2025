"""冻结并打包 CMS 网络重构实验的 Windows ONNX release。"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import torch
from sklearn import __version__ as sklearn_version

from cms_network_redesign import (
    ARTIFACT_DIR,
    STUDY_NAME,
    _load_selected_model,
)
from cms_regional import (
    CORE_FEATURES,
    FILTERED_DATA_PATH,
    FILTERED_DIAGNOSTICS_PATH,
)
from data_loader import PROJECT_ROOT, TARGET_COLS
from export_cms_onnx import _onnx_metadata
from export_onnx import (
    DeploymentNet,
    INPUT_NAME,
    MAX_ALLOWED_DIFF,
    OPSET_VERSION,
    OUTPUT_NAME,
    REFERENCE_INPUTS,
    WRAPPER_FILES,
    _export_model,
    _sha256,
    _verify_model,
    _write_checksums,
    _write_csv,
    _write_interface,
    _write_onnx_metadata,
)


RELEASE_VERSION = "wdf_cms_network_redesign_v1"
RELEASE_DIR = PROJECT_ROOT / "deployment" / "releases" / RELEASE_VERSION
SOURCE_CMS_DIR = PROJECT_ROOT / "trained_models" / "wdf_cms_orig_core6_v1"
ONNX_FILENAME = "wdf_cms_orig_core6_v1.onnx"
COMPATIBILITY_ALIAS = "wdf_drifter.onnx"
WINDOWS_STAGING_PATH = (
    r"D:\OilspillModel\OilSpillModel\ModelRun\release_onnx"
    rf"\{RELEASE_VERSION}"
)
WINDOWS_ACCEPTANCE_THRESHOLD = 1e-4

SUPPORT_FILES: dict[Path, str] = {
    ARTIFACT_DIR / "selected" / "best_mlp.pth": "best_mlp.pth",
    ARTIFACT_DIR / "selected" / "x_scaler.pkl": "x_scaler.pkl",
    ARTIFACT_DIR / "selected" / "model_config.json": "model_config.json",
    ARTIFACT_DIR
    / "selected"
    / "training_history.json": "training_history.json",
    ARTIFACT_DIR / "selection_lock.json": "selection_lock.json",
    ARTIFACT_DIR / "cv_fold_manifest.json": "cv_fold_manifest.json",
    ARTIFACT_DIR
    / "cv_results_partial.json": "architecture_cv_results.json",
    ARTIFACT_DIR / "test_evaluation.json": "test_evaluation.json",
    SOURCE_CMS_DIR / "split_manifest.json": "split_manifest.json",
    SOURCE_CMS_DIR / "cms_region_mask.json": "cms_region_mask.json",
    SOURCE_CMS_DIR
    / "cms_region_row_index.npz": "cms_region_row_index.npz",
    SOURCE_CMS_DIR
    / "cms_data_statistics.json": "cms_data_statistics.json",
    SOURCE_CMS_DIR
    / "regional_linear_analysis.json": "regional_linear_analysis.json",
    SOURCE_CMS_DIR
    / "linear_baseline_metrics.json": "linear_baseline_metrics.json",
    SOURCE_CMS_DIR
    / "mlp_metrics.json": "original_cms_mlp_metrics.json",
    FILTERED_DIAGNOSTICS_PATH: FILTERED_DIAGNOSTICS_PATH.name,
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_locked_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    selection_path = ARTIFACT_DIR / "selection_lock.json"
    evaluation_path = ARTIFACT_DIR / "test_evaluation.json"
    for path in (selection_path, evaluation_path, *SUPPORT_FILES):
        if not path.is_file():
            raise FileNotFoundError(path)

    selection = _read_json(selection_path)
    evaluation = _read_json(evaluation_path)
    if selection["test_status"] != "sealed_not_evaluated":
        raise RuntimeError("selection lock 已被修改，拒绝冻结。")
    if evaluation["test_status"] != "evaluated_once_locked":
        raise RuntimeError("缺少唯一一次固定 test 评价。")
    if evaluation["selection_lock_sha256"] != _sha256(selection_path):
        raise RuntimeError("test 评价与 selection lock SHA256 不匹配。")
    if evaluation["selected_architecture"] != selection[
        "selected_architecture"
    ]:
        raise RuntimeError("selection 与 test 评价的模型结构不一致。")
    if evaluation["recommendation"] != "do_not_replace_frozen_global":
        raise RuntimeError("冻结脚本的安全状态与 test 结论不一致。")
    return selection, evaluation


def _deployment_model(
    selection: dict[str, Any],
) -> tuple[DeploymentNet, Any]:
    model, scaler = _load_selected_model(selection)
    if int(scaler.n_features_in_) != len(CORE_FEATURES):
        raise ValueError("Scaler 特征数不等于冻结 core6 接口。")
    if not np.all(np.isfinite(scaler.mean_)):
        raise ValueError("Scaler mean_ 含非有限值。")
    if not np.all(np.isfinite(scaler.scale_)):
        raise ValueError("Scaler scale_ 含非有限值。")
    if np.any(np.asarray(scaler.scale_) <= 0):
        raise ValueError("Scaler scale_ 必须全部大于 0。")

    deployment = DeploymentNet(
        model,
        torch.as_tensor(scaler.mean_, dtype=torch.float32),
        torch.as_tensor(scaler.scale_, dtype=torch.float32),
    )
    deployment.eval()
    return deployment, scaler


def _copy_support_files(release_dir: Path) -> None:
    for source, filename in SUPPORT_FILES.items():
        shutil.copy2(source, release_dir / filename)
    for filename in WRAPPER_FILES:
        source = PROJECT_ROOT / "src" / "models" / filename
        destination = release_dir / filename
        if source.suffix.lower() == ".bat":
            text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
            destination.write_bytes(
                text.replace("\n", "\r\n").encode("utf-8")
            )
        else:
            shutil.copy2(source, destination)


def _write_contract_check(
    release_dir: Path,
    selection: dict[str, Any],
    onnx_metadata: dict[str, Any],
) -> dict[str, Any]:
    training_contract = selection["training_contract"]
    checks = {
        "feature_order_unchanged": (
            training_contract["features"] == CORE_FEATURES
        ),
        "target_definition_unchanged": (
            training_contract["target"]
            == {
                "residual_u": "ve - cfsv2_u",
                "residual_v": "vn - cfsv2_v",
            }
        ),
        "input_name_unchanged": (
            onnx_metadata["input"]["name"] == INPUT_NAME
        ),
        "output_name_unchanged": (
            onnx_metadata["output"]["name"] == OUTPUT_NAME
        ),
        "input_width_unchanged": (
            onnx_metadata["input"]["shape"] == ["batch_size", 6]
        ),
        "output_width_unchanged": (
            onnx_metadata["output"]["shape"] == ["batch_size", 2]
        ),
        "float32_interface_unchanged": (
            onnx_metadata["input"]["element_type"] == 1
            and onnx_metadata["output"]["element_type"] == 1
        ),
        "scaler_inside_onnx": (
            onnx_metadata["custom_metadata"]["scaler_inside_graph"] == "true"
        ),
        "fortran_sources_unchanged": all(
            _sha256(release_dir / filename)
            == _sha256(PROJECT_ROOT / "src" / "models" / filename)
            for filename in (
                "onnx_wrapper.cpp",
                "onnx_wrapper.h",
                "wdf_model_mod.f90",
                "test_wdf_onnx.f90",
            )
        ),
    }
    result = {
        "schema_version": 1,
        "changed_surface": "MLP architecture only",
        "selected_architecture": selection["selected_architecture"],
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    if not result["all_passed"]:
        raise RuntimeError(f"冻结接口检查失败: {checks}")
    _write_json(release_dir / "frozen_contract_check.json", result)
    return result


def _write_readme(
    release_dir: Path,
    manifest: dict[str, Any],
) -> None:
    metrics = manifest["test_evaluation"]["subsets"]["CMS_overall"]
    selected = metrics["selected_network"]
    global_model = metrics["frozen_global"]
    release_dir.joinpath("README.md").write_text(
        f"""# {RELEASE_VERSION} Windows Handoff

## 状态

- 结构：`plain_64_32`，参数量 2,594
- 科学状态：已冻结的 CMS 网络重构实验
- 部署状态：Windows staging 候选，禁止自动激活
- 权威 ONNX：`{ONNX_FILENAME}`
- Fortran 兼容别名：`{COMPATIBILITY_ALIAS}`
- Windows staging：`{WINDOWS_STAGING_PATH}`
- 激活建议：`do_not_activate`

本 release 不覆盖旧 `wdf_cms_orig_core6_v1` release，也不修改
`onnx_active`。两个 ONNX 文件字节级相同；兼容别名仅用于保持既有
Fortran 验证程序不变。

## 冻结接口

- 输入：`input`, float32, `(batch_size, 6)`
- 输出：`output`, float32, `(batch_size, 2)`
- StandardScaler 已烘焙进 ONNX，Fortran 端不得再次标准化
- 输入顺序：
  `era5_u10, era5_v10, era5_swh, era5_mwp,`
  `era5_wave_dir_sin, era5_wave_dir_cos`
- target：
  `residual_u = ve - cfsv2_u`,
  `residual_v = vn - cfsv2_v`
- opset：{OPSET_VERSION}
- Python PyTorch/ONNX 最大绝对误差：
  {manifest['verification']['max_absolute_difference']:.3e}

## 固定 test 结果

- 新 regional 网络：joint R² {selected['r2_joint']:.6f}，
  RMSE {selected['rmse']:.6f} m/s
- 旧 CMS MLP：joint R²
  {manifest['test_evaluation']['references']['original_cms_mlp']['r2_joint']:.6f}
- 区域线性基准：joint R²
  {manifest['test_evaluation']['references']['regional_linear']['r2_joint']:.6f}
- 冻结 global MLP（同一批行）：joint R²
  {global_model['r2_joint']:.6f}，RMSE {global_model['rmse']:.6f} m/s

新结构改善了旧 CMS MLP 和区域线性基准，但仍弱于冻结 global MLP。
因此该模型仅作为可复现实验交付，不应替换当前运行模型。

## 数据限制

- CMS：23 个 original_ID，21,074 行
- train/val/test：15/4/4 个 original_ID，交集为 0
- BYS/ECS/NSCS 行数：0/13,956/7,118
- 当前源数据没有 BYS 样本，不能声称具备渤海—黄海泛化能力

## Windows 验证

1. 保持 release 文件完整，不与旧版本混用。
2. 在 staging 目录运行 `verify_windows.bat`。
3. 要求所有输出与 `expected_output.csv` 的绝对误差 `< 1e-4`。
4. 记录 `SHA256SUMS.txt` 中权威 ONNX 的 SHA256。
5. 不修改 release root 或 `onnx_active`。
""",
        encoding="utf-8",
    )


def _build_release_contents(
    release_dir: Path,
    *,
    freeze_commit: str,
) -> dict[str, Any]:
    selection, evaluation = _load_locked_inputs()
    deployment_model, scaler = _deployment_model(selection)
    release_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = release_dir / ONNX_FILENAME
    alias_path = release_dir / COMPATIBILITY_ALIAS
    _export_model(deployment_model, onnx_path, CORE_FEATURES)
    metadata_properties = _write_onnx_metadata(
        onnx_path,
        model_version=RELEASE_VERSION,
        feature_cols=CORE_FEATURES,
        training_run=str(ARTIFACT_DIR.relative_to(PROJECT_ROOT)),
    )
    reference_output, verification = _verify_model(
        deployment_model,
        onnx_path,
        CORE_FEATURES,
    )
    shutil.copy2(onnx_path, alias_path)
    if _sha256(onnx_path) != _sha256(alias_path):
        raise RuntimeError("权威 ONNX 与 Fortran 兼容别名不一致。")

    _write_csv(release_dir / "test_input.csv", CORE_FEATURES, REFERENCE_INPUTS)
    _write_csv(
        release_dir / "expected_output.csv",
        TARGET_COLS,
        reference_output,
    )
    _write_interface(
        release_dir / "interface.json",
        CORE_FEATURES,
        RELEASE_VERSION,
    )
    _copy_support_files(release_dir)
    detailed_onnx_metadata = _onnx_metadata(onnx_path)
    _write_json(release_dir / "onnx_metadata.json", detailed_onnx_metadata)
    contract_check = _write_contract_check(
        release_dir,
        selection,
        detailed_onnx_metadata,
    )

    split_manifest = _read_json(SOURCE_CMS_DIR / "split_manifest.json")
    data_statistics = _read_json(
        SOURCE_CMS_DIR / "cms_data_statistics.json"
    )
    diagnostics = _read_json(FILTERED_DIAGNOSTICS_PATH)
    dataset_sha256 = _sha256(FILTERED_DATA_PATH)
    expected_dataset_sha256 = diagnostics["file_integrity"]["output_sha256"]
    if dataset_sha256 != expected_dataset_sha256:
        raise RuntimeError("CMS 训练数据 SHA256 与诊断记录不一致。")

    selected_metrics = evaluation["subsets"]["CMS_overall"][
        "selected_network"
    ]
    global_metrics = evaluation["subsets"]["CMS_overall"]["frozen_global"]
    selection_assessment = {
        "beats_original_cms_mlp": (
            selected_metrics["r2_joint"]
            > evaluation["references"]["original_cms_mlp"]["r2_joint"]
            and selected_metrics["rmse"]
            < evaluation["references"]["original_cms_mlp"]["rmse"]
        ),
        "beats_regional_linear": (
            selected_metrics["r2_joint"]
            > evaluation["references"]["regional_linear"]["r2_joint"]
            and selected_metrics["rmse"]
            < evaluation["references"]["regional_linear"]["rmse"]
        ),
        "beats_frozen_global_same_rows": (
            selected_metrics["r2_joint"] > global_metrics["r2_joint"]
            and selected_metrics["rmse"] < global_metrics["rmse"]
        ),
        "activation_recommendation": "do_not_activate",
        "reason": (
            "The redesigned regional MLP improves the original CMS MLP and "
            "regional linear baseline, but underperforms the frozen global "
            "MLP on the identical test rows."
        ),
    }

    excluded_from_source_hashes = {
        ONNX_FILENAME,
        COMPATIBILITY_ALIAS,
        "release_manifest.json",
        "README.md",
        "SHA256SUMS.txt",
    }
    source_hashes = {
        path.name: _sha256(path)
        for path in sorted(release_dir.iterdir())
        if path.is_file() and path.name not in excluded_from_source_hashes
    }
    manifest = {
        "schema_version": 1,
        "model_version": RELEASE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "frozen_experimental_cms_architecture_redesign_not_selected"
        ),
        "deployment_status": "candidate_pending_windows_validation",
        "activation_recommendation": "do_not_activate",
        "freeze_code_git_commit": freeze_commit,
        "selection_code_git_commit": selection[
            "selection_code_git_commit"
        ],
        "test_evaluation_code_git_commit": evaluation[
            "evaluation_code_git_commit"
        ],
        "selected_architecture": selection["selected_architecture"],
        "selected_architecture_spec": selection["candidate_registry"][
            selection["selected_architecture"]
        ],
        "selected_parameter_count": evaluation["selected_parameter_count"],
        "training_contract": selection["training_contract"],
        "selection": {
            "method": (
                "5-fold original_ID GroupKFold on fixed train IDs; "
                "top two evaluated on fixed validation IDs"
            ),
            "cv_ranking": selection["cv_ranking"],
            "fixed_validation_ranking": selection[
                "fixed_validation_ranking"
            ],
            "selected_validation_metrics": selection[
                "fixed_validation_results"
            ][selection["selected_architecture"]]["validation_metrics"],
            "selection_lock_sha256": _sha256(
                ARTIFACT_DIR / "selection_lock.json"
            ),
        },
        "test_evaluation": evaluation,
        "selection_assessment": selection_assessment,
        "regional_scope": {
            "name": "China Marginal Seas",
            "expression": "BYS OR ECS OR NSCS",
            "dataset": data_statistics["cms_dataset"],
            "region_membership_counts": data_statistics[
                "region_membership_counts"
            ],
            "month_counts": data_statistics["month_counts"],
            "observed_support_limitation": (
                "BYS has zero rows in the supplied source dataset."
            ),
        },
        "split_method": split_manifest["split_provenance"]["strategy"],
        "split_random_seed": split_manifest["random_seed"],
        "split_counts": {
            name: {
                key: split_manifest["splits"][name][key]
                for key in ("n_original_ids", "n_segments", "n_samples")
            }
            for name in ("train", "val", "test")
        },
        "pairwise_original_id_intersections": data_statistics["split"][
            "pairwise_original_id_intersections"
        ],
        "dataset": {
            "filename": FILTERED_DATA_PATH.name,
            "size_bytes": FILTERED_DATA_PATH.stat().st_size,
            "sha256": dataset_sha256,
            "diagnostics_filename": FILTERED_DIAGNOSTICS_PATH.name,
            "diagnostics_sha256": _sha256(FILTERED_DIAGNOSTICS_PATH),
        },
        "onnx": {
            "filename": ONNX_FILENAME,
            "compatibility_alias": COMPATIBILITY_ALIAS,
            "sha256": _sha256(onnx_path),
            "compatibility_alias_sha256": _sha256(alias_path),
            "opset": OPSET_VERSION,
            "input_name": INPUT_NAME,
            "output_name": OUTPUT_NAME,
            "scaler_inside_graph": True,
            "metadata_properties": metadata_properties,
        },
        "verification": verification,
        "frozen_contract_check": contract_check,
        "scaler": {
            "n_features_in": int(scaler.n_features_in_),
            "n_samples_seen": int(np.asarray(scaler.n_samples_seen_).max()),
            "mean": np.asarray(scaler.mean_).tolist(),
            "scale": np.asarray(scaler.scale_).tolist(),
        },
        "source_hashes": source_hashes,
        "windows": {
            "staging_path": WINDOWS_STAGING_PATH,
            "onnx_active_modified": False,
            "python_export_acceptance_threshold": MAX_ALLOWED_DIFF,
            "fortran_acceptance_threshold": WINDOWS_ACCEPTANCE_THRESHOLD,
        },
        "software": {
            "python_environment": "Miniforge3 conda env buoy-drifter",
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "scikit_learn": sklearn_version,
            "joblib": joblib.__version__,
        },
    }
    _write_json(release_dir / "release_manifest.json", manifest)
    _write_readme(release_dir, manifest)
    _write_checksums(release_dir)
    return manifest


def build_redesign_release(freeze_commit: str) -> dict[str, Any]:
    """在临时目录完整构建，通过后原子移动到全新 release 目录。"""
    if RELEASE_DIR.exists():
        raise FileExistsError(f"拒绝覆盖已有 release: {RELEASE_DIR}")
    releases_root = RELEASE_DIR.parent
    releases_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{RELEASE_VERSION}.",
            dir=releases_root,
        )
    )
    try:
        manifest = _build_release_contents(
            temporary,
            freeze_commit=freeze_commit,
        )
        temporary.rename(RELEASE_DIR)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="冻结 CMS 网络重构 ONNX Windows release"
    )
    parser.add_argument("--freeze-commit", required=True)
    args = parser.parse_args()
    manifest = build_redesign_release(args.freeze_commit)
    print(f"Release: {RELEASE_DIR}")
    print(f"ONNX: {RELEASE_DIR / ONNX_FILENAME}")
    print(f"SHA256: {manifest['onnx']['sha256']}")
    print(
        "PyTorch/ONNX max abs diff: "
        f"{manifest['verification']['max_absolute_difference']:.3e}"
    )
    print(
        "Activation recommendation: "
        f"{manifest['activation_recommendation']}"
    )


if __name__ == "__main__":
    main()
