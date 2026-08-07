"""冻结并打包 ``wdf_cms_orig_core6_v1`` Windows ONNX release。"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import onnx

from cms_regional import (
    FILTERED_DIAGNOSTICS_PATH,
    MODEL_VERSION,
)
from data_loader import PROJECT_ROOT
from export_onnx import (
    _sha256,
    _write_checksums,
    build_release,
)


RUN_DIR = PROJECT_ROOT / "trained_models" / MODEL_VERSION
RESULT_DIR = PROJECT_ROOT / "results" / MODEL_VERSION
RELEASE_DIR = PROJECT_ROOT / "deployment" / "releases" / MODEL_VERSION
ONNX_FILENAME = f"{MODEL_VERSION}.onnx"
COMPATIBILITY_ALIAS = "wdf_drifter.onnx"
WINDOWS_STAGING_PATH = (
    rf"D:\OilspillModel\OilSpillModel\ModelRun\release_onnx\{MODEL_VERSION}"
)

SUPPORT_ARTIFACTS = [
    "best_mlp.pth",
    "x_scaler.pkl",
    "split_manifest.json",
    "cms_region_mask.json",
    "cms_region_row_index.npz",
    "cms_data_statistics.json",
    "regional_linear_analysis.json",
    "regional_evaluation.json",
    "mlp_metrics.json",
    "linear_baseline_metrics.json",
    "linear_baseline.joblib",
    "model_config.json",
    "frozen_contract_check.json",
    "training_history.json",
]


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _onnx_metadata(onnx_path: Path) -> dict:
    model = onnx.load(onnx_path)
    input_tensor = model.graph.input[0]
    output_tensor = model.graph.output[0]

    def dimensions(value_info: onnx.ValueInfoProto) -> list[str | int | None]:
        result: list[str | int | None] = []
        for dimension in value_info.type.tensor_type.shape.dim:
            if dimension.dim_param:
                result.append(dimension.dim_param)
            elif dimension.HasField("dim_value"):
                result.append(int(dimension.dim_value))
            else:
                result.append(None)
        return result

    return {
        "filename": onnx_path.name,
        "sha256": _sha256(onnx_path),
        "ir_version": int(model.ir_version),
        "opset_imports": [
            {
                "domain": value.domain or "ai.onnx",
                "version": int(value.version),
            }
            for value in model.opset_import
        ],
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "input": {
            "name": input_tensor.name,
            "shape": dimensions(input_tensor),
            "element_type": int(input_tensor.type.tensor_type.elem_type),
        },
        "output": {
            "name": output_tensor.name,
            "shape": dimensions(output_tensor),
            "element_type": int(output_tensor.type.tensor_type.elem_type),
        },
        "custom_metadata": {
            item.key: item.value for item in model.metadata_props
        },
    }


def _write_handoff_readme(
    release_dir: Path,
    manifest: dict,
    statistics: dict,
    evaluation: dict,
) -> None:
    overall = evaluation["subsets"]["CMS_overall"]
    regional = overall["regional_mlp"]
    global_model = overall["frozen_global_mlp"]
    release_dir.joinpath("README.md").write_text(
        f"""# {MODEL_VERSION} Windows Handoff

## Identity and status

- Model: `{MODEL_VERSION}`
- Scientific status: frozen single CMS regional core6 MLP
- Deployment status: Windows candidate; Python ONNX verification passed
- Authoritative ONNX: `{ONNX_FILENAME}`
- Unchanged Fortran-test alias: `{COMPATIBILITY_ALIAS}`
- Intended Windows staging directory:
  `{WINDOWS_STAGING_PATH}`
- `onnx_active` is not changed by this handoff.

The two ONNX filenames are byte-identical. The named CMS file is the
authoritative handoff artifact; the alias keeps the existing
`test_wdf_onnx.f90` logic unchanged.

## Frozen interface

- Input: `input`, float32, `(batch_size, 6)`
- Output: `output`, float32, `(batch_size, 2)`
- StandardScaler is inside ONNX; do not standardize again in Fortran.
- Feature order:
  `era5_u10, era5_v10, era5_swh, era5_mwp,`
  `era5_wave_dir_sin, era5_wave_dir_cos`
- Target:
  `residual_u = ve - cfsv2_u`,
  `residual_v = vn - cfsv2_v`
- ONNX opset: {manifest['onnx']['opset']}
- PyTorch/ONNX maximum absolute difference:
  {manifest['verification']['max_absolute_difference']:.3e}
- Dynamic batches verified:
  {manifest['verification']['dynamic_batch_sizes_passed']}

## Regional data support

- CMS IDs / rows:
  {statistics['cms_dataset']['n_original_ids']} /
  {statistics['cms_dataset']['n_samples']}
- BYS / ECS / NSCS rows:
  {statistics['region_membership_counts']['BYS']} /
  {statistics['region_membership_counts']['ECS']} /
  {statistics['region_membership_counts']['NSCS']}
- Split strategy:
  `{statistics['split']['provenance']['strategy']}`

The supplied source contains zero BYS rows. The release is valid for the exact
CMS-mask experiment, but its observed training support is ECS + NSCS; no BYS
performance claim can be made.

## Same-test-set comparison

- Regional CMS test joint R2: {regional['r2_joint']:.6f}
- Regional CMS test RMSE: {regional['rmse']:.6f} m/s
- Frozen global joint R2 on the same rows: {global_model['r2_joint']:.6f}
- Frozen global RMSE on the same rows: {global_model['rmse']:.6f} m/s

## Verification

1. Keep all release files together.
2. Run `verify_windows.bat` in the staging directory.
3. The unchanged Fortran test loads `{COMPATIBILITY_ALIAS}`.
4. Require every output difference from `expected_output.csv` to be `< 1e-4`.
5. Record the authoritative ONNX SHA256 from `SHA256SUMS.txt`.

Checkpoint, scaler, split manifest, row-level region index, region definition,
data statistics, metrics, fixed test vectors, ONNX metadata and checksums are
included for reproducibility.
""",
        encoding="utf-8",
    )


def build_cms_release(training_commit: str) -> dict:
    if not RESULT_DIR.joinpath("experiment.json").is_file():
        raise FileNotFoundError(RESULT_DIR / "experiment.json")
    manifest = build_release(
        RUN_DIR,
        RELEASE_DIR,
        model_version=MODEL_VERSION,
        data_diagnostics_path=FILTERED_DIAGNOSTICS_PATH,
        training_commit=training_commit,
        onnx_filename=ONNX_FILENAME,
        compatibility_alias=COMPATIBILITY_ALIAS,
    )

    for filename in SUPPORT_ARTIFACTS:
        source = RUN_DIR / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, RELEASE_DIR / filename)
    shutil.copy2(
        FILTERED_DIAGNOSTICS_PATH,
        RELEASE_DIR / FILTERED_DIAGNOSTICS_PATH.name,
    )
    shutil.copy2(
        RESULT_DIR / "experiment.json",
        RELEASE_DIR / "experiment.json",
    )

    onnx_path = RELEASE_DIR / ONNX_FILENAME
    alias_path = RELEASE_DIR / COMPATIBILITY_ALIAS
    if _sha256(onnx_path) != _sha256(alias_path):
        raise RuntimeError("权威 ONNX 与 Fortran 兼容别名不是字节级一致。")
    onnx_metadata = _onnx_metadata(onnx_path)
    onnx_metadata_path = RELEASE_DIR / "onnx_metadata.json"
    _write_json(onnx_metadata_path, onnx_metadata)

    statistics = json.loads(
        (RUN_DIR / "cms_data_statistics.json").read_text(encoding="utf-8")
    )
    region_mask = json.loads(
        (RUN_DIR / "cms_region_mask.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (RUN_DIR / "regional_evaluation.json").read_text(encoding="utf-8")
    )
    split_manifest = json.loads(
        (RUN_DIR / "split_manifest.json").read_text(encoding="utf-8")
    )
    linear_analysis = json.loads(
        (RUN_DIR / "regional_linear_analysis.json").read_text(encoding="utf-8")
    )

    manifest.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scientific_status": "frozen_single_cms_regional_core6",
            "regional_scope": {
                "name": "China Marginal Seas",
                "expression": "BYS OR ECS OR NSCS",
                "region_mask": region_mask,
                "statistics": statistics["cms_dataset"],
                "region_membership_counts": statistics[
                    "region_membership_counts"
                ],
                "month_counts": statistics["month_counts"],
                "observed_support_limitation": (
                    "BYS has zero rows in the supplied source dataset."
                ),
            },
            "split_method": split_manifest["split_provenance"]["strategy"],
            "split_provenance": split_manifest["split_provenance"],
            "regional_linear_analysis": linear_analysis,
            "regional_evaluation": evaluation,
            "onnx_metadata_file": onnx_metadata_path.name,
            "windows_staging_path": WINDOWS_STAGING_PATH,
            "onnx_active_modified": False,
        }
    )
    manifest["onnx"]["sha256"] = onnx_metadata["sha256"]
    manifest["onnx"]["compatibility_alias_sha256"] = _sha256(alias_path)
    manifest["source_hashes"].update(
        {
            filename.replace(".", "_") + "_sha256": _sha256(
                RELEASE_DIR / filename
            )
            for filename in SUPPORT_ARTIFACTS
            if filename != "best_mlp.pth"
        }
    )
    _write_json(RELEASE_DIR / "release_manifest.json", manifest)
    _write_handoff_readme(RELEASE_DIR, manifest, statistics, evaluation)
    _write_checksums(RELEASE_DIR)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 CMS regional ONNX release")
    parser.add_argument(
        "--training-commit",
        required=True,
        help="正式训练所用的 Git commit。",
    )
    args = parser.parse_args()
    manifest = build_cms_release(args.training_commit)
    print(f"Release: {RELEASE_DIR}")
    print(f"ONNX: {RELEASE_DIR / ONNX_FILENAME}")
    print(f"SHA256: {manifest['onnx']['sha256']}")
    print(
        "PyTorch/ONNX max abs diff: "
        f"{manifest['verification']['max_absolute_difference']:.3e}"
    )


if __name__ == "__main__":
    main()
