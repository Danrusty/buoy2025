"""
将指定 original_ID MLP 导出为带内部标准化的 ONNX 候选包。

发布包包含 ONNX、接口元数据、固定测试向量、Python 预期输出、Windows
wrapper 源码和 SHA256 校验值。Windows/Fortran 端只输入原始物理量，
不能再次执行 StandardScaler。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from data_loader import FEATURE_COLS, PROJECT_ROOT, TARGET_COLS
from train_mlp import ResidualMLP


MODEL_VERSION = "wdf_full9_ablation_reference_v1"
DEFAULT_RUN_NAME = "ablation_study/full_9"
OPSET_VERSION = 12
INPUT_NAME = "input"
OUTPUT_NAME = "output"
MAX_ALLOWED_DIFF = 1e-5

FEATURE_UNITS = [
    "m/s",
    "m/s",
    "m/s",
    "dimensionless",
    "dimensionless",
    "m",
    "s",
    "dimensionless",
    "dimensionless",
]
TARGET_UNITS = ["m/s", "m/s"]

# 三组有限、物理量级合理且方向编码自洽的固定测试向量。
REFERENCE_INPUTS = np.asarray(
    [
        [5.0, 0.0, 5.0, 0.0, 1.0, 1.5, 7.0, 0.0, 1.0],
        [-4.0, 3.0, 5.0, 0.6, -0.8, 2.5, 9.0, 1.0, 0.0],
        [0.0, -8.0, 8.0, -1.0, 0.0, 4.0, 12.0, -0.70710677, 0.70710677],
    ],
    dtype=np.float32,
)

WRAPPER_FILES = [
    "onnx_wrapper.cpp",
    "onnx_wrapper.h",
    "wdf_model_mod.f90",
    "test_wdf_onnx.f90",
    "build_wrapper.bat",
]


class DeploymentNet(nn.Module):
    """把训练集 StandardScaler 固化到 MLP 计算图内部。"""

    def __init__(
        self,
        trained_mlp: nn.Module,
        scaler_mean: torch.Tensor,
        scaler_scale: torch.Tensor,
    ):
        super().__init__()
        self.mlp = trained_mlp
        self.register_buffer("scaler_mean", scaler_mean.unsqueeze(0))
        self.register_buffer("scaler_scale", scaler_scale.unsqueeze(0))

    def forward(self, x_physical: torch.Tensor) -> torch.Tensor:
        x_scaled = (x_physical - self.scaler_mean) / self.scaler_scale
        return self.mlp(x_scaled)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, columns: list[str], values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(
            [[f"{float(value):.9g}" for value in row] for row in values]
        )


def _load_deployment_model(
    checkpoint_path: Path,
    scaler_path: Path,
) -> tuple[DeploymentNet, Any]:
    model = ResidualMLP(input_size=len(FEATURE_COLS), output_size=len(TARGET_COLS))
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    scaler = joblib.load(scaler_path)
    if int(scaler.n_features_in_) != len(FEATURE_COLS):
        raise ValueError(
            f"Scaler 特征数为 {scaler.n_features_in_}，预期 {len(FEATURE_COLS)}。"
        )
    if not np.all(np.isfinite(scaler.mean_)) or not np.all(
        np.isfinite(scaler.scale_)
    ):
        raise ValueError("Scaler 包含非有限参数。")
    if np.any(np.asarray(scaler.scale_) <= 0):
        raise ValueError("Scaler scale_ 必须全部大于 0。")

    deployment_model = DeploymentNet(
        model,
        torch.as_tensor(scaler.mean_, dtype=torch.float32),
        torch.as_tensor(scaler.scale_, dtype=torch.float32),
    )
    deployment_model.eval()
    return deployment_model, scaler


def _export_model(model: DeploymentNet, onnx_path: Path) -> None:
    dummy_input = torch.zeros((1, len(FEATURE_COLS)), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAME: {0: "batch_size"},
            OUTPUT_NAME: {0: "batch_size"},
        },
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)


def _verify_model(
    model: DeploymentNet,
    onnx_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    with torch.no_grad():
        pytorch_output = model(torch.from_numpy(REFERENCE_INPUTS)).numpy()

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    onnx_output = session.run(
        [OUTPUT_NAME],
        {INPUT_NAME: REFERENCE_INPUTS},
    )[0]

    # 单样本验证动态 batch，而不仅是导出时使用的 batch=1。
    single_output = session.run(
        [OUTPUT_NAME],
        {INPUT_NAME: REFERENCE_INPUTS[:1]},
    )[0]
    if single_output.shape != (1, len(TARGET_COLS)):
        raise RuntimeError(f"动态 batch 验证失败: {single_output.shape}")

    difference = np.abs(pytorch_output - onnx_output)
    max_diff = float(difference.max())
    mean_diff = float(difference.mean())
    if max_diff >= MAX_ALLOWED_DIFF:
        raise RuntimeError(
            f"PyTorch/ONNX 最大绝对误差 {max_diff:.3e} "
            f">= {MAX_ALLOWED_DIFF:.1e}"
        )
    if not np.all(np.isfinite(onnx_output)):
        raise RuntimeError("ONNX 固定测试输出包含 NaN 或 Inf。")

    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    expected_input_shape = ["batch_size", len(FEATURE_COLS)]
    expected_output_shape = ["batch_size", len(TARGET_COLS)]
    if input_meta.name != INPUT_NAME or input_meta.shape != expected_input_shape:
        raise RuntimeError(
            f"ONNX 输入接口异常: {input_meta.name}, {input_meta.shape}"
        )
    if output_meta.name != OUTPUT_NAME or output_meta.shape != expected_output_shape:
        raise RuntimeError(
            f"ONNX 输出接口异常: {output_meta.name}, {output_meta.shape}"
        )

    verification = {
        "reference_batch_size": int(len(REFERENCE_INPUTS)),
        "max_absolute_difference": max_diff,
        "mean_absolute_difference": mean_diff,
        "acceptance_threshold": MAX_ALLOWED_DIFF,
        "dynamic_batch_single_sample_passed": True,
    }
    return onnx_output, verification


def _write_interface(path: Path) -> None:
    interface = {
        "model_version": MODEL_VERSION,
        "opset": OPSET_VERSION,
        "input": {
            "name": INPUT_NAME,
            "shape": ["batch_size", len(FEATURE_COLS)],
            "dtype": "float32",
            "physical_values": True,
            "standardization": "inside_onnx",
            "features": [
                {"index": index + 1, "name": name, "unit": unit}
                for index, (name, unit) in enumerate(
                    zip(FEATURE_COLS, FEATURE_UNITS)
                )
            ],
        },
        "output": {
            "name": OUTPUT_NAME,
            "shape": ["batch_size", len(TARGET_COLS)],
            "dtype": "float32",
            "variables": [
                {
                    "index": index + 1,
                    "name": name,
                    "unit": unit,
                    "positive_direction": "east" if index == 0 else "north",
                }
                for index, (name, unit) in enumerate(
                    zip(TARGET_COLS, TARGET_UNITS)
                )
            ],
        },
        "input_validation": {
            "nan_or_inf_supported": False,
            "missing_value_handling_inside_model": False,
        },
        "fortran_layout": {
            "input_declaration": "real(c_float) :: features(9, N)",
            "output_declaration": "real(c_float) :: drift_uv(2, N)",
        },
    }
    path.write_text(
        json.dumps(interface, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_readme(path: Path, verification: dict[str, Any]) -> None:
    path.write_text(
        f"""# WDF ONNX Windows Handoff

- Model version: `{MODEL_VERSION}`
- Scientific status: full_9 ablation reference
- Windows status: candidate, pending repeated C++/Fortran validation
- ONNX opset: {OPSET_VERSION}
- Input: `input`, float32, `(batch_size, 9)`
- Output: `output`, float32, `(batch_size, 2)`
- StandardScaler: baked into ONNX; do not standardize again in Fortran
- PyTorch/ONNX max absolute difference: {verification['max_absolute_difference']:.3e}

## Files

- `wdf_drifter.onnx`: Windows runtime model.
- `interface.json`: authoritative feature/output contract.
- `test_input.csv`: fixed raw physical input vectors.
- `expected_output.csv`: Python ONNX Runtime reference output.
- `release_manifest.json`: source model, split and metric provenance.
- `SHA256SUMS.txt`: release file integrity checks.
- `onnx_wrapper.*`, `wdf_model_mod.f90`: C++/Fortran interface.
- `test_wdf_onnx.f90`: Windows chain verification program.
- `build_wrapper.bat`: VS2022 x64 wrapper build script.

## Windows acceptance

1. Replace only `wdf_drifter.onnx`; wrapper ABI is unchanged.
2. Build in VS2022 x64 Developer Command Prompt with oneAPI Fortran.
3. Run `test_wdf_onnx.exe`.
4. Compare all outputs with `expected_output.csv`; require absolute error `< 1e-4`.
5. Record the deployed ONNX SHA256 from `SHA256SUMS.txt`.

The previous trajectory-index split model is internal legacy and must not be
mixed with this release.
""",
        encoding="utf-8",
    )


def _write_checksums(release_dir: Path) -> None:
    checksum_path = release_dir / "SHA256SUMS.txt"
    files = sorted(
        path for path in release_dir.iterdir()
        if path.is_file() and path.name != checksum_path.name
    )
    lines = [f"{_sha256(path)}  {path.name}" for path in files]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def build_release(
    run_dir: Path,
    release_dir: Path,
) -> dict[str, Any]:
    checkpoint_path = run_dir / "best_mlp.pth"
    scaler_path = run_dir / "x_scaler.pkl"
    split_manifest_path = run_dir / "split_manifest.json"
    metrics_path = run_dir / "mlp_metrics.json"
    for required_path in (
        checkpoint_path,
        scaler_path,
        split_manifest_path,
        metrics_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    release_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = release_dir / "wdf_drifter.onnx"

    model, scaler = _load_deployment_model(checkpoint_path, scaler_path)
    _export_model(model, onnx_path)
    onnx_output, verification = _verify_model(model, onnx_path)

    _write_csv(release_dir / "test_input.csv", FEATURE_COLS, REFERENCE_INPUTS)
    _write_csv(release_dir / "expected_output.csv", TARGET_COLS, onnx_output)
    _write_interface(release_dir / "interface.json")

    for filename in WRAPPER_FILES:
        shutil.copy2(
            PROJECT_ROOT / "src" / "models" / filename,
            release_dir / filename,
        )

    shutil.copy2(scaler_path, release_dir / "x_scaler.pkl")

    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": "ablation_reference_full_9",
        "deployment_status": "candidate_pending_windows_validation",
        "split_method": "original_ID",
        "split_random_seed": split_manifest["random_seed"],
        "split_counts": {
            name: {
                key: split_manifest["splits"][name][key]
                for key in ("n_original_ids", "n_segments", "n_samples")
            }
            for name in ("train", "val", "test")
        },
        "metrics": metrics,
        "onnx": {
            "filename": onnx_path.name,
            "opset": OPSET_VERSION,
            "input_name": INPUT_NAME,
            "output_name": OUTPUT_NAME,
            "scaler_inside_graph": True,
        },
        "verification": verification,
        "source_hashes": {
            "checkpoint_sha256": _sha256(checkpoint_path),
            "scaler_sha256": _sha256(scaler_path),
            "split_manifest_sha256": _sha256(split_manifest_path),
        },
        "scaler": {
            "n_features_in": int(scaler.n_features_in_),
            "n_samples_seen": int(scaler.n_samples_seen_),
            "mean": np.asarray(scaler.mean_).tolist(),
            "scale": np.asarray(scaler.scale_).tolist(),
        },
        "software": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_readme(release_dir / "README.md", verification)
    _write_checksums(release_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 WDF ONNX Windows 交接包")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=PROJECT_ROOT / "deployment" / "releases" / MODEL_VERSION,
    )
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "trained_models" / args.run_name
    release_dir = args.release_dir.resolve()
    manifest = build_release(run_dir, release_dir)

    print("WDF ONNX 发布包生成完成")
    print(f"  version : {manifest['model_version']}")
    print(f"  source  : {run_dir}")
    print(f"  release : {release_dir}")
    print(
        "  max diff: "
        f"{manifest['verification']['max_absolute_difference']:.3e}"
    )
    print("  Windows validation: pending")


if __name__ == "__main__":
    main()
