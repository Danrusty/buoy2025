"""Frozen-global CMS 低阶区域校正的拟合、分组验证和轨迹代理。"""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.model_selection import GroupKFold

from cms_regional import (
    CORE_FEATURES,
    FILTERED_DATA_PATH,
    GLOBAL_ONNX_PATH,
    GLOBAL_SPLIT_MANIFEST_PATH,
    region_memberships,
)
from data_loader import PROJECT_ROOT
from evaluation import regression_metrics


MODEL_VERSION = "wdf_cms_global_adapter_v1"
ARTIFACT_DIR = PROJECT_ROOT / "trained_models" / MODEL_VERSION
RESULT_DIR = PROJECT_ROOT / "results" / MODEL_VERSION
SOURCE_CMS_SPLIT_MANIFEST = (
    PROJECT_ROOT
    / "trained_models"
    / "wdf_cms_orig_core6_v1"
    / "split_manifest.json"
)
EXPECTED_GLOBAL_ONNX_SHA256 = (
    "787d1d6a663677e30161a70493c70a7e46434414fb59085fbb68477939f18941"
)
RANDOM_SEED = 42
LAMBDA_GRID = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
OUTER_SPLITS = 5
INNER_SPLITS = 4
TRAJECTORY_HORIZONS_HOURS = (6, 12, 24, 48, 72)


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    family: str
    parameter_names: tuple[str, ...]
    description: str

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names)


ADAPTER_SPECS: OrderedDict[str, AdapterSpec] = OrderedDict(
    (
        spec.name,
        spec,
    )
    for spec in (
        AdapterSpec(
            name="G0_global_only",
            family="none",
            parameter_names=(),
            description="Frozen global ONNX without correction.",
        ),
        AdapterSpec(
            name="G1_bias2",
            family="bias",
            parameter_names=("bias_u", "bias_v"),
            description="Two-component regional bias.",
        ),
        AdapterSpec(
            name="G2_wind_rotation4",
            family="wind_rotation",
            parameter_names=(
                "along_wind",
                "cross_wind",
                "bias_u",
                "bias_v",
            ),
            description=(
                "Rotation-structured wind correction plus component bias."
            ),
        ),
        AdapterSpec(
            name="G3_wind_full6",
            family="wind_full",
            parameter_names=(
                "u_from_u10",
                "u_from_v10",
                "v_from_u10",
                "v_from_v10",
                "bias_u",
                "bias_v",
            ),
            description="Full 2x2 wind correction plus component bias.",
        ),
        AdapterSpec(
            name="G4_global_calibration6",
            family="global_calibration",
            parameter_names=(
                "u_from_global_u",
                "u_from_global_v",
                "v_from_global_u",
                "v_from_global_v",
                "bias_u",
                "bias_v",
            ),
            description=(
                "Additive calibration of frozen-global output plus bias."
            ),
        ),
        AdapterSpec(
            name="G5_core6_linear14",
            family="core6",
            parameter_names=tuple(
                [f"u_from_{name}" for name in CORE_FEATURES]
                + [f"v_from_{name}" for name in CORE_FEATURES]
                + ["bias_u", "bias_v"]
            ),
            description="Full core6 linear discrepancy plus component bias.",
        ),
    )
)


@dataclass
class AdapterData:
    frame: pd.DataFrame
    features: np.ndarray
    target: np.ndarray
    global_prediction: np.ndarray
    groups: np.ndarray
    memberships: dict[str, np.ndarray]

    def subset(self, indices: np.ndarray | list[int]) -> "AdapterData":
        selected = np.asarray(indices)
        return AdapterData(
            frame=self.frame.iloc[selected].reset_index(drop=True),
            features=self.features[selected],
            target=self.target[selected],
            global_prediction=self.global_prediction[selected],
            groups=self.groups[selected],
            memberships={
                name: mask[selected]
                for name, mask in self.memberships.items()
            },
        )


@dataclass(frozen=True)
class FittedAdapter:
    spec_name: str
    lambda_value: float | None
    coefficients: tuple[float, ...]
    basis_rms_scales: tuple[float, ...]
    training_original_ids: tuple[str, ...]

    def to_dict(
        self,
        *,
        model_version: str = MODEL_VERSION,
    ) -> dict[str, Any]:
        spec = ADAPTER_SPECS[self.spec_name]
        return {
            "schema_version": 1,
            "model_version": model_version,
            "adapter_spec": asdict(spec),
            "lambda": self.lambda_value,
            "coefficients": {
                name: value
                for name, value in zip(
                    spec.parameter_names,
                    self.coefficients,
                )
            },
            "coefficient_vector": list(self.coefficients),
            "basis_rms_scales": list(self.basis_rms_scales),
            "training_original_ids": list(self.training_original_ids),
            "n_training_original_ids": len(self.training_original_ids),
            "base_onnx_sha256": EXPECTED_GLOBAL_ONNX_SHA256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FittedAdapter":
        spec_name = value["adapter_spec"]["name"]
        if spec_name not in ADAPTER_SPECS:
            raise ValueError(f"未知 adapter spec: {spec_name}")
        return cls(
            spec_name=spec_name,
            lambda_value=value["lambda"],
            coefficients=tuple(
                float(item) for item in value["coefficient_vector"]
            ),
            basis_rms_scales=tuple(
                float(item) for item in value["basis_rms_scales"]
            ),
            training_original_ids=tuple(
                str(item) for item in value["training_original_ids"]
            ),
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_ids(values: Iterable[str]) -> str:
    canonical = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _canonical_original_id(frame: pd.DataFrame) -> str:
    values = frame["original_ID"].dropna().astype(str).str.strip().unique()
    if len(values) != 1 or not values[0]:
        raise ValueError(
            f"区域 episode original_ID 异常: {values[:5].tolist()}"
        )
    return str(values[0])


def load_filtered_frames(
    path: Path = FILTERED_DATA_PATH,
) -> list[pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as file:
        frames = pickle.load(file)
    if not isinstance(frames, list) or not frames:
        raise ValueError("CMS filtered dataset 必须是非空 DataFrame 列表。")
    return frames


def derive_global_lineage_split(
    frames: list[pd.DataFrame],
    manifest_path: Path = GLOBAL_SPLIT_MANIFEST_PATH,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cms_ids = {_canonical_original_id(frame) for frame in frames}
    split_ids = {
        name: sorted(
            cms_ids
            & {
                str(value)
                for value in manifest["splits"][name]["original_ids"]
            }
        )
        for name in ("train", "val", "test")
    }
    union = set().union(*(set(values) for values in split_ids.values()))
    if union != cms_ids:
        raise RuntimeError(
            f"global split 未覆盖全部 CMS ID: {sorted(cms_ids - union)}"
        )
    if (
        set(split_ids["train"]) & set(split_ids["val"])
        or set(split_ids["train"]) & set(split_ids["test"])
        or set(split_ids["val"]) & set(split_ids["test"])
    ):
        raise RuntimeError("global lineage split original_ID 存在交集。")
    return {
        "schema_version": 1,
        "strategy": "inherit_frozen_global_original_id_lineage",
        "random_seed": manifest.get("random_seed", RANDOM_SEED),
        "source_manifest": str(
            manifest_path.resolve().relative_to(PROJECT_ROOT)
        ),
        "source_manifest_sha256": sha256_file(manifest_path),
        "splits": {
            "development": {
                "global_source_split": "train",
                "original_ids": split_ids["train"],
                "n_original_ids": len(split_ids["train"]),
            },
            "gate": {
                "global_source_split": "val",
                "original_ids": split_ids["val"],
                "n_original_ids": len(split_ids["val"]),
            },
            "confirmation": {
                "global_source_split": "test",
                "original_ids": split_ids["test"],
                "n_original_ids": len(split_ids["test"]),
            },
        },
        "pairwise_original_id_intersections": {
            "development_gate": 0,
            "development_confirmation": 0,
            "gate_confirmation": 0,
        },
    }


def validate_frozen_global() -> dict[str, Any]:
    if not GLOBAL_ONNX_PATH.is_file():
        raise FileNotFoundError(GLOBAL_ONNX_PATH)
    actual_sha256 = sha256_file(GLOBAL_ONNX_PATH)
    if actual_sha256 != EXPECTED_GLOBAL_ONNX_SHA256:
        raise RuntimeError(
            "frozen global ONNX SHA256 不匹配: "
            f"{actual_sha256} != {EXPECTED_GLOBAL_ONNX_SHA256}"
        )
    session = ort.InferenceSession(
        str(GLOBAL_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()
    output_meta = session.get_outputs()
    if len(input_meta) != 1 or len(output_meta) != 1:
        raise RuntimeError("frozen global ONNX 不是单输入单输出。")
    if (
        input_meta[0].name != "input"
        or input_meta[0].shape != ["batch_size", 6]
        or input_meta[0].type != "tensor(float)"
    ):
        raise RuntimeError(f"frozen global 输入接口异常: {input_meta[0]}")
    if (
        output_meta[0].name != "output"
        or output_meta[0].shape != ["batch_size", 2]
        or output_meta[0].type != "tensor(float)"
    ):
        raise RuntimeError(f"frozen global 输出接口异常: {output_meta[0]}")
    return {
        "path": str(GLOBAL_ONNX_PATH.resolve().relative_to(PROJECT_ROOT)),
        "sha256": actual_sha256,
        "input": {
            "name": input_meta[0].name,
            "shape": input_meta[0].shape,
            "dtype": input_meta[0].type,
        },
        "output": {
            "name": output_meta[0].name,
            "shape": output_meta[0].shape,
            "dtype": output_meta[0].type,
        },
    }


def build_adapter_data(
    frames: list[pd.DataFrame],
    selected_ids: Iterable[str],
    session: ort.InferenceSession,
    *,
    membership_function: Callable[
        [np.ndarray | pd.Series, np.ndarray | pd.Series],
        dict[str, np.ndarray],
    ] = region_memberships,
    required_membership: str | None = "CMS",
) -> AdapterData:
    selected_set = {str(value) for value in selected_ids}
    pieces = []
    found_ids: set[str] = set()
    for episode_index, source in enumerate(frames):
        original_id = _canonical_original_id(source)
        if original_id not in selected_set:
            continue
        found_ids.add(original_id)
        frame = source.copy()
        frame["original_ID"] = original_id
        frame["_cms_episode_key"] = f"{original_id}:{episode_index}"
        frame["_cms_episode_step"] = np.arange(len(frame), dtype=np.int64)
        pieces.append(frame)
    if found_ids != selected_set:
        raise RuntimeError(
            f"CMS 数据缺少 ID: {sorted(selected_set - found_ids)}"
        )
    if not pieces:
        raise ValueError("selected_ids 为空。")

    combined = pd.concat(pieces, ignore_index=True)
    required = {
        *CORE_FEATURES,
        "original_ID",
        "time",
        "latitude",
        "longitude",
        "ve",
        "vn",
        "cfsv2_u",
        "cfsv2_v",
    }
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"CMS adapter 数据缺列: {sorted(missing)}")
    if combined[list(required)].isna().any().any():
        raise ValueError("CMS adapter 必需列存在缺测。")
    combined["time"] = pd.to_datetime(combined["time"])

    features = combined[CORE_FEATURES].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    target = np.column_stack(
        (
            combined["ve"].to_numpy(dtype=np.float64)
            - combined["cfsv2_u"].to_numpy(dtype=np.float64),
            combined["vn"].to_numpy(dtype=np.float64)
            - combined["cfsv2_v"].to_numpy(dtype=np.float64),
        )
    )
    global_prediction = session.run(
        ["output"],
        {"input": features},
    )[0].astype(np.float64)
    memberships = membership_function(
        combined["latitude"],
        combined["longitude"],
    )
    if required_membership is not None:
        if required_membership not in memberships:
            raise KeyError(
                f"membership 缺少必需区域 {required_membership!r}。"
            )
        if not memberships[required_membership].all():
            raise RuntimeError(
                f"filtered 数据包含 {required_membership} 区域外行。"
            )
    return AdapterData(
        frame=combined,
        features=features.astype(np.float64),
        target=target,
        global_prediction=global_prediction,
        groups=combined["original_ID"].astype(str).to_numpy(),
        memberships=memberships,
    )


def combine_adapter_data(parts: list[AdapterData]) -> AdapterData:
    if not parts:
        raise ValueError("parts 不能为空。")
    return AdapterData(
        frame=pd.concat(
            [part.frame for part in parts],
            ignore_index=True,
        ),
        features=np.concatenate([part.features for part in parts]),
        target=np.concatenate([part.target for part in parts]),
        global_prediction=np.concatenate(
            [part.global_prediction for part in parts]
        ),
        groups=np.concatenate([part.groups for part in parts]),
        memberships={
            name: np.concatenate(
                [part.memberships[name] for part in parts]
            )
            for name in parts[0].memberships
        },
    )


def _basis(spec: AdapterSpec, data: AdapterData) -> np.ndarray:
    count = len(data.target)
    if spec.family == "none":
        return np.zeros((count, 2, 0), dtype=np.float64)
    if spec.family == "bias":
        basis = np.zeros((count, 2, 2), dtype=np.float64)
        basis[:, 0, 0] = 1.0
        basis[:, 1, 1] = 1.0
        return basis

    if spec.family == "wind_rotation":
        u10 = data.features[:, 0]
        v10 = data.features[:, 1]
        basis = np.zeros((count, 2, 4), dtype=np.float64)
        basis[:, 0, 0] = u10
        basis[:, 0, 1] = -v10
        basis[:, 1, 0] = v10
        basis[:, 1, 1] = u10
        basis[:, 0, 2] = 1.0
        basis[:, 1, 3] = 1.0
        return basis

    if spec.family in {"wind_full", "global_calibration"}:
        values = (
            data.features[:, :2]
            if spec.family == "wind_full"
            else data.global_prediction
        )
        basis = np.zeros((count, 2, 6), dtype=np.float64)
        basis[:, 0, 0:2] = values
        basis[:, 1, 2:4] = values
        basis[:, 0, 4] = 1.0
        basis[:, 1, 5] = 1.0
        return basis

    if spec.family == "core6":
        width = len(CORE_FEATURES)
        basis = np.zeros((count, 2, 2 * width + 2), dtype=np.float64)
        basis[:, 0, :width] = data.features
        basis[:, 1, width : 2 * width] = data.features
        basis[:, 0, -2] = 1.0
        basis[:, 1, -1] = 1.0
        return basis
    raise ValueError(f"未知 adapter family: {spec.family}")


def equal_id_row_weights(groups: np.ndarray) -> np.ndarray:
    groups = np.asarray(groups).astype(str)
    unique, counts = np.unique(groups, return_counts=True)
    count_map = dict(zip(unique.tolist(), counts.tolist()))
    weights = np.asarray(
        [1.0 / count_map[value] for value in groups],
        dtype=np.float64,
    )
    if not np.isclose(weights.sum(), len(unique)):
        raise RuntimeError("equal-ID row weight 归一化异常。")
    return weights


def fit_adapter(
    data: AdapterData,
    spec: AdapterSpec,
    lambda_value: float | None,
) -> FittedAdapter:
    training_ids = tuple(sorted(np.unique(data.groups).tolist()))
    if spec.family == "none":
        if lambda_value is not None:
            raise ValueError("G0 不接受 lambda。")
        return FittedAdapter(
            spec_name=spec.name,
            lambda_value=None,
            coefficients=(),
            basis_rms_scales=(),
            training_original_ids=training_ids,
        )
    if lambda_value is None or lambda_value < 0:
        raise ValueError("Ridge adapter 需要非负 lambda。")

    basis = _basis(spec, data)
    design = basis.reshape(-1, spec.parameter_count)
    target = (
        data.target - data.global_prediction
    ).reshape(-1)
    row_weights = equal_id_row_weights(data.groups)
    weights = np.repeat(row_weights / 2.0, 2)
    weights = weights / weights.sum()

    rms = np.sqrt(np.sum(weights[:, None] * design**2, axis=0))
    rms = np.where(rms > 1e-12, rms, 1.0)
    scaled = design / rms
    if lambda_value == 0:
        root_weight = np.sqrt(weights)
        coefficient_scaled = np.linalg.lstsq(
            scaled * root_weight[:, None],
            target * root_weight,
            rcond=None,
        )[0]
    else:
        gram = scaled.T @ (weights[:, None] * scaled)
        rhs = scaled.T @ (weights * target)
        coefficient_scaled = np.linalg.solve(
            gram + lambda_value * np.eye(spec.parameter_count),
            rhs,
        )
    coefficients = coefficient_scaled / rms
    if not np.all(np.isfinite(coefficients)):
        raise RuntimeError(f"{spec.name} 拟合产生非有限系数。")
    return FittedAdapter(
        spec_name=spec.name,
        lambda_value=float(lambda_value),
        coefficients=tuple(float(value) for value in coefficients),
        basis_rms_scales=tuple(float(value) for value in rms),
        training_original_ids=training_ids,
    )


def predict_correction(
    adapter: FittedAdapter,
    data: AdapterData,
) -> np.ndarray:
    spec = ADAPTER_SPECS[adapter.spec_name]
    if spec.family == "none":
        return np.zeros_like(data.global_prediction)
    coefficients = np.asarray(adapter.coefficients, dtype=np.float64)
    if len(coefficients) != spec.parameter_count:
        raise ValueError("adapter 系数长度与 spec 不匹配。")
    return np.einsum("nop,p->no", _basis(spec, data), coefficients)


def correction_magnitude_summary(
    correction: np.ndarray,
    global_prediction: np.ndarray,
) -> dict[str, float]:
    correction_norm = np.linalg.norm(correction, axis=1)
    global_norm = np.linalg.norm(global_prediction, axis=1)
    correction_p99 = float(np.quantile(correction_norm, 0.99))
    global_p99 = float(np.quantile(global_norm, 0.99))
    return {
        "correction_mean_magnitude": float(correction_norm.mean()),
        "correction_rms_magnitude": float(
            np.sqrt(np.mean(correction_norm**2))
        ),
        "correction_p95_magnitude": float(
            np.quantile(correction_norm, 0.95)
        ),
        "correction_p99_magnitude": correction_p99,
        "correction_max_magnitude": float(correction_norm.max()),
        "global_p99_magnitude": global_p99,
        "correction_to_global_p99_ratio": (
            correction_p99 / global_p99
            if global_p99 > 0
            else float("inf")
        ),
    }


def compare_predictions(
    data: AdapterData,
    adapted_prediction: np.ndarray,
) -> dict[str, Any]:
    if adapted_prediction.shape != data.target.shape:
        raise ValueError("adapted_prediction shape 异常。")
    per_id = []
    for original_id in sorted(np.unique(data.groups)):
        mask = data.groups == original_id
        base_error = data.global_prediction[mask] - data.target[mask]
        adapted_error = adapted_prediction[mask] - data.target[mask]
        base_rmse = float(np.sqrt(np.mean(base_error**2)))
        adapted_rmse = float(np.sqrt(np.mean(adapted_error**2)))
        per_id.append(
            {
                "original_ID": str(original_id),
                "n_samples": int(mask.sum()),
                "base_mse": float(np.mean(base_error**2)),
                "adapted_mse": float(np.mean(adapted_error**2)),
                "delta_mse": float(
                    np.mean(adapted_error**2) - np.mean(base_error**2)
                ),
                "base_rmse": base_rmse,
                "adapted_rmse": adapted_rmse,
                "delta_rmse": adapted_rmse - base_rmse,
                "relative_rmse_change": (
                    (adapted_rmse - base_rmse) / base_rmse
                    if base_rmse > 0
                    else 0.0
                ),
            }
        )
    base_rmse_values = np.asarray(
        [value["base_rmse"] for value in per_id],
        dtype=np.float64,
    )
    adapted_rmse_values = np.asarray(
        [value["adapted_rmse"] for value in per_id],
        dtype=np.float64,
    )
    delta_mse_values = np.asarray(
        [value["delta_mse"] for value in per_id],
        dtype=np.float64,
    )
    adapted_error = adapted_prediction - data.target
    base_error = data.global_prediction - data.target
    return {
        "n_samples": len(data.target),
        "n_original_ids": len(per_id),
        "base_point_metrics": regression_metrics(
            data.target,
            data.global_prediction,
        ),
        "adapted_point_metrics": regression_metrics(
            data.target,
            adapted_prediction,
        ),
        "base_bias": base_error.mean(axis=0).tolist(),
        "adapted_bias": adapted_error.mean(axis=0).tolist(),
        "macro_id_base_rmse": float(base_rmse_values.mean()),
        "macro_id_adapted_rmse": float(adapted_rmse_values.mean()),
        "macro_id_relative_rmse_improvement": float(
            (
                base_rmse_values.mean() - adapted_rmse_values.mean()
            )
            / base_rmse_values.mean()
        ),
        "id_win_rate": float(
            np.mean(adapted_rmse_values < base_rmse_values)
        ),
        "maximum_id_relative_rmse_degradation": float(
            max(value["relative_rmse_change"] for value in per_id)
        ),
        "mean_id_delta_mse": float(delta_mse_values.mean()),
        "se_id_delta_mse": float(
            delta_mse_values.std(ddof=1) / np.sqrt(len(delta_mse_values))
            if len(delta_mse_values) > 1
            else 0.0
        ),
        "per_id": per_id,
    }


def compare_by_region(
    data: AdapterData,
    adapted_prediction: np.ndarray,
    *,
    region_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    result = {}
    if region_names is None:
        preferred = ("CMS", "BYS", "ECS", "NSCS")
        names = [
            name for name in preferred
            if name in data.memberships
        ]
        names.extend(
            name for name in data.memberships
            if name not in names
        )
    else:
        names = list(region_names)
    for name in names:
        if name not in data.memberships:
            raise KeyError(f"未知 membership: {name}")
        mask = data.memberships[name]
        if not mask.any():
            result[name] = {
                "status": "no_samples",
                "n_samples": 0,
                "n_original_ids": 0,
            }
            continue
        subset = data.subset(np.flatnonzero(mask))
        result[name] = {
            "status": "ok",
            **compare_predictions(subset, adapted_prediction[mask]),
        }
    return result


def _splitter(
    data: AdapterData,
    n_splits: int,
    seed: int,
) -> GroupKFold:
    n_groups = len(np.unique(data.groups))
    if n_groups < 2:
        raise ValueError("GroupKFold 至少需要2个 original_ID。")
    return GroupKFold(
        n_splits=min(n_splits, n_groups),
        shuffle=True,
        random_state=seed,
    )


def candidate_pairs() -> list[tuple[AdapterSpec, float | None]]:
    pairs = [(ADAPTER_SPECS["G0_global_only"], None)]
    for spec in list(ADAPTER_SPECS.values())[1:]:
        pairs.extend((spec, value) for value in LAMBDA_GRID)
    return pairs


def pair_key(spec: AdapterSpec, lambda_value: float | None) -> str:
    return (
        spec.name
        if lambda_value is None
        else f"{spec.name}|lambda={lambda_value:g}"
    )


def cross_validate_pair(
    data: AdapterData,
    spec: AdapterSpec,
    lambda_value: float | None,
    *,
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    prediction = np.empty_like(data.target)
    seen = np.zeros(len(data.target), dtype=bool)
    fold_manifest = []
    splitter = _splitter(data, n_splits, seed)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(data.features, groups=data.groups),
        start=1,
    ):
        train_data = data.subset(train_index)
        validation_data = data.subset(validation_index)
        if set(train_data.groups) & set(validation_data.groups):
            raise RuntimeError(f"fold {fold} original_ID 泄漏。")
        adapter = fit_adapter(
            train_data,
            spec,
            lambda_value,
        )
        prediction[validation_index] = (
            validation_data.global_prediction
            + predict_correction(adapter, validation_data)
        )
        seen[validation_index] = True
        fold_manifest.append(
            {
                "fold": fold,
                "train_original_ids": sorted(
                    np.unique(train_data.groups).tolist()
                ),
                "validation_original_ids": sorted(
                    np.unique(validation_data.groups).tolist()
                ),
                "n_train_samples": len(train_data.target),
                "n_validation_samples": len(validation_data.target),
            }
        )
    if not seen.all():
        raise RuntimeError("CV 未覆盖全部开发样本。")
    comparison = compare_predictions(data, prediction)
    return {
        "key": pair_key(spec, lambda_value),
        "adapter_name": spec.name,
        "lambda": lambda_value,
        "parameter_count": spec.parameter_count,
        "mean_id_delta_mse": comparison["mean_id_delta_mse"],
        "se_id_delta_mse": comparison["se_id_delta_mse"],
        "comparison": comparison,
        "fold_manifest": fold_manifest,
        "_prediction": prediction,
    }


def select_candidate_by_cv(
    data: AdapterData,
    *,
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    results = {}
    for spec, lambda_value in candidate_pairs():
        result = cross_validate_pair(
            data,
            spec,
            lambda_value,
            n_splits=n_splits,
            seed=seed,
        )
        results[result["key"]] = result
    best = min(results.values(), key=lambda value: value["mean_id_delta_mse"])
    threshold = best["mean_id_delta_mse"] + best["se_id_delta_mse"]
    eligible = [
        value
        for value in results.values()
        if value["mean_id_delta_mse"] <= threshold
    ]
    selected = min(
        eligible,
        key=lambda value: (
            value["parameter_count"],
            -(
                float(value["lambda"])
                if value["lambda"] is not None
                else float("inf")
            ),
            value["adapter_name"],
        ),
    )
    serializable_results = {
        key: {
            result_key: result_value
            for result_key, result_value in value.items()
            if result_key != "_prediction"
        }
        for key, value in results.items()
    }
    return {
        "best_mean_candidate": best["key"],
        "best_mean_id_delta_mse": best["mean_id_delta_mse"],
        "best_standard_error": best["se_id_delta_mse"],
        "one_standard_error_threshold": threshold,
        "eligible_candidates": sorted(value["key"] for value in eligible),
        "selected_key": selected["key"],
        "selected_adapter_name": selected["adapter_name"],
        "selected_lambda": selected["lambda"],
        "candidate_results": serializable_results,
    }


def run_nested_development_cv(
    data: AdapterData,
    *,
    outer_splits: int = OUTER_SPLITS,
    inner_splits: int = INNER_SPLITS,
    seed: int = RANDOM_SEED,
    model_version: str = MODEL_VERSION,
) -> dict[str, Any]:
    prediction = np.empty_like(data.target)
    correction = np.empty_like(data.target)
    seen = np.zeros(len(data.target), dtype=bool)
    outer_results = []
    splitter = _splitter(data, outer_splits, seed)
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(data.features, groups=data.groups),
        start=1,
    ):
        outer_train = data.subset(train_index)
        outer_validation = data.subset(validation_index)
        selection = select_candidate_by_cv(
            outer_train,
            n_splits=inner_splits,
            seed=seed + fold,
        )
        spec = ADAPTER_SPECS[selection["selected_adapter_name"]]
        adapter = fit_adapter(
            outer_train,
            spec,
            selection["selected_lambda"],
        )
        fold_correction = predict_correction(adapter, outer_validation)
        prediction[validation_index] = (
            outer_validation.global_prediction + fold_correction
        )
        correction[validation_index] = fold_correction
        seen[validation_index] = True
        outer_results.append(
            {
                "fold": fold,
                "train_original_ids": sorted(
                    np.unique(outer_train.groups).tolist()
                ),
                "validation_original_ids": sorted(
                    np.unique(outer_validation.groups).tolist()
                ),
                "selection": {
                    key: value
                    for key, value in selection.items()
                    if key != "candidate_results"
                },
                "selected_adapter": adapter.to_dict(
                    model_version=model_version,
                ),
                "validation_comparison": compare_predictions(
                    outer_validation,
                    outer_validation.global_prediction + fold_correction,
                ),
            }
        )
    if not seen.all():
        raise RuntimeError("nested outer CV 未覆盖全部开发样本。")
    comparison = compare_predictions(data, prediction)
    regions = compare_by_region(data, prediction)
    magnitude = correction_magnitude_summary(
        correction,
        data.global_prediction,
    )
    family_counts = {
        name: sum(
            value["selection"]["selected_adapter_name"] == name
            for value in outer_results
        )
        for name in ADAPTER_SPECS
    }
    return {
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "random_seed": seed,
        "comparison": comparison,
        "regions": regions,
        "correction_magnitude": magnitude,
        "selected_family_counts": family_counts,
        "outer_folds": outer_results,
        "_prediction": prediction,
        "_correction": correction,
    }


def trajectory_proxy(
    data: AdapterData,
    adapted_prediction: np.ndarray,
    *,
    horizons: tuple[int, ...] = TRAJECTORY_HORIZONS_HOURS,
) -> dict[str, Any]:
    base_error = data.global_prediction - data.target
    adapted_error = adapted_prediction - data.target
    records: dict[int, list[dict[str, Any]]] = {
        horizon: [] for horizon in horizons
    }
    for episode_key, positions in data.frame.groupby(
        "_cms_episode_key",
        sort=True,
    ).groups.items():
        index = np.asarray(list(positions), dtype=np.int64)
        order = np.argsort(
            data.frame.loc[index, "_cms_episode_step"].to_numpy()
        )
        index = index[order]
        times = data.frame.loc[index, "time"].to_numpy(dtype="datetime64[s]")
        if len(times) > 1:
            differences = np.diff(times).astype("timedelta64[s]").astype(int)
            if not np.all(differences == 3600):
                raise RuntimeError(f"episode {episode_key} 不是连续小时序列。")
        original_id = str(data.groups[index[0]])
        for horizon in horizons:
            for start in range(0, len(index) - horizon + 1, horizon):
                window = index[start : start + horizon]
                base_displacement = (
                    base_error[window].sum(axis=0) * 3600.0 / 1000.0
                )
                adapted_displacement = (
                    adapted_error[window].sum(axis=0) * 3600.0 / 1000.0
                )
                records[horizon].append(
                    {
                        "original_ID": original_id,
                        "episode_key": str(episode_key),
                        "start_step": int(start),
                        "base_endpoint_error_km": float(
                            np.linalg.norm(base_displacement)
                        ),
                        "adapted_endpoint_error_km": float(
                            np.linalg.norm(adapted_displacement)
                        ),
                    }
                )

    summaries = {}
    for horizon, values in records.items():
        if not values:
            summaries[str(horizon)] = {
                "status": "no_windows",
                "n_windows": 0,
                "n_original_ids": 0,
                "per_id": [],
            }
            continue
        per_id = []
        for original_id in sorted(
            {value["original_ID"] for value in values}
        ):
            selected = [
                value for value in values
                if value["original_ID"] == original_id
            ]
            base_values = np.asarray(
                [value["base_endpoint_error_km"] for value in selected]
            )
            adapted_values = np.asarray(
                [
                    value["adapted_endpoint_error_km"]
                    for value in selected
                ]
            )
            per_id.append(
                {
                    "original_ID": original_id,
                    "n_windows": len(selected),
                    "base_median_km": float(np.median(base_values)),
                    "adapted_median_km": float(
                        np.median(adapted_values)
                    ),
                    "base_p90_km": float(np.quantile(base_values, 0.9)),
                    "adapted_p90_km": float(
                        np.quantile(adapted_values, 0.9)
                    ),
                }
            )
        summaries[str(horizon)] = {
            "status": "ok",
            "n_windows": len(values),
            "n_original_ids": len(per_id),
            "macro_id_base_median_km": float(
                np.mean([value["base_median_km"] for value in per_id])
            ),
            "macro_id_adapted_median_km": float(
                np.mean(
                    [value["adapted_median_km"] for value in per_id]
                )
            ),
            "macro_id_base_p90_km": float(
                np.mean([value["base_p90_km"] for value in per_id])
            ),
            "macro_id_adapted_p90_km": float(
                np.mean([value["adapted_p90_km"] for value in per_id])
            ),
            "per_id": per_id,
        }
    return {
        "definition": (
            "Non-overlapping windows accumulate (predicted residual - "
            "observed residual) * 3600 seconds; background current cancels."
        ),
        "horizons_hours": list(horizons),
        "summaries": summaries,
    }
