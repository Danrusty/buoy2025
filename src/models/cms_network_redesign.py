"""CMS core6 小样本网络重构与严格 test-seal 评价。"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import shutil
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from cms_regional import (
    CORE_FEATURES,
    FILTERED_DATA_PATH,
    GLOBAL_ONNX_PATH,
    region_memberships,
)
from data_loader import PROJECT_ROOT
from evaluation import regression_metrics
from train_mlp import (
    BATCH_SIZE,
    EPOCHS,
    LR,
    LR_MIN,
    MIN_DELTA,
    PATIENCE,
    RANDOM_SEED,
    GpuTensorDataset,
    ResidualMLP,
)


STUDY_NAME = "wdf_cms_core6_network_redesign_v1"
ARTIFACT_DIR = PROJECT_ROOT / "trained_models" / STUDY_NAME
RESULT_DIR = PROJECT_ROOT / "results" / STUDY_NAME
SOURCE_SPLIT_MANIFEST = (
    PROJECT_ROOT
    / "trained_models"
    / "wdf_cms_orig_core6_v1"
    / "split_manifest.json"
)
SOURCE_CMS_METRICS = (
    PROJECT_ROOT
    / "trained_models"
    / "wdf_cms_orig_core6_v1"
    / "mlp_metrics.json"
)
SOURCE_LINEAR_METRICS = (
    PROJECT_ROOT
    / "trained_models"
    / "wdf_cms_orig_core6_v1"
    / "linear_baseline_metrics.json"
)
N_CV_SPLITS = 5
N_FIXED_VAL_FINALISTS = 2

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    family: str
    hidden_sizes: tuple[int, ...]
    normalization: str
    residual_blocks: int
    description: str


class PlainCore6MLP(nn.Module):
    """无 BatchNorm/Dropout 的小型 ReLU MLP。"""

    def __init__(
        self,
        hidden_sizes: tuple[int, ...],
        normalization: str = "none",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        input_size = len(CORE_FEATURES)
        for width in hidden_sizes:
            layers.append(nn.Linear(input_size, width))
            if normalization == "layernorm":
                layers.append(nn.LayerNorm(width))
            elif normalization != "none":
                raise ValueError(f"未知 normalization: {normalization}")
            layers.append(nn.ReLU())
            input_size = width
        layers.append(nn.Linear(input_size, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class ResidualCore6MLP(nn.Module):
    """小宽度 LayerNorm residual MLP。"""

    def __init__(self, width: int = 64, n_blocks: int = 2):
        super().__init__()
        self.input_projection = nn.Linear(len(CORE_FEATURES), width)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(width),
                    nn.Linear(width, width),
                    nn.ReLU(),
                    nn.Linear(width, width),
                )
                for _ in range(n_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.input_projection(values))
        for block in self.blocks:
            hidden = torch.relu(hidden + block(hidden))
        return self.output(self.output_norm(hidden))


class LinearSkipCore6MLP(nn.Module):
    """线性 WDF 直连加小型非线性修正支路。"""

    def __init__(self, hidden_sizes: tuple[int, ...]):
        super().__init__()
        self.linear = nn.Linear(len(CORE_FEATURES), 2)
        self.nonlinear = PlainCore6MLP(
            hidden_sizes=hidden_sizes,
            normalization="none",
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.linear(values) + self.nonlinear(values)


ARCHITECTURES: OrderedDict[str, ArchitectureSpec] = OrderedDict(
    (
        spec.name,
        spec,
    )
    for spec in (
        ArchitectureSpec(
            name="linear_core6",
            family="plain",
            hidden_sizes=(),
            normalization="none",
            residual_blocks=0,
            description="14-parameter linear neural control using all core6 inputs.",
        ),
        ArchitectureSpec(
            name="plain_16_16",
            family="plain",
            hidden_sizes=(16, 16),
            normalization="none",
            residual_blocks=0,
            description="Very small two-layer ReLU MLP without normalization.",
        ),
        ArchitectureSpec(
            name="plain_32_32",
            family="plain",
            hidden_sizes=(32, 32),
            normalization="none",
            residual_blocks=0,
            description="Small two-layer ReLU MLP without normalization.",
        ),
        ArchitectureSpec(
            name="plain_64_32",
            family="plain",
            hidden_sizes=(64, 32),
            normalization="none",
            residual_blocks=0,
            description="Moderate tapered ReLU MLP without normalization.",
        ),
        ArchitectureSpec(
            name="linear_skip_32_32",
            family="linear_skip",
            hidden_sizes=(32, 32),
            normalization="none",
            residual_blocks=0,
            description=(
                "Direct linear WDF path plus a small nonlinear correction."
            ),
        ),
        ArchitectureSpec(
            name="linear_skip_64_32",
            family="linear_skip",
            hidden_sizes=(64, 32),
            normalization="none",
            residual_blocks=0,
            description=(
                "Direct linear WDF path plus a moderate nonlinear correction."
            ),
        ),
        ArchitectureSpec(
            name="layernorm_32_32",
            family="plain",
            hidden_sizes=(32, 32),
            normalization="layernorm",
            residual_blocks=0,
            description="Small MLP with batch-size-independent LayerNorm.",
        ),
        ArchitectureSpec(
            name="layernorm_64_32",
            family="plain",
            hidden_sizes=(64, 32),
            normalization="layernorm",
            residual_blocks=0,
            description="Moderate tapered MLP with LayerNorm.",
        ),
        ArchitectureSpec(
            name="residual_layernorm_64",
            family="residual",
            hidden_sizes=(64,),
            normalization="layernorm",
            residual_blocks=2,
            description="Two small residual blocks with LayerNorm.",
        ),
        ArchitectureSpec(
            name="legacy_core6_433k",
            family="legacy",
            hidden_sizes=(512, 512, 256, 128),
            normalization="batchnorm",
            residual_blocks=0,
            description="Frozen 433k BatchNorm/Dropout network reference.",
        ),
    )
)


def build_model(spec: ArchitectureSpec) -> nn.Module:
    if spec.family == "plain":
        return PlainCore6MLP(
            hidden_sizes=spec.hidden_sizes,
            normalization=spec.normalization,
        )
    if spec.family == "residual":
        return ResidualCore6MLP(
            width=spec.hidden_sizes[0],
            n_blocks=spec.residual_blocks,
        )
    if spec.family == "linear_skip":
        return LinearSkipCore6MLP(hidden_sizes=spec.hidden_sizes)
    if spec.family == "legacy":
        return ResidualMLP(input_size=len(CORE_FEATURES), output_size=2)
    raise ValueError(f"未知 architecture family: {spec.family}")


def _json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_all() -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _canonical_id(frame: pd.DataFrame) -> str:
    values = frame["original_ID"].dropna().astype(str).str.strip().unique()
    if len(values) != 1:
        raise ValueError(f"episode original_ID 异常: {values[:5].tolist()}")
    return str(values[0])


def _load_frames_and_manifest() -> tuple[list[pd.DataFrame], dict[str, Any]]:
    for path in (FILTERED_DATA_PATH, SOURCE_SPLIT_MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)
    with FILTERED_DATA_PATH.open("rb") as file:
        frames = pickle.load(file)
    manifest = json.loads(SOURCE_SPLIT_MANIFEST.read_text(encoding="utf-8"))
    return frames, manifest


def _frames_for_ids(
    frames: list[pd.DataFrame],
    selected_ids: set[str],
) -> pd.DataFrame:
    selected = [
        frame for frame in frames if _canonical_id(frame) in selected_ids
    ]
    found_ids = {_canonical_id(frame) for frame in selected}
    if found_ids != selected_ids:
        raise ValueError(
            f"数据未完整覆盖 ID: {sorted(selected_ids - found_ids)}"
        )
    combined = pd.concat(selected, ignore_index=True)
    required = {
        *CORE_FEATURES,
        "original_ID",
        "ve",
        "vn",
        "cfsv2_u",
        "cfsv2_v",
        "latitude",
        "longitude",
    }
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"CMS 数据缺列: {sorted(missing)}")
    clean = combined.dropna(subset=list(required)).copy()
    if len(clean) != len(combined):
        raise ValueError("CMS 架构研究数据存在必需列缺测。")
    clean["original_ID"] = clean["original_ID"].astype(str)
    clean["residual_u"] = clean["ve"] - clean["cfsv2_u"]
    clean["residual_v"] = clean["vn"] - clean["cfsv2_v"]
    return clean


def _arrays(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
) -> dict[str, Any]:
    scaler = StandardScaler()
    x_train_raw = train_frame[CORE_FEATURES].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    x_validation_raw = validation_frame[CORE_FEATURES].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    return {
        "X_train": scaler.fit_transform(x_train_raw).astype(
            np.float32,
            copy=False,
        ),
        "y_train": train_frame[["residual_u", "residual_v"]].to_numpy(
            dtype=np.float32,
            copy=True,
        ),
        "X_validation": scaler.transform(x_validation_raw).astype(
            np.float32,
            copy=False,
        ),
        "y_validation": validation_frame[
            ["residual_u", "residual_v"]
        ].to_numpy(dtype=np.float32, copy=True),
        "scaler": scaler,
    }


@torch.no_grad()
def _predict(
    model: nn.Module,
    values: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions = []
    for start in range(0, len(values), BATCH_SIZE):
        batch = torch.from_numpy(values[start : start + BATCH_SIZE]).to(
            device=device,
            dtype=torch.float32,
        )
        predictions.append(model(batch).cpu().numpy())
    return np.concatenate(predictions)


@torch.no_grad()
def _evaluate_with_predictions(
    model: nn.Module,
    dataset: GpuTensorDataset,
    criterion: nn.Module,
) -> tuple[float, np.ndarray]:
    """完全复用冻结流程的 batch 加权 PyTorch MSE 语义。"""
    model.eval()
    predictions = []
    loss_sum = 0.0
    n_samples = 0
    for x_batch, y_batch in dataset:
        prediction = model(x_batch)
        loss_sum += criterion(prediction, y_batch).item() * len(x_batch)
        n_samples += len(x_batch)
        predictions.append(prediction.cpu().numpy())
    return loss_sum / n_samples, np.concatenate(predictions)


def train_candidate(
    spec: ArchitectureSpec,
    arrays: dict[str, Any],
) -> dict[str, Any]:
    """使用与冻结 core6 相同的训练超参数训练一个结构候选。"""
    _seed_all()
    device = _device()
    model = build_model(spec).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    train_dataset = GpuTensorDataset(
        arrays["X_train"],
        arrays["y_train"],
        device,
        BATCH_SIZE,
        shuffle=True,
    )
    validation_dataset = GpuTensorDataset(
        arrays["X_validation"],
        arrays["y_validation"],
        device,
        BATCH_SIZE,
        shuffle=False,
    )
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=60,
        T_mult=2,
        eta_min=LR_MIN,
    )

    best_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    best_state: dict[str, torch.Tensor] | None = None
    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_r2_joint": [],
        "learning_rate": [],
    }

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        n_samples = 0
        for x_batch, y_batch in train_dataset:
            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(x_batch)
            n_samples += len(x_batch)
        train_loss = loss_sum / n_samples

        validation_loss, validation_prediction = _evaluate_with_predictions(
            model,
            validation_dataset,
            criterion,
        )
        validation_metrics = regression_metrics(
            arrays["y_validation"],
            validation_prediction,
        )
        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["validation_r2_joint"].append(
            validation_metrics["r2_joint"]
        )
        history["learning_rate"].append(current_lr)
        scheduler.step(epoch)

        if validation_loss < best_loss - MIN_DELTA:
            best_loss = validation_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    if best_state is None:
        raise RuntimeError(f"{spec.name} 未生成 checkpoint。")
    model.load_state_dict(best_state)
    _, final_prediction = _evaluate_with_predictions(
        model,
        validation_dataset,
        criterion,
    )
    final_metrics = regression_metrics(
        arrays["y_validation"],
        final_prediction,
    )
    result = {
        "architecture": asdict(spec),
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "validation_metrics": final_metrics,
        "history": history,
        "state_dict": best_state,
    }
    del train_dataset, validation_dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _serializable_training_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"state_dict", "history"}
    } | {"history": result["history"]}


def _prepare_cv_folds(
    train_frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    splitter = GroupKFold(
        n_splits=N_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    groups = train_frame["original_ID"].to_numpy()
    folds = []
    for fold_index, (train_index, validation_index) in enumerate(
        splitter.split(train_frame, groups=groups),
        start=1,
    ):
        fold_train = train_frame.iloc[train_index].copy()
        fold_validation = train_frame.iloc[validation_index].copy()
        train_ids = sorted(fold_train["original_ID"].unique().tolist())
        validation_ids = sorted(
            fold_validation["original_ID"].unique().tolist()
        )
        if set(train_ids) & set(validation_ids):
            raise RuntimeError(f"CV fold {fold_index} original_ID 泄漏。")
        folds.append(
            {
                "fold": fold_index,
                "train_ids": train_ids,
                "validation_ids": validation_ids,
                "arrays": _arrays(fold_train, fold_validation),
            }
        )
    return folds


def _rank_cv(results: dict[str, Any]) -> list[str]:
    return sorted(
        results,
        key=lambda name: (
            -results[name]["summary"]["mean_r2_joint"],
            results[name]["summary"]["mean_rmse"],
            results[name]["summary"]["parameter_count"],
        ),
    )


def _rank_fixed_validation(results: dict[str, Any]) -> list[str]:
    return sorted(
        results,
        key=lambda name: (
            -results[name]["validation_metrics"]["r2_joint"],
            results[name]["validation_metrics"]["rmse"],
            results[name]["parameter_count"],
        ),
    )


def _training_contract() -> dict[str, Any]:
    return {
        "features": CORE_FEATURES,
        "target": {
            "residual_u": "ve - cfsv2_u",
            "residual_v": "vn - cfsv2_v",
        },
        "loss": "MSELoss",
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "checkpoint_monitor": "validation_loss",
        "optimizer": "AdamW",
        "learning_rate": LR,
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingWarmRestarts(T_0=60,T_mult=2)",
        "minimum_learning_rate": LR_MIN,
        "random_seed": RANDOM_SEED,
        "input_scaler": "StandardScaler fit on each training fold only",
        "forbidden_changes": [
            "fixed 0.03 correction",
            "regional target correction",
            "LSTM or sequence model",
            "new features",
            "automatic hyperparameter search",
            "subregion-specific models",
            "ensemble deployment",
        ],
    }


def run_selection(code_commit: str) -> dict[str, Any]:
    """在 test 封存状态下完成 CV 排序和固定 validation 决选。"""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    test_output = ARTIFACT_DIR / "test_evaluation.json"
    if test_output.exists():
        raise FileExistsError(
            "test_evaluation.json 已存在；拒绝重新进行架构选择。"
        )

    frames, manifest = _load_frames_and_manifest()
    split_ids = {
        name: set(manifest["splits"][name]["original_ids"])
        for name in ("train", "val", "test")
    }
    if (
        split_ids["train"] & split_ids["val"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["val"] & split_ids["test"]
    ):
        raise RuntimeError("源 split manifest original_ID 存在交集。")

    train_frame = _frames_for_ids(frames, split_ids["train"])
    validation_frame = _frames_for_ids(frames, split_ids["val"])
    del frames
    if set(train_frame["original_ID"]) & split_ids["test"]:
        raise RuntimeError("test ID 进入 train frame。")
    if set(validation_frame["original_ID"]) & split_ids["test"]:
        raise RuntimeError("test ID 进入 validation frame。")

    folds = _prepare_cv_folds(train_frame)
    fold_manifest = {
        "n_splits": N_CV_SPLITS,
        "source_population": "fixed CMS train IDs only",
        "folds": [
            {
                "fold": fold["fold"],
                "train_ids": fold["train_ids"],
                "validation_ids": fold["validation_ids"],
                "n_train_samples": len(fold["arrays"]["X_train"]),
                "n_validation_samples": len(
                    fold["arrays"]["X_validation"]
                ),
            }
            for fold in folds
        ],
    }
    _json_write(ARTIFACT_DIR / "cv_fold_manifest.json", fold_manifest)

    cv_results: dict[str, Any] = {}
    for name, spec in ARCHITECTURES.items():
        logger.info("CV architecture: %s", name)
        fold_results = []
        for fold in folds:
            trained = train_candidate(spec, fold["arrays"])
            fold_results.append(
                {
                    "fold": fold["fold"],
                    "train_ids": fold["train_ids"],
                    "validation_ids": fold["validation_ids"],
                    **{
                        key: value
                        for key, value in _serializable_training_result(
                            trained
                        ).items()
                        if key != "history"
                    },
                }
            )
        r2_values = np.asarray(
            [
                value["validation_metrics"]["r2_joint"]
                for value in fold_results
            ],
            dtype=np.float64,
        )
        rmse_values = np.asarray(
            [
                value["validation_metrics"]["rmse"]
                for value in fold_results
            ],
            dtype=np.float64,
        )
        cv_results[name] = {
            "architecture": asdict(spec),
            "summary": {
                "parameter_count": fold_results[0]["parameter_count"],
                "mean_r2_joint": float(r2_values.mean()),
                "std_r2_joint": float(r2_values.std(ddof=0)),
                "minimum_r2_joint": float(r2_values.min()),
                "maximum_r2_joint": float(r2_values.max()),
                "mean_rmse": float(rmse_values.mean()),
                "std_rmse": float(rmse_values.std(ddof=0)),
            },
            "folds": fold_results,
        }
        _json_write(
            ARTIFACT_DIR / "cv_results_partial.json",
            {
                "completed_architectures": list(cv_results),
                "results": cv_results,
            },
        )

    cv_ranking = _rank_cv(cv_results)
    finalists = cv_ranking[:N_FIXED_VAL_FINALISTS]
    fixed_arrays = _arrays(train_frame, validation_frame)
    fixed_results: dict[str, Any] = {}
    for name in finalists:
        spec = ARCHITECTURES[name]
        logger.info("Fixed validation finalist: %s", name)
        trained = train_candidate(spec, fixed_arrays)
        candidate_dir = ARTIFACT_DIR / "finalists" / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = candidate_dir / "best_mlp.pth"
        scaler_path = candidate_dir / "x_scaler.pkl"
        torch.save(trained["state_dict"], checkpoint_path)
        joblib.dump(fixed_arrays["scaler"], scaler_path)
        _json_write(candidate_dir / "training_history.json", trained["history"])
        fixed_results[name] = {
            **{
                key: value
                for key, value in _serializable_training_result(trained).items()
                if key != "history"
            },
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "scaler": str(scaler_path.relative_to(PROJECT_ROOT)),
            "scaler_sha256": _sha256(scaler_path),
        }

    fixed_ranking = _rank_fixed_validation(fixed_results)
    selected_name = fixed_ranking[0]
    selected_dir = ARTIFACT_DIR / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    source_dir = ARTIFACT_DIR / "finalists" / selected_name
    selected_checkpoint = selected_dir / "best_mlp.pth"
    selected_scaler = selected_dir / "x_scaler.pkl"
    shutil.copy2(source_dir / "best_mlp.pth", selected_checkpoint)
    shutil.copy2(source_dir / "x_scaler.pkl", selected_scaler)
    shutil.copy2(
        source_dir / "training_history.json",
        selected_dir / "training_history.json",
    )
    model_config = {
        "study": STUDY_NAME,
        "selected_architecture": asdict(ARCHITECTURES[selected_name]),
        "parameter_count": fixed_results[selected_name]["parameter_count"],
        "training_contract": _training_contract(),
        "selection_method": (
            "5-fold original_ID GroupKFold on fixed train IDs; top two "
            "evaluated on fixed validation IDs; test IDs sealed"
        ),
    }
    _json_write(selected_dir / "model_config.json", model_config)

    selection_lock = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": STUDY_NAME,
        "selection_code_git_commit": code_commit,
        "test_status": "sealed_not_evaluated",
        "test_original_ids": sorted(split_ids["test"]),
        "test_original_ids_sha256": hashlib.sha256(
            "\n".join(sorted(split_ids["test"])).encode("utf-8")
        ).hexdigest(),
        "training_contract": _training_contract(),
        "candidate_registry": {
            name: asdict(spec) for name, spec in ARCHITECTURES.items()
        },
        "cv_ranking": cv_ranking,
        "cv_results": cv_results,
        "fixed_validation_finalists": finalists,
        "fixed_validation_ranking": fixed_ranking,
        "fixed_validation_results": fixed_results,
        "selected_architecture": selected_name,
        "selected_checkpoint": str(
            selected_checkpoint.relative_to(PROJECT_ROOT)
        ),
        "selected_checkpoint_sha256": _sha256(selected_checkpoint),
        "selected_scaler": str(selected_scaler.relative_to(PROJECT_ROOT)),
        "selected_scaler_sha256": _sha256(selected_scaler),
    }
    _json_write(ARTIFACT_DIR / "selection_lock.json", selection_lock)
    _json_write(
        RESULT_DIR / "selection_summary.json",
        {
            "study": STUDY_NAME,
            "test_status": "sealed_not_evaluated",
            "cv_ranking": cv_ranking,
            "fixed_validation_ranking": fixed_ranking,
            "selected_architecture": selected_name,
            "selected_validation_metrics": fixed_results[selected_name][
                "validation_metrics"
            ],
        },
    )
    logger.info("Locked selection: %s", selected_name)
    return selection_lock


def _load_selected_model(
    selection_lock: dict[str, Any],
) -> tuple[nn.Module, StandardScaler]:
    name = selection_lock["selected_architecture"]
    spec = ARCHITECTURES[name]
    checkpoint_path = PROJECT_ROOT / selection_lock["selected_checkpoint"]
    scaler_path = PROJECT_ROOT / selection_lock["selected_scaler"]
    if _sha256(checkpoint_path) != selection_lock["selected_checkpoint_sha256"]:
        raise RuntimeError("selected checkpoint SHA256 不匹配。")
    if _sha256(scaler_path) != selection_lock["selected_scaler_sha256"]:
        raise RuntimeError("selected scaler SHA256 不匹配。")
    model = build_model(spec)
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, joblib.load(scaler_path)


def _metrics_with_bias(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
) -> dict[str, Any]:
    if len(y_true) == 0:
        return {
            "status": "no_samples",
            "r2_u": None,
            "r2_v": None,
            "r2_joint": None,
            "rmse": None,
            "mae": None,
            "bias_u": None,
            "bias_v": None,
        }
    metrics = regression_metrics(y_true, y_prediction)
    error = y_prediction.astype(np.float64) - y_true.astype(np.float64)
    return {
        "status": "ok",
        **metrics,
        "bias_u": float(error[:, 0].mean()),
        "bias_v": float(error[:, 1].mean()),
    }


def run_test_once(code_commit: str) -> dict[str, Any]:
    """解封固定 test 一次；已有结果时拒绝覆盖或重复评价。"""
    output_path = ARTIFACT_DIR / "test_evaluation.json"
    if output_path.exists():
        raise FileExistsError(
            "test_evaluation.json 已存在；test 只允许一次性评价。"
        )
    selection_path = ARTIFACT_DIR / "selection_lock.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["test_status"] != "sealed_not_evaluated":
        raise RuntimeError(f"test 状态异常: {selection['test_status']}")

    frames, manifest = _load_frames_and_manifest()
    test_ids = set(manifest["splits"]["test"]["original_ids"])
    expected_hash = hashlib.sha256(
        "\n".join(sorted(test_ids)).encode("utf-8")
    ).hexdigest()
    if expected_hash != selection["test_original_ids_sha256"]:
        raise RuntimeError("固定 test original_ID 清单发生变化。")
    test_frame = _frames_for_ids(frames, test_ids)
    del frames

    model, scaler = _load_selected_model(selection)
    x_raw = test_frame[CORE_FEATURES].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    x_scaled = scaler.transform(x_raw).astype(np.float32, copy=False)
    y_true = test_frame[["residual_u", "residual_v"]].to_numpy(
        dtype=np.float32,
        copy=True,
    )
    selected_prediction = _predict(model, x_scaled, torch.device("cpu"))

    session = ort.InferenceSession(
        str(GLOBAL_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    global_prediction = session.run(["output"], {"input": x_raw})[0]
    memberships = region_memberships(
        test_frame["latitude"],
        test_frame["longitude"],
    )
    subset_masks = {
        "CMS_overall": memberships["CMS"],
        "Bohai_Yellow_Sea": memberships["BYS"],
        "East_China_Sea": memberships["ECS"],
        "Northern_South_China_Sea": memberships["NSCS"],
    }
    subsets = {
        name: {
            "n_samples": int(mask.sum()),
            "n_original_ids": int(
                test_frame.loc[mask, "original_ID"].nunique()
            ),
            "selected_network": _metrics_with_bias(
                y_true[mask],
                selected_prediction[mask],
            ),
            "frozen_global": _metrics_with_bias(
                y_true[mask],
                global_prediction[mask],
            ),
        }
        for name, mask in subset_masks.items()
    }

    source_cms = json.loads(SOURCE_CMS_METRICS.read_text(encoding="utf-8"))
    source_linear = json.loads(
        SOURCE_LINEAR_METRICS.read_text(encoding="utf-8")
    )
    selected_metrics = subsets["CMS_overall"]["selected_network"]
    global_metrics = subsets["CMS_overall"]["frozen_global"]
    comparisons = {
        "selected_minus_original_cms_joint_r2": (
            selected_metrics["r2_joint"]
            - source_cms["test"]["r2_joint"]
        ),
        "selected_minus_original_cms_rmse": (
            selected_metrics["rmse"] - source_cms["test"]["rmse"]
        ),
        "selected_minus_linear_joint_r2": (
            selected_metrics["r2_joint"]
            - source_linear["test"]["r2_joint"]
        ),
        "selected_minus_linear_rmse": (
            selected_metrics["rmse"] - source_linear["test"]["rmse"]
        ),
        "selected_minus_frozen_global_joint_r2": (
            selected_metrics["r2_joint"] - global_metrics["r2_joint"]
        ),
        "selected_minus_frozen_global_rmse": (
            selected_metrics["rmse"] - global_metrics["rmse"]
        ),
    }
    recommendation = (
        "eligible_for_independent_onnx_freeze"
        if selected_metrics["r2_joint"] > global_metrics["r2_joint"]
        and selected_metrics["rmse"] < global_metrics["rmse"]
        else "do_not_replace_frozen_global"
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": STUDY_NAME,
        "evaluation_code_git_commit": code_commit,
        "test_status": "evaluated_once_locked",
        "selection_lock_sha256": _sha256(selection_path),
        "selected_architecture": selection["selected_architecture"],
        "selected_parameter_count": selection["fixed_validation_results"][
            selection["selected_architecture"]
        ]["parameter_count"],
        "test_original_ids": sorted(test_ids),
        "subsets": subsets,
        "references": {
            "original_cms_mlp": source_cms["test"],
            "regional_linear": source_linear["test"],
            "frozen_global_same_rows": global_metrics,
        },
        "comparisons": comparisons,
        "recommendation": recommendation,
    }
    _json_write(output_path, report)
    _json_write(RESULT_DIR / "test_evaluation.json", report)
    logger.info("One-time test evaluation complete: %s", recommendation)
    return report


def _setup_logging(phase: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    )
    logger.info("Study=%s phase=%s device=%s", STUDY_NAME, phase, _device())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CMS core6 网络结构受控重构研究"
    )
    parser.add_argument(
        "--phase",
        choices=["select", "evaluate-test-once"],
        required=True,
    )
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    _setup_logging(args.phase)
    if args.phase == "select":
        result = run_selection(args.code_commit)
        print(f"Selected: {result['selected_architecture']}")
        print("Test status: sealed_not_evaluated")
    else:
        result = run_test_once(args.code_commit)
        overall = result["subsets"]["CMS_overall"]["selected_network"]
        print(f"Selected: {result['selected_architecture']}")
        print(f"Test joint R2: {overall['r2_joint']:.6f}")
        print(f"Test RMSE: {overall['rmse']:.6f}")
        print(f"Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    main()
