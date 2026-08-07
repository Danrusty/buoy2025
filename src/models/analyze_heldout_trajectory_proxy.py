"""冻结 held-out drifter test 的多时长位移误差代理分析。

本脚本不训练、不修改模型，也不引入 Fortran 轨迹动力学。它严格继承冻结
global core6 的 original_ID test split 和逐行有效性掩码，在每条源子轨迹
内部按严格 1 小时间隔切分连续 episode，再用非重叠窗口累计

    (预测残差速度 - 观测残差速度) × 3600 秒

得到 6/12/24/48/72 h 的二维位移误差代理。主比较为冻结 Linear baseline
与冻结 core6 MLP；可选读取证据 tag 上已冻结的 lat7 预测作为辅助敏感性分析。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pickle
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import matplotlib
import numpy as np
import onnxruntime as ort
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data_loader import (  # noqa: E402
    CURRENT_COLS,
    GROUP_COL,
    OBS_COLS,
    PROJECT_ROOT,
    _canonical_original_id,
)
from evaluation import regression_metrics  # noqa: E402


ANALYSIS_NAME = "heldout_trajectory_proxy_v1"
SOURCE_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "trajectories_with_all_features_circular_mwd_v2.pkl"
)
FROZEN_RUN_DIR = (
    PROJECT_ROOT
    / "trained_models"
    / "ablation_circular_mwd_v2_final"
    / "core_6"
)
SPLIT_MANIFEST_PATH = FROZEN_RUN_DIR / "split_manifest.json"
LINEAR_MODEL_PATH = FROZEN_RUN_DIR / "linear_baseline.joblib"
FROZEN_METRICS_PATH = FROZEN_RUN_DIR / "mlp_metrics.json"
RELEASE_MANIFEST_PATH = (
    PROJECT_ROOT
    / "deployment"
    / "releases"
    / "wdf_core6_circular_mwd_v2"
    / "release_manifest.json"
)
FROZEN_ONNX_PATH = (
    PROJECT_ROOT
    / "deployment"
    / "releases"
    / "wdf_core6_circular_mwd_v2"
    / "wdf_drifter.onnx"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / ANALYSIS_NAME
LAT7_EVIDENCE_REF = "paper/global-mlp-lat7+lat9"
LAT7_RUN_MANIFEST_REPO_PATH = (
    "trained_models/global_mlp_lat7_v1/run_manifest.json"
)
LAT7_METRICS_REPO_PATH = "trained_models/global_mlp_lat7_v1/metrics.json"
HORIZONS_HOURS = (6, 12, 24, 48, 72)
RANDOM_SEED = 42
BOOTSTRAP_REPLICATES = 10_000
EXPECTED_SOURCE_SHA256 = (
    "22ab0a32ff9472a6f8b8f57af5fd96b93cdeb76d45b4ef6b0a798fa1befb937e"
)
EXPECTED_ONNX_SHA256 = (
    "787d1d6a663677e30161a70493c70a7e46434414fb59085fbb68477939f18941"
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Episode:
    """拼接后 test 数组中的一个严格连续小时片段。"""

    original_id: str
    source_segment_index: int
    episode_index_within_segment: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass
class HeldoutData:
    """冻结 test rows 及其连续片段元数据。"""

    core6: np.ndarray
    target: np.ndarray
    sin_latitude: np.ndarray
    episodes: list[Episode]
    test_original_ids: list[str]
    source_segment_count: int
    non_hourly_gap_count: int


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    """分块计算文件 SHA256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    """仓库内路径写为相对路径，便于跨机器复现。"""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def git_output(*arguments: str, binary: bool = False) -> str | bytes:
    """读取当前仓库 Git 对象，不切换工作树。"""

    result = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout


def git_json(reference: str, repo_path: str) -> dict[str, Any]:
    """从证据 tag 读取 JSON。"""

    payload = git_output("show", f"{reference}:{repo_path}")
    if not isinstance(payload, str):
        raise TypeError("Git JSON 输出类型异常。")
    return json.loads(payload)


def split_continuous_bounds(
    times: np.ndarray | Sequence[Any],
) -> list[tuple[int, int]]:
    """按严格 3600 秒间隔切分，返回左闭右开局部边界。"""

    values = np.asarray(times, dtype="datetime64[s]")
    if values.ndim != 1:
        raise ValueError(f"时间数组必须是一维，实际为 {values.shape}")
    if len(values) == 0:
        return []
    if np.isnat(values).any():
        raise ValueError("保留行的 time 含 NaT。")
    if len(values) == 1:
        return [(0, 1)]

    seconds = np.diff(values).astype("timedelta64[s]").astype(np.int64)
    boundaries = np.flatnonzero(seconds != 3600) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(values)]))
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
    ]


def _frozen_valid_mask(frame: pd.DataFrame, core6: Sequence[str]) -> pd.Series:
    """完全复刻冻结 core6 loader 的逐行 dropna 成员规则。"""

    required = list(dict.fromkeys([*core6, *OBS_COLS, *CURRENT_COLS]))
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"源轨迹缺少冻结字段: {sorted(missing)}")
    return frame[required].notna().all(axis=1)


def _validate_split_manifest(
    manifest: Mapping[str, Any],
) -> tuple[list[str], tuple[str, ...]]:
    """验证冻结 test split 的基本防泄漏约束。"""

    if manifest.get("group_column") != GROUP_COL:
        raise ValueError("split manifest group_column 不是 original_ID。")
    if int(manifest.get("random_seed", -1)) != RANDOM_SEED:
        raise ValueError("split manifest seed 不是 42。")

    split_sets: dict[str, set[str]] = {}
    for name in ("train", "val", "test"):
        ids = [str(value) for value in manifest["splits"][name]["original_ids"]]
        if len(ids) != int(manifest["splits"][name]["n_original_ids"]):
            raise ValueError(f"{name} original_ID 数量与 manifest 不一致。")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} original_ID 存在重复。")
        split_sets[name] = set(ids)
    if (
        split_sets["train"] & split_sets["val"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["val"] & split_sets["test"]
    ):
        raise ValueError("冻结 split 出现 original_ID 泄漏。")

    core6 = tuple(str(value) for value in manifest["feature_columns"])
    expected = (
        "era5_u10",
        "era5_v10",
        "era5_swh",
        "era5_mwp",
        "era5_wave_dir_sin",
        "era5_wave_dir_cos",
    )
    if core6 != expected:
        raise ValueError(f"冻结特征顺序异常: {core6}")
    return (
        [str(value) for value in manifest["splits"]["test"]["original_ids"]],
        core6,
    )


def load_heldout_data(
    source_path: Path = SOURCE_PATH,
    split_manifest_path: Path = SPLIT_MANIFEST_PATH,
    *,
    verify_source_sha256: bool = True,
) -> HeldoutData:
    """读取冻结 test rows，并构建不跨源片段、不跨缺口的 episode。"""

    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    test_original_ids, core6_columns = _validate_split_manifest(manifest)
    test_id_set = set(test_original_ids)

    if verify_source_sha256:
        logger.info("验证源数据 SHA256...")
        actual_source_sha256 = sha256_file(source_path)
        if actual_source_sha256 != EXPECTED_SOURCE_SHA256:
            raise ValueError(
                "源数据 SHA256 不匹配: "
                f"{actual_source_sha256} != {EXPECTED_SOURCE_SHA256}"
            )

    logger.info("读取源轨迹 pickle: %s", source_path)
    with source_path.open("rb") as file:
        trajectories = pickle.load(file)
    logger.info("源子轨迹总数: %d", len(trajectories))

    core6_pieces: list[np.ndarray] = []
    target_pieces: list[np.ndarray] = []
    latitude_pieces: list[np.ndarray] = []
    episodes: list[Episode] = []
    seen_ids: set[str] = set()
    row_offset = 0
    segment_count = 0
    non_hourly_gap_count = 0

    for trajectory_index, frame in enumerate(trajectories):
        original_id = _canonical_original_id(frame, trajectory_index)
        if original_id not in test_id_set:
            continue
        required_metadata = {"time", "latitude"}
        missing = required_metadata - set(frame.columns)
        if missing:
            raise ValueError(
                f"第 {trajectory_index} 条 test 轨迹缺元数据: {sorted(missing)}"
            )

        valid = _frozen_valid_mask(frame, core6_columns)
        n_rows = int(valid.sum())
        if n_rows == 0:
            continue

        selected_core6 = frame.loc[
            valid, list(core6_columns)
        ].to_numpy(dtype=np.float32, copy=True)
        # 先按原列精度做减法，再转 float32，与冻结 loader 保持一致。
        selected_target = np.column_stack(
            (
                frame.loc[valid, OBS_COLS[0]].to_numpy(dtype=np.float64)
                - frame.loc[valid, CURRENT_COLS[0]].to_numpy(
                    dtype=np.float64
                ),
                frame.loc[valid, OBS_COLS[1]].to_numpy(dtype=np.float64)
                - frame.loc[valid, CURRENT_COLS[1]].to_numpy(
                    dtype=np.float64
                ),
            )
        ).astype(np.float32)
        latitude = frame.loc[valid, "latitude"].to_numpy(
            dtype=np.float64,
            copy=True,
        )
        if not np.all(np.isfinite(latitude)):
            raise ValueError(
                f"第 {trajectory_index} 条 test 轨迹保留行 latitude 非有限。"
            )
        if np.any((latitude < -90.0) | (latitude > 90.0)):
            raise ValueError(
                f"第 {trajectory_index} 条 test 轨迹 latitude 超界。"
            )
        sin_latitude = np.sin(np.deg2rad(latitude)).astype(np.float32)

        parsed_times = pd.to_datetime(
            frame.loc[valid, "time"],
            errors="coerce",
            utc=True,
        ).to_numpy(dtype="datetime64[s]")
        local_bounds = split_continuous_bounds(parsed_times)
        non_hourly_gap_count += max(0, len(local_bounds) - 1)
        for episode_index, (local_start, local_stop) in enumerate(
            local_bounds
        ):
            episodes.append(
                Episode(
                    original_id=original_id,
                    source_segment_index=trajectory_index,
                    episode_index_within_segment=episode_index,
                    start=row_offset + local_start,
                    stop=row_offset + local_stop,
                )
            )

        core6_pieces.append(selected_core6)
        target_pieces.append(selected_target)
        latitude_pieces.append(sin_latitude)
        seen_ids.add(original_id)
        row_offset += n_rows
        segment_count += 1

    del trajectories
    if seen_ids != test_id_set:
        raise ValueError(
            "源数据未覆盖全部冻结 test original_ID: "
            f"{sorted(test_id_set - seen_ids)[:10]}"
        )
    if not core6_pieces:
        raise ValueError("冻结 test 没有有效数据。")

    core6 = np.concatenate(core6_pieces)
    target = np.concatenate(target_pieces)
    sin_latitude = np.concatenate(latitude_pieces)
    expected = manifest["splits"]["test"]
    checks = {
        "n_samples": (len(target), int(expected["n_samples"])),
        "n_segments": (segment_count, int(expected["n_segments"])),
        "n_original_ids": (len(seen_ids), int(expected["n_original_ids"])),
    }
    for label, (actual, frozen) in checks.items():
        if actual != frozen:
            raise ValueError(
                f"冻结 test {label} 不一致: {actual} != {frozen}"
            )
    if not (len(core6) == len(target) == len(sin_latitude) == row_offset):
        raise RuntimeError("test 数组拼接长度不一致。")

    logger.info(
        "冻结 test: %d IDs / %d 源片段 / %d 连续 episodes / %d rows",
        len(seen_ids),
        segment_count,
        len(episodes),
        len(target),
    )
    return HeldoutData(
        core6=core6,
        target=target,
        sin_latitude=sin_latitude,
        episodes=episodes,
        test_original_ids=test_original_ids,
        source_segment_count=segment_count,
        non_hourly_gap_count=non_hourly_gap_count,
    )


def predict_linear(core6: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """用冻结训练集拟合的二输出线性基准做 held-out 推理。"""

    artifact = joblib.load(LINEAR_MODEL_PATH)
    if set(artifact) != {"reg_u", "reg_v"}:
        raise ValueError("linear_baseline.joblib 内容异常。")
    wind = core6[:, :2]
    prediction = np.column_stack(
        (
            artifact["reg_u"].predict(wind),
            artifact["reg_v"].predict(wind),
        )
    ).astype(np.float32, copy=False)
    return prediction, {
        "path": relative_path(LINEAR_MODEL_PATH),
        "sha256": sha256_file(LINEAR_MODEL_PATH),
        "input_features": ["era5_u10", "era5_v10"],
    }


def predict_frozen_core6(
    core6: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """通过冻结 release ONNX 批量推理 core6 MLP。"""

    actual_sha256 = sha256_file(FROZEN_ONNX_PATH)
    if actual_sha256 != EXPECTED_ONNX_SHA256:
        raise ValueError(
            f"冻结 ONNX SHA256 不匹配: {actual_sha256}"
        )
    session = ort.InferenceSession(
        str(FROZEN_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()
    output_meta = session.get_outputs()
    if len(input_meta) != 1 or len(output_meta) != 1:
        raise ValueError("冻结 ONNX 必须是单输入、单输出。")
    if input_meta[0].name != "input" or output_meta[0].name != "output":
        raise ValueError("冻结 ONNX 输入输出名称异常。")

    prediction = np.empty((len(core6), 2), dtype=np.float32)
    for start in range(0, len(core6), batch_size):
        stop = min(start + batch_size, len(core6))
        prediction[start:stop] = session.run(
            ["output"],
            {"input": np.ascontiguousarray(core6[start:stop])},
        )[0]
    return prediction, {
        "path": relative_path(FROZEN_ONNX_PATH),
        "sha256": actual_sha256,
        "providers": session.get_providers(),
        "scaler_inside_graph": True,
    }


def load_lat7_prediction(
    expected_rows: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """校验并读取证据 tag 对应的 lat7 test prediction cache。"""

    run_manifest = git_json(
        LAT7_EVIDENCE_REF,
        LAT7_RUN_MANIFEST_REPO_PATH,
    )
    prediction_record = run_manifest["artifacts"]["prediction_cache"]
    cache_path = PROJECT_ROOT / prediction_record["path"]
    if not cache_path.is_file():
        raise FileNotFoundError(
            "lat7 prediction cache 不在本地；可从证据 tag 的 checkpoint "
            f"重放后再运行: {cache_path}"
        )
    actual_sha256 = sha256_file(cache_path)
    if actual_sha256 != prediction_record["sha256"]:
        raise ValueError(
            "lat7 prediction cache SHA256 不匹配: "
            f"{actual_sha256} != {prediction_record['sha256']}"
        )
    prediction = np.load(cache_path, mmap_mode="r")
    expected_shape = (expected_rows, 2)
    if prediction.shape != expected_shape or prediction.dtype != np.float32:
        raise ValueError(
            f"lat7 prediction cache 异常: {prediction.shape}, "
            f"{prediction.dtype}"
        )
    evidence_commit = str(
        git_output("rev-list", "-n", "1", LAT7_EVIDENCE_REF)
    ).strip()
    return prediction, {
        "evidence_ref": LAT7_EVIDENCE_REF,
        "evidence_commit": evidence_commit,
        "run_manifest_path": LAT7_RUN_MANIFEST_REPO_PATH,
        "training_code_git_commit": run_manifest["training_code_git_commit"],
        "path": relative_path(cache_path),
        "sha256": actual_sha256,
        "features": run_manifest["features"],
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


def validate_point_metrics(
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    """重放逐时指标，并与已冻结记录核对。"""

    frozen = json.loads(FROZEN_METRICS_PATH.read_text(encoding="utf-8"))
    expected: dict[str, Mapping[str, float]] = {
        "linear": frozen["linear_baseline_test"],
        "frozen_core6_mlp": frozen["test"],
    }
    if "mlp_lat7" in predictions:
        lat7_metrics = git_json(
            LAT7_EVIDENCE_REF,
            LAT7_METRICS_REPO_PATH,
        )
        expected["mlp_lat7"] = lat7_metrics["test"]["row_weighted"]

    result = {}
    for name, prediction in predictions.items():
        values = _metric_with_bias(target, prediction)
        reference = expected[name]
        differences = {
            metric: abs(values[metric] - float(reference[metric]))
            for metric in ("r2_u", "r2_v", "r2_joint", "rmse", "mae")
        }
        maximum = max(differences.values())
        if maximum > 2e-6:
            raise ValueError(
                f"{name} 逐时指标未通过冻结重放: max diff={maximum}"
            )
        result[name] = {
            **values,
            "maximum_absolute_difference_from_frozen_record": maximum,
        }
    return result


def endpoint_errors_for_episodes(
    velocity_error: np.ndarray,
    episodes: Sequence[Episode],
    horizons: Sequence[int] = HORIZONS_HOURS,
) -> dict[int, np.ndarray]:
    """按 episode 计算各时长非重叠窗口的二维端点误差（km）。"""

    error = np.asarray(velocity_error, dtype=np.float64)
    if error.ndim != 2 or error.shape[1] != 2:
        raise ValueError(f"速度误差必须为 (N, 2)，实际为 {error.shape}")
    pieces: dict[int, list[np.ndarray]] = {
        int(horizon): [] for horizon in horizons
    }
    for episode in episodes:
        if episode.start < 0 or episode.stop > len(error):
            raise ValueError(f"episode 越界: {episode}")
        for horizon in horizons:
            horizon = int(horizon)
            n_windows = episode.length // horizon
            if n_windows == 0:
                continue
            stop = episode.start + n_windows * horizon
            block = error[episode.start:stop].reshape(
                n_windows,
                horizon,
                2,
            )
            # 每行代表 1 小时平均速度；m/s × 3600 / 1000 = 3.6 km。
            displacement = block.sum(axis=1, dtype=np.float64) * 3.6
            pieces[horizon].append(
                np.linalg.norm(displacement, axis=1)
            )
    return {
        horizon: (
            np.concatenate(values)
            if values
            else np.empty((0,), dtype=np.float64)
        )
        for horizon, values in pieces.items()
    }


def window_group_indices(
    episodes: Sequence[Episode],
    id_order: Sequence[str],
    horizons: Sequence[int] = HORIZONS_HOURS,
) -> dict[int, np.ndarray]:
    """生成与 endpoint 数组顺序完全一致的 original_ID 索引。"""

    id_lookup = {
        original_id: index for index, original_id in enumerate(id_order)
    }
    pieces: dict[int, list[np.ndarray]] = {
        int(horizon): [] for horizon in horizons
    }
    for episode in episodes:
        if episode.original_id not in id_lookup:
            raise ValueError(f"episode ID 不在冻结 test: {episode.original_id}")
        group_index = id_lookup[episode.original_id]
        for horizon in horizons:
            horizon = int(horizon)
            n_windows = episode.length // horizon
            if n_windows:
                pieces[horizon].append(
                    np.full(n_windows, group_index, dtype=np.int32)
                )
    return {
        horizon: (
            np.concatenate(values)
            if values
            else np.empty((0,), dtype=np.int32)
        )
        for horizon, values in pieces.items()
    }


def paired_id_bootstrap(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    replicates: int,
    batch_size: int = 500,
) -> dict[str, Any]:
    """以 original_ID 为抽样单元计算配对均值差和改善率区间。"""

    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if base.shape != cand.shape or base.ndim != 1 or len(base) == 0:
        raise ValueError("bootstrap 输入必须是同长度非空一维数组。")
    if replicates <= 0:
        raise ValueError("bootstrap replicates 必须大于 0。")

    rng = np.random.default_rng(seed)
    delta_samples = np.empty(replicates, dtype=np.float64)
    improvement_samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(
            0,
            len(base),
            size=(stop - start, len(base)),
        )
        base_means = base[indices].mean(axis=1)
        candidate_means = cand[indices].mean(axis=1)
        delta_samples[start:stop] = candidate_means - base_means
        improvement_samples[start:stop] = (
            (base_means - candidate_means) / base_means * 100.0
        )

    delta_ci = np.quantile(delta_samples, [0.025, 0.975])
    improvement_ci = np.quantile(
        improvement_samples,
        [0.025, 0.975],
    )
    return {
        "sampling_unit": "original_ID",
        "paired": True,
        "random_seed": seed,
        "replicates": replicates,
        "delta_candidate_minus_linear_km_ci95": [
            float(delta_ci[0]),
            float(delta_ci[1]),
        ],
        "relative_improvement_percent_ci95": [
            float(improvement_ci[0]),
            float(improvement_ci[1]),
        ],
    }


def summarize_horizon(
    horizon: int,
    group_indices: np.ndarray,
    endpoint_errors: Mapping[str, np.ndarray],
    id_order: Sequence[str],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    """按 ID 等权汇总单个时长，并生成相对 Linear 的配对比较。"""

    groups = np.asarray(group_indices, dtype=np.int32)
    if len(groups) == 0:
        raise ValueError(f"{horizon} h 没有可用窗口。")
    for name, values in endpoint_errors.items():
        if np.asarray(values).shape != groups.shape:
            raise ValueError(
                f"{horizon} h {name} endpoint 数量与窗口 ID 不一致。"
            )
    if "linear" not in endpoint_errors:
        raise ValueError("endpoint_errors 缺少 linear。")

    present = np.unique(groups)
    per_id = []
    id_statistics: dict[str, dict[str, np.ndarray]] = {
        name: {"median": [], "p90": []}
        for name in endpoint_errors
    }
    for group_index in present:
        selected = groups == group_index
        original_id = id_order[int(group_index)]
        record: dict[str, Any] = {
            "original_ID": original_id,
            "n_windows": int(selected.sum()),
            "models": {},
        }
        for name, values in endpoint_errors.items():
            selected_values = np.asarray(values)[selected]
            median = float(np.median(selected_values))
            p90 = float(np.quantile(selected_values, 0.9))
            record["models"][name] = {
                "median_endpoint_error_km": median,
                "p90_endpoint_error_km": p90,
            }
            id_statistics[name]["median"].append(median)
            id_statistics[name]["p90"].append(p90)
        per_id.append(record)

    arrays = {
        name: {
            statistic: np.asarray(values, dtype=np.float64)
            for statistic, values in statistics.items()
        }
        for name, statistics in id_statistics.items()
    }
    models = {}
    for name, values in endpoint_errors.items():
        models[name] = {
            "pooled_window_median_km": float(np.median(values)),
            "pooled_window_p90_km": float(np.quantile(values, 0.9)),
            "equal_id_mean_of_window_medians_km": float(
                arrays[name]["median"].mean()
            ),
            "equal_id_mean_of_window_p90_km": float(
                arrays[name]["p90"].mean()
            ),
        }

    comparisons = {}
    baseline_windows = np.asarray(endpoint_errors["linear"])
    for candidate_name, candidate_windows in endpoint_errors.items():
        if candidate_name == "linear":
            continue
        comparison: dict[str, Any] = {}
        for statistic in ("median", "p90"):
            baseline = arrays["linear"][statistic]
            candidate = arrays[candidate_name][statistic]
            delta = candidate - baseline
            ties = np.isclose(delta, 0.0, rtol=0.0, atol=1e-12)
            base_mean = float(baseline.mean())
            candidate_mean = float(candidate.mean())
            comparison[statistic] = {
                "equal_id_delta_candidate_minus_linear_km": (
                    candidate_mean - base_mean
                ),
                "equal_id_relative_improvement_percent": (
                    (base_mean - candidate_mean) / base_mean * 100.0
                ),
                "id_wins": int(np.sum(delta < -1e-12)),
                "id_ties": int(ties.sum()),
                "id_losses": int(np.sum(delta > 1e-12)),
                "id_win_rate": float(np.mean(delta < -1e-12)),
                "paired_bootstrap": paired_id_bootstrap(
                    baseline,
                    candidate,
                    seed=seed,
                    replicates=bootstrap_replicates,
                ),
            }
        candidate_windows = np.asarray(candidate_windows)
        window_ties = np.isclose(
            candidate_windows,
            baseline_windows,
            rtol=0.0,
            atol=1e-12,
        )
        comparison["window_level"] = {
            "wins": int(
                np.sum(candidate_windows < baseline_windows - 1e-12)
            ),
            "ties": int(window_ties.sum()),
            "losses": int(
                np.sum(candidate_windows > baseline_windows + 1e-12)
            ),
            "win_rate": float(
                np.mean(candidate_windows < baseline_windows - 1e-12)
            ),
        }
        comparisons[candidate_name] = comparison

    return {
        "horizon_hours": horizon,
        "n_windows": len(groups),
        "n_original_ids": len(present),
        "models": models,
        "comparisons_vs_linear": comparisons,
        "per_id": per_id,
    }


def dataset_summary(
    data: HeldoutData,
    group_indices: Mapping[int, np.ndarray],
) -> dict[str, Any]:
    """生成 episode 长度与各时长覆盖统计。"""

    lengths = np.asarray(
        [episode.length for episode in data.episodes],
        dtype=np.int64,
    )
    return {
        "n_original_ids": len(data.test_original_ids),
        "n_samples": len(data.target),
        "n_source_segments": data.source_segment_count,
        "n_continuous_episodes": len(data.episodes),
        "n_non_hourly_gaps": data.non_hourly_gap_count,
        "episode_length_hours": {
            "minimum": int(lengths.min()),
            "median": float(np.median(lengths)),
            "p90": float(np.quantile(lengths, 0.9)),
            "maximum": int(lengths.max()),
        },
        "window_coverage": {
            str(horizon): {
                "n_windows": len(values),
                "n_original_ids": int(len(np.unique(values))),
            }
            for horizon, values in group_indices.items()
        },
    }


def write_summary_csv(
    path: Path,
    horizon_summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    """写出便于论文制表的扁平汇总。"""

    fieldnames = [
        "horizon_hours",
        "model",
        "n_windows",
        "n_original_ids",
        "pooled_window_median_km",
        "pooled_window_p90_km",
        "equal_id_mean_of_window_medians_km",
        "equal_id_mean_of_window_p90_km",
        "median_improvement_percent_vs_linear",
        "median_improvement_ci95_low",
        "median_improvement_ci95_high",
        "median_id_win_rate",
        "p90_improvement_percent_vs_linear",
        "p90_improvement_ci95_low",
        "p90_improvement_ci95_high",
        "p90_id_win_rate",
        "window_win_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for horizon in sorted(
            horizon_summaries,
            key=lambda value: int(value),
        ):
            summary = horizon_summaries[horizon]
            for model_name, model in summary["models"].items():
                row = {
                    "horizon_hours": int(horizon),
                    "model": model_name,
                    "n_windows": summary["n_windows"],
                    "n_original_ids": summary["n_original_ids"],
                    **model,
                }
                if model_name != "linear":
                    comparison = summary["comparisons_vs_linear"][model_name]
                    median_ci = comparison["median"]["paired_bootstrap"][
                        "relative_improvement_percent_ci95"
                    ]
                    p90_ci = comparison["p90"]["paired_bootstrap"][
                        "relative_improvement_percent_ci95"
                    ]
                    row.update(
                        {
                            "median_improvement_percent_vs_linear": (
                                comparison["median"][
                                    "equal_id_relative_improvement_percent"
                                ]
                            ),
                            "median_improvement_ci95_low": median_ci[0],
                            "median_improvement_ci95_high": median_ci[1],
                            "median_id_win_rate": comparison["median"][
                                "id_win_rate"
                            ],
                            "p90_improvement_percent_vs_linear": (
                                comparison["p90"][
                                    "equal_id_relative_improvement_percent"
                                ]
                            ),
                            "p90_improvement_ci95_low": p90_ci[0],
                            "p90_improvement_ci95_high": p90_ci[1],
                            "p90_id_win_rate": comparison["p90"][
                                "id_win_rate"
                            ],
                            "window_win_rate": comparison["window_level"][
                                "win_rate"
                            ],
                        }
                    )
                writer.writerow(row)


def plot_summary(
    path: Path,
    horizon_summaries: Mapping[str, Mapping[str, Any]],
) -> None:
    """绘制端点误差及相对 Linear 改善率。"""

    horizons = sorted(int(value) for value in horizon_summaries)
    first = horizon_summaries[str(horizons[0])]
    model_names = list(first["models"])
    labels = {
        "linear": "Linear baseline",
        "frozen_core6_mlp": "Frozen core6 MLP",
        "mlp_lat7": "MLP lat7 (auxiliary)",
    }
    colors = {
        "linear": "#4C566A",
        "frozen_core6_mlp": "#2E86AB",
        "mlp_lat7": "#D1495B",
    }

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for model_name in model_names:
        values = [
            horizon_summaries[str(horizon)]["models"][model_name][
                "equal_id_mean_of_window_medians_km"
            ]
            for horizon in horizons
        ]
        axes[0].plot(
            horizons,
            values,
            marker="o",
            linewidth=2,
            label=labels.get(model_name, model_name),
            color=colors.get(model_name),
        )
    axes[0].set(
        xlabel="Integration horizon (h)",
        ylabel="Equal-ID mean endpoint error (km)",
        title="Median within-ID displacement-error proxy",
        xticks=horizons,
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    for model_name in model_names:
        if model_name == "linear":
            continue
        values = [
            horizon_summaries[str(horizon)]["comparisons_vs_linear"][
                model_name
            ]["median"]["equal_id_relative_improvement_percent"]
            for horizon in horizons
        ]
        low = [
            horizon_summaries[str(horizon)]["comparisons_vs_linear"][
                model_name
            ]["median"]["paired_bootstrap"][
                "relative_improvement_percent_ci95"
            ][0]
            for horizon in horizons
        ]
        high = [
            horizon_summaries[str(horizon)]["comparisons_vs_linear"][
                model_name
            ]["median"]["paired_bootstrap"][
                "relative_improvement_percent_ci95"
            ][1]
            for horizon in horizons
        ]
        axes[1].plot(
            horizons,
            values,
            marker="o",
            linewidth=2,
            label=labels.get(model_name, model_name),
            color=colors.get(model_name),
        )
        axes[1].fill_between(
            horizons,
            low,
            high,
            color=colors.get(model_name),
            alpha=0.16,
        )
    axes[1].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[1].set(
        xlabel="Integration horizon (h)",
        ylabel="Improvement vs Linear (%)",
        title="Equal-ID median proxy (paired bootstrap 95% CI)",
        xticks=horizons,
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_analysis(
    *,
    output_dir: Path,
    include_lat7: bool,
    batch_size: int,
    bootstrap_replicates: int,
    verify_source_sha256: bool,
) -> dict[str, Any]:
    """执行完整分析并冻结 JSON、CSV 与图。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_heldout_data(
        verify_source_sha256=verify_source_sha256,
    )

    logger.info("重放冻结 Linear baseline...")
    linear_prediction, linear_provenance = predict_linear(data.core6)
    logger.info("重放冻结 core6 ONNX...")
    core6_prediction, core6_provenance = predict_frozen_core6(
        data.core6,
        batch_size=batch_size,
    )
    predictions: dict[str, np.ndarray] = {
        "linear": linear_prediction,
        "frozen_core6_mlp": core6_prediction,
    }
    model_provenance = {
        "linear": linear_provenance,
        "frozen_core6_mlp": core6_provenance,
    }
    if include_lat7:
        logger.info("读取 lat7 辅助预测...")
        lat7_prediction, lat7_provenance = load_lat7_prediction(
            len(data.target)
        )
        predictions["mlp_lat7"] = lat7_prediction
        model_provenance["mlp_lat7"] = lat7_provenance

    point_metrics = validate_point_metrics(data.target, predictions)
    groups_by_horizon = window_group_indices(
        data.episodes,
        data.test_original_ids,
    )
    endpoint_by_model = {}
    for name, prediction in predictions.items():
        logger.info("累计 %s 的多时长位移误差...", name)
        velocity_error = (
            np.asarray(prediction, dtype=np.float64)
            - np.asarray(data.target, dtype=np.float64)
        )
        endpoint_by_model[name] = endpoint_errors_for_episodes(
            velocity_error,
            data.episodes,
        )

    horizon_summaries = {}
    for horizon in HORIZONS_HOURS:
        values = {
            name: endpoint_by_model[name][horizon]
            for name in predictions
        }
        horizon_summaries[str(horizon)] = summarize_horizon(
            horizon,
            groups_by_horizon[horizon],
            values,
            data.test_original_ids,
            bootstrap_replicates=bootstrap_replicates,
            seed=RANDOM_SEED + horizon,
        )

    release_manifest = json.loads(
        RELEASE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": 1,
        "analysis_name": ANALYSIS_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_code_git_commit": str(
            git_output("rev-parse", "HEAD")
        ).strip(),
        "scientific_scope": {
            "primary_comparison": "linear vs frozen_core6_mlp",
            "auxiliary_comparison": (
                "mlp_lat7 vs linear on identical rows"
                if include_lat7
                else None
            ),
            "no_model_training": True,
            "no_fortran_or_oilspill_dynamics": True,
        },
        "protocol": {
            "split": (
                "frozen global original_ID held-out test; no refit or "
                "selection on test"
            ),
            "continuity": (
                "within each source trajectory segment, split wherever "
                "adjacent retained timestamps differ from exactly 3600 s"
            ),
            "windows": (
                "non-overlapping windows anchored at each continuous "
                "episode start; incomplete tail discarded"
            ),
            "displacement_proxy": (
                "Euclidean norm of sum((predicted residual - observed "
                "residual) * 3600 seconds) / 1000, in km"
            ),
            "background_current": (
                "cancels because every model and observation shares the "
                "same CFSv2 background current"
            ),
            "primary_aggregation": (
                "median endpoint error within original_ID, then equal-ID "
                "mean across IDs"
            ),
            "tail_aggregation": (
                "P90 endpoint error within original_ID, then equal-ID "
                "mean across IDs"
            ),
            "uncertainty": (
                "paired nonparametric bootstrap over original_ID"
            ),
            "horizons_hours": list(HORIZONS_HOURS),
        },
        "lineage": {
            "frozen_release_training_code_git_commit": release_manifest[
                "training_code_git_commit"
            ],
            "source": {
                "path": relative_path(SOURCE_PATH),
                "size_bytes": SOURCE_PATH.stat().st_size,
                "sha256": (
                    EXPECTED_SOURCE_SHA256
                    if verify_source_sha256
                    else "not_recomputed"
                ),
            },
            "split_manifest": {
                "path": relative_path(SPLIT_MANIFEST_PATH),
                "sha256": sha256_file(SPLIT_MANIFEST_PATH),
            },
            "models": model_provenance,
        },
        "dataset": dataset_summary(data, groups_by_horizon),
        "point_metrics": point_metrics,
        "trajectory_proxy": {
            "bootstrap_replicates": bootstrap_replicates,
            "horizons": horizon_summaries,
        },
    }

    json_path = output_dir / "trajectory_proxy.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary_csv(
        output_dir / "trajectory_proxy_summary.csv",
        horizon_summaries,
    )
    plot_summary(
        output_dir / "trajectory_proxy.png",
        horizon_summaries,
    )
    logger.info("分析产物已保存: %s", output_dir)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="冻结 held-out test 多时长位移误差代理分析"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="结果目录",
    )
    parser.add_argument(
        "--include-lat7",
        action="store_true",
        help="增加证据 tag 上 lat7 MLP 的同 test rows 辅助分析",
    )
    parser.add_argument(
        "--onnx-batch-size",
        type=int,
        default=131_072,
        help="冻结 ONNX CPU 推理 batch size",
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
        help="按 original_ID 配对 bootstrap 次数",
    )
    parser.add_argument(
        "--skip-source-sha256",
        action="store_true",
        help="仅用于快速调试；正式结果不得使用",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    if args.onnx_batch_size <= 0:
        raise ValueError("--onnx-batch-size 必须大于 0。")
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates 必须大于 0。")
    run_analysis(
        output_dir=args.output_dir.resolve(),
        include_lat7=args.include_lat7,
        batch_size=args.onnx_batch_size,
        bootstrap_replicates=args.bootstrap_replicates,
        verify_source_sha256=not args.skip_source_sha256,
    )


if __name__ == "__main__":
    main()
