"""运行 105–170 E expanded frozen-global adapter 严格实验。"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnxruntime as ort

from cms_global_adapter import (
    ADAPTER_SPECS,
    EXPECTED_GLOBAL_ONNX_SHA256,
    GLOBAL_ONNX_PATH,
    LAMBDA_GRID,
    OUTER_SPLITS,
    RANDOM_SEED,
    FittedAdapter,
    build_adapter_data,
    combine_adapter_data,
    compare_by_region,
    compare_predictions,
    correction_magnitude_summary,
    derive_global_lineage_split,
    fit_adapter,
    hash_ids,
    load_filtered_frames,
    predict_correction,
    run_nested_development_cv,
    select_candidate_by_cv,
    sha256_file,
    trajectory_proxy,
    validate_frozen_global,
)
from data_loader import PROJECT_ROOT
from eastasia_wnp_regional import (
    ARTIFACT_DIR,
    EXPECTED_POPULATION,
    FILTERED_DATA_PATH,
    FILTERED_DIAGNOSTICS_PATH,
    LATITUDE_RANGE,
    LONGITUDE_RANGE,
    MODEL_VERSION,
    eawnp_memberships,
)
from cms_regional import GLOBAL_SPLIT_MANIFEST_PATH
from run_cms_global_adapter import (
    _confirmation_acceptance,
    _selection_gate,
)


RESULT_DIR = PROJECT_ROOT / "results" / MODEL_VERSION
logger = logging.getLogger(__name__)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


def _session() -> ort.InferenceSession:
    return ort.InferenceSession(
        str(GLOBAL_ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )


def _build_data(
    frames: list,
    original_ids: list[str],
    session: ort.InferenceSession,
):
    return build_adapter_data(
        frames,
        original_ids,
        session,
        membership_function=eawnp_memberships,
        required_membership="EAWNP",
    )


def _protocol(code_commit: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_code_git_commit": code_commit,
        "scientific_change_from_cms_adapter_v1": (
            "Only the row-level geographic filter and inherited population "
            "change; adapter candidates, fitting, selection and gates remain "
            "the same."
        ),
        "region": {
            "latitude": list(LATITUDE_RANGE),
            "longitude": list(LONGITUDE_RANGE),
            "boundary_policy": "inclusive",
            "minimum_hourly_rows_per_original_id": 24,
            "expected_population": EXPECTED_POPULATION,
            "count_threshold_override": {
                "source_branch": "wdf_cms_range_search_v1",
                "source_commit": (
                    "2c7111d451411bb2df96b6e75e111d9ef2451d7a"
                ),
                "target_or_model_metrics_seen_before_override": False,
            },
        },
        "base_model": {
            "path": _relative(GLOBAL_ONNX_PATH),
            "required_sha256": EXPECTED_GLOBAL_ONNX_SHA256,
            "weights_trainable": False,
        },
        "adapter_target": (
            "observed residual - authoritative frozen-global ONNX prediction"
        ),
        "target_definition": {
            "residual_u": "drifter_ve - CFSv2_u",
            "residual_v": "drifter_vn - CFSv2_v",
        },
        "candidates": {
            name: {
                "name": spec.name,
                "family": spec.family,
                "parameter_names": list(spec.parameter_names),
                "parameter_count": spec.parameter_count,
                "description": spec.description,
            }
            for name, spec in ADAPTER_SPECS.items()
        },
        "lambda_grid": list(LAMBDA_GRID),
        "fit_weighting": (
            "each original_ID has total weight 1; rows within ID weight 1/n"
        ),
        "ridge": (
            "RMS-scaled basis; all coefficients including bias penalized; "
            "lambda shrinks exactly toward frozen global"
        ),
        "selection": {
            "development_split": (
                "75 regional IDs inherited from frozen-global train"
            ),
            "gate_split": (
                "12 regional IDs inherited from frozen-global validation"
            ),
            "confirmation_split": (
                "9 regional IDs inherited from frozen-global test"
            ),
            "nested_outer_group_folds": OUTER_SPLITS,
            "nested_inner_group_folds": 4,
            "random_seed": RANDOM_SEED,
            "primary_score": "equal-ID mean adapter-minus-global MSE",
            "tie_rule": (
                "one-standard-error; fewer parameters; stronger lambda"
            ),
        },
        "development_gate": {
            "nonzero_adapter": True,
            "minimum_macro_id_rmse_improvement": 0.02,
            "minimum_id_win_rate": 0.60,
            "minimum_outer_family_frequency": 0.60,
            "maximum_ecs_nscs_macro_rmse_degradation": 0.02,
            "maximum_correction_to_global_p99_ratio": 1.0,
        },
        "validation_gate": {
            "point_rmse_must_improve": True,
            "macro_id_rmse_must_improve": True,
            "maximum_single_id_rmse_degradation": 0.05,
        },
        "confirmation_acceptance": {
            "point_joint_r2_must_improve": True,
            "point_rmse_must_improve": True,
            "maximum_single_id_rmse_degradation": 0.05,
            "trajectory_24h_macro_median_must_improve": True,
            "maximum_trajectory_24h_macro_p90_degradation": 0.05,
        },
        "reported_subsets": [
            "EAWNP",
            "original CMS",
            "BYS",
            "ECS",
            "NSCS",
            "WEST_105_140",
            "EAST_140_170",
        ],
        "explicitly_forbidden": [
            "global weight fine-tuning",
            "nonlinear adapter",
            "new input features",
            "subregion-specific models",
            "adapter ensemble",
            "test-driven reselection",
            "rows east of 170 E",
        ],
    }


def _strip_private(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if not key.startswith("_")
    }


def _validate_prepared_data() -> dict[str, Any]:
    required = {
        "filtered_data": FILTERED_DATA_PATH,
        "diagnostics": FILTERED_DIAGNOSTICS_PATH,
        "statistics": ARTIFACT_DIR / "data_statistics.json",
        "region_mask": ARTIFACT_DIR / "region_mask.json",
        "split_manifest": ARTIFACT_DIR / "split_manifest.json",
        "row_index": ARTIFACT_DIR / "region_row_index.npz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"expanded 数据 artifact 缺失: {missing}")
    diagnostics = _read_json(required["diagnostics"])
    statistics = _read_json(required["statistics"])
    split_manifest = _read_json(required["split_manifest"])
    population = {
        "total": statistics["dataset"]["n_original_ids"],
        "train": split_manifest["splits"]["train"]["n_original_ids"],
        "val": split_manifest["splits"]["val"]["n_original_ids"],
        "test": split_manifest["splits"]["test"]["n_original_ids"],
        "samples": statistics["dataset"]["n_samples"],
    }
    if population != EXPECTED_POPULATION:
        raise RuntimeError(
            f"prepared population 异常: {population} != "
            f"{EXPECTED_POPULATION}"
        )
    if sha256_file(FILTERED_DATA_PATH) != diagnostics[
        "output_integrity"
    ]["filtered_data_sha256"]:
        raise RuntimeError("filtered expanded data SHA256 不匹配。")
    return {
        name: {
            "path": _relative(path),
            "sha256": sha256_file(path),
        }
        for name, path in required.items()
    }


def run_selection(code_commit: str) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    selection_lock_path = ARTIFACT_DIR / "selection_lock.json"
    test_output_path = ARTIFACT_DIR / "test_evaluation.json"
    if selection_lock_path.exists() or test_output_path.exists():
        raise FileExistsError(
            "expanded selection/test artifact 已存在；拒绝覆盖。"
        )

    prepared_data = _validate_prepared_data()
    base_contract = validate_frozen_global()
    frames = load_filtered_frames(FILTERED_DATA_PATH)
    lineage = derive_global_lineage_split(frames)
    counts = {
        name: lineage["splits"][name]["n_original_ids"]
        for name in ("development", "gate", "confirmation")
    }
    if counts != {"development": 75, "gate": 12, "confirmation": 9}:
        raise RuntimeError(f"预期 expanded lineage 75/12/9，实际为 {counts}")

    protocol = _protocol(code_commit)
    protocol_path = ARTIFACT_DIR / "selection_protocol.json"
    _write_json(protocol_path, protocol)
    session = _session()
    development_ids = lineage["splits"]["development"]["original_ids"]
    gate_ids = lineage["splits"]["gate"]["original_ids"]
    confirmation_ids = lineage["splits"]["confirmation"]["original_ids"]
    logger.info("构造 development=75 ID 与 gate=12 ID")
    development = _build_data(frames, development_ids, session)
    gate = _build_data(frames, gate_ids, session)
    lineage["splits"]["development"]["n_samples"] = len(
        development.target
    )
    lineage["splits"]["gate"]["n_samples"] = len(gate.target)
    lineage["splits"]["confirmation"]["n_samples"] = None
    lineage["confirmation_data_status"] = "sealed_not_loaded"
    lineage_path = ARTIFACT_DIR / "global_lineage_split_manifest.json"
    _write_json(lineage_path, lineage)

    logger.info("运行75-ID nested GroupKFold adapter 选择")
    nested = run_nested_development_cv(
        development,
        model_version=MODEL_VERSION,
    )
    logger.info("在全部75个 development ID 上锁定 family/lambda")
    final_cv = select_candidate_by_cv(
        development,
        n_splits=OUTER_SPLITS,
        seed=RANDOM_SEED,
    )
    selected_name = final_cv["selected_adapter_name"]
    selected_lambda = final_cv["selected_lambda"]
    selected_spec = ADAPTER_SPECS[selected_name]
    logger.info("选择结果: %s lambda=%s", selected_name, selected_lambda)

    development_adapter = fit_adapter(
        development,
        selected_spec,
        selected_lambda,
    )
    development_adapter_path = (
        ARTIFACT_DIR / "adapter_fit_development.json"
    )
    _write_json(
        development_adapter_path,
        {
            **development_adapter.to_dict(model_version=MODEL_VERSION),
            "selection_code_git_commit": code_commit,
            "fit_population": "75 frozen-global train regional IDs",
        },
    )
    gate_correction = predict_correction(development_adapter, gate)
    gate_prediction = gate.global_prediction + gate_correction
    gate_comparison = compare_predictions(gate, gate_prediction)
    gate_regions = compare_by_region(gate, gate_prediction)
    gate_magnitude = correction_magnitude_summary(
        gate_correction,
        gate.global_prediction,
    )
    decision = _selection_gate(
        selected_name=selected_name,
        nested=nested,
        gate_comparison=gate_comparison,
    )

    development_path = ARTIFACT_DIR / "development_cv.json"
    _write_json(
        development_path,
        {
            "schema_version": 1,
            "model_version": MODEL_VERSION,
            "nested_selection": _strip_private(nested),
            "final_development_selection": final_cv,
        },
    )
    gate_path = ARTIFACT_DIR / "gate_evaluation.json"
    _write_json(
        gate_path,
        {
            "schema_version": 1,
            "model_version": MODEL_VERSION,
            "adapter_name": selected_name,
            "lambda": selected_lambda,
            "comparison": gate_comparison,
            "regions": gate_regions,
            "correction_magnitude": gate_magnitude,
            "decision": decision,
        },
    )

    adapter_for_test_path: Path | None = None
    adapter_for_test_sha256: str | None = None
    if decision["passed"]:
        logger.info("门控通过；用 development+gate 87 ID 拟合锁定 adapter")
        development_and_gate = combine_adapter_data([development, gate])
        adapter_for_test = fit_adapter(
            development_and_gate,
            selected_spec,
            selected_lambda,
        )
        adapter_for_test_path = ARTIFACT_DIR / "adapter_for_test.json"
        _write_json(
            adapter_for_test_path,
            {
                **adapter_for_test.to_dict(model_version=MODEL_VERSION),
                "selection_code_git_commit": code_commit,
                "fit_population": (
                    "75 frozen-global train plus 12 frozen-global "
                    "validation regional IDs"
                ),
                "confirmation_ids_used_for_fit": False,
            },
        )
        adapter_for_test_sha256 = sha256_file(adapter_for_test_path)
    else:
        logger.info("门控未通过；confirmation test 不获授权")

    selection_lock = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_code_git_commit": code_commit,
        "base_contract": base_contract,
        "prepared_data": prepared_data,
        "protocol": {
            "path": _relative(protocol_path),
            "sha256": sha256_file(protocol_path),
        },
        "lineage_manifest": {
            "path": _relative(lineage_path),
            "sha256": sha256_file(lineage_path),
        },
        "development_cv": {
            "path": _relative(development_path),
            "sha256": sha256_file(development_path),
        },
        "gate_evaluation": {
            "path": _relative(gate_path),
            "sha256": sha256_file(gate_path),
        },
        "selected_adapter_name": selected_name,
        "selected_lambda": selected_lambda,
        "selection_gate_passed": decision["passed"],
        "selection_gate": decision,
        "adapter_for_test": (
            {
                "path": _relative(adapter_for_test_path),
                "sha256": adapter_for_test_sha256,
            }
            if adapter_for_test_path is not None
            else None
        ),
        "confirmation_original_ids": confirmation_ids,
        "confirmation_original_ids_sha256": hash_ids(confirmation_ids),
        "confirmation_disclosure": (
            "Only geographic counts were inspected before this selection; "
            "no target, base metric, or adapter prediction on these IDs was "
            "evaluated."
        ),
        "test_status": (
            "sealed_not_evaluated"
            if decision["passed"]
            else "not_authorized_gate_failed"
        ),
    }
    _write_json(selection_lock_path, selection_lock)
    summary = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "selected_adapter_name": selected_name,
        "selected_lambda": selected_lambda,
        "development_comparison": nested["comparison"],
        "gate_comparison": gate_comparison,
        "selection_gate": decision,
        "test_status": selection_lock["test_status"],
    }
    _write_json(RESULT_DIR / "selection_summary.json", summary)
    return selection_lock


def run_test_once(code_commit: str) -> dict[str, Any]:
    output_path = ARTIFACT_DIR / "test_evaluation.json"
    result_output_path = RESULT_DIR / "test_evaluation.json"
    if output_path.exists() or result_output_path.exists():
        raise FileExistsError("expanded confirmation 已评价；拒绝重复运行。")
    selection_path = ARTIFACT_DIR / "selection_lock.json"
    if not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    selection = _read_json(selection_path)
    if not selection["selection_gate_passed"]:
        raise RuntimeError("selection gate 未通过，禁止评价 confirmation。")
    if selection["test_status"] != "sealed_not_evaluated":
        raise RuntimeError(f"test 状态异常: {selection['test_status']}")
    if validate_frozen_global()["sha256"] != selection["base_contract"][
        "sha256"
    ]:
        raise RuntimeError("frozen global ONNX 合同发生变化。")

    adapter_record = selection["adapter_for_test"]
    adapter_path = PROJECT_ROOT / adapter_record["path"]
    if sha256_file(adapter_path) != adapter_record["sha256"]:
        raise RuntimeError("adapter_for_test SHA256 不匹配。")
    adapter = FittedAdapter.from_dict(_read_json(adapter_path))
    frames = load_filtered_frames(FILTERED_DATA_PATH)
    lineage = derive_global_lineage_split(frames)
    confirmation_ids = lineage["splits"]["confirmation"]["original_ids"]
    if hash_ids(confirmation_ids) != selection[
        "confirmation_original_ids_sha256"
    ]:
        raise RuntimeError("confirmation original_ID 清单发生变化。")
    if set(confirmation_ids) & set(adapter.training_original_ids):
        raise RuntimeError("confirmation ID 进入 adapter fitting population。")

    session = _session()
    logger.info("解封9个 frozen-global test regional IDs，一次性评价")
    confirmation = _build_data(frames, confirmation_ids, session)
    correction = predict_correction(adapter, confirmation)
    prediction = confirmation.global_prediction + correction
    comparison = compare_predictions(confirmation, prediction)
    regions = compare_by_region(confirmation, prediction)
    magnitude = correction_magnitude_summary(
        correction,
        confirmation.global_prediction,
    )
    proxy = trajectory_proxy(confirmation, prediction)
    acceptance = _confirmation_acceptance(comparison, proxy)
    report = {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_code_git_commit": code_commit,
        "selection_lock_sha256": sha256_file(selection_path),
        "adapter_for_test_sha256": sha256_file(adapter_path),
        "base_onnx_sha256": EXPECTED_GLOBAL_ONNX_SHA256,
        "test_status": "evaluated_once_locked",
        "confirmation_original_ids": confirmation_ids,
        "confirmation": {
            "comparison": comparison,
            "regions": regions,
            "correction_magnitude": magnitude,
            "trajectory_proxy": proxy,
        },
        "acceptance": acceptance,
        "recommendation": (
            "eligible_for_onnx_freeze"
            if acceptance["passed"]
            else "stop_no_onnx_freeze"
        ),
    }
    _write_json(output_path, report)
    _write_json(result_output_path, report)
    return report


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="105–170E frozen-global linear adapter"
    )
    parser.add_argument(
        "--phase",
        choices=["select", "evaluate-test-once"],
        required=True,
    )
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    _setup_logging()
    logger.info(
        "model=%s phase=%s base_sha256=%s",
        MODEL_VERSION,
        args.phase,
        EXPECTED_GLOBAL_ONNX_SHA256,
    )
    if args.phase == "select":
        result = run_selection(args.code_commit)
        print(f"Selected: {result['selected_adapter_name']}")
        print(f"Lambda: {result['selected_lambda']}")
        print(f"Gate passed: {result['selection_gate_passed']}")
        print(f"Test status: {result['test_status']}")
    else:
        result = run_test_once(args.code_commit)
        comparison = result["confirmation"]["comparison"]
        print(
            "Confirmation joint R2: "
            f"{comparison['adapted_point_metrics']['r2_joint']:.6f}"
        )
        print(
            "Confirmation RMSE: "
            f"{comparison['adapted_point_metrics']['rmse']:.6f}"
        )
        print(f"Recommendation: {result['recommendation']}")


if __name__ == "__main__":
    main()
