"""运行 frozen-global CMS adapter 的严格选择与一次性确认评价。"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnxruntime as ort

from cms_global_adapter import (
    ADAPTER_SPECS,
    ARTIFACT_DIR,
    EXPECTED_GLOBAL_ONNX_SHA256,
    GLOBAL_ONNX_PATH,
    LAMBDA_GRID,
    MODEL_VERSION,
    OUTER_SPLITS,
    RANDOM_SEED,
    RESULT_DIR,
    SOURCE_CMS_SPLIT_MANIFEST,
    AdapterData,
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
from cms_regional import FILTERED_DATA_PATH, GLOBAL_SPLIT_MANIFEST_PATH
from data_loader import PROJECT_ROOT


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


def _protocol(code_commit: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_code_git_commit": code_commit,
        "base_model": {
            "path": _relative(GLOBAL_ONNX_PATH),
            "required_sha256": EXPECTED_GLOBAL_ONNX_SHA256,
            "weights_trainable": False,
        },
        "adapter_target": (
            "observed residual - authoritative frozen-global ONNX prediction"
        ),
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
                "CMS IDs inherited from frozen-global train"
            ),
            "gate_split": "CMS IDs inherited from frozen-global validation",
            "confirmation_split": (
                "CMS IDs inherited from frozen-global test"
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
            "maximum_subregion_macro_rmse_degradation": 0.02,
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
        "explicitly_forbidden": [
            "global weight fine-tuning",
            "nonlinear adapter",
            "new input features",
            "subregion-specific models",
            "adapter ensemble",
            "test-driven reselection",
        ],
    }


def _strip_private(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if not key.startswith("_")
    }


def _selection_gate(
    *,
    selected_name: str,
    nested: dict[str, Any],
    gate_comparison: dict[str, Any],
) -> dict[str, Any]:
    nested_comparison = nested["comparison"]
    regions = nested["regions"]
    family_frequency = (
        nested["selected_family_counts"].get(selected_name, 0)
        / nested["outer_splits"]
    )
    subregion_checks = {}
    for name in ("ECS", "NSCS"):
        values = regions[name]
        if values["status"] != "ok":
            subregion_checks[name] = {
                "status": "no_samples",
                "passed": False,
            }
            continue
        base_rmse = values["macro_id_base_rmse"]
        adapted_rmse = values["macro_id_adapted_rmse"]
        relative_change = (adapted_rmse - base_rmse) / base_rmse
        subregion_checks[name] = {
            "status": "ok",
            "relative_macro_id_rmse_change": relative_change,
            "passed": relative_change <= 0.02,
        }

    checks = {
        "selected_adapter_is_nonzero": selected_name != "G0_global_only",
        "development_macro_id_rmse_improves_at_least_2_percent": (
            nested_comparison["macro_id_relative_rmse_improvement"] >= 0.02
        ),
        "development_id_win_rate_at_least_60_percent": (
            nested_comparison["id_win_rate"] >= 0.60
        ),
        "outer_selected_family_frequency_at_least_60_percent": (
            family_frequency >= 0.60
        ),
        "development_ecs_not_degraded_over_2_percent": (
            subregion_checks["ECS"]["passed"]
        ),
        "development_nscs_not_degraded_over_2_percent": (
            subregion_checks["NSCS"]["passed"]
        ),
        "development_correction_p99_not_larger_than_global_p99": (
            nested["correction_magnitude"][
                "correction_to_global_p99_ratio"
            ]
            <= 1.0
        ),
        "gate_point_rmse_improves": (
            gate_comparison["adapted_point_metrics"]["rmse"]
            < gate_comparison["base_point_metrics"]["rmse"]
        ),
        "gate_macro_id_rmse_improves": (
            gate_comparison["macro_id_adapted_rmse"]
            < gate_comparison["macro_id_base_rmse"]
        ),
        "no_gate_id_degrades_over_5_percent": (
            gate_comparison["maximum_id_relative_rmse_degradation"] <= 0.05
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "selected_family_outer_frequency": family_frequency,
        "subregion_checks": subregion_checks,
    }


def _confirmation_acceptance(
    comparison: dict[str, Any],
    proxy: dict[str, Any],
) -> dict[str, Any]:
    horizon = proxy["summaries"]["24"]
    trajectory_available = horizon["status"] == "ok"
    checks = {
        "point_joint_r2_improves": (
            comparison["adapted_point_metrics"]["r2_joint"]
            > comparison["base_point_metrics"]["r2_joint"]
        ),
        "point_rmse_improves": (
            comparison["adapted_point_metrics"]["rmse"]
            < comparison["base_point_metrics"]["rmse"]
        ),
        "no_confirmation_id_degrades_over_5_percent": (
            comparison["maximum_id_relative_rmse_degradation"] <= 0.05
        ),
        "trajectory_24h_available": trajectory_available,
        "trajectory_24h_macro_median_improves": (
            trajectory_available
            and horizon["macro_id_adapted_median_km"]
            < horizon["macro_id_base_median_km"]
        ),
        "trajectory_24h_macro_p90_not_degraded_over_5_percent": (
            trajectory_available
            and horizon["macro_id_adapted_p90_km"]
            <= 1.05 * horizon["macro_id_base_p90_km"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def run_selection(code_commit: str) -> dict[str, Any]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    selection_lock_path = ARTIFACT_DIR / "selection_lock.json"
    test_output_path = ARTIFACT_DIR / "test_evaluation.json"
    if selection_lock_path.exists():
        raise FileExistsError(
            "selection_lock.json 已存在；拒绝覆盖已锁定选择。"
        )
    if test_output_path.exists():
        raise FileExistsError(
            "test_evaluation.json 已存在；拒绝重新选择。"
        )

    logger.info("验证 frozen global ONNX 合同与 SHA256")
    base_contract = validate_frozen_global()
    frames = load_filtered_frames()
    lineage = derive_global_lineage_split(frames)
    counts = {
        name: lineage["splits"][name]["n_original_ids"]
        for name in ("development", "gate", "confirmation")
    }
    if counts != {"development": 19, "gate": 2, "confirmation": 2}:
        raise RuntimeError(f"预期 global lineage 19/2/2，实际为 {counts}")

    protocol = _protocol(code_commit)
    protocol_path = ARTIFACT_DIR / "selection_protocol.json"
    _write_json(protocol_path, protocol)

    session = _session()
    development_ids = lineage["splits"]["development"]["original_ids"]
    gate_ids = lineage["splits"]["gate"]["original_ids"]
    confirmation_ids = lineage["splits"]["confirmation"]["original_ids"]
    logger.info("构造 development=%d ID 与 gate=%d ID", 19, 2)
    development = build_adapter_data(frames, development_ids, session)
    gate = build_adapter_data(frames, gate_ids, session)
    lineage["splits"]["development"]["n_samples"] = len(
        development.target
    )
    lineage["splits"]["gate"]["n_samples"] = len(gate.target)
    lineage["splits"]["confirmation"]["n_samples"] = None
    lineage["confirmation_data_status"] = "sealed_not_loaded"
    lineage_path = ARTIFACT_DIR / "global_lineage_split_manifest.json"
    _write_json(lineage_path, lineage)

    logger.info("运行19-ID nested GroupKFold adapter 选择")
    nested = run_nested_development_cv(development)
    nested_serializable = _strip_private(nested)
    logger.info("在全部19个 development ID 上锁定 family/lambda")
    final_cv = select_candidate_by_cv(
        development,
        n_splits=OUTER_SPLITS,
        seed=RANDOM_SEED,
    )
    selected_name = final_cv["selected_adapter_name"]
    selected_lambda = final_cv["selected_lambda"]
    selected_spec = ADAPTER_SPECS[selected_name]
    logger.info(
        "选择结果: %s lambda=%s",
        selected_name,
        selected_lambda,
    )

    development_adapter = fit_adapter(
        development,
        selected_spec,
        selected_lambda,
    )
    development_adapter_path = ARTIFACT_DIR / "adapter_fit_development.json"
    _write_json(
        development_adapter_path,
        {
            **development_adapter.to_dict(),
            "selection_code_git_commit": code_commit,
            "fit_population": "19 frozen-global train CMS IDs",
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
            "nested_selection": nested_serializable,
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
        logger.info("门控通过；用 development+gate 21 ID 拟合锁定 adapter")
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
                **adapter_for_test.to_dict(),
                "selection_code_git_commit": code_commit,
                "fit_population": (
                    "19 frozen-global train CMS IDs plus "
                    "2 frozen-global validation CMS IDs"
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
        "data_source": {
            "path": _relative(FILTERED_DATA_PATH),
            "sha256": sha256_file(FILTERED_DATA_PATH),
        },
        "global_split_manifest": {
            "path": _relative(GLOBAL_SPLIT_MANIFEST_PATH),
            "sha256": sha256_file(GLOBAL_SPLIT_MANIFEST_PATH),
        },
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
            "Base-only aggregate metrics were inspected during lineage audit; "
            "no adapter prediction has been evaluated on these IDs."
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
    logger.info(
        "selection lock 完成: gate_passed=%s test_status=%s",
        decision["passed"],
        selection_lock["test_status"],
    )
    return selection_lock


def run_test_once(code_commit: str) -> dict[str, Any]:
    output_path = ARTIFACT_DIR / "test_evaluation.json"
    result_output_path = RESULT_DIR / "test_evaluation.json"
    if output_path.exists() or result_output_path.exists():
        raise FileExistsError("confirmation test 已评价；拒绝重复运行。")
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

    frames = load_filtered_frames()
    lineage = derive_global_lineage_split(frames)
    confirmation_ids = lineage["splits"]["confirmation"]["original_ids"]
    if hash_ids(confirmation_ids) != selection[
        "confirmation_original_ids_sha256"
    ]:
        raise RuntimeError("confirmation original_ID 清单发生变化。")
    if set(confirmation_ids) & set(adapter.training_original_ids):
        raise RuntimeError("confirmation ID 进入 adapter fitting population。")

    session = _session()
    logger.info("解封2个 frozen-global test CMS ID，一次性评价")
    confirmation = build_adapter_data(frames, confirmation_ids, session)
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

    legacy_manifest = _read_json(SOURCE_CMS_SPLIT_MANIFEST)
    legacy_ids = legacy_manifest["splits"]["test"]["original_ids"]
    legacy = build_adapter_data(frames, legacy_ids, session)
    legacy_correction = predict_correction(adapter, legacy)
    legacy_prediction = legacy.global_prediction + legacy_correction
    legacy_report = {
        "status": "secondary_mixed_global_provenance_and_partly_in_sample",
        "original_ids": legacy_ids,
        "adapter_training_id_overlap": sorted(
            set(legacy_ids) & set(adapter.training_original_ids)
        ),
        "comparison": compare_predictions(legacy, legacy_prediction),
        "regions": compare_by_region(legacy, legacy_prediction),
    }

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
        "legacy_cms_15_4_4_test": legacy_report,
        "acceptance": acceptance,
        "recommendation": (
            "eligible_for_onnx_freeze"
            if acceptance["passed"]
            else "stop_no_onnx_freeze"
        ),
    }
    _write_json(output_path, report)
    _write_json(result_output_path, report)
    logger.info(
        "confirmation 完成: base R2=%.6f adapted R2=%.6f "
        "base RMSE=%.6f adapted RMSE=%.6f recommendation=%s",
        comparison["base_point_metrics"]["r2_joint"],
        comparison["adapted_point_metrics"]["r2_joint"],
        comparison["base_point_metrics"]["rmse"],
        comparison["adapted_point_metrics"]["rmse"],
        report["recommendation"],
    )
    return report


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s - %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen-global CMS linear adapter"
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
