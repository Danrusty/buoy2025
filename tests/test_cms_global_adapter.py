"""Frozen-global CMS adapter 数学与分组约束测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.adapters.cms_global_adapter import (  # noqa: E402
    ADAPTER_SPECS,
    AdapterData,
    _basis,
    compare_predictions,
    equal_id_row_weights,
    fit_adapter,
    predict_correction,
    trajectory_proxy,
)
from src.models.adapters.run_cms_global_adapter import (  # noqa: E402
    _confirmation_acceptance,
)


def _synthetic_data() -> AdapterData:
    rng = np.random.default_rng(42)
    groups = np.repeat(["a", "b", "c"], [5, 20, 8])
    count = len(groups)
    features = rng.normal(size=(count, 6))
    global_prediction = rng.normal(scale=0.05, size=(count, 2))
    correction = np.column_stack(
        (
            0.02 * features[:, 0]
            - 0.01 * features[:, 1]
            + 0.03,
            0.015 * features[:, 0]
            + 0.025 * features[:, 1]
            - 0.02,
        )
    )
    target = global_prediction + correction
    frame = pd.DataFrame(
        {
            "original_ID": groups,
            "_cms_episode_key": [f"{value}:0" for value in groups],
            "_cms_episode_step": np.zeros(count, dtype=int),
            "time": pd.Timestamp("2020-01-01"),
        }
    )
    memberships = {
        "CMS": np.ones(count, dtype=bool),
        "BYS": np.zeros(count, dtype=bool),
        "ECS": np.ones(count, dtype=bool),
        "NSCS": np.zeros(count, dtype=bool),
    }
    return AdapterData(
        frame=frame,
        features=features,
        target=target,
        global_prediction=global_prediction,
        groups=groups,
        memberships=memberships,
    )


class CmsGlobalAdapterTest(unittest.TestCase):
    def test_candidate_basis_shapes_match_parameter_counts(self) -> None:
        data = _synthetic_data()
        for spec in ADAPTER_SPECS.values():
            self.assertEqual(
                _basis(spec, data).shape,
                (len(data.target), 2, spec.parameter_count),
                msg=spec.name,
            )

    def test_equal_id_weights_do_not_favor_long_ids(self) -> None:
        data = _synthetic_data()
        weights = equal_id_row_weights(data.groups)
        for original_id in np.unique(data.groups):
            self.assertAlmostEqual(
                float(weights[data.groups == original_id].sum()),
                1.0,
            )

    def test_unregularized_full_wind_recovers_synthetic_correction(self) -> None:
        data = _synthetic_data()
        spec = ADAPTER_SPECS["G3_wind_full6"]
        adapter = fit_adapter(data, spec, 0.0)
        prediction = (
            data.global_prediction + predict_correction(adapter, data)
        )
        np.testing.assert_allclose(prediction, data.target, atol=1e-10)
        comparison = compare_predictions(data, prediction)
        self.assertLess(
            comparison["adapted_point_metrics"]["rmse"],
            1e-10,
        )

    def test_zero_adapter_is_exact_frozen_global(self) -> None:
        data = _synthetic_data()
        spec = ADAPTER_SPECS["G0_global_only"]
        adapter = fit_adapter(data, spec, None)
        correction = predict_correction(adapter, data)
        np.testing.assert_array_equal(correction, np.zeros_like(correction))

    def test_trajectory_proxy_accumulates_known_velocity_error(self) -> None:
        count = 24
        groups = np.asarray(["a"] * count)
        frame = pd.DataFrame(
            {
                "original_ID": groups,
                "_cms_episode_key": ["a:0"] * count,
                "_cms_episode_step": np.arange(count),
                "time": pd.date_range(
                    "2020-01-01",
                    periods=count,
                    freq="h",
                ),
            }
        )
        target = np.zeros((count, 2))
        base = np.column_stack((np.full(count, 1.0), np.zeros(count)))
        adapted = np.column_stack((np.full(count, 0.5), np.zeros(count)))
        data = AdapterData(
            frame=frame,
            features=np.zeros((count, 6)),
            target=target,
            global_prediction=base,
            groups=groups,
            memberships={
                "CMS": np.ones(count, dtype=bool),
                "BYS": np.zeros(count, dtype=bool),
                "ECS": np.ones(count, dtype=bool),
                "NSCS": np.zeros(count, dtype=bool),
            },
        )
        result = trajectory_proxy(data, adapted, horizons=(24,))
        summary = result["summaries"]["24"]
        self.assertAlmostEqual(
            summary["macro_id_base_median_km"],
            86.4,
        )
        self.assertAlmostEqual(
            summary["macro_id_adapted_median_km"],
            43.2,
        )

    def test_confirmation_acceptance_requires_point_and_trajectory_gain(
        self,
    ) -> None:
        comparison = {
            "base_point_metrics": {"r2_joint": 0.1, "rmse": 0.4},
            "adapted_point_metrics": {"r2_joint": 0.2, "rmse": 0.3},
            "maximum_id_relative_rmse_degradation": 0.0,
        }
        proxy = {
            "summaries": {
                "24": {
                    "status": "ok",
                    "macro_id_base_median_km": 10.0,
                    "macro_id_adapted_median_km": 8.0,
                    "macro_id_base_p90_km": 20.0,
                    "macro_id_adapted_p90_km": 19.0,
                }
            }
        }
        self.assertTrue(
            _confirmation_acceptance(comparison, proxy)["passed"]
        )
        proxy["summaries"]["24"]["macro_id_adapted_p90_km"] = 22.0
        self.assertFalse(
            _confirmation_acceptance(comparison, proxy)["passed"]
        )


if __name__ == "__main__":
    unittest.main()
