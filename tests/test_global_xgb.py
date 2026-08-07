"""Global XGBoost 受控搜索与效率门槛测试。"""

import unittest

import numpy as np

from src.models.training.run_global_xgb import (
    CANDIDATES,
    accuracy_gate,
    build_features,
    choose_candidate,
)


class GlobalXgbTest(unittest.TestCase):
    def test_core6_and_lat7_order(self):
        cached = {
            "core6": np.arange(12, dtype=np.float32).reshape(2, 6),
            "sin_latitude": np.asarray([0.25, -0.5], dtype=np.float32),
        }
        core6 = build_features(cached, "core6")
        lat7 = build_features(cached, "lat7")
        np.testing.assert_array_equal(core6, cached["core6"])
        np.testing.assert_array_equal(lat7[:, :6], cached["core6"])
        np.testing.assert_array_equal(lat7[:, 6], [0.25, -0.5])

    def test_candidate_selection_uses_validation_then_tree_count(self):
        results = [
            {
                "name": CANDIDATES[0]["name"],
                "validation": {"rmse": 0.20},
                "total_boosted_rounds": 100,
            },
            {
                "name": CANDIDATES[1]["name"],
                "validation": {"rmse": 0.19},
                "total_boosted_rounds": 200,
            },
        ]
        self.assertEqual(
            choose_candidate(results)["name"],
            CANDIDATES[1]["name"],
        )
        results[0]["validation"]["rmse"] = 0.19
        self.assertEqual(
            choose_candidate(results)["name"],
            CANDIDATES[0]["name"],
        )

    def test_efficiency_gate_requires_all_accuracy_checks(self):
        frozen = {
            "row_weighted": {"r2_joint": 0.12, "rmse": 0.20},
            "macro_original_id": {"r2_joint": -0.30, "rmse": 0.19},
        }
        candidate = {
            "row_weighted": {"r2_joint": 0.14, "rmse": 0.197},
            "macro_original_id": {"r2_joint": -0.25, "rmse": 0.18},
        }
        self.assertTrue(accuracy_gate(candidate, frozen)["passed"])
        candidate["macro_original_id"]["rmse"] = 0.20
        self.assertFalse(accuracy_gate(candidate, frozen)["passed"])


if __name__ == "__main__":
    unittest.main()
