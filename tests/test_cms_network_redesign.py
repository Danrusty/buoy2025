"""CMS core6 网络重构候选的接口与排序测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
sys.path.insert(0, str(MODELS_DIR))

from cms_network_redesign import (  # noqa: E402
    ARCHITECTURES,
    _rank_cv,
    _rank_fixed_validation,
    _training_contract,
    build_model,
)


class CmsNetworkRedesignTest(unittest.TestCase):
    def test_every_candidate_preserves_core6_to_two_output_interface(self) -> None:
        values = torch.zeros((7, 6), dtype=torch.float32)
        parameter_counts = {}
        for name, spec in ARCHITECTURES.items():
            model = build_model(spec)
            output = model(values)
            self.assertEqual(output.shape, (7, 2), msg=name)
            parameter_counts[name] = sum(
                parameter.numel() for parameter in model.parameters()
            )
        self.assertEqual(parameter_counts["linear_core6"], 14)
        self.assertLess(
            parameter_counts["plain_16_16"],
            parameter_counts["legacy_core6_433k"],
        )
        self.assertLess(
            parameter_counts["linear_skip_32_32"],
            parameter_counts["legacy_core6_433k"],
        )
        self.assertEqual(
            parameter_counts["legacy_core6_433k"],
            433_538,
        )

    def test_training_contract_keeps_non_architecture_controls_frozen(self) -> None:
        contract = _training_contract()
        self.assertEqual(len(contract["features"]), 6)
        self.assertEqual(contract["loss"], "MSELoss")
        self.assertEqual(contract["batch_size"], 8192)
        self.assertEqual(contract["early_stopping_patience"], 20)
        self.assertEqual(contract["optimizer"], "AdamW")
        self.assertEqual(contract["random_seed"], 42)
        self.assertIn("new features", contract["forbidden_changes"])
        self.assertIn(
            "automatic hyperparameter search",
            contract["forbidden_changes"],
        )

    def test_rankings_use_r2_then_rmse_then_capacity(self) -> None:
        cv = {
            "a": {
                "summary": {
                    "mean_r2_joint": 0.2,
                    "mean_rmse": 0.4,
                    "parameter_count": 100,
                }
            },
            "b": {
                "summary": {
                    "mean_r2_joint": 0.3,
                    "mean_rmse": 0.5,
                    "parameter_count": 200,
                }
            },
            "c": {
                "summary": {
                    "mean_r2_joint": 0.2,
                    "mean_rmse": 0.3,
                    "parameter_count": 300,
                }
            },
        }
        self.assertEqual(_rank_cv(cv), ["b", "c", "a"])

        fixed = {
            "a": {
                "validation_metrics": {"r2_joint": 0.1, "rmse": 0.4},
                "parameter_count": 100,
            },
            "b": {
                "validation_metrics": {"r2_joint": 0.2, "rmse": 0.5},
                "parameter_count": 200,
            },
            "c": {
                "validation_metrics": {"r2_joint": 0.2, "rmse": 0.3},
                "parameter_count": 300,
            },
        }
        self.assertEqual(_rank_fixed_validation(fixed), ["c", "b", "a"])


if __name__ == "__main__":
    unittest.main()
