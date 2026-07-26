"""统一回归指标计算测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score


MODELS_DIR = Path(__file__).resolve().parents[1] / "src" / "models"
sys.path.insert(0, str(MODELS_DIR))

from evaluation import regression_metrics  # noqa: E402


class RegressionMetricsTest(unittest.TestCase):
    def test_joint_r2_is_float64_component_mean(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.normal(size=(10_000, 2)).astype(np.float32)
        y_pred = (y_true + rng.normal(scale=0.5, size=y_true.shape)).astype(
            np.float32
        )

        metrics = regression_metrics(y_true, y_pred)
        expected = r2_score(
            y_true.astype(np.float64),
            y_pred.astype(np.float64),
            multioutput="raw_values",
        )
        self.assertAlmostEqual(metrics["r2_u"], expected[0], places=14)
        self.assertAlmostEqual(metrics["r2_v"], expected[1], places=14)
        self.assertAlmostEqual(metrics["r2_joint"], expected.mean(), places=14)


if __name__ == "__main__":
    unittest.main()
