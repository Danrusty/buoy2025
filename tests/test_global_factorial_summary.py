"""四格析因效应计算测试。"""

import unittest

from src.models.training.summarize_global_factorial import (
    factorial_effects,
)


class GlobalFactorialSummaryTest(unittest.TestCase):
    def test_factorial_effects(self):
        cells = {
            "A": {"metric": 1.0},
            "B": {"metric": 3.0},
            "C": {"metric": 4.0},
            "D": {"metric": 10.0},
        }
        effect = factorial_effects(cells)["metric"]
        self.assertEqual(effect["latitude_under_mlp_B_minus_A"], 2.0)
        self.assertEqual(effect["model_without_latitude_C_minus_A"], 3.0)
        self.assertEqual(effect["latitude_under_xgboost_D_minus_C"], 6.0)
        self.assertEqual(effect["model_with_latitude_D_minus_B"], 7.0)
        self.assertEqual(
            effect["interaction_D_minus_C_minus_B_plus_A"],
            4.0,
        )

    def test_rejects_missing_cell(self):
        with self.assertRaisesRegex(ValueError, "实验格"):
            factorial_effects(
                {
                    "A": {"metric": 1.0},
                    "B": {"metric": 2.0},
                    "C": {"metric": 3.0},
                }
            )


if __name__ == "__main__":
    unittest.main()
