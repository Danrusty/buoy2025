"""回归评估指标的统一实现。"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    用 float64 计算双输出速度回归指标。

    joint R2 明确定义为 residual_u 和 residual_v R2 的等权平均，避免
    数百万个 float32 样本在二维按列累加时产生可见数值误差。
    """
    true64 = np.asarray(y_true, dtype=np.float64)
    pred64 = np.asarray(y_pred, dtype=np.float64)
    if true64.shape != pred64.shape:
        raise ValueError(
            f"y_true/y_pred shape 不一致: {true64.shape} vs {pred64.shape}"
        )
    if true64.ndim != 2 or true64.shape[1] != 2:
        raise ValueError(f"预期双输出数组 (N, 2)，实际为 {true64.shape}")

    r2_values = np.asarray(
        r2_score(true64, pred64, multioutput="raw_values"),
        dtype=np.float64,
    )
    error = pred64 - true64
    return {
        "r2_u": float(r2_values[0]),
        "r2_v": float(r2_values[1]),
        "r2_joint": float(r2_values.mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }
