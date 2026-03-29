"""
baseline.py
===========
线性回归物理基准模型：估算恒定风漂移系数（WDF）。

物理背景：
  漂流浮标残差速度 ≈ α * 风速向量
  即 residual_u ≈ α * u10，residual_v ≈ α * v10
  WDF（Wind Drift Factor）α 通常在 0.02 ~ 0.04 之间。

本脚本：
  1. 调用 data_loader 加载数据（采样或完整）
  2. 用 (u10, v10) → (residual_u, residual_v) 拟合线性回归
  3. 打印回归系数矩阵、WDF 估算值、R²、RMSE
  4. 与"预测零"基准比较

运行方式：
  cd src/models
  conda run -n buoy-drifter python baseline.py            # 采样模式（快速）
  conda run -n buoy-drifter python baseline.py --full     # 完整数据集
"""

import sys
import logging
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from data_loader import load_and_split_data

logger = logging.getLogger(__name__)


# ==============================================================================
# WDF 基准模型
# ==============================================================================
def run_linear_baseline(splits: dict) -> dict:
    """
    在原始尺度的风速分量上拟合线性回归，估算 WDF 系数。

    Parameters
    ----------
    splits : load_and_split_data() 的返回值

    Returns
    -------
    results : dict，包含 r2_u, r2_v, rmse, wdf_estimate, models
    """
    logger.info("=== 开始线性回归 WDF 基准 ===")

    X_tr_w  = splits['X_train_wind']   # shape (N_train, 2)，[u10, v10]
    X_te_w  = splits['X_test_wind']    # shape (N_test,  2)
    y_train = splits['y_train']        # shape (N_train, 2)，[residual_u, residual_v]
    y_test  = splits['y_test']         # shape (N_test,  2)

    # ------------------------------------------------------------------
    # 拟合两个独立的线性回归（u 分量 和 v 分量）
    # 不强制无截距，让数据决定（Stokes 漂移等会引入轻微偏置）
    # ------------------------------------------------------------------
    reg_u = LinearRegression()
    reg_v = LinearRegression()

    reg_u.fit(X_tr_w, y_train[:, 0])
    reg_v.fit(X_tr_w, y_train[:, 1])

    # 系数矩阵（2×2）：
    #   行 = [residual_u 方程, residual_v 方程]
    #   列 = [u10 系数, v10 系数]
    coef_matrix = np.array([reg_u.coef_, reg_v.coef_])
    intercepts  = np.array([reg_u.intercept_, reg_v.intercept_])

    # ------------------------------------------------------------------
    # WDF 估算方法：
    #   各向同性假设下，系数矩阵应近似为 α * I（α = WDF）
    #   取对角线均值作为 WDF 估计值
    # ------------------------------------------------------------------
    wdf_estimate = float(np.mean(np.diag(coef_matrix)))
    wdf_offdiag  = float(np.mean([coef_matrix[0, 1], coef_matrix[1, 0]]))  # 理想为 0

    # ------------------------------------------------------------------
    # 在测试集上评估
    # ------------------------------------------------------------------
    pred_u = reg_u.predict(X_te_w)
    pred_v = reg_v.predict(X_te_w)
    y_pred = np.column_stack([pred_u, pred_v])

    r2_u      = r2_score(y_test[:, 0], pred_u)
    r2_v      = r2_score(y_test[:, 1], pred_v)
    r2_joint  = r2_score(y_test, y_pred)
    rmse      = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae       = float(np.mean(np.abs(y_pred - y_test)))

    # 对比基准：全预测为 0（即忽略风场影响）
    rmse_zero = float(np.sqrt(np.mean(y_test ** 2)))
    r2_zero   = r2_score(y_test, np.zeros_like(y_test))

    # ------------------------------------------------------------------
    # 输出报告
    # ------------------------------------------------------------------
    sep = "=" * 55

    logger.info(sep)
    logger.info("  线性回归 WDF 基准 — 结果报告")
    logger.info(sep)

    logger.info("\n【回归系数矩阵】（行: 目标分量, 列: 输入特征）")
    logger.info(f"  {'':12s}   u10        v10        截距")
    logger.info(f"  residual_u : {coef_matrix[0,0]:+.5f}   {coef_matrix[0,1]:+.5f}   {intercepts[0]:+.5f}")
    logger.info(f"  residual_v : {coef_matrix[1,0]:+.5f}   {coef_matrix[1,1]:+.5f}   {intercepts[1]:+.5f}")

    logger.info("\n【WDF 系数估算】")
    logger.info(f"  对角线均值 WDF ≈ {wdf_estimate:.4f}  "
                f"（{wdf_estimate*100:.2f}%，预期 2~4%）")
    logger.info(f"  非对角项均值    ≈ {wdf_offdiag:.4f}  "
                f"（接近 0 表示各向同性）")

    logger.info("\n【测试集性能 (线性回归)】")
    logger.info(f"  R²  (residual_u) : {r2_u:.4f}")
    logger.info(f"  R²  (residual_v) : {r2_v:.4f}")
    logger.info(f"  R²  (联合)       : {r2_joint:.4f}")
    logger.info(f"  RMSE             : {rmse:.4f} m/s")
    logger.info(f"  MAE              : {mae:.4f} m/s")

    logger.info("\n【对比：预测全零基准（忽略风场）】")
    logger.info(f"  R²   (全零) : {r2_zero:.4f}")
    logger.info(f"  RMSE (全零) : {rmse_zero:.4f} m/s")
    logger.info(f"  线性回归提升 RMSE: {(rmse_zero - rmse):.4f} m/s "
                f"({(rmse_zero - rmse)/rmse_zero*100:.1f}%)")

    logger.info(sep)
    logger.info("  ↑ 以上指标将作为 MLP 的超越目标")
    logger.info(sep)

    return {
        'r2_u':         r2_u,
        'r2_v':         r2_v,
        'r2_joint':     r2_joint,
        'rmse':         rmse,
        'mae':          mae,
        'wdf_estimate': wdf_estimate,
        'coef_matrix':  coef_matrix,
        'intercepts':   intercepts,
        'reg_u':        reg_u,
        'reg_v':        reg_v,
    }


# ==============================================================================
# 独立运行入口
# ==============================================================================
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%H:%M:%S',
    )
    sample = '--full' not in sys.argv

    splits  = load_and_split_data(sample_mode=sample, sample_size=200)
    results = run_linear_baseline(splits)

    print(f"\n>>> WDF 估算: {results['wdf_estimate']*100:.2f}%")
    print(f">>> 线性基准 R²(联合): {results['r2_joint']:.4f}")
    print(f">>> 线性基准 RMSE: {results['rmse']:.4f} m/s")
