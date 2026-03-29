"""
data_loader.py
==============
数据加载、目标计算、防泄漏切分与特征标准化模块。

职责：
  1. 从 trajectories_with_all_features.pkl 加载轨迹列表
  2. 计算残差漂移目标 (residual_u / residual_v)
  3. 按轨迹 ID 切分（70/15/15），防止数据泄漏
  4. 对输入特征 X 做 StandardScaler（仅在 train 上 fit）
  5. 用 joblib 保存 scaler，供后续 ONNX/Fortran 部署使用

运行方式（独立测试）：
  cd src/models
  python data_loader.py
"""

import os
import pickle
import logging
from datetime import datetime

import gc

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# 路径配置（从 src/models/ 目录运行）
# ==============================================================================
DATA_PATH      = '../../processed_data/trajectories_with_all_features.pkl'
TRAINED_MODELS_DIR = '../../trained_models'
LOG_DIR        = '../../logs'
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================================================================
# 常量定义
# ==============================================================================
# 9 个输入特征（风场 5 个 + 波浪 4 个）
FEATURE_COLS = [
    'era5_u10', 'era5_v10', 'era5_wind_speed',
    'era5_wind_dir_sin', 'era5_wind_dir_cos',
    'era5_swh', 'era5_mwp',
    'era5_wave_dir_sin', 'era5_wave_dir_cos',
]
# 线性基准只用风速分量（物理意义：WDF 模型 v_drift = α * v_wind）
WIND_COLS = ['era5_u10', 'era5_v10']

# 目标（残差漂移）
TARGET_COLS = ['residual_u', 'residual_v']

# 背景流列（来自 CFSv2）
CURRENT_COLS = ['cfsv2_u', 'cfsv2_v']

# 浮标观测速度列
OBS_COLS = ['ve', 'vn']

RANDOM_SEED = 42


# ==============================================================================
# 内存监控辅助
# ==============================================================================
def _mem_mb() -> float:
    """返回当前进程 RSS 内存（MB）。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 ** 2
    except ImportError:
        return float('nan')


# ==============================================================================
# 日志配置（避免重复添加 handler）
# ==============================================================================
def _setup_logger(name: str = 'data_loader') -> logging.Logger:
    log_path = os.path.join(
        LOG_DIR,
        f"data_loader_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ==============================================================================
# 核心函数
# ==============================================================================
def load_and_split_data(
    filepath: str = DATA_PATH,
    random_seed: int = RANDOM_SEED,
    sample_mode: bool = False,
    sample_size: int = 200,
):
    """
    加载数据、计算残差目标、按轨迹切分并标准化。

    Parameters
    ----------
    filepath    : pkl 文件路径（List[pd.DataFrame]）
    random_seed : 随机种子，保证可复现
    sample_mode : True 时只取 sample_size 条轨迹用于快速调试
    sample_size : 采样模式下的轨迹数量

    Returns
    -------
    splits : dict，包含：
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        X_train_wind, X_val_wind, X_test_wind,   # 线性基准用（原始尺度）
        x_scaler,                                 # 已 fit 的 StandardScaler
        feature_cols,                             # 特征列名列表
    """
    logger = _setup_logger()
    mode_tag = "【采样模式】" if sample_mode else "【完整模式】"
    logger.info(f"=== 开始数据加载 {mode_tag} ===")

    # ------------------------------------------------------------------
    # 步骤 1/4: 加载原始轨迹列表
    # ------------------------------------------------------------------
    logger.info(f"步骤 1/4: 从 '{filepath}' 加载轨迹数据...")
    try:
        with open(filepath, 'rb') as f:
            all_trajs = pickle.load(f)
    except FileNotFoundError:
        logger.error(f"文件未找到: {filepath}")
        raise

    logger.info(f"原始轨迹总数: {len(all_trajs)}")
    logger.info(f"当前内存（加载后）: {_mem_mb():.0f} MB")

    if sample_mode:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(all_trajs), size=min(sample_size, len(all_trajs)), replace=False)
        all_trajs = [all_trajs[i] for i in sorted(idx)]
        logger.info(f"采样后轨迹数: {len(all_trajs)}")

    # ------------------------------------------------------------------
    # 步骤 2/4: 过滤缺失必要列的轨迹，计算残差目标
    # ------------------------------------------------------------------
    logger.info("步骤 2/4: 过滤无效轨迹并计算漂移残差...")
    required_cols = set(FEATURE_COLS + OBS_COLS + CURRENT_COLS)
    # 只保留后续需要的列，减少内存占用
    keep_cols = FEATURE_COLS + OBS_COLS + CURRENT_COLS  # 不含重复
    keep_cols = list(dict.fromkeys(keep_cols))          # 去重保序
    valid_trajs = []
    skip_count = 0

    for i, df in enumerate(all_trajs):
        missing = required_cols - set(df.columns)
        if missing:
            skip_count += 1
            continue
        # 只取必要列，立即丢弃其余列（减少内存）
        sub = df[keep_cols].dropna()
        if len(sub) == 0:
            skip_count += 1
            continue
        df_clean = sub.copy()
        # 计算残差目标（浮标观测速度 - 背景流速度 = 风浪致漂移）
        df_clean['residual_u'] = df_clean['ve'] - df_clean['cfsv2_u']
        df_clean['residual_v'] = df_clean['vn'] - df_clean['cfsv2_v']
        df_clean['_traj_id'] = i  # 轨迹级别唯一 ID，防泄漏切分用
        valid_trajs.append(df_clean)

    # ★ 释放原始轨迹列表（2-4 GB），不再需要
    del all_trajs
    gc.collect()
    logger.info(f"有效轨迹数: {len(valid_trajs)}（跳过: {skip_count}）  内存: {_mem_mb():.0f} MB")
    if not valid_trajs:
        raise ValueError("没有有效轨迹，请检查数据列名是否正确。")

    # ------------------------------------------------------------------
    # 步骤 3/4: 按轨迹 ID 切分（70/15/15），防止数据泄漏
    # ------------------------------------------------------------------
    logger.info("步骤 3/4: 按轨迹 ID 进行防泄漏切分（70/15/15）...")

    traj_ids = np.arange(len(valid_trajs))

    # 先切出 15% test
    train_val_ids, test_ids = train_test_split(
        traj_ids, test_size=0.15, random_state=random_seed
    )
    # 再从 train+val 中切出 val（使 val ≈ 15% of total，故比例 ≈ 15/85 ≈ 0.176）
    train_ids, val_ids = train_test_split(
        train_val_ids, test_size=0.176, random_state=random_seed
    )

    def _collect(ids):
        frames = [valid_trajs[i] for i in ids]
        return pd.concat(frames, ignore_index=True)

    df_train = _collect(train_ids)
    df_val   = _collect(val_ids)
    df_test  = _collect(test_ids)

    # ★ 释放 valid_trajs 列表，已完成切分
    del valid_trajs
    gc.collect()

    logger.info(
        f"切分完成 | 训练: {len(train_ids)} 条轨迹 / {len(df_train)} 点 | "
        f"验证: {len(val_ids)} 条轨迹 / {len(df_val)} 点 | "
        f"测试: {len(test_ids)} 条轨迹 / {len(df_test)} 点  内存: {_mem_mb():.0f} MB"
    )

    # ------------------------------------------------------------------
    # 步骤 4/4: 特征标准化（仅 X，y 保持原始 m/s 尺度）
    # ------------------------------------------------------------------
    logger.info("步骤 4/4: 对输入特征 X 进行标准化（只在 train 上 fit）...")

    X_train_raw  = df_train[FEATURE_COLS].values.astype(np.float32)
    X_train_wind = df_train[WIND_COLS].values.astype(np.float32)
    y_train      = df_train[TARGET_COLS].values.astype(np.float32)
    del df_train; gc.collect()  # ★ 立即释放，转为 numpy 后不再需要 DataFrame

    X_val_raw  = df_val[FEATURE_COLS].values.astype(np.float32)
    X_val_wind = df_val[WIND_COLS].values.astype(np.float32)
    y_val      = df_val[TARGET_COLS].values.astype(np.float32)
    del df_val; gc.collect()

    X_test_raw  = df_test[FEATURE_COLS].values.astype(np.float32)
    X_test_wind = df_test[WIND_COLS].values.astype(np.float32)
    y_test      = df_test[TARGET_COLS].values.astype(np.float32)
    del df_test; gc.collect()

    logger.info(f"DataFrame → NumPy 转换完成  内存: {_mem_mb():.0f} MB")

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_val   = x_scaler.transform(X_val_raw)
    X_test  = x_scaler.transform(X_test_raw)

    # 保存 scaler（joblib，供部署使用）
    scaler_path = os.path.join(TRAINED_MODELS_DIR, 'x_scaler.pkl')
    joblib.dump(x_scaler, scaler_path)
    logger.info(f"StandardScaler 已保存到: {scaler_path}")

    # 统计摘要
    logger.info(
        f"X_train 形状: {X_train.shape}，"
        f"X_val: {X_val.shape}，"
        f"X_test: {X_test.shape}"
    )
    logger.info(
        f"y_train 形状: {y_train.shape}，"
        f"残差 u 均值: {y_train[:, 0].mean():.4f} m/s，"
        f"残差 v 均值: {y_train[:, 1].mean():.4f} m/s"
    )
    logger.info("=== 数据加载完毕 ===")

    return {
        'X_train': X_train,  'y_train': y_train,
        'X_val':   X_val,    'y_val':   y_val,
        'X_test':  X_test,   'y_test':  y_test,
        'X_train_wind': X_train_wind,
        'X_val_wind':   X_val_wind,
        'X_test_wind':  X_test_wind,
        'x_scaler':   x_scaler,
        'feature_cols': FEATURE_COLS,
    }


# ==============================================================================
# 独立运行测试
# ==============================================================================
if __name__ == '__main__':
    # 采样模式快速验证（只加载 200 条轨迹）
    import sys
    sample = '--full' not in sys.argv

    splits = load_and_split_data(sample_mode=sample, sample_size=200)

    print("\n===== 数据集摘要 =====")
    for key in ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test']:
        arr = splits[key]
        print(f"  {key:12s}: shape={arr.shape}, dtype={arr.dtype}")

    print(f"\n特征列 ({len(splits['feature_cols'])} 个): {splits['feature_cols']}")
    print(f"Scaler 均值 (前3): {splits['x_scaler'].mean_[:3]}")
    print("===== 验证通过 =====")
