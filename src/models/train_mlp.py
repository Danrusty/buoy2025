"""
train_mlp.py
============
Stateless MLP 训练脚本：预测漂流浮标残差速度（residual_u / residual_v）。

网络结构：
  Input(9) -> Linear(256) -> BN -> ReLU
           -> Linear(128) -> BN -> ReLU
           -> Linear(64)  -> BN -> ReLU
           -> Linear(2)

训练配置：
  - Loss      : MSELoss
  - Optimizer : AdamW (lr=1e-3)
  - Scheduler : ReduceLROnPlateau（监控 val_loss）
  - Early Stop: patience=5, max_epochs=50
  - Batch Size: 16384（适合大数据集）

运行方式：
  cd src/models
  conda run -n buoy-drifter python train_mlp.py            # 采样模式（快速验证）
  conda run -n buoy-drifter python train_mlp.py --full     # 完整训练
"""

import os
import sys
import logging
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境下保存图片
import matplotlib.pyplot as plt

from data_loader import load_and_split_data, _setup_logger
from baseline import run_linear_baseline

# ==============================================================================
# 路径配置
# ==============================================================================
TRAINED_MODELS_DIR = '../../trained_models'
RESULTS_DIR        = '../../results'
LOG_DIR            = '../../logs'
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================================================================
# 超参数
# ==============================================================================
BATCH_SIZE    = 16384
EPOCHS        = 50
PATIENCE      = 5
LR            = 1e-3
LR_FACTOR     = 0.5    # ReduceLROnPlateau 衰减因子
LR_PATIENCE   = 3      # 学习率衰减耐心值
LR_MIN        = 1e-5   # 学习率下限
RANDOM_SEED   = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ==============================================================================
# 设备检测（CUDA → MPS → CPU）
# ==============================================================================
def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ==============================================================================
# 模型定义
# ==============================================================================
class ResidualMLP(nn.Module):
    """
    带 BatchNorm 的多层感知机，预测漂流残差速度。

    Input(9) -> [Linear(256)->BN->ReLU] -> [Linear(128)->BN->ReLU]
             -> [Linear(64)->BN->ReLU]  -> Linear(2)
    """
    def __init__(self, input_size: int = 9, output_size: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==============================================================================
# 辅助函数
# ==============================================================================
def _make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    pin = torch.cuda.is_available()  # pin_memory 只在 CUDA 下有效，CPU/MPS 下反而浪费内存
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      pin_memory=pin, num_workers=0)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
              criterion: nn.Module) -> tuple[float, float]:
    """返回 (val_loss_mse, val_r2_joint)。"""
    model.eval()
    preds_list, targets_list = [], []
    total_loss, total_samples = 0.0, 0

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        pred = model(X_b)
        loss = criterion(pred, y_b)
        total_loss    += loss.item() * len(X_b)
        total_samples += len(X_b)
        preds_list.append(pred.cpu().numpy())
        targets_list.append(y_b.cpu().numpy())

    preds   = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)
    avg_loss = total_loss / total_samples
    r2 = r2_score(targets, preds)
    return avg_loss, r2


# ==============================================================================
# 训练主函数
# ==============================================================================
def train(splits: dict, logger: logging.Logger) -> dict:
    device = _get_device()
    logger.info(f"使用设备: {device}")

    # 构建 DataLoader
    train_loader = _make_loader(splits['X_train'], splits['y_train'], shuffle=True)
    val_loader   = _make_loader(splits['X_val'],   splits['y_val'],   shuffle=False)
    test_loader  = _make_loader(splits['X_test'],  splits['y_test'],  shuffle=False)

    # 初始化模型
    model = ResidualMLP().to(device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN, verbose=False
    )

    best_val_loss  = float('inf')
    no_improve     = 0
    best_model_path = os.path.join(TRAINED_MODELS_DIR, 'best_mlp.pth')
    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'lr': []}

    logger.info(f"\n{'='*60}")
    logger.info("  开始训练  (max_epochs={}, patience={}, batch={})".format(
        EPOCHS, PATIENCE, BATCH_SIZE))
    logger.info(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        # --- 训练阶段 ---
        model.train()
        running_loss, n_samples = 0.0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(X_b)
            n_samples    += len(X_b)

        train_loss = running_loss / n_samples

        # --- 验证阶段 ---
        val_loss, val_r2 = _evaluate(model, val_loader, device, criterion)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['lr'].append(current_lr)

        logger.info(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_R²={val_r2:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # 学习率调度
        scheduler.step(val_loss)

        # 早停 & 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"  ✓ val_loss 改善，已保存最佳模型 -> {best_model_path}")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                logger.info(f"\n早停触发（连续 {PATIENCE} epoch 无改善）")
                break

    logger.info("--- 训练结束 ---")

    # 加载最佳权重进行最终评估
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return {
        'model':      model,
        'history':    history,
        'test_loader': test_loader,
        'device':     device,
        'criterion':  criterion,
    }


# ==============================================================================
# 最终评估（Test 集）& 与线性基准对比
# ==============================================================================
def evaluate_and_compare(train_result: dict, baseline_result: dict,
                         logger: logging.Logger) -> None:
    model       = train_result['model']
    test_loader = train_result['test_loader']
    device      = train_result['device']
    criterion   = train_result['criterion']

    test_loss, _ = _evaluate(model, test_loader, device, criterion)

    # 收集完整预测结果用于细粒度指标
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds_list.append(model(X_b.to(device)).cpu().numpy())
            targets_list.append(y_b.numpy())

    preds   = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    r2_u    = r2_score(targets[:, 0], preds[:, 0])
    r2_v    = r2_score(targets[:, 1], preds[:, 1])
    r2_joint = r2_score(targets, preds)
    rmse    = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae     = float(np.mean(np.abs(preds - targets)))

    sep = "=" * 60
    logger.info(f"\n{sep}")
    logger.info("  最终评估（Test 集）& 与线性基准对比")
    logger.info(sep)
    logger.info(f"\n{'指标':<20} {'线性回归 (WDF)':>18} {'MLP':>14}")
    logger.info("-" * 55)
    logger.info(f"{'R² (residual_u)':<20} {baseline_result['r2_u']:>18.4f} {r2_u:>14.4f}")
    logger.info(f"{'R² (residual_v)':<20} {baseline_result['r2_v']:>18.4f} {r2_v:>14.4f}")
    logger.info(f"{'R² (联合)':<20} {baseline_result['r2_joint']:>18.4f} {r2_joint:>14.4f}")
    logger.info(f"{'RMSE (m/s)':<20} {baseline_result['rmse']:>18.4f} {rmse:>14.4f}")
    logger.info(f"{'MAE (m/s)':<20} {baseline_result['mae']:>18.4f} {mae:>14.4f}")
    logger.info("-" * 55)

    rmse_improve = (baseline_result['rmse'] - rmse) / baseline_result['rmse'] * 100
    r2_improve   = r2_joint - baseline_result['r2_joint']
    logger.info(f"MLP vs 线性基准: RMSE 提升 {rmse_improve:+.1f}%，R² 提升 {r2_improve:+.4f}")
    logger.info(sep)

    if rmse < baseline_result['rmse']:
        logger.info("✓ MLP 成功击败线性 WDF 基准！")
    else:
        logger.info("✗ MLP 未超过线性基准，建议检查特征或增大训练轮数。")
    logger.info(sep)


# ==============================================================================
# 绘图
# ==============================================================================
def plot_history(history: dict) -> None:
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss 曲线
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='tab:blue')
    axes[0].plot(epochs, history['val_loss'],   label='Val Loss',   color='tab:orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True)

    # R² 曲线
    axes[1].plot(epochs, history['val_r2'], color='tab:green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R²')
    axes[1].set_title('Validation R²')
    axes[1].grid(True)

    # 学习率曲线
    axes[2].semilogy(epochs, history['lr'], color='tab:red')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True)

    fig.tight_layout()
    out_path = os.path.join(RESULTS_DIR, 'mlp_training_curve.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n训练曲线已保存: {out_path}")


# ==============================================================================
# 主入口
# ==============================================================================
if __name__ == '__main__':
    sample = '--full' not in sys.argv
    mode_tag = "【采样模式 200 条轨迹】" if sample else "【完整数据集】"

    # 设置日志（含时间戳，避免文件名冲突）
    log_path = os.path.join(
        LOG_DIR, f"train_mlp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )
    logger = logging.getLogger('train_mlp')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

    logger.info(f"{'='*60}")
    logger.info(f"  WDF_DL_Param Phase 2 — MLP 训练  {mode_tag}")
    logger.info(f"{'='*60}")

    # 步骤 1: 加载数据
    splits = load_and_split_data(sample_mode=sample, sample_size=200)

    # 步骤 2: 线性基准（用于最终对比）
    logger.info("\n--- 运行线性基准 ---")
    baseline_result = run_linear_baseline(splits)

    # 步骤 3: MLP 训练
    logger.info("\n--- 开始 MLP 训练 ---")
    train_result = train(splits, logger)

    # 步骤 4: 评估 & 对比
    evaluate_and_compare(train_result, baseline_result, logger)

    # 步骤 5: 保存训练曲线
    plot_history(train_result['history'])

    logger.info(f"\n日志已保存: {log_path}")
    logger.info("=== 全流程完成 ===")
