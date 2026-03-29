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
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境下保存图片
import matplotlib.pyplot as plt

from data_loader import load_and_split_data
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
BATCH_SIZE    = 8192
EPOCHS        = 200    # 给模型足够的收敛空间
PATIENCE      = 20     # 曲线仍在上升时 5 太激进，改为 20
LR            = 1e-3
LR_FACTOR     = 0.5    # ReduceLROnPlateau 衰减因子
LR_PATIENCE   = 8      # 避免 LR 过早衰减（之前 3 导致 epoch17 就腰斩）
LR_MIN        = 1e-6   # 允许更低 LR 继续精细收敛
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
# GPU 数据集：一次性将数据移到显存，消除每 batch 的 CPU→GPU 传输瓶颈
# ==============================================================================
class GpuTensorDataset:
    """
    将 numpy 数组一次性上传到指定 device（显存），
    每次迭代直接在 GPU 上切片，避免 DataLoader 逐 batch 传输的 CPU 瓶颈。
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 device: torch.device, batch_size: int, shuffle: bool):
        self.X = torch.from_numpy(X).float().to(device)
        self.y = torch.from_numpy(y).float().to(device)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.n          = len(X)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = torch.randperm(self.n, device=self.X.device) if self.shuffle \
              else torch.arange(self.n, device=self.X.device)
        for start in range(0, self.n, self.batch_size):
            batch_idx = idx[start: start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]


@torch.no_grad()
def _evaluate(model: nn.Module, dataset: GpuTensorDataset,
              criterion: nn.Module) -> tuple[float, float]:
    """返回 (val_loss_mse, val_r2_joint)。数据已在 GPU，无需 .to(device)。"""
    model.eval()
    preds_list, targets_list = [], []
    total_loss, total_samples = 0.0, 0

    for X_b, y_b in dataset:
        pred = model(X_b)
        loss = criterion(pred, y_b)
        total_loss    += loss.item() * len(X_b)
        total_samples += len(X_b)
        preds_list.append(pred.cpu().numpy())
        targets_list.append(y_b.cpu().numpy())

    preds    = np.concatenate(preds_list)
    targets  = np.concatenate(targets_list)
    avg_loss = total_loss / total_samples
    r2       = r2_score(targets, preds)
    return avg_loss, r2


# ==============================================================================
# 训练主函数
# ==============================================================================
_logger = logging.getLogger(__name__)


def train(splits: dict) -> dict:
    device = _get_device()
    _logger.info(f"使用设备: {device}")
    if device.type == 'cuda':
        _logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                    f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # 一次性将全部数据上传到 GPU 显存（消除逐 batch CPU→GPU 传输瓶颈）
    _logger.info("将训练/验证/测试数据上传到 GPU 显存...")
    train_ds = GpuTensorDataset(splits['X_train'], splits['y_train'], device, BATCH_SIZE, shuffle=True)
    val_ds   = GpuTensorDataset(splits['X_val'],   splits['y_val'],   device, BATCH_SIZE, shuffle=False)
    test_ds  = GpuTensorDataset(splits['X_test'],  splits['y_test'],  device, BATCH_SIZE, shuffle=False)
    if device.type == 'cuda':
        _logger.info(f"显存占用（数据上传后）: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    # 初始化模型
    model = ResidualMLP().to(device)
    _logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR,
        patience=LR_PATIENCE, min_lr=LR_MIN, verbose=False
    )

    best_val_loss  = float('inf')
    best_val_r2    = float('-inf')
    no_improve     = 0
    best_model_path = os.path.join(TRAINED_MODELS_DIR, 'best_mlp.pth')
    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'lr': []}

    _logger.info(f"\n{'='*60}")
    _logger.info("  开始训练  (max_epochs={}, patience={}, batch={})".format(
        EPOCHS, PATIENCE, BATCH_SIZE))
    _logger.info(f"{'='*60}")

    for epoch in range(1, EPOCHS + 1):
        # --- 训练阶段 ---
        model.train()
        running_loss, n_samples = 0.0, 0
        for X_b, y_b in train_ds:   # 数据已在 GPU，直接用
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(X_b)
            n_samples    += len(X_b)

        train_loss = running_loss / n_samples

        # --- 验证阶段 ---
        val_loss, val_r2 = _evaluate(model, val_ds, criterion)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['lr'].append(current_lr)

        _logger.info(
            f"Epoch [{epoch:03d}/{EPOCHS}] | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"val_R²={val_r2:.4f} | "
            f"lr={current_lr:.2e}"
        )

        # 学习率调度（监控 val_loss）
        scheduler.step(val_loss)

        # 早停：同时监控 val_loss 和 val_r2，任一改善即重置计数器
        # （val_r2 更能反映真实泛化能力，对 loss 的微小数值波动更鲁棒）
        improved = (val_loss < best_val_loss) or (val_r2 > best_val_r2)
        if improved:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            _logger.info(f"  ✓ 改善（val_loss={best_val_loss:.6f}, val_R²={best_val_r2:.4f}），已保存最佳模型")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                _logger.info(f"\n早停触发（连续 {PATIENCE} epoch 无改善）")
                break

    _logger.info("--- 训练结束 ---")

    # 加载最佳权重进行最终评估
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    return {
        'model':    model,
        'history':  history,
        'test_ds':  test_ds,
        'device':   device,
        'criterion': criterion,
    }


# ==============================================================================
# 最终评估（Test 集）& 与线性基准对比
# ==============================================================================
def evaluate_and_compare(train_result: dict, baseline_result: dict) -> None:
    model     = train_result['model']
    test_ds   = train_result['test_ds']
    criterion = train_result['criterion']

    # 收集完整预测结果用于细粒度指标
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for X_b, y_b in test_ds:   # 数据已在 GPU
            preds_list.append(model(X_b).cpu().numpy())
            targets_list.append(y_b.cpu().numpy())

    test_loss = float(np.mean((np.concatenate(preds_list) - np.concatenate(targets_list)) ** 2))

    preds   = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    r2_u    = r2_score(targets[:, 0], preds[:, 0])
    r2_v    = r2_score(targets[:, 1], preds[:, 1])
    r2_joint = r2_score(targets, preds)
    rmse    = float(np.sqrt(np.mean((preds - targets) ** 2)))
    mae     = float(np.mean(np.abs(preds - targets)))

    sep = "=" * 60
    _logger.info(f"\n{sep}")
    _logger.info("  最终评估（Test 集）& 与线性基准对比")
    _logger.info(sep)
    _logger.info(f"\n{'指标':<20} {'线性回归 (WDF)':>18} {'MLP':>14}")
    _logger.info("-" * 55)
    _logger.info(f"{'R² (residual_u)':<20} {baseline_result['r2_u']:>18.4f} {r2_u:>14.4f}")
    _logger.info(f"{'R² (residual_v)':<20} {baseline_result['r2_v']:>18.4f} {r2_v:>14.4f}")
    _logger.info(f"{'R² (联合)':<20} {baseline_result['r2_joint']:>18.4f} {r2_joint:>14.4f}")
    _logger.info(f"{'RMSE (m/s)':<20} {baseline_result['rmse']:>18.4f} {rmse:>14.4f}")
    _logger.info(f"{'MAE (m/s)':<20} {baseline_result['mae']:>18.4f} {mae:>14.4f}")
    _logger.info("-" * 55)

    rmse_improve = (baseline_result['rmse'] - rmse) / baseline_result['rmse'] * 100
    r2_improve   = r2_joint - baseline_result['r2_joint']
    _logger.info(f"MLP vs 线性基准: RMSE 提升 {rmse_improve:+.1f}%，R² 提升 {r2_improve:+.4f}")
    _logger.info(sep)

    if rmse < baseline_result['rmse']:
        _logger.info("✓ MLP 成功击败线性 WDF 基准！")
    else:
        _logger.info("✗ MLP 未超过线性基准，建议检查特征或增大训练轮数。")
    _logger.info(sep)


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
def _setup_logging() -> logging.Logger:
    """
    统一日志配置入口（仅在 train_mlp.py 作为主程序时调用）。
    配置根 logger，使 data_loader / baseline 的模块级 logger 自动继承，
    全程只写一个带时间戳的 log 文件。
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )
    # 避免重复添加（防止 IDE 等环境多次调用）
    if not root.handlers:
        fh = logging.FileHandler(log_path, encoding='utf-8')
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(sh)
    logging.info(f"日志文件: {log_path}")
    return logging.getLogger(__name__)


if __name__ == '__main__':
    sample = '--full' not in sys.argv
    mode_tag = "【采样模式 200 条轨迹】" if sample else "【完整数据集】"

    logger = _setup_logging()
    _logger.info(f"{'='*60}")
    _logger.info(f"  WDF_DL_Param Phase 2 — MLP 训练  {mode_tag}")
    _logger.info(f"{'='*60}")

    # 步骤 1: 加载数据
    splits = load_and_split_data(sample_mode=sample, sample_size=200)

    # 步骤 2: 线性基准（用于最终对比）
    _logger.info("\n--- 运行线性基准 ---")
    baseline_result = run_linear_baseline(splits)

    # 步骤 3: MLP 训练
    _logger.info("\n--- 开始 MLP 训练 ---")
    train_result = train(splits)

    # 步骤 4: 评估 & 对比
    evaluate_and_compare(train_result, baseline_result)

    # 步骤 5: 保存训练曲线
    plot_history(train_result['history'])

    _logger.info("=== 全流程完成 ===")
