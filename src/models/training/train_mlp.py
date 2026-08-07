"""Stateless MLP 训练、评估与实验产物保存。"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境下保存图片
import matplotlib.pyplot as plt

from ..data_loader import DEFAULT_RUN_NAME, PROJECT_ROOT, load_and_split_data
from ..evaluation import regression_metrics
from .baseline import run_linear_baseline

# ==============================================================================
# 路径配置
# ==============================================================================
TRAINED_MODELS_DIR = PROJECT_ROOT / "trained_models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"
TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 超参数
# ==============================================================================
BATCH_SIZE    = 8192
EPOCHS        = 200
PATIENCE      = 20
LR            = 3e-4   # 大网络建议 3e-4；1e-4 对 400K 参数网络收敛太慢
LR_MIN        = 1e-6
RANDOM_SEED   = 42
MIN_DELTA     = 0.0

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
    大容量 MLP（~430K 参数），带 BatchNorm + Dropout。

    Input(input_size) -> [512->BN->ReLU->Drop] -> [512->BN->ReLU->Drop]
                      -> [256->BN->ReLU->Drop] -> [128->BN->ReLU] -> Linear(2)

    core6/full9 的参数量均约 430K，与 11M 级训练样本相匹配。
    Dropout(0.1) 防止大网络过拟合。
    """
    def __init__(self, input_size: int = 9, output_size: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_size),
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
        if not self.shuffle:
            for start in range(0, self.n, self.batch_size):
                end = start + self.batch_size
                yield self.X[start:end], self.y[start:end]
            return

        indices = torch.randperm(self.n, device=self.X.device)
        for start in range(0, self.n, self.batch_size):
            batch_indices = indices[start: start + self.batch_size]
            yield self.X[batch_indices], self.y[batch_indices]


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
    metrics = regression_metrics(targets, preds)
    return avg_loss, metrics["r2_joint"]


# ==============================================================================
# 训练主函数
# ==============================================================================
_logger = logging.getLogger(__name__)


def train(splits: dict) -> dict:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = _get_device()
    artifact_dir = Path(splits["artifact_dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    _logger.info(f"使用设备: {device}")
    if device.type == 'cuda':
        _logger.info(f"GPU: {torch.cuda.get_device_name(0)}, "
                    f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # 一次性将全部数据上传到 GPU 显存（消除逐 batch CPU→GPU 传输瓶颈）
    _logger.info("将训练/验证/测试数据转换为 %s 张量...", device)
    train_ds = GpuTensorDataset(splits['X_train'], splits['y_train'], device, BATCH_SIZE, shuffle=True)
    val_ds   = GpuTensorDataset(splits['X_val'],   splits['y_val'],   device, BATCH_SIZE, shuffle=False)
    test_ds  = GpuTensorDataset(splits['X_test'],  splits['y_test'],  device, BATCH_SIZE, shuffle=False)
    if device.type == 'cuda':
        _logger.info(f"显存占用（数据上传后）: {torch.cuda.memory_allocated()/1024**2:.0f} MB")

    # 初始化模型
    input_size = len(splits["feature_cols"])
    model = ResidualMLP(input_size=input_size).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    _logger.info(f"模型参数量: {parameter_count:,}")

    config_path = artifact_dir / "model_config.json"
    config = {
        "model_class": "ResidualMLP",
        "architecture": [input_size, 512, 512, 256, 128, 2],
        "batch_norm": True,
        "dropout": 0.1,
        "feature_columns": splits["feature_cols"],
        "target_columns": splits["target_cols"],
        "batch_size": BATCH_SIZE,
        "max_epochs": EPOCHS,
        "early_stopping_patience": PATIENCE,
        "checkpoint_monitor": "validation_loss",
        "learning_rate": LR,
        "minimum_learning_rate": LR_MIN,
        "optimizer": "AdamW",
        "weight_decay": 1e-4,
        "scheduler": "CosineAnnealingWarmRestarts(T_0=60,T_mult=2)",
        "random_seed": RANDOM_SEED,
    }
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _logger.info("模型配置已保存: %s", config_path)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    # CosineAnnealingWarmRestarts：
    #   T_0=60  → 第一个余弦周期 60 epoch（LR 从 3e-4 降到 1e-6）
    #   T_mult=2 → 第二个周期 120 epoch（总计 180 epoch 内完成两次重启）
    #   eta_min  → LR 下限，避免降到 0 导致梯度停滞（旧 CosineAnnealingLR 的 bug）
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=60, T_mult=2, eta_min=LR_MIN
    )

    best_val_loss = float("inf")
    best_val_r2_at_checkpoint = float("-inf")
    best_epoch = 0
    no_improve = 0
    best_model_path = artifact_dir / "best_mlp.pth"
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

        # CosineAnnealingWarmRestarts 按 epoch 步进（不需要监控 val_loss）
        scheduler.step(epoch)

        # checkpoint 与 early stopping 使用同一唯一标准，避免最佳指标和权重错位。
        improved = val_loss < (best_val_loss - MIN_DELTA)
        if improved:
            best_val_loss = val_loss
            best_val_r2_at_checkpoint = val_r2
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            _logger.info(
                "  改善（val_loss=%.6f, val_R²=%.4f），已保存最佳模型",
                best_val_loss,
                best_val_r2_at_checkpoint,
            )
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                _logger.info(f"\n早停触发（连续 {PATIENCE} epoch 无改善）")
                break

    _logger.info("--- 训练结束 ---")

    # 加载最佳权重进行最终评估
    model.load_state_dict(
        torch.load(best_model_path, map_location=device, weights_only=True)
    )
    _logger.info(
        "加载 epoch %d 的最佳权重: val_loss=%.6f, val_R²=%.4f",
        best_epoch,
        best_val_loss,
        best_val_r2_at_checkpoint,
    )

    return {
        'model':    model,
        'history':  history,
        'test_ds':  test_ds,
        'device':   device,
        'criterion': criterion,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'best_val_r2': best_val_r2_at_checkpoint,
        'best_model_path': best_model_path,
        'artifact_dir': artifact_dir,
        'parameter_count': parameter_count,
    }


# ==============================================================================
# 最终评估（测试集）及与线性基准对比
# ==============================================================================
def evaluate_and_compare(train_result: dict, baseline_result: dict) -> dict:
    model     = train_result['model']
    test_ds   = train_result['test_ds']
    # 收集完整预测结果用于细粒度指标
    model.eval()
    preds_list, targets_list = [], []
    with torch.no_grad():
        for X_b, y_b in test_ds:   # 数据已在 GPU
            preds_list.append(model(X_b).cpu().numpy())
            targets_list.append(y_b.cpu().numpy())

    preds   = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    test_metrics = regression_metrics(targets, preds)
    r2_u = test_metrics["r2_u"]
    r2_v = test_metrics["r2_v"]
    r2_joint = test_metrics["r2_joint"]
    rmse = test_metrics["rmse"]
    mae = test_metrics["mae"]

    sep = "=" * 60
    _logger.info(f"\n{sep}")
    _logger.info("  最终评估（测试集）及与线性基准对比")
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
    _logger.info(
        "MLP 相对线性基准: RMSE 提升 %+.1f%%，R² 提升 %+.4f",
        rmse_improve,
        r2_improve,
    )
    _logger.info(sep)

    if rmse < baseline_result['rmse']:
        _logger.info("[通过] MLP 的测试集 RMSE 低于线性 WDF 基准。")
    else:
        _logger.info("[未通过] MLP 的测试集 RMSE 未低于线性 WDF 基准。")
    _logger.info(sep)

    metrics = {
        "checkpoint": {
            "path": str(
                Path(train_result["best_model_path"]).relative_to(PROJECT_ROOT)
            ),
            "best_epoch": train_result["best_epoch"],
            "validation_loss": train_result["best_val_loss"],
            "validation_r2_joint": train_result["best_val_r2"],
            "parameter_count": train_result["parameter_count"],
        },
        "test": {
            "r2_u": r2_u,
            "r2_v": r2_v,
            "r2_joint": r2_joint,
            "rmse": rmse,
            "mae": mae,
        },
        "linear_baseline_test": {
            key: float(baseline_result[key])
            for key in ("r2_u", "r2_v", "r2_joint", "rmse", "mae")
        },
        "mlp_vs_linear": {
            "rmse_improvement_percent": rmse_improve,
            "r2_joint_difference": r2_improve,
        },
    }
    metrics_path = Path(train_result["artifact_dir"]) / "mlp_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _logger.info("MLP 指标已保存: %s", metrics_path)
    return metrics


# ==============================================================================
# 绘图
# ==============================================================================
def plot_history(
    history: dict,
    artifact_dir: Path,
    result_dir: Path | None = None,
) -> tuple[Path, Path]:
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 损失曲线
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
    result_dir = result_dir or RESULTS_DIR / artifact_dir.name
    result_dir.mkdir(parents=True, exist_ok=True)
    out_path = result_dir / "mlp_training_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    history_path = artifact_dir / "training_history.json"
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _logger.info("训练曲线已保存: %s", out_path)
    _logger.info("训练历史已保存: %s", history_path)
    return out_path, history_path


# ==============================================================================
# 主入口
# ==============================================================================
def _setup_logging(run_name: str) -> logging.Logger:
    """
    统一日志配置入口（仅在 train_mlp.py 作为主程序时调用）。
    配置根 logger，使 data_loader / baseline 的模块级 logger 自动继承，
    全程只写一个带时间戳的 log 文件。
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / (
        f"train_{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )
    # 长驻 IDE 进程可能保留旧 handler；先关闭，再为本次运行建立唯一日志。
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    logging.info(f"日志文件: {log_path}")
    return logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练 Stateless WDF MLP")
    parser.add_argument("--full", action="store_true", help="使用完整数据集")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    args = parser.parse_args()

    sample = not args.full
    mode_tag = (
        f"【采样模式 {args.sample_size} 个 original_ID】"
        if sample
        else "【完整数据集】"
    )
    artifact_dir = TRAINED_MODELS_DIR / args.run_name

    logger = _setup_logging(args.run_name)
    _logger.info(f"{'='*60}")
    _logger.info(f"  WDF_DL_Param 第二阶段：MLP 训练  {mode_tag}")
    _logger.info(f"{'='*60}")

    # 步骤 1: 加载数据
    splits = load_and_split_data(
        sample_mode=sample,
        sample_size=args.sample_size,
        artifact_dir=artifact_dir,
    )

    # 步骤 2: 线性基准（用于最终对比）
    _logger.info("\n--- 运行线性基准 ---")
    baseline_result = run_linear_baseline(splits)

    # 步骤 3: MLP 训练
    _logger.info("\n--- 开始 MLP 训练 ---")
    train_result = train(splits)

    # 步骤 4: 评估 & 对比
    evaluate_and_compare(train_result, baseline_result)

    # 步骤 5: 保存训练曲线
    plot_history(train_result['history'], artifact_dir)

    _logger.info("=== 全流程完成 ===")
