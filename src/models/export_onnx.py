# coding: utf-8
"""
export_onnx.py
==============
WDF_DL_Param 项目 Phase 3 - 任务 A

功能：
  1. 加载已训练的 PyTorch MLP 模型 (best_mlp.pth)。
  2. 加载用于输入的 scikit-learn StandardScaler (x_scaler.pkl)。
  3. 创建一个包含 "数据标准化" 和 "MLP 推理" 的端到端部署模型。
     - 将 StandardScaler 的均值 (mean) 和标准差 (scale) "烘焙" 为模型内部的
       torch.Tensor 常量，消除对 scikit-learn 的运行时依赖。
  4. 将该部署模型导出为 ONNX 格式 (wdf_drifter.onnx)，并支持动态 batch size。

该脚本为 Fortran/C++ 业务化部署提供模型中间表示（ONNX）。

运行方式：
  cd src/models
  python export_onnx.py
"""

import os
import joblib
import torch
import torch.nn as nn

# ==============================================================================
# 1. 路径配置 (与 train_mlp.py 保持一致)
# ==============================================================================
# 假设脚本在 src/models/ 目录下运行
TRAINED_MODELS_DIR = '../../trained_models'
ONNX_OUTPUT_DIR = '../../trained_models'  # ONNX 模型也存放在此
os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, 'best_mlp.pth')
SCALER_PATH = os.path.join(TRAINED_MODELS_DIR, 'x_scaler.pkl')
ONNX_PATH = os.path.join(ONNX_OUTPUT_DIR, 'wdf_drifter.onnx')


# ==============================================================================
# 2. 模型定义 (必须与训练时完全一致)
# ==============================================================================
class ResidualMLP(nn.Module):
    """
    该定义拷贝自 train_mlp.py，用于加载 best_mlp.pth 的 state_dict。
    """
    def __init__(self, input_size: int = 9, output_size: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        # 在导出 ONNX 时，BatchNorm 和 Dropout 的行为与训练时不同。
        # .eval() 模式会自动处理，无需手动修改。
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
# 3. 创建包含预处理的部署模型 (关键步骤)
# ==============================================================================
class DeploymentNet(nn.Module):
    """
    一个包装模型，集成了 "标准化预处理" 和 "MLP推理"。
    这是最终要导出为 ONNX 的模型。
    """
    def __init__(self, trained_mlp: nn.Module, scaler_mean: torch.Tensor, scaler_scale: torch.Tensor):
        """
        参数:
          - trained_mlp: 已经加载了权重和偏差的原始 MLP 模型实例。
          - scaler_mean: 标准化器的均值 (1D Tensor, shape [num_features])。
          - scaler_scale: 标准化器的标准差 (1D Tensor, shape [num_features])。
        """
        super().__init__()
        self.mlp = trained_mlp

        # 使用 register_buffer 将 scaler 的参数注册为模型的一部分。
        # 这样做有几个好处：
        #   1. 这些张量会被自动移动到正确的设备 (CPU/GPU)。
        #   2. 它们会被包含在模型的 state_dict 中 (虽然这里我们只用于导出)。
        #   3. 最重要的是，它们成为 ONNX 计算图的一部分，无需外部传入。
        #   4. 我们使用 unsqueeze(0) 将其 shape 从 (9,) 变为 (1, 9) 以便广播
        self.register_buffer('scaler_mean', scaler_mean.unsqueeze(0))
        self.register_buffer('scaler_scale', scaler_scale.unsqueeze(0))

    def forward(self, x_physical: torch.Tensor) -> torch.Tensor:
        """
        模型的前向传播逻辑。
        接收原始物理量，返回预测的残差。

        参数:
          - x_physical: 原始物理量输入 (未标准化), shape [batch_size, 9]

        返回:
          - residual_uv: 预测的残差, shape [batch_size, 2]
        """
        # 步骤 1: "烘焙"在模型内部的标准化
        # (x - mean) / scale
        x_scaled = (x_physical - self.scaler_mean) / self.scaler_scale

        # 步骤 2: 将标准化后的数据传入原 MLP 模型
        residual_uv = self.mlp(x_scaled)

        return residual_uv


# ==============================================================================
# 4. 主执行函数
# ==============================================================================
def main():
    """
    执行模型加载、包装和 ONNX 导出的主流程。
    """
    print("WDF_DL_Param ONNX 导出脚本")
    print("-" * 40)

    # --- 步骤 1: 加载原始 MLP 模型 ---
    print(f"  [1] 加载 PyTorch MLP 模型结构...")
    # 注意：我们仅初始化结构，权重将在下一步加载
    original_mlp = ResidualMLP(input_size=9, output_size=2)

    print(f"  [2] 从 '{os.path.basename(MODEL_PATH)}' 加载已训练的权重...")
    # 加载状态字典。map_location='cpu' 确保即使模型在 GPU 上训练，也能在无 GPU 环境下加载。
    original_mlp.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    # **极其重要**: 必须将模型切换到评估模式 (.eval())
    # 这会固定 BatchNorm 和 Dropout 的行为，使其在推理时保持确定性。
    original_mlp.eval()
    print("      - 模型已切换到 .eval() 模式。")

    # --- 步骤 2: 加载并转换 Scaler ---
    print(f"  [3] 从 '{os.path.basename(SCALER_PATH)}' 加载 StandardScaler...")
    scaler = joblib.load(SCALER_PATH)

    # 将 scaler 的 mean_ 和 scale_ (numpy 数组) 转换为 torch.Tensor
    scaler_mean_tensor = torch.tensor(scaler.mean_, dtype=torch.float32)
    scaler_scale_tensor = torch.tensor(scaler.scale_, dtype=torch.float32)
    print("      - 已将 Scaler 均值和标准差转换为 Torch Tensor。")
    print(f"      - 特征均值 (前3个): {scaler.mean_[:3]}")
    print(f"      - 特征标准差 (前3个): {scaler.scale_[:3]}")

    # --- 步骤 3: 创建并准备部署模型 ---
    print("  [4] 创建包含预处理的 DeploymentNet...")
    deployment_model = DeploymentNet(
        trained_mlp=original_mlp,
        scaler_mean=scaler_mean_tensor,
        scaler_scale=scaler_scale_tensor
    )
    deployment_model.eval() # 同样设置为评估模式

    # --- 步骤 4: 导出为 ONNX ---
    print(f"  [5] 导出模型到 '{os.path.basename(ONNX_PATH)}'...")

    # 创建一个符合输入尺寸的 "虚拟" 输入张量。
    # batch_size=1 是一个占位符，因为我们下面会将其设为动态。
    # 尺寸必须是 (batch_size, num_features)，即 (1, 9)。
    dummy_input = torch.randn(1, 9)

    # 定义动态维度。这允许 ONNX 模型接受不同大小的 batch。
    # 'batch_size' 是我们给这个动态维度起的名字，可以是任意字符串。
    dynamic_axes = {
        'input': {0: 'batch_size'},   # key 'input' 对应下面的 input_names
        'output': {0: 'batch_size'}  # key 'output' 对应下面的 output_names
    }

    try:
        torch.onnx.export(
            deployment_model,        # 要导出的模型实例
            dummy_input,             # 一个示例输入
            ONNX_PATH,               # 输出文件路径
            export_params=True,      # 导出训练好的参数
            opset_version=12,        # ONNX 算子集版本，12 是一个稳定且广泛支持的版本
            do_constant_folding=True,# 执行常量折叠优化
            input_names=['input'],   # 指定输入节点的名称
            output_names=['output'], # 指定输出节点的名称
            dynamic_axes=dynamic_axes # 指定动态维度
        )
        print("\n  ✓ ONNX 模型导出成功!")
        print(f"    - 输入节点名: 'input'")
        print(f"    - 输出节点名: 'output'")
        print(f"    - 动态 batch: 是")
        print(f"    - 模型已保存到: {ONNX_PATH}")

    except Exception as e:
        print(f"\n  ✗ ONNX 导出失败: {e}")

    print("-" * 40)


def verify():
    """
    对比 PyTorch DeploymentNet 输出与 ONNX Runtime 输出的数值一致性。
    使用固定种子生成 100 组随机物理量输入，确认 max |diff| < 1e-5。
    同时打印测试样本的预测值，供 Fortran 端到端验证时参考。
    """
    import numpy as np
    import onnxruntime as ort

    print("\n" + "=" * 40)
    print("  ONNX 数值一致性验证")
    print("=" * 40)

    # --- 加载 PyTorch 部署模型 ---
    original_mlp = ResidualMLP(input_size=9, output_size=2)
    original_mlp.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    original_mlp.eval()

    scaler = joblib.load(SCALER_PATH)
    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32)

    deploy_model = DeploymentNet(original_mlp, scaler_mean, scaler_scale)
    deploy_model.eval()

    # --- 生成测试数据（物理量级别的随机输入）---
    np.random.seed(42)
    # 9 个特征的典型范围（基于训练集统计）
    test_input_np = np.random.randn(100, 9).astype(np.float32) * 5.0

    # --- PyTorch 推理 ---
    with torch.no_grad():
        pt_output = deploy_model(torch.from_numpy(test_input_np)).numpy()

    # --- ONNX Runtime 推理 ---
    sess = ort.InferenceSession(ONNX_PATH)
    ort_output = sess.run(['output'], {'input': test_input_np})[0]

    # --- 对比 ---
    diff = np.abs(pt_output - ort_output)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"  测试样本数: {len(test_input_np)}")
    print(f"  最大绝对误差: {max_diff:.2e}")
    print(f"  平均绝对误差: {mean_diff:.2e}")

    if max_diff < 1e-5:
        print("  ✓ 验证通过：PyTorch 与 ONNX Runtime 输出一致")
    else:
        print(f"  ✗ 验证失败：max_diff={max_diff:.6f} > 1e-5")

    # --- 打印 3 组参考值（供 Fortran 端对比）---
    print("\n  参考测试向量（Fortran 端验证用）:")
    print(f"  {'粒子':>4s}  {'输入 (前5个特征)':>40s}  {'pred_u':>10s}  {'pred_v':>10s}")
    for i in range(3):
        feat_str = ", ".join(f"{v:+.4f}" for v in test_input_np[i, :5])
        print(f"  #{i+1:>3d}  [{feat_str}, ...]  {ort_output[i,0]:+.6f}  {ort_output[i,1]:+.6f}")

    print("=" * 40)
    return max_diff < 1e-5


if __name__ == '__main__':
    main()
    verify()
