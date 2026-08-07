# 模型代码目录

当前维护中的 global 模型代码只采用一层浅目录：

- `data_loader.py`、`evaluation.py`：共用数据与指标契约；
- `training/`：global baseline、MLP 训练、消融、ONNX 导出，以及纬度信息 ×
  模型类型析因实验的共用协议；
- `training/global_mlp_spatial.py`：lat7/lat9 共用的 MLP 训练与冻结协议；
- `training/global_longitude.py`：在冻结行顺序上追加经度循环编码缓存；
- `training/run_global_mlp_lat7.py`、`training/run_global_mlp_lat9.py`：
  两个空间信息 MLP 实验入口；
- `deployment/`：C++/Fortran wrapper 和 Windows 验证脚本；
- `legacy/`：已归档的早期 MLP/RNN 探索脚本。

所有入口均从仓库根目录以模块方式运行，例如：

```bash
conda run -n buoy-drifter python -m src.models.training.run_ablation --help
```

测试统一保存在仓库顶层 `tests/`。
