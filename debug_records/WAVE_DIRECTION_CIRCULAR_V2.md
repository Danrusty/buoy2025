# Wave Direction Circular v2

## 问题与修复

- v1 对 ERA5 `mean_wave_direction` 角度直接做空间/时间线性插值，再转
  `sin/cos`，跨越 0/360 度时不满足圆周变量的物理性质。
- v2 固定 ERA5 `coming-from`、北为 0 度、顺时针为正，不加 180 度。
- 原始 `mwd` 先编码为 `sin/cos`，分别插值后归一化，再写入训练特征。
- 海冰附近相邻时次的全 NaN 层会污染三维插值。最终实现改为选取精确
  ERA5 小时层后只做空间双线性插值；局部窗口全 NaN 时显式逐级扩展。

## 数据验收

- v2 数据集：`trajectories_with_all_features_circular_mwd_v2.pkl`
- SHA256：
  `22ab0a32ff9472a6f8b8f57af5fd96b93cdeb76d45b4ef6b0a798fa1befb937e`
- 4,141 条子轨迹、16,734,351 行、2,439 个 `original_ID`。
- 除 `era5_wave_dir_sin/cos` 外，索引、时间、ID、目标和其他特征与 v1
  逐项一致；train/val/test 的 `original_ID` 切分与 v1 完全一致。
- 近零向量事件为 0；最小插值向量模为 `0.00935445`，单位模最大误差为
  `2.22e-16`。
- 全量生成耗时约 1 小时 44 分，峰值 RSS 约 8.14 GiB，无 swap/OOM。

## 训练注意事项

- 不得让两个进程写入同一 `study-name/feature-set` 目录，否则
  `best_mlp.pth`、指标和训练历史可能互相覆盖。
- 正式结果使用严格串行重跑目录
  `ablation_circular_mwd_v2_final`；此前同名非 final 目录已废弃。
- core6 与 full9 的 validation 联合 R2 仅相差 `0.000174`，full9 在 test
  上略差，因此继续冻结更简单、已有 Windows 接口基础的 core6。

## 发布结论

- v2 core6 ONNX：
  `deployment/releases/wdf_core6_circular_mwd_v2/wdf_drifter.onnx`
- ONNX SHA256：
  `787d1d6a663677e30161a70493c70a7e46434414fb59085fbb68477939f18941`
- PyTorch/ONNX 最大绝对误差 `5.215e-08`，动态 batch 1/3/17 通过。
- Windows VS2022 + oneAPI ifx + ONNX Runtime 1.17.1 全链路通过，最大绝对
  误差 `1.4901e-08`。
