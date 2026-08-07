# wdf_cms_network_redesign_v1 Windows Handoff

## 状态

- 结构：`plain_64_32`，参数量 2,594
- 科学状态：已冻结的 CMS 网络重构实验
- 部署状态：Windows staging 候选，禁止自动激活
- 权威 ONNX：`wdf_cms_orig_core6_v1.onnx`
- Fortran 兼容别名：`wdf_drifter.onnx`
- Windows staging：`D:\OilspillModel\OilSpillModel\ModelRun\release_onnx\wdf_cms_network_redesign_v1`
- 激活建议：`do_not_activate`

本 release 不覆盖旧 `wdf_cms_orig_core6_v1` release，也不修改
`onnx_active`。两个 ONNX 文件字节级相同；兼容别名仅用于保持既有
Fortran 验证程序不变。

## 冻结接口

- 输入：`input`, float32, `(batch_size, 6)`
- 输出：`output`, float32, `(batch_size, 2)`
- StandardScaler 已烘焙进 ONNX，Fortran 端不得再次标准化
- 输入顺序：
  `era5_u10, era5_v10, era5_swh, era5_mwp,`
  `era5_wave_dir_sin, era5_wave_dir_cos`
- target：
  `residual_u = ve - cfsv2_u`,
  `residual_v = vn - cfsv2_v`
- opset：12
- Python PyTorch/ONNX 最大绝对误差：
  1.490e-08

## 固定 test 结果

- 新 regional 网络：joint R² 0.047973，
  RMSE 0.368909 m/s
- 旧 CMS MLP：joint R²
  -0.066276
- 区域线性基准：joint R²
  0.011372
- 冻结 global MLP（同一批行）：joint R²
  0.134824，RMSE 0.352383 m/s

新结构改善了旧 CMS MLP 和区域线性基准，但仍弱于冻结 global MLP。
因此该模型仅作为可复现实验交付，不应替换当前运行模型。

## 数据限制

- CMS：23 个 original_ID，21,074 行
- train/val/test：15/4/4 个 original_ID，交集为 0
- BYS/ECS/NSCS 行数：0/13,956/7,118
- 当前源数据没有 BYS 样本，不能声称具备渤海—黄海泛化能力

## Windows 验证

1. 保持 release 文件完整，不与旧版本混用。
2. 在 staging 目录运行 `verify_windows.bat`。
3. 要求所有输出与 `expected_output.csv` 的绝对误差 `< 1e-4`。
4. 记录 `SHA256SUMS.txt` 中权威 ONNX 的 SHA256。
5. 不修改 release root 或 `onnx_active`。
