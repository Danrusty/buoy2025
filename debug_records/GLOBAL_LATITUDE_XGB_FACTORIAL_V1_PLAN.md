# Global 纬度信息 × 模型类型析因实验 v1

## 研究问题

冻结 global core6 MLP 继续作为基准模型。本轮受控析因实验用于判断其轨迹改善
较弱主要与以下哪种原因有关：

1. 缺少大尺度空间信息；
2. MLP 模型类型的函数逼近能力有限；
3. 两项因素都有效，且存在非加性交互；
4. 两项因素均无效。

新增空间信号严格限定为 `sin(deg2rad(latitude))`。因此本轮是“纬度信息实验”，
不是完整空间编码实验。

## 四个受控实验格

| 实验格 | 模型 | 输入 | 分支 |
|---|---|---|---|
| A（冻结基准） | MLP | 冻结 core6 | `master@f2a0170` |
| B | MLP | core6 + `sin_latitude` | `wdf_global_mlp_lat7_v1` |
| C | XGBoost | 冻结 core6 | `wdf_global_xgb_core6_v1` |
| D | XGBoost | core6 + `sin_latitude` | `wdf_global_xgb_lat7_v1` |

core6 输入顺序固定为：

1. `era5_u10`
2. `era5_v10`
3. `era5_swh`
4. `era5_mwp`
5. `era5_wave_dir_sin`
6. `era5_wave_dir_cos`

B 和 D 仅在第 7 列追加 `sin_latitude`。

## 冻结控制项

- 数据集：
  `processed_data/trajectories_with_all_features_circular_mwd_v2.pkl`
- 数据集 SHA256：
  `22ab0a32ff9472a6f8b8f57af5fd96b93cdeb76d45b4ef6b0a798fa1befb937e`
- 精确继承的 `original_ID` 切分：
  `trained_models/ablation_circular_mwd_v2_final/core_6/split_manifest.json`
- 集合数量：train 1707 个 ID、validation 366 个 ID、test 366 个 ID。
- target 保持 `ve - cfsv2_u` 和 `vn - cfsv2_v`。
- 四格实验的样本成员、逐行顺序和 target 完全一致。
- 不引入经度、区域标签、新环境特征、target correction、序列模型、
  regional adapter 或分区域模型。
- 随机种子保持 42。

共享缓存只保存一份 core6，并单独保存 `sin_latitude`，避免复制整套 global
数组，同时可精确构造两种特征集合。缓存的受控溯源记录为
`trained_models/global_factorial_v1/data_manifest.json`；实际数组保存在已被
Git 忽略的 `processed_data/global_factorial_v1/`。

## 训练与模型选择

实验格 B 保持冻结 MLP 的网络结构、MSE loss、AdamW、scheduler、batch size、
最大 epoch、early stopping 规则和随机种子策略。唯一结构变化是首层输入由
6 维变为 7 维。

实验格 C 和 D 分别用两个独立 XGBoost regressor 预测 u、v target。小规模且
预先声明的候选组合只在 validation 集评价；配置锁定后才允许评价 test 集。
两个 XGBoost 实验使用完全相同的候选空间和选择规则。

固定公共参数为 `reg:squarederror`、`hist`、`device=cuda`、`max_bin=256`、
`subsample=0.8`、`colsample_bytree=1.0`、`seed=42`。最多训练 1000 轮，
validation RMSE 连续 50 轮不改善即 early stopping，并保留最佳迭代。

候选配置在访问 test 前预先冻结为：

| 名称 | grow policy | 深度/叶数 | eta | min child weight | L2 |
|---|---|---:|---:|---:|---:|
| `depth6_eta005` | depthwise | depth 6 | 0.05 | 128 | 10 |
| `depth8_eta003` | depthwise | depth 8 | 0.03 | 128 | 10 |
| `lossguide64_eta005` | lossguide | 64 leaves | 0.05 | 128 | 10 |
| `lossguide128_eta003` | lossguide | 128 leaves | 0.03 | 256 | 20 |

每个候选分别训练 u、v booster，再以两分量合并后的 validation RMSE 选择唯一
配置；相同时优先总树数较少者，其次按上表顺序。选择结果、模型 SHA256 和
validation 记录先写入不可回改的 `selection_lock.json`，随后才加载 test 特征
和 target。

同一时刻只运行一个高显存训练任务。不会争用训练资源的 CPU 数据准备、校验和
报告生成可以并行。

## 统一评价

四格实验均使用同一批冻结 test 行。主指标为逐行加权的 `R2_u`、`R2_v`、
joint R2（两分量 R2 的算术平均）、RMSE 和 MAE。另外统一报告：

- u、v residual bias；
- 每个 `original_ID` 等权的宏平均指标；
- 相对冻结实验格 A 的逐 ID RMSE 胜/平/负比例；
- 固定纬度带诊断。

析因效应定义为：

- MLP 下的纬度贡献：B − A；
- 不含纬度时的模型类型贡献：C − A；
- XGBoost 下的纬度贡献：D − C；
- 含纬度时的模型类型贡献：D − B；
- 交互项：D − C − B + A。

## XGBoost 效率评价门槛

只有 XGBoost 实验格相对 A 达到有意义的精度改善时，才进入目标环境推理效率
评价：

- joint R2 绝对提高至少 0.01；
- RMSE 相对降低至少 1%；
- macro-ID 指标也改善，排除收益只集中于长记录的情况。

通过门槛后，测试 batch size 1 和实际业务 batch，包含 warmup、p50/p95
latency、throughput、模型加载时间、模型体积、内存和显式线程数。是否最终部署
仍与本轮 global baseline 实验分开决策。

## 停止条件

四格对比完成后停止扩展 global 训练。是否进一步训练 regional adapter，应只以
本轮得到的最佳 global baseline 为起点，并另行决策。
