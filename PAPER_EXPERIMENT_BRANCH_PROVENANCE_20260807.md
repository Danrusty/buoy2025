# 论文诊断实验分支与提交来源

更新时间：2026-08-10

## 1. 记录目的

本文档冻结论文成文阶段所用模型诊断实验的 Git 来源。这些分支是从同一工程基线
分叉的互斥实验谱系，不是等待全部合并到 `master` 的功能分支。论文引用结果时
必须同时记录实验角色、准确提交、evidence tag、切分或区域、gate 状态、产物位置
和可比性。

## 2. 工程基线

- 冻结部署模型谱系：`master@f2a01709c8cac05f580341df70b477a633614aae`
- 当前 `master`：在冻结部署谱系上增加论文 provenance、共用评价
  与一项隔离的 wind-only 严格消融，不改变 active model
- 冻结归档 tag：`archive/wdf-core6-circular-mwd-v2`
- 当前工程部署主模型：`wdf_core6_circular_mwd_v2`
- 部署状态：Python、C++、Fortran 和 Windows 数值验证通过
- 研究状态：工程接口继续冻结；后续模型只作为论文诊断证据，未替换 Windows
  Fortran 溢油模拟中的 global core6

`master` 维护稳定工程基线、部署模型、论文 provenance 和共用评价。
2026-08-10 根据论文章节衔接需求，`master` 新增 wind-only 严格消融作为单一
论文诊断例外；它使用独立产物目录，并明确冻结为
`active_deployment_changed=false`。其他互斥训练代码、失败 release 和实验
模型仍不整体合并回主线。

## 3. CMS 实验谱系

| 角色 | 准确提交 | Evidence tag | 当前状态 |
|---|---|---|---|
| CMS 原始 core6 冻结基准 | `68b1c32` | `paper/cms-core6-baseline-v1` | frozen baseline |
| CMS frozen-global adapter 选择 | `422a4db` | `paper/cms-adapter-v1` | failed gate |
| 15–45 N、105–170 E adapter 选择 | `68473f4` | `paper/cms-adapter-170e-v1` | failed gate |
| CMS umbrella 汇总 | `3365743` | `paper/cms-umbrella-v1` | 远端分支保留 |
| CMS 网络重设计 | `5401c2d` | `paper/cms-network-redesign-v1` | tag 归档，分支已删除 |
| CMS 范围搜索 | `2c7111d` | `paper/cms-range-search-v1` | tag 归档，分支已删除 |

准确拓扑为：

```text
master f2a0170
└─ b9bcfc2 -> 68b1c32  CMS 原始 core6
   ├─ 3f8edb0 -> 5f752d4 -> a194e70 -> 3a60076 -> cfacef3 -> 5401c2d
   │  CMS 网络重设计
   └─ b207503 -> 422a4db  CMS adapter
      ├─ 476474a -> 0ad58d8 -> 3cddb3f -> 2c7111d
      │  范围搜索与 170 E 决策
      └─ 63003b5 -> fa2c208 -> 68473f4 -> 3365743
         170 E adapter、结果冻结与 umbrella 整理
```

范围决策 `2c7111d` 与 170 E adapter 实现 `63003b5` 是语义上的前后关系，
但 `63003b5` 的 Git 父提交是 `422a4db`，不是 `2c7111d`。不得通过 merge、
rebase 或 cherry-pick 把这两条历史伪造成祖先关系；论文 provenance 采用
commit 和 tag 交叉引用。

### 3.1 CMS 原始网络与网络重设计

CMS 严格区域筛选只得到 23 个 `original_ID`、21,074 行；BYS 为零样本。
CMS 原始 core6 MLP 在 4-ID test 上的结果为：

| 模型 | joint R² | RMSE (m/s) |
|---|---:|---:|
| CMS 原始 core6 MLP | -0.066276 | 0.390745 |
| CMS 区域线性基准 | 0.011372 | 0.376188 |
| Frozen global MLP，同一批 CMS test 行 | 0.134824 | 0.352383 |

网络重设计从 validation-locked 候选中选择 `plain_64_32`，参数量从约 433k
降至 2,594。它在同一批 CMS test 行上达到 joint R² `0.047973`、RMSE
`0.368909 m/s`，优于 CMS 原始网络和区域线性基准，但仍显著差于 frozen
global。

主要证据位置：

- `paper/cms-umbrella-v1:debug_records/CMS_REGIONAL_CORE6_V1.md`
- `paper/cms-network-redesign-v1:results/wdf_cms_core6_network_redesign_v1/test_evaluation.json`
- `paper/cms-network-redesign-v1:deployment/releases/wdf_cms_network_redesign_v1/WINDOWS_VALIDATION.md`

网络重设计 release 只证明实验模型可复现和工程接口可运行，不代表它应替换
当前部署模型。

### 3.2 两轮 frozen-global adapter

原始 CMS adapter 严格继承 global lineage：19 个 development ID、2 个 gate
ID、2 个 sealed confirmation ID。五个 nested outer folds和全 development
选择均得到 `G0_global_only`。最近的非零候选 `G1_bias2, lambda=10` 仍使
macro-ID RMSE 退化 `0.0328%`，只改善 `42.1%` 的 ID，因此在 confirmation
之前停止。

范围搜索是 count-only 决策，不读取 target、不拟合模型：

| 东界 | 总 ID | Train | Validation | Test | 行数 |
|---:|---:|---:|---:|---:|---:|
| 140 E | 53 | 41 | 9 | 3 | 101,127 |
| 150 E | 74 | 59 | 11 | 4 | 238,036 |
| 160 E | 87 | 69 | 11 | 7 | 367,649 |
| 170 E | 96 | 75 | 12 | 9 | 454,892 |
| 180 E | 172 | 131 | 18 | 23 | 526,212 |

原门槛下 180 E 首次通过；用户在查看计数且尚未评价 target/model 前，明确接受
170 E 的 `96 / 75 / 12 / 9`，因此第二轮 adapter 使用
`15–45 N, 105–170 E`。

170 E adapter 的四个 outer folds选择 `G0_global_only`，一个选择强正则
`G3_wind_full6`；全 development 仍只有 G0 位于 one-standard-error 区间。
最近的非零候选 `G4_global_calibration6, lambda=10` 仅把逐行 RMSE 从
`0.270941` 降至 `0.270933 m/s`，同时使 macro-ID RMSE 退化 `0.0033%`。
gate 因而失败，9 个 confirmation ID 保持 sealed。

主要证据位置：

- `paper/cms-adapter-v1:trained_models/wdf_cms_global_adapter_v1/development_cv.json`
- `paper/cms-adapter-v1:trained_models/wdf_cms_global_adapter_v1/gate_evaluation.json`
- `paper/cms-range-search-v1:results/wdf_cms_extent_search_v1/range_search.json`
- `paper/cms-range-search-v1:results/wdf_cms_extent_search_v1/range_selection_override.json`
- `paper/cms-adapter-170e-v1:trained_models/wdf_eastasia_wnp_global_adapter_v1/development_cv.json`
- `paper/cms-adapter-170e-v1:trained_models/wdf_eastasia_wnp_global_adapter_v1/gate_evaluation.json`

两轮 adapter 都没有生成可接受的新 ONNX，也没有更改 active model。

## 4. Global 空间信息与模型类型诊断

### 4.1 实验来源

| 角色 | 准确提交 | Evidence tag | 当前状态 |
|---|---|---|---|
| MLP lat7 + lat9 汇总 | `21f8c85` | `paper/global-mlp-lat7+lat9` | 远端分支保留 |
| XGBoost core6 | `2d77514` | `paper/global-xgb-core6-v1` | 远端分支保留 |
| XGBoost lat7 与析因汇总 | `fb972f5` | `paper/global-xgb-lat7-v1` | 远端分支保留 |

`wdf_global_mlp_lat7_v1` 的历史分支名保留不变，但 tip 同时包含 lat7、lat9
及五模型汇总。论文和 tag 使用更准确的 `global-mlp-lat7+lat9` 名称。已提交
产物内部目录 `results/global_mlp_spatial_v1` 保持不改名，以免改写实验记录。

本研究没有训练 XGBoost lat9。五个实际 global 单元是：

| 格 | 模型/输入 | test joint R² | RMSE (m/s) |
|---|---|---:|---:|
| A | Frozen MLP core6 | 0.127398 | 0.194682 |
| B | MLP core6 + `sin(lat)` | 0.130840 | 0.194291 |
| E | MLP core6 + `sin(lat)` + `sin/cos(lon)` | 0.130127 | 0.194376 |
| C | XGBoost core6 | 0.127254 | 0.194698 |
| D | XGBoost core6 + `sin(lat)` | 0.131584 | 0.194207 |

MLP 下纬度带来 joint R² `+0.003442`；单独替换 XGBoost 为
`-0.000144`；XGBoost 下加入纬度为 `+0.004330`。lat9 相对 lat7 的 test
joint R² 为 `-0.000713`，虽然 validation 提高，但没有形成独立 test 增益。
数值最优的 D 相对 A 也只有 `+0.004186` joint R² 和 `0.244%` RMSE 改善，
未达到进入 Windows 推理效率评价的预注册门槛。

主要证据位置：

- `paper/global-mlp-lat7+lat9:results/global_mlp_spatial_v1/README.md`
- `paper/global-mlp-lat7+lat9:results/global_mlp_spatial_v1/comparison.json`
- `paper/global-xgb-lat7-v1:results/global_factorial_v1/README.md`
- `paper/global-xgb-lat7-v1:results/global_factorial_v1/comparison.json`

### 4.2 Wind-only 波浪输入严格消融

| 角色 | 准确提交 | Evidence tag | 当前状态 |
|---|---|---|---|
| 实验代码 | `0fa4030b0515079c2c4275c898cb6b3f15bc8820` | 由结果 tag 交叉引用 | frozen training code |
| 正式结果 | `dca1fd55a68b3aacab090eae790f13ca06647e3a` | `paper/wind-only-ablation-v1` | frozen diagnostic |

wind-only MLP 只输入 `era5_u10/era5_v10`；对照 core6 输入风速两分量及
`era5_swh`、`era5_mwp`、`era5_wave_dir_sin/cos`。wind-only 仍使用 core6
六特征有效行掩码。清单校验确认两者的 `original_ID`、子轨迹、逐行样本、
target 和 seed 全部一致：train/validation/test 分别为
`11,694,073 / 2,614,863 / 2,425,415` 行和 `1707 / 366 / 366` 个 ID。
共同线性基准重放的最大绝对差为零。

| 模型 | test joint R² | RMSE (m/s) | MAE (m/s) |
|---|---:|---:|---:|
| 共同线性 WDF | 0.122317 | 0.195250 | 0.143718 |
| Wind-only MLP | 0.123691 | 0.195094 | 0.143574 |
| Wind+wave core6 MLP | 0.127398 | 0.194682 | 0.143305 |

core6 相对 wind-only 的 joint R² 增加 `0.003707`，RMSE 降低
`0.000412 m/s`（`0.211%`）。该结果的绝对幅度较小，表明风场承担了
水平残差速度的主要可预测信息。以共同线性 WDF 为参照，core6 的
joint R² 总增量为 `0.005081`，wind-only 的非线性增量为 `0.001374`；
加入波浪后的额外增量占 core6 相对线性基准总差值的 `73.0%`。

同一 held-out test 上的水平位移误差代理为：

| 时长 | Wind-only (km) | Core6 (km) | Core6 改善率 | 配对 95% CI |
|---:|---:|---:|---:|---:|
| 6 h | 4.365 | 4.355 | 0.229% | [-0.084%, 0.543%] |
| 12 h | 7.976 | 7.963 | 0.159% | [-0.231%, 0.531%] |
| 24 h | 14.761 | 14.723 | 0.257% | [-0.256%, 0.741%] |
| 48 h | 27.883 | 27.729 | 0.552% | [0.073%, 1.018%] |
| 72 h | 39.831 | 39.653 | 0.448% | [-0.026%, 0.924%] |

五个时长的点估计均支持 core6，48 h 的配对区间下界高于零。该结果为
波浪增量信息在中期水平时间积分中的可识别性提供了独立证据。
其他时长的区间覆盖零，因而该证据定位为幅度较小、具有时长依赖性的
水平过程信号。

主要证据位置：

- `results/wind_only_ablation_circular_mwd_v2/README.md`
- `results/wind_only_ablation_circular_mwd_v2/comparison.json`
- `results/wind_only_ablation_circular_mwd_v2/trajectory_proxy.json`
- `trained_models/wind_only_ablation_circular_mwd_v2/wind_2/run_manifest.json`

## 5. Held-out drifter 多时长位移误差代理

分析代码固定于 `master@ce52512`，没有训练或修改模型。它严格重放 frozen
global held-out test 的 366 个 `original_ID`、646 个源片段和 2,425,415 行，
在非 1 h 间隔处切成 903 个连续 episode。每个 episode 使用非重叠
6/12/24/48/72 h 窗口，累计
`(predicted residual - observed residual) × 3600 s` 后取二维端点距离。
主统计量先在每个 ID 内取窗口中位数，再对 ID 等权；95% 区间以 ID 为单位做
10,000 次配对 bootstrap。

冻结 core6 MLP 相对 Linear 的结果为：

| 时长 | Linear (km) | core6 MLP (km) | 改善率 | 配对 95% CI |
|---:|---:|---:|---:|---:|
| 6 h | 4.367 | 4.355 | 0.278% | [-0.064%, 0.618%] |
| 12 h | 7.978 | 7.963 | 0.184% | [-0.227%, 0.595%] |
| 24 h | 14.771 | 14.723 | 0.328% | [-0.174%, 0.827%] |
| 48 h | 27.871 | 27.729 | 0.509% | [0.016%, 0.992%] |
| 72 h | 39.964 | 39.653 | 0.779% | [0.269%, 1.293%] |

因此逐时 RMSE 的 `0.291%` 改善没有被积分迅速淹没；中位端点代理在
48–72 h 仍存在，并随时长小幅增强。但绝对收益只有 48 h 的 `0.142 km` 和
72 h 的 `0.311 km`，长时 P90 改善区间也跨零，不能据此宣称真实溢油轨迹已有
明显改善。

辅助的 MLP lat7 使用相同 test rows，在 48/72 h 相对 Linear 分别改善
`1.210%` 和 `1.214%`，对应 `0.337/0.485 km`，配对区间高于零；长时 P90
仍不稳定。该结果强化了“存在小幅、具有时间相关性的信号”，但仍未形成数量级
突破。

主要证据位置：

- `results/heldout_trajectory_proxy_v1/README.md`
- `results/heldout_trajectory_proxy_v1/trajectory_proxy.json`
- `results/heldout_trajectory_proxy_v1/trajectory_proxy_summary.csv`

这只是沿观测位置和真实逐时 forcing 的 open-loop displacement-error proxy，
不是递归更新位置的 Fortran 轨迹；它不包含空间偏离后的 forcing、扩散、岸线和
溢油动力学。

## 6. 可比性边界

只有共享 `original_ID` 切分、测试样本、逐行顺序、target 和评价脚本的结果
才能作严格数值排名。

- Global A/B/C/D/E 使用完全相同的 test 行，可以严格比较。
- Wind-only/core6 消融共享 core6 有效行掩码与全部 test 行，可以严格
  解释为移除四项波浪输入后的性能变化。
- CMS 原始网络、CMS 网络重设计和 frozen global 的 CMS 子集评价使用同一批
  4-ID CMS test 行，可以在该区域内直接比较。
- 两轮 adapter 的 development/gate 数量与区域不同；它们回答“校正是否通过
  各自预注册 gate”，不能把 CV 或 gate 指标当成统一 blind-test 排名。
- Range search 只提供样本数量和区域决策证据，不能作为模型性能结果。
- Global test 与 CMS/170 E 子集的区域、ID 支持和权重不同，绝对指标不能直接
  混排成模型优劣表。
- 逐行指标会让长记录获得更高权重；论文同时报告 equal-ID macro 指标和逐 ID
  胜/负比例，避免把少数长轨迹的微小收益解释为普遍改善。

## 7. 分支与归档规则

截至 2026-08-10，以下远端分支保留用于论文写作：

- `wdf_cms_regional_core6_v1@3365743`
- `wdf_global_mlp_lat7_v1@21f8c85`
- `wdf_global_xgb_core6_v1@2d77514`
- `wdf_global_xgb_lat7_v1@fb972f5`

`wdf_cms_range_search_v1` 和 `wdf_cms_network_redesign_v1` 已在远端 annotated
tags 验证后删除本地分支。它们的完整历史和产物分别由
`paper/cms-range-search-v1` 与 `paper/cms-network-redesign-v1` 保留。

`master` 上的 wind-only 消融不新建分支，正式结果由
`paper/wind-only-ablation-v1` 标记，且没有更改 active release。

处理规则：

1. 不 rename、rebase、squash 或重写论文 evidence commits；
2. 不把 CMS、range-search、adapter 或 global factorial 分支整体合并进
   `master`；
3. 论文写作期间保留上述四个远端实验分支；
4. 论文正式冻结后，只要 tags 已验证并在 provenance 中引用，实验分支可以删除；
5. 若需要把通用基础设施带回主线，从干净 master 建立 integration branch，
   只选择基础设施与文档，不携带失败 release 或互斥模型产物。

## 8. 论文叙事边界

core6 circular-MWD v2 仍是 Windows Fortran 案例中的部署主模型。XGBoost、
位置特征、CMS 网络重设计和 adapter 用于回答：

- 更换模型族是否明显突破离线信息上限；
- 简单加入空间坐标是否解决区域差异；
- 缩小到 CMS 或扩大到东亚–西北太平洋后，网络重设计或低阶校正是否改善迁移；
- 当前性能上限是否能归因于单一网络结构，而非目标定义、数据支持、对象差异或
  背景流误差。
- 波浪变量在以风场为主的水平残差速度参数化中能否提供可识别的
  增量信息。

现有证据的共同结论是：纬度和局部网络重设计能带来小幅、局部改善，但简单经度
编码、XGBoost 模型替换和低阶 regional adapter 都没有形成足够稳定的独立测试
增益。工程部署主线因此保持冻结，论文将这些正负结果作为诊断链，而不是把每个
实验分支解释为候选 release。

章节衔接使用两层证据。第一层明确记录波浪输入的绝对预测增益较小，
风场包含了水平残差速度的主要可预测信息。第二层强调波浪输入占 core6
相对线性 WDF 的 joint R² 总增量约 `73%`，并在 48 h 水平位移误差代理上
得到下界高于零的配对区间。这两层结果用于建立“波浪对沉潜油垂向过程
重要—波浪变量对水平残差速度提供小幅增量信息—风浪结合参数化用于
溢油案例”的论文叙事链。
