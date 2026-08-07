# WDF_DL_Param

本仓库研究海表漂流浮标相对背景流的风浪驱动残差，并维护可供
C++/Fortran/Windows 溢油模型调用的 ONNX 推理接口。当前工程主线冻结在
global circular-MWD core6 模型；后续 CMS、adapter、空间坐标和 XGBoost
实验用于论文中的机制诊断，没有替换 active deployment。

`master` 另包含 held-out drifter 的 6–72 h 位移误差代理：沿严格连续小时片段
积分逐时速度误差，用于检验 MLP 的小幅点级收益是否能保留到漂移应用时标；该
分析不训练模型，也不改变部署接口。

分支概况：

- `master`：稳定工程和已验证的 global core6 部署；
- `wdf_cms_regional_core6_v1`：CMS baseline 与两轮 adapter 诊断；
- `wdf_global_mlp_lat7_v1`：MLP lat7、lat9 及空间输入汇总；
- `wdf_global_xgb_core6_v1`、`wdf_global_xgb_lat7_v1`：模型类型析因实验；
- CMS range-search 与 network-redesign 已转为 `paper/*` evidence tags，
  不再保留工作分支。

论文实验的准确提交、关键数值、产物位置和可比性边界见
[PAPER_EXPERIMENT_BRANCH_PROVENANCE_20260807.md](PAPER_EXPERIMENT_BRANCH_PROVENANCE_20260807.md)。
最新轨迹代理结论见
[results/heldout_trajectory_proxy_v1/README.md](results/heldout_trajectory_proxy_v1/README.md)。
