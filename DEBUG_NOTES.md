# ERA5 数据匹配问题分析与修复

## 问题诊断：为什么63小时后结果是空列表

### 根本原因：坐标范围不匹配

在原代码中，执行顺序如下：

```
1. 加载 ERA5 数据 ds_era5_raw （坐标可能是 -180 到 180）
2. 计算轨迹空间范围 → 转换为 0-360 范围
3. 尝试用 0-360 范围的经度去裁剪 ds_era5_raw ← 坐标范围不匹配！
4. xr.sel() 返回空集
5. 为空的数据集导致后续插值失败
6. 所有异常被 except 捕获，轨迹被跳过
7. 最终：4171 条轨迹 → 0 条成功处理
```

### 具体位置

**match_era5_wind.py 原问题：**
- 第63-75行：按时间过滤 ✓
- 第81-97行：加载数据 ✓
- 第102行：concat ✓
- 第107-118行：准备坐标，计算空间范围 - **轨迹经度已转为 0-360**
- **第124行：尝试裁剪 ds_era5_raw (坐标可能还是 -180~180) ← OOM 或空集**
- 第138-140行：才对 ds_era5 做坐标转换（已经太晚！）

## 修复内容

### 修复1：重新排列坐标处理顺序

**原来的错误流程：**
```
加载 → concat → 准备插值坐标 → 计算范围 → 空间裁剪(坐标不匹配) → 转换坐标
```

**修复后的正确流程：**
```
加载 → concat → [立即转换坐标名称和范围] → 准备插值坐标 → 计算范围 → 空间裁剪(坐标一致) → 插值
```

### 修复2：自动处理坐标名称

```python
# 添加在 concat 之后，裁剪之前
coord_mapping = {}
if 'latitude' in ds.dims and 'lat' not in ds.dims:
    coord_mapping['latitude'] = 'lat'
if 'longitude' in ds.dims and 'lon' not in ds.dims:
    coord_mapping['longitude'] = 'lon'
if coord_mapping:
    ds = ds.rename(coord_mapping)

# 立即转换经度范围（在任何 slice 操作之前）
if ds.lon.min() < 0:
    ds['lon'] = (ds['lon'] + 360) % 360
    ds = ds.sortby('lon')
```

### 修复3：创建验证脚本

新增 `validate_era5_match.py`，用于：
- 自动选取最短的轨迹（快速验证）
- 采样该轨迹的5个点
- 对每个点：重新从原始ERA5文件读取值，与插值结果进行比对
- 输出详细的匹配报告（差异 < 0.01 算匹配）

## 操作步骤

### 第1步：运行修复后的风场脚本（预计 1-2 天）

```powershell
cd C:\Users\dan\tmp_buoy
python src\data_process\match_era5_wind.py
```

### 第2步：验证结果

```powershell
# 复制验证脚本到 Windows (在 WSL 中)
scp src/data_process/validate_era5_match.py dan@localhost:/c/Users/dan/tmp_buoy/src/data_process/

# 在 Windows 中运行验证
python src\data_process\validate_era5_match.py
```

验证脚本会：
1. 加载 `trajectories_with_currents_and_wind.pkl`
2. 找到最短的轨迹
3. 从原始ERA5风场文件重新读取5个采样点的值
4. 与插值结果进行比对
5. 输出匹配报告（✓ 或 ✗）

### 第3步：若验证通过，运行波浪脚本（预计 1 天）

```powershell
python src\data_process\match_era5_wave.py
```

### 第4步：验证波浪数据

修改 `validate_era5_match.py` 的最后一行指向 `trajectories_with_all_features.pkl`，再运行一次验证。

## 预期改进

| 指标 | 原来 | 修复后 |
|-----|------|-------|
| 处理时间 | 63小时 | 1-2天（更稳定） |
| 结果 | 0 条轨迹 | 预期 3500+ 条轨迹 |
| 数据正确性 | N/A | 通过验证脚本确认 |

## 关键要点

1. **不能跳过验证**：由于修复涉及坐标处理的核心逻辑，必须验证结果
2. **最短轨迹优先**：验证脚本自动选择最短轨迹，运行快速
3. **点对点比对**：验证脚本直接从原始ERA5文件重新读取值，确保数据完整性

---

## ERA5 波浪数据匹配调试全记录（match_era5_wave.py）

**背景**：风场匹配成功后，波浪匹配出现 ~53%（2218/4155）轨迹插值后全为 NaN 的问题。以下是完整的试错过程和最终根本原因分析。

### 尝试一：坐标系不匹配假说（× 不成立）

**初始假说**：ERA5 波浪数据的经度是 -180~180，而轨迹经度转换为了 0~360，导致 `xr.interp` 时坐标范围不匹配，插值在范围之外返回 NaN。

**诊断方法**：在脚本中打印 ERA5 波浪文件的实际坐标范围。

**诊断结果**：ERA5 波浪数据的经度本就是 0~359.5（已是 0-360 格式），纬度 -90~90，坐标没有问题。

**结论**：假说不成立，但这一步修复了其他几个次要问题：`valid_time` 命名冲突、经度比较精度（加 `float()` 包装）。

---

### 尝试二：ocean-masked 加权插值（× 方向正确但不完整）

**假说**：ERA5 波浪模型只在海洋格点有值，陆地格点为 NaN。普通线性插值遇到 NaN 邻格时直接返回 NaN（NaN 传播）。解决方案是"先填零再归一化"。

**方法**：
```python
ocean_mask = da.notnull().astype(float)   # 海洋=1，陆地=0
da_filled = da.fillna(0.0)
interp_data = da_filled.interp(...)
interp_mask = ocean_mask.interp(...)
result = np.where(mask >= 0.1, data / mask, np.nan)
```

**诊断结果**：问题依然存在，相同轨迹仍全为 NaN。

**深入原因**：该方法要求插值点周围**至少有一个**海洋格点有值。但对于完全被陆地/海冰包围的格点，整个裁剪区域内所有格点均为 NaN，`mask` 全为 0，`result` 仍为 NaN。该方法对"只有部分邻格为 NaN"有效，对"全部邻格为 NaN"无效。

---

### 尝试三：文件句柄提前关闭假说（× 部分成立但不是主因）

**假说**：`ds_cropped` 是 `ds` 的 lazy 视图，`ds.close()` 关闭了底层文件句柄，导致后续 `interp` 读取数据时全为 NaN 而不报错。

**理论推导**：ERA5 波浪的纬度本已是升序（-90→90），`sortby('lat')` 条件为假不执行，`ds` 保持原始文件对象引用，`ds.close()` 真正关闭文件 → 后续 lazy 读取失败。风场数据纬度降序，`sortby` 被触发，`ds` 被重赋值为派生对象，`close()` 是 no-op，因此风场脚本碰巧没有此问题。

**修复**：在 `ds.close()` 前加 `ds_cropped = ds_cropped.load()`。

**实际测试结果**：加 `load()` 后问题仍然存在。

**重新诊断**：直接在 WSL terminal 中运行，观察到失败轨迹的坐标：
- Traj 2：lat=65.9°N，lon=-19.8°（**冰岛海岸**）
- Traj 7：lat=-70.25°S，lon=163.6°（**南极海冰区**）
- Traj 14：lat=68.3°N，lon=-53.9°（**格陵兰峡湾**）

**结论**：`load()` 修复是必要的防御性改进，但并非主因。主因是地理位置本身的物理特性。

---

### 根本原因确认：ERA5 波浪模型的物理边界 + NaN 全覆盖

**真正的根本原因**：ERA5 波浪模型（WAM 模式）的计算域严格限于开阔海洋，不包括：
1. **陆地格点**：明显的陆地返回 NaN（分辨率 0.5°）
2. **海冰覆盖区**：南北极的海冰格点也返回 NaN，季节性变化
3. **内陆封闭水体**：格陵兰峡湾、波罗的海等部分封闭海域可能不在模型域内

当浮标漂移到上述区域时，空间裁剪后的局部网格（2×2 度范围）**全部**为 NaN，无论是普通插值、ocean-masked 插值，还是任何依赖"至少有一个有效邻格"的方法都会失败。

**诊断验证**：在 WSL 中用诊断脚本直接读取 Traj 7 区域的 ERA5 wave SWH 值，确认裁剪区域内所有时间步的所有格点均为 NaN（海冰区无波浪数据）。

---

### 最终修复：Coast-fill 海岸外推

**方法**：在三维线性插值之前，对每个时间步的 NaN 格点，用 `scipy.interpolate.NearestNDInterpolator` 填补为最近有效海洋格点的值：

```python
from scipy.interpolate import NearestNDInterpolator

lat_2d, lon_2d = np.meshgrid(
    ds_era5_wave.lat.values, ds_era5_wave.lon.values, indexing='ij'
)
for var in wave_vars:
    data_orig = ds_era5_wave[var].values  # (time, lat, lon)
    for t_idx in range(data_orig.shape[0]):
        slc = data_orig[t_idx]
        nan_mask = np.isnan(slc)
        if not nan_mask.any() or not (~nan_mask).any():
            continue
        nn = NearestNDInterpolator(
            np.column_stack([lat_2d[~nan_mask], lon_2d[~nan_mask]]),
            slc[~nan_mask]
        )
        data_orig[t_idx][nan_mask] = nn(lat_2d[nan_mask], lon_2d[nan_mask])
```

**效果（15 条最短轨迹测试）**：
| 失败类型 | 修复前 | 修复后 |
|---------|--------|--------|
| 冰岛海岸 | 全 NaN | ✅ 成功 |
| 南极海冰 | 全 NaN | ❌ 仍失败（物理上确实无数据） |
| 格陵兰峡湾 | 全 NaN | ❌ 仍失败（WAM 模型域外） |
| 总成功率 | 12/15 (80%) | 13/15 (87%) |

残留失败（2/15）是物理上真正无法获取波浪数据的区域，coast-fill 对"局部裁剪区域全为 NaN"也无能为力，属合理行为。

---

### 试错过程中的物理启示

**关于研究目的与数据选取的思考**：

本研究的核心目标是通过浮标漂移数据反演风拖曳因子（wind drift factor），用于修正溢油漂移模型。从物理机制出发：

1. **极地海冰区浮标**：冰区浮标的漂移受海冰动力学主导（冰内应力、冰脊等），而非风-波-流的直接作用，与目标研究问题（开阔海面风拖曳）关联甚少，可直接剔除。

2. **峡湾/内陆水道浮标**：ERA5 波浪模型不覆盖峡湾，且峡湾内的漂移受地形约束（峡湾环流、潮汐），机制与开阔海面截然不同。

3. **西风带浮标**：西风带（40°~60°S 及北半球同纬度）是高能环境，波浪强烈、风速大，ERA5 在此精度尚可，是研究的良好案例。

4. **近岸浮标**（研究的主要目标）：溢油事故多发于近岸，浮标数据最相关。ERA5 0.5° 分辨率在近岸略显粗糙，但 coast-fill 能有效弥补近岸格点 NaN 问题。

**初期选择全球数据的合理性**：在流程跑通阶段保留全部数据是合理的——可以检验管道在各种边缘情况下的鲁棒性，避免因过早过滤引入偏差。后续可通过地理筛选（去除极地冰区 |lat|>65°、去除封闭海域内的轨迹）显著提升匹配成功率。

---

### 各版本 ERA5 文件格式兼容性问题汇总

调试过程中发现 ERA5 不同下载批次的文件格式存在差异，需要在加载时逐一处理：

| 问题 | 表现 | 修复 |
|------|------|------|
| 新版 CDS API 同时含 `time` 和 `valid_time` | `rename` 报目标名已存在 | 若 `time` 已存在则 `drop_vars('valid_time')` |
| 旧版波浪文件时间为 float64 | 格式 YYYYMMDD.fraction（如 20200101.5 = 2020-01-01 12:00）| 手动解析：整数部分=日期，小数×24=小时 |
| 重复时间戳（expver 混合） | `sel(time=slice())` 报 non-unique label | `isel` 去重后必须 `assign_coords` 强制重建 pandas Index |
| 纬度降序（90→-90） | `sel(lat=slice(south,north))` 返回空集 | `sortby('lat')` 确保升序后再裁剪 |

**教训**：`isel` 去重后不能仅靠 `isel`，必须追加 `assign_coords(time=('time', ds.time.values))` 刷新内部索引，否则 xarray 底层的 pandas Index 不会更新，`sel(slice())` 仍会失败。
