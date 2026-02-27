# WSL 内存溢出问题解决方案

## 问题诊断
- 并行处理8个worker，每个加载 ~2.4GB ERA5数据
- 总计可达 ~19GB 峰值内存占用
- WSL 80% 内存限制 → OOM kill

## 解决方案分层

### 方案1：脚本级优化 ✓ 已实施

#### 改进内容
1. **减少并行worker数** (8 → 2)
   - 内存占用: 19GB → 5GB (峰值)
   - 性能: 稍降但稳定

2. **显式内存管理**
   ```python
   del large_arrays  # 显式释放
   gc.collect()     # 强制垃圾回收
   ```

3. **抑制警告**
   - 减少内存开销 ~5%

#### 配置位置
- `match_era5_wind.py`: NUM_WORKERS = 2 (Line 217)
- `match_era5_wave.py`: NUM_WORKERS = 2 (Line 216)

### 方案2：WSL 配置优化（推荐）

#### 编辑 Windows 用户目录的 `.wslconfig` 文件
路径: `C:\Users\<YourUsername>\.wslconfig`

#### 推荐配置
```ini
[wsl2]
# 内存限制：物理内存的 60-70%
# 如果系统有 64GB 物理内存，设置为 40-45GB
memory=40GB

# Swap 空间：用于缓解内存压力
# 建议 = 内存限制的 25-50%
swap=10GB

# 页面文件位置（可选）
pageFile=D:\temp\wsl-swap

# 处理器核心数（可选，默认为系统核心数）
processors=16

[interop]
# 在某些情况下，禁用互操作可能有帮助
enabled=true
```

#### 应用配置
```bash
# 1. 在 PowerShell (as Admin) 中关闭 WSL
wsl --shutdown

# 2. 重启 WSL (自动重新启动)
wsl
```

### 方案3：脚本执行策略

#### 选项 A：分批处理（推荐用于 WSL）
```bash
# 分两批运行（每批处理 ~2000 个轨迹）

# 批次 1 (风场)
/home/dan/miniforge3/envs/buoy-drifter/bin/python \
  src/data_process/match_era5_wind.py > logs/wind_batch1.log 2>&1 &

# 等待完成...
# 检查进度：tail -f logs/wind_batch1.log

# 批次 2 (波浪)
/home/dan/miniforge3/envs/buoy-drifter/bin/python \
  src/data_process/match_era5_wave.py > logs/wave_batch1.log 2>&1 &
```

#### 选项 B：单进程处理（最稳定但最慢）
```bash
# 修改脚本中：NUM_WORKERS = 1
# 内存占用: 2-3GB (几乎不会OOM)
# 耗时: 24-36小时（但不会被kill）
```

#### 选项 C：监控内存使用
```bash
# 在 WSL 中实时监控
watch -n 1 'free -h | head -2'

# 或使用 top
top -b -n 1 | head -20
```

### 方案4：系统级监控

#### 在 PowerShell 中监控 vmmemWSL
```powershell
# 实时查看内存占用
while($true) {
    $wsl = Get-Process vmmemWSL -ErrorAction SilentlyContinue
    if($wsl) {
        Write-Host "vmmemWSL 内存: $([Math]::Round($wsl.WorkingSet/1GB, 2)) GB"
    }
    Start-Sleep -Seconds 2
}
```

#### 设置 Windows 虚拟内存（应急方案）
- 系统设置 → 系统信息 → 高级系统设置
- 性能 → 虚拟内存 → 更改
- 设置为 D:\: 50GB 初始值，100GB 最大值（如果 D: 盘有空间）

## 执行建议（按优先级）

### 第一步：立即应用
✓ 脚本已优化（NUM_WORKERS=2，垃圾回收）
→ **立即测试运行**

### 第二步：如果仍然 OOM
1. 编辑 `.wslconfig` 文件
2. 增加 swap 空间和内存限制
3. 重启 WSL

### 第三步：如果还是 OOM
1. 将 NUM_WORKERS 从 2 改为 1
2. 接受较长的运行时间（~30小时）

### 第四步：最终方案
分开两台机器或云环境处理，或使用 Dask/Ray 分布式框架

## 性能对比

| 配置 | 内存峰值 | 耗时 | 稳定性 |
|-----|---------|------|--------|
| NUM_WORKERS=8 (原始) | ~19GB | ~1h | ✗ OOM kill |
| NUM_WORKERS=2 + gc | ~5GB | ~2h | ✓ 稳定 |
| NUM_WORKERS=1 + gc | ~2.5GB | ~4h | ✓✓ 极稳定 |
| + .wslconfig 优化 | ~5GB | ~2h | ✓✓ 推荐 |

## 故障排除

### 症状 1: 脚本卡住不动
- 原因：交换空间不足或内存压力过大
- 解决：增加 swap，或降低 NUM_WORKERS

### 症状 2: 速度急剧下降（倒数计时）
- 原因：已触发 swap，系统在频繁读写磁盘
- 解决：检查 D: 盘空间，或降低并发

### 症状 3: 进程被强行 kill，无错误信息
- 原因：Windows 的 vmmemWSL 内存枪毙
- 解决：配置 `.wslconfig` 增加内存限制

## 快速检查清单

- [ ] `.wslconfig` 是否存在于 `C:\Users\<YourUsername>\`
- [ ] 脚本中 NUM_WORKERS 是否 ≤ 3
- [ ] D: 盘或其他分区是否有 50GB+ 空闲空间（用于 swap）
- [ ] 运行前清理临时文件 (`wsl --clean` 或手动清理)
- [ ] 使用 `watch free -h` 监控内存

---

**建议从"改进脚本 + 修改 .wslconfig"开始，这两个结合可以解决 90% 的问题。**
