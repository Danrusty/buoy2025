import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import pickle
from datetime import datetime
from tqdm import tqdm

# --- 1. 配置与设置 ---
# ==============================================================================
LOG_DIR = '../../logs'
os.makedirs(LOG_DIR, exist_ok=True)
log_filename = f"data_sanitization_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, log_filename), encoding='utf-8'),
        logging.StreamHandler()
    ]·
)

# --- 输入/输出配置 ---
# 注意：这里改为直接读取 NC 文件
INPUT_NC_PATH = '../../drifter_hourly_qc_2cd0_7581_6b62.nc' 
OUTPUT_SANITIZED_PATH = '../../processed_data/trajectories_sanitized.pkl'
RESULTS_OUTPUT_DIR = '../../results/data_analysis'
os.makedirs(RESULTS_OUTPUT_DIR, exist_ok=True)

# --- 筛选参数 ---
# 时间窗口 (用户指定)
START_DATE = pd.Timestamp("2020-01-01")
END_DATE = pd.Timestamp("2022-10-31 23:59:59")

# 序列分割参数
# 既然是 hourly 数据，容忍度设为 1.5 小时。
# 如果差值 > 1.5小时，说明中间缺了至少一个点，必须切断。
MAX_GAP_HOURS = 1.5 
MIN_TRAJ_LENGTH = 72  # 最小保留长度 (小时)

# ==============================================================================

def decode_ids(id_char_array):
    """
    将 NetCDF 的二维字符数组 (row, 15) 解码为字符串列表。
    优化速度：使用 numpy view 或列表推导。
    """
    # 假设 ID 数组是 byte 类型 (S1)
    # 转换为 Unicode 字符串并去除空格
    return ["".join(row.astype(str)).strip() for row in id_char_array]

def process_and_sanitize_nc(nc_path, output_path):
    logging.info(f"--- 启动数据清洗流程 (Target: Undrogued Drifters) ---")
    logging.info(f"读取原始 NetCDF 文件: {nc_path}")

    if not os.path.exists(nc_path):
        logging.error("文件不存在！")
        return

    # 1. 使用 xarray 懒加载
    ds = xr.open_dataset(nc_path, decode_times=False) # 保持时间为数字以便快速比较
    
    # 获取原始行数
    total_rows = ds.dims['row']
    logging.info(f"原始数据总行数: {total_rows}")

    # 2. 提取关键列 (Numpy Arrays) 以便快速过滤
    logging.info("提取关键变量到内存...")
    times = ds['time'].values # seconds since 1970
    drogue_lost_dates = ds['drogue_lost_date'].values
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    ves = ds['ve'].values
    vns = ds['vn'].values
    
    # 转换时间边界为 timestamp float
    t_start = (START_DATE - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    t_end = (END_DATE - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

    # --- 核心过滤逻辑 ---
    logging.info("执行向量化过滤...")

    # A. 时间范围过滤
    mask_time = (times >= t_start) & (times <= t_end)
    
    # B. Undrogued (无锚系) 筛选
    # 逻辑: 当前时间 >= 丢失时间 且 丢失时间有效(>0)
    # 注意: GDP 中未丢失通常为 0 或 1e14 (取决于版本，但通常 time < lost_date 代表有锚)
    # 我们只想要 time >= lost_date
    mask_undrogued = (times >= drogue_lost_dates) & (drogue_lost_dates > 0)
    
    # C. 有效速度过滤 (去除 NaN 和 绝对静止/搁浅)
    # 0.001 m/s 作为搁浅阈值
    mask_valid_vel = (~np.isnan(ves)) & (~np.isnan(vns)) & \
                     ((np.abs(ves) > 0.001) | (np.abs(vns) > 0.001))
    
    # D. 坐标有效性
    mask_valid_geo = (lats >= -90) & (lats <= 90) & (lons >= -180) & (lons <= 180)

    # 综合 Mask
    final_mask = mask_time & mask_undrogued & mask_valid_vel & mask_valid_geo
    
    selected_count = np.sum(final_mask)
    logging.info(f"筛选结果: {selected_count} 行符合要求 (占比 {selected_count/total_rows:.2%})")
    
    if selected_count == 0:
        logging.error("没有符合条件的数据！请检查筛选逻辑或源数据。")
        return

    # 3. 构建 DataFrame
    logging.info("构建 Pandas DataFrame...")
    
    # 只解码符合条件的 ID 以节省时间
    raw_ids = ds['ID'].values[final_mask]
    logging.info("正在解码 ID (这可能需要一点时间)...")
    decoded_ids = decode_ids(raw_ids)
    
    df = pd.DataFrame({
        'ID': decoded_ids,
        'time': pd.to_datetime(times[final_mask], unit='s'),
        'latitude': lats[final_mask],
        'longitude': lons[final_mask],
        've': ves[final_mask],
        'vn': vns[final_mask]
    })
    
    ds.close()
    
    # 4. 按 ID 分组并分割轨迹
    logging.info("开始按 ID 分组并检查时间连续性...")
    sanitized_trajectories = []
    
    # 进度条
    unique_ids = df['ID'].unique()
    logging.info(f"共发现 {len(unique_ids)} 个唯一的无锚系浮标 ID")

    for buoy_id, group in tqdm(df.groupby('ID'), desc="Splitting Trajectories"):
        # 确保按时间排序
        group = group.sort_values('time')
        
        # 计算时间差 (小时)
        time_diffs = group['time'].diff().dt.total_seconds() / 3600.0
        
        # 找到断点: 只要间隔 > 1.5小时 (考虑到 hourly 数据可能有微小偏差，1.5是安全的判定)
        split_indices = np.where(time_diffs > MAX_GAP_HOURS)[0]
        
        # 添加起始和结束点
        split_points = [0] + split_indices.tolist() + [len(group)]
        
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i+1]
            
            sub_traj = group.iloc[start:end].copy()
            
            # 最小长度检查
            if len(sub_traj) >= MIN_TRAJ_LENGTH:
                # 重置索引
                sub_traj = sub_traj.reset_index(drop=True)
                # 添加元数据
                sub_traj['original_ID'] = buoy_id
                sub_traj['segment_index'] = i
                sanitized_trajectories.append(sub_traj)

    final_count = len(sanitized_trajectories)
    logging.info(f"处理完成。生成 {final_count} 条连续的子轨迹。")

    # 5. 可视化统计 (可选)
    plot_distribution(sanitized_trajectories)

    # 6. 保存
    logging.info(f"保存结果到: {output_path}")
    with open(output_path, 'wb') as f:
        pickle.dump(sanitized_trajectories, f)
    
    logging.info("--- 数据清洗全部完成 ---")

def plot_distribution(trajectories):
    """绘制轨迹长度分布"""
    lengths = [len(t) for t in trajectories]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black', log=True)
    plt.title(f'Sanitized Undrogued Trajectory Lengths (Count={len(trajectories)})')
    plt.xlabel('Length (Hours)')
    plt.ylabel('Count (Log Scale)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plot_path = os.path.join(RESULTS_OUTPUT_DIR, 'sanitized_length_dist.png')
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"长度分布图已保存至: {plot_path}")

if __name__ == '__main__':
    process_and_sanitize_nc(INPUT_NC_PATH, OUTPUT_SANITIZED_PATH)