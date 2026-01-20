import xarray as xr
import pandas as pd
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm


def match_cfsv2_currents(processed_buoy_file, cfsv2_dir, output_dir):
    """
    Matches CFSv2 reanalysis current data with preprocessed buoy trajectories.

    For each point in the hourly trajectories, it performs spatio-temporal
    interpolation on the CFSv2 dataset to find the corresponding background
    sea water velocity (u and v components) at 5m depth.

    Args:
        processed_buoy_file (str): Path to the 'trajectories_sanitized.pkl' file.
        cfsv2_dir (str): Directory containing the yearly CFSv2 NetCDF files.
        output_dir (str): Directory to save the final processed file.
    """
    print("--- 开始匹配CFSv2海流数据 ---")

    # --- 1. 加载预处理过的浮标轨迹 ---
    print(f"步骤 1/4: 加载浮标数据从: {processed_buoy_file}")
    try:
        with open(processed_buoy_file, 'rb') as f:
            buoy_trajectories = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 浮标数据文件未找到 at '{processed_buoy_file}'")
        return
    print(f"加载了 {len(buoy_trajectories)} 段连续的无水帆浮标轨迹。")


    # --- 2. 加载CFSv2数据 ---
    print(f"步骤 2/4: 使用 xarray.open_mfdataset 加载CFSv2数据...")
    u_files = sorted(glob.glob(os.path.join(cfsv2_dir, '*ocnu*.nc')))
    v_files = sorted(glob.glob(os.path.join(cfsv2_dir, '*ocnv*.nc')))

    if not u_files or not v_files:
        print(f"错误: 在目录 '{cfsv2_dir}' 中未找到CFSv2 NetCDF文件。")
        print("请确保文件名包含 'ocnu' (u) 和 'ocnv' (v)。")
        return
    print(f"找到 {len(u_files)} 个 U-分量文件和 {len(v_files)} 个 V-分量文件。")

    ds_u = xr.open_mfdataset(u_files, combine='by_coords', parallel=True)
    ds_v = xr.open_mfdataset(v_files, combine='by_coords', parallel=True)
    
    print("正在合并数据集...")
    ds_cfsv2 = xr.merge([ds_u, ds_v])

    # --- FIX: 删除重复的时间戳 ---
    # open_mfdataset 会连接年度文件，可能导致交界处时间戳重复 (e.g., 2021-01-01 00:00 同时存在于2020和2021文件中)
    # 这会导致插值错误 "Reindexing only valid with uniquely valued Index objects"
    print("正在剔除数据中的重复时间戳...")
    _, unique_indices = np.unique(ds_cfsv2['time'], return_index=True)
    ds_cfsv2 = ds_cfsv2.isel(time=unique_indices)
    print(f"去重后有效时间维度大小为: {len(ds_cfsv2['time'])}")
    
    # 移除深度维度 (如果存在且为单一值)
    if 'depth' in ds_cfsv2.coords and len(ds_cfsv2.coords['depth']) == 1:
        ds_cfsv2 = ds_cfsv2.squeeze('depth')
        
    print("CFSv2 数据集加载并合并完成。")


    # --- 3. 迭代、匹配和插值 ---
    print("步骤 3/4: 迭代所有轨迹并进行时空插值...")
    enriched_trajectories = []
    for traj_df in tqdm(buoy_trajectories, desc="插值海流数据中"):
        # 准备插值所需的坐标数组
        lats = xr.DataArray(traj_df['latitude'], dims="points")
        lons = xr.DataArray(traj_df['longitude'], dims="points")
        times = xr.DataArray(traj_df['time'], dims="points")

        # --- 关键: 经度坐标转换 ---
        # 浮标经度是 -180 到 180, CFSv2 是 0 到 360
        lons_360 = (lons + 360) % 360

        try:
            # 使用 xarray 的高级插值功能
            interpolated_currents = ds_cfsv2[['ocnu5','ocnv5']].interp(
                lat=lats,
                lon=lons_360,
                time=times,
                method="linear"
            )

            # 将插值结果添加回 DataFrame
            traj_df['cfsv2_u'] = interpolated_currents['ocnu5'].values
            traj_df['cfsv2_v'] = interpolated_currents['ocnv5'].values

            # 删除插值失败（结果为NaN）的行
            traj_df.dropna(subset=['cfsv2_u', 'cfsv2_v'], inplace=True)

            # 只有当轨迹仍然有足够数据时才保留
            if len(traj_df) > 1:
                enriched_trajectories.append(traj_df)

        except Exception as e:
            print(f"\n警告: 处理某段轨迹时发生插值错误: {e}")
            print("该段轨迹将被跳过。")
            continue

    print(f"插值完成。剩余 {len(enriched_trajectories)} 段轨迹拥有匹配的海流数据。")

    # --- 4. 保存结果 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_filename = os.path.join(output_dir, 'trajectories_with_cfsv2_currents.pkl')
    print(f"\n步骤 4/4: 将富含海流数据的结果保存到: {output_filename}")
    with open(output_filename, 'wb') as f:
        pickle.dump(enriched_trajectories, f)

    print("--- CFSv2海流数据匹配完成！---")

    if enriched_trajectories:
        print(f"\n最终产出是一个Python列表，其中包含 {len(enriched_trajectories)} 个pandas.DataFrame。")
        print("每个DataFrame代表一个【无水帆】浮标的一段【1小时间隔】【连续】轨迹，并包含了匹配的CFSv2海流速度。")
        print("\n第一个轨迹的头部数据示例:")
        print(enriched_trajectories[0].head())
    else:
        print("\n警告: 没有轨迹成功匹配海流数据。请检查输入数据和时间范围。")


if __name__ == '__main__':
    # --- 用户配置 ---
    # 1. 上一步生成的预处理浮标文件
    PROCESSED_BUOY_FILE = '../../processed_data/trajectories_sanitized.pkl'

    # 2. 存放所有CFSv2 NetCDF文件的目录
    CFSV2_DATA_DIRECTORY = '../../reanalysis/CFSv2/'

    # 3. 输出目录
    OUTPUT_DIRECTORY = '../../processed_data'

    # --- 运行脚本 ---
    if not os.path.exists(PROCESSED_BUOY_FILE):
        print(f"错误: 输入的浮标文件 '{PROCESSED_BUOY_FILE}' 不存在。请先运行第一步预处理脚本。")
    elif not os.path.exists(CFSV2_DATA_DIRECTORY):
        print(f"错误: CFSv2数据目录 '{CFSV2_DATA_DIRECTORY}' 不存在。请检查路径。")
    else:
        match_cfsv2_currents(PROCESSED_BUOY_FILE, CFSV2_DATA_DIRECTORY, OUTPUT_DIRECTORY)
