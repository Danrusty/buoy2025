"""
验证脚本：对ERA5风场/波浪数据匹配结果进行验证和比对
1. 选取一条短轨迹
2. 逐点检查插值数据与原始ERA5数据的一致性
3. 输出详细的比对报告
"""
import xarray as xr
import pandas as pd
import numpy as np
import os
import pickle
import glob


def validate_trajectory_era5_wind(
    trajectory_df,
    era5_wind_dir,
    num_sample_points=5
):
    """
    验证单条轨迹的ERA5风场匹配结果

    Args:
        trajectory_df: 包含插值风场数据的轨迹DataFrame
        era5_wind_dir: ERA5风数据目录
        num_sample_points: 要验证的点数（从轨迹中均匀采样）

    Returns:
        验证报告字典
    """
    print("\n" + "="*80)
    print("验证 ERA5 风场数据匹配结果")
    print("="*80)

    # 检查必需的列
    required_cols = ['latitude', 'longitude', 'time', 'era5_u10', 'era5_v10']
    missing_cols = [col for col in required_cols if col not in trajectory_df.columns]
    if missing_cols:
        print(f"错误: 缺少列 {missing_cols}")
        return None

    # 采样点
    n_total = len(trajectory_df)
    indices = np.linspace(0, n_total - 1, num_sample_points, dtype=int)
    print(f"\n轨迹总长度: {n_total} 个点")
    print(f"采样 {num_sample_points} 个点进行验证")

    results = []

    for idx in indices:
        row = trajectory_df.iloc[idx]
        lat = row['latitude']
        lon = (row['longitude'] + 360) % 360  # 转为 0-360
        time_point = row['time']

        print(f"\n--- 点 #{idx} ---")
        print(f"位置: lat={lat:.2f}, lon={lon:.2f}")
        print(f"时间: {time_point}")
        print(f"插值风场: u10={row['era5_u10']:.3f}, v10={row['era5_v10']:.3f}")

        # 计算应该用哪个文件
        year = pd.Timestamp(time_point).year
        month = pd.Timestamp(time_point).month
        yyyymm = f"{year:04d}{month:02d}"

        # 查找对应的风场文件
        wind_files = sorted(glob.glob(os.path.join(era5_wind_dir, f'wind_{yyyymm}*.nc')))

        if not wind_files:
            print(f"  警告: 未找到 {yyyymm} 的风场文件")
            continue

        wind_file = wind_files[0]
        print(f"  读取文件: {os.path.basename(wind_file)}")

        try:
            # 加载风场数据
            ds = xr.open_dataset(wind_file)

            # 标准化坐标名称
            if 'latitude' in ds.dims and 'lat' not in ds.dims:
                ds = ds.rename({'latitude': 'lat'})
            if 'longitude' in ds.dims and 'lon' not in ds.dims:
                ds = ds.rename({'longitude': 'lon'})

            # 标准化经度范围
            if ds.lon.min() < 0:
                ds['lon'] = (ds['lon'] + 360) % 360
                ds = ds.sortby('lon')

            # ERA5 lat is descending, sort ascending
            if float(ds.lat[0]) > float(ds.lat[-1]):
                ds = ds.sortby('lat')

            # 在该点进行插值
            try:
                interp_result = ds[['u10', 'v10']].interp(
                    lat=float(lat),
                    lon=lon,
                    time=time_point,
                    method='linear'
                )

                u10_raw = float(interp_result['u10'].values)
                v10_raw = float(interp_result['v10'].values)

                print(f"  原始ERA5: u10={u10_raw:.3f}, v10={v10_raw:.3f}")

                # 比较
                u10_diff = abs(u10_raw - row['era5_u10'])
                v10_diff = abs(v10_raw - row['era5_v10'])

                match_status = "✓ 匹配" if (u10_diff < 0.01 and v10_diff < 0.01) else "✗ 不匹配"
                print(f"  差异: Δu10={u10_diff:.3f}, Δv10={v10_diff:.3f} {match_status}")

                results.append({
                    'index': idx,
                    'lat': lat,
                    'lon': lon,
                    'time': time_point,
                    'u10_match': u10_diff < 0.01,
                    'v10_match': v10_diff < 0.01
                })

            except Exception as e:
                print(f"  插值失败: {e}")

            ds.close()

        except Exception as e:
            print(f"  读取文件失败: {e}")

    # 总结
    print(f"\n" + "="*80)
    if results:
        match_count = sum(1 for r in results if r['u10_match'] and r['v10_match'])
        print(f"验证完成: {match_count}/{len(results)} 个点数据匹配")
        if match_count == len(results):
            print("✓ 所有验证点数据正确！")
        else:
            print(f"⚠ 发现 {len(results) - match_count} 个不匹配的点")
    else:
        print("验证失败: 无法进行数据比对")
    print("="*80)

    return results


def validate_trajectory_era5_wave(
    trajectory_df,
    era5_wave_dir,
    num_sample_points=5
):
    """
    验证单条轨迹的ERA5波浪数据匹配结果
    """
    print("\n" + "="*80)
    print("验证 ERA5 波浪数据匹配结果")
    print("="*80)

    # 检查必需的列
    required_cols = ['latitude', 'longitude', 'time', 'era5_swh', 'era5_mwp']
    missing_cols = [col for col in required_cols if col not in trajectory_df.columns]
    if missing_cols:
        print(f"错误: 缺少列 {missing_cols}")
        return None

    # 采样点
    n_total = len(trajectory_df)
    indices = np.linspace(0, n_total - 1, num_sample_points, dtype=int)
    print(f"\n轨迹总长度: {n_total} 个点")
    print(f"采样 {num_sample_points} 个点进行验证")

    results = []

    for idx in indices:
        row = trajectory_df.iloc[idx]
        lat = row['latitude']
        lon = (row['longitude'] + 360) % 360  # 转为 0-360
        time_point = row['time']

        print(f"\n--- 点 #{idx} ---")
        print(f"位置: lat={lat:.2f}, lon={lon:.2f}")
        print(f"时间: {time_point}")
        print(f"插值波浪: swh={row['era5_swh']:.3f}, mwp={row['era5_mwp']:.3f}")

        # 计算应该用哪个文件
        year = pd.Timestamp(time_point).year
        month = pd.Timestamp(time_point).month
        yyyymm = f"{year:04d}{month:02d}"

        # 查找对应的波浪文件
        wave_files = sorted(glob.glob(os.path.join(era5_wave_dir, f'wave_{yyyymm}*.nc')))

        if not wave_files:
            print(f"  警告: 未找到 {yyyymm} 的波浪文件")
            continue

        wave_file = wave_files[0]
        print(f"  读取文件: {os.path.basename(wave_file)}")

        try:
            # 加载波浪数据
            ds = xr.open_dataset(wave_file)

            # 处理 valid_time → time 重命名
            if 'valid_time' in ds.coords:
                ds = ds.rename({'valid_time': 'time'})

            # Wave files have time as float64 (YYYYMMDD.fraction), convert to datetime64
            if ds.time.dtype == np.float64 or ds.time.dtype == np.float32:
                time_float = ds.time.values.astype(np.float64)
                date_ints = time_float.astype(np.int64)
                fracs = time_float - date_ints
                hours = np.round(fracs * 24).astype(int)
                base_dates = pd.to_datetime(date_ints, format='%Y%m%d')
                datetime_index = base_dates + pd.to_timedelta(hours, unit='h')
                ds['time'] = datetime_index.values

            # 标准化坐标名称
            if 'latitude' in ds.dims and 'lat' not in ds.dims:
                ds = ds.rename({'latitude': 'lat'})
            if 'longitude' in ds.dims and 'lon' not in ds.dims:
                ds = ds.rename({'longitude': 'lon'})

            # 标准化经度范围
            if ds.lon.min() < 0:
                ds['lon'] = (ds['lon'] + 360) % 360
                ds = ds.sortby('lon')

            # ERA5 lat is descending, sort ascending
            if float(ds.lat[0]) > float(ds.lat[-1]):
                ds = ds.sortby('lat')

            # 在该点进行插值
            try:
                interp_result = ds[['swh', 'mwp', 'mwd']].interp(
                    lat=float(lat),
                    lon=lon,
                    time=time_point,
                    method='linear'
                )

                swh_raw = float(interp_result['swh'].values)
                mwp_raw = float(interp_result['mwp'].values)
                mwd_raw = float(interp_result['mwd'].values)

                print(f"  原始ERA5: swh={swh_raw:.3f}, mwp={mwp_raw:.3f}, mwd={mwd_raw:.1f}°")

                # 比较
                swh_diff = abs(swh_raw - row['era5_swh'])
                mwp_diff = abs(mwp_raw - row['era5_mwp'])

                match_status = "✓ 匹配" if (swh_diff < 0.01 and mwp_diff < 0.01) else "✗ 不匹配"
                print(f"  差异: Δswh={swh_diff:.3f}, Δmwp={mwp_diff:.3f} {match_status}")

                results.append({
                    'index': idx,
                    'lat': lat,
                    'lon': lon,
                    'time': time_point,
                    'swh_match': swh_diff < 0.01,
                    'mwp_match': mwp_diff < 0.01
                })

            except Exception as e:
                print(f"  插值失败: {e}")

            ds.close()

        except Exception as e:
            print(f"  读取文件失败: {e}")

    # 总结
    print(f"\n" + "="*80)
    if results:
        match_count = sum(1 for r in results if r['swh_match'] and r['mwp_match'])
        print(f"验证完成: {match_count}/{len(results)} 个点数据匹配")
        if match_count == len(results):
            print("✓ 所有验证点数据正确！")
        else:
            print(f"⚠ 发现 {len(results) - match_count} 个不匹配的点")
    else:
        print("验证失败: 无法进行数据比对")
    print("="*80)

    return results


if __name__ == '__main__':
    print("ERA5 数据匹配验证工具")
    print("="*80)

    # 配置 - 优先使用样本文件进行快速验证
    SAMPLE_WIND_FILE = '../../processed_data/trajectories_with_currents_and_wind_samples.pkl'
    SAMPLE_WAVE_FILE = '../../processed_data/trajectories_with_all_features_samples.pkl'
    FULL_WIND_FILE = '../../processed_data/trajectories_with_currents_and_wind.pkl'
    FULL_WAVE_FILE = '../../processed_data/trajectories_with_all_features.pkl'

    ERA5_WIND_DIR = '../../reanalysis/wind'
    ERA5_WAVE_DIR = '../../reanalysis/wave'

    # 确定使用哪个文件进行验证
    wind_file = None
    wave_file = None

    print("检查可用的文件...")
    if os.path.exists(SAMPLE_WIND_FILE):
        wind_file = SAMPLE_WIND_FILE
        print(f"✓ 找到样本风场文件 (快速验证): {os.path.basename(SAMPLE_WIND_FILE)}")
    elif os.path.exists(FULL_WIND_FILE):
        wind_file = FULL_WIND_FILE
        print(f"✓ 找到完整风场文件: {os.path.basename(FULL_WIND_FILE)}")
    else:
        print(f"✗ 未找到风场文件")

    if os.path.exists(SAMPLE_WAVE_FILE):
        wave_file = SAMPLE_WAVE_FILE
        print(f"✓ 找到样本波浪文件 (快速验证): {os.path.basename(SAMPLE_WAVE_FILE)}")
    elif os.path.exists(FULL_WAVE_FILE):
        wave_file = FULL_WAVE_FILE
        print(f"✓ 找到完整波浪文件: {os.path.basename(FULL_WAVE_FILE)}")
    else:
        print(f"✗ 未找到波浪文件")

    print()

    # 验证风场
    if wind_file and os.path.exists(wind_file):
        print(f"加载轨迹数据: {wind_file}")
        with open(wind_file, 'rb') as f:
            trajectories_wind = pickle.load(f)

        print(f"加载了 {len(trajectories_wind)} 条轨迹\n")

        if len(trajectories_wind) > 0 and 'era5_u10' in trajectories_wind[0].columns:
            shortest_traj = min(trajectories_wind, key=len)
            print(f"选取最短的轨迹进行验证 (长度: {len(shortest_traj)} 个点)\n")
            validate_trajectory_era5_wind(shortest_traj, ERA5_WIND_DIR, num_sample_points=min(5, len(shortest_traj)))
        else:
            print("轨迹列表为空或缺少风场数据列")

    # 验证波浪
    if wave_file and os.path.exists(wave_file):
        print(f"\n加载轨迹数据: {wave_file}")
        with open(wave_file, 'rb') as f:
            trajectories_wave = pickle.load(f)

        print(f"加载了 {len(trajectories_wave)} 条轨迹\n")

        if len(trajectories_wave) > 0 and 'era5_swh' in trajectories_wave[0].columns:
            shortest_traj = min(trajectories_wave, key=len)
            print(f"选取最短的轨迹进行验证 (长度: {len(shortest_traj)} 个点)\n")
            validate_trajectory_era5_wave(shortest_traj, ERA5_WAVE_DIR, num_sample_points=min(5, len(shortest_traj)))
        else:
            print("轨迹列表为空或缺少波浪数据列")
