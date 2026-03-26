import xarray as xr
import pandas as pd
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm


def match_era5_wind(processed_buoy_file_with_currents, era5_dir, output_dir, sample_mode=False, sample_size=10):
    """
    Matches ERA5 reanalysis wind data with buoy trajectories using serial processing.

    For each trajectory point, performs spatio-temporal interpolation on the ERA5 dataset
    to find the corresponding 10-meter wind components (u10, v10).

    Feature engineering:
    - era5_u10, era5_v10: 10-meter wind components
    - era5_wind_speed: Wind speed magnitude
    - era5_wind_dir_sin, era5_wind_dir_cos: Periodic wind direction encoding

    Args:
        processed_buoy_file_with_currents (str): Path to trajectories with CFS current data
        era5_dir (str): Directory containing ERA5 wind NetCDF files
        output_dir (str): Directory to save output
        sample_mode (bool): If True, only process shortest trajectories for quick validation
        sample_size (int): Number of shortest trajectories to process in sample mode
    """
    mode_info = "【采样验证模式】" if sample_mode else "【完整处理模式】"
    print(f"--- 开始匹配ERA5风场数据 {mode_info} (串行处理) ---")

    # --- 步骤 1/4: 加载浮标轨迹 ---
    print(f"步骤 1/4: 加载浮标数据从: {processed_buoy_file_with_currents}")
    try:
        with open(processed_buoy_file_with_currents, 'rb') as f:
            trajectories_with_currents = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 浮标数据文件未找到 at '{processed_buoy_file_with_currents}'")
        return
    if not trajectories_with_currents:
        print("错误: 加载的浮标轨迹列表为空，无法继续。")
        return

    print(f"加载了 {len(trajectories_with_currents)} 段连续轨迹。")

    # Sample mode: select shortest trajectories for quick validation
    if sample_mode:
        trajectories_with_currents = sorted(trajectories_with_currents, key=len)
        trajectories_with_currents = trajectories_with_currents[:sample_size]
        print(f"采样模式: 选择最短的 {len(trajectories_with_currents)} 条轨迹，长度: {[len(t) for t in trajectories_with_currents]}")

    # --- 步骤 2/4: 检查ERA5数据 ---
    print(f"步骤 2/4: 检查ERA5风场数据...")
    era5_all_files = sorted(glob.glob(os.path.join(era5_dir, '*.nc')))
    if not era5_all_files:
        print(f"错误: ERA5数据目录 '{era5_dir}' 中未找到 .nc 文件。")
        return
    print(f"找到 {len(era5_all_files)} 个ERA5风场文件。")

    # --- 步骤 3/4: 串行处理各轨迹 ---
    print("步骤 3/4: 逐条处理轨迹并进行时空插值...")
    fully_enriched_trajectories = []

    # Debug counters
    fail_stats = {
        'no_files': 0,
        'load_failed': 0,
        'concat_failed': 0,
        'coord_failed': 0,
        'crop_failed': 0,
        'interp_failed': 0,
        'all_nan': 0,
        'too_short': 0,
        'success': 0
    }

    for traj_idx, traj_df in enumerate(tqdm(trajectories_with_currents, desc="处理轨迹中")):
        traj_df = traj_df.copy()

        # Determine which months this trajectory spans
        time_min = traj_df['time'].min()
        time_max = traj_df['time'].max()

        # Filter files to only those covering the trajectory's time range
        era5_files = []
        for f in era5_all_files:
            basename = os.path.basename(f)
            if basename.startswith('wind_') and basename.endswith('.nc'):
                try:
                    file_yyyymm = basename.split('_')[1][:6]  # Extract YYYYMM
                    file_date = pd.Timestamp(year=int(file_yyyymm[:4]), month=int(file_yyyymm[4:6]), day=1)
                    if file_date <= time_max + pd.Timedelta(days=1) and \
                       file_date + pd.DateOffset(months=1) > time_min - pd.Timedelta(days=1):
                        era5_files.append(f)
                except (ValueError, IndexError):
                    continue

        if not era5_files:
            fail_stats['no_files'] += 1
            continue

        # --- Calculate spatial range BEFORE loading files ---
        lat_min = float(traj_df['latitude'].min() - 1)
        lat_max = float(traj_df['latitude'].max() + 1)
        lon_min = float((traj_df['longitude'].min() - 1 + 360) % 360)
        lon_max = float((traj_df['longitude'].max() + 1 + 360) % 360)

        try:
            # Load, standardize, and crop each file BEFORE concatenation
            datasets = []
            load_errors = []
            for f in era5_files:
                try:
                    ds = xr.open_dataset(f)

                    # === Step 1: Standardize coordinates on single file (memory-efficient) ===
                    # Rename coordinates if needed
                    if 'latitude' in ds.dims and 'lat' not in ds.dims:
                        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

                    # Standardize longitude to 0-360 range
                    if ds.lon.min() < 0:
                        ds['lon'] = (ds['lon'] + 360) % 360
                        ds = ds.sortby('lon')

                    # Sort latitude ascending (ERA5 is descending)
                    # This sortby on single file (~2-3 GB) is memory-safe
                    if float(ds.lat[0]) > float(ds.lat[-1]):
                        ds = ds.sortby('lat')

                    # 统一时间精度为 datetime64[ns]（与轨迹数据一致）。
                    # 新版 CDS API 下载的文件时间可能为 datetime64[us]，xr.interp 时
                    # 两者 float64 数值差 1000 倍，导致插值点超出范围全为 NaN。
                    if ds.time.dtype != np.dtype('datetime64[ns]'):
                        ds['time'] = ds.time.values.astype('datetime64[ns]')

                    # === Step 2: Remove duplicate timestamps BEFORE slicing ===
                    # 注意：仅用 isel 去重后，xarray 内部的 pandas Index 不会自动重建，
                    # 需要 assign_coords 强制刷新，否则 .sel(time=slice()) 仍会报
                    # "Cannot get left slice bound for non-unique label" 错误。
                    _, unique_indices = np.unique(ds.time.values, return_index=True)
                    if len(unique_indices) < len(ds.time):
                        ds = ds.isel(time=np.sort(unique_indices))
                        ds = ds.assign_coords(time=('time', ds.time.values))

                    # === Step 3: Time cropping ===
                    file_time_min = pd.Timestamp(ds.time.values[0])
                    file_time_max = pd.Timestamp(ds.time.values[-1])
                    select_min = max(file_time_min, time_min - pd.Timedelta(days=1))
                    select_max = min(file_time_max, time_max + pd.Timedelta(days=1))

                    if select_min > select_max:
                        ds.close()
                        continue

                    ds = ds.sel(time=slice(select_min, select_max))
                    if len(ds.time) == 0:
                        ds.close()
                        continue

                    # === Step 4: Spatial cropping (CRITICAL - reduces memory before concat) ===
                    if lon_max < lon_min:
                        # Trajectory crosses dateline - split into two regions
                        ds1 = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, 360))
                        ds2 = ds.sel(lat=slice(lat_min, lat_max), lon=slice(0, lon_max))
                        ds_cropped = xr.concat([ds1, ds2], dim='lon')
                    else:
                        ds_cropped = ds.sel(
                            lat=slice(lat_min, lat_max),
                            lon=slice(lon_min, lon_max)
                        )

                    if len(ds_cropped.lon) == 0 or len(ds_cropped.lat) == 0:
                        ds.close()
                        continue

                    # Append the small, cropped dataset (~tens of MB)
                    datasets.append(ds_cropped)
                    ds.close()

                except Exception as e:
                    load_errors.append((os.path.basename(f), str(e)))
                    continue

            if not datasets:
                fail_stats['load_failed'] += 1
                if traj_idx < 5 or len(load_errors) > 0:
                    print(f"\n[Traj {traj_idx}] 无法加载任何ERA5文件。时间范围: {time_min} to {time_max}")
                    if load_errors:
                        print(f"  加载错误: {load_errors[:3]}")
                continue

            # Concatenate already-cropped small datasets (total ~hundreds of MB, no OOM)
            ds_era5 = xr.concat(datasets, dim='time')

        except Exception as e:
            fail_stats['concat_failed'] += 1
            if traj_idx < 5:
                print(f"\n[Traj {traj_idx}] Concat失败: {e}")
            continue

        # Prepare interpolation coordinate arrays
        lats = xr.DataArray(traj_df['latitude'], dims="points")
        lons = xr.DataArray(traj_df['longitude'], dims="points")
        times = xr.DataArray(traj_df['time'], dims="points")
        lons_360 = (lons + 360) % 360

        try:
            # Spatio-temporal interpolation on ERA5 dataset
            interpolated_wind = ds_era5[['u10', 'v10']].interp(
                lat=lats,
                lon=lons_360,
                time=times,
                method="linear"
            )

            u10 = interpolated_wind['u10'].values
            v10 = interpolated_wind['v10'].values
            traj_df['era5_u10'] = u10
            traj_df['era5_v10'] = v10

            # Feature engineering: wind speed (magnitude) and direction encoding
            wind_speed = np.sqrt(u10 ** 2 + v10 ** 2)
            traj_df['era5_wind_speed'] = wind_speed

            # Encode periodic wind direction into sine and cosine components
            wind_angle_rad = np.arctan2(v10, u10)
            traj_df['era5_wind_dir_sin'] = np.sin(wind_angle_rad)
            traj_df['era5_wind_dir_cos'] = np.cos(wind_angle_rad)

            # Check interpolation results
            n_nan = np.isnan(u10).sum()
            n_total = len(u10)

            traj_df.dropna(subset=['era5_u10', 'era5_v10'], inplace=True)

            if len(traj_df) == 0:
                fail_stats['all_nan'] += 1
                if traj_idx < 5:
                    print(f"\n[Traj {traj_idx}] 插值后全为NaN ({n_nan}/{n_total} NaN)")
                    print(f"  轨迹时间范围: {time_min} to {time_max}")
                    print(f"  ERA5时间范围: {ds_era5.time.values[0]} to {ds_era5.time.values[-1]}")
                continue
            elif len(traj_df) == 1:
                fail_stats['too_short'] += 1
                continue
            else:
                if n_nan > 0 and traj_idx < 3:
                    print(f"\n[Traj {traj_idx}] 部分NaN: {n_nan}/{n_total} ({100*n_nan/n_total:.1f}%)")
                fully_enriched_trajectories.append(traj_df)
                fail_stats['success'] += 1

        except Exception as e:
            fail_stats['interp_failed'] += 1
            print(f"\n[Traj {traj_idx}] 插值错误: {e}")
            print(f"  轨迹时间范围: {time_min} to {time_max}")
            continue

        finally:
            # Cleanup
            try:
                ds_era5.close()
            except Exception:
                pass
            del ds_era5, lats, lons, times, lons_360

    print(f"插值完成。共 {len(fully_enriched_trajectories)} 段轨迹获得了风场数据。")

    # Print detailed failure statistics
    print("\n=== 处理统计 ===")
    print(f"总轨迹数: {len(trajectories_with_currents)}")
    print(f"成功: {fail_stats['success']}")
    print(f"失败分解:")
    print(f"  - 未找到对应ERA5文件: {fail_stats['no_files']}")
    print(f"  - ERA5文件加载失败: {fail_stats['load_failed']}")
    print(f"  - 多文件concat失败: {fail_stats['concat_failed']}")
    print(f"  - 坐标标准化失败: {fail_stats['coord_failed']}")
    print(f"  - 空间裁剪失败: {fail_stats['crop_failed']}")
    print(f"  - 插值失败: {fail_stats['interp_failed']}")
    print(f"  - 插值后全为NaN: {fail_stats['all_nan']}")
    print(f"  - 有效点数<=1: {fail_stats['too_short']}")
    print(f"总失败: {sum(fail_stats.values()) - fail_stats['success']}")
    print("================\n")

    # --- 步骤 4/4: 保存结果 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if sample_mode:
        output_filename = os.path.join(output_dir, 'trajectories_with_currents_and_wind_samples.pkl')
    else:
        output_filename = os.path.join(output_dir, 'trajectories_with_currents_and_wind.pkl')

    print(f"\n步骤 4/4: 将富含风场数据的结果保存到: {output_filename}")
    with open(output_filename, 'wb') as f:
        pickle.dump(fully_enriched_trajectories, f)

    print("--- ERA5风场数据匹配完成！---")

    if fully_enriched_trajectories:
        print(f"\n最终产出是一个Python列表，其中包含 {len(fully_enriched_trajectories)} 个pandas.DataFrame。")
        print("每个DataFrame现在都包含了浮标、CFS海流、ERA5风场以及风场衍生特征。")
        print("\n第一个轨迹的头部数据示例:")
        print(fully_enriched_trajectories[0].head())
        print("\n查看新增的列名:")
        print(fully_enriched_trajectories[0].columns.tolist())
    else:
        print("\n警告: 没有轨迹成功匹配风场数据。请检查输入数据和时间范围。")


if __name__ == '__main__':
    # --- 用户配置 ---
    PROCESSED_BUOY_FILE_WITH_CURRENTS = '../../processed_data/trajectories_with_cfsv2_currents.pkl'
    ERA5_DATA_DIRECTORY = '../../reanalysis/wind'
    OUTPUT_DIRECTORY = '../../processed_data'

    # --- 采样模式配置 ---
    SAMPLE_MODE = False          # 设置为 True 进行快速验证，False 进行完整处理
    SAMPLE_SIZE = 10             # 采样轨迹数量（最短的N条轨迹）

    # --- 运行脚本 ---
    if not os.path.exists(PROCESSED_BUOY_FILE_WITH_CURRENTS):
        print(f"错误: 输入的浮标文件 '{PROCESSED_BUOY_FILE_WITH_CURRENTS}' 不存在。请先运行CFS流场匹配脚本。")
    elif not os.path.exists(ERA5_DATA_DIRECTORY):
        print(f"错误: ERA5数据目录 '{ERA5_DATA_DIRECTORY}' 不存在。请检查路径。")
    else:
        match_era5_wind(PROCESSED_BUOY_FILE_WITH_CURRENTS, ERA5_DATA_DIRECTORY, OUTPUT_DIRECTORY,
                        sample_mode=SAMPLE_MODE, sample_size=SAMPLE_SIZE)
