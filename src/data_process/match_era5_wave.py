import xarray as xr
import pandas as pd
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm
from scipy.interpolate import NearestNDInterpolator


def match_era5_wave(processed_buoy_file_with_wind, era5_wave_dir, output_dir, sample_mode=False, sample_size=10):
    """
    Matches ERA5 reanalysis wave data with buoy trajectories using serial processing.

    For each trajectory point, performs spatio-temporal interpolation on the ERA5 dataset
    to find the corresponding wave parameters (swh, mwp, mwd).

    Feature engineering:
    - era5_swh: Significant wave height
    - era5_mwp: Mean wave period
    - era5_wave_dir_sin, era5_wave_dir_cos: Periodic wave direction encoding

    Args:
        processed_buoy_file_with_wind (str): Path to trajectories with wind data
        era5_wave_dir (str): Directory containing ERA5 wave NetCDF files
        output_dir (str): Directory to save output
        sample_mode (bool): If True, only process shortest trajectories for quick validation
        sample_size (int): Number of shortest trajectories to process in sample mode
    """
    mode_info = "【采样验证模式】" if sample_mode else "【完整处理模式】"
    print(f"--- 开始匹配ERA5波浪数据 {mode_info} (串行处理) ---")

    # --- 步骤 1/4: 加载已匹配海流和风场的浮标轨迹 ---
    print(f"步骤 1/4: 加载浮标数据从: {processed_buoy_file_with_wind}")
    try:
        with open(processed_buoy_file_with_wind, 'rb') as f:
            trajectories_with_wind = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 浮标数据文件未找到 at '{processed_buoy_file_with_wind}'")
        return
    if not trajectories_with_wind:
        print("错误: 加载的浮标轨迹列表为空，无法继续。")
        return

    print(f"加载了 {len(trajectories_with_wind)} 段已匹配海流和风场的轨迹。")

    # Sample mode: select shortest trajectories for quick validation
    if sample_mode:
        trajectories_with_wind = sorted(trajectories_with_wind, key=len)
        trajectories_with_wind = trajectories_with_wind[:sample_size]
        print(f"采样模式: 选择最短的 {len(trajectories_with_wind)} 条轨迹，长度: {[len(t) for t in trajectories_with_wind]}")

    # --- 步骤 2/4: 检查ERA5波浪数据 ---
    print("步骤 2/4: 检查ERA5波浪数据...")
    era5_all_files = sorted(glob.glob(os.path.join(era5_wave_dir, '*.nc')))
    if not era5_all_files:
        print(f"错误: ERA5波浪数据目录 '{era5_wave_dir}' 中未找到 .nc 文件。")
        return
    print(f"找到 {len(era5_all_files)} 个ERA5波浪文件。")

    # --- 步骤 3/4: 串行处理各轨迹 ---
    print("步骤 3/4: 逐条处理轨迹并进行时空插值...")
    final_trajectories = []
    diagnosed_era5_coords = False  # 用于首次打印ERA5波浪坐标范围（诊断用）

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

    for traj_idx, traj_df in enumerate(tqdm(trajectories_with_wind, desc="处理轨迹中")):
        traj_df = traj_df.copy()

        # Determine which months this trajectory spans
        time_min = traj_df['time'].min()
        time_max = traj_df['time'].max()

        # Filter files to only those covering the trajectory's time range
        era5_wave_files = []
        for f in era5_all_files:
            basename = os.path.basename(f)
            if basename.startswith('wave_') and basename.endswith('.nc'):
                try:
                    file_yyyymm = basename.split('_')[1][:6]  # Extract YYYYMM
                    file_date = pd.Timestamp(year=int(file_yyyymm[:4]), month=int(file_yyyymm[4:6]), day=1)
                    if file_date <= time_max + pd.Timedelta(days=1) and \
                       file_date + pd.DateOffset(months=1) > time_min - pd.Timedelta(days=1):
                        era5_wave_files.append(f)
                except (ValueError, IndexError):
                    continue

        if not era5_wave_files:
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
            for f in era5_wave_files:
                try:
                    ds = xr.open_dataset(f)

                    # === Step 1: Handle wave-specific time format ===
                    # Rename 'valid_time' to 'time' if present
                    # 注意：新版 CDS API 下载的 ERA5 文件同时含有 'time'（预报参考时间）
                    # 和 'valid_time'（实际有效时间）两个坐标。直接 rename 会因目标名已存在而报错。
                    if 'valid_time' in ds.coords:
                        if 'time' in ds.dims or 'time' in ds.coords:
                            # time 维度/坐标已存在，valid_time 只是辅助标注，丢弃即可
                            ds = ds.drop_vars('valid_time')
                        else:
                            # valid_time 就是主时间维度，重命名为 time
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

                    # === Step 2: Standardize coordinates on single file (memory-efficient) ===
                    # Rename coordinates if needed
                    if 'latitude' in ds.dims and 'lat' not in ds.dims:
                        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

                    # Standardize longitude to 0-360 range
                    # 用 float() 确保是标量比较，避免 xarray DataArray 比较在某些版本下行为异常
                    if float(ds.lon.min()) < 0:
                        ds['lon'] = (ds['lon'] + 360) % 360
                        ds = ds.sortby('lon')

                    # Sort latitude ascending (ERA5 is descending)
                    # This sortby on single file (~2-3 GB) is memory-safe
                    if float(ds.lat[0]) > float(ds.lat[-1]):
                        ds = ds.sortby('lat')

                    # === 首次成功加载时，打印ERA5波浪坐标信息（诊断用，只打印一次） ===
                    if not diagnosed_era5_coords:
                        print(f"\n[诊断] ERA5波浪文件坐标范围 ({os.path.basename(f)}):")
                        print(f"  经度范围(标准化后): {float(ds.lon.min()):.2f} ~ {float(ds.lon.max()):.2f}")
                        print(f"  纬度范围(标准化后): {float(ds.lat.min()):.2f} ~ {float(ds.lat.max()):.2f}")
                        print(f"  可用变量: {list(ds.data_vars.keys())}")
                        diagnosed_era5_coords = True

                    # === Step 3: Remove duplicate timestamps BEFORE slicing ===
                    # 同风场脚本：isel 去重后必须 assign_coords 强制刷新内部 pandas Index，
                    # 否则 .sel(time=slice()) 仍会报 non-unique label 错误。
                    _, unique_indices = np.unique(ds.time.values, return_index=True)
                    if len(unique_indices) < len(ds.time):
                        ds = ds.isel(time=np.sort(unique_indices))
                        ds = ds.assign_coords(time=('time', ds.time.values))

                    # === Step 4: Time cropping ===
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

                    # === Step 5: Spatial cropping (CRITICAL - reduces memory before concat) ===
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

                    # 关键：关闭文件句柄前，必须先将裁剪后的数据强制加载到内存。
                    # ERA5 波浪数据纬度本来就是升序，sortby('lat') 不会被调用，
                    # 因此 ds 始终是原始文件对象。ds.close() 会真正关闭底层文件句柄，
                    # 导致 ds_cropped（lazy 视图）后续无法读取任何数据，interp 全为 NaN。
                    # 此时 ds_cropped 已经过时空双重裁剪，数据量极小（< 几 MB），
                    # 显式 load() 是安全的，不会造成 OOM。
                    ds_cropped = ds_cropped.load()
                    datasets.append(ds_cropped)
                    ds.close()

                except Exception as e:
                    load_errors.append((os.path.basename(f), str(e)))
                    continue

            if not datasets:
                fail_stats['load_failed'] += 1
                if traj_idx < 5 or len(load_errors) > 0:
                    print(f"\n[Traj {traj_idx}] 无法加载任何ERA5波浪文件。时间范围: {time_min} to {time_max}")
                    if load_errors:
                        print(f"  加载错误: {load_errors[:3]}")
                continue

            # Concatenate already-cropped small datasets (total ~hundreds of MB, no OOM)
            ds_era5_wave = xr.concat(datasets, dim='time')

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
            wave_vars = ['swh', 'mwp', 'mwd']

            # === Coast-fill：海岸外推 ===
            # ERA5 波浪模型只在海洋格点有值，陆地/海冰格点为 NaN。
            # 当浮标位于海岸线、小岛或极地海冰附近时，周围所有 ERA5 格点均为
            # NaN，导致线性插值（以及 ocean-masked 加权插值）完全失败（全 NaN）。
            #
            # 修复方案：在插值前，对每个时间步用 NearestNDInterpolator 将 NaN
            # 格点替换为最近有效海洋格点的值（最近邻外推）。填补后的网格不含 NaN，
            # 再做普通线性插值可以得到有效结果。
            #
            # 注意：coast-fill 只影响真实陆地/海冰格点，不修改有效海洋数据。
            # 填补值只用于最终的三线性插值权重中，不会引入额外误差（浮标坐标
            # 已在海面，插值权重主要来自周边真实海洋格点）。
            lat_2d, lon_2d = np.meshgrid(
                ds_era5_wave.lat.values, ds_era5_wave.lon.values, indexing='ij'
            )
            coast_filled = {}
            for var in wave_vars:
                data_orig = ds_era5_wave[var].values  # shape: (time, lat, lon)
                data_filled = data_orig.copy()
                for t_idx in range(data_orig.shape[0]):
                    slc = data_orig[t_idx]
                    nan_mask = np.isnan(slc)
                    if not nan_mask.any():
                        continue  # 该时间步无 NaN，跳过
                    valid_mask = ~nan_mask
                    if not valid_mask.any():
                        continue  # 全 NaN（极端情况），跳过
                    nn_interp = NearestNDInterpolator(
                        np.column_stack([lat_2d[valid_mask], lon_2d[valid_mask]]),
                        slc[valid_mask]
                    )
                    data_filled[t_idx][nan_mask] = nn_interp(
                        lat_2d[nan_mask], lon_2d[nan_mask]
                    )
                coast_filled[var] = xr.DataArray(
                    data_filled,
                    dims=ds_era5_wave[var].dims,
                    coords=ds_era5_wave[var].coords
                )
            ds_era5_wave = xr.Dataset(coast_filled)

            # === 普通线性插值（coast-fill 后无 NaN 邻域，可直接插值）===
            wave_results = {}
            for var in wave_vars:
                interp_result = ds_era5_wave[var].interp(
                    lat=lats, lon=lons_360, time=times, method='linear'
                )
                wave_results[var] = interp_result.values

            # Add wave height and period directly
            traj_df['era5_swh'] = wave_results['swh']
            traj_df['era5_mwp'] = wave_results['mwp']

            # Feature engineering: encode periodic wave direction into sine and cosine components
            mwd_deg = wave_results['mwd']
            mwd_rad = np.deg2rad(mwd_deg)
            traj_df['era5_wave_dir_sin'] = np.sin(mwd_rad)
            traj_df['era5_wave_dir_cos'] = np.cos(mwd_rad)

            # Check interpolation results
            n_nan = int(np.isnan(wave_results['swh']).sum())
            n_total = len(wave_results['swh'])

            # Drop rows where interpolation failed
            # 在 dropna 前保留原始坐标信息，用于 all_nan 时的诊断打印
            orig_lat_min = traj_df['latitude'].min()
            orig_lat_max = traj_df['latitude'].max()
            orig_lon_min = traj_df['longitude'].min()
            orig_lon_max = traj_df['longitude'].max()
            traj_df.dropna(subset=['era5_swh', 'era5_mwp', 'era5_wave_dir_sin'], inplace=True)

            if len(traj_df) == 0:
                fail_stats['all_nan'] += 1
                if fail_stats['all_nan'] <= 10:
                    print(f"\n[Traj {traj_idx}] 插值后全为NaN ({n_nan}/{n_total} NaN)")
                    print(f"  轨迹时间范围: {time_min} to {time_max}")
                    print(f"  轨迹经纬度范围: lat=[{orig_lat_min:.2f}, {orig_lat_max:.2f}], "
                          f"lon=[{orig_lon_min:.2f}, {orig_lon_max:.2f}]")
                    print(f"  插值用 lon_360 范围: [{float(lons_360.min()):.2f}, {float(lons_360.max()):.2f}]")
                    print(f"  ERA5波浪 lon范围: [{float(ds_era5_wave.lon.min()):.2f}, {float(ds_era5_wave.lon.max()):.2f}]")
                    print(f"  ERA5波浪 lat范围: [{float(ds_era5_wave.lat.min()):.2f}, {float(ds_era5_wave.lat.max()):.2f}]")
                    print(f"  ERA5波浪时间范围: {ds_era5_wave.time.values[0]} to {ds_era5_wave.time.values[-1]}")
                    print(f"  ERA5时间dtype: {ds_era5_wave.time.dtype}  轨迹时间dtype: {times.dtype}")
                    print(f"  ERA5时间int64[0]: {ds_era5_wave.time.values[0].astype('int64')}  轨迹时间int64[0]: {times.values[0].astype('int64')}")
                continue
            elif len(traj_df) == 1:
                fail_stats['too_short'] += 1
                continue
            else:
                if n_nan > 0 and traj_idx < 3:
                    print(f"\n[Traj {traj_idx}] 部分NaN: {n_nan}/{n_total} ({100*n_nan/n_total:.1f}%)")
                final_trajectories.append(traj_df)
                fail_stats['success'] += 1

        except Exception as e:
            fail_stats['interp_failed'] += 1
            print(f"\n[Traj {traj_idx}] 插值错误: {e}")
            print(f"  轨迹时间范围: {time_min} to {time_max}")
            continue

        finally:
            # Cleanup
            try:
                ds_era5_wave.close()
            except Exception:
                pass
            try:
                del ds_era5_wave, lats, lons, times, lons_360
            except Exception:
                pass
            try:
                del coast_filled, lat_2d, lon_2d
            except Exception:
                pass

    print(f"插值完成。共 {len(final_trajectories)} 段轨迹获得了波浪数据。")

    # Print detailed failure statistics
    print("\n=== 处理统计 ===")
    print(f"总轨迹数: {len(trajectories_with_wind)}")
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

    # --- 步骤 4/4: 保存最终结果 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if sample_mode:
        output_filename = os.path.join(output_dir, 'trajectories_with_all_features_samples.pkl')
    else:
        output_filename = os.path.join(output_dir, 'trajectories_with_all_features.pkl')

    print(f"\n步骤 4/4: 将包含所有特征的最终数据集保存到: {output_filename}")
    with open(output_filename, 'wb') as f:
        pickle.dump(final_trajectories, f)

    print("\n" + "=" * 80)
    print("--- 所有数据预处理和特征工程步骤已全部完成！---")
    print("=" * 80)

    if final_trajectories:
        print(f"\n最终产出文件 '{output_filename}' 是一个Python列表。")
        print("列表中的每个DataFrame都包含了构建深度学习模型所需的全部输入特征:")
        print("  - 浮标观测数据 (ID, time, lat, lon, ve, vn)")
        print("  - CFS背景海流 (cfsv2_u, cfsv2_v)")
        print("  - ERA5背景风场 (era5_u10, era5_v10)及其衍生特征 (speed, dir_sin, dir_cos)")
        print("  - ERA5背景波浪 (era5_swh, era5_mwp)及其衍生特征 (dir_sin, dir_cos)")
        print("\n第一个轨迹的头部数据示例:")
        print(final_trajectories[0].head())
        print("\n最终数据集的完整列名:")
        print(final_trajectories[0].columns.tolist())
    else:
        print("\n警告: 没有轨迹成功匹配波浪数据。请检查输入数据和时间范围。")


if __name__ == '__main__':
    # --- 用户配置 ---
    # 1. 上一步生成的、已匹配海流和风场的文件
    PROCESSED_BUOY_FILE_WITH_WIND = '../../processed_data/trajectories_with_currents_and_wind.pkl'

    # 2. 存放所有ERA5波浪NetCDF文件的目录
    ERA5_WAVE_DATA_DIRECTORY = '../../reanalysis/wave'

    # 3. 输出目录
    OUTPUT_DIRECTORY = '../../processed_data'

    # --- 采样模式配置 ---
    SAMPLE_MODE = False          # 设置为 True 进行快速验证，False 进行完整处理
    SAMPLE_SIZE = 15             # 采样轨迹数量（最短的N条轨迹）

    # --- 运行脚本 ---
    if not os.path.exists(PROCESSED_BUOY_FILE_WITH_WIND):
        print(f"错误: 输入文件 '{PROCESSED_BUOY_FILE_WITH_WIND}' 不存在。请先运行风场匹配脚本。")
    elif not os.path.exists(ERA5_WAVE_DATA_DIRECTORY):
        print(f"错误: ERA5波浪数据目录 '{ERA5_WAVE_DATA_DIRECTORY}' 不存在。请检查路径。")
    else:
        match_era5_wave(PROCESSED_BUOY_FILE_WITH_WIND, ERA5_WAVE_DATA_DIRECTORY, OUTPUT_DIRECTORY,
                        sample_mode=SAMPLE_MODE, sample_size=SAMPLE_SIZE)
