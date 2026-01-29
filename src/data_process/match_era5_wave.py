import xarray as xr
import pandas as pd
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm


def match_era5_wave(processed_buoy_file_with_wind, era5_wave_dir, output_dir):
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
    """
    print("--- 开始匹配ERA5波浪数据 (串行处理模式) ---")

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

    for traj_df in tqdm(trajectories_with_wind, desc="处理轨迹中"):
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
                    # Include file if it overlaps with trajectory's time range (with buffer)
                    if file_date <= time_max + pd.Timedelta(days=1) and \
                       file_date + pd.DateOffset(months=1) > time_min - pd.Timedelta(days=1):
                        era5_wave_files.append(f)
                except (ValueError, IndexError):
                    continue

        if not era5_wave_files:
            continue

        try:
            # Load and concatenate wave files for the trajectory's time range
            def select_time_range_and_rename(ds):
                # Rename 'valid_time' to 'time' to match other datasets and avoid conflicts
                if 'valid_time' in ds.coords:
                    ds = ds.rename({'valid_time': 'time'})
                selected = ds.sel(time=slice(time_min - pd.Timedelta(days=1), time_max + pd.Timedelta(days=1)))
                if len(selected.time) == 0:
                    return None
                return selected

            datasets = []
            for f in era5_wave_files:
                try:
                    ds = xr.open_dataset(f)
                    ds_sel = select_time_range_and_rename(ds)
                    if ds_sel is not None:
                        datasets.append(ds_sel)
                    ds.close()
                except Exception:
                    continue

            if not datasets:
                continue

            ds_era5_wave_raw = xr.concat(datasets, dim='time')

        except Exception:
            continue

        # Standardize coordinate names
        ds_era5_wave = ds_era5_wave_raw.rename({'latitude': 'lat', 'longitude': 'lon'})

        # Ensure longitude is in 0-360 range
        if ds_era5_wave.lon.min() < 0:
            ds_era5_wave['lon'] = (ds_era5_wave['lon'] + 360) % 360
        ds_era5_wave = ds_era5_wave.sortby('lon')

        # Prepare interpolation coordinate arrays
        lats = xr.DataArray(traj_df['latitude'], dims="points")
        lons = xr.DataArray(traj_df['longitude'], dims="points")
        times = xr.DataArray(traj_df['time'], dims="points")
        lons_360 = (lons + 360) % 360

        try:
            # Spatio-temporal interpolation on ERA5 wave dataset
            interpolated_wave = ds_era5_wave[['swh', 'mwp', 'mwd']].interp(
                lat=lats,
                lon=lons_360,
                time=times,
                method="linear"
            )

            # Add wave height and period directly
            traj_df['era5_swh'] = interpolated_wave['swh'].values
            traj_df['era5_mwp'] = interpolated_wave['mwp'].values

            # Feature engineering: encode periodic wave direction into sine and cosine components
            mwd_deg = interpolated_wave['mwd'].values
            # Convert degrees to radians for trigonometric functions
            mwd_rad = np.deg2rad(mwd_deg)
            traj_df['era5_wave_dir_sin'] = np.sin(mwd_rad)
            traj_df['era5_wave_dir_cos'] = np.cos(mwd_rad)

            # Drop rows where interpolation failed
            traj_df.dropna(subset=['era5_swh', 'era5_mwp', 'era5_wave_dir_sin'], inplace=True)

            if len(traj_df) > 1:
                final_trajectories.append(traj_df)

        except Exception:
            pass

        finally:
            # Cleanup
            try:
                ds_era5_wave.close()
                ds_era5_wave_raw.close()
            except Exception:
                pass
            del ds_era5_wave_raw, ds_era5_wave, lats, lons, times, lons_360

    print(f"处理完成。共 {len(final_trajectories)} 段轨迹获得了波浪数据。")

    # --- 步骤 4/4: 保存最终结果 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # --- 运行脚本 ---
    if not os.path.exists(PROCESSED_BUOY_FILE_WITH_WIND):
        print(f"错误: 输入文件 '{PROCESSED_BUOY_FILE_WITH_WIND}' 不存在。请先运行风场匹配脚本。")
    elif not os.path.exists(ERA5_WAVE_DATA_DIRECTORY):
        print(f"错误: ERA5波浪数据目录 '{ERA5_WAVE_DATA_DIRECTORY}' 不存在。请检查路径。")
    else:
        match_era5_wave(PROCESSED_BUOY_FILE_WITH_WIND, ERA5_WAVE_DATA_DIRECTORY, OUTPUT_DIRECTORY)
