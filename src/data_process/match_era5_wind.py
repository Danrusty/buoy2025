import xarray as xr
import pandas as pd
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm


def match_era5_wind(processed_buoy_file_with_currents, era5_dir, output_dir):
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
    """
    print("--- 开始匹配ERA5风场数据 (串行处理模式) ---")

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

    # --- 步骤 2/4: 检查ERA5数据 ---
    print("步骤 2/4: 检查ERA5风场数据...")
    era5_all_files = sorted(glob.glob(os.path.join(era5_dir, '*.nc')))
    if not era5_all_files:
        print(f"错误: ERA5数据目录 '{era5_dir}' 中未找到 .nc 文件。")
        return
    print(f"找到 {len(era5_all_files)} 个ERA5风场文件。")

    # --- 步骤 3/4: 串行处理各轨迹 ---
    print("步骤 3/4: 逐条处理轨迹并进行时空插值...")
    fully_enriched_trajectories = []

    for traj_df in tqdm(trajectories_with_currents, desc="处理轨迹中"):
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
                    # Include file if it overlaps with trajectory's time range (with buffer)
                    if file_date <= time_max + pd.Timedelta(days=1) and \
                       file_date + pd.DateOffset(months=1) > time_min - pd.Timedelta(days=1):
                        era5_files.append(f)
                except (ValueError, IndexError):
                    continue

        if not era5_files:
            continue

        try:
            # Load and concatenate wind files for the trajectory's time range
            def select_time_range(ds):
                selected = ds.sel(time=slice(time_min - pd.Timedelta(days=1), time_max + pd.Timedelta(days=1)))
                if len(selected.time) == 0:
                    return None
                return selected

            datasets = []
            for f in era5_files:
                try:
                    ds = xr.open_dataset(f)
                    ds_sel = select_time_range(ds)
                    if ds_sel is not None:
                        datasets.append(ds_sel)
                    ds.close()
                except Exception:
                    continue

            if not datasets:
                continue

            ds_era5_raw = xr.concat(datasets, dim='time')

        except Exception:
            continue

        # ERA5 coordinates are already named 'lon' and 'lat'
        ds_era5 = ds_era5_raw

        # Standardize longitude to 0-360 range
        if ds_era5.lon.min() < 0:
            ds_era5['lon'] = (ds_era5['lon'] + 360) % 360
        ds_era5 = ds_era5.sortby('lon')

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

            traj_df.dropna(subset=['era5_u10', 'era5_v10'], inplace=True)

            if len(traj_df) > 1:
                fully_enriched_trajectories.append(traj_df)

        except Exception:
            pass

        finally:
            # Cleanup
            try:
                ds_era5.close()
                ds_era5_raw.close()
            except Exception:
                pass
            del ds_era5_raw, ds_era5, lats, lons, times, lons_360

    print(f"处理完成。共 {len(fully_enriched_trajectories)} 段轨迹获得了风场数据。")

    # --- 步骤 4/4: 保存结果 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        print(fully_enriched_trajectories[0].columns)
    else:
        print("\n警告: 没有轨迹成功匹配风场数据。请检查输入数据和时间范围。")


if __name__ == '__main__':
    # --- 用户配置 ---
    PROCESSED_BUOY_FILE_WITH_CURRENTS = '../../processed_data/trajectories_with_cfsv2_currents.pkl'
    ERA5_DATA_DIRECTORY = '../../reanalysis/wind'
    OUTPUT_DIRECTORY = '../../processed_data'

    # --- 运行脚本 ---
    if not os.path.exists(PROCESSED_BUOY_FILE_WITH_CURRENTS):
        print(f"错误: 输入的浮标文件 '{PROCESSED_BUOY_FILE_WITH_CURRENTS}' 不存在。请先运行CFS流场匹配脚本。")
    elif not os.path.exists(ERA5_DATA_DIRECTORY):
        print(f"错误: ERA5数据目录 '{ERA5_DATA_DIRECTORY}' 不存在。请检查路径。")
    else:
        match_era5_wind(PROCESSED_BUOY_FILE_WITH_CURRENTS, ERA5_DATA_DIRECTORY, OUTPUT_DIRECTORY)
