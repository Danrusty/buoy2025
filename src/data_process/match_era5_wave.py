import xarray as xr
import pandas as pd
import numpy as np
import os
import pickle
import glob
import gc
import json
from tqdm import tqdm
from scipy.interpolate import NearestNDInterpolator

from wave_direction import (
    NEAR_ZERO_THRESHOLD,
    coast_fill_mwd_components,
    normalize_direction_components,
)


def _wave_angles_degrees(trajectory):
    """从现有 sin/cos 特征恢复 0~360 度角，仅用于 v1/v2 诊断比较。"""
    required = {'era5_wave_dir_sin', 'era5_wave_dir_cos'}
    if not required.issubset(trajectory.columns):
        return None
    angles = np.rad2deg(np.arctan2(
        trajectory['era5_wave_dir_sin'].to_numpy(),
        trajectory['era5_wave_dir_cos'].to_numpy(),
    ))
    return np.mod(angles, 360.0)


def select_representative_trajectories(
    trajectories,
    sample_size=50,
    random_seed=42,
):
    """选择长度分层、波向边界、波向跳变和固定随机轨迹的去重组合。"""
    if sample_size >= len(trajectories):
        return list(trajectories), list(range(len(trajectories))), {
            str(index): ['all'] for index in range(len(trajectories))
        }

    lengths = np.asarray([len(trajectory) for trajectory in trajectories])
    if sample_size <= 10:
        quantile_targets = np.quantile(lengths, [0.0, 0.5, 1.0])
        direction_group_size = 1
    else:
        group_size = max(1, sample_size // 5)
        quantile_targets = np.quantile(
            lengths,
            np.linspace(0.0, 1.0, group_size),
        )
        direction_group_size = group_size
    length_quantiles = []
    for target in quantile_targets:
        candidates = np.argsort(np.abs(lengths - target))
        selected = next(
            int(index) for index in candidates
            if int(index) not in length_quantiles
        )
        length_quantiles.append(selected)

    boundary_scores = np.full(len(trajectories), -np.inf)
    jump_scores = np.full(len(trajectories), -np.inf)
    for index, trajectory in enumerate(trajectories):
        angles = _wave_angles_degrees(trajectory)
        if angles is None or len(angles) == 0:
            continue
        distance_to_north = np.minimum(angles, 360.0 - angles)
        boundary_scores[index] = float(np.mean(distance_to_north <= 10.0))
        if len(angles) > 1:
            jump_scores[index] = float(np.max(np.abs(np.diff(angles))))

    boundary = np.argsort(
        boundary_scores
    )[-direction_group_size:][::-1].tolist()
    jumps = np.argsort(
        jump_scores
    )[-direction_group_size:][::-1].tolist()

    reasons = {}

    def add(indices, reason):
        for index in indices:
            reasons.setdefault(int(index), []).append(reason)

    add(length_quantiles, 'length_quantile')
    add(boundary, 'near_0_360')
    add(jumps, 'large_angle_jump')

    rng = np.random.default_rng(random_seed)
    remaining = [
        index for index in range(len(trajectories))
        if index not in reasons
    ]
    rng.shuffle(remaining)
    add(remaining[: max(0, sample_size - len(reasons))], 'random')

    # 组间可能重叠；继续用固定随机序列补足到准确的 sample_size。
    if len(reasons) < sample_size:
        fallback = [
            index for index in range(len(trajectories))
            if index not in reasons
        ]
        rng.shuffle(fallback)
        add(fallback[: sample_size - len(reasons)], 'random_fill')

    selected_indices = sorted(reasons)[:sample_size]
    selected = [trajectories[index] for index in selected_indices]
    selected_reasons = {
        str(index): reasons[index] for index in selected_indices
    }
    return selected, selected_indices, selected_reasons


def _circular_angle_difference_degrees(old_sin, old_cos, new_sin, new_cos):
    """返回两个单位方向向量之间的最短夹角，范围 0~180 度。"""
    cross = old_cos * new_sin - old_sin * new_cos
    dot = old_sin * new_sin + old_cos * new_cos
    return np.abs(np.rad2deg(np.arctan2(cross, dot)))


def match_era5_wave(
    processed_buoy_file_with_wind,
    era5_wave_dir,
    output_dir,
    sample_mode=False,
    sample_size=10,
    output_filename=None,
):
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
        sample_mode (bool): 是否只处理分层代表性轨迹用于快速验证
        sample_size (int): 采样模式下的代表性轨迹数量
    """
    mode_info = "【采样验证模式】" if sample_mode else "【完整处理模式】"
    print(f"--- 开始匹配ERA5波浪数据 {mode_info} (串行处理) ---")

    # --- 步骤 1/4: 加载已匹配海流和风场的浮标轨迹 ---
    print(f"步骤 1/4: 加载浮标数据从: {processed_buoy_file_with_wind}")
    try:
        with open(processed_buoy_file_with_wind, 'rb') as f:
            trajectories_with_wind = pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 未找到浮标数据文件 '{processed_buoy_file_with_wind}'")
        return
    if not trajectories_with_wind:
        print("错误: 加载的浮标轨迹列表为空，无法继续。")
        return

    print(f"加载了 {len(trajectories_with_wind)} 段已匹配海流和风场的轨迹。")

    selected_indices = list(range(len(trajectories_with_wind)))
    selected_reasons = {
        str(index): ['full'] for index in selected_indices
    }

    # 采样模式选择具有代表性的轨迹用于验证。
    if sample_mode:
        (
            trajectories_with_wind,
            selected_indices,
            selected_reasons,
        ) = select_representative_trajectories(
            trajectories_with_wind,
            sample_size=sample_size,
        )
        gc.collect()
        print(
            f"采样模式: 选择 {len(trajectories_with_wind)} 条代表性轨迹，"
            f"原始索引: {selected_indices}"
        )

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

    # 失败类型统计
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
    circular_stats = {
        'near_zero_count': 0,
        'below_0_1_count': 0,
        'minimum_resultant_length': float('inf'),
        'finite_resultant_count': 0,
        'near_zero_examples': [],
        'angle_differences': [],
    }

    for traj_idx, traj_df in enumerate(tqdm(trajectories_with_wind, desc="处理轨迹中")):
        traj_df = traj_df.copy()

        # 确定当前轨迹覆盖的月份。
        time_min = traj_df['time'].min()
        time_max = traj_df['time'].max()

        # 只保留时间范围与当前轨迹相交的文件。
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

        # --- 加载文件前计算空间范围 ---
        lat_min = float(traj_df['latitude'].min() - 1)
        lat_max = float(traj_df['latitude'].max() + 1)
        lon_min = float((traj_df['longitude'].min() - 1 + 360) % 360)
        lon_max = float((traj_df['longitude'].max() + 1 + 360) % 360)

        try:
            # 每个文件先加载、统一坐标并裁剪，再执行拼接。
            datasets = []
            load_errors = []
            for f in era5_wave_files:
                try:
                    ds = xr.open_dataset(f)

                    # === 步骤 1：处理波浪文件的时间格式 ===
                    # 存在 valid_time 时，将其统一为 time。
                    # 注意：新版 CDS API 下载的 ERA5 文件同时含有 'time'（预报参考时间）
                    # 和 'valid_time'（实际有效时间）两个坐标。直接 rename 会因目标名已存在而报错。
                    if 'valid_time' in ds.coords:
                        if 'time' in ds.dims or 'time' in ds.coords:
                            # time 维度/坐标已存在，valid_time 只是辅助标注，丢弃即可
                            ds = ds.drop_vars('valid_time')
                        else:
                            # valid_time 就是主时间维度，重命名为 time
                            ds = ds.rename({'valid_time': 'time'})

                    # 将浮点 YYYYMMDD.fraction 时间转换为 datetime64。
                    if ds.time.dtype == np.float64 or ds.time.dtype == np.float32:
                        time_float = ds.time.values.astype(np.float64)
                        date_ints = time_float.astype(np.int64)
                        fracs = time_float - date_ints
                        hours = np.round(fracs * 24).astype(int)
                        base_dates = pd.to_datetime(date_ints, format='%Y%m%d')
                        datetime_index = base_dates + pd.to_timedelta(hours, unit='h')
                        ds['time'] = datetime_index.values

                    # 统一时间精度为 datetime64[ns]（与轨迹数据一致）。
                    # 新版 CDS API 下载的 ERA5 文件时间编码为 datetime64[us]（微秒），
                    # 而轨迹数据为 datetime64[ns]（纳秒）。xr.interp 将 datetime64 转为
                    # float64 时直接取底层整数值：us 量级约 1.66e15，ns 量级约 1.66e18，
                    # 两者相差 1000 倍，导致插值点完全超出 ERA5 时间轴范围，全部返回 NaN。
                    if ds.time.dtype != np.dtype('datetime64[ns]'):
                        ds['time'] = ds.time.values.astype('datetime64[ns]')

                    # === 步骤 2：在单文件内统一坐标，控制内存占用 ===
                    # 必要时统一坐标名称。
                    if 'latitude' in ds.dims and 'lat' not in ds.dims:
                        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})

                    # 将经度统一到 0~360 度。
                    # 用 float() 确保是标量比较，避免 xarray DataArray 比较在某些版本下行为异常
                    if float(ds.lon.min()) < 0:
                        ds['lon'] = (ds['lon'] + 360) % 360
                        ds = ds.sortby('lon')

                    # ERA5 纬度通常为降序，这里统一为升序。
                    # 对单个文件执行 sortby，避免多文件拼接后产生额外内存峰值。
                    if float(ds.lat[0]) > float(ds.lat[-1]):
                        ds = ds.sortby('lat')

                    # === 首次成功加载时，打印ERA5波浪坐标信息（诊断用，只打印一次） ===
                    if not diagnosed_era5_coords:
                        print(f"\n[诊断] ERA5波浪文件坐标范围 ({os.path.basename(f)}):")
                        print(f"  经度范围(标准化后): {float(ds.lon.min()):.2f} ~ {float(ds.lon.max()):.2f}")
                        print(f"  纬度范围(标准化后): {float(ds.lat.min()):.2f} ~ {float(ds.lat.max()):.2f}")
                        print(f"  可用变量: {list(ds.data_vars.keys())}")
                        diagnosed_era5_coords = True

                    # === 步骤 3：裁剪前删除重复时间戳 ===
                    # 同风场脚本：isel 去重后必须 assign_coords 强制刷新内部 pandas Index，
                    # 否则 .sel(time=slice()) 仍会报 non-unique label 错误。
                    _, unique_indices = np.unique(ds.time.values, return_index=True)
                    if len(unique_indices) < len(ds.time):
                        ds = ds.isel(time=np.sort(unique_indices))
                        ds = ds.assign_coords(time=('time', ds.time.values))

                    # === 步骤 4：按时间裁剪 ===
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

                    # === 步骤 5：按空间裁剪，降低拼接前的内存占用 ===
                    if lon_max < lon_min:
                        # 轨迹跨越日期变更线，拆分为两个经度区域。
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
                    print(f"\n[轨迹 {traj_idx}] 无法加载任何 ERA5 波浪文件。时间范围: {time_min} 至 {time_max}")
                    if load_errors:
                        print(f"  加载错误: {load_errors[:3]}")
                continue

            # 拼接已裁剪的数据集，总量通常为数百 MB。
            ds_era5_wave = xr.concat(datasets, dim='time')

        except Exception as e:
            fail_stats['concat_failed'] += 1
            if traj_idx < 5:
                print(f"\n[轨迹 {traj_idx}] 数据拼接失败: {e}")
            continue

        # 构造插值坐标数组。
        lats = xr.DataArray(traj_df['latitude'], dims="points")
        lons = xr.DataArray(traj_df['longitude'], dims="points")
        times = xr.DataArray(traj_df['time'], dims="points")
        lons_360 = (lons + 360) % 360

        try:
            scalar_wave_vars = ['swh', 'mwp']
            old_direction = None
            if {
                'era5_wave_dir_sin',
                'era5_wave_dir_cos',
            }.issubset(traj_df.columns):
                old_direction = (
                    traj_df['era5_wave_dir_sin'].to_numpy(copy=True),
                    traj_df['era5_wave_dir_cos'].to_numpy(copy=True),
                )

            # === 海岸缺测填补（coast-fill）===
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
            for var in scalar_wave_vars:
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

            # mwd 是圆周变量，必须先编码为单位向量分量，再做缺测填补和插值。
            mwd_sin, mwd_cos = coast_fill_mwd_components(
                ds_era5_wave['mwd']
            )
            coast_filled['mwd_sin'] = mwd_sin
            coast_filled['mwd_cos'] = mwd_cos
            ds_era5_wave = xr.Dataset(coast_filled)

            # === 普通线性插值（coast-fill 后无 NaN 邻域，可直接插值）===
            wave_results = {}
            for var in ['swh', 'mwp', 'mwd_sin', 'mwd_cos']:
                interp_result = ds_era5_wave[var].interp(
                    lat=lats, lon=lons_360, time=times, method='linear'
                )
                wave_results[var] = interp_result.values

            # 直接写入有效波高和平均波周期。
            traj_df['era5_swh'] = wave_results['swh']
            traj_df['era5_mwp'] = wave_results['mwp']

            direction = normalize_direction_components(
                wave_results['mwd_sin'],
                wave_results['mwd_cos'],
                near_zero_threshold=NEAR_ZERO_THRESHOLD,
            )
            traj_df['era5_wave_dir_sin'] = direction.sin
            traj_df['era5_wave_dir_cos'] = direction.cos

            finite_resultant = direction.resultant_length[
                np.isfinite(direction.resultant_length)
            ]
            if finite_resultant.size:
                circular_stats['finite_resultant_count'] += int(
                    finite_resultant.size
                )
                circular_stats['minimum_resultant_length'] = min(
                    circular_stats['minimum_resultant_length'],
                    float(finite_resultant.min()),
                )
                circular_stats['below_0_1_count'] += int(
                    np.count_nonzero(finite_resultant < 0.1)
                )

            near_zero_positions = np.flatnonzero(direction.near_zero)
            circular_stats['near_zero_count'] += int(
                near_zero_positions.size
            )
            source_trajectory_index = selected_indices[traj_idx]
            for row_position in near_zero_positions:
                if len(circular_stats['near_zero_examples']) >= 20:
                    break
                row = traj_df.iloc[int(row_position)]
                circular_stats['near_zero_examples'].append({
                    'source_trajectory_index': int(source_trajectory_index),
                    'row_position': int(row_position),
                    'original_ID': str(row.get('original_ID', row.get('ID'))),
                    'time': str(row['time']),
                    'latitude': float(row['latitude']),
                    'longitude': float(row['longitude']),
                    'resultant_length': float(
                        direction.resultant_length[row_position]
                    ),
                })

            if old_direction is not None:
                angle_difference = _circular_angle_difference_degrees(
                    old_direction[0],
                    old_direction[1],
                    direction.sin,
                    direction.cos,
                )
                finite_difference = angle_difference[
                    np.isfinite(angle_difference)
                ]
                if finite_difference.size:
                    circular_stats['angle_differences'].append(
                        finite_difference.astype(np.float32, copy=False)
                    )

            # 检查插值结果。
            n_nan = int(np.isnan(wave_results['swh']).sum())
            n_total = len(wave_results['swh'])

            # 删除插值失败的记录。
            # 在 dropna 前保留原始坐标信息，用于 all_nan 时的诊断打印
            orig_lat_min = traj_df['latitude'].min()
            orig_lat_max = traj_df['latitude'].max()
            orig_lon_min = traj_df['longitude'].min()
            orig_lon_max = traj_df['longitude'].max()
            traj_df.dropna(
                subset=[
                    'era5_swh',
                    'era5_mwp',
                    'era5_wave_dir_sin',
                    'era5_wave_dir_cos',
                ],
                inplace=True,
            )

            if len(traj_df) == 0:
                fail_stats['all_nan'] += 1
                if fail_stats['all_nan'] <= 10:
                    print(f"\n[轨迹 {traj_idx}] 插值后全为 NaN ({n_nan}/{n_total})")
                    print(f"  轨迹时间范围: {time_min} 至 {time_max}")
                    print(f"  轨迹经纬度范围: lat=[{orig_lat_min:.2f}, {orig_lat_max:.2f}], "
                          f"lon=[{orig_lon_min:.2f}, {orig_lon_max:.2f}]")
                    print(f"  插值用 lon_360 范围: [{float(lons_360.min()):.2f}, {float(lons_360.max()):.2f}]")
                    print(f"  ERA5波浪 lon范围: [{float(ds_era5_wave.lon.min()):.2f}, {float(ds_era5_wave.lon.max()):.2f}]")
                    print(f"  ERA5波浪 lat范围: [{float(ds_era5_wave.lat.min()):.2f}, {float(ds_era5_wave.lat.max()):.2f}]")
                    print(
                        "  ERA5 波浪时间范围: "
                        f"{ds_era5_wave.time.values[0]} 至 "
                        f"{ds_era5_wave.time.values[-1]}"
                    )
                    print(f"  ERA5时间dtype: {ds_era5_wave.time.dtype}  轨迹时间dtype: {times.dtype}")
                    print(f"  ERA5时间int64[0]: {ds_era5_wave.time.values[0].astype('int64')}  轨迹时间int64[0]: {times.values[0].astype('int64')}")
                continue
            elif len(traj_df) == 1:
                fail_stats['too_short'] += 1
                continue
            else:
                if n_nan > 0 and traj_idx < 3:
                    print(
                        f"\n[轨迹 {traj_idx}] 部分为 NaN: "
                        f"{n_nan}/{n_total} ({100*n_nan/n_total:.1f}%)"
                    )
                final_trajectories.append(traj_df)
                fail_stats['success'] += 1

        except Exception as e:
            fail_stats['interp_failed'] += 1
            print(f"\n[轨迹 {traj_idx}] 插值错误: {e}")
            print(f"  轨迹时间范围: {time_min} 至 {time_max}")
            continue

        finally:
            # 释放当前轨迹使用的临时对象。
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

    # 输出详细的失败类型统计。
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

    angle_difference_summary = None
    if circular_stats['angle_differences']:
        angle_differences = np.concatenate(
            circular_stats.pop('angle_differences')
        )
        angle_difference_summary = {
            'finite_count': int(angle_differences.size),
            'changed_over_1_degree': int(np.count_nonzero(
                angle_differences > 1.0
            )),
            'changed_over_10_degrees': int(np.count_nonzero(
                angle_differences > 10.0
            )),
            'p50_degrees': float(np.percentile(angle_differences, 50)),
            'p90_degrees': float(np.percentile(angle_differences, 90)),
            'p99_degrees': float(np.percentile(angle_differences, 99)),
            'maximum_degrees': float(angle_differences.max()),
        }
    else:
        circular_stats.pop('angle_differences')

    if not np.isfinite(circular_stats['minimum_resultant_length']):
        circular_stats['minimum_resultant_length'] = None
    circular_stats['near_zero_threshold'] = NEAR_ZERO_THRESHOLD
    circular_stats['angle_difference_v1_vs_v2'] = angle_difference_summary

    print("=== 波向圆周插值统计 ===")
    print(f"有限结果数: {circular_stats['finite_resultant_count']}")
    print(
        "最小合成向量长度: "
        f"{circular_stats['minimum_resultant_length']}"
    )
    print(f"r < 0.1 数量: {circular_stats['below_0_1_count']}")
    print(
        f"r < {NEAR_ZERO_THRESHOLD:g} 近零数量: "
        f"{circular_stats['near_zero_count']}"
    )
    if angle_difference_summary:
        print(f"v1/v2 角度差: {angle_difference_summary}")
    print("==========================\n")

    # --- 步骤 4/4: 保存最终结果 ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if output_filename is None:
        if sample_mode:
            filename = (
                'trajectories_with_all_features_circular_mwd_v2_samples.pkl'
            )
        else:
            filename = 'trajectories_with_all_features_circular_mwd_v2.pkl'
        output_filename = os.path.join(output_dir, filename)
    elif not os.path.isabs(output_filename):
        output_filename = os.path.join(output_dir, output_filename)

    print(f"\n步骤 4/4: 将包含所有特征的最终数据集保存到: {output_filename}")
    with open(output_filename, 'wb') as f:
        pickle.dump(final_trajectories, f)

    diagnostics_path = os.path.splitext(output_filename)[0] + '_diagnostics.json'
    diagnostics = {
        'schema_version': 1,
        'direction_convention': {
            'source': 'ERA5 mean_wave_direction',
            'meaning': 'coming-from',
            'zero_direction': 'north',
            'positive_rotation': 'clockwise',
            'add_180_degrees': False,
            'sin_definition': 'sin(deg2rad(mwd))',
            'cos_definition': 'cos(deg2rad(mwd))',
        },
        'sample_mode': bool(sample_mode),
        'sample_size': int(len(trajectories_with_wind)),
        'selected_source_indices': [
            int(index) for index in selected_indices
        ],
        'selection_reasons': selected_reasons,
        'failure_statistics': fail_stats,
        'circular_statistics': circular_stats,
        'output_file': os.path.abspath(output_filename),
    }
    with open(diagnostics_path, 'w', encoding='utf-8') as file:
        json.dump(diagnostics, file, ensure_ascii=False, indent=2)
    print(f"波向诊断报告已保存: {diagnostics_path}")

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
    SAMPLE_SIZE = 50             # 采样轨迹数量（最短的N条轨迹）

    # --- 运行脚本 ---
    if not os.path.exists(PROCESSED_BUOY_FILE_WITH_WIND):
        print(f"错误: 输入文件 '{PROCESSED_BUOY_FILE_WITH_WIND}' 不存在。请先运行风场匹配脚本。")
    elif not os.path.exists(ERA5_WAVE_DATA_DIRECTORY):
        print(f"错误: ERA5波浪数据目录 '{ERA5_WAVE_DATA_DIRECTORY}' 不存在。请检查路径。")
    else:
        match_era5_wave(PROCESSED_BUOY_FILE_WITH_WIND, ERA5_WAVE_DATA_DIRECTORY, OUTPUT_DIRECTORY,
                        sample_mode=SAMPLE_MODE, sample_size=SAMPLE_SIZE)
