# Minimal Feature Ablation: core_6 vs full_9

Both models use the same physical-buoy `original_ID` split, random seed,
network body, optimizer, scheduler, early stopping rule, and test set. Only
the input feature set changes.

## Feature Sets

- `core_6`: `era5_u10, era5_v10, era5_swh, era5_mwp, era5_wave_dir_sin, era5_wave_dir_cos`
- `full_9`: `era5_u10, era5_v10, era5_wind_speed, era5_wind_dir_sin, era5_wind_dir_cos, era5_swh, era5_mwp, era5_wave_dir_sin, era5_wave_dir_cos`

## Results

| Feature set | Params | Best epoch | Val joint R2 | Test R2 u | Test R2 v | Test joint R2 | Test RMSE | Test MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| core_6 | 433,538 | 12 | 0.128506 | 0.127345 | 0.127450 | 0.127398 | 0.194682 | 0.143305 |
| full_9 | 435,074 | 9 | 0.128680 | 0.127351 | 0.126607 | 0.126979 | 0.194728 | 0.143330 |

`full_9 - core_6`:

- Test joint R2: -0.000419
- Test RMSE: +0.000045 m/s
- Test MAE: +0.000025 m/s
- Validation joint R2: +0.000174

Lower test RMSE in this controlled run: **core_6**.

Joint R2 is calculated in float64 as the arithmetic mean of the separate
`residual_u` and `residual_v` R2 values.
