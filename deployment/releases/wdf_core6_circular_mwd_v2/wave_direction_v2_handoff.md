# Wave Direction Circular v2 - Windows Handoff

## Release identity

- Model version: `wdf_core6_circular_mwd_v2`
- WSL source branch: `wave_direction_circular_v2`
- Training code commit: `514170a3e3dc4c015df7159df3ae7279d5f3a1b2`
- Wave repair code commit: `8f599cd66be2513c910d66790d65d27bfd79efe8`
- Windows status: C++/Fortran acceptance passed on 2026-07-28
- Windows staging target:
  `D:\OilspillModel\OilSpillModel\ModelRun\release_onnx\wdf_core6_circular_mwd_v2`
- `onnx_active` was not changed during acceptance. Activation remains a
  separate, intentional Windows-side integration step.

## Data and model provenance

- v2 dataset: `trajectories_with_all_features_circular_mwd_v2.pkl`
- Dataset SHA256:
  `22ab0a32ff9472a6f8b8f57af5fd96b93cdeb76d45b4ef6b0a798fa1befb937e`
- Rows / trajectory segments / physical IDs:
  `16,734,351 / 4,141 / 2,439`
- Direction repair algorithm: `mwd_circular_exact_hour_local_month_v3`
- Only `era5_wave_dir_sin/cos` changed from v1. All other fields,
  indices, timestamps and `original_ID` values were verified unchanged.
- Train/validation/test `original_ID` lists are identical to v1:
  `1707 / 366 / 366` IDs.
- Frozen checkpoint:
  `trained_models/ablation_circular_mwd_v2_final/core_6/best_mlp.pth`
- Checkpoint SHA256:
  `1a3e2ab9091fa318f57dc00bfa10954a0faff62242be3860d151a06f48e15241`
- Scaler SHA256:
  `b73dbde04ae9b4059b6137e50b59cdbeab8c3f0613b1daf68ef3a102e0436f83`
- ONNX SHA256:
  `787d1d6a663677e30161a70493c70a7e46434414fb59085fbb68477939f18941`

## Model selection

The controlled v2 ablation used the same split, random seed, network body and
training configuration for both feature sets.

| Model | Validation joint R2 | Test joint R2 | Test RMSE (m/s) |
|---|---:|---:|---:|
| core6 | 0.128506 | 0.127398 | 0.194682 |
| full9 | 0.128680 | 0.126979 | 0.194728 |

The validation difference is only `0.000174`; full9 is slightly worse on the
held-out test set. The simpler core6 interface remains frozen for deployment.
The linear test baseline has joint R2 `0.122317` and RMSE `0.195250 m/s`.

## ONNX contract

- Opset: `12`
- Input: `input`, float32, `(batch_size, 6)`
- Output: `output`, float32, `(batch_size, 2)`
- Dynamic batches verified in WSL: `1`, `3`, `17`
- StandardScaler is inside the ONNX graph. Do not standardize in Fortran.
- NaN/Inf and missing values are not supported; validate before inference.

Strict input order:

1. `era5_u10` - eastward 10 m wind, m/s
2. `era5_v10` - northward 10 m wind, m/s
3. `era5_swh` - significant wave height, m
4. `era5_mwp` - mean wave period, s
5. `era5_wave_dir_sin` - dimensionless
6. `era5_wave_dir_cos` - dimensionless

Wave direction is ERA5 `mean_wave_direction`: `coming-from`, north is
0 degrees, angle increases clockwise, and no 180-degree rotation is applied:

```text
era5_wave_dir_sin = sin(mwd * pi / 180)
era5_wave_dir_cos = cos(mwd * pi / 180)
```

Strict output order:

1. `residual_u` - eastward drift residual, m/s
2. `residual_v` - northward drift residual, m/s

These are residual velocities, not total particle velocity. The oil-spill
model must add them to the intended background-current term exactly once.

Fortran arrays must be `real(c_float) :: features(6, N)` and
`real(c_float) :: drift_uv(2, N)`. This layout maps directly to C row-major
`(N, 6)` and `(N, 2)`.

## WSL verification

- ONNX checker: passed
- ONNX Runtime provider: CPUExecutionProvider
- PyTorch/ONNX maximum absolute difference: `5.2154064e-08`
- Acceptance threshold used during export: `< 1e-5`
- Fixed expected outputs:

```text
 0.0776273087,  0.00420927256
-0.0705249459,  0.0309366882
 0.000777993351,-0.107235432
```

`test_wdf_onnx.f90` reads `test_input.csv` and `expected_output.csv` at
runtime, so expected values are version-matched rather than hard-coded.

## Windows acceptance

Completed on 2026-07-28:

- VS2022 17.14.29 / MSVC 19.44.35225 wrapper build: passed
- Intel oneAPI `ifx` 2022.1.0 Build 20220316 compile/link: passed
- ONNX Runtime 1.17.1 Fortran -> C++ -> ONNX inference: passed
- Observed maximum absolute error: `1.4901e-08`
- Required tolerance: `< 1e-4`
- The final-run re-export retained the validated ONNX SHA256 exactly
- Detailed record: `WINDOWS_VALIDATION.md`
- `onnx_active` remained unchanged

Do not mix this ONNX file with the v1 scaler, test vectors or interface files.
