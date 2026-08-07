# wdf_cms_orig_core6_v1 Windows Handoff

## Identity and status

- Model: `wdf_cms_orig_core6_v1`
- Scientific status: frozen single CMS regional core6 MLP
- Deployment status: Windows C++/Fortran validation passed; not activated
- Authoritative ONNX: `wdf_cms_orig_core6_v1.onnx`
- Unchanged Fortran-test alias: `wdf_drifter.onnx`
- Intended Windows staging directory:
  `D:\OilspillModel\OilSpillModel\ModelRun\release_onnx\wdf_cms_orig_core6_v1`
- `onnx_active` is not changed by this handoff.
- Activation recommendation:
  `do_not_activate`

The two ONNX filenames are byte-identical. The named CMS file is the
authoritative handoff artifact; the alias keeps the existing
`test_wdf_onnx.f90` logic unchanged.

## Frozen interface

- Input: `input`, float32, `(batch_size, 6)`
- Output: `output`, float32, `(batch_size, 2)`
- StandardScaler is inside ONNX; do not standardize again in Fortran.
- Feature order:
  `era5_u10, era5_v10, era5_swh, era5_mwp,`
  `era5_wave_dir_sin, era5_wave_dir_cos`
- Target:
  `residual_u = ve - cfsv2_u`,
  `residual_v = vn - cfsv2_v`
- ONNX opset: 12
- PyTorch/ONNX maximum absolute difference:
  2.235e-08
- Dynamic batches verified:
  [1, 3, 17]

## Regional data support

- CMS IDs / rows:
  23 /
  21074
- BYS / ECS / NSCS rows:
  0 /
  13956 /
  7118
- Split strategy:
  `regenerated_group_shuffle_split`

The supplied source contains zero BYS rows. The release is valid for the exact
CMS-mask experiment, but its observed training support is ECS + NSCS; no BYS
performance claim can be made.

## Same-test-set comparison

- Regional CMS test joint R2: -0.066276
- Regional CMS test RMSE: 0.390745 m/s
- Frozen global joint R2 on the same rows: 0.134824
- Frozen global RMSE on the same rows: 0.352383 m/s

The regional MLP is worse than both the regional linear baseline and the
frozen global MLP on the same test rows. This release freezes the requested
experiment for reproducibility and Windows handoff; it is not selected for
operational activation.

## Verification

1. Keep all release files together.
2. Run `verify_windows.bat` in the staging directory.
3. The unchanged Fortran test loads `wdf_drifter.onnx`.
4. Require every output difference from `expected_output.csv` to be `< 1e-4`.
5. Record the authoritative ONNX SHA256 from `SHA256SUMS.txt`.

Checkpoint, scaler, split manifest, row-level region index, region definition,
data statistics, metrics, fixed test vectors, ONNX metadata and checksums are
included for reproducibility.

## Completed Windows validation

- Date: 2026-08-07
- VS2022 17.14.29 / MSVC 19.44.35225 wrapper build: passed
- Intel oneAPI ifx 2022.1.0 Build 20220316 compile/link: passed
- ONNX Runtime 1.17.1 Fortran -> C++ -> ONNX inference: passed
- Maximum absolute difference: `2.2352e-08`
- Required tolerance: `< 1e-4`
- Validated authoritative ONNX SHA256:
  `5e89aeac80c96b122a957b2fb849db65f984667779712ef4b8a602ced4b3eb83`
- Staging directory:
  `D:\OilspillModel\OilSpillModel\ModelRun\release_onnx\wdf_cms_orig_core6_v1`
- The old release root and `onnx_active` remain at SHA256
  `787d1d6a663677e30161a70493c70a7e46434414fb59085fbb68477939f18941`.
