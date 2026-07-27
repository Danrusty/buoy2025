# WDF ONNX Windows Handoff

- Model version: `wdf_core6_original_id_v1`
- Scientific status: frozen core_6 model selected by controlled ablation
- Windows status: validated with C++/Fortran end-to-end test
- ONNX opset: 12
- Input: `input`, float32, `(batch_size, 6)`
- Output: `output`, float32, `(batch_size, 2)`
- StandardScaler: baked into ONNX; do not standardize again in Fortran
- PyTorch/ONNX max absolute difference: 2.980e-08

## Frozen input order

1. `era5_u10`
2. `era5_v10`
3. `era5_swh`
4. `era5_mwp`
5. `era5_wave_dir_sin`
6. `era5_wave_dir_cos`

## Files

- `wdf_drifter.onnx`: Windows runtime model.
- `interface.json`: authoritative feature/output contract.
- `test_input.csv`: fixed raw physical input vectors.
- `expected_output.csv`: Python ONNX Runtime reference output.
- `release_manifest.json`: source model, split and metric provenance.
- `SHA256SUMS.txt`: release file integrity checks.
- `onnx_wrapper.*`, `wdf_model_mod.f90`: C++/Fortran interface.
- `test_wdf_onnx.f90`: Windows chain verification program.
- `build_wrapper.bat`: VS2022 x64 wrapper build script.
- `verify_windows.bat`: one-command VS2022/oneAPI build and acceptance test.

## Windows acceptance

1. Use this complete release as one version; do not mix it with any 9-feature wrapper.
2. Run `verify_windows.bat` from a normal Windows command prompt.
3. The script loads VS2022/oneAPI, builds the wrapper and runs the Fortran test.
4. Compare all outputs with `expected_output.csv`; require absolute error `< 1e-4`.
5. Record the deployed ONNX SHA256 from `SHA256SUMS.txt`.

## Completed validation

- Date: 2026-07-26
- Target: Windows x64, VS2022 17.14.29, MSVC 19.44.35225, Intel oneAPI `ifx`
- ONNX Runtime: 1.17.1
- Result: C++ wrapper build passed; Fortran -> C++ -> ONNX test passed
- Maximum absolute difference from Python reference: `2.9802e-08`
- Windows handoff path:
  `D:\OilspillModel\OilSpillModel\ModelRun\deployment\releases\wdf_core6_original_id_v1`

The previous trajectory-index split model is internal legacy and must not be
mixed with this release.
