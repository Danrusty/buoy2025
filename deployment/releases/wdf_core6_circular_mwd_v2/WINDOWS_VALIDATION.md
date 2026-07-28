# Windows Acceptance Record

- Date: 2026-07-28
- Release: `wdf_core6_circular_mwd_v2`
- Environment: Windows x64, VS2022 17.14.29, MSVC 19.44.35225,
  Intel oneAPI `ifx` 2022.1.0 Build 20220316
- ONNX Runtime: 1.17.1
- Command: `verify_windows.bat`
- C++ wrapper build: passed
- Fortran compile and link: passed
- Fortran -> C++ -> ONNX inference: passed
- Maximum absolute difference from `expected_output.csv`: `1.4901e-08`
- Required tolerance: `< 1e-4`

Validated Windows staging directory:

`D:\OilspillModel\OilSpillModel\ModelRun\release_onnx\wdf_core6_circular_mwd_v2`

The existing `onnx_active` directory was not modified during validation.
Switching the oil-spill model to this release remains an explicit
Windows-side integration step.
