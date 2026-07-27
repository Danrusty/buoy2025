# Windows Acceptance Record

- Date: 2026-07-26
- Release: `wdf_core6_original_id_v1`
- Environment: Windows x64, VS2022 17.14.29, MSVC 19.44.35225,
  Intel oneAPI `ifx`, ONNX Runtime 1.17.1
- Command: `verify_windows.bat`
- C++ wrapper build: passed
- Fortran compile and link: passed
- Fortran -> C++ -> ONNX inference: passed
- Maximum absolute difference from `expected_output.csv`: `2.9802e-08`
- Required tolerance: `< 1e-4`

Validated Windows handoff directory:

`D:\OilspillModel\OilSpillModel\ModelRun\deployment\releases\wdf_core6_original_id_v1`

The active ModelRun source and model were not overwritten during this
acceptance test. Integration into the oil-spill executable remains a separate
Windows-side change.
