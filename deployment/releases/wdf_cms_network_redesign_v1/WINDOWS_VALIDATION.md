# Windows Acceptance Record

- Date: 2026-08-07
- Release: `wdf_cms_network_redesign_v1`
- Scientific selection: improved regional experiment; not recommended for
  activation because it remains weaker than the frozen global MLP
- Environment: Windows x64, VS2022 17.14.29, MSVC 19.44.35225,
  Intel oneAPI `ifx` 2022.1.0 Build 20220316
- ONNX Runtime: 1.17.1
- Command: `verify_windows.bat`
- C++ wrapper build: passed
- Fortran compile and link: passed
- Fortran -> C++ -> ONNX inference: passed
- Maximum absolute difference from `expected_output.csv`: `0.0000E+00`
- Required tolerance: `< 1e-4`
- Validated authoritative ONNX SHA256:
  `b79afcbf0f70add4ec6dfdf4f3299e69254e40c03a343c10171d1b1b44d09ff5`
- `wdf_drifter.onnx` compatibility alias: byte-identical to the authoritative
  `wdf_cms_orig_core6_v1.onnx`

Validated Windows staging directory:

`D:\OilspillModel\OilSpillModel\ModelRun\release_onnx\wdf_cms_network_redesign_v1`

The staging directory was newly created. The existing release root and
`onnx_active` were not modified; both retain frozen global ONNX SHA256
`787d1d6a663677e30161a70493c70a7e46434414fb59085fbb68477939f18941`.
The previous CMS staging release was also unchanged and retains SHA256
`5e89aeac80c96b122a957b2fb849db65f984667779712ef4b8a602ced4b3eb83`.
