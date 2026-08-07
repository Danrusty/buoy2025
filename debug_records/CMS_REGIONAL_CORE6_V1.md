# CMS Regional Core6 v1

## Baseline identity

- Repository baseline: `master` at
  `f2a01709c8cac05f580341df70b477a633614aae`
- Frozen archive tag: `archive/wdf-core6-circular-mwd-v2`
- Development branch: `wdf_cms_regional_core6_v1`
- Frozen global release: `wdf_core6_circular_mwd_v2`
- New model version: `wdf_cms_orig_core6_v1`
- Python distribution/environment: Miniforge3 / `buoy-drifter`

The frozen global ONNX has already passed Python, C++, Fortran and Windows
numerical validation. This regional experiment does not reinterpret that
engineering validation as evidence of trajectory-level scientific benefit.

## What the old framework established

The global circular-v2 experiment established a reliable six-feature deployment
contract and a leakage-free physical-buoy split:

- target:
  `residual_u = ve - cfsv2_u`,
  `residual_v = vn - cfsv2_v`
- input:
  `era5_u10, era5_v10, era5_swh, era5_mwp,`
  `era5_wave_dir_sin, era5_wave_dir_cos`
- scaler inside ONNX
- dynamic batch ONNX interface `(N, 6) -> (N, 2)`
- `original_ID` train/validation/test separation
- circular `coming-from` ERA5 mean-wave-direction convention

The global core6 test result was joint R2 `0.127398` and RMSE
`0.194682 m/s`. The corresponding linear baseline was joint R2 `0.122317`
and RMSE `0.195250 m/s`. Thus the MLP exceeded the linear baseline by only
`0.005081` joint R2 and `0.29%` RMSE.

## Problems not resolved by the old framework

1. **Engineering acceptance did not imply trajectory improvement.**
   The ONNX and language bindings were numerically correct, but the actual
   Fortran oil-spill trajectories improved only weakly.

2. **The scientific objective and acceptance metric were at different
   levels.** Training and model selection used row-weighted hourly velocity
   loss/R2. The operational question concerns accumulated trajectory
   displacement, direction, landfall and separation over time.

3. **Global geographic mixing can average regional wind response.**
   The frozen global linear map has effective WDF about `0.01367`, with small
   global off-diagonal terms. A single global loss can weaken or cancel
   directionally distinct regional responses even when the deployment chain is
   correct.

4. **Hourly rows are not independent scientific replicates.**
   The global dataset contains millions of highly correlated points from
   2,439 physical IDs. Splitting by ID prevents direct leakage, but the loss and
   reported metrics still weight long records more heavily.

5. **The global held-out result does not diagnose operational distribution
   shift.** Sanchi and A Symphony occupy a small geographic and environmental
   part of the global support.

6. **The old baseline's `wdf_offdiag` was only the mean of the two
   off-diagonal entries.** The regional analysis additionally reports the
   physically interpretable rotational cross-wind coefficient
   `(A[1,0] - A[0,1]) / 2`, while retaining the full matrix.

These are reasons for a controlled regional data experiment, not permission to
change the frozen target, model, features or deployment contract.

## CMS mask and source decision

CMS is the row-level union of three inclusive rectangles:

- BYS: 31–41 N, 117–127 E
- ECS: 23–33 N, 117–131 E
- NSCS: 15–23 N, 105–122 E

The geographic mask is computed from the requested
`trajectories_with_all_features.pkl`. The selected rows are taken at identical
positions from
`trajectories_with_all_features_circular_mwd_v2.pkl`, because the latter is the
actual frozen global-v2 feature source.

On all 21,074 CMS rows, the two files have identical ID, time, location,
target/current, wind, SWH and MWP values. Their wave-direction `sin/cos` values
differ on every CMS row. Training directly from the v1 wave-direction values
would therefore change both geography and wave preprocessing, violating the
"only change training-data range" control.

Rows are retained only while they are inside the CMS union. An `original_ID`
entering CMS does not admit its out-of-region rows. Selected rows are separated
into continuous hourly episodes for provenance, while the MLP remains
stateless. IDs with fewer than 24 total CMS hourly rows are excluded.

Rectangle memberships are stored as independent booleans. CMS union rows are
deduplicated; a geometrically overlapping row may be included in both stated
subregion reports.

## Pre-training data audit

Strict masking of the supplied source gives:

- 23 unique `original_ID`
- 21,074 hourly rows
- 46 continuous in-region episodes from 24 source segments
- BYS: 0 rows
- ECS: 13,956 rows
- NSCS: 7,118 rows
- no missing values in the target or six frozen features
- all 23 IDs already exceed the 24-hour minimum

Inheriting the frozen global manifest would yield 19/2/2 train/validation/test
IDs. Because validation and test would each contain only two physical IDs, the
pipeline follows the specified fallback: two-stage
`GroupShuffleSplit`, seed 42, with zero ID intersection.

The zero-row BYS result is a source-data limitation, not a mask failure. The
model can still be trained under the exact CMS definition, but its observed
support is ECS + NSCS and BYS metrics must remain unavailable.

## Frozen controls

The pipeline enforces equality with the frozen global `model_config.json` for:

- network architecture, BatchNorm and dropout
- loss-monitor/early-stopping behavior
- batch size and maximum epochs
- optimizer, weight decay and learning-rate scheduler
- feature and target order
- random seed

It does not implement fixed `0.03` correction, regional target correction,
sequence models, new features, hyperparameter search or separate subregion
models.

## Planned artifacts and gates

Training artifacts are written under:

`trained_models/wdf_cms_orig_core6_v1`

Analysis results are written under:

`results/wdf_cms_orig_core6_v1`

The Windows release is written under:

`deployment/releases/wdf_cms_orig_core6_v1`

Required gates:

1. unit tests for row masking, thresholding and ID split;
2. zero pairwise ID intersection;
3. exact frozen-contract comparison;
4. regional linear analysis and global reference;
5. regional/global MLP evaluation on identical CMS test rows;
6. overall/ECS/NSCS test metrics and explicit no-sample BYS status;
7. ONNX checker, dynamic batches and PyTorch/ONNX numerical consistency;
8. authoritative ONNX SHA256 and fixed input/output vectors;
9. unchanged C++/Fortran wrapper validation on Windows.

Formal training results and Windows acceptance details are appended only after
their corresponding gates pass.

## Formal training result

The formal run used Miniforge3 environment `buoy-drifter`, CUDA on an NVIDIA
GeForce RTX 4070 Ti SUPER, and training-code commit
`b9bcfc2df853117863da28e3d8e734a3f01d96a9`.

- train/validation/test IDs: 15 / 4 / 4
- train/validation/test rows: 14,173 / 3,974 / 2,927
- pairwise ID intersections: 0
- checkpoint epoch: 1
- early stopping: epoch 21 after 20 non-improving epochs
- parameter count: 433,538
- all 15 frozen configuration fields: identical to global core6

Regional linear fit:

```text
A = [[ 0.00530079, -0.00578002],
     [-0.00217295,  0.00598808]]
b = [-0.07374081, -0.04901418]
```

- effective WDF: `0.00564443`
- cross-wind coefficient: `0.00180353`
- test joint R2: `0.011372`
- test RMSE: `0.376188 m/s`

CMS regional MLP test result:

- R2 u / v / joint:
  `-0.056290 / -0.076261 / -0.066276`
- RMSE: `0.390745 m/s`
- MAE: `0.299108 m/s`

Frozen global MLP on the identical CMS test rows:

- R2 u / v / joint:
  `0.119860 / 0.149788 / 0.134824`
- RMSE: `0.352383 m/s`
- MAE: `0.263393 m/s`

The regional MLP is worse than both the regional linear baseline and the
frozen global MLP. The regional linear effective WDF is also smaller, not
larger, than the global value (`0.00564` versus `0.01367`). This controlled run
therefore does not support the hypothesis that geographic restriction alone
reveals a stronger learnable wind-drift signal.

The most direct explanation is inadequate independent regional support: only
23 physical IDs, with four validation and four test IDs, no BYS observations,
large target-mean shifts between splits, and a frozen 433k-parameter network.
No architecture, target, loss, feature, weighting or hyperparameter change was
made to compensate.

The requested ONNX is frozen as an experimental result for reproducibility and
Windows chain validation, but it is not recommended for `onnx_active` or
operational oil-spill activation based on these held-out results.

## ONNX freeze

- authoritative file: `wdf_cms_orig_core6_v1.onnx`
- SHA256:
  `5e89aeac80c96b122a957b2fb849db65f984667779712ef4b8a602ced4b3eb83`
- PyTorch/ONNX maximum absolute difference: `2.2351742e-08`
- acceptance threshold: `< 1e-5`
- dynamic batches passed: 1, 3, 17
- unchanged-Fortran compatibility alias: `wdf_drifter.onnx`
- alias and authoritative ONNX are byte-identical

## Windows acceptance

Completed on 2026-08-07 in:

`D:\OilspillModel\OilSpillModel\ModelRun\release_onnx\wdf_cms_orig_core6_v1`

- VS2022 17.14.29 / MSVC 19.44.35225 C++ wrapper build: passed
- Intel oneAPI ifx 2022.1.0 Build 20220316 compile/link: passed
- ONNX Runtime 1.17.1 Fortran -> C++ -> ONNX inference: passed
- maximum absolute difference: `2.2352e-08`
- required tolerance: `< 1e-4`
- validated ONNX SHA256:
  `5e89aeac80c96b122a957b2fb849db65f984667779712ef4b8a602ced4b3eb83`

The old release root and `onnx_active` were checked before and after staging;
both remain at frozen global SHA256
`787d1d6a663677e30161a70493c70a7e46434414fb59085fbb68477939f18941`.
Only the new versioned staging directory was created.

## Release retraction

After the scratch CMS MLP and both frozen-global adapter studies failed their
scientific generalization gates, the repository release directory
`deployment/releases/wdf_cms_orig_core6_v1` was removed. The historical
training, metrics and validation records remain under `trained_models/`,
`results/` and this document, but the model is no longer represented as a
release candidate.

This retraction does not alter the authoritative frozen-global release or its
active SHA256. The historical Windows staging copy is outside this repository
and was not modified by the repository cleanup.
