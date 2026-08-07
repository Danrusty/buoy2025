# Global Latitude × Model-Class Factorial v1

## Research question

The frozen global core6 MLP remains the reference model. This controlled
factorial asks whether its weak trajectory impact is primarily associated with:

1. missing large-scale spatial information;
2. limited function approximation by the MLP model class;
3. both factors through a non-additive interaction; or
4. neither factor.

The added spatial signal is intentionally limited to
`sin(deg2rad(latitude))`. It is a latitude experiment, not a complete spatial
encoding experiment.

## Four controlled cells

| Cell | Model | Inputs | Branch |
|---|---|---|---|
| A (frozen reference) | MLP | frozen core6 | `master@f2a0170` |
| B | MLP | core6 + `sin_latitude` | `wdf_global_mlp_lat7_v1` |
| C | XGBoost | frozen core6 | `wdf_global_xgb_core6_v1` |
| D | XGBoost | core6 + `sin_latitude` | `wdf_global_xgb_lat7_v1` |

Core6 order is fixed:

1. `era5_u10`
2. `era5_v10`
3. `era5_swh`
4. `era5_mwp`
5. `era5_wave_dir_sin`
6. `era5_wave_dir_cos`

For B and D, `sin_latitude` is appended as input 7.

## Frozen controls

- Dataset:
  `processed_data/trajectories_with_all_features_circular_mwd_v2.pkl`
- Dataset SHA256:
  `22ab0a32ff9472a6f8b8f57af5fd96b93cdeb76d45b4ef6b0a798fa1befb937e`
- Exact `original_ID` split:
  `trained_models/ablation_circular_mwd_v2_final/core_6/split_manifest.json`
- Split counts: train 1707 IDs, validation 366 IDs, test 366 IDs.
- Targets remain `ve - cfsv2_u` and `vn - cfsv2_v`.
- Row membership and row order are identical in all four cells.
- No longitude, region label, new environmental feature, target correction,
  sequence model, regional adapter, or per-region model is introduced.
- Random seed remains 42.

The shared cache stores core6 once and stores `sin_latitude` separately. This
avoids duplicating the global arrays while allowing exact construction of
either feature set. Its tracked provenance is
`trained_models/global_factorial_v1/data_manifest.json`; the arrays themselves
remain under ignored `processed_data/global_factorial_v1/`.

## Training and model selection

Cell B keeps the frozen MLP architecture, MSE loss, AdamW optimizer, scheduler,
batch size, maximum epochs, early-stopping rule, and random-seed strategy. Only
the first layer changes from six to seven inputs.

Cells C and D use two independent XGBoost regressors, one for each target
component. A small, predeclared search is evaluated only on validation data.
The selected configuration is locked before the test set is evaluated. Both
XGBoost cells use the same search space and selection rule.

Only one memory-intensive GPU training job runs at a time. CPU-only
preparation, validation, and reporting may overlap when they do not contend for
the training job's resources.

## Evaluation

Every cell is evaluated on the identical frozen test rows. The primary metrics
are row-weighted `R2_u`, `R2_v`, joint R2 (the arithmetic mean of the two),
RMSE, and MAE. The report additionally includes:

- residual bias for u and v;
- equal-weight macro-`original_ID` metrics;
- per-ID RMSE win/tie/loss rate against frozen cell A;
- fixed latitude-band diagnostics.

The factorial effects are:

- latitude under MLP: B − A;
- model class without latitude: C − A;
- latitude under XGBoost: D − C;
- model class with latitude: D − B;
- interaction: D − C − B + A.

## XGBoost efficiency gate

Target-environment inference benchmarking is required only if an XGBoost cell
shows a meaningful accuracy gain against A:

- joint R2 improves by at least 0.01 absolute;
- RMSE improves by at least 1%;
- macro-ID metrics also improve rather than shifting gain to long records.

If the gate passes, benchmark batch size 1 and an operational batch with warmup,
p50/p95 latency, throughput, model load time, model size, memory, and explicit
thread count. The final deployment decision remains separate from this global
baseline experiment.

## Stop condition

After all four cells are compared, stop global training expansion. A later
regional adapter is considered only from the resulting best global baseline
and requires a separate decision.
