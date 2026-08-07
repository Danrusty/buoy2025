# wdf_cms_orig_core6_v1

## Scope

One unified China Marginal Seas MLP was trained with the frozen global core6
target, features, network, loss, optimizer, scheduler, early stopping and
random-seed strategy. Only the row-level geographic training-data selection
and the leakage-free `original_ID` split population changed.

## Dataset

- CMS original IDs: 23
- CMS hourly samples: 21074
- Continuous in-region episodes: 46
- BYS / ECS / NSCS membership rows:
  0 /
  13956 /
  7118
- Split strategy: `regenerated_group_shuffle_split`
- Train / val / test IDs:
  15 /
  4 /
  4
- Train / val / test rows:
  14173 /
  3974 /
  2927
- Pairwise split ID intersections: 0

The supplied source contains no qualifying BYS rows. BYS metrics are therefore
reported as `N/A`; this model is a CMS-mask model whose observed support in this
dataset is ECS + NSCS.

## Regional linear baseline

- A matrix: `[[0.005300786346197128, -0.005780019797384739], [-0.0021729543805122375, 0.005988077726215124]]`
- Intercept: `[-0.07374081015586853, -0.04901418089866638]`
- Effective WDF (`trace(A)/2`): 0.00564443
- Cross-wind coefficient: 0.00180353
- Test joint R2: 0.011372
- Test RMSE: 0.376188 m/s

## MLP test metrics

- R2 u: -0.056290
- R2 v: -0.076261
- Joint R2: -0.066276
- RMSE: 0.390745 m/s
- MAE: 0.299108 m/s
- Frozen global MLP on the identical CMS test rows:
  joint R2 0.134824,
  RMSE 0.352383 m/s

## Test subsets

Bias is `mean(predicted residual - observed residual)`.

| Subset | Rows | IDs | Regional joint R2 | Regional RMSE | Bias u | Bias v | Global joint R2 | Global RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CMS_overall | 2927 | 4 | -0.066276 | 0.390745 | 0.106035 | -0.097247 | 0.134824 | 0.352383 |
| Bohai_Yellow_Sea | 0 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| East_China_Sea | 1869 | 4 | -0.082675 | 0.371750 | 0.021834 | -0.148886 | 0.079239 | 0.342962 |
| Northern_South_China_Sea | 1058 | 2 | -0.193158 | 0.422217 | 0.254780 | -0.006025 | 0.081163 | 0.368438 |

## Frozen contract

All 15 checked training/interface fields are
identical to the frozen global core6 configuration.
