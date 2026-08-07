# CMS Frozen-Global Adapter v1

## Objective

Keep the authoritative frozen global core6 ONNX unchanged as the base model
and learn only a low-order China Marginal Seas discrepancy correction:

```text
observed residual = frozen_global(core6) + regional_adapter
```

The adapter study must not fine-tune global weights, add input features, train
nonlinear networks, split one `original_ID` across folds, or select a model
using test results.

## Frozen base

- Release: `wdf_core6_circular_mwd_v2`
- ONNX:
  `deployment/releases/wdf_core6_circular_mwd_v2/wdf_drifter.onnx`
- Expected SHA256:
  `787d1d6a663677e30161a70493c70a7e46434414fb59085fbb68477939f18941`
- Input: raw physical core6, float32, `(batch_size, 6)`
- Output: `residual_u, residual_v`, float32, `(batch_size, 2)`
- StandardScaler remains inside the frozen global graph.

## Data lineage

The primary adapter split inherits the frozen global model's original
`original_ID` split:

- development: the 19 CMS IDs in global train;
- gate: the 2 CMS IDs in global validation;
- confirmation: the 2 CMS IDs in global test.

The previously regenerated CMS 15/4/4 split is retained only as a secondary
legacy comparison because it mixes frozen-global training provenance into its
validation and test sets.

The confirmation IDs have already been used for a base-only lineage audit.
No adapter prediction has been evaluated on them. They are therefore a locked
confirmation set, not a pristine blind test.

## Pre-registered candidates

All candidates predict:

```text
adapter_target = observed_residual - frozen_global_prediction
```

| Name | Correction | Parameters |
|---|---|---:|
| `G0_global_only` | zero | 0 |
| `G1_bias2` | component bias | 2 |
| `G2_wind_rotation4` | isotropic scale/cross-wind matrix + bias | 4 |
| `G3_wind_full6` | full 2x2 wind matrix + bias | 6 |
| `G4_global_calibration6` | full 2x2 global-output correction + bias | 6 |
| `G5_core6_linear14` | full core6 linear correction + bias | 14 |

No combined, nonlinear, fine-tuned, subregion-specific, or ensemble adapter is
allowed in v1.

## Fitting and selection

- Each `original_ID` has total fitting weight 1.
- Rows within an ID have weight `1 / n_rows_for_ID`.
- The linear basis is RMS-scaled inside each training fold.
- Ridge penalizes every coefficient, including bias, so increasing
  regularization shrinks exactly toward the frozen global model.
- Fixed regularization grid:
  `0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10`.
- Development uses nested, shuffled `GroupKFold`, seed 42.
- The primary score is equal-ID mean adapter-minus-global MSE.
- The one-standard-error rule selects the smallest parameter count and then
  the strongest regularization among statistically equivalent candidates.

## Gate

Test evaluation is authorized only if all conditions hold:

1. the selected candidate is not `G0_global_only`;
2. nested development macro-ID RMSE improves by at least 2%;
3. at least 60% of development IDs improve;
4. the selected family appears in at least 60% of outer folds;
5. ECS and NSCS development macro-ID RMSE do not degrade by more than 2%;
6. correction P99 magnitude does not exceed global prediction P99;
7. gate point RMSE and macro-ID RMSE both improve;
8. no gate ID RMSE degrades by more than 5%.

If the gate passes, the locked candidate and lambda are refit on the 19
development plus 2 gate IDs. Those coefficients are hashed and committed
before the confirmation set is evaluated once.

## Confirmation and trajectory proxy

The locked 2-ID confirmation set must improve joint R2 and RMSE, with no ID
degrading by more than 5%.

For each continuous regional episode, non-overlapping 6/12/24/48/72-hour
windows accumulate velocity error into displacement error. At 24 hours:

- macro-ID median endpoint error must improve;
- macro-ID P90 endpoint error must not degrade by more than 5%.

Failure at either point or trajectory level stops the study without ONNX
freeze or Windows activation.

## ONNX handoff

Only an accepted adapter may be appended directly to the authoritative global
ONNX graph. The global graph is not re-exported. Its final tensor is renamed
internally to `global_output`; the linear correction and final `Add` are
appended while the public `input` and `output` contract remains unchanged.

`onnx_active` is never changed automatically.

## Executed selection result

The pre-registered selection was executed from code commit
`b207503ba08478c38727c3c10177edc5b2185372`.

- Frozen-global lineage was exactly 19 development / 2 gate / 2 sealed
  confirmation IDs.
- Development contained 15,823 rows and gate contained 4,508 rows.
- All five nested outer folds selected `G0_global_only`.
- The all-development best-mean candidate was also `G0_global_only`; it was
  the only candidate within the one-standard-error threshold.
- Frozen-global nested-development joint R2 was `0.073760` and RMSE was
  `0.300126 m/s`.
- The nearest nonzero candidate was `G1_bias2` at lambda `10`, but it
  increased equal-ID mean MSE by `0.000042001`, reduced macro-ID RMSE by
  `-0.0328%` (that is, a degradation), and improved only `42.1%` of IDs.
- Every adapter family's best cross-validated setting was more strongly
  regularized at lambda `10` and still worse than no correction.

The selection gate therefore failed before confirmation evaluation. The two
global-test CMS IDs remain sealed from adapter prediction. No
`adapter_for_test`, confirmation report, ONNX, Windows staging directory, or
active-model change was produced.

This result does not support a transferable low-order CMS correction from the
available 19 global-train regional IDs. It does support retaining the frozen
global model unchanged. BYS still has no observed training rows, so this study
also provides no empirical basis for claiming a Yellow Sea calibration.
