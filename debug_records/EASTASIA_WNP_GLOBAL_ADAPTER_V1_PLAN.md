# East Asia–WNP Frozen-Global Adapter v1

## Controlled change

Repeat the locked `wdf_cms_global_adapter_v1` experiment with only one
scientific change: replace the original CMS union row filter with the
user-accepted rectangle:

```text
15–45 N, 105–170 E, inclusive
```

The count-only range search found 96 eligible `original_ID`, 454,892 rows and
frozen-global lineage counts of 75 train / 12 validation / 9 test. The user
explicitly accepted these counts before any expanded-range target or model
metric was evaluated. Provenance is locked on `wdf_cms_range_search_v1` at
commit `2c7111d451411bb2df96b6e75e111d9ef2451d7a`.

The resulting population is an East Asia–western North Pacific band, not a
China Marginal Seas-only dataset.

## Frozen controls

- Authoritative frozen-global ONNX and weights remain unchanged.
- Target remains `ve - cfsv2_u`, `vn - cfsv2_v`.
- Input remains the same raw physical core6.
- Each ID receives total fit weight 1.
- Candidate families G0–G5 and lambda grid remain unchanged.
- Nested shuffled GroupKFold, seed 42 and the one-standard-error rule remain
  unchanged.
- Development, gate and confirmation inherit the frozen-global lineage.
- Gate and confirmation thresholds remain unchanged.
- No neural-network retraining, fine-tuning, new feature, nonlinear adapter,
  ensemble, regional target correction or subregion-specific model is allowed.

## Data rules

- Only in-rectangle rows are retained.
- An ID entering the rectangle does not admit its outside rows.
- The 24-hour minimum is accumulated by `original_ID` across source segments.
- Complete 72-hour residence is not required.
- The original v1 file supplies the geographic mask, while aligned rows from
  the circular-v2 file supply model features.
- Every row east of 170 E is excluded.

## Evaluation sequence

1. Fit/select only on the 75 frozen-global train-lineage IDs using nested CV.
2. Evaluate the selected family/lambda on the 12 validation-lineage gate IDs.
3. Only if every locked development and gate condition passes, refit on
   75+12 IDs, hash and commit the adapter.
4. Evaluate the 9 confirmation IDs exactly once.
5. Freeze an appended ONNX and stage Windows validation only if point and
   24-hour trajectory-proxy acceptance both pass.

Reports include expanded overall, original CMS, BYS, ECS, NSCS,
105–140 E and 140–170 E subsets. The previous CMS adapter artifacts remain
immutable under their original version.

## Prepared data result

Data preparation used code commit
`63003b5b7f8adb6ec620cbefae41a106f40d2513`.

- 96 eligible `original_ID`, 454,892 selected hourly rows.
- 75 / 12 / 9 frozen-global train / validation / test lineage IDs.
- 371,404 / 55,691 / 27,797 rows by lineage split.
- Pairwise ID intersections are all zero.
- 162 source segments were separated into 219 continuous hourly episodes.
- Coordinate extrema are 15.000012–44.995888 N and
  110.461182–169.999725 E; no row east of 170 E is present.
- All target/current/core6 fields have zero missing values.
- All selected identity columns match between the v1 mask and circular-v2
  feature source.
- All 454,892 rows use the repaired circular-v2 wave direction.
- Original CMS subset: 21,074 rows.
- 105–140 E / 140–170 E support: 101,127 / 353,765 rows.
- Filtered-data SHA256:
  `3991faf3f5503cb69a089e1f432711977f64f2c53af069229c3e8826d73d9a85`.
