# CMS Extent Search v1

## Purpose

Search for the smallest requested cumulative rectangle with enough independent
regional buoys to repeat the frozen-global adapter experiment.

This branch performs count-only selection. It does not inspect targets, fit an
adapter, evaluate prediction metrics, create an ONNX model, or change an active
deployment.

## Fixed search protocol

- Latitude: 15–45 N, inclusive.
- West longitude: 105 E, inclusive.
- First east boundary: 140 E.
- If a range is insufficient, expand eastward by 10 degrees through
  150/160/170/180 E and select the first passing boundary.
- 180 E is the fixed maximum because the source uses the
  `-180 to 180` longitude convention; this v1 search does not cross the
  dateline.
- Only in-rectangle rows count.
- Entering the region does not admit an ID's out-of-region rows.
- An `original_ID` must contribute at least 24 in-region hourly rows summed
  across its source segments.
- Split membership is inherited strictly from the frozen-global split
  manifest; no regenerated split is allowed for this search.

The first candidate is accepted only when all four conditions hold:

- total eligible `original_ID` >= 100;
- frozen-global train lineage >= 75;
- frozen-global validation lineage >= 15;
- frozen-global test lineage >= 10.

The mask source is the requested
`processed_data/trajectories_with_all_features.pkl`. The search validates its
known SHA256 before scanning.

## Branch workflow

The search is implemented and recorded on `wdf_cms_range_search_v1`, branched
from adapter result commit `422a4db`. After the range is locked and pushed,
adapter work returns to `wdf_cms_regional_core6_v1`. The previous
`wdf_cms_global_adapter_v1` result remains immutable; any expanded-range study
must use a new model/artifact version.

## Executed result

The final count-only scan used code commit
`0ad58d8480edb3003bff1e241a2c3b32b88a1d1e`.

| Longitude range | Total IDs | Train | Validation | Test | Rows | Pass |
|---|---:|---:|---:|---:|---:|---|
| 105–140 E | 53 | 41 | 9 | 3 | 101,127 | no |
| 105–150 E | 74 | 59 | 11 | 4 | 238,036 | no |
| 105–160 E | 87 | 69 | 11 | 7 | 367,649 | no |
| 105–170 E | 96 | 75 | 12 | 9 | 454,892 | no |
| 105–180 E | 172 | 131 | 18 | 23 | 526,212 | yes |

The selected minimum 10-degree candidate is therefore:

```text
15–45 N, 105–180 E
```

This selected support reaches the dateline and must not be described as a
China Marginal Seas-only dataset. It is an expanded East Asia–western North
Pacific latitude band designed to obtain enough independent frozen-lineage
IDs for the adapter generalization test.

## User count-threshold override

After reviewing only the count table above, and before any expanded-range
adapter data were constructed or any target/model metric was evaluated, the
user accepted the `105–170 E` candidate:

```text
96 total IDs / 75 train / 12 validation / 9 test / 454,892 rows
```

This explicitly overrides the initial `100 / 75 / 15 / 10` count requirement.
The adapter experiment must therefore use `15–45 N, 105–170 E`. Rows from
`170–180 E` are excluded. The 180 E count remains in the audit table only and
must not be used to describe the selected modeling population.
