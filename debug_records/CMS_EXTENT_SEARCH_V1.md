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
- If the first range is insufficient, expand eastward by 10 degrees to 150 E.
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
