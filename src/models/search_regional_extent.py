"""搜索满足 frozen-global lineage 数量门槛的最小区域矩形。"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from cms_regional import (
    GLOBAL_SPLIT_MANIFEST_PATH,
    MASK_SOURCE_PATH,
    MIN_REGIONAL_POINTS,
)
from data_loader import PROJECT_ROOT


SEARCH_VERSION = "wdf_cms_extent_search_v1"
LATITUDE_RANGE = (15.0, 45.0)
WEST_LONGITUDE = 105.0
EAST_LONGITUDE_CANDIDATES = (
    140.0,
    150.0,
    160.0,
    170.0,
    180.0,
)
MINIMUM_TOTAL_IDS = 100
MINIMUM_LINEAGE_IDS = {
    "train": 75,
    "val": 15,
    "test": 10,
}
EXPECTED_MASK_SOURCE_SHA256 = (
    "7f516210e0198a40584f19519cea5ac6e524dd27208925d902619d7234734608"
)
DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / SEARCH_VERSION
    / "range_search.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_original_id(frame: pd.DataFrame) -> str:
    values = frame["original_ID"].dropna().astype(str).str.strip().unique()
    if len(values) != 1 or not values[0]:
        raise ValueError(
            f"每个源片段必须只有一个非空 original_ID，实际为 {values[:5]}"
        )
    return str(values[0])


def validate_global_lineage(
    global_manifest: dict[str, Any],
) -> dict[str, set[str]]:
    lineage = {
        name: set(
            map(str, global_manifest["splits"][name]["original_ids"])
        )
        for name in ("train", "val", "test")
    }
    if (
        lineage["train"] & lineage["val"]
        or lineage["train"] & lineage["test"]
        or lineage["val"] & lineage["test"]
    ):
        raise ValueError("frozen-global manifest 的 original_ID 存在交集。")
    return lineage


def count_rows_by_id_and_extent(
    trajectories: Iterable[pd.DataFrame],
    east_longitudes: tuple[float, ...] = EAST_LONGITUDE_CANDIDATES,
    *,
    latitude_range: tuple[float, float] = LATITUDE_RANGE,
    west_longitude: float = WEST_LONGITUDE,
) -> tuple[dict[float, Counter[str]], dict[str, int]]:
    """逐行累计每个 ID 在各累积矩形内的小时点数。"""
    ordered_east = tuple(sorted(map(float, east_longitudes)))
    if not ordered_east:
        raise ValueError("east_longitudes 不能为空。")
    if ordered_east[0] <= west_longitude:
        raise ValueError("东界必须严格位于西界以东。")
    lat_min, lat_max = map(float, latitude_range)
    if lat_min >= lat_max:
        raise ValueError("纬度上界必须大于下界。")

    counts = {east: Counter() for east in ordered_east}
    source_segments = 0
    source_rows = 0
    source_ids: set[str] = set()
    for index, frame in enumerate(trajectories):
        required = {"original_ID", "latitude", "longitude"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"源片段 {index} 缺列: {sorted(missing)}")
        original_id = canonical_original_id(frame)
        source_ids.add(original_id)
        latitude = frame["latitude"].to_numpy(dtype=np.float64)
        longitude = frame["longitude"].to_numpy(dtype=np.float64)
        base = (
            np.isfinite(latitude)
            & np.isfinite(longitude)
            & (latitude >= lat_min)
            & (latitude <= lat_max)
            & (longitude >= west_longitude)
        )
        for east in ordered_east:
            selected = int(np.count_nonzero(base & (longitude <= east)))
            if selected:
                counts[east][original_id] += selected
        source_segments += 1
        source_rows += len(frame)
    source_summary = {
        "n_source_segments": source_segments,
        "n_source_rows": source_rows,
        "n_source_original_ids": len(source_ids),
    }
    return counts, source_summary


def summarize_extent_search(
    trajectories: Iterable[pd.DataFrame],
    global_manifest: dict[str, Any],
    *,
    east_longitudes: tuple[float, ...] = EAST_LONGITUDE_CANDIDATES,
    latitude_range: tuple[float, float] = LATITUDE_RANGE,
    west_longitude: float = WEST_LONGITUDE,
    min_regional_points: int = MIN_REGIONAL_POINTS,
    minimum_total_ids: int = MINIMUM_TOTAL_IDS,
    minimum_lineage_ids: dict[str, int] = MINIMUM_LINEAGE_IDS,
) -> dict[str, Any]:
    lineage = validate_global_lineage(global_manifest)
    lineage_union = set().union(*lineage.values())
    counts_by_east, source_summary = count_rows_by_id_and_extent(
        trajectories,
        east_longitudes,
        latitude_range=latitude_range,
        west_longitude=west_longitude,
    )

    candidates = []
    selected = None
    for east in sorted(counts_by_east):
        row_counts = counts_by_east[east]
        eligible_ids = sorted(
            original_id
            for original_id, count in row_counts.items()
            if count >= min_regional_points
        )
        eligible_set = set(eligible_ids)
        uncovered = sorted(eligible_set - lineage_union)
        if uncovered:
            raise RuntimeError(
                "区域 ID 未被 frozen-global lineage 覆盖: "
                f"{uncovered[:10]}"
            )
        lineage_ids = {
            name: sorted(eligible_set & values)
            for name, values in lineage.items()
        }
        lineage_counts = {
            name: len(values) for name, values in lineage_ids.items()
        }
        checks = {
            "total_original_ids_at_least_minimum": (
                len(eligible_ids) >= minimum_total_ids
            ),
            **{
                f"{name}_original_ids_at_least_{minimum_lineage_ids[name]}": (
                    lineage_counts[name] >= minimum_lineage_ids[name]
                )
                for name in ("train", "val", "test")
            },
        }
        candidate = {
            "latitude": list(map(float, latitude_range)),
            "longitude": [float(west_longitude), east],
            "minimum_regional_hourly_points_per_original_id": (
                min_regional_points
            ),
            "n_original_ids_before_minimum": len(row_counts),
            "n_original_ids": len(eligible_ids),
            "n_samples": int(
                sum(row_counts[value] for value in eligible_ids)
            ),
            "lineage_counts": lineage_counts,
            "eligible_original_ids": eligible_ids,
            "lineage_original_ids": lineage_ids,
            "checks": checks,
            "passed": all(checks.values()),
        }
        candidates.append(candidate)
        if selected is None and candidate["passed"]:
            selected = candidate

    return {
        "schema_version": 1,
        "search_version": SEARCH_VERSION,
        "mask": {
            "coordinate_system": "WGS84 latitude/longitude degrees",
            "boundary_policy": "inclusive",
            "latitude": list(map(float, latitude_range)),
            "west_longitude": float(west_longitude),
            "east_longitude_candidates": sorted(
                map(float, east_longitudes)
            ),
            "row_selection": (
                "Only rows inside each rectangle are counted; an ID entering "
                "the rectangle does not admit its outside rows."
            ),
        },
        "minimum_regional_hourly_points_per_original_id": (
            min_regional_points
        ),
        "requirements": {
            "minimum_total_original_ids": minimum_total_ids,
            "minimum_inherited_lineage_original_ids": minimum_lineage_ids,
        },
        "source_summary": source_summary,
        "candidates": candidates,
        "selected_range": selected,
        "status": "selected" if selected is not None else "no_range_passed",
    }


def run_search(
    *,
    source_path: Path = MASK_SOURCE_PATH,
    manifest_path: Path = GLOBAL_SPLIT_MANIFEST_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    code_commit: str,
) -> dict[str, Any]:
    source_path = Path(source_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_path = Path(output_path).resolve()
    actual_source_sha256 = sha256_file(source_path)
    if actual_source_sha256 != EXPECTED_MASK_SOURCE_SHA256:
        raise RuntimeError(
            "mask source SHA256 不匹配: "
            f"{actual_source_sha256} != {EXPECTED_MASK_SOURCE_SHA256}"
        )
    with source_path.open("rb") as file:
        trajectories = pickle.load(file)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = summarize_extent_search(trajectories, manifest)
    result.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "search_code_git_commit": code_commit,
            "source": {
                "path": str(source_path.relative_to(PROJECT_ROOT)),
                "size_bytes": source_path.stat().st_size,
                "sha256": actual_source_sha256,
            },
            "global_split_manifest": {
                "path": str(manifest_path.relative_to(PROJECT_ROOT)),
                "size_bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
            },
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖范围搜索结果: {output_path}")
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="搜索满足 frozen-global lineage 门槛的最小矩形"
    )
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    args = parser.parse_args()
    result = run_search(
        output_path=args.output,
        code_commit=args.code_commit,
    )
    for candidate in result["candidates"]:
        print(
            f"{candidate['longitude']}: total={candidate['n_original_ids']} "
            f"train={candidate['lineage_counts']['train']} "
            f"val={candidate['lineage_counts']['val']} "
            f"test={candidate['lineage_counts']['test']} "
            f"samples={candidate['n_samples']} "
            f"passed={candidate['passed']}"
        )
    if result["selected_range"] is None:
        print("Selected: none")
    else:
        print(
            "Selected: "
            f"lat={result['selected_range']['latitude']} "
            f"lon={result['selected_range']['longitude']}"
        )


if __name__ == "__main__":
    main()
